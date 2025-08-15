# app.py — Streamlit Board Game Recommender
import os, json, time, math, random, uuid, datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# ✨ NEW: Supabase client import
from supabase import create_client

# Load .env if present
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
GAMES_PATH = DATA_DIR / "games.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
RESPONSES_CSV = Path(__file__).parent / "responses.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change-me")

# ✨ NEW: Supabase environment + client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        supabase = None
        # Don't crash the app if misconfigured; show a gentle note later.
        st.sidebar.warning(f"Supabase init failed: {e}")


# ------------------------------ Models --------------------------------------
class Game(BaseModel):
    id: str
    name: str
    bgg_slug: str = ""
    players_min: int = 1
    players_max: int = 4
    solo: bool = False
    coop: bool = False
    playtime_min: int = 60
    playtime_max: int = 120
    complexity: float = 3.0  # 1..5
    themes: List[str] = Field(default_factory=list)
    mechanics: List[str] = Field(default_factory=list)
    image_url: str = ""
    notes: str = ""


class Questions(BaseModel):
    version: int = 1
    weights: Dict[str, float]
    questions: List[Dict[str, Any]]


# --------------------------- Storage helpers --------------------------------
def load_games() -> List[Game]:
    if not GAMES_PATH.exists():
        return []
    data = json.loads(GAMES_PATH.read_text(encoding="utf-8"))
    out = []
    for g in data:
        try:
            out.append(Game(**g))
        except ValidationError as e:
            st.warning(f"Skipping invalid game: {g.get('name', g.get('id'))} — {e}")
    return out


def save_games(games: List[Game]):
    GAMES_PATH.write_text(
        json.dumps([g.model_dump() for g in games], indent=2), encoding="utf-8"
    )


def load_questions() -> Questions:
    if not QUESTIONS_PATH.exists():
        st.stop()
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return Questions(**data)


def save_questions(q: Questions):
    QUESTIONS_PATH.write_text(json.dumps(q.model_dump(), indent=2), encoding="utf-8")


def append_response(row: Dict[str, Any]):
    # (kept) Save to CSV locally
    df = pd.DataFrame([row])
    if RESPONSES_CSV.exists():
        df.to_csv(RESPONSES_CSV, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(RESPONSES_CSV, index=False, encoding="utf-8")

    # ✨ NEW: Also save to Supabase if configured
    if supabase:
        try:
            payload = {
                "ts": row["ts"],
                "user": row["user"],
                # Store as JSON (dict), not string
                "answers_json": (
                    json.loads(row["answers_json"])
                    if isinstance(row["answers_json"], str)
                    else row["answers_json"]
                ),
                "ranking_json": (
                    json.loads(row["ranking_json"])
                    if isinstance(row["ranking_json"], str)
                    else row["ranking_json"]
                ),
            }
            supabase.table("survey_responses").insert(payload).execute()
        except Exception as e:
            st.error(f"Supabase insert failed: {e}")


# ✨ NEW: Helper to fetch all responses from Supabase (used by Leaderboard tab)
def fetch_all_responses() -> pd.DataFrame:
    if not supabase:
        return pd.DataFrame()
    try:
        res = supabase.table("survey_responses").select("*").execute()
        if res.data:
            return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Supabase fetch failed: {e}")
    return pd.DataFrame()


# --------------------------- GPT integration --------------------------------
def propose_questions_via_gpt(games: List[Game], n: int = 8) -> Questions:
    """
    Ask GPT to propose a small set of discriminative questions for the given games.
    Returns a Questions object. Requires OPENAI_API_KEY.
    """
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not set. Cannot generate questions.")
        st.stop()

    # Try both modern and legacy OpenAI clients for compatibility
    prompt = f"""
You are helping create a short survey that picks the best board game among candidates,
based on user preferences. The candidates (with features) are:

{json.dumps([g.model_dump() for g in games], indent=2)}

Return exactly this JSON with {n} multiple-choice questions:
{{
  "version": 1,
  "weights": {{
    "<qkey>": <float weight>  // relative importance; ~1.0 typical
  }},
  "questions": [
    {{
      "key": "<snake_case_key>",
      "text": "<concise question>",
      "options": [
        {{"label":"<user-facing>", "value":"<machine_value>"}}
      ]
    }}
  ]
}}

Guidelines:
- Focus on **discriminating** between these specific games (themes, coop vs comp, player counts, time, complexity, campaign vs one-shot, solo).
- Keep 3–6 options per question, clear and mutually exclusive.
- Use short keys, e.g., "complexity", "mode", "time", "theme", "group_size", "campaign", "solo".
- Provide a "weights" dict assigning relative importance (0.5–1.5 typical).
- Be concise. No extra commentary. JSON ONLY.
""".strip()

    try:
        # New-style SDK
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = resp.choices[0].message.content
    except Exception:
        # Legacy fallback
        try:
            import openai

            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

    try:
        data = json.loads(content)
        return Questions(**data)
    except Exception as e:
        st.error("Failed to parse GPT JSON. Showing raw output below for debugging.")
        st.code(content)
        st.stop()


# ---------------------------- Scoring logic ---------------------------------
def score_game(
    game: Game, answers: Dict[str, str], weights: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (total_score, details) where details is per-question contribution.
    All contributions are positive numbers (higher is better). We normalize per-question to [0,1].
    """
    details = {}
    total = 0.0

    # helper to clamp
    def clamp01(x):
        return max(0.0, min(1.0, x))

    # group_size
    if "group_size" in answers:
        v = answers["group_size"]
        if v == "solo":
            s = 1.0 if game.solo else 0.0
        elif v == "2":
            # favor games that allow 2; max if (players_min<=2<=players_max)
            s = (
                1.0
                if (game.players_min <= 2 <= game.players_max)
                else 0.5 if game.players_min <= 3 else 0.0
            )
        elif v == "3-4":
            s = (
                1.0
                if game.players_max >= 4 and game.players_min <= 3
                else 0.5 if game.players_max >= 3 else 0.0
            )
        else:  # 5+
            # none of the starters go 5+, so low
            s = 0.0 if game.players_max < 5 else 1.0
        details["group_size"] = s * weights.get("group_size", 1.0)
        total += details["group_size"]

    # mode (coop vs comp)
    if "mode" in answers:
        v = answers["mode"]
        if v == "coop":
            s = 1.0 if game.coop else 0.0
        elif v == "comp":
            s = 1.0 if not game.coop else 0.0
        else:
            s = 0.7  # either
        details["mode"] = s * weights.get("mode", 1.0)
        total += details["mode"]

    # session_length
    if "session_length" in answers:
        midpoint = (game.playtime_min + game.playtime_max) / 2
        v = answers["session_length"]
        target = {"short": 45, "medium": 75, "long": 120, "epic": 170}.get(v, 90)
        # similarity by absolute difference (smaller is better)
        diff = abs(midpoint - target)
        # 0 min diff → 1.0 ; 120+ min diff → ~0.0
        s = clamp01(1.0 - (diff / 120.0))
        details["session_length"] = s * weights.get("session_length", 1.0)
        total += details["session_length"]

    # complexity
    if "complexity" in answers:
        v = answers["complexity"]
        target_w = {
            "light": 2.0,
            "medium": 3.0,
            "heavy": 4.0,
            "any": game.complexity,
        }.get(v, 3.0)
        diff = abs(game.complexity - target_w)
        s = clamp01(1.0 - (diff / 2.0))  # 0 diff → 1.0 ; 2.0 diff → 0.0
        details["complexity"] = s * weights.get("complexity", 1.0)
        total += details["complexity"]

    # theme
    if "theme" in answers:
        v = answers["theme"]
        if v == "any":
            s = 0.7
        else:
            keyword_map = {
                "animals": ["animals", "zoo", "conservation"],
                "exploration": ["exploration", "narrative", "adventure"],
                "survival": ["survival", "city-building", "post-apocalyptic"],
                "sci-fi": ["sci-fi", "space"],
                "innovation": ["innovation", "civilization", "history"],
            }
            desired = set(keyword_map.get(v, []))
            overlap = len(desired.intersection(set(game.themes)))
            s = 1.0 if overlap > 0 else 0.0
        details["theme"] = s * weights.get("theme", 1.0)
        total += details["theme"]

    # campaign
    if "campaign" in answers:
        v = answers["campaign"]
        if v == "campaign_yes":
            s = (
                1.0
                if "campaign" in game.mechanics or "campaign" in game.themes
                else (1.0 if "story" in game.mechanics else 0.0)
            )
        elif v == "campaign_no":
            # reward non-campaign one-shots (Ark Nova, Inventions, SETI)
            s = 1.0 if "campaign" not in game.mechanics else 0.2
        else:
            s = 0.7
        details["campaign"] = s * weights.get("campaign", 1.0)
        total += details["campaign"]

    # solo
    if "solo" in answers:
        v = answers["solo"]
        if v == "solo_required":
            s = 1.0 if game.solo else 0.0
        elif v == "solo_nice":
            s = 1.0 if game.solo else 0.6
        else:
            s = 1.0 if not game.solo else 0.8
        details["solo"] = s * weights.get("solo", 1.0)
        total += details["solo"]

    # randomness
    if "randomness" in answers:
        v = answers["randomness"]
        # Heuristic: heavy euros → low randomness; coop survival/story → medium/high
        desired = {"low": 0.2, "medium": 0.5, "high": 0.8, "any": 0.5}[v]
        game_r = 0.5
        if game.name.lower().startswith("inventions") or game.name.lower().startswith(
            "ark nova"
        ):
            game_r = 0.3
        if game.coop and ("survival" in game.themes or "narrative" in game.themes):
            game_r = 0.6
        s = 1.0 - abs(desired - game_r)  # closer is better
        details["randomness"] = s * weights.get("randomness", 1.0)
        total += details["randomness"]

    # --- EXTRA TRAIT MAPPERS (personality-style) ---

    # Helper: get value safely
    def _val(key):
        return answers.get(key)

    # vibe_night -> soft preferences by game id
    if _val("vibe_night"):
        pref = {
            "deep_think": {
                "inventions": 1.0,
                "ark_nova": 0.9,
                "seti": 0.8,
                "sleeping_gods": 0.6,
                "frostpunk": 0.4,
            },
            "steady_build": {
                "ark_nova": 1.0,
                "seti": 0.9,
                "inventions": 0.7,
                "sleeping_gods": 0.5,
                "frostpunk": 0.4,
            },
            "desperate_survive": {
                "frostpunk": 1.0,
                "sleeping_gods": 0.6,
                "seti": 0.4,
                "ark_nova": 0.4,
                "inventions": 0.3,
            },
            "story_odyssey": {
                "sleeping_gods": 1.0,
                "frostpunk": 0.6,
                "ark_nova": 0.4,
                "seti": 0.4,
                "inventions": 0.4,
            },
            "curious_discovery": {
                "seti": 1.0,
                "sleeping_gods": 0.8,
                "ark_nova": 0.6,
                "inventions": 0.6,
                "frostpunk": 0.4,
            },
        }
        s = pref.get(answers["vibe_night"], {}).get(game.id, 0.5)
        details["vibe_night"] = s * weights.get("vibe_night", 1.0)
        total += details["vibe_night"]

    # decision_style
    if _val("decision_style"):
        pref = {
            "optimize_numbers": {
                "inventions": 1.0,
                "ark_nova": 0.9,
                "seti": 0.8,
                "frostpunk": 0.5,
                "sleeping_gods": 0.5,
            },
            "plan_long_term": {
                "ark_nova": 1.0,
                "inventions": 0.9,
                "seti": 0.7,
                "sleeping_gods": 0.6,
                "frostpunk": 0.6,
            },
            "improvise_on_the_fly": {
                "sleeping_gods": 1.0,
                "frostpunk": 0.8,
                "seti": 0.5,
                "ark_nova": 0.4,
                "inventions": 0.4,
            },
            "follow_heart": {
                "sleeping_gods": 0.9,
                "ark_nova": 0.7,
                "frostpunk": 0.7,
                "seti": 0.5,
                "inventions": 0.5,
            },
        }
        s = pref.get(answers["decision_style"], {}).get(game.id, 0.5)
        details["decision_style"] = s * weights.get("decision_style", 1.0)
        total += details["decision_style"]

    # map_aesthetic
    if _val("map_aesthetic"):
        pref = {
            "star_charts": {
                "seti": 1.0,
                "sleeping_gods": 0.6,
                "inventions": 0.5,
                "ark_nova": 0.4,
                "frostpunk": 0.4,
            },
            "rugged_wasteland": {
                "frostpunk": 1.0,
                "sleeping_gods": 0.5,
                "seti": 0.5,
                "ark_nova": 0.4,
                "inventions": 0.4,
            },
            "lush_habitats": {
                "ark_nova": 1.0,
                "sleeping_gods": 0.6,
                "seti": 0.4,
                "inventions": 0.4,
                "frostpunk": 0.3,
            },
            "ancient_routes": {
                "inventions": 1.0,
                "sleeping_gods": 0.7,
                "ark_nova": 0.5,
                "seti": 0.4,
                "frostpunk": 0.4,
            },
            "hand_drawn_journal": {
                "sleeping_gods": 1.0,
                "seti": 0.6,
                "ark_nova": 0.5,
                "inventions": 0.5,
                "frostpunk": 0.4,
            },
        }
        s = pref.get(answers["map_aesthetic"], {}).get(game.id, 0.5)
        details["map_aesthetic"] = s * weights.get("map_aesthetic", 1.0)
        total += details["map_aesthetic"]

    # pressure_tolerance
    if _val("pressure_tolerance"):
        pref = {
            "low_pressure": {
                "ark_nova": 0.9,
                "inventions": 0.9,
                "seti": 0.8,
                "sleeping_gods": 0.6,
                "frostpunk": 0.3,
            },
            "moderate_pressure": {
                "ark_nova": 0.9,
                "inventions": 0.8,
                "seti": 0.8,
                "sleeping_gods": 0.7,
                "frostpunk": 0.6,
            },
            "high_pressure": {
                "frostpunk": 1.0,
                "sleeping_gods": 0.7,
                "seti": 0.5,
                "ark_nova": 0.4,
                "inventions": 0.4,
            },
            "swingy_pressure": {
                "sleeping_gods": 1.0,
                "seti": 0.7,
                "ark_nova": 0.6,
                "inventions": 0.6,
                "frostpunk": 0.6,
            },
        }
        s = pref.get(answers["pressure_tolerance"], {}).get(game.id, 0.5)
        details["pressure_tolerance"] = s * weights.get("pressure_tolerance", 1.0)
        total += details["pressure_tolerance"]

    # project_scope
    if _val("project_scope"):
        pref = {
            "perfect_system": {
                "inventions": 1.0,
                "ark_nova": 0.9,
                "seti": 0.7,
                "sleeping_gods": 0.5,
                "frostpunk": 0.5,
            },
            "explore_unknown": {
                "sleeping_gods": 1.0,
                "seti": 0.9,
                "ark_nova": 0.5,
                "inventions": 0.5,
                "frostpunk": 0.5,
            },
            "keep_alive": {
                "frostpunk": 1.0,
                "sleeping_gods": 0.7,
                "seti": 0.4,
                "ark_nova": 0.4,
                "inventions": 0.4,
            },
            "build_sanctuary": {
                "ark_nova": 1.0,
                "sleeping_gods": 0.6,
                "seti": 0.4,
                "inventions": 0.4,
                "frostpunk": 0.3,
            },
            "spark_ideas": {
                "inventions": 1.0,
                "ark_nova": 0.7,
                "seti": 0.6,
                "sleeping_gods": 0.5,
                "frostpunk": 0.4,
            },
        }
        s = pref.get(answers["project_scope"], {}).get(game.id, 0.5)
        details["project_scope"] = s * weights.get("project_scope", 1.0)
        total += details["project_scope"]

    # risk_attitude
    if _val("risk_attitude"):
        pref = {
            "avoid_variance": {
                "inventions": 0.9,
                "ark_nova": 0.9,
                "seti": 0.8,
                "frostpunk": 0.4,
                "sleeping_gods": 0.5,
            },
            "measured_risks": {
                "ark_nova": 0.9,
                "inventions": 0.8,
                "seti": 0.7,
                "sleeping_gods": 0.7,
                "frostpunk": 0.6,
            },
            "push_luck": {
                "sleeping_gods": 1.0,
                "frostpunk": 0.8,
                "seti": 0.6,
                "ark_nova": 0.5,
                "inventions": 0.5,
            },
        }
        s = pref.get(answers["risk_attitude"], {}).get(game.id, 0.5)
        details["risk_attitude"] = s * weights.get("risk_attitude", 1.0)
        total += details["risk_attitude"]

    # puzzle_flavor
    if _val("puzzle_flavor"):
        pref = {
            "efficiency_engine": {
                "ark_nova": 1.0,
                "inventions": 1.0,
                "seti": 0.8,
                "frostpunk": 0.5,
                "sleeping_gods": 0.5,
            },
            "navigation": {
                "sleeping_gods": 1.0,
                "seti": 0.7,
                "ark_nova": 0.5,
                "inventions": 0.5,
                "frostpunk": 0.5,
            },
            "workforce": {
                "seti": 1.0,
                "frostpunk": 0.9,
                "ark_nova": 0.6,
                "inventions": 0.6,
                "sleeping_gods": 0.6,
            },
            "multiobjective": {
                "inventions": 1.0,
                "ark_nova": 0.9,
                "seti": 0.7,
                "frostpunk": 0.6,
                "sleeping_gods": 0.6,
            },
            "narrative_problem": {
                "sleeping_gods": 1.0,
                "frostpunk": 0.8,
                "seti": 0.5,
                "ark_nova": 0.4,
                "inventions": 0.4,
            },
        }
        s = pref.get(answers["puzzle_flavor"], {}).get(game.id, 0.5)
        details["puzzle_flavor"] = s * weights.get("puzzle_flavor", 1.0)
        total += details["puzzle_flavor"]

    # moral_dilemmas
    if _val("moral_dilemmas"):
        pref = {
            "love_morals": {
                "frostpunk": 1.0,
                "sleeping_gods": 0.8,
                "inventions": 0.5,
                "ark_nova": 0.5,
                "seti": 0.5,
            },
            "ok_if_earned": {
                "sleeping_gods": 0.9,
                "frostpunk": 0.8,
                "inventions": 0.6,
                "ark_nova": 0.6,
                "seti": 0.6,
            },
            "prefer_clean": {
                "ark_nova": 0.9,
                "inventions": 0.9,
                "seti": 0.8,
                "sleeping_gods": 0.5,
                "frostpunk": 0.3,
            },
        }
        s = pref.get(answers["moral_dilemmas"], {}).get(game.id, 0.5)
        details["moral_dilemmas"] = s * weights.get("moral_dilemmas", 1.0)
        total += details["moral_dilemmas"]

    # collection_drive
    if _val("collection_drive"):
        pref = {
            "love_sets": {
                "seti": 1.0,
                "ark_nova": 0.9,
                "inventions": 0.6,
                "sleeping_gods": 0.5,
                "frostpunk": 0.5,
            },
            "prefer_networks": {
                "inventions": 1.0,
                "ark_nova": 0.7,
                "seti": 0.6,
                "sleeping_gods": 0.5,
                "frostpunk": 0.5,
            },
            "either_sets_networks": {
                "ark_nova": 0.7,
                "inventions": 0.7,
                "seti": 0.6,
                "sleeping_gods": 0.6,
                "frostpunk": 0.6,
            },
        }
        s = pref.get(answers["collection_drive"], {}).get(game.id, 0.5)
        details["collection_drive"] = s * weights.get("collection_drive", 1.0)
        total += details["collection_drive"]

    # solo_vibes (softly reinforces solo-capable titles but still subtle)
    if _val("solo_vibes"):
        pref = {
            "quiet_engine": {
                "ark_nova": 0.9,
                "inventions": 0.9,
                "seti": 0.5,
                "sleeping_gods": 0.7,
                "frostpunk": 0.7,
            },
            "survival_challenge": {
                "frostpunk": 1.0,
                "sleeping_gods": 0.7,
                "ark_nova": 0.6,
                "inventions": 0.5,
                "seti": 0.3,
            },
            "story_escape": {
                "sleeping_gods": 1.0,
                "ark_nova": 0.6,
                "inventions": 0.5,
                "seti": 0.4,
                "frostpunk": 0.6,
            },
            "no_solo_tonight": {
                "seti": 0.8,
                "ark_nova": 0.5,
                "inventions": 0.5,
                "sleeping_gods": 0.4,
                "frostpunk": 0.4,
            },
        }
        s = pref.get(answers["solo_vibes"], {}).get(game.id, 0.5)
        details["solo_vibes"] = s * weights.get("solo_vibes", 1.0)
        total += details["solo_vibes"]

    return total, details


# ------------------------------ UI Helpers ----------------------------------
def tag_row(text, tooltip=None):
    st.markdown(f"<span class='tag'>{text}</span>", unsafe_allow_html=True)
    if tooltip:
        st.tooltip(tooltip)


def game_card(game: Game, score: float, details: Dict[str, float]):
    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            if game.image_url:
                st.image(game.image_url, use_container_width=True)
            else:
                st.write("🧩")
        with cols[1]:
            st.subheader(f"{game.name}")
            st.caption(", ".join(game.themes) if game.themes else "—")
            st.progress(min(1.0, score / 7.0), text=f"Match score: {score:.2f}")
            # quick facts
            st.write(
                f"**Players:** {game.players_min}–{game.players_max}"
                f" | **Solo:** {'Yes' if game.solo else 'No'}"
                f" | **Mode:** {'Co-op' if game.coop else 'Competitive'}"
                f" | **Time:** {game.playtime_min}–{game.playtime_max} min"
                f" | **Weight:** {game.complexity:.1f}/5"
            )
            with st.expander("Why this score? (per-question contributions)"):
                df = pd.DataFrame([details]).T.reset_index()
                df.columns = ["question", "contribution"]
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={"contribution": {"format": "{:.2f}"}},
                )
            if game.notes:
                st.caption(game.notes)


# ------------------------------ Streamlit App -------------------------------
st.set_page_config(page_title="Board Game Recommender", page_icon="🎲", layout="wide")
st.markdown(
    """
<style>
.tag {
  display: inline-block;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  color: #3730a3;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 12px;
  margin-right: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎲 Board Game Recommender")

# Top bar: admin login and dataset status
right = st.sidebar
right.header("Controls")
mode = right.radio("Mode", ["Survey", "Admin"], horizontal=False)
if mode == "Admin":
    pw = right.text_input(
        "Admin password", type="password", help="Set ADMIN_PASSWORD env var to change"
    )
    if pw != ADMIN_PASSWORD:
        st.warning("Enter the correct admin password to edit settings.")
        st.stop()

# Load data
games = load_games()
if not games:
    st.error("No games defined. Add at least one in Admin mode.")
    st.stop()
qs = load_questions()

# Main content
if mode == "Survey":
    st.subheader("Tell us a bit, we'll pick the best fit")
    with st.form("survey_form", clear_on_submit=False):
        user_name = st.text_input("Your name (for saving results)", max_chars=64)
        answers = {}
        for q in qs.questions:
            key = q["key"]
            opts = q["options"]
            labels = [o["label"] for o in opts]
            values = [o["value"] for o in opts]
            default_index = 0
            selected_label = st.selectbox(
                q["text"], labels, index=default_index, key=f"q_{key}"
            )
            v = values[labels.index(selected_label)]
            answers[key] = v

        submitted = st.form_submit_button("Get my matches ➜", type="primary")

    if submitted:
        weights = qs.weights
        scores = []
        for g in games:
            s, det = score_game(g, answers, weights)
            scores.append((g, s, det))

        scores.sort(key=lambda x: x[1], reverse=True)
        topk = scores[:3]

        st.success(
            f"Top recommendation: **{topk[0][0].name}** (score {topk[0][1]:.2f})"
        )
        cols = st.columns(3)
        for i, (g, s, det) in enumerate(topk):
            with cols[i]:
                game_card(g, s, det)

        # save response
        if user_name.strip():
            row = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "user": user_name.strip(),
                "answers_json": json.dumps(answers, ensure_ascii=False),
                "ranking_json": json.dumps(
                    [{"game_id": g.id, "score": s} for g, s, _ in scores],
                    ensure_ascii=False,
                ),
            }
            append_response(row)
            # (kept) Existing caption
            st.caption("Saved to responses.csv")
        else:
            st.caption("Tip: enter your name to save your result.")
else:
    st.subheader("Admin")
    # ✨ NEW: add Leaderboard tab without removing others
    tabs = st.tabs(["Games", "Questions", "Responses", "Leaderboard"])

    with tabs[0]:
        st.markdown("### Current games")
        for i, g in enumerate(games):
            with st.expander(f"{g.name}"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.write(
                        f"**Players:** {g.players_min}–{g.players_max}  |  **Solo:** {g.solo}  |  **Coop:** {g.coop}"
                    )
                    st.write(
                        f"**Playtime:** {g.playtime_min}–{g.playtime_max}  |  **Weight:** {g.complexity}"
                    )
                    st.write(f"**Themes:** {', '.join(g.themes)}")
                    st.write(f"**Mechanics:** {', '.join(g.mechanics)}")
                    st.write(f"**Image URL:** {g.image_url or '—'}")
                    st.write(f"**Notes:** {g.notes or '—'}")
                with c2:
                    if st.button(f"Delete '{g.name}'", key=f"del_{g.id}"):
                        del games[i]
                        save_games(games)
                        st.rerun()

        st.divider()
        st.markdown("### Add / Edit a game")
        with st.form("add_game"):
            is_edit = st.checkbox(
                "Edit existing", value=False, help="Enable to update a game by id"
            )
            gid = st.text_input("ID (unique, snake_case)", value="new_game")
            name = st.text_input("Name", value="New Game")
            bgg = st.text_input("BGG slug (optional)", value="")
            pmin, pmax = st.columns(2)
            players_min = pmin.number_input(
                "Players min", min_value=1, max_value=8, value=1, step=1
            )
            players_max = pmax.number_input(
                "Players max", min_value=1, max_value=12, value=4, step=1
            )
            solo, coop = st.columns(2)
            solo_b = solo.checkbox("Has solo", value=False)
            coop_b = coop.checkbox("Is cooperative", value=False)
            tmin, tmax = st.columns(2)
            playtime_min = tmin.number_input(
                "Playtime min (min)", min_value=10, max_value=600, value=60, step=5
            )
            playtime_max = tmax.number_input(
                "Playtime max (min)", min_value=10, max_value=600, value=120, step=5
            )
            complexity = st.slider(
                "Complexity (1–5)", min_value=1.0, max_value=5.0, value=3.0, step=0.1
            )
            themes = st.text_input("Themes (comma-separated)", value="")
            mechanics = st.text_input("Mechanics (comma-separated)", value="")
            image_url = st.text_input("Image URL", value="")
            notes = st.text_area("Notes", value="")
            submitted = st.form_submit_button("Save game", type="primary")
            if submitted:
                new_game = Game(
                    id=gid.strip(),
                    name=name.strip(),
                    bgg_slug=bgg.strip(),
                    players_min=int(players_min),
                    players_max=int(players_max),
                    solo=bool(solo_b),
                    coop=bool(coop_b),
                    playtime_min=int(playtime_min),
                    playtime_max=int(playtime_max),
                    complexity=float(complexity),
                    themes=[t.strip() for t in themes.split(",") if t.strip()],
                    mechanics=[m.strip() for m in mechanics.split(",") if m.strip()],
                    image_url=image_url.strip(),
                    notes=notes.strip(),
                )
                if is_edit:
                    # replace by id
                    replaced = False
                    for i, g in enumerate(games):
                        if g.id == new_game.id:
                            games[i] = new_game
                            replaced = True
                            break
                    if not replaced:
                        st.error(f"No game with id={new_game.id} to edit.")
                    else:
                        save_games(games)
                        st.success(f"Updated {new_game.name}")
                        st.rerun()
                else:
                    games.append(new_game)
                    save_games(games)
                    st.success(f"Added {new_game.name}")
                    st.rerun()

    with tabs[1]:
        st.markdown("### Questions & Weights")
        st.caption(
            "You can use the fallback set below or ask GPT to propose a fresh set tailored to your current games."
        )
        st.write("**Current weights:**")
        for k, v in qs.weights.items():
            qs.weights[k] = st.slider(
                f"Weight for '{k}'",
                min_value=0.0,
                max_value=2.0,
                value=float(v),
                step=0.1,
            )
        if st.button("Save weights"):
            save_questions(qs)
            st.success("Weights saved.")

        st.divider()
        st.write("**Current questions:**")
        for i, q in enumerate(qs.questions):
            with st.expander(q.get("text", f"Q{i+1}")):
                new_text = st.text_input(
                    "Question text", value=q.get("text", ""), key=f"qt_{i}"
                )
                qs.questions[i]["text"] = new_text
                # options editor
                opt_df = pd.DataFrame(q.get("options", []))
                st.dataframe(opt_df, hide_index=True, use_container_width=True)
        if st.button("Save question text (options unchanged)"):
            save_questions(qs)
            st.success("Questions saved.")

        st.divider()
        st.write("**Generate new questions via GPT**")
        n_q = st.slider("How many questions?", min_value=5, max_value=12, value=8)
        if st.button("Ask GPT to propose questions", type="primary"):
            new_qs = propose_questions_via_gpt(games, n=n_q)
            # keep old weights if GPT didn't provide any
            if not new_qs.weights:
                new_qs.weights = qs.weights
            save_questions(new_qs)
            st.success("New questions saved from GPT.")
            st.rerun()

    with tabs[2]:
        st.markdown("### Responses")
        if RESPONSES_CSV.exists():
            df = pd.read_csv(RESPONSES_CSV)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                data=RESPONSES_CSV.read_bytes(),
                file_name="responses.csv",
            )
        else:
            st.info("No responses yet.")

    # ✨ NEW: Leaderboard tab (Top 3 by total points across ALL Supabase responses)
    with tabs[3]:
        st.markdown("### 🏆 Top 3 Games (Total Points from Supabase)")
        if not supabase:
            st.info(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in secrets/.env to enable the leaderboard."
            )
        else:
            all_responses = fetch_all_responses()
            if all_responses.empty:
                st.info("No responses yet in Supabase.")
            else:
                # Compute totals (sum only; ignore per-question details)
                weights = qs.weights
                totals = {g.id: 0.0 for g in games}
                for _, row in all_responses.iterrows():
                    answers = row.get("answers_json", {})
                    if isinstance(answers, str):
                        try:
                            answers = json.loads(answers)
                        except Exception:
                            answers = {}
                    # score each game for this response
                    for g in games:
                        s, _ = score_game(g, answers, weights)
                        totals[g.id] += s

                leaderboard = [
                    {"Game": g.name, "Total Points": round(totals[g.id], 2)}
                    for g in games
                ]
                top3 = sorted(
                    leaderboard, key=lambda x: x["Total Points"], reverse=True
                )[:3]
                st.dataframe(
                    pd.DataFrame(top3), hide_index=True, use_container_width=True
                )
