# Streamlit Board Game Recommender (Plug‑and‑Play)

A simple platform that lets you:
- Define **candidate games** (admin)
- Generate or edit **survey questions** (optionally via GPT)
- Collect **user answers** with name
- Produce a **weighted match score** for each game with clear reasoning
- Persist responses to a CSV

## Quickstart

1. **Install** (use a fresh venv):
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (optional, only needed for GPT-generated questions):
   - `OPENAI_API_KEY=...` (for Chat Completions)
   - `ADMIN_PASSWORD=change-me` (to enter Admin Mode)

   You can also put them in a `.env` file (supported automatically).

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

4. Open the URL Streamlit prints (usually http://localhost:8501).

## Files
- `data/games.json` – starter game list (your five BGG picks are here)
- `data/questions.json` – fallback question set
- `responses.csv` – will be created as users submit surveys
- `app.py` – Streamlit app

## Notes
- If no `OPENAI_API_KEY` is provided, the app uses the bundled `questions.json`.
- Admin Mode (top right) lets you edit/add games, tweak weights, and call GPT to propose questions that **discriminate** between your current games.
- All math stays transparent; hover over scores to see feature-by-feature contributions.
- To add images, set `image_url` per game (BGG image link works).

---

**Created:** 2025-08-15 09:45
