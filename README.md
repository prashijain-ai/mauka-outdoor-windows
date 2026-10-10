# Mauka (मौका) 🍃

**Find your moment outside. A local-first AI guide to the clean-air hours of a hot, hazy city, built on open data and open models that run on your own laptop.**

Built for the Hacktoberfest 2026 "Touch Grass" DEV challenge.

## What it does

1. Downloads hourly weather (ERA5) and air quality (CAMS) history for a city from [Open-Meteo](https://open-meteo.com) (free, no API key).
2. Defines a *good outdoor hour*: US AQI ≤ 100, feels-like 8–35 °C, no real rain (see `mauka/config.py`).
3. Trains **TabPFN** (a pretrained tabular model) to predict, for each waking hour of tomorrow, the chance it will be a good hour, from today's conditions plus short history.
4. Compares TabPFN with four baselines on a strictly later test period.
5. A small open-weight **Gemma** model, running locally through Ollama, turns the numbers into a short nudge in English, Hinglish or Hindi. It only sees numbers we computed. If Ollama is not running, a plain template is used and the app says so.

Nothing leaves your machine except the public Open-Meteo requests.

## Setup (macOS, Apple Silicon)

```bash
cd mauka
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# local model for the nudges
brew install ollama          # or download from ollama.com
ollama serve &               # leave running
ollama pull gemma3:1b        # about 1 GB; use a bigger Gemma if you have the RAM
```

Everything is inside this folder and `.venv`. Delete the folder (and `ollama rm gemma3:1b`) to remove it all.

### TabPFN notes
- `pip install tabpfn` downloads model weights on first use. Check the [TabPFN docs](https://github.com/PriorLabs/TabPFN) for any license or token step required for your version.
- If TabPFN is not installed, the app falls back to a baseline model and says so.

## Run

```bash
python scripts/evaluate.py --city Delhi     # fetch data, compare models, save results/
streamlit run app.py                        # the app
pytest -q                                   # tests
```

No internet? `python scripts/evaluate.py --demo` and the app's "Offline demo" mode use **synthetic** data. That is only for checking the app works; never quote its numbers.

Use another Gemma by setting `MAUKA_MODEL` (an Ollama tag) or editing the sidebar field.

## How the evaluation works (and what it does not prove)

- One row per (day, waking hour). Target: is the **same hour tomorrow** good?
- Chronological split: the last 25% of days are the test set, with a gap so no training target overlaps a test input.
- Baselines: *persistence* (tomorrow = today), *climatology* (historical rate for that month and hour), *logistic regression*, *gradient-boosted trees*.
- Metrics: ROC AUC, Brier score (calibration), accuracy, F1.

Limitations to state in your post:
- Gridded reanalysis and model data, not street-level sensors.
- Training uses archived "today" values; live use mixes observations and forecasts for the rest of today.
- The thresholds are a simple definition of "good", not medical advice.
- Results differ by city and season. Report yours, including where TabPFN does **not** win.

## What I could not test in the build environment
These need your machine: live Open-Meteo calls (the air-quality history start date may limit `--start`), TabPFN weights and speed, and Ollama/Gemma generation. Tests and the demo mode were run with synthetic data and a baseline model.

## Challenge checklist
- [ ] New public repo, started inside the challenge window (Oct 5 to 11); note any later commits in this README.
- [ ] Run `evaluate.py` for your city and paste the real table into the post.
- [ ] Take it outside for a day and report what happened.
- [ ] Demo video, English DEV post from the official template, tags `devchallenge` and `hf26challenge`.
- [ ] Post explains why open models matter here (local, private, inspectable).

## Credits
Weather and air quality: Open-Meteo.com (CC BY 4.0); air quality from Copernicus CAMS. Models: TabPFN (Prior Labs), Gemma (Google).
