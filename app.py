# Mauka: find your moment outside.
# Run with:  streamlit run app.py

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from mauka.config import CITIES, THRESHOLDS
from mauka.data import DataError, fetch_recent, load_history, synthetic_series
from mauka.features import live_features
from mauka.model import evaluate_models, fit_for_app, positive_proba, tabpfn_available
from mauka.nudge import (DEFAULT_MODEL, LANGUAGES, build_facts, find_windows,
                         generate_nudge, ollama_status, show_hour)

RESULTS_DIR = Path(__file__).parent / "results"
HISTORY_START = "2022-08-01"

# colours
INK = "#14232E"
PAPER = "#F2F5F6"
SKY = "#2F7FA8"
SMOG = "#8A7B6B"
LINE = "#D3DCE0"
MUTED = "#5B6B76"

st.set_page_config(page_title="Mauka", page_icon="🍃", layout="centered")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=Instrument+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stCaption {{ font-family: 'Instrument Sans', system-ui, sans-serif; color: {INK}; }}
.stApp {{ background: {PAPER}; }}
h1.mauka-title {{ font-family: 'Bricolage Grotesque', system-ui, sans-serif; font-weight: 700; font-size: 3.4rem;
  letter-spacing: -0.02em; margin: 0 0 .1rem 0; line-height: 1; }}
.mauka-sub {{ color: {MUTED}; margin: 0 0 1.6rem 0; font-size: 1.05rem; }}
.day-label {{ font-family: 'Bricolage Grotesque', system-ui, sans-serif; font-weight: 500; font-size: 1.15rem; margin: 0 0 .6rem 0; }}
.strip {{ display: flex; align-items: flex-end; gap: 4px; height: 96px; margin-bottom: 4px; }}
.cell {{ flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: stretch; height: 100%; }}
.bar {{ border-radius: 3px 3px 0 0; min-height: 10px; }}
.hours {{ display: flex; gap: 4px; margin-bottom: 1.4rem; }}
.hours span {{ flex: 1; text-align: center; font-size: .68rem; color: {MUTED}; }}
.win {{ border-top: 1px solid {LINE}; padding: .7rem 0; display: flex; justify-content: space-between; align-items: baseline; }}
.win:last-of-type {{ border-bottom: 1px solid {LINE}; }}
.win b {{ font-family: 'Bricolage Grotesque', system-ui, sans-serif; font-size: 1.25rem; font-weight: 500; }}
.win span {{ color: {MUTED}; }}
.nudge {{ border-left: 4px solid {SKY}; padding: .4rem 0 .4rem 1rem; margin: 1.6rem 0 .3rem 0; font-size: 1.12rem; line-height: 1.55; }}
.source {{ color: {MUTED}; font-size: .8rem; margin-bottom: 1.4rem; }}
.legend {{ color: {MUTED}; font-size: .8rem; margin: -.8rem 0 1.2rem 0; }}
@media (prefers-reduced-motion: no-preference) {{ .bar {{ transition: height .4s ease; }} }}
</style>
""", unsafe_allow_html=True)


def blend(color1, color2, amount):
    # a colour between color1 (amount=0) and color2 (amount=1)
    a = [int(color1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(color2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02x}" for x, y in zip(a, b))


def hour_strip(hours, probs):
    # one bar per hour: taller and bluer means a better chance
    bars = ""
    labels = ""
    for h, p in zip(hours, probs):
        height = 12 + int(p * 84)
        color = blend(SMOG, SKY, float(p))
        bars += (f'<div class="cell" title="{show_hour(h)}: {p:.0%} chance">'
                 f'<div class="bar" style="height:{height}px;background:{color}"></div></div>')
        labels += f"<span>{h:02d}</span>"
    return f'<div class="strip">{bars}</div><div class="hours">{labels}</div>'


# ---- sidebar settings ----
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Data", ["Live (Open-Meteo)", "Offline demo (synthetic data)"],
                    help="The demo needs no internet, but the data is made up.")
    cities = sorted(CITIES)
    city = st.selectbox("City", cities)

    model_choices = (["tabpfn"] if tabpfn_available() else []) + ["hgb", "logistic"]
    model_name = st.selectbox("Prediction model", model_choices,
                              help="TabPFN is the main model. The others are baselines.")
    language = st.selectbox("Nudge language", LANGUAGES)
    gemma = st.text_input("Local model (Ollama tag)", DEFAULT_MODEL)

    running, installed = ollama_status(gemma)
    if installed:
        st.caption(f"Local model ready: {gemma}")
    elif running:
        st.caption(f"Ollama is running but '{gemma}' isn't downloaded. Run: ollama pull {gemma}")
    else:
        st.caption("Ollama isn't running, so the nudge will use a plain template.")
    if not tabpfn_available():
        st.caption("TabPFN isn't installed (pip install tabpfn), so a baseline model is used.")

use_demo = mode.startswith("Offline")
lat, lon = CITIES[city]
today = pd.Timestamp.now().normalize()


# ---- load data and train the model ----
@st.cache_data(ttl=6 * 3600, show_spinner="Loading weather and air quality history...")
def get_data(city_name, demo):
    if demo:
        end = (today + pd.Timedelta(days=1)).date().isoformat()
        fake = synthetic_series("2023-10-01", end, seed=5)
        history = fake.loc[: today - pd.Timedelta(days=3)]
        recent = fake.loc[today - pd.Timedelta(days=4):]
        return history, recent
    end = (date.today() - timedelta(days=7)).isoformat()
    return load_history(city_name, lat, lon, HISTORY_START, end), fetch_recent(lat, lon)


@st.cache_resource(show_spinner="Training the model...")
def get_model(city_name, demo, name):
    history, _ = get_data(city_name, demo)
    return fit_for_app(history, name)


try:
    history, recent = get_data(city, use_demo)
    model = get_model(city, use_demo, model_name)
    today_rows, used_day = live_features(recent, today=today)
except (DataError, requests.RequestException, ValueError) as e:
    st.error(f"Could not get data for {city}: {e}")
    st.info("Check your internet connection, or pick 'Offline demo' in the sidebar.")
    st.stop()

hours = today_rows["hour"].astype(int).to_numpy()
probs = positive_proba(model, today_rows)
tomorrow = used_day + pd.Timedelta(days=1)
windows = find_windows(hours, probs, threshold=0.5, top=3)

# ---- the page ----
st.markdown('<h1 class="mauka-title">Mauka</h1>', unsafe_allow_html=True)
st.markdown('<p class="mauka-sub">Find your moment outside 🍃<br>'
            'A local-first AI guide to the clean-air hours</p>', unsafe_allow_html=True)

if use_demo:
    st.warning("Offline demo: this data is made up. It only shows how the app works.")

st.markdown(f'<p class="day-label">{city}, {tomorrow.strftime("%A %d %B")}</p>', unsafe_allow_html=True)
st.markdown(hour_strip(hours, probs), unsafe_allow_html=True)
st.markdown(
    f'<p class="legend">Taller and bluer means a better chance of a good hour. A good hour has '
    f'US AQI of {THRESHOLDS.max_us_aqi:.0f} or less, feels like {THRESHOLDS.min_apparent_temp_c:.0f} to '
    f'{THRESHOLDS.max_apparent_temp_c:.0f} °C, and no real rain.</p>', unsafe_allow_html=True)

if windows:
    for w in windows:
        st.markdown(f'<div class="win"><b>{show_hour(w["start"])} to {show_hour(w["end"])}</b>'
                    f'<span>{w["mean_p"]:.0%} chance · {w["hours"]} h</span></div>',
                    unsafe_allow_html=True)
else:
    st.markdown("No hour has a 50% chance or better tomorrow. Keep any time outside short.")

first_row = today_rows.iloc[0]
facts = build_facts(city, windows, float(first_row["aqi_day_mean"]), float(first_row["temp_day_max"]))
text, writer = generate_nudge(facts, language, gemma)
st.markdown(f'<div class="nudge">{text}</div><div class="source">Written by: {writer}</div>',
            unsafe_allow_html=True)

with st.expander("How accurate is this?"):
    results_path = RESULTS_DIR / f"metrics_{'synthetic' if use_demo else city.lower()}.json"

    if st.button("Run a quick check now"):
        names = ["persistence", "climatology", "logistic", "hgb"]
        if model_name == "tabpfn":
            names.append("tabpfn")
        with st.spinner("Comparing the models on the most recent part of the history..."):
            out = evaluate_models(history, models=names, max_train=3000, max_test=2000)
        out["city"] = city
        RESULTS_DIR.mkdir(exist_ok=True)
        results_path.write_text(json.dumps(out, indent=2))

    if results_path.exists():
        out = json.loads(results_path.read_text())
        st.caption(f"Tested on {out['n_test']} hours from {out['test_period'][0]} to "
                   f"{out['test_period'][1]}, using a model trained only on earlier data. "
                   f"{out['test_base_rate']:.0%} of those hours were good. "
                   "Higher AUC, accuracy and F1 are better. Lower Brier is better.")
        st.dataframe(pd.DataFrame(out["results"]).round(3), hide_index=True, width="stretch")
        for name, reason in out.get("skipped", {}).items():
            st.caption(f"Skipped {name}: {reason}")
    else:
        st.caption("No results yet. Press the button, or run: python scripts/evaluate.py --city " + city)

    st.caption("Limits: this uses gridded model data (ERA5 weather, CAMS air quality), not street-level "
               "sensors. The thresholds are a simple idea of 'good', not medical advice.")

st.caption("Weather and air quality data: Open-Meteo.com (CC BY 4.0). Air quality from Copernicus CAMS.")
