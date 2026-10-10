# Gets weather and air quality data from Open-Meteo (free, no API key needed).
# Open-Meteo asks for credit (CC BY 4.0). Air quality is from Copernicus CAMS.

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
                "precipitation", "wind_speed_10m"]
AIR_VARS = ["pm2_5", "us_aqi"]
ALL_VARS = WEATHER_VARS + AIR_VARS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DataError(Exception):
    pass


def get_json(url, params):
    # try up to 3 times in case the network is flaky
    error = ""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            error = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 400:
                break  # a bad request won't work on retry
        except requests.RequestException as e:
            error = str(e)
        time.sleep(2 * (attempt + 1))
    raise DataError(f"Open-Meteo request failed: {error}")


def to_table(data, variables):
    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        raise DataError("Open-Meteo sent back no hourly data.")
    table = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for name in variables:
        table[name] = hourly.get(name)
    return table.set_index("time")


def date_ranges(start, end, days=180):
    # split a long period into smaller requests
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    while first <= last:
        chunk_end = min(first + pd.Timedelta(days=days - 1), last)
        yield first.date().isoformat(), chunk_end.date().isoformat()
        first = chunk_end + pd.Timedelta(days=1)


def tidy(df):
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=ALL_VARS)
    if df.empty:
        raise DataError("No rows left after joining weather and air quality.")
    return df[ALL_VARS].astype(float)


def fetch_history(lat, lon, start, end):
    # weather history joined with air quality history
    weather_parts, air_parts, skipped = [], [], []
    for s, e in date_ranges(start, end):
        w = get_json(ARCHIVE_URL, dict(latitude=lat, longitude=lon, start_date=s, end_date=e,
                                       hourly=",".join(WEATHER_VARS), timezone="auto"))
        weather_parts.append(to_table(w, WEATHER_VARS))
        try:
            a = get_json(AIR_URL, dict(latitude=lat, longitude=lon, start_date=s, end_date=e,
                                       hourly=",".join(AIR_VARS), timezone="auto"))
            air_parts.append(to_table(a, AIR_VARS))
        except DataError:
            skipped.append((s, e))  # air quality history may not go back this far
    if not air_parts:
        raise DataError("No air quality data came back. Try a later --start date. "
                        f"Skipped: {skipped[:3]}")
    both = pd.concat(weather_parts).join(pd.concat(air_parts), how="inner")
    return tidy(both)


def fetch_recent(lat, lon, past_days=4, forecast_days=2):
    # last few days (what happened) plus the next day or two (forecast)
    w = get_json(FORECAST_URL, dict(latitude=lat, longitude=lon, past_days=past_days,
                                    forecast_days=forecast_days,
                                    hourly=",".join(WEATHER_VARS), timezone="auto"))
    a = get_json(AIR_URL, dict(latitude=lat, longitude=lon, past_days=past_days,
                               forecast_days=forecast_days,
                               hourly=",".join(AIR_VARS), timezone="auto"))
    return tidy(to_table(w, WEATHER_VARS).join(to_table(a, AIR_VARS), how="inner"))


def load_history(name, lat, lon, start, end):
    # download once, then reuse the saved CSV so later runs work offline
    DATA_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    path = DATA_DIR / f"{slug}_{start}_{end}.csv"
    if path.exists():
        return pd.read_csv(path, index_col="time", parse_dates=True)
    df = fetch_history(lat, lon, start, end)
    df.to_csv(path, index_label="time")
    return df


# ---- fake data, only for tests and the offline demo ----

PM_POINTS = [0, 9.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]
AQI_POINTS = [0, 50, 100, 150, 200, 300, 400, 500]


def synthetic_series(start, end, seed=7):
    # made-up data that looks a bit like north India:
    # smoggy winters, hot before the monsoon, rain in July-September
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    days = times.normalize().unique()

    def drifting_noise(keep, spread):
        # today's value depends a bit on yesterday's
        x = np.zeros(len(days))
        for i in range(1, len(days)):
            x[i] = keep * x[i - 1] + rng.normal(0, spread)
        return x

    smog_noise = drifting_noise(0.8, 0.35)
    wet_noise = drifting_noise(0.6, 0.5)
    temp_noise = drifting_noise(0.85, 1.0)

    day_number = pd.Series(np.arange(len(days)), index=days).reindex(times.normalize()).to_numpy()
    day_of_year = times.dayofyear.to_numpy()
    hour = times.hour.to_numpy()
    monsoon = (times.month >= 7) & (times.month <= 9)

    pm25 = 50 + 45 * np.cos(2 * np.pi * (day_of_year - 355) / 365)
    pm25 = np.where(monsoon, pm25 * 0.45, pm25)
    pm25 = pm25 * np.exp(smog_noise[day_number]) * (1 + 0.25 * np.cos(2 * np.pi * (hour - 7) / 24))
    pm25 = np.clip(pm25, 3, 450)
    aqi = np.interp(pm25, PM_POINTS, AQI_POINTS)

    temp = (25 + 9 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
            + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + temp_noise[day_number])
    humidity = np.clip(45 + 25 * monsoon - 18 * np.sin(2 * np.pi * (hour - 9) / 24)
                       + rng.normal(0, 6, len(times)), 15, 99)
    feels_like = temp + 3.5 * (humidity / 100) + rng.normal(0, 0.5, len(times))

    rain_chance = np.where(monsoon, 0.30, 0.03) + 0.1 * np.tanh(wet_noise[day_number])
    is_raining = rng.random(len(times)) < np.clip(rain_chance * 0.4, 0, 0.9)
    rain = np.where(is_raining, rng.exponential(2.0, len(times)), 0.0)
    wind = rng.gamma(3.0, 2.0, len(times))

    return pd.DataFrame({
        "temperature_2m": temp, "apparent_temperature": feels_like,
        "relative_humidity_2m": humidity, "precipitation": rain, "wind_speed_10m": wind,
        "pm2_5": pm25, "us_aqi": aqi,
    }, index=pd.DatetimeIndex(times, name="time"))
