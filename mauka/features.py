# Turns the hourly data into a table the models can learn from.
#
# One row = one hour of one day. We use what we know at that hour (plus a bit of
# recent history) to guess if the SAME hour TOMORROW will be a good one.

import pandas as pd

from .config import THRESHOLDS

FEATURES = [
    "hour", "month",
    "pm2_5", "us_aqi", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "wind_speed_10m",
    "aqi_day_mean", "aqi_prev_day_mean", "aqi_3d_mean", "temp_day_max",
    "ok_now",
]


def add_features(df, th=THRESHOLDS):
    d = df.sort_index().asfreq("h").copy()

    # is this hour good? (1 = yes, 0 = no, blank if data is missing)
    has_data = d[["us_aqi", "apparent_temperature", "precipitation"]].notna().all(axis=1)
    good = ((d["us_aqi"] <= th.max_us_aqi)
            & (d["apparent_temperature"] <= th.max_apparent_temp_c)
            & (d["apparent_temperature"] >= th.min_apparent_temp_c)
            & (d["precipitation"] <= th.max_precip_mm))
    d["ok_now"] = good.astype(float).where(has_data)

    # daily numbers: average air quality, hottest "feels like"
    day = d.index.normalize()
    all_days = pd.date_range(day.min(), day.max(), freq="D")
    aqi_by_day = d["us_aqi"].groupby(day).mean().reindex(all_days)
    peak_temp_by_day = d["apparent_temperature"].groupby(day).max().reindex(all_days)

    d["aqi_day_mean"] = aqi_by_day.reindex(day).to_numpy()
    d["aqi_prev_day_mean"] = aqi_by_day.shift(1).reindex(day).to_numpy()
    d["aqi_3d_mean"] = aqi_by_day.rolling(3, min_periods=2).mean().reindex(day).to_numpy()
    d["temp_day_max"] = peak_temp_by_day.reindex(day).to_numpy()
    d["hour"] = d.index.hour
    d["month"] = d.index.month

    # what we want to predict: the same hour, 24 hours later
    d["target"] = d["ok_now"].shift(-24)
    return d


def build_training(df, th=THRESHOLDS):
    d = add_features(df, th)
    keep = (d["hour"].between(th.first_hour, th.last_hour)
            & d[FEATURES].notna().all(axis=1)
            & d["target"].notna())
    X = d.loc[keep, FEATURES].copy()
    y = d.loc[keep, "target"].astype(int)
    return X, y


def live_features(recent, today=None, th=THRESHOLDS):
    # rows for today's waking hours, used to predict tomorrow
    d = add_features(recent, th)
    usable = d[FEATURES].notna().all(axis=1) & d["hour"].between(th.first_hour, th.last_hour)
    days = d.index.normalize()

    if today is None:
        today = pd.Timestamp.now().normalize()
    today = pd.Timestamp(today).normalize()

    rows = d[usable & (days == today)]
    if rows.empty:
        # today isn't complete yet, so use the latest day that is
        complete_days = days[usable.to_numpy()]
        if len(complete_days) == 0:
            raise ValueError("Not enough recent data to build features.")
        today = complete_days.max()
        rows = d[usable & (days == today)]
    return rows[FEATURES], today
