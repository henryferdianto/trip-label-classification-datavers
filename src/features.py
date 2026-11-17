"""Spatiotemporal feature engineering for ride-hailing trips."""
import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p = np.pi / 180
    a = (
        np.sin((lat2 - lat1) * p / 2) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(a))


def build(df: pd.DataFrame) -> pd.DataFrame:
    if {"pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"} <= set(df.columns):
        df["distance_km"] = haversine(
            df.pickup_lat, df.pickup_lon, df.dropoff_lat, df.dropoff_lon
        )
    if "pickup_time" in df.columns:
        ts = pd.to_datetime(df["pickup_time"])
        df["hour"] = ts.dt.hour
        df["weekday"] = ts.dt.weekday
    return df
