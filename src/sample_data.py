"""Generate synthetic ride-hailing trips with a predictable label signal."""
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pickup_lat = rng.uniform(-6.4, -6.1, n)
    pickup_lon = rng.uniform(106.6, 106.95, n)
    dropoff_lat = pickup_lat + rng.normal(0, 0.08, n)
    dropoff_lon = pickup_lon + rng.normal(0, 0.09, n)
    distance = np.hypot((dropoff_lat - pickup_lat) * 111,
                        (dropoff_lon - pickup_lon) * 111)
    hour = rng.integers(0, 24, n)
    duration = distance / rng.uniform(0.3, 0.7, n) + rng.uniform(2, 10, n)

    score = distance + 0.15 * duration + 0.6 * ((hour >= 17) & (hour <= 21))
    threshold_lo, threshold_hi = np.quantile(score, [0.35, 0.7])
    label = np.where(score <= threshold_lo, "short",
                     np.where(score <= threshold_hi, "medium", "long"))
    base_time = pd.Timestamp("2026-03-01")
    pickup_time = [base_time + pd.Timedelta(hours=int(h)) + pd.Timedelta(days=int(d))
                   for h, d in zip(hour, rng.integers(0, 7, n))]

    return pd.DataFrame({
        "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat, "dropoff_lon": dropoff_lon,
        "pickup_time": [t.isoformat() for t in pickup_time],
        "distance_km": distance, "duration_min": duration,
        "trip_label": label,
    })


def main() -> Path:
    out = Path("data/raw/trips.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"wrote trips -> {out}")
    return out


if __name__ == "__main__":
    main()
