import numpy as np

from src.features import build, haversine
from src.sample_data import make
from src.train import prepare, train


def test_haversine_known_distance():
    d = haversine(-6.2, 106.8, -6.2, 106.8)
    assert d == 0
    d2 = haversine(0, 0, 0, 1)
    assert 110 < d2 < 112


def test_build_drops_time_and_adds_features():
    out = build(make(100))
    assert "pickup_time" not in out.columns
    assert {"hour", "weekday", "is_peak"} <= set(out.columns)


def test_pipeline_macro_f1():
    _, _, score = train(make(800))
    assert score > 0.5
