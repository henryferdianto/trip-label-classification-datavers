"""Train a multi-class trip label classifier."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.features import build


def prepare(df: pd.DataFrame, target: str = "trip_label"):
    feats = build(df)
    X = feats.drop(columns=[target]).select_dtypes("number")
    return X, feats[target]


def train(df: pd.DataFrame, target: str = "trip_label"):
    X, y = prepare(df, target)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_tr, y_tr)
    macro_f1 = f1_score(y_te, model.predict(X_te), average="macro")
    return model, list(X.columns), float(macro_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/trips.csv")
    ap.add_argument("--target", default="trip_label")
    args = ap.parse_args()
    if not Path(args.input).exists():
        from src.sample_data import main as make_sample

        make_sample()
    model, columns, score = train(pd.read_csv(args.input), args.target)
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": columns}, "models/trip_label.joblib")
    print(f"macro F1: {score:.4f}")


if __name__ == "__main__":
    main()
