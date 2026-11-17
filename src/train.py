"""Train a multi-class trip label classifier."""
import argparse
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.features import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target", default="trip_label")
    args = ap.parse_args()

    df = build(pd.read_csv(args.input))
    X = pd.get_dummies(df.drop(columns=[args.target]), drop_first=True)
    y = df[args.target]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = LGBMClassifier(
        n_estimators=800, learning_rate=0.03, class_weight="balanced", random_state=42
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"macro F1: {f1_score(y_te, pred, average='macro'):.4f}")
    joblib.dump(model, "models/trip_label.joblib")


if __name__ == "__main__":
    main()
