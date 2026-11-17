"""Evaluate trip label predictions."""
import argparse
import pandas as pd
from sklearn.metrics import classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.pred)
    print(classification_report(df["y_true"], df["y_pred"]))


if __name__ == "__main__":
    main()
