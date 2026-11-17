import pandas as pd

from src.sample_data import make
from src.train import train


def main():
    df = make()
    df.to_csv("data/raw/trips.csv", index=False)
    _, _, score = train(df)
    print(f"trip label macro F1: {score:.4f}")


if __name__ == "__main__":
    main()
