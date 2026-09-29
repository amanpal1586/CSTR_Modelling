"""Train a Random Forest surrogate: (t, CA0, T0, Tc, q, CAf, Tf) -> (CA(t), T(t))."""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from .cstr_model import OP_COLS, STATE_COLS

FEATURES = ["t"] + OP_COLS


def train(df, n_estimators=200, min_samples_leaf=2, seed=0):
    tr = df[df.split == "train"]
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                               n_jobs=-1, random_state=seed)
    rf.fit(tr[FEATURES], tr[STATE_COLS])
    return rf


def feature_importance(rf):
    return pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)


def main(data="data/cstr_dataset.csv", out="models/rf_surrogate.joblib"):
    df = pd.read_csv(data)
    rf = train(df)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out)
    fi = feature_importance(rf)
    Path("results").mkdir(exist_ok=True)
    fi.to_csv("results/feature_importance.csv", header=["importance"])
    print("Model saved ->", out); print(fi.round(4).to_string())
    return rf


if __name__ == "__main__":
    main()
