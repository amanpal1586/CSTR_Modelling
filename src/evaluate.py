"""Controlled experiments: surrogate vs physics simulation.

Produces metrics (MAE, RMSE, R2), residual analysis, feature importance,
trajectory plots, and a computational-time benchmark.
"""
import json, time, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .cstr_model import simulate, OP_COLS, STATE_COLS
from .generate_data import T_GRID
from .train_surrogate import FEATURES, feature_importance

FIG = Path("results/figures")


def metrics(y, yhat):
    out = {}
    for j, s in enumerate(STATE_COLS):
        out[s] = dict(MAE=float(mean_absolute_error(y[:, j], yhat[:, j])),
                      RMSE=float(np.sqrt(mean_squared_error(y[:, j], yhat[:, j]))),
                      R2=float(r2_score(y[:, j], yhat[:, j])))
    return out


def predict_traj(rf, op, t=T_GRID):
    X = pd.DataFrame({"t": t, **{k: op[k] for k in OP_COLS}})[FEATURES]
    return rf.predict(X)


def plot_importance(rf):
    fi = feature_importance(rf)[::-1]
    plt.figure(figsize=(6, 3.5)); plt.barh(fi.index, fi.values, color="#3b7dd8")
    plt.xlabel("Impurity-based importance"); plt.title("Random Forest feature importance")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=150); plt.close()


def plot_parity_residuals(df, rf, split, tag):
    d = df[df.split == split]; yhat = rf.predict(d[FEATURES]); y = d[STATE_COLS].values
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    for j, s in enumerate(STATE_COLS):
        ax[0, j].scatter(y[:, j], yhat[:, j], s=4, alpha=.3)
        lo, hi = y[:, j].min(), y[:, j].max(); ax[0, j].plot([lo, hi], [lo, hi], "r--", lw=1)
        ax[0, j].set(xlabel=f"Actual {s}", ylabel=f"Predicted {s}", title=f"{s}: parity ({tag})")
        ax[1, j].hist(yhat[:, j] - y[:, j], bins=50, color="#888")
        ax[1, j].set(xlabel=f"Residual (pred - actual) {s}", title=f"{s}: residuals ({tag})")
    plt.tight_layout(); plt.savefig(FIG / f"parity_residuals_{tag}.png", dpi=150); plt.close()


def plot_trajectories(df, rf, split, tag, n=3, seed=1):
    ids = np.random.default_rng(seed).choice(df[df.split == split].case_id.unique(), n, replace=False)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 6), sharex=True)
    for c, cid in enumerate(ids):
        d = df[df.case_id == cid]; yhat = rf.predict(d[FEATURES])
        for j, s in enumerate(STATE_COLS):
            ax[j, c].plot(d.t, d[s], "k-", label="Physics (ODE)")
            ax[j, c].plot(d.t, yhat[:, j], "r--", label="RF surrogate")
            ax[j, c].set_ylabel(s); ax[j, c].set_title(f"case {cid}" if j == 0 else "")
        ax[1, c].set_xlabel("time [min]")
    ax[0, 0].legend(); plt.suptitle(f"Actual vs predicted trajectories ({tag})")
    plt.tight_layout(); plt.savefig(FIG / f"trajectories_{tag}.png", dpi=150); plt.close()


def benchmark(rf, df, n_cases=100):
    ops = [df[df.case_id == c].iloc[0][OP_COLS].to_dict() for c in df[df.split == "test"].case_id.unique()[:n_cases]]
    t0 = time.perf_counter(); [simulate(o, T_GRID) for o in ops]; t_phys = time.perf_counter() - t0
    X = pd.concat([pd.DataFrame({"t": T_GRID, **{k: o[k] for k in OP_COLS}}) for o in ops])[FEATURES]
    t0 = time.perf_counter(); rf.predict(X); t_batch = time.perf_counter() - t0
    t0 = time.perf_counter(); [predict_traj(rf, o) for o in ops]; t_single = time.perf_counter() - t0
    return dict(n_cases=len(ops), physics_s=t_phys, surrogate_batched_s=t_batch, surrogate_per_case_s=t_single,
                speedup_batched=t_phys / t_batch, speedup_per_case=t_phys / t_single)


def main(data="data/cstr_dataset.csv", model="models/rf_surrogate.joblib"):
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data); rf = joblib.load(model)
    res = {}
    for split in ["train", "val", "test", "extrap"]:
        d = df[df.split == split]; res[split] = metrics(d[STATE_COLS].values, rf.predict(d[FEATURES]))
    res["benchmark"] = benchmark(rf, df)
    plot_importance(rf)
    for sp, tag in [("test", "test"), ("extrap", "extrap")]:
        plot_parity_residuals(df, rf, sp, tag); plot_trajectories(df, rf, sp, tag)
    Path("results/metrics.json").write_text(json.dumps(res, indent=2))
    rows = [{"split": sp, "state": s, **res[sp][s]} for sp in ["train", "val", "test", "extrap"] for s in STATE_COLS]
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print(json.dumps(res["benchmark"], indent=2))
    return res


if __name__ == "__main__":
    main()
