"""Generate process data by simulating the CSTR across varied operating conditions.

Splits (by *operating condition*, never by time step, to avoid leakage):
  train / val / test : sampled from the nominal operating envelope
  extrap             : sampled from a shifted envelope the surrogate never sees
                       (used to test generalisation to unseen operating conditions)
"""
import numpy as np, pandas as pd
from pathlib import Path
from .cstr_model import simulate, OP_COLS

T_END, N_STEPS = 20.0, 81   # minutes, samples per trajectory
T_GRID = np.linspace(0.0, T_END, N_STEPS)

# (low, high) sampling ranges
NOMINAL = dict(CA0=(0.6, 1.0), T0=(300., 360.), Tc=(295., 310.),
               q=(80., 120.), CAf=(0.85, 1.0), Tf=(335., 355.))
EXTRAP = dict(CA0=(0.4, 0.6), T0=(360., 390.), Tc=(310., 320.),
              q=(120., 150.), CAf=(0.70, 0.85), Tf=(355., 370.))


def sample_ops(n, ranges, rng):
    return [{k: rng.uniform(*ranges[k]) for k in OP_COLS} for _ in range(n)]


def build_df(ops, split, start_id=0):
    rows = []
    for i, op in enumerate(ops):
        try:
            traj = simulate(op, T_GRID)
        except RuntimeError:
            continue
        cid = start_id + i
        for t, (ca, T) in zip(T_GRID, traj):
            rows.append({"case_id": cid, "split": split, "t": t, **op, "CA": ca, "T": T})
    return pd.DataFrame(rows)


def main(n_nominal=400, n_extrap=60, seed=0, out="data/cstr_dataset.csv"):
    rng = np.random.default_rng(seed)
    ops = sample_ops(n_nominal, NOMINAL, rng)
    idx = rng.permutation(n_nominal)
    n_tr, n_va = int(0.7 * n_nominal), int(0.15 * n_nominal)
    parts, cid = [], 0
    for name, sel in [("train", idx[:n_tr]), ("val", idx[n_tr:n_tr + n_va]), ("test", idx[n_tr + n_va:])]:
        parts.append(build_df([ops[i] for i in sel], name, cid)); cid += len(sel)
    parts.append(build_df(sample_ops(n_extrap, EXTRAP, rng), "extrap", cid))
    df = pd.concat(parts, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows / {df.case_id.nunique()} trajectories -> {out}")
    print(df.groupby("split").case_id.nunique().to_string())
    return df


if __name__ == "__main__":
    main()
