"""Streamlit dashboard: physics-based CSTR vs Random Forest surrogate.
Run from project root:  streamlit run app/streamlit_app.py
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from src.cstr_model import simulate, OP_COLS, STATE_COLS
from src.train_surrogate import FEATURES
from src.generate_data import NOMINAL

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="CSTR ML Surrogate", layout="wide")
st.title("CSTR: Physics-based simulation vs ML surrogate")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "rf_surrogate.joblib")

if not (ROOT / "models" / "rf_surrogate.joblib").exists():
    st.error("Model not found. Run `python run_pipeline.py` first."); st.stop()
rf = load_model()

st.sidebar.header("Operating conditions")
lab = dict(CA0="Initial CA0 [mol/L]", T0="Initial T0 [K]", Tc="Coolant Tc [K]",
           q="Feed flow q [L/min]", CAf="Feed CAf [mol/L]", Tf="Feed Tf [K]")
op = {}
for k in OP_COLS:
    lo, hi = NOMINAL[k]; span = hi - lo
    op[k] = st.sidebar.slider(lab[k], float(lo - span), float(hi + span), float((lo + hi) / 2),
                              step=float(span / 100))
    if not (lo <= op[k] <= hi):
        st.sidebar.caption(f":orange[{k} outside training range - extrapolation]")
t_end = st.sidebar.slider("Horizon [min]", 5.0, 20.0, 20.0)
t = np.linspace(0, t_end, 81)

t0 = time.perf_counter(); y_phys = simulate(op, t); t_phys = time.perf_counter() - t0
X = pd.DataFrame({"t": t, **{k: op[k] for k in OP_COLS}})[FEATURES]
t0 = time.perf_counter(); y_ml = rf.predict(X); t_ml = time.perf_counter() - t0

err = y_ml - y_phys
c = st.columns(4)
for j, s in enumerate(STATE_COLS):
    c[j].metric(f"MAE ({s})", f"{np.abs(err[:, j]).mean():.4g}")
    c[j + 2].metric(f"RMSE ({s})", f"{np.sqrt((err[:, j] ** 2).mean()):.4g}")

fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
for j, s in enumerate(STATE_COLS):
    ax[0, j].plot(t, y_phys[:, j], "k-", label="Physics (ODE)"); ax[0, j].plot(t, y_ml[:, j], "r--", label="RF surrogate")
    ax[0, j].set_ylabel(s); ax[1, j].plot(t, err[:, j], color="#555"); ax[1, j].axhline(0, c="k", lw=.5)
    ax[1, j].set(ylabel=f"error in {s}", xlabel="time [min]")
ax[0, 0].legend(); plt.tight_layout(); st.pyplot(fig)

st.subheader("Computation time (this trajectory)")
st.write(f"Physics: **{t_phys * 1e3:.2f} ms** | Surrogate: **{t_ml * 1e3:.2f} ms** | "
         f"speed-up: **{t_phys / max(t_ml, 1e-9):.1f}x**")

mp = ROOT / "results" / "metrics.json"
if mp.exists():
    st.subheader("Offline evaluation")
    m = json.loads(mp.read_text())
    st.dataframe(pd.DataFrame([{"split": sp, "state": s, **m[sp][s]} for sp in ["train", "val", "test", "extrap"] for s in STATE_COLS]).round(4))
    fp = ROOT / "results" / "figures"
    a, b = st.columns(2)
    if (fp / "feature_importance.png").exists(): a.image(str(fp / "feature_importance.png"))
    if (fp / "parity_residuals_extrap.png").exists(): b.image(str(fp / "parity_residuals_extrap.png"))
