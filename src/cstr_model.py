"""Physics-based CSTR model: first-order exothermic reaction A -> B with a cooling jacket.

States : CA  [mol/L]  reactant concentration
         T   [K]      reactor temperature
Inputs : q   [L/min]  volumetric feed flow (V constant -> residence time V/q)
         CAf [mol/L]  feed concentration
         Tf  [K]      feed temperature
         Tc  [K]      coolant temperature

Balances (Seborg, Edgar, Mellichamp - CSTR example):
  dCA/dt = q/V (CAf - CA) - k0 exp(-E/RT) CA
  dT/dt  = q/V (Tf - T) + (-dHr)/(rho Cp) k0 exp(-E/RT) CA + UA/(V rho Cp) (Tc - T)
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CSTRParams:
    V: float = 100.0          # L
    rho_cp: float = 239.0     # J/(L K)   (rho * Cp)
    dHr: float = -5.0e4       # J/mol     (exothermic)
    E_over_R: float = 8750.0  # K
    k0: float = 7.2e10        # 1/min
    UA: float = 5.0e4         # J/(min K)


# Names of operating-condition inputs used throughout the project
OP_COLS = ["CA0", "T0", "Tc", "q", "CAf", "Tf"]
STATE_COLS = ["CA", "T"]


def rhs(t, x, q, CAf, Tf, Tc, p: CSTRParams):
    CA, T = x
    k = p.k0 * np.exp(-p.E_over_R / T)
    dCA = q / p.V * (CAf - CA) - k * CA
    dT = (q / p.V * (Tf - T)
          + (-p.dHr) / p.rho_cp * k * CA
          + p.UA / (p.V * p.rho_cp) * (Tc - T))
    return [dCA, dT]


def simulate(op: dict, t_eval: np.ndarray, p: CSTRParams = CSTRParams()):
    """Integrate the CSTR ODEs from initial state (CA0, T0) under constant inputs.

    Returns array of shape (len(t_eval), 2) with columns [CA, T].
    """
    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), [op["CA0"], op["T0"]],
        t_eval=t_eval, args=(op["q"], op["CAf"], op["Tf"], op["Tc"], p),
        method="LSODA", rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def params_dict(p: CSTRParams = CSTRParams()):
    return asdict(p)
