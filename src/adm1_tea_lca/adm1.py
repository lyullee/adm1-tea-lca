"""Direct adapter around preserved equations; no process-simulator dependency.

Concentrations g/L equal kg/m3 numerically. Volumes and inflows are supplied
in m3 and m3/d, so concentration times flow gives kg/d. Gas is converted from
component mass to dry-normal Nm3, never by pressure-only conversion.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from importlib.resources import files
import math
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from ._legacy.adm1_basic import ADM1

STATES = (
    "S_su S_aa S_fa S_va S_bu S_pro S_ac S_h2 S_ch4 S_IC S_IN S_h2o "
    "X_ch X_pr X_li X_su X_aa X_fa X_va X_bu X_pro X_ac X_h2 S_cat S_an "
    "S_va_i S_bu_i S_pro_i S_ac_i S_hco3_i S_nh3 S_gas_h2 S_gas_ch4 S_gas_co2"
).split()
NORMAL_DENSITIES = {"ch4": .717, "co2": 1.977, "h2": .0899}  # kg/Nm3


def read_table(name: str) -> list[dict]:
    with files("adm1_tea_lca").joinpath("data", name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class ADM1Config:
    liquid_volume_m3: float = 10000.
    headspace_volume_m3: float = 1000.
    integration_days: float = 300.
    rtol: float = 1e-3
    atol: float = 1e-6
    max_step_days: float = 1.
    methane_drift_limit: float = .01
    ph_drift_limit: float = .05
    ph_min: float = 5.5
    ph_max: float = 9.
    max_influent_ts: float = .12
    density_t_m3: float = 1.
    silage_ts: float = .201
    silage_vs_ts: float = .815
    slurry_ts: float = .068
    slurry_vs_ts: float = .759

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(self.liquid_volume_m3, self.headspace_volume_m3, self.rtol,
               self.atol, self.max_step_days, self.density_t_m3) <= 0:
            raise ValueError("Volumes, tolerances, step and density must be positive")
        if self.integration_days < 30 or not 0 < self.max_influent_ts <= 1:
            raise ValueError("Require integration_days >= 30 and 0 < max_influent_ts <= 1")
        if self.ph_min >= self.ph_max or min(self.methane_drift_limit, self.ph_drift_limit) < 0:
            raise ValueError("Invalid pH range or convergence limits")
        if any(not 0 < v <= 1 for v in (self.silage_ts, self.silage_vs_ts, self.slurry_ts, self.slurry_vs_ts)):
            raise ValueError("Feed TS and VS/TS must be fractions in (0,1]")


class ADM1Simulator:
    """Fixed-feed, fixed-volume steady-state screen using the preserved RHS."""

    def __init__(self, config: ADM1Config | None = None, parameter_overrides: dict | None = None):
        self.config = config or ADM1Config()
        self.parameters = {r["parameter"]: float(r["value"]) for r in read_table("adm1_parameters.csv")}
        for name, value in (parameter_overrides or {}).items():
            if name not in self.parameters or not math.isfinite(float(value)):
                raise ValueError(f"Unknown/nonfinite biochemical parameter: {name}")
            self.parameters[name] = float(value)
        self.initial = {r["parameter"]: float(r["value"]) for r in read_table("adm1_initial_conditions.csv")}
        self.feeds = {name: {} for name in ("Grass_Silage_A", "Cattle_Slurry_A", "Water")}
        for row in read_table("adm1_feed_input_provenance.csv"):
            if row["substrate"] in self.feeds and row["ADM1_input_state"] in STATES:
                # Already effective: do not apply DQ a second time.
                self.feeds[row["substrate"]][row["ADM1_input_state"]] = float(row["current_effective_input"])
        self._cache: dict[tuple, dict] = {}

    def feed(self, silage_vs_fraction: float, hrt_days: float, olr: float) -> dict:
        s = self.config
        if (not all(math.isfinite(v) for v in (silage_vs_fraction, hrt_days, olr))
                or not 0 <= silage_vs_fraction <= 1 or min(hrt_days, olr) <= 0):
            raise ValueError("Require finite VS fraction in [0,1], positive HRT and OLR")
        q = s.liquid_volume_m3 / hrt_days
        vs_t_d = s.liquid_volume_m3 * olr / 1000.
        silage = vs_t_d * silage_vs_fraction / (s.silage_ts * s.silage_vs_ts)
        slurry = vs_t_d * (1 - silage_vs_fraction) / (s.slurry_ts * s.slurry_vs_ts)
        water = q * s.density_t_m3 - silage - slurry
        ts_t_d = silage * s.silage_ts + slurry * s.slurry_ts
        ts = ts_t_d / (q * s.density_t_m3)
        if water < -1e-8 or ts > s.max_influent_ts + 1e-10:
            raise ValueError("Infeasible feed: negative dilution water or influent TS above limit")
        return dict(silage_vs_fraction=silage_vs_fraction, hrt_days=hrt_days,
                    olr_kg_vs_m3_d=olr, liquid_volume_m3=s.liquid_volume_m3,
                    q_m3_d=q, wet_feed_t_d=q*s.density_t_m3,
                    silage_t_d=silage, slurry_t_d=slurry, water_t_d=max(water, 0.),
                    vs_kg_d=vs_t_d*1000, input_ts_t_d=ts_t_d, influent_ts_pct=100*ts)

    def simulate(self, silage_vs_fraction: float=.8, hrt_days: float=30., olr: float=2.) -> dict:
        key = (silage_vs_fraction, hrt_days, olr)
        if key in self._cache:
            # Protect cached source simulations from downstream mutation.
            import copy
            return copy.deepcopy(self._cache[key])
        s = self.config
        feed = self.feed(*key)
        weights = [feed[n]/feed["wet_feed_t_d"] for n in ("silage_t_d", "slurry_t_d", "water_t_d")]
        mixed = {state: sum(w*self.feeds[label].get(state, 0.) for w, label in zip(weights, self.feeds))
                 for state in STATES}
        inputs = SimpleNamespace(time=[0., 1e9], q_in=[feed["q_m3_d"]]*2,
                                 **{name: [value]*2 for name, value in mixed.items()})
        initial = SimpleNamespace(**self.initial)
        model = ADM1(s.liquid_volume_m3, s.headspace_volume_m3,
                     (0., s.integration_days), inputs, initial, SimpleNamespace(**self.parameters))
        sol = solve_ivp(model.adm1_mass, (0., s.integration_days),
                        model.get_x_from_initial(initial), method="BDF", rtol=s.rtol,
                        atol=s.atol, max_step=s.max_step_days, dense_output=True)
        if not sol.success or abs(sol.t[-1]-s.integration_days) > 1e-8:
            raise RuntimeError("ADM1 integration failed: " + sol.message)
        times = np.linspace(0., s.integration_days, 31)
        outputs = np.array([model.adm1_mass_output(sol.sol(t)) for t in times])
        previous = model.adm1_mass_output(sol.sol(s.integration_days-30))
        final = outputs[-1]
        if not np.isfinite(final).all() or np.min(final[:34]) < -1e-5 or final[43] <= 0:
            raise ValueError("ADM1 returned invalid states or nonpositive gas flow")
        gas = {name: float(final[index]*final[43]) for name, index in (("h2",31),("ch4",32),("co2",33))}
        if min(gas.values()) < 0 or gas["ch4"] <= 0:
            raise ValueError("ADM1 returned negative gas or nonpositive methane")
        normal = {name: value/NORMAL_DENSITIES[name] for name, value in gas.items()}
        drift = abs(gas["ch4"]-previous[32]*previous[43])/gas["ch4"]
        ph_drift = abs(final[38]-previous[38])
        converged = drift <= s.methane_drift_limit and ph_drift <= s.ph_drift_limit
        result = {
            "feed": feed, "gas_kg_d": gas, "gas_dry_normal_nm3_d": normal,
            "raw_biogas_nm3_d": sum(normal.values()), "pH": float(final[38]),
            "state": dict(zip(STATES, map(float, final[:34]))),
            "input_states": mixed,
            "specific_methane_nm3_kg_vs": normal["ch4"]/feed["vs_kg_d"],
            "numerical": {"solver_success": True, "solver": "SciPy BDF",
                "nfev": int(sol.nfev), "converged": bool(converged),
                "methane_relative_drift_30d": float(drift), "ph_absolute_drift_30d": float(ph_drift),
                "pH_in_screen": bool(s.ph_min <= final[38] <= s.ph_max)},
            "trajectory": [{"day": float(t), "pH": float(row[38]),
                            "ch4_kg_d": float(row[32]*row[43])} for t,row in zip(times,outputs)],
            "configuration": asdict(s), "parameters": dict(self.parameters),
            "initial_condition_provenance": "archived local ADM1_Tisocco_NSGA vector; original generator unavailable",
            "scope": "direct preserved equations with archived feed mapping; no independent biological validation claimed",
        }
        import copy
        self._cache[key] = copy.deepcopy(result)
        return result
