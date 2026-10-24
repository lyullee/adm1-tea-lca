"""One-way ADM1 -> gas/material ledger -> TEA/LCA pipeline."""
from dataclasses import asdict

from .adm1 import ADM1Simulator, NORMAL_DENSITIES
from .boundaries import BoundaryParams, incremental_hrt_loss, rheology_multipliers
from .tea import TEAParams, evaluate_tea
from .lca import LCAParams, evaluate_lca


def process_balance(adm1: dict, lca: LCAParams, boundary: BoundaryParams) -> dict:
    feed = adm1["feed"]
    loss_fraction = incremental_hrt_loss(feed["hrt_days"], boundary)
    generated = dict(adm1["gas_kg_d"])
    yield_loss = generated["ch4"]*loss_fraction
    generated["ch4"] -= yield_loss
    leak = generated["ch4"]*lca.digester_leak_fraction
    collected = {name: value*(1-lca.digester_leak_fraction) for name,value in generated.items()}
    upgrade_loss = collected["ch4"]*lca.upgrading_loss_fraction
    saleable = collected["ch4"]-upgrade_loss
    raw_volumes = {name: value/NORMAL_DENSITIES[name] for name,value in collected.items()}
    raw_volume = sum(raw_volumes.values())
    # Ledger closure only: digestate is obtained by difference. This is not an
    # independent measurement of effluent mass or residual dry solids.
    digestate = feed["wet_feed_t_d"]-sum(generated.values())/1000
    if digestate <= 0 or raw_volume <= 0 or saleable <= 0:
        raise ValueError("Nonpositive digestate, gas or product after postprocessing")
    if boundary.ts_basis == "influent":
        ts = feed["influent_ts_pct"]
    else:
        ts = 100*feed["input_ts_t_d"]/digestate
    multipliers = rheology_multipliers(ts) if boundary.apply_rheology else {"electricity":1.,"maintenance":1.}
    return {**feed, "generated_gas_kg_d":generated, "collected_gas_kg_d":collected,
            "raw_biogas_nm3_d":raw_volume, "co2_volume_fraction":raw_volumes["co2"]/raw_volume,
            "saleable_methane_nm3_d":saleable/NORMAL_DENSITIES["ch4"],
            "saleable_methane_kg_d":saleable, "digester_leak_ch4_kg_d":leak,
            "upgrading_loss_ch4_kg_d":upgrade_loss,
            "incremental_yield_loss_fraction":loss_fraction,
            "incremental_yield_loss_kg_ch4_d":yield_loss,
            "digestate_t_d":digestate,
            "ledger_mass_residual_kg_d":feed["wet_feed_t_d"]*1000-digestate*1000-sum(generated.values()),
            "ts_for_penalty_pct":ts, "ts_basis":boundary.ts_basis,
            "electricity_multiplier":multipliers["electricity"],
            "maintenance_multiplier":multipliers["maintenance"],
            "digestate_mass_basis":"difference-based ledger; not independent ADM1 elemental closure",
            "gas_volume_basis":"dry normal component masses; kg/Nm3 densities explicitly recorded"}


class PlantEvaluator:
    def __init__(self, simulator: ADM1Simulator | None=None, tea: TEAParams | None=None,
                 lca: LCAParams | None=None, boundary: BoundaryParams | None=None):
        self.simulator = simulator or ADM1Simulator()
        self.tea_params = tea or TEAParams()
        self.lca_params = lca or LCAParams()
        self.boundary_params = boundary or BoundaryParams()

    def evaluate(self, silage_vs_fraction: float=.8, hrt_days: float=30., olr: float=2.) -> dict:
        adm1 = self.simulator.simulate(silage_vs_fraction,hrt_days,olr)
        process = process_balance(adm1,self.lca_params,self.boundary_params)
        tea = evaluate_tea(process,self.tea_params)
        lca = evaluate_lca(process,tea,self.lca_params,operating_days=self.tea_params.operating_days)
        return {"schema":"adm1-tea-lca/evaluation/v1", "adm1":adm1, "process":process,
                "tea":tea, "lca":lca,
                "screen_valid":adm1["numerical"]["converged"] and adm1["numerical"]["pH_in_screen"],
                "assumptions":{"tea":asdict(self.tea_params), "lca":asdict(self.lca_params),
                               "boundary":asdict(self.boundary_params)},
                "implementation_status":"ADM1 source preserved; adapter/TEA/LCA/analysis reconstructed 2026-10-24"}
