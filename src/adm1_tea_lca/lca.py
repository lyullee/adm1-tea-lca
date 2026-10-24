"""Recorded partial GHG screen, with incurred burdens and credits separated."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LCAParams:
    gwp_ch4: float = 28.
    gwp_n2o: float = 265.
    grid_kgco2e_kwh: float = .35
    transport_kgco2e_t_km: float = .10
    transport_distance_km: float = 10.
    fertilizer_credit_kgco2e_t_digestate: float = 22.1
    fossil_gas_credit_kgco2e_kwh: float = .20
    digester_leak_fraction: float = .01
    upgrading_loss_fraction: float = .01
    storage_mode: str = "closed"
    open_storage_mcf: float = .15
    residual_methane_potential_kg_d: float | None = None
    digestate_nitrogen_kg_t: float = 3.6
    storage_n2o_n_fraction: float | None = None

    def __post_init__(self):
        if self.storage_mode not in ("closed", "open"):
            raise ValueError("Storage mode must be closed or open")
        if not all(math.isfinite(v) for v in asdict(self).values() if isinstance(v,(int,float))):
            raise ValueError("LCA factors must be finite")
        if not all(0 <= v < 1 for v in (self.digester_leak_fraction,self.upgrading_loss_fraction,self.open_storage_mcf)):
            raise ValueError("Leak/loss/MCF fractions must lie in [0,1)")
        if self.residual_methane_potential_kg_d is not None and self.residual_methane_potential_kg_d < 0:
            raise ValueError("Residual methane potential cannot be negative")
        if self.storage_n2o_n_fraction is not None and not 0 <= self.storage_n2o_n_fraction <= 1:
            raise ValueError("N2O-N fraction must lie in [0,1]")


def evaluate_lca(process: dict, tea: dict, p: LCAParams | None = None, *, operating_days: float=365.) -> dict:
    p = p or LCAParams()
    d = operating_days
    terms = {
        "feed_transport":process["silage_t_d"]*d*p.transport_distance_km*p.transport_kgco2e_t_km,
        "digester_methane_leak":process["digester_leak_ch4_kg_d"]*d*p.gwp_ch4,
        "upgrading_methane_loss":process["upgrading_loss_ch4_kg_d"]*d*p.gwp_ch4,
        "purchased_electricity":tea["electricity_kwh_y"]*p.grid_kgco2e_kwh,
    }
    missing = ["feedstock cultivation inventory", "plant construction/end-of-life inventory",
               "complete heating/utilities inventory", "site-specific digestate application inventory"]
    if p.storage_mode == "open":
        if p.residual_methane_potential_kg_d is None:
            missing.append("measured/justified digestate residual methane potential")
        else:
            terms["open_storage_methane"] = p.residual_methane_potential_kg_d*p.open_storage_mcf*d*p.gwp_ch4
        if p.storage_n2o_n_fraction is None:
            missing.append("open-storage N2O-N emission factor")
        else:
            terms["open_storage_n2o"] = (process["digestate_t_d"]*d*p.digestate_nitrogen_kg_t*
                                          p.storage_n2o_n_fraction*44/28*p.gwp_n2o)
    credits = {
        "fertilizer_substitution":-process["digestate_t_d"]*d*p.fertilizer_credit_kgco2e_t_digestate,
        "fossil_gas_substitution":-tea["biomethane_kwh_y"]*p.fossil_gas_credit_kgco2e_kwh,
    }
    incurred = sum(terms.values())
    net = incurred+sum(credits.values())
    return {"incurred_components_kgco2e_y":terms, "counterfactual_credits_kgco2e_y":credits,
            "resolved_incurred_kgco2e_y":incurred, "resolved_net_kgco2e_y":net,
            "resolved_net_gco2e_kwh":1000*net/tea["biomethane_kwh_y"],
            "uncharacterized_incremental_yield_loss_kg_ch4_d":process["incremental_yield_loss_kg_ch4_d"],
            "full_lca_kgco2e_y":None, "missing_inventory":missing,
            "scope":"partial historical GHG screen, not a complete product LCA; yield loss is not atmospheric methane"}
