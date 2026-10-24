"""TEA formulas reconstructed from 2026-07-11 notes and 08-29 parameter CSV."""
from dataclasses import dataclass
import math

from scipy.optimize import brentq


@dataclass(frozen=True)
class TEAParams:
    discount_rate: float = .06
    project_lifetime_years: int = 20
    tax_rate: float = .125
    salvage_rate: float = .05
    working_capital_rate: float = .15
    capex_subsidy_rate: float = .20
    reactor_unit_eur_m3: float = 241.3
    storage_unit_eur_m3: float = 37.4
    reception_eur_t_d: float = 1549.
    pasteurization_eur_t_d: float = 2536.
    upgrading_capex_anchor_eur: float = 2104897.
    upgrading_anchor_flow_m3h: float = 1000.
    grid_connection_eur: float = 400000.
    civil_factor: float = .30
    eng_rate: float = .07
    contingency_rate: float = .10
    dev_legal_rate: float = .09
    scale_exp_equipment: float = .69
    maintenance_rate: float = .04
    insurance_rate: float = .015
    labour_anchor_eur: float = 145455.
    capex_anchor_volume_m3: float = 12750.
    labour_scale_exp: float = .25
    admin_eur_year: float = 39935.
    electricity_price_eur_kwh: float = .16
    transport_eur_t_km: float = .13
    transport_distance_km: float = 10.
    ad_base_power_kwh_m3: float = .05
    upgrading_power_ref_kwh_m3: float = .31
    upgrading_power_co2_ref: float = .45
    silage_price_eur_t_dm: float = 165.
    silage_dm_fraction: float = .201
    slurry_price_eur_t: float = 0.
    biomethane_price_eur_kwh: float = .115
    digestate_value_eur_t: float = 1.5
    lhv_kwh_nm3: float = 9.97
    operating_days: float = 365.
    extra_capex_eur: float = 0.
    extra_opex_eur_year: float = 0.

    def __post_init__(self):
        from dataclasses import asdict
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("TEA parameters must be finite")
        if self.discount_rate < 0 or self.project_lifetime_years < 1 or int(self.project_lifetime_years) != self.project_lifetime_years:
            raise ValueError("Require nonnegative discount rate and integer project life >=1")
        if not 0 < self.operating_days <= 365:
            raise ValueError("Operating days must be in (0,365]")
        if not all(0 <= v <= 1 for v in (self.tax_rate,self.salvage_rate,self.working_capital_rate,self.capex_subsidy_rate)):
            raise ValueError("Financial fractions must lie in [0,1]")
        if min(self.upgrading_anchor_flow_m3h, self.capex_anchor_volume_m3,
               self.upgrading_power_co2_ref, self.lhv_kwh_nm3) <= 0:
            raise ValueError("Scaling denominators and heating value must be positive")


def capital_recovery_factor(rate: float, years: int) -> float:
    return 1/years if rate == 0 else rate*(1+rate)**years/((1+rate)**years-1)


def npv(cashflows: list[float], rate: float) -> float:
    return sum(value/(1+rate)**year for year,value in enumerate(cashflows))


def irr(cashflows: list[float]) -> float | None:
    # Conventional investment profiles have only one sign change. Report
    # undefined for nonconventional profiles instead of choosing a hidden root.
    signs = [1 if value > 0 else -1 for value in cashflows if value != 0]
    changes = sum(a!=b for a,b in zip(signs,signs[1:]))
    if changes != 1:
        return None
    low, high = -.95, 1.
    for _ in range(8):
        if npv(cashflows,low)*npv(cashflows,high) <= 0:
            return float(brentq(lambda r:npv(cashflows,r),low,high))
        high *= 2
    return None


def payback(cashflows: list[float], rate: float = 0.) -> float | None:
    cumulative = cashflows[0]
    for year,value in enumerate(cashflows[1:],1):
        discounted = value/(1+rate)**year
        if cumulative < 0 <= cumulative+discounted:
            return year-1-cumulative/discounted
        cumulative += discounted
    return None


def evaluate_tea(process: dict, p: TEAParams | None = None) -> dict:
    p = p or TEAParams()
    d = p.operating_days
    volume = process["liquid_volume_m3"]
    wet = process["wet_feed_t_d"]
    direct = {
        "reactor": p.reactor_unit_eur_m3*volume,
        "storage": p.storage_unit_eur_m3*volume,
        "reception": p.reception_eur_t_d*wet,
        "pasteurization": p.pasteurization_eur_t_d*wet,
        "upgrading": p.upgrading_capex_anchor_eur*(process["raw_biogas_nm3_d"]/24/p.upgrading_anchor_flow_m3h)**p.scale_exp_equipment,
        "grid": p.grid_connection_eur,
    }
    physical = sum(direct.values())
    civil = physical*p.civil_factor
    indirect = (physical+civil)*(p.eng_rate+p.contingency_rate+p.dev_legal_rate)
    gross_capex = physical+civil+indirect+p.extra_capex_eur
    capex = gross_capex*(1-p.capex_subsidy_rate)
    ad_power = process["raw_biogas_nm3_d"]*p.ad_base_power_kwh_m3*process["electricity_multiplier"]
    upgrading_power = process["raw_biogas_nm3_d"]*p.upgrading_power_ref_kwh_m3*process["co2_volume_fraction"]/p.upgrading_power_co2_ref
    electricity = (ad_power+upgrading_power)*d
    opex = {
        "maintenance": physical*p.maintenance_rate*process["maintenance_multiplier"],
        "insurance": physical*p.insurance_rate,
        "labour": p.labour_anchor_eur*(volume/p.capex_anchor_volume_m3)**p.labour_scale_exp,
        "administration": p.admin_eur_year,
        "electricity": electricity*p.electricity_price_eur_kwh,
        "silage": process["silage_t_d"]*p.silage_dm_fraction*p.silage_price_eur_t_dm*d,
        "slurry": process["slurry_t_d"]*p.slurry_price_eur_t*d,
        "transport": process["silage_t_d"]*p.transport_distance_km*p.transport_eur_t_km*d,
        "extra": p.extra_opex_eur_year,
    }
    annual_opex = sum(opex.values())
    energy = process["saleable_methane_nm3_d"]*p.lhv_kwh_nm3*d
    if energy <= 0:
        raise ValueError("Cannot calculate TEA for nonpositive product energy")
    revenues = {"biomethane":energy*p.biomethane_price_eur_kwh,
                "digestate":process["digestate_t_d"]*p.digestate_value_eur_t*d}
    revenue = sum(revenues.values())
    working_capital = capex*p.working_capital_rate
    depreciation = capex*(1-p.salvage_rate)/p.project_lifetime_years
    flows, taxes = [-(capex+working_capital)], []
    loss_carry = 0.
    for _ in range(p.project_lifetime_years):
        earnings = revenue-annual_opex-depreciation
        taxable = max(0.,earnings-loss_carry)
        loss_carry = max(0.,loss_carry-earnings)
        tax = taxable*p.tax_rate
        flows.append(revenue-annual_opex-tax)
        taxes.append(tax)
    flows[-1] += capex*p.salvage_rate+working_capital
    return {
        "capex_direct_eur":direct, "capex_gross_eur":gross_capex, "capex_net_eur":capex,
        "opex_eur_y":annual_opex, "opex_components_eur_y":opex,
        "revenue_eur_y":revenue, "revenue_components_eur_y":revenues,
        "electricity_kwh_y":electricity, "biomethane_kwh_y":energy,
        "npv_eur":npv(flows,p.discount_rate), "irr_fraction":irr(flows),
        "lcob_eur_mwh":1000*(capex*capital_recovery_factor(p.discount_rate,p.project_lifetime_years)+annual_opex)/energy,
        "simple_payback_years":payback(flows), "discounted_payback_years":payback(flows,p.discount_rate),
        "cashflows_eur":flows, "tax_eur_y":taxes,
        "scope":"reconstructed common-basis screening TEA; historical source years not escalated; LCOB excludes co-product revenue",
    }
