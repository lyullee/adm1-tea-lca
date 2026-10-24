"""Optional historical empirical post-processing, reconstructed from notes."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BoundaryParams:
    apply_hrt_loss: bool = False
    hrt_reference_days: float = 20.
    hrt_anchor_days: float = 16.
    hrt_critical_days: float = 4.
    anchor_incremental_loss: float = .21
    baseline_loss_fraction: float = .1917
    curvature: float = 1.
    apply_rheology: bool = False
    ts_basis: str = "influent"

    def __post_init__(self):
        if not self.hrt_critical_days < self.hrt_anchor_days < self.hrt_reference_days:
            raise ValueError("Require HRTcritical < HRTanchor < HRTreference")
        if not 0 < self.baseline_loss_fraction < 1 or not 0 <= self.anchor_incremental_loss < 1:
            raise ValueError("Invalid baseline or anchor loss")
        if not math.isfinite(self.curvature) or self.curvature <= 0:
            raise ValueError("Curvature must be positive and finite")
        if self.ts_basis not in ("influent", "legacy_digestate_proxy"):
            raise ValueError("Unknown TS basis")


def incremental_hrt_loss(hrt_days: float, p: BoundaryParams) -> float:
    """Additional yield loss; never classify it as an atmospheric emission."""
    if not math.isfinite(hrt_days) or hrt_days <= 0:
        raise ValueError("HRT must be positive and finite")
    if not p.apply_hrt_loss or hrt_days >= p.hrt_reference_days:
        return 0.
    if hrt_days <= p.hrt_critical_days:
        return 1.
    anchor = ((p.hrt_reference_days-p.hrt_anchor_days)/
              (p.hrt_anchor_days-p.hrt_critical_days))**p.curvature
    base = math.log1p(p.anchor_incremental_loss/p.baseline_loss_fraction)/anchor
    exponent = base*((p.hrt_reference_days-hrt_days)/(hrt_days-p.hrt_critical_days))**p.curvature
    # Cap the incremental fraction of ADM1 methane, not baseline+incremental.
    limit = math.log1p(1/p.baseline_loss_fraction)
    return 1. if exponent >= limit else p.baseline_loss_fraction*math.expm1(exponent)


def rheology_multipliers(ts_pct: float) -> dict:
    if not math.isfinite(ts_pct) or ts_pct < 0:
        raise ValueError("TS must be nonnegative and finite")
    if ts_pct <= 5.4:
        electricity = 1 + ts_pct/5.4*.031
    elif ts_pct <= 12:
        electricity = 1.031 + (ts_pct-5.4)/6.6*(1.20-1.031)
    else:
        electricity = 1.20*math.exp(.331*(ts_pct-12))
        if ts_pct >= 13:
            electricity *= 2
    return {"electricity": electricity, "maintenance": min(1+.28*max(ts_pct-10,0),2.4)}
