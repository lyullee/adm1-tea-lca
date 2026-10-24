"""Small recorded grid/robustness/sensitivity analyses; no hidden simulator."""
from __future__ import annotations

from dataclasses import replace
from itertools import product
import math

import numpy as np
from scipy.stats import spearmanr

from .pipeline import PlantEvaluator, process_balance
from .tea import TEAParams, evaluate_tea
from .lca import evaluate_lca


def compact(result: dict) -> dict:
    p, t, a = result['process'], result['tea'], result['adm1']
    return {"silage_vs_fraction":p['silage_vs_fraction'], "hrt_days":p['hrt_days'],
            "olr_kg_vs_m3_d":p['olr_kg_vs_m3_d'], "pH":a['pH'],
            "methane_nm3_d":p['saleable_methane_nm3_d'], "npv_eur":t['npv_eur'],
            "irr_fraction":t['irr_fraction'], "lcob_eur_mwh":t['lcob_eur_mwh'],
            "resolved_net_gco2e_kwh":result['lca']['resolved_net_gco2e_kwh'],
            "screen_valid":result['screen_valid']}


def grid_search(evaluator: PlantEvaluator, fractions=(.6,.8), hrts=(25.,30.), olrs=(2.,)) -> dict:
    accepted, rejected = [], []
    for fraction,hrt,olr in product(fractions,hrts,olrs):
        coordinates = dict(silage_vs_fraction=float(fraction),hrt_days=float(hrt),olr=float(olr))
        try:
            result = evaluator.evaluate(**coordinates)
            if result['screen_valid']:
                accepted.append(compact(result))
            else:
                rejected.append({**coordinates,"reason":"convergence or pH screen not satisfied"})
        except (ValueError,RuntimeError) as exc:
            rejected.append({**coordinates,"reason":str(exc)})
    return {"schema":"adm1-tea-lca/grid-search/v1", "accepted":accepted,"rejected":rejected,
            "best_by_npv":max(accepted,key=lambda r:r['npv_eur']) if accepted else None,
            "best_by_lcob":min(accepted,key=lambda r:r['lcob_eur_mwh']) if accepted else None,
            "scope":"best evaluated finite-grid point; not a continuous or global optimum"}


def pareto_front(rows: list[dict]) -> list[dict]:
    """Non-dominated NPV-max / partial-GHG-min rows on the supplied grid."""
    return [r for r in rows if not any(
        q['npv_eur'] >= r['npv_eur'] and q['resolved_net_gco2e_kwh'] <= r['resolved_net_gco2e_kwh']
        and (q['npv_eur'] > r['npv_eur'] or q['resolved_net_gco2e_kwh'] < r['resolved_net_gco2e_kwh'])
        for q in rows)]


def robust_grid_screen(rows: list[dict], cv: float=.12) -> dict:
    """Normalized lognormal-kernel average on available VLR/composition rows.

    This records the historic idea with explicit finite/truncated support.
    It is not the unavailable original convolution or a failure probability.
    """
    if not math.isfinite(cv) or cv < 0:
        raise ValueError("CV must be finite and nonnegative")
    if not rows:
        return {"cv":cv,"evaluated_centers":[],"best_center":None}
    if len({r['olr_kg_vs_m3_d'] for r in rows}) > 1:
        raise ValueError("Robust screen accepts one OLR slice; run slices separately")
    if any(not 0 < r['silage_vs_fraction'] < 1 for r in rows):
        raise ValueError("Lognormal composition kernel requires interior fractions (0,1)")
    coordinates=np.array([[1/r['hrt_days'],r['silage_vs_fraction']] for r in rows])
    scores=np.array([r['npv_eur'] for r in rows])
    screened=[]
    for i,center in enumerate(coordinates):
        if cv == 0:
            weights=np.zeros(len(rows));weights[i]=1.
        else:
            sigma=math.sqrt(math.log1p(cv*cv))
            mu=np.log(center)-.5*sigma*sigma
            log_density=(-np.log(coordinates)-.5*((np.log(coordinates)-mu)/sigma)**2).sum(axis=1)
            weights=np.exp(log_density-log_density.max());weights/=weights.sum()
        screened.append({**rows[i],"kernel_mean_npv_eur":float(weights@scores),
                         "effective_grid_points":float(1/(weights@weights))})
    return {"cv":cv,"evaluated_centers":screened,
            "best_center":max(screened,key=lambda r:r['kernel_mean_npv_eur']),
            "scope":"normalized kernel over available finite grid; omitted support is not characterized; not original E[NPV] reproduction"}


def oat_sensitivity(result: dict, evaluator: PlantEvaluator, fraction: float=.1) -> list[dict]:
    if not 0 < fraction < 1:
        raise ValueError("OAT fraction must be in (0,1)")
    rows=[]
    p=evaluator.tea_params
    for name in ('biomethane_price_eur_kwh','silage_price_eur_t_dm','electricity_price_eur_kwh','reactor_unit_eur_m3'):
        base=getattr(p,name)
        for factor in (1-fraction,1+fraction):
            value=base*factor
            recalculated=evaluate_tea(result['process'],replace(p,**{name:value}))
            rows.append({"parameter":name,"value":value,"factor":factor,
                         "npv_eur":recalculated['npv_eur'],"lcob_eur_mwh":recalculated['lcob_eur_mwh'],
                         "npv_change_eur":recalculated['npv_eur']-result['tea']['npv_eur']})
    return rows


def sampled_sensitivity(result: dict, evaluator: PlantEvaluator, n: int=64, seed: int=20261024) -> dict:
    """Small fixed-design Spearman screen, not historical Borgonovo delta GSA."""
    if n < 8:
        raise ValueError("Use at least eight samples for a descriptive screen")
    rng=np.random.default_rng(seed)
    p=evaluator.tea_params
    economic={
        'biomethane_price_eur_kwh':(.069,.115,.180),
        'silage_price_eur_t_dm':(140.,165.,245.),
        'electricity_price_eur_kwh':(.10,.16,.22),
        'reactor_unit_eur_m3':(.8*p.reactor_unit_eur_m3,p.reactor_unit_eur_m3,1.25*p.reactor_unit_eur_m3),
    }
    environmental={
        'grid_kgco2e_kwh':(.20,.35,.50),
        'fertilizer_credit_kgco2e_t_digestate':(15.,22.1,30.),
    }
    samples={name:rng.triangular(*bounds,size=n) for name,bounds in {**economic,**environmental}.items()}
    outputs=[]
    for i in range(n):
        tea_p=replace(p,**{name:float(samples[name][i]) for name in economic})
        lca_p=replace(evaluator.lca_params,**{name:float(samples[name][i]) for name in environmental})
        tea=evaluate_tea(result['process'],tea_p)
        lca=evaluate_lca(result['process'],tea,lca_p,operating_days=tea_p.operating_days)
        outputs.append({"sample":i,**{name:float(values[i]) for name,values in samples.items()},
                        "npv_eur":tea['npv_eur'],"lcob_eur_mwh":tea['lcob_eur_mwh'],
                        "resolved_net_gco2e_kwh":lca['resolved_net_gco2e_kwh']})
    correlations=[]
    for name, values in samples.items():
        row={"parameter":name}
        for metric in ('npv_eur','lcob_eur_mwh','resolved_net_gco2e_kwh'):
            target=np.array([r[metric] for r in outputs])
            row[metric]=float(spearmanr(values,target).statistic) if np.ptp(target)>0 else None
        correlations.append(row)
    return {"sample_count":n,"seed":seed,"draws":outputs,"spearman_screen":correlations,
            "scope":"fixed operating design, independent triangular assumptions; small descriptive rank screen, no convergence claim or historical delta reproduction"}
