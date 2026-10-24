import json
import math

import pytest

from adm1_tea_lca import ADM1Simulator, BoundaryParams, LCAParams, PlantEvaluator
from adm1_tea_lca.analysis import grid_search, oat_sensitivity, robust_grid_screen, sampled_sensitivity
from adm1_tea_lca.boundaries import incremental_hrt_loss, rheology_multipliers
from adm1_tea_lca.records import record_run
from adm1_tea_lca.tea import capital_recovery_factor


@pytest.fixture(scope='module')
def evaluated():
    plant=PlantEvaluator()
    return plant,plant.evaluate()


def test_representative_integrated_run(evaluated):
    plant,result=evaluated
    assert len(result['adm1']['state'])==34
    assert result['adm1']['numerical']['solver_success']
    assert result['screen_valid']
    assert result['process']['saleable_methane_nm3_d']>0
    assert math.isfinite(result['tea']['npv_eur'])
    assert result['tea']['lcob_eur_mwh']>0
    assert len(result['tea']['cashflows_eur'])==21
    assert result['lca']['full_lca_kgco2e_y'] is None
    p=result['process']
    assert abs(p['ledger_mass_residual_kg_d'])<1e-6
    assert p['generated_gas_kg_d']['ch4']==pytest.approx(
        p['saleable_methane_kg_d']+p['upgrading_loss_ch4_kg_d']+p['digester_leak_ch4_kg_d'])
    assert capital_recovery_factor(0,20)==.05


def test_small_grid_and_analysis_wiring(evaluated):
    plant,result=evaluated
    grid=grid_search(plant)
    assert len(grid['accepted'])==4
    assert grid['best_by_npv']['npv_eur']==max(r['npv_eur'] for r in grid['accepted'])
    deterministic=robust_grid_screen(grid['accepted'],0)
    assert deterministic['best_center']['kernel_mean_npv_eur']==grid['best_by_npv']['npv_eur']
    assert len(oat_sensitivity(result,plant))==8
    sampled=sampled_sensitivity(result,plant,n=16)
    assert len(sampled['draws'])==16
    assert len(sampled['spearman_screen'])==6


def test_historical_anchor_and_input_guard():
    p=BoundaryParams(apply_hrt_loss=True)
    assert incremental_hrt_loss(20,p)==0
    assert incremental_hrt_loss(16,p)==pytest.approx(.21)
    assert incremental_hrt_loss(4,p)==1
    assert rheology_multipliers(13)['electricity']>rheology_multipliers(12)['electricity']
    with pytest.raises(ValueError):
        ADM1Simulator().feed(.8,60,4)


def test_open_storage_does_not_invent_residual_emissions(evaluated):
    plant,result=evaluated
    from adm1_tea_lca.lca import evaluate_lca
    lca=evaluate_lca(result['process'],result['tea'],LCAParams(storage_mode='open'))
    assert 'open_storage_methane' not in lca['incurred_components_kgco2e_y']
    assert any('residual methane' in s for s in lca['missing_inventory'])


def test_record_and_no_overwrite(evaluated,tmp_path):
    _,result=evaluated
    target=tmp_path/'smoke'
    record_run(target,result)
    manifest=json.loads((target/'run_manifest.json').read_text())
    assert manifest['result_hashes']['evaluation.json']
    assert any(r['path']=='_legacy/adm1_basic.py' for r in manifest['source_and_input_hashes'])
    with pytest.raises(FileExistsError):
        record_run(target,result)
