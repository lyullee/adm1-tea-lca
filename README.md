# ADM1-TEA-LCA

An executable record of earlier **ADM1-based grass-silage/cattle-slurry
biomethane research**, including simulation, techno-economic analysis,
partial environmental accounting, finite-grid optimization, robustness and
sensitivity analysis.

[한국어 설명](docs/README_KO.md) · [Reconstruction record](docs/RECONSTRUCTION.md)
· [Light test record](docs/TEST_REPORT.md)

## Purpose and provenance

The existing mass-based ADM1/R1–R4 equation sources and archived input tables
are preserved byte-for-byte. The missing analysis interfaces were reconstructed
on **2026-10-24** from the project's July–August research notes. This is a
research-code record, not a claim of complete historical numerical reproduction.
No DWSIM, Aspen or downstream hydrogen optimization is included.

## Install

Python 3.10 or newer:

```bash
python -m pip install adm1-tea-lca
adm1-tea-lca run --output runs/baseline
adm1-tea-lca demo --output runs/demo
```

From a source checkout:

```bash
python -m pip install -e ".[dev]"
python run.py run --config examples/recorded_baseline.json --output runs/baseline
python run.py demo --output runs/demo
python -m pytest -q
```

Existing run directories are not overwritten. The default representative point
uses a silage VS fraction of 0.8, HRT of 30 days, OLR of 2 kg VS/m3/d, and
10,000 m3 liquid volume. `demo` adds only a four-point grid, an eight-row OAT
screen and 64 descriptive sensitivity samples.

```python
from adm1_tea_lca import PlantEvaluator
from adm1_tea_lca.analysis import grid_search

plant = PlantEvaluator()
result = plant.evaluate(silage_vs_fraction=0.8, hrt_days=30, olr=2)
print(result["tea"]["npv_eur"], result["tea"]["lcob_eur_mwh"])
grid = grid_search(plant, fractions=[0.6, 0.8], hrts=[25., 30.], olrs=[2.])
```

## Workflow and modules

`feed characterization -> direct ADM1 integration -> gas/product/material
ledger -> TEA and partial GHG screen -> finite-grid analyses -> hashed records`

| Module | Role | Status |
|---|---|---|
| `_legacy/adm1_basic.py`, `adm1_r1.py`–`adm1_r4.py` | Existing model equations | Preserved source bytes |
| `_legacy/input.py`, `initial.py`, `parameters.py`, `main.py` | Existing inputs and runner | Preserved source bytes |
| `adm1.py` | Feed mixing, BDF integration, state diagnostics and gas conversion | New adapter |
| `tea.py` | CAPEX/OPEX, cash flows, NPV, IRR, LCOB and payback | Reconstructed from notes |
| `lca.py` | Incurred GHG, counterfactual credits and missing inventories | Reconstructed from notes |
| `boundaries.py` | Optional empirical HRT-loss and TS-rheology postprocessing | Reconstructed from notes |
| `pipeline.py` | One-way calculation and consistent product-loss ledger | Reconstructed interface |
| `analysis.py` | Grid search, Pareto screen, robustness kernel, OAT and sampled ranks | Lightweight implementation |
| `records.py` | Settings, environment, source and result SHA-256 records | New recording interface |

## Recorded execution

The release includes `recorded_runs/demo_20261024/`:

- `evaluation.json`: feed quantities, 34 states, pH, gas, trajectory,
  numerical diagnostics, economic/environmental outputs and assumptions;
- `analyses.json`: finite-grid, Pareto, robustness and sensitivity results;
- `grid.csv`: the four accepted evaluated points; and
- `run_manifest.json`: versions, command and source/input/output hashes.

The **five light tests passed**. The representative run converged at pH 7.7640
and produced 5,773.5003 Nm3/d saleable methane under the declared losses.
The tests check execution and bookkeeping, rather than experimental validity.
`SOURCE_MANIFEST.json` identifies the preserved originals. The release record
ZIP includes all code, documentation and the small executed example.

## Interpretation limits

- The integrated runner uses the full 34-state model. R1–R4 are retained for
  provenance and were not individually revalidated in this light test scope.
- Effective feed inputs already include their degradability mapping; DQ is
  not applied twice. The archived parameter temperature remains 311 K.
- Gas is converted from component mass to dry-normal Nm3. Digester leak and
  upgrading losses consistently affect both product energy and methane emissions.
- Digestate mass is obtained by difference. Its ledger closure is not an
  independent elemental-balance validation.
- Empirical HRT/rheology corrections default to off. Enable them with
  `--historical-boundaries`. They are recorded postprocessing assumptions, not
  validated collapse predictions. Influent TS is the default solids basis;
  the old digestate proxy is an explicitly named optional approximation.
- LCOB is EUR/MWh and excludes co-product revenue; the cash-flow metrics
  include the declared digestate revenue and financial assumptions. No new
  cost-year escalation has been applied.
- Environmental outputs are a **partial historical GHG screen**. Full LCA
  remains null. In particular, the archived fertilizer credit scales with wet
  digestate mass including dilution water and can overstate credits. Negative
  partial balances must not be reported as product footprints or verified savings.
- Robustness uses a normalized kernel on the available finite grid. It is
  not the original full-domain expected-NPV convolution. The sampled rank
  screen is descriptive Spearman analysis, not a reproduction of historical
  Borgonovo-delta/PAWN GSA or a convergence claim.

## Citation and license

Software creator: **Ugwiyeon Lee**, Korea Gas Safety Corporation. Cite the
versioned software archive described in `CITATION.cff`; the Zenodo DOI is added
after the first release is archived.

The repository uses the [MIT license](LICENSE). Upstream attribution is retained
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`third_party_licenses/Weinrich-MIT.txt`.
