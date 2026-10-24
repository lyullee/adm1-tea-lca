# ADM1–TEA–LCA 연구 코드 기록

2026년 7~8월의 **ADM1 기반 grass silage–cattle slurry 바이오메탄 연구**를
실행 가능한 통합 코드로 정리했다. DWSIM·Aspen·수소 후단 공정은 포함하지 않는다.

원본이 남아 있는 ADM1/R1~R4 방정식·초기값·파라미터·입력 코드와 원고의 입력표는
그대로 보존했다. 유실된 TEA/LCA·최적화 통합 코드는 인수인계서에 기록된 수식과
계수를 바탕으로 **2026-10-24에 재구성**했다. 이 패키지는 옛 원본의 완전 복원이나
과거 논문의 모든 수치 재현을 주장하지 않는다. 목적은 계산 구조·코드·실행 이력의
보존이다.

## 빠른 실행

Python 3.10 이상, NumPy·SciPy가 필요하다.

```powershell
python -m pip install -e ".[dev]"
python run.py run --config examples/recorded_baseline.json --output runs/baseline
python run.py demo --output runs/demo
python -m pytest -q
```

`run`은 대표 조건(사일리지 VS분율 0.8, HRT 30일, OLR 2 kg VS/m³/d)을 계산한다.
`demo`는 여기에 4점 격자 탐색, 부분 환경지표 파레토, 강건성 커널 스크린,
4인자 ±10% OAT, 64표본 순위 민감도 스크린을 추가한다. 큰 격자나 장시간 GSA는
자동 실행하지 않는다. 기존 출력 폴더는 덮어쓰지 않는다.

설치 후에는 `adm1-tea-lca` 또는 `python -m adm1_tea_lca`도 같은 명령을 제공한다.

## 통합 흐름

`기질 혼합 → ADM1 직접 적분 → 건조 표준상태 가스·제품·소화액 장부 → TEA/LCA
→ 유한 격자 최적화·강건성·민감도 → JSON/CSV와 해시 기록`

| 코드 | 역할 | 보존/재구성 |
|---|---|---|
| `_legacy/adm1_basic.py`, `adm1_r1.py`~`adm1_r4.py` | 기존 방정식 | 기존 파일 바이트 보존 |
| `_legacy/input.py`, `initial.py`, `parameters.py`, `main.py` | 기존 입력·실행 코드 | 기존 파일 바이트 보존 |
| `adm1.py` | 패키지 입력표·혼합·BDF 직접 적분·단위 변환 | 새 어댑터 |
| `tea.py` | CAPEX/OPEX, 현금흐름·NPV·IRR·LCOB·회수기간 | 기록에서 재구성 |
| `lca.py` | 배출·회피 크레딧·누락 인벤토리 | 기록에서 재구성 |
| `boundaries.py` | 선택적 HRT 손실·TS 유변학 후처리 | 기록에서 재구성 |
| `pipeline.py` | 모듈 연결과 일관된 가스·제품 장부 | 재구성 |
| `analysis.py` | 격자·파레토·강건성·OAT·표본 민감도 | 새 경량 구현 |
| `records.py` | 설정·실행환경·코드 및 결과 SHA-256 | 신규 기록 기능 |

## 출력

- `evaluation.json`: 투입량, 34상태, pH·메탄·궤적·수치진단, 경제·환경 결과 및 설정.
- `analyses.json`: demo의 격자·강건성·민감도 결과.
- `grid.csv`: 수치·물리 스크린을 통과한 평가점.
- `run_manifest.json`: 실행 시각, Python/NumPy/SciPy 버전, 명령, 코드·입력·결과 해시.

Python 사용 예:

```python
from adm1_tea_lca import PlantEvaluator
from adm1_tea_lca.analysis import grid_search

plant = PlantEvaluator()
result = plant.evaluate(silage_vs_fraction=.8, hrt_days=30, olr=2)
print(result["tea"]["npv_eur"], result["tea"]["lcob_eur_mwh"])
grid = grid_search(plant, fractions=[.6, .8], hrts=[25., 30.], olrs=[2.])
```

## 해석 범위

- 통합 실행은 34상태 ADM1을 사용한다. R1~R4는 원본 코드 보관이며 이번 경량
  테스트에서 각 축소모델을 재검증하지 않는다.
- 기본은 순수 ADM1 출력과 유입 TS 제한이다. `--historical-boundaries`를 주면
  기록의 경험적 HRT 손실·유변학 모델을 후처리한다. 검증된 붕괴 예측으로 해석하지
  않는다. TS 기본은 유입 TS다. 옛 `input_TS/digestate` 근사는 선택 옵션으로만
  남겼으며 실제 소화액 TS라고 주장하지 않는다.
- HRT 앵커·곡률은 실제 식에 반영한다. HRT 추가 수율손실과 대기 메탄 배출은
  구분한다. 개방저장의 잔류 메탄·N₂O는 입력이 없으면 누락으로 기록한다.
- 기질 입력의 DQ는 이미 적용된 `current_effective_input`을 한 번만 사용한다.
- 방정식과 입력표의 온도는 311 K다. 7월 노트의 41°C를 임의로 덮어쓰지 않는다.
- 가스는 질량에서 Nm³로 변환하고 CH₄ 누출·정제손실을 제품량과 환경성에 같은
  비율로 반영한다. 소화액은 차감식 장부값이므로 장부 수지 일치가 독립적인
  ADM1 원소수지 검증을 뜻하지 않는다.
- LCOB는 €/MWh이며 부산물 수익은 차감하지 않는다. 세후 NPV/IRR에는 소화액
  수익·이월결손금·운전자본·잔존가치를 반영한다. 비용 연도는 추가 보정하지 않았다.
- LCA는 역사적 계수에 따른 **부분 GHG 스크린**이다. 전체 LCA는 null로 유지한다.
- 강건성은 보유 격자상의 정규화 커널 평균이다. 옛 연속 E[NPV] 적분의 동일 재현이
  아니다. 표본 민감도는 Spearman 스크린이며 과거 Borgonovo δ/PAWN GSA를 대체하거나
  수렴을 주장하지 않는다.

근거와 차이는 [재구성 기록](docs/RECONSTRUCTION.md), 테스트 결과는
[테스트 기록](docs/TEST_REPORT.md)에 정리한다. 기존 파일 해시는
`SOURCE_MANIFEST.json`에 남긴다. 원본 프로젝트 파일은 수정하지 않았다.
