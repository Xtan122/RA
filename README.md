# RA — Refai & Ahmed component-level formulation evaluation

Faithful re-implementation of the component- and function-level evaluation
framework from **Refai & Ahmed, _Peering Inside the Black Box_
(arXiv:2510.16943)** for LLM-generated optimization formulations.

Given a ground-truth formulation and a candidate formulation, this repository
computes the paper's metrics:

- constraint precision / recall,
- decision-variable precision / recall,
- optimality gap,
- objective RMSE (Obj-RMSE),
- constraint RMSE (Cons-RMSE) over matched constraints,
- efficiency (latency, input/output tokens).

The ground-truth dataset is the four ComplexOR problems: Knapsack,
AircraftAssignment, Diet, AircraftLanding (`src/baseline/ground_truth.py`).

The LLM-generation and cross-method comparison live in the separate
`equiceval-llm-eval` experiment repo; this repo only implements the method.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Test

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

## Score a candidate

```bash
PYTHONPATH=. .venv/bin/python -m experiments.run_ra_eval \
  --problem Knapsack --candidate data/samples/knapsack_perfect.json

PYTHONPATH=. .venv/bin/python -m experiments.run_ra_eval \
  --problem Knapsack --candidate data/samples/knapsack_wrong_rhs.json
```

## Layout

- `src/baseline/metrics.py` — Eqs. 1–7 of the paper.
- `src/baseline/ground_truth.py` — `PROBLEMS_REGISTRY` (four ComplexOR problems).
- `src/baseline/evaluator.py` — `BaselineEvaluator.evaluate_candidate(...)`.
- `src/utils/solver_wrapper.py` — LP → HiGHS, MILP → SCIP, fallback PuLP/CBC.
- `data/samples/` — sample candidates for a perfect and a faulty Knapsack model.
- `arXiv-2510.16943v1/` — the original paper source.
