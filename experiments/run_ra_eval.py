"""Refai & Ahmed (arXiv:2510.16943) component-level evaluator CLI.

Scores one candidate formulation against one of the four ComplexOR ground-truth
problems and prints the paper's metrics: constraint / decision-variable precision
and recall, optimality gap, objective RMSE, constraint RMSE.

Usage:
    PYTHONPATH=. python -m experiments.run_ra_eval \
        --problem Knapsack --candidate data/samples/knapsack_perfect.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline.evaluator import BaselineEvaluator
from src.baseline.ground_truth import PROBLEMS_REGISTRY
from src.utils.solver_wrapper import UnifiedSolver


def _match_constraints(gt_names, cand_names):
    """Name/keyword-overlap matching with a sequential fallback (paper's manual match)."""
    matched, used = [], set()
    for gt_i, gt_name in enumerate(gt_names):
        gt_keywords = set(gt_name.lower().replace("-", "_").split("_"))
        best_i, best_score = None, 0
        for cand_i, cand_name in enumerate(cand_names):
            if cand_i in used:
                continue
            cand_keywords = set(cand_name.lower().replace("-", "_").split("_"))
            overlap = len(gt_keywords & cand_keywords)
            if overlap > best_score:
                best_score, best_i = overlap, cand_i
        if best_i is not None:
            matched.append((gt_i, best_i))
            used.add(best_i)
    matched_gt = {m[0] for m in matched}
    free = [i for i in range(len(cand_names)) if i not in used]
    seq = 0
    for gt_i in range(len(gt_names)):
        if gt_i not in matched_gt and seq < len(free):
            matched.append((gt_i, free[seq]))
            seq += 1
    return matched


def _constraint_func(spec):
    """Candidate row -> ``f(x)`` in the ground-truth convention (``<= 0`` when satisfied)."""
    coeffs = {k: float(v) for k, v in spec["coeffs"].items()}
    rhs = float(spec["rhs"])
    sense = spec.get("sense", "<=")
    if sense == ">=":
        return lambda x, c=coeffs, r=rhs: r - sum(a * x.get(v, 0.0) for v, a in c.items())
    return lambda x, c=coeffs, r=rhs: sum(a * x.get(v, 0.0) for v, a in c.items()) - r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=sorted(PROBLEMS_REGISTRY))
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    payload = json.loads(args.candidate.read_text())
    variables = set(payload["variables"])
    specs = payload["constraints"]
    funcs = [(s.get("name", f"c{i}"), _constraint_func(s)) for i, s in enumerate(specs)]
    cand_names = [name for name, _ in funcs]

    gt = PROBLEMS_REGISTRY[args.problem]
    matched = payload.get("matched")
    if matched is None:
        matched = _match_constraints([c.name for c in gt.constraints], cand_names)
    matched = [tuple(m) for m in matched]

    cand_opt = payload.get("optimal_value")
    solver_status = "given"
    if cand_opt is None:
        var_specs = {
            v: (
                "binary" if payload.get("binary", {}).get(v) else "continuous",
                float(payload.get("lower", {}).get(v, 0.0)),
                float(payload.get("upper", {}).get(v, payload.get("default_upper", 1.0))),
            )
            for v in variables
        }
        objective = payload.get("objective", {})
        constraints_list = [
            (s["coeffs"], s.get("sense", "<="), float(s["rhs"])) for s in specs
        ]
        res = UnifiedSolver.solve_pulp_model(
            var_specs=var_specs,
            objective_coeffs={k: float(v) for k, v in objective.get("coeffs", {}).items()},
            objective_sense=objective.get("sense", gt.objective_sense),
            constraints_list=constraints_list,
        )
        cand_opt = res.objective_value if res.status == "Optimal" else None
        solver_status = res.status

    metrics = BaselineEvaluator(args.problem).evaluate_candidate(
        candidate_variables=variables,
        candidate_constraints=funcs,
        candidate_optimal_val=cand_opt,
        matched_constraint_indices=matched,
        num_samples=args.num_samples,
    )
    print(json.dumps({
        "problem": args.problem,
        "candidate": str(args.candidate),
        "solver_status": solver_status,
        "candidate_optimal_value": cand_opt,
        "cons_precision": metrics.cons_precision,
        "cons_recall": metrics.cons_recall,
        "dv_precision": metrics.dv_precision,
        "dv_recall": metrics.dv_recall,
        "optimality_gap": metrics.optimality_gap,
        "obj_rmse": metrics.obj_rmse,
        "cons_rmse": metrics.cons_rmse,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
