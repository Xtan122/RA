"""
Unit tests for Refai & Ahmed (2025) baseline evaluation metrics and problem registry.
"""

import pytest
import numpy as np
from src.baseline.ground_truth import PROBLEMS_REGISTRY, KNAPSACK_PROBLEM, AIRCRAFT_ASSIGNMENT_PROBLEM, DIET_PROBLEM, ALP_PROBLEM
from src.baseline.metrics import (
    compute_precision_recall,
    compute_optimality_gap,
    compute_obj_rmse,
    compute_cons_rmse,
    generate_evaluation_samples,
)
from src.baseline.evaluator import BaselineEvaluator
from src.utils.solver_wrapper import UnifiedSolver


def test_ground_truth_registry():
    assert len(PROBLEMS_REGISTRY) == 4
    assert "Knapsack" in PROBLEMS_REGISTRY
    assert "AircraftAssignment" in PROBLEMS_REGISTRY
    assert "Diet" in PROBLEMS_REGISTRY
    assert "AircraftLanding" in PROBLEMS_REGISTRY


def test_precision_recall_computation():
    # TP=2, FP=1, FN=1 => Precision=2/3 (~0.6667), Recall=2/3 (~0.6667)
    p, r = compute_precision_recall(2, 1, 1)
    assert pytest.approx(p, 0.001) == 2 / 3
    assert pytest.approx(r, 0.001) == 2 / 3

    # TP=3, FP=0, FN=0 => Precision=1.0, Recall=1.0
    p, r = compute_precision_recall(3, 0, 0)
    assert p == 1.0
    assert r == 1.0


def test_optimality_gap():
    # Normal case: true = 100, llm = 100 => gap = 0.0
    gap = compute_optimality_gap(100.0, 100.0)
    assert gap == 0.0

    # Deviation: true = 100, llm = 90 => gap = |100 - 90| / 100 = 0.10
    gap = compute_optimality_gap(100.0, 90.0)
    assert pytest.approx(gap, 0.001) == 0.10

    # Edge case z* = 0 (Aircraft Landing): true = 0, llm = 5 => gap = 5.0
    gap = compute_optimality_gap(0.0, 5.0)
    assert gap == 5.0

    # None case
    gap = compute_optimality_gap(100.0, None)
    assert gap is None


def test_knapsack_solver_ground_truth():
    # Test solving Knapsack ground truth using UnifiedSolver
    # Maximize 60 x_1 + 100 x_2 + 120 x_3
    # s.t. 10 x_1 + 20 x_2 + 30 x_3 <= 50
    # x_i in {0, 1}
    var_specs = {
        "x_1": ("binary", 0.0, 1.0),
        "x_2": ("binary", 0.0, 1.0),
        "x_3": ("binary", 0.0, 1.0),
    }
    obj_coeffs = {"x_1": 60.0, "x_2": 100.0, "x_3": 120.0}
    constraints = [({"x_1": 10.0, "x_2": 20.0, "x_3": 30.0}, "<=", 50.0)]

    res = UnifiedSolver.solve_pulp_model(
        var_specs=var_specs,
        objective_coeffs=obj_coeffs,
        objective_sense="maximize",
        constraints_list=constraints,
    )

    assert res.status == "Optimal"
    assert res.objective_value == 220.0
    # Optimal selection: x_2=1, x_3=1 (weight 50, value 220)
    assert res.variable_values["x_2"] == 1.0
    assert res.variable_values["x_3"] == 1.0


def test_baseline_evaluator_perfect_knapsack():
    evaluator = BaselineEvaluator("Knapsack")

    cand_vars = {"x_1", "x_2", "x_3"}
    cand_cons = [
        ("capacity", lambda x: 10 * x["x_1"] + 20 * x["x_2"] + 30 * x["x_3"] - 50)
    ]
    cand_opt_val = 220.0

    metrics = evaluator.evaluate_candidate(
        candidate_variables=cand_vars,
        candidate_constraints=cand_cons,
        candidate_optimal_val=cand_opt_val,
        matched_constraint_indices=[(0, 0)],
    )

    assert metrics.cons_precision == 1.0
    assert metrics.cons_recall == 1.0
    assert metrics.dv_precision == 1.0
    assert metrics.dv_recall == 1.0
    assert metrics.optimality_gap == 0.0
    assert metrics.cons_rmse == pytest.approx(0.0, abs=1e-12)
