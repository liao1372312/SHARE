from scripts import run_share_budget_matched_comparison as matched


def test_scalar_calibration_uses_hir_band_then_distance(monkeypatch):
    class FakePolicy:
        scalar = 0.0

    policy = FakePolicy()
    rates = {0.0: 0.40, 1.0: 0.215, 2.0: 0.19, 3.0: 0.10}
    monkeypatch.setattr(matched, "episode_hir", lambda current, episodes: rates[current.scalar])
    result = matched.calibrate_scalar_validation_only(
        policy,
        [object()],
        [0.0, 1.0, 2.0, 3.0],
        lambda value: setattr(policy, "scalar", value),
        target=0.20,
        lower=0.18,
        upper=0.22,
    )
    assert result["validation_hir_in_band"]
    assert result["validation_hir"] == 0.19
    assert policy.scalar == 2.0


def test_pareto_excludes_comparator_outside_hir_band():
    base = {
        "repair_sr_mean": 0.9,
        "uar_mean": 0.1,
        "unsafe_autonomy_cost_rate_mean": 0.1,
        "stopped_rate_mean": 0.01,
        "normalized_return_mean": 1.0,
        "collaboration_regret_mean": 0.5,
        "mean_hir_in_band": True,
    }
    rows = [{"policy": "share_h", **base}]
    for name in matched.COMPARATORS:
        rows.append(
            {
                "policy": name,
                **base,
                "repair_sr_mean": 1.0,
                "mean_hir_in_band": False,
            }
        )
    result = matched.pareto_assessment(rows)
    assert result["conclusion"] == "inconclusive_no_budget_matched_comparator"
    assert result["share_h_on_frontier_among_mean_hir_eligible_methods"] is None
    assert not any(row["dominates_share_h"] for row in result["comparisons"])
