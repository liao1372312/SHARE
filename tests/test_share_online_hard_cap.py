from share.policies import Policy
from share.types import AuthorityAction
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from scripts.run_share_online_hard_cap_curve import nondominated


class HumanFirstPolicy(Policy):
    name = "fake"

    def select(self, state, feasible):
        if AuthorityAction.HUMAN in feasible:
            return AuthorityAction.HUMAN
        return AuthorityAction.STOP


class State:
    step = 0

    def __init__(self, episode_id):
        self.episode_id = episode_id


def test_binary_hard_cap_falls_back_to_ai():
    policy = OnlineHardHumanCapPolicy(HumanFirstPolicy(), cap=1, binary_fallback=True)
    feasible = list(AuthorityAction)
    assert policy.select(State("one"), feasible) == AuthorityAction.HUMAN
    assert policy.select(State("two"), feasible) == AuthorityAction.AI
    assert policy.human_interventions == 1
    assert policy.forced_autonomous_fallbacks == 1


def test_value_hard_cap_replans_without_human():
    policy = OnlineHardHumanCapPolicy(HumanFirstPolicy(), cap=0, binary_fallback=False)
    assert policy.select(State("one"), list(AuthorityAction)) == AuthorityAction.STOP
    assert policy.human_interventions == 0
    assert policy.value_policy_replans == 1


def test_curve_nondominance_respects_all_declared_dimensions():
    rows = []
    for budget in (0.0, 0.1, 0.2, 0.3, 0.5):
        for index, policy in enumerate(("share_conf", "share_static", "share_bandit", "share_ddqn", "share_h")):
            rows.append(
                {
                    "nominal_budget": budget,
                    "policy": policy,
                    "repair_sr_mean": 1.0 if index == 0 else 0.5,
                    "normalized_return_mean": 1.0 if index == 0 else 0.5,
                    "uar_mean": 0.0 if index == 0 else 0.5,
                    "unsafe_autonomy_cost_rate_mean": 0.0 if index == 0 else 0.5,
                    "stopped_rate_mean": 0.0 if index == 0 else 0.5,
                    "collaboration_regret_mean": 0.0 if index == 0 else 0.5,
                }
            )
    assert all(row["nondominated"] == ["share_conf"] for row in nondominated(rows))
