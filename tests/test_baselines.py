"""基线方法的性质测试：社会最优不劣于任何基线、输出为合法分布。"""

import numpy as np

from egtsat import default_scenario, make_payoff
from egtsat.baselines import (
    fixed_proportions,
    nearest_proportions,
    optimal_proportions,
    random_proportions,
)
from egtsat.metrics import mean_payoff


def _user_positions_hotspots(n=1000, seed=0):
    """与 README 场景一致的双热点用户分布。"""
    rng = np.random.default_rng(seed)
    halves = rng.random(n) > 0.5
    pos = np.where(halves[:, None],
                   rng.normal(loc=[20.0, 80.0], scale=10.0, size=(n, 2)),
                   rng.normal(loc=[80.0, 20.0], scale=10.0, size=(n, 2)))
    return pos


def test_optimal_dominates_all_baselines():
    scenario = default_scenario()
    payoff = make_payoff(scenario)
    x_opt = optimal_proportions(payoff, scenario.m)
    assert np.isclose(x_opt.sum(), 1.0) and x_opt.min() >= 0.0

    u_opt = mean_payoff(x_opt, payoff(x_opt))
    for name, x in {
        "nearest": nearest_proportions(_user_positions_hotspots(), scenario),
        "fixed": fixed_proportions(scenario),
        "random": random_proportions(scenario.m, np.random.default_rng(1)),
    }.items():
        assert np.isclose(x.sum(), 1.0) and x.min() >= 0.0
        assert mean_payoff(x, payoff(x)) <= u_opt + 1e-6, f"{name} 不应优于社会最优"


def test_nearest_is_half_and_half_for_symmetric_hotspots():
    """双热点恰好落在两颗卫星星下点上：最近邻应为 50/50。"""
    scenario = default_scenario()
    positions = np.vstack([
        np.repeat([[20.0, 80.0]], 50, axis=0),
        np.repeat([[80.0, 20.0]], 50, axis=0),
    ])
    x = nearest_proportions(positions, scenario)
    assert x[0] == 0.5  # Sat A 在 (20, 80)
    assert x[3] == 0.5  # Sat D 在 (80, 20)
