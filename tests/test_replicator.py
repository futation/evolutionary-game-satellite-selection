"""复制动态方程的性质测试：单纯形守恒、均衡合理性。"""

import numpy as np

from egtsat import Scenario, Satellite, default_scenario, make_payoff, solve_replicator
from egtsat.replicator import replicator_rhs


def test_simplex_conservation_along_trajectory():
    scenario = default_scenario()
    payoff = make_payoff(scenario)
    t, X = solve_replicator(np.full(scenario.m, 0.25), payoff)
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)
    assert X.min() > -1e-12


def test_uniform_is_fixed_point_when_satellites_identical():
    """四颗完全相同的卫星下，均匀分布处效用相等，应为不动点。"""
    scenario = Scenario(
        satellites=tuple(
            Satellite(f"Sat {i}", snr_db=15.0, service_rate=800.0, trust=0.5, ground_pos=(0.0, 0.0))
            for i in range(4)
        ),
    )
    payoff = make_payoff(scenario)
    x0 = np.full(4, 0.25)
    assert np.allclose(replicator_rhs(0.0, x0, payoff), 0.0, atol=1e-12)


def test_converges_to_rest_point_with_zero_gradient():
    """从均匀初值出发，终点处所有"仍被使用"的策略效用应相等（无迁移压力）。

    复制动态在均衡附近为多项式收敛，故积分到足够长的时间。"""
    scenario = default_scenario()
    payoff = make_payoff(scenario)
    _, X = solve_replicator(np.full(scenario.m, 0.25), payoff, t_span=(0.0, 300.0))
    x_final = np.clip(X[-1], 0.0, None)
    u_final = payoff(x_final / x_final.sum(), 1e6)
    active = x_final > 1e-3
    assert active.sum() >= 1
    assert np.ptp(u_final[active]) < 1e-3
