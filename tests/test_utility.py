"""效用函数的性质测试：有界性、单调性、稳定域截断。"""

import numpy as np

from egtsat import default_scenario
from egtsat.utility import (
    capacity_score,
    delay_score,
    make_payoff,
    per_user_rate,
    queue_delay,
)


def test_capacity_score_bounded_and_monotone():
    scenario = default_scenario()
    x = np.full(scenario.m, 0.25)
    rate = per_user_rate(scenario.snr_linear, scenario.bandwidth_hz, x, scenario.num_users)
    score = capacity_score(rate, scenario.capacity_half)
    assert np.all(score > 0) and np.all(score < 1)
    # 信道越好分数越高
    assert np.all(np.diff(score) < 0)  # snr 从 A 到 D 递减，分数也应递减
    # 负载越重分数越低
    rate_heavy = per_user_rate(scenario.snr_linear, scenario.bandwidth_hz, np.array([1.0, 0, 0, 0]), 1000)
    assert capacity_score(rate_heavy, scenario.capacity_half)[0] < score[0]


def test_delay_score_decreases_with_load():
    scenario = default_scenario()
    x_light = np.full(scenario.m, 0.1)
    x_heavy = np.full(scenario.m, 0.9)
    d_light = delay_score(
        queue_delay(x_light, scenario.service_rates, scenario.arrival_per_user, scenario.num_users),
        scenario.delay_half,
    )
    d_heavy = delay_score(
        queue_delay(x_heavy, scenario.service_rates, scenario.arrival_per_user, scenario.num_users),
        scenario.delay_half,
    )
    assert np.all(d_heavy < d_light)


def test_queue_delay_clamped_when_unstable():
    """全部用户挤到最弱卫星：lambda 远超 mu，时延必须被截断且有限。"""
    scenario = default_scenario()
    x = np.array([0.0, 0.0, 0.0, 1.0])
    delay = queue_delay(x, scenario.service_rates, scenario.arrival_per_user, scenario.num_users)
    assert np.all(np.isfinite(delay))
    margin = scenario.delay_margin
    boundary = 1.0 / (margin * scenario.service_rates)
    assert delay[3] == boundary[3]
    # 截断后 delay 分数仍严格为正
    assert delay_score(delay, scenario.delay_half)[3] > 0


def test_payoff_bounded_and_consistent():
    """payoff 必须落在 [0, 1]（三个分量均有界且权重和为 1）。"""
    scenario = default_scenario()
    payoff = make_payoff(scenario)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.dirichlet(np.ones(scenario.m))
        u = payoff(x, 0.0)
        assert u.shape == (scenario.m,)
        assert np.all(u >= 0.0) and np.all(u <= 1.0)
