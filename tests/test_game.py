"""主体模仿仿真的性质测试：单纯形守恒与均场一致性（核心验证）。"""

import numpy as np

from egtsat import default_scenario, make_payoff, simulate_game, solve_replicator


def test_proportions_stay_on_simplex():
    scenario = default_scenario()
    payoff = make_payoff(scenario)
    result = simulate_game(np.full(scenario.m, 0.25), payoff, n_steps=100,
                           rng=np.random.default_rng(0), num_agents=500)
    assert np.allclose(result.proportions.sum(axis=1), 1.0, atol=1e-12)
    assert result.proportions.min() >= 0.0


def test_beta_must_be_in_range():
    scenario = default_scenario()
    payoff = make_payoff(scenario)
    try:
        simulate_game(np.full(scenario.m, 0.25), payoff, beta=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("beta=0 应当触发 ValueError")


def test_abm_mean_field_matches_ode():
    """核心验证：成对比较模仿的均场极限是复制动态方程。

    多种子平均后的 ABM 终点比例应贴近 ODE 终点比例。
    """
    scenario = default_scenario()
    payoff = make_payoff(scenario)
    _, X_ode = solve_replicator(np.full(scenario.m, 0.25), payoff,
                                t_span=(0.0, 30.0), n_eval=100)
    x_ode_final = X_ode[-1]

    finals = []
    for seed in range(5):
        result = simulate_game(
            np.full(scenario.m, 0.25), payoff,
            n_steps=400, beta=0.1, num_agents=2000,
            rng=np.random.default_rng(seed),
        )
        finals.append(result.final_proportions)
    x_abm_final = np.mean(finals, axis=0)

    assert np.abs(x_abm_final - x_ode_final).sum() < 0.05, (
        f"ABM 均值 {x_abm_final} 偏离 ODE 终点 {x_ode_final}"
    )
