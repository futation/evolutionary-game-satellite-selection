"""实验 2：不同用户规模下，演化博弈解与社会最优/经典基线的平均效用对比。

结论预期：演化博弈均衡贴近社会最优（上界），显著优于
最近邻 / 固定 / 随机等无博弈基线；用户越多（拥塞越重），差距越大。

运行：python experiments/exp2_utility_vs_N.py
"""

from __future__ import annotations

import numpy as np

from _common import hotspot_users, make_scenario, save_fig
import matplotlib.pyplot as plt

from egtsat import make_payoff, simulate_game, solve_replicator
from egtsat.baselines import (
    fixed_proportions,
    nearest_proportions,
    optimal_proportions,
    random_proportions,
)
from egtsat.metrics import mean_payoff

N_LIST = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
N_SEEDS_ABM = 3


def payoff_at_final(payoff, x, t=1e6):
    """在近似收敛点上评估平均效用。"""
    x = np.clip(x, 0.0, None)
    return mean_payoff(x / x.sum(), payoff(x / x.sum(), t))


def main() -> None:
    m = 4
    curves: dict[str, list[float]] = {k: [] for k in
                                      ["EGT (ODE)", "EGT (agents)", "Optimal",
                                       "Nearest", "Fixed", "Random"]}
    for n_users in N_LIST:
        scenario = make_scenario(num_users=n_users)
        payoff = make_payoff(scenario)
        x0 = np.full(m, 1.0 / m)

        _, X = solve_replicator(x0, payoff, t_span=(0.0, 120.0), n_eval=61)
        curves["EGT (ODE)"].append(payoff_at_final(payoff, X[-1]))

        abm = [
            simulate_game(x0, payoff, n_steps=400, beta=0.1, num_agents=n_users,
                          rng=np.random.default_rng(seed)).mean_payoff[-50:].mean()
            for seed in range(N_SEEDS_ABM)
        ]
        curves["EGT (agents)"].append(float(np.mean(abm)))

        x_opt = optimal_proportions(payoff, m)
        curves["Optimal"].append(mean_payoff(x_opt, payoff(x_opt)))

        x_near = nearest_proportions(hotspot_users(n_users), scenario)
        curves["Nearest"].append(mean_payoff(x_near, payoff(x_near)))

        x_fixed = fixed_proportions(scenario)
        curves["Fixed"].append(mean_payoff(x_fixed, payoff(x_fixed)))

        rng = np.random.default_rng(7)
        curves["Random"].append(float(np.mean([
            mean_payoff(x, payoff(x)) for x in
            (random_proportions(m, rng) for _ in range(20))
        ])))
        print(f"N={n_users:5d}  "
              + "  ".join(f"{k}={vals[-1]:.3f}" for k, vals in curves.items() if vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "Optimal": dict(color="black", ls=":", lw=2),
        "EGT (ODE)": dict(marker="o", lw=2),
        "EGT (agents)": dict(marker="s", ls="--", lw=1.5),
        "Nearest": dict(marker="^", lw=1.5),
        "Fixed": dict(marker="v", lw=1.5),
        "Random": dict(marker="d", lw=1.5, alpha=0.8),
    }
    for name, values in curves.items():
        ax.plot(N_LIST, values, label=name, **styles.get(name, {}))
    ax.set_xlabel("Number of users N")
    ax.set_ylabel("Mean payoff at equilibrium")
    ax.set_title("Evolutionary-game selection approaches the social optimum")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "exp2_utility_vs_N.png")


if __name__ == "__main__":
    main()
