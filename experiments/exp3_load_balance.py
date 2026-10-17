"""实验 3：负载均衡——演化博弈如何把拥塞用户疏散到异构卫星上。

左图：各卫星承载的用户比例随时间收敛；
右图：平均效用随时间上升，并与静态基线（固定/最近邻/随机/社会最优）对照。
运行：python experiments/exp3_load_balance.py
"""

from __future__ import annotations

import numpy as np

from _common import hotspot_users, make_scenario, save_fig
import matplotlib.pyplot as plt

from egtsat import make_payoff, simulate_game
from egtsat.baselines import fixed_proportions, nearest_proportions, optimal_proportions, random_proportions
from egtsat.metrics import jain_index, mean_payoff


def main() -> None:
    scenario = make_scenario()
    payoff = make_payoff(scenario)
    m = scenario.m
    x0 = np.full(m, 1.0 / m)

    result = simulate_game(x0, payoff, n_steps=300, beta=0.1,
                           rng=np.random.default_rng(0), num_agents=scenario.num_users)
    steps = np.arange(result.proportions.shape[0])

    static_levels = {}
    x_fixed = fixed_proportions(scenario)
    static_levels["Fixed"] = mean_payoff(x_fixed, payoff(x_fixed, 300))
    x_near = nearest_proportions(hotspot_users(scenario.num_users), scenario)
    static_levels["Nearest"] = mean_payoff(x_near, payoff(x_near, 300))
    rng = np.random.default_rng(7)
    static_levels["Random"] = float(np.mean([
        mean_payoff(x, payoff(x, 300)) for x in (random_proportions(m, rng) for _ in range(20))
    ]))
    x_opt = optimal_proportions(payoff, m)
    static_levels["Optimal"] = mean_payoff(x_opt, payoff(x_opt, 300))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for i in range(m):
        ax.plot(steps, result.proportions[:, i], lw=2, label=scenario.satellites[i].name)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Population share")
    ax.set_title("Convergence of access shares")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(steps, result.mean_payoff, lw=2, color="C2", label="EGT (agents)")
    colors = {"Fixed": "C3", "Nearest": "C4", "Random": "C5", "Optimal": "black"}
    for name, level in static_levels.items():
        ax.axhline(level, ls=":" if name == "Optimal" else "--", lw=1.2,
                   color=colors[name], label=name)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean payoff")
    ax.set_title("System payoff vs static baselines")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    jain_egt = jain_index(result.final_proportions * scenario.num_users)
    jain_fixed = jain_index(x_fixed * scenario.num_users)
    fig.suptitle(
        f"Load balancing: Jain index {jain_egt:.3f} (EGT) vs {jain_fixed:.3f} (Fixed)",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, "exp3_load_balance.png")

    print("EGT final shares :", np.round(result.final_proportions, 4))
    print("Fixed shares     :", np.round(x_fixed, 4))
    print("Nearest shares   :", np.round(x_near, 4))
    print("Optimal shares   :", np.round(x_opt, 4))
    print(f"EGT mean payoff  : {result.mean_payoff[-1]:.4f}  (Jain {jain_egt:.3f})")
    print(f"static levels    : " + "  ".join(f"{k}={v:.3f}" for k, v in static_levels.items()))


if __name__ == "__main__":
    main()
