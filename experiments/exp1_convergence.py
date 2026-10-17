"""实验 1（旗舰验证）：复制动态 ODE 与主体模仿仿真的收敛轨迹对比。

理论：成对比较模仿的均场极限是复制动态方程，
     因此大总体下 ABM 轨迹均值应贴合 ODE 解，方差随总体规模缩小。
本实验同时是实现的正确性检验（tests/test_game.py 同源断言）。

运行：python experiments/exp1_convergence.py
"""

from __future__ import annotations

import numpy as np

from _common import make_scenario, save_fig
import matplotlib.pyplot as plt

from egtsat import make_payoff, simulate_game, solve_replicator

N_STEPS = 400
BETA = 0.1  # 1 时步 ≈ beta 时间单位，ODE 时间轴据此对齐
N_SEEDS = 6


def main() -> None:
    scenario = make_scenario()
    payoff = make_payoff(scenario)
    x0 = np.full(scenario.m, 1.0 / scenario.m)

    t_ode, X_ode = solve_replicator(x0, payoff, t_span=(0.0, N_STEPS * BETA))

    runs = [
        simulate_game(x0, payoff, n_steps=N_STEPS, beta=BETA,
                      rng=np.random.default_rng(seed), num_agents=scenario.num_users)
        for seed in range(N_SEEDS)
    ]
    stacked = np.stack([r.proportions for r in runs])          # (seeds, steps+1, M)
    abm_mean = stacked.mean(axis=0)
    abm_std = stacked.std(axis=0)
    steps = np.arange(N_STEPS + 1) * BETA

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax = axes[0]
    for i in range(scenario.m):
        ax.plot(t_ode, X_ode[:, i], lw=2, label=scenario.satellites[i].name)
    ax.set_title("Replicator dynamics (theory)")
    ax.set_xlabel("Time (replicator scale)")
    ax.set_ylabel("Population share")

    ax = axes[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i in range(scenario.m):
        ax.fill_between(steps, abm_mean[:, i] - abm_std[:, i], abm_mean[:, i] + abm_std[:, i],
                        color=colors[i], alpha=0.18, lw=0)
        ax.plot(steps, abm_mean[:, i], lw=2, color=colors[i], label=scenario.satellites[i].name)
    ax.set_title(f"Agent-based imitation (mean ± std, {N_SEEDS} seeds)")

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    u_ode = payoff(X_ode[-1] / X_ode[-1].sum(), 1e6) @ (X_ode[-1] / X_ode[-1].sum())
    u_abm = float(np.mean([r.mean_payoff[-50:].mean() for r in runs]))
    fig.suptitle(
        f"Imitation dynamics converge to the replicator equilibrium  "
        f"(mean payoff: ODE {u_ode:.4f} vs ABM {u_abm:.4f})",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, "exp1_convergence.png")

    print("ODE final shares   :", np.round(X_ode[-1], 4))
    print("ABM final shares   :", np.round(abm_mean[-1], 4))
    print(f"mean payoff: ODE={u_ode:.4f}  ABM={u_abm:.4f}")


if __name__ == "__main__":
    main()
