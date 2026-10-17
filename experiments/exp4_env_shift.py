"""实验 4：环境突变下的自适应——最优卫星突然劣化时种群如何迁移。

在第 300 步让原本最优的 Sat A 信噪比骤降（模拟故障/干扰），
观察平均效用的瞬时跌落与用户向其余卫星的迁移恢复。
这是分布式演化算法"无需中心控制器即可适应环境变化"的直观展示。

运行：python experiments/exp4_env_shift.py
"""

from __future__ import annotations

import numpy as np

from _common import make_scenario, save_fig
import matplotlib.pyplot as plt

from egtsat import make_payoff, simulate_game

SHOCK_STEP = 300
N_STEPS = 600
SAT_INDEX = 0  # Sat A
SNR_AFTER_DB = 6.0


def main() -> None:
    scenario = make_scenario()
    base_snr = scenario.snr_linear.copy()

    def snr_of_t(t: float) -> np.ndarray:
        """t >= SHOCK_STEP 后 Sat A 信道质量骤降。"""
        snr = base_snr.copy()
        if t >= SHOCK_STEP:
            snr[SAT_INDEX] = 10 ** (SNR_AFTER_DB / 10)
        return snr

    payoff = make_payoff(scenario, snr_of_t=snr_of_t)
    x0 = np.full(scenario.m, 1.0 / scenario.m)

    result = simulate_game(x0, payoff, n_steps=N_STEPS, beta=0.1,
                           rng=np.random.default_rng(0), num_agents=scenario.num_users)
    steps = np.arange(N_STEPS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(steps, result.mean_payoff, lw=2, color="C2")
    ax.axvline(SHOCK_STEP, color="C3", ls="--", lw=1.5)
    ax.annotate("Sat A degraded", xy=(SHOCK_STEP, result.mean_payoff.max() * 0.97),
                xytext=(SHOCK_STEP + 15, result.mean_payoff.max() * 0.97),
                color="C3", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean payoff")
    ax.set_title("System payoff recovers after the shock")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i in range(scenario.m):
        ax.plot(steps, result.proportions[:, i], lw=2, label=scenario.satellites[i].name)
    ax.axvline(SHOCK_STEP, color="C3", ls="--", lw=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Population share")
    ax.set_title("Users migrate away from the degraded satellite")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Distributed adaptation: Sat A SNR {scenario.satellites[0].snr_db:.0f} dB -> {SNR_AFTER_DB:.0f} dB at step {SHOCK_STEP}",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, "exp4_env_shift.png")

    pre = result.mean_payoff[SHOCK_STEP - 1]
    trough = result.mean_payoff[SHOCK_STEP:SHOCK_STEP + 40].min()
    post = result.mean_payoff[-100:].mean()
    print(f"mean payoff before shock : {pre:.4f}")
    print(f"mean payoff trough       : {trough:.4f}")
    print(f"mean payoff recovered    : {post:.4f}")
    print("shares before / after    :",
          np.round(result.proportions[SHOCK_STEP - 1], 3), "->",
          np.round(result.proportions[-1], 3))


if __name__ == "__main__":
    main()
