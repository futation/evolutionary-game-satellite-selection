"""基于主体的模仿动态（pairwise comparison imitation）。

每个时间步，每个主体（用户）：

1. 观察自己当前策略的效用 ``U_cur``；
2. 按当前种群比例随机抽取一个模仿对象，得其策略 ``j`` 和效用 ``U_j``
   （等价于从总体中均匀抽一个人并沿用其选择）；
3. 若 ``U_j > U_cur``，以概率 ``beta * (U_j - U_cur)`` 切换到 ``j``；
   效用不高于自己则不动作。

这就是成对比较模仿（pairwise comparison imitation）。它的均场极限恰为
复制动态方程（见 replicator.py）：设 i 类主体转向 j 的速率为
``x_i * x_j * beta * (U_j - U_i)^+``，正负相消后恰好得到
``dx_i/dt = x_i * beta * (U_i - U_bar)``。因此总体规模较大时，本仿真得到
的比例轨迹应围绕 ODE 解随机波动——exp1 用这一性质验证实现的正确性，
它也是"分布式算法收敛到理论均衡"这一结论的依据。

本模块只负责模仿机制本身，不感知任何通信模型：效用完全由 ``payoff``
闭包提供，声誉（信任）等额外状态通过可选回调注入。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from egtsat.utility import PayoffFn

#: 声誉更新回调：输入当前声誉 G、种群比例 x 与时步 t，返回更新后的声誉。
ReputationStep = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class GameResult:
    """演化仿真的时间序列输出。

    Attributes:
        proportions: 每步的种群比例，(n_steps+1, M)，第 0 行为初始分布。
        mean_payoff: 每步的种群平均效用，(n_steps+1,)。
        reputation:  若启用声誉动态，为每步的声誉值 (n_steps+1, M)；否则为 None。
    """

    proportions: np.ndarray
    mean_payoff: np.ndarray
    reputation: np.ndarray | None = None

    @property
    def final_proportions(self) -> np.ndarray:
        """收敛（或终止时刻）的种群比例，(M,)。"""
        return self.proportions[-1]


def simulate_game(
    x0: np.ndarray,
    payoff: PayoffFn,
    n_steps: int = 300,
    beta: float = 0.1,
    rng: np.random.Generator | None = None,
    num_agents: int = 1000,
    reputation: np.ndarray | None = None,
    reputation_step: ReputationStep | None = None,
) -> GameResult:
    """运行有限总体的成对比较模仿仿真。

    Args:
        x0:              初始比例 (M,)，内部归一化。
        payoff:          效用函数 ``U(x, t)``，t 为整数时步；总体规模 N 等
                         场景信息由该闭包持有，不在此处重复传入。
        n_steps:         时步数。
        beta:            模仿强度，0 < beta <= 1；概率 beta*(U_j-U_cur) 天然有界，
                         beta 同时给出与 ODE 时间轴的换算（1 时步 ≈ beta 时间单位）。
        rng:             numpy 随机源，传入可复现。
        num_agents:      主体数量。它是有限总体噪声的来源：越大轨迹越贴近 ODE。
        reputation:      可选的声誉（信任）初值 (M,)。配合 ``reputation_step``
                         使用；仿真过程对它的更新通过回调完成，本函数不解释其语义。
        reputation_step: 可选的声誉更新回调 ``G' = f(G, x, t)``。

    Returns:
        :class:`GameResult`，比例、平均效用（及声誉）的完整轨迹。
    """
    if not 0 < beta <= 1:
        raise ValueError("beta 必须位于 (0, 1]")
    rng = np.random.default_rng() if rng is None else rng
    x0 = np.asarray(x0, dtype=float)
    m = x0.size
    x0 = x0 / x0.sum()

    strategies = rng.choice(m, size=num_agents, p=x0)

    proportions = np.empty((n_steps + 1, m))
    mean_payoff = np.empty(n_steps + 1)
    rep_history = None
    if reputation is not None:
        reputation = np.asarray(reputation, dtype=float).copy()
        rep_history = np.empty((n_steps + 1, m))
        rep_history[0] = reputation

    def _record(step: int, x: np.ndarray, u: np.ndarray) -> None:
        proportions[step] = x
        mean_payoff[step] = u @ x

    x = _shares(strategies, m)
    u = payoff(x, 0)
    _record(0, x, u)

    for step in range(1, n_steps + 1):
        models = strategies[rng.integers(0, num_agents, num_agents)]
        gain = u[models] - u[strategies]
        switch = rng.random(num_agents) < beta * np.maximum(gain, 0.0)
        strategies[switch] = models[switch]

        x = _shares(strategies, m)
        u = payoff(x, step)
        _record(step, x, u)

        if reputation is not None and reputation_step is not None:
            reputation = reputation_step(reputation, x, step)
            rep_history[step] = reputation

    return GameResult(proportions=proportions, mean_payoff=mean_payoff, reputation=rep_history)


def _shares(strategies: np.ndarray, m: int) -> np.ndarray:
    """由策略数组统计各策略占比，(M,)。"""
    return np.bincount(strategies, minlength=m) / strategies.size
