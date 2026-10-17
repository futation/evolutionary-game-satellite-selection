"""复制动态方程（replicator dynamics）及其数值求解。

    dx_i/dt = x_i * ( U_i(x) - U_bar(x) ),    U_bar = sum_j x_j * U_j

几何意义：x 在单纯形（各比例非负、和为 1）上流动，效用高于平均值的
策略，其种群占比上升；等效用策略之间互不流动。

它在本仓库中的角色是**理论基准**：当下一个体的更新规则是"成对比较模仿"
（见 game.py，按效用差为正的概率切换到被模仿者）时，无限总体的均场极限
恰为上述方程（Björnerstedt & Weibull, 1996；Weibull, *Evolutionary Game
Theory*, 1995, ch.5）。因此 exp1 中 ODE 曲线与 ABM 轨迹均值应当重合，
重合本身就是对算法实现正确性的验证。
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from egtsat.utility import PayoffFn


def replicator_rhs(t: float, x: np.ndarray, payoff: PayoffFn) -> np.ndarray:
    """复制动态右端项。数值积分中 x 可能出现 -1e-16 级别的越界，
    计算 payoff 前截回单纯形，避免下溢噪声进入物理量。"""
    x = np.clip(x, 0.0, None)
    x = x / x.sum()
    u = payoff(x, t)
    return x * (u - u @ x)


def solve_replicator(
    x0: np.ndarray,
    payoff: PayoffFn,
    t_span: tuple[float, float] = (0.0, 24.0),
    n_eval: int = 241,
    rtol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """从初始比例 x0 积分复制动态方程。

    Returns:
        t:  均匀采样的时间点，(n_eval,)。
        X:  对应的种群比例轨迹，(n_eval, M)。
    """
    x0 = np.asarray(x0, dtype=float)
    x0 = x0 / x0.sum()
    sol = solve_ivp(
        replicator_rhs,
        t_span,
        x0,
        args=(payoff,),
        method="RK45",
        t_eval=np.linspace(t_span[0], t_span[1], n_eval),
        rtol=rtol,
    )
    if not sol.success:
        raise RuntimeError(f"复制动态积分失败: {sol.message}")
    return sol.t, sol.y.T
