"""对照基线：与演化博弈解比较的四种经典接入选择方式。

- ``optimal_proportions``: 在单纯形上直接最大化平均效用（社会最优），
  性能上界；用 SLSQP 从多个起点求解，避免陷入局部最优。
- ``nearest_proportions``: 用户接入星下点最近的卫星——纯地理规则，
  不感知负载与服务能力。
- ``fixed_proportions``: 固定比例堆积到"名义速率最高"的卫星——
  对应现实中盲目跟风选热门节点的行为。
- ``random_proportions``: Dirichlet 随机比例——无信息基线。

所有函数返回与演化博弈解同构的比例向量 (M,)，可直接代入同一个
payoff 函数计算平均效用（exp2）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from egtsat.config import Scenario
from egtsat.metrics import mean_payoff
from egtsat.utility import PayoffFn


def optimal_proportions(payoff: PayoffFn, m: int, n_starts: int = 4,
                        t: float = 0.0) -> np.ndarray:
    """社会最优比例：max_x sum_i x_i U_i(x)，约束 sum(x)=1, x>=0。

    多起点 SLSQP；静态场景传 t=0。返回比例向量 (M,)。
    """
    def objective(x: np.ndarray) -> float:
        return -mean_payoff(x, payoff(np.clip(x, 0.0, None), t))

    constraints = {"type": "eq", "fun": lambda x: x.sum() - 1.0}
    bounds = [(0.0, 1.0)] * m
    best_x, best_val = None, np.inf
    for k in range(n_starts):
        x_start = np.full(m, 1.0 / m) if k == 0 else _rand_simplex(m, np.random.default_rng(k))
        res = minimize(objective, x_start, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 300, "ftol": 1e-10})
        if res.success and res.fun < best_val:
            best_x, best_val = res.x, res.fun
    if best_x is None:
        raise RuntimeError("社会最优求解未收敛")
    return np.clip(best_x, 0.0, None) / np.clip(best_x, 0.0, None).sum()


def nearest_proportions(user_positions: np.ndarray, scenario: Scenario) -> np.ndarray:
    """最近邻基线：每个用户接入星下点几何距离最近的卫星。

    Args:
        user_positions: 用户坐标 (N, 2)。
        scenario:       提供卫星星下点坐标。
    """
    sats = scenario.ground_positions                       # (M, 2)
    dist = np.linalg.norm(user_positions[:, None, :] - sats[None, :, :], axis=2)  # (N, M)
    choices = dist.argmin(axis=1)                          # (N,)
    return np.bincount(choices, minlength=scenario.m) / len(user_positions)


def fixed_proportions(scenario: Scenario, dominant_share: float = 0.7) -> np.ndarray:
    """固定基线：dominant_share 的用户固定选择速率最高的卫星，其余均匀分给其余卫星。"""
    m = scenario.m
    best = int(scenario.snr_linear.argmax())
    props = np.full(m, (1.0 - dominant_share) / max(m - 1, 1))
    props[best] = dominant_share
    return props


def random_proportions(m: int, rng: np.random.Generator) -> np.ndarray:
    """随机基线：Dirichlet(1) 采样的单纯形均匀随机比例。"""
    return rng.dirichlet(np.ones(m))


def _rand_simplex(m: int, rng: np.random.Generator) -> np.ndarray:
    """单纯形上的均匀随机点，用作多起点优化的扰动起点。"""
    x = rng.dirichlet(np.ones(m))
    return 0.5 * x + 0.5 / m
