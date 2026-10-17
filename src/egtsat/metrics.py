"""系统级指标：用于在实验中量化"负载均衡程度"与整体性能。"""

from __future__ import annotations

import numpy as np


def mean_payoff(x: np.ndarray, u: np.ndarray) -> float:
    """种群平均效用 = 比例加权的效用均值，即复制动态中的 U_bar。"""
    return float(np.asarray(u) @ np.asarray(x))


def jain_index(loads: np.ndarray) -> float:
    """Jain 公平性指数：1 为完全均衡，M 为完全独占。"""
    loads = np.asarray(loads, dtype=float)
    return float(loads.sum() ** 2 / (loads.size * (loads**2).sum()))


def load_cv(loads: np.ndarray) -> float:
    """负载变异系数（标准差/均值）：越小越均衡；均值 0 时返回 0。"""
    loads = np.asarray(loads, dtype=float)
    mean = loads.mean()
    return float(loads.std() / mean) if mean > 0 else 0.0
