"""多维效用函数：把容量、排队时延、信任三个异构指标压缩为可比的 [0,1] 分数。

本模块体现了整个仓库最重要的两条设计决定：

1. 分量归一化统一用半饱和有理函数 ``score = s / (s + half)``：
   单调、有界、无指数溢出，且 ``half`` 有直接的物理含义——
   "物理量等于 half 时恰好得 0.5 分"。不使用 sigmoid + 魔术缩放系数。

2. 效用聚合方式全仓库唯一：
       payoff_i(x) = w_c * cap_i(x) + w_d * delay_i(x) + w_t * trust_i
   复制动态方程（replicator）、主体模仿仿真（game）、最优化基线（baselines）
   共用 :func:`make_payoff` 返回的同一个函数，
   这是"理论均衡 / 仿真收敛 / 社会最优"三者可以严格对比的前提。

M/M/1 平均时延 W = 1/(mu - lambda) 在 lambda -> mu^- 处发散，这是模型边界
（队列失稳）而非数值事故：本模块在 lambda >= (1-margin)*mu 时把时延截断在
稳定域边界的取值 1/(margin*mu)，保证 payoff 有限、单调、可微性无关紧要。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from egtsat.config import Scenario

#: payoff 函数的统一签名：输入比例向量 x 与时间 t，返回各策略效用 (M,)。
#: 时间参数允许实验构造时变环境（如 exp4 的信道突变）。
PayoffFn = Callable[[np.ndarray, float], np.ndarray]


def per_user_rate(snr_linear: np.ndarray, bandwidth_hz: float, x: np.ndarray, num_users: int) -> np.ndarray:
    """每用户速率份额 (bps)。

    卫星 i 的总速率 R_i = B * log2(1 + SNR_i)（香农公式），
    按接入人数均分：R_i / (1 + x_i * N)。分母加 1 避免空载时除零。
    """
    total_rate = bandwidth_hz * np.log2(1.0 + snr_linear)
    return total_rate / (1.0 + x * num_users)


def queue_delay(x: np.ndarray, service_rates: np.ndarray, arrival_per_user: float, num_users: int,
                margin: float = 0.02) -> np.ndarray:
    """M/M/1 平均排队时延 (s)，含稳定域截断，(M,)。

    到达率 lambda_i = x_i * N * arrival_per_user；
    稳定域内 W_i = 1 / (mu_i - lambda_i)；
    lambda_i >= (1 - margin) * mu_i 时取边界值 1 / (margin * mu_i)。
    """
    arrival = x * num_users * arrival_per_user
    headroom = service_rates - arrival
    boundary = 1.0 / (margin * service_rates)
    return np.where(headroom > margin * service_rates, 1.0 / np.maximum(headroom, 1e-12), boundary)


def capacity_score(per_user_rate_: np.ndarray, capacity_half: float) -> np.ndarray:
    """容量分数 = s / (s + half)，(M,)，越大越好。"""
    return per_user_rate_ / (per_user_rate_ + capacity_half)


def delay_score(delay_s: np.ndarray, delay_half: float) -> np.ndarray:
    """时延分数 = half / (half + W)，(M,)，时延越小分数越高。"""
    return delay_half / (delay_half + delay_s)


def make_payoff(
    scenario: Scenario,
    snr_of_t: Callable[[float], np.ndarray] | None = None,
    trust_of_t: Callable[[float], np.ndarray] | None = None,
) -> PayoffFn:
    """构造统一的 payoff 函数 ``U(x, t) -> (M,)``。

    Args:
        scenario:  场景参数。
        snr_of_t:  可选的时变信噪比回调，``snr_of_t(t) -> (M,)``（线性值）。
                   供环境突变实验使用；缺省为静态 SNR。
        trust_of_t: 可选的时变信任回调 ``trust_of_t(t) -> (M,)``；
                   缺省使用场景中的外生信任值。
    """
    weights = np.array([scenario.weight_capacity, scenario.weight_delay, scenario.weight_trust])
    service_rates = scenario.service_rates

    def payoff(x: np.ndarray, t: float = 0.0) -> np.ndarray:
        snr = scenario.snr_linear if snr_of_t is None else np.asarray(snr_of_t(t))
        trust = scenario.trust_scores if trust_of_t is None else np.asarray(trust_of_t(t))
        cap = capacity_score(per_user_rate(snr, scenario.bandwidth_hz, x, scenario.num_users),
                             scenario.capacity_half)
        queuing = queue_delay(x, service_rates, scenario.arrival_per_user, scenario.num_users,
                              scenario.delay_margin)
        return weights[0] * cap + weights[1] * delay_score(queuing, scenario.delay_half) + weights[2] * trust

    return payoff
