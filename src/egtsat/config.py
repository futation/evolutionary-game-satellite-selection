"""场景与参数配置。

所有模型参数集中在 :class:`Scenario` 数据类里，实验脚本只从它取值，
保证不同实验之间参数一致、且修改一处即可全局生效。

卫星参数（SNR、服务速率、信任值）都是"场景设计值"而非任何真实星座的数据——
教学实现关心的是模型行为，不是复现某个运营商的网络。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Satellite:
    """单颗卫星的静态属性。

    Attributes:
        name:          显示名称（图例用）。
        snr_db:        接收信噪比 (dB)。静态场景中决定链路速率，是卫星间"质量差异"的来源之一。
        service_rate:  任务服务速率 mu (任务/秒)，M/M/1 排队模型的服务强度。
        trust:         外生信任/声誉值，[0,1]。代表用户对该卫星的历史口碑，
                       与动态计算的容量、时延分量共同构成效用。
        ground_pos:    星下点坐标 (km)，仅供"最近邻"基线计算几何距离使用。
    """

    name: str
    snr_db: float
    service_rate: float
    trust: float
    ground_pos: tuple[float, float]


@dataclass(frozen=True)
class Scenario:
    """一次仿真实验的完整参数集。

    Attributes:
        satellites:        卫星列表（数量 M 不限，效用计算对 M 向量化）。
        num_users:         用户总数 N。
        arrival_per_user:  每用户平均任务到达率 lambda_user (任务/秒)。
                           卫星 i 的总到达率为 lambda_i = x_i * N * lambda_user。
        bandwidth_hz:      每颗卫星链路带宽 B (Hz)。
        weight_capacity:   容量分量权重 w_c。
        weight_delay:      时延分量权重 w_d。
        weight_trust:      信任分量权重 w_t。三个权重无需强制归一，但默认和为 1，
                           使 payoff 天然落在 [0,1]，便于模仿概率的解释。
        capacity_half:     容量分数的半饱和点 (bps)：每用户速率 = half 时得分 0.5。
        delay_half:        时延分数的半饱和点 (s)：排队时延 = half 时得分 0.5。
        delay_margin:      M/M/1 稳定域保护带宽 epsilon：到达率超过 (1-epsilon)*mu 时，
                           时延按时延在稳定域边界的取值截断（见 utility 模块说明）。
    """

    satellites: tuple[Satellite, ...]
    num_users: int = 1000
    arrival_per_user: float = 1.0
    bandwidth_hz: float = 20e6
    weight_capacity: float = 0.4
    weight_delay: float = 0.4
    weight_trust: float = 0.2
    capacity_half: float = 0.5e6
    delay_half: float = 2.0e-3
    delay_margin: float = 0.02

    @property
    def m(self) -> int:
        """卫星数量 M。"""
        return len(self.satellites)

    @property
    def snr_linear(self) -> np.ndarray:
        """各卫星线性信噪比，(M,)。"""
        return 10 ** (np.array([s.snr_db for s in self.satellites]) / 10)

    @property
    def service_rates(self) -> np.ndarray:
        """各卫星服务速率 mu，(M,)。"""
        return np.array([s.service_rate for s in self.satellites], dtype=float)

    @property
    def trust_scores(self) -> np.ndarray:
        """各卫星外生信任值，(M,)。"""
        return np.array([s.trust for s in self.satellites], dtype=float)

    @property
    def ground_positions(self) -> np.ndarray:
        """各卫星星下点坐标，(M, 2)。"""
        return np.array([s.ground_pos for s in self.satellites], dtype=float)

    def __post_init__(self) -> None:
        if self.m == 0:
            raise ValueError("scenario 至少需要一颗卫星")
        if min(self.weight_capacity, self.weight_delay, self.weight_trust) < 0:
            raise ValueError("效用分量权重不能为负")
        if self.trust_scores.min() < 0 or self.trust_scores.max() > 1:
            raise ValueError("信任值必须位于 [0, 1]")
        if self.delay_margin <= 0 or self.delay_margin >= 1:
            raise ValueError("delay_margin 应位于 (0, 1)")


def default_scenario() -> Scenario:
    """默认教学场景：4 颗异构 LEO 卫星 + 1000 个双热点分布的用户。

    卫星从 A 到 D 信道质量（SNR）递减、服务能力（mu）递减，
    信任值刻意与信道质量错开，使"最优接入分布"不是简单的好上加好，
    拥塞最重的卫星到达率逼近其服务速率，演化压力明显。
    """
    return Scenario(
        satellites=(
            Satellite("Sat A", snr_db=20.0, service_rate=1200.0, trust=0.7, ground_pos=(20.0, 80.0)),
            Satellite("Sat B", snr_db=17.0, service_rate=900.0, trust=0.5, ground_pos=(80.0, 80.0)),
            Satellite("Sat C", snr_db=15.0, service_rate=600.0, trust=0.6, ground_pos=(20.0, 20.0)),
            Satellite("Sat D", snr_db=12.0, service_rate=400.0, trust=0.3, ground_pos=(80.0, 20.0)),
        ),
        num_users=1000,
        arrival_per_user=1.0,
        bandwidth_hz=20e6,
    )
