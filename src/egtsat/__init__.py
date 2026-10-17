"""egtsat —— 演化博弈卫星接入选择的教学参考实现。

这个包用尽可能少的代码演示一条完整的技术链路：

    多维效用建模（容量 / 排队时延 / 信任）
        → 复制动态方程（群体层面的理论演化）
        → 成对比例模仿算法（个体层面的分布式实现，均场极限即上述方程）
        → 与经典基线（社会最优 / 最近邻 / 固定 / 随机）对比

设计原则见 README。典型用法：

    from egtsat import default_scenario, make_payoff, solve_replicator, simulate_game

    scenario = default_scenario()
    payoff = make_payoff(scenario)
    t, X = solve_replicator([0.25] * 4, payoff)          # 理论演化
    result = simulate_game([0.25] * 4, payoff, n_steps=300)  # 主体仿真
"""

from egtsat.config import Satellite, Scenario, default_scenario
from egtsat.utility import make_payoff
from egtsat.replicator import solve_replicator
from egtsat.game import GameResult, simulate_game

__all__ = [
    "Satellite",
    "Scenario",
    "default_scenario",
    "make_payoff",
    "solve_replicator",
    "GameResult",
    "simulate_game",
]

__version__ = "0.1.0"
