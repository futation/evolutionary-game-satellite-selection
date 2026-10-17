# egtsat — 演化博弈卫星接入选择

> 中文说明（英文主文档见 [README.md](README.md)）

一个小而自包含的 Python 包，演示**演化博弈论如何解决低轨卫星（LEO）接入选择问题**：
大量地面用户反复选择接入哪颗卫星，两种分布式决策机制——**复制动态方程**
（群体层面的理论）与**成对比较模仿**（个体层面的算法）——把用户分布推向一个
在**容量、排队时延、信任**三者之间取得平衡的均衡点。

本仓库是教科书方法（复制动态、M/M/1 排队、香农容量）的**独立教学实现**，
不是任何论文的官方代码。所有模型的经典出处见文末参考文献。

## 为什么做这个

"1000 个分散的用户，如何在没有中心控制器的前提下选卫星，既不把好卫星挤爆，
也不盲从口碑差的卫星？"——演化博弈论对此有干净的答案，但 GitHub 上可运行、
经过测试的开源实现很少。本仓库用约 600 行带注释的代码和 4 个可复现实验填补这个空缺。

## 模型原理

每颗卫星的**效用**由三个归一化分量构成（统一用半饱和函数 `s/(s+half)` 映射到 `[0,1]`）：

```
U_i(x) = w_c · 容量_i(x) + w_d · 时延_i(x) + w_t · 信任_i
```

- **容量**：每用户香农速率 `R_i / (1 + x_i·N)`——卫星越拥挤，人均速率越低；
- **时延**：M/M/1 平均逗留时间 `1/(μ_i − λ_i)`，其中 `λ_i = x_i·N·λ_user`；
  在稳定域边界 `λ = (1−ε)μ` 处截断（队列失稳是模型边界，不是数值事故）；
- **信任**：每颗卫星的外生声誉分。

`x` 是各卫星承载的用户比例向量。两个更新过程作用于 `x`：

| 过程 | 更新规则 | 角色 |
|---|---|---|
| 复制动态 | `dx_i/dt = x_i (U_i − Ū)` | 理论：数值积分 ODE |
| 主体模仿 | 每个用户随机抽取一个模仿对象，以概率 `β·(U_j − U_cur)⁺` 切换到对方策略 | 分布式算法，任何用户可独立执行 |

模仿规则的**均场极限恰好是复制动态方程**（成对效用差的正负部分精确相消），
因此主体仿真必须——也确实——收敛到 ODE 轨迹上。实验 1 验证了这一点，
`tests/test_game.py` 在 CI 中持续断言它。

## 快速上手

```bash
git clone https://github.com/futation/evolutionary-game-satellite-selection
cd evolutionary-game-satellite-selection
pip install -e .[dev]

pytest            # 12 个性质测试，< 1 秒
python experiments/exp1_convergence.py    # 重新生成全部图片到 figures/
```

最小示例：

```python
import numpy as np
from egtsat import default_scenario, make_payoff, simulate_game, solve_replicator

scenario = default_scenario()          # 4 颗异构 LEO 卫星，N = 1000
payoff = make_payoff(scenario)
x0 = np.full(scenario.m, 0.25)

t, trajectory = solve_replicator(x0, payoff)             # 理论
result = simulate_game(x0, payoff, num_agents=1000)      # 分布式算法
print(result.final_proportions)                          # ≈ trajectory[-1]
```

## 实验一览

所有图片由 `experiments/` 下脚本生成，无手工修饰。

| 图 | 内容 |
|---|---|
| ![收敛验证](figures/exp1_convergence.png) | **验证**：模仿动态（右，6 种子均值±标准差）落在复制动态 ODE（左）上，终点平均效用一致到小数点后 4 位。 |
| ![效用 vs N](figures/exp2_utility_vs_N.png) | **近似最优**：N = 100…1000 全程，博弈均衡紧贴社会最优（点线），显著高于最近邻/固定/随机；拥塞越重差距越大。 |
| ![负载均衡](figures/exp3_load_balance.png) | **负载均衡**：接入比例收敛；Jain 指数 0.712，而"都挤最优卫星"的固定策略仅 0.481。 |
| ![环境突变](figures/exp4_env_shift.png) | **自适应**：最优卫星信噪比骤降后用户自动迁走，系统效用收敛到**新环境**的社会最优——全程无中心控制器。 |

## 仓库结构

```
src/egtsat/
├── config.py      # Satellite / Scenario 数据类，默认 4 星场景
├── utility.py     # 容量 / M/M/1 时延 / 信任 → 统一 payoff 函数
├── replicator.py  # 复制动态 ODE（scipy.solve_ivp）
├── game.py        # 主体级成对比较模仿
├── baselines.py   # 社会最优（SLSQP）、最近邻、固定、随机
└── metrics.py     # Jain 指数、负载变异系数、平均效用
experiments/       # exp1–exp4，各自生成一张图
tests/             # 性质测试，含均场一致性验证
```

## 扩展

- 任意替换效用分量——一切都只是比例向量 `x`（以及时间）的函数：
  时变环境参见 `exp4` 与 `make_payoff(snr_of_t=...)`。
- 通过 `simulate_game(reputation=..., reputation_step=...)` 与
  `make_payoff(trust_of_t=...)` 接入动态声誉模型。
- 扩展到更多卫星：payoff 对 `M` 向量化，8 星、16 星只需改 `Scenario`。

Roadmap：遍历保密容量分量、真实轨道递推 `d_i(t)`、声誉 EMA 演示实验。

## 参考文献

- J. Hofbauer & K. Sigmund, *Evolutionary Games and Population Dynamics*, Cambridge University Press, 1998.
- J. W. Weibull, *Evolutionary Game Theory*, MIT Press, 1995（第 5 章：模仿动态）.
- J. Björnerstedt & J. W. Weibull, "Homo economicus and homo sociologicus: an elementary cartoon," 1996 —— "成对模仿 ⇒ 复制动态"的出处.
- L. Kleinrock, *Queueing Systems, Vol. 1*, Wiley, 1975（M/M/1）.
- 博弈论卫星选择方向的公开工作：
  [基于群与演化博弈的 LEO 切换算法 (2018)](https://jns.nju.edu.cn/EN/Y2018/V54/I4/855) ·
  [超密集 LEO 网络中 Cybertwin 辅助的模式选择](https://www.researchgate.net/publication/366842756_Cybertwin_Assisted_Mode_Selection_in_Ultra_Dense_LEO_Integrated_Satellite_Terrestrial_Network) ·
  [GT4Satellite：卫星网络博弈论文献集](https://github.com/jwwthu/GT4Satellite).

## 许可证

[MIT](LICENSE)
