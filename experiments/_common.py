"""实验公共工具：路径引导、场景构造、双热点用户分布、图片保存。

实验脚本既支持 `pip install -e .` 后直接运行，
也支持克隆仓库后免安装运行（此处把 src 加入 sys.path）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 无显示环境出图

import matplotlib.pyplot as plt  # noqa: E402

from egtsat import Scenario, default_scenario  # noqa: E402

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"


def make_scenario(num_users: int | None = None) -> Scenario:
    """默认场景；可覆盖用户数（exp2 的 N 扫描用）。"""
    base = default_scenario()
    if num_users is None:
        return base
    return Scenario(
        satellites=base.satellites,
        num_users=num_users,
        arrival_per_user=base.arrival_per_user,
        bandwidth_hz=base.bandwidth_hz,
    )


def hotspot_users(n: int, seed: int = 0) -> np.ndarray:
    """双热点用户分布：一半用户聚在 (20,80)，一半在 (80,20)，单位 km。

    模拟灾区/偏远区域中用户聚集在少数定居点的几何形态，
    供最近邻基线使用。
    """
    rng = np.random.default_rng(seed)
    north = rng.random(n) > 0.5
    return np.where(
        north[:, None],
        rng.normal(loc=[20.0, 80.0], scale=10.0, size=(n, 2)),
        rng.normal(loc=[80.0, 20.0], scale=10.0, size=(n, 2)),
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    """把图片存入仓库根目录的 figures/。"""
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")
