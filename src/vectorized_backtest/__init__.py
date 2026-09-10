"""vectorized_backtest: 向量化多因子回测引擎(雏形)

给定(可以是多个)因子, 按固定频率调仓、期间买入持有, 算出多空组合的
净值曲线和一些诊断指标。全程以 `(date, ticker)` 为索引的长表/面板数据
流转, 只有真正需要矩阵运算的地方(以后 MPT 的协方差/优化器)才局部转
宽表。

内部按"这一步在管什么"分了几个文件夹, 而不是平铺一堆 .py:
    rebalance.py     -- 哪天是调仓日(跨好几个模块都要用, 放在顶层)
    signals/         -- 价格/因子面板 -> 一个"该不该买、买多少"的信号
                        (收益率、多因子标准化合成、IC 打分期望收益)
    portfolio/       -- 信号 -> 组合权重(分位数多空、以后的 MPT)
    engine.py        -- 权重 -> 逐日净值(这个包真正的核心)
    evaluation/       -- 净值/权重/IC 序列 -> 夏普、回撤、换手率、IR
    io/              -- 唯一 import load 的地方, 把存好的 parquet 拼成
                        上面这些模块要用的面板形状

除了 io/ 里那一处, 其余模块不 import load、也不 import momfactor
(因子已经算好存在 parquet 里了, 这里不重新算因子)——跟 sources/load/
momfactor 之间"不依赖、不 import 上下游"的规矩一致, 只是这里全部收在
一个包里, 用文件夹分工, 不是拆成好几个 git 仓库(试过拆成独立的
`backtesttools` 包, 两个仓库要互相对齐版本、多一层管理成本, 对现在
这个规模不划算, 就合回来了; 分文件夹已经能达到"看代码知道这段是干嘛的"
这个目的)。

公开 API:
    rebalance_dates / rebalance_block             -- 调仓日期的生成与归属
    daily_returns / forward_returns               -- 收益率计算
    zscore_by_date / combine_scores               -- 多因子标准化与合成
    rolling_ic / rolling_sigma / ic_scaled_alpha   -- 期望收益(μ)估计
    quantile_long_short / mean_variance            -- 仓位构造
    run / BacktestResult                           -- 组合净值演算
    sharpe_ratio / max_drawdown / turnover /
    information_ratio                              -- 评估指标
    factor_panel / price_panel                     -- 从 load 的 parquet 拼面板
"""
from __future__ import annotations

from vectorized_backtest.engine import BacktestResult, run
from vectorized_backtest.evaluation import (
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from vectorized_backtest.io import factor_panel, price_panel
from vectorized_backtest.portfolio import mean_variance, quantile_long_short
from vectorized_backtest.rebalance import rebalance_block, rebalance_dates
from vectorized_backtest.signals import (
    combine_scores,
    daily_returns,
    forward_returns,
    ic_scaled_alpha,
    rolling_ic,
    rolling_sigma,
    zscore_by_date,
)

__all__ = [
    "BacktestResult",
    "combine_scores",
    "daily_returns",
    "factor_panel",
    "forward_returns",
    "ic_scaled_alpha",
    "information_ratio",
    "max_drawdown",
    "mean_variance",
    "price_panel",
    "quantile_long_short",
    "rebalance_block",
    "rebalance_dates",
    "rolling_ic",
    "rolling_sigma",
    "run",
    "sharpe_ratio",
    "turnover",
    "zscore_by_date",
]


def main() -> None:
    print("Hello from vectorized-backtest!")
