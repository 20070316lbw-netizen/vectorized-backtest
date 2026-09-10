"""vectorized_backtest: 向量化回测引擎包(雏形)

给定调仓日的因子/期望收益, 按"调仓日定权重、期间买入持有"的方式算组合
净值——全程以 (date, ticker) 为索引的长表/面板数据流转(跟 load/momfactor
的约定一致), 只有真正需要矩阵运算的地方(以后 MPT 的协方差/优化器)才
局部转宽表。不做任何存储, 只有 panels.py 一处 import load 作为 io 边界。

内部按职责分文件: rebalance.py 决定哪天是调仓日; returns.py 算前瞻/
逐日收益率; combine.py 做多因子横截面标准化与合成; expected_returns.py
把无量纲的综合分数换算成收益率量纲的期望收益(Grinold-Kahn IC 打分
模型); portfolio.py 把期望收益变成组合权重(分位数多空是向量化实现,
MPT 优化器还没定, 占位); engine.py 把权重变成逐日净值; metrics.py 算
夏普、回撤、换手率、信息比率这些诊断指标; panels.py 是唯一 import load
的地方, 把存好的 parquet 读成上面这些函数要用的面板形状。

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

from vectorized_backtest.combine import combine_scores, zscore_by_date
from vectorized_backtest.engine import BacktestResult, run
from vectorized_backtest.expected_returns import ic_scaled_alpha, rolling_ic, rolling_sigma
from vectorized_backtest.metrics import information_ratio, max_drawdown, sharpe_ratio, turnover
from vectorized_backtest.panels import factor_panel, price_panel
from vectorized_backtest.portfolio import mean_variance, quantile_long_short
from vectorized_backtest.rebalance import rebalance_block, rebalance_dates
from vectorized_backtest.returns import daily_returns, forward_returns

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
