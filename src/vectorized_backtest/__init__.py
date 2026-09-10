"""vectorized_backtest: 向量化回测引擎(雏形)

只负责一件事: 给定调仓日的目标权重, 按"调仓日定权重、期间买入持有"的
方式算组合净值——这是"回测引擎"这一层, 不负责仓位该怎么算(因子面板、
多因子合成、期望收益、仓位构造、评估指标这些跟"用哪种方式模拟账户"
无关的逻辑, 都在上游的
[`backtesttools`](https://github.com/20070316lbw-netizen/backtesttools)
包里, 这里只 `import backtesttools.rebalance` 拿"哪天是调仓日"这一个
工具函数)。

以后如果要做事件驱动回测(逐笔/逐事件模拟, 而不是这种整段买入持有的
向量化算法), 会是另一个独立的包, 跟这里平级, 共用同一个
`backtesttools`, 互相不依赖。

公开 API:
    run / BacktestResult   -- 组合净值演算
"""
from __future__ import annotations

from vectorized_backtest.engine import BacktestResult, run

__all__ = [
    "BacktestResult",
    "run",
]


def main() -> None:
    print("Hello from vectorized-backtest!")
