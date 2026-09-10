# vectorized-backtest

个人向量化回测引擎: 给定调仓日的目标权重, 按"调仓日定权重、期间买入
持有"的方式算组合净值曲线。

只负责净值模拟这一件事——因子面板拼接、多因子合成、期望收益估计、
仓位构造(分位数多空、以后的 MPT)、评估指标这些跟"用哪种方式模拟账户"
无关的逻辑, 都在
[`backtesttools`](https://github.com/20070316lbw-netizen/backtesttools)
包里, 这里只 `import backtesttools.rebalance` 拿"哪天是调仓日"这一个
工具函数。依赖方向是单向的: 这里可以 `import backtesttools`, 反过来
不行。

以前这个包里塞了因子合成、期望收益、仓位构造这些逻辑, 后来发现它们跟
"向量化"还是"事件驱动"这种模拟方式完全无关(不管以后拿去配哪种回测
引擎, "这批股票该怎么分仓"这一步的代码应该是同一份), 就把它们拆到
`backtesttools` 去了, 这里只剩 `engine.py` 一个核心模块。

## 快速开始

```python
from backtesttools import factor_panel, price_panel, combine_scores, zscore_by_date, \
    daily_returns, forward_returns, rebalance_dates, rolling_ic, rolling_sigma, \
    ic_scaled_alpha, quantile_long_short, sharpe_ratio, max_drawdown
from vectorized_backtest import run

FREQ = 21  # 调仓间隔(交易日)

# 仓位怎么算是 backtesttools 的事(见它的 README), 这里只演示最后一步:
# 拿到调仓日的目标权重之后, 交给 run() 算净值。
weights = ...  # backtesttools.quantile_long_short(...) 或以后的 mean_variance(...) 的输出
prices = price_panel("data/prices_mi.parquet")

result = run(weights, prices, freq=FREQ)
print(sharpe_ratio(result.nav, periods_per_year=252))
print(max_drawdown(result.nav))
```

## 设计取舍

- **全程以 `(date, ticker)` 为索引的长表/面板数据流转**: 传进来的
  `weights`、算出来的 `nav`/`shares`/`daily_value` 都是 `[date, ticker]`
  MultiIndex 的 `pd.Series`, 跟 `backtesttools` 的输出天然对得上。
- **调仓频率固定, 不天天调仓**: 天天调仓在向量化回测里数值上免费, 但
  现实里是真金白银的交易成本, 所以固定"每 N 个交易日调一次仓", 非调仓
  日不交易、不重新算权重, 组合按调仓日买入的份额一直持有到下次调仓,
  期间市值自然漂移。
- **净值口径是"本金 x (1 + 累计盈亏比例)", 不是持仓市值加总**: 多空
  dollar-neutral 的组合, 每天把多头空头市值加总只剩下净盈亏、本金那部分
  会丢掉, 所以 `engine.run` 的 `nav` 是按这个口径算的; 持仓市值
  (`daily_value`)、份额(`shares`)是另一套东西, 只用来报告敞口, 不能
  拿去加总当 nav 用。这一段的数值在 `test/test_engine.py` 里是手算验证
  过的, 不是回归测试上一次跑出来的数字。
- **"分段买入持有"的复利全程向量化, 唯一的链式依赖是调仓日这个小数组
  上的 `cumprod`**: 每个调仓区间内, 每只票相对区间起点的价格比率可以
  整段一次性算(`groupby` + `transform`), 区间与区间之间"下一区间起始
  资产 = 上一区间期末资产"这个接力关系, 用 `cumprod` 在调仓日数组
  (比如 5 年数据也就几十个)上一次性算完, 不需要遍历全部交易日。

## 当前状态

`engine.run` 已实现、有单测(手算净值路径, 见 `test/test_engine.py`)。
仓位怎么构造(包括还没实现的 MPT)是 `backtesttools` 的事, 见那个包的
README。

## 开发

```bash
uv sync
uv run ruff check .
uv run pytest -v
```
