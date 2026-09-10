# vectorized-backtest

个人向量化多因子回测引擎: 给定(可以是多个)因子, 按固定频率调仓、期间
买入持有, 算出多空组合的净值曲线和一些诊断指标。

跟 [`load`](https://github.com/20070316lbw-netizen/load)、
[`momfactor`](https://github.com/20070316lbw-netizen/momfactor) 是上下游
关系而不是平级——这里是消费方: 只有 `panels.py` 一处 `import load`(读
存好的 parquet), 不 `import momfactor`(因子已经算好存在 parquet 里了,
这里不重新算因子)。核心计算模块(`returns.py`/`combine.py`/
`expected_returns.py`/`portfolio.py`/`engine.py`/`metrics.py`)不 import
`load`, 只吃/吐 `pandas` 结构, 跟 `momfactor` 的规矩一致。

## 快速开始

```python
from vectorized_backtest import (
    factor_panel, price_panel,          # panels.py, 读 load 存的 parquet
    zscore_by_date, combine_scores,     # combine.py, 多因子横截面标准化 + 合成
    daily_returns, forward_returns,     # returns.py
    rebalance_dates,                    # rebalance.py
    rolling_ic, rolling_sigma, ic_scaled_alpha,  # expected_returns.py
    quantile_long_short,                # portfolio.py
    run,                                # engine.py
    sharpe_ratio, max_drawdown, turnover, information_ratio,  # metrics.py
)

FREQ = 21  # 调仓间隔(交易日)

factors = factor_panel("data/factors_long.parquet")   # 一列一个因子
prices = price_panel("data/prices_mi.parquet")

z = zscore_by_date(factors)
daily_score = combine_scores(z)                        # 每天都有的综合分数

fwd = forward_returns(prices, horizon=FREQ)
ic = rolling_ic(daily_score, fwd, window=250)             # 窗口长度自己定, 没有默认值
sigma = rolling_sigma(daily_returns(prices), window=60)   # 同上

rebal_dates = rebalance_dates(prices.index.get_level_values("date"), freq=FREQ)
rebalance_score = daily_score[daily_score.index.get_level_values("date").isin(rebal_dates)]
mu = ic_scaled_alpha(rebalance_score, sigma, ic)

weights = quantile_long_short(mu, n_quantiles=5)
result = run(weights, prices, freq=FREQ)

print(sharpe_ratio(result.nav, periods_per_year=252))
print(max_drawdown(result.nav))
print(turnover(weights))
```

（上面这段是把各模块串起来的示意, 不是现成脚本; 真正跑一次端到端还没
接线, 见下面"当前状态"。）

## 设计取舍

- **全程以 `(date, ticker)` 为索引的长表/面板数据流转**: 因子分数、
  权重、组合收益都是 `[date, ticker]` MultiIndex 的 `pd.Series`, 跟
  `load` 的因子长表天然对得上。只有真正需要矩阵运算的地方(`portfolio.
  mean_variance` 要算协方差矩阵)才局部 `.unstack()` 成 `date x ticker`
  宽表, 算完再 `.stack()` 回来——不为了以后的 MPT 就把整个引擎都改成
  宽表。
- **调仓频率固定, 不天天调仓**: 天天调仓在向量化回测里数值上免费, 但
  现实里是真金白银的交易成本, 所以固定"每 N 个交易日调一次仓"(见
  `rebalance.py`), 非调仓日不交易、不重新算权重, 组合按调仓日买入的
  份额一直持有到下次调仓, 期间市值自然漂移。
- **净值口径是"本金 x (1 + 累计盈亏比例)", 不是持仓市值加总**: 多空
  dollar-neutral 的组合, 每天把多头空头市值加总只剩下净盈亏、本金那部分
  会丢掉, 所以 `engine.run` 的 `nav` 是按这个口径算的; 持仓市值
  (`daily_value`)、份额(`shares`)是另一套东西, 只用来报告敞口, 不能
  拿去加总当 nav 用。这一段的数值在 `test/test_engine.py` 里是手算验证
  过的, 不是回归测试上一次跑出来的数字。
- **只做多空, 不做纯多头**: 多空刚好配合以后要做的资金分散/错峰调仓
  (把资金分散成几份、错开约 `freq / n` 个交易日入场, 摊平单次调仓点的
  运气成分, 类似经典的 overlapping portfolio 做法)——这个还没做, 但
  架构上不冲突, 是 `engine.run` 之外再加一层"跑 N 份错开的策略、把权重
  加总/平均"的外层循环, `engine.run` 本身不用改。
- **无量纲的因子分数换算成收益率量纲的期望收益, 用的是 Grinold-Kahn
  的 IC 打分模型**(`expected_returns.ic_scaled_alpha`): `μ = IC × σ ×
  score`。`score` 是横截面标准化后的综合分数(均值 0、标准差 1, 无量纲),
  `σ` 是个股波动率, `IC` 是这个因子历史上的信息系数——比直接拿个股历史
  均值收益当期望收益噪声小得多(历史均值的估计误差收敛极慢, 是
  Michaud 1989 说的经典的"error maximization"问题)。`score` 用调仓日
  那天的横截面分数, `IC` 用每天(不只调仓日)滚动估计——两者故意不是
  同一个采样频率: 只用调仓日的分数去估 IC, 几年数据也就几十个样本点,
  噪声太大; `score`/`sigma`/`IC` 各自的滚动窗口长度都没有默认值, 需要
  调用方显式传, 这是需要单独想清楚的建模选择。
- **`combine.combine_scores` 按行重新归一化权重, 不把缺失因子当 0**:
  某只票某天缺几个因子不该被当成因子值为 0 参与加权(那样会把分数拉向
  0, 变相惩罚数据不全的股票), 而是用它实际有值的那些因子、按这些因子
  的权重重新归一化。
- **`expected_returns.rolling_ic` 的 spearman 相关系数是手动
  rank + pearson 实现的, 不用 pandas 内置的 `method="spearman"`**——
  那条路径要装 `scipy`, 而秩相关系数数学上就等于"先按截面排秩、再算
  普通 pearson 相关系数", 两者数值完全一致, 没必要为了这一个相关系数
  多引入一个依赖。

## 当前状态

已实现、有单测(`uv run pytest -v`, 见 `test/`):
`rebalance.py`、`returns.py`、`combine.py`、`expected_returns.py`、
`portfolio.quantile_long_short`、`engine.run`、`metrics.py`、
`panels.py`。

还没实现: `portfolio.mean_variance`(均值-方差优化仓位构造), 卡在还没定
用哪个优化器(`scipy.optimize` 还是 `cvxpy`)、以及具体约束(dollar-neutral
的严格程度、杠杆上限、单票权重上限、协方差估计窗口), 调用会直接
`raise NotImplementedError`。

## 开发

```bash
uv sync
uv run ruff check .
uv run pytest -v
```

测试全部用手造的小 DataFrame 算出来验证(`test/test_engine.py` 里那组
净值路径是纸笔手算的), 不触发网络请求, 也不 import `load`(只有
`panels.py` 依赖 `load`, 测试没有覆盖到这个 io 边界, 因为没有现成的
样例 parquet)。
