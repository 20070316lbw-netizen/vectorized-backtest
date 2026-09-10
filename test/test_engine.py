"""engine.py 的单元测试。

净值路径是手算出来的(见 conftest.two_ticker_prices 的价格数字和下面
每个测试里的注释), 不是回归测试上一次跑出来的数字——这样如果以后改动
弄错了口径(比如误把持仓市值加总当 nav), 数值会跟手算的对不上而报错。
"""
from __future__ import annotations

import pandas as pd
import pytest

from vectorized_backtest.engine import run


@pytest.fixture
def two_block_weights(two_ticker_prices) -> pd.Series:
    _, dates = two_ticker_prices
    w = pd.Series(
        {
            (dates[0], "A"): 1.0,
            (dates[0], "B"): -1.0,
            (dates[3], "A"): -0.5,
            (dates[3], "B"): 0.5,
        }
    )
    w.index = pd.MultiIndex.from_tuples(w.index, names=["date", "ticker"])
    return w


def test_run_nav_matches_hand_computed_long_short_path(two_ticker_prices, two_block_weights):
    prices, dates = two_ticker_prices
    result = run(two_block_weights, prices, freq=3, log_rebalances=False)

    # 手算(见模块 docstring): 第一区间 +1 A / -1 B, 第二区间 -0.5 A / +0.5 B,
    # 净值口径是"本金 x (1 + 累计盈亏比例)", 不是持仓市值加总。
    expected = {
        dates[0]: 1.0,
        dates[1]: 1.20,
        dates[2]: 0.95,
        dates[3]: 0.95,
        dates[4]: 0.87875,
        dates[5]: 0.95,
        dates[6]: 0.95,
    }
    for day, exp in expected.items():
        assert result.nav[day] == pytest.approx(exp)


def test_run_shares_match_weight_times_starting_nav_over_entry_price(
    two_ticker_prices, two_block_weights
):
    prices, dates = two_ticker_prices
    result = run(two_block_weights, prices, freq=3, log_rebalances=False)

    price_a_day0 = prices.loc[(dates[0], "A"), "adj_close"]
    assert result.shares[(dates[0], "A")] == pytest.approx(1.0 * 1.0 / price_a_day0)

    price_a_day3 = prices.loc[(dates[3], "A"), "adj_close"]
    nav_at_day3 = 0.95  # 见 test_run_nav_matches_hand_computed_long_short_path
    assert result.shares[(dates[3], "A")] == pytest.approx(-0.5 * nav_at_day3 / price_a_day3)


def test_run_rejects_weights_whose_dates_are_not_rebalance_dates(two_ticker_prices):
    prices, dates = two_ticker_prices
    bad_weights = pd.Series({(dates[1], "A"): 1.0})
    bad_weights.index = pd.MultiIndex.from_tuples(bad_weights.index, names=["date", "ticker"])
    with pytest.raises(ValueError):
        run(bad_weights, prices, freq=3, log_rebalances=False)
