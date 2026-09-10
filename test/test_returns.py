"""returns.py 的单元测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from vectorized_backtest.signals.returns import daily_returns, forward_returns


def test_daily_returns_arithmetic_matches_manual_calc(two_ticker_prices):
    prices, dates = two_ticker_prices
    dr = daily_returns(prices, log=False)
    price_a_day0 = prices.loc[(dates[0], "A"), "adj_close"]
    price_a_day1 = prices.loc[(dates[1], "A"), "adj_close"]
    assert dr[(dates[1], "A")] == pytest.approx(price_a_day1 / price_a_day0 - 1)
    assert pd.isna(dr[(dates[0], "A")])


def test_forward_returns_matches_manual_calc(two_ticker_prices):
    prices, dates = two_ticker_prices
    fr = forward_returns(prices, horizon=3)
    price_a_day0 = prices.loc[(dates[0], "A"), "adj_close"]
    price_a_day3 = prices.loc[(dates[3], "A"), "adj_close"]
    assert fr[(dates[0], "A")] == pytest.approx(price_a_day3 / price_a_day0 - 1)


def test_forward_returns_rejects_non_positive_horizon(two_ticker_prices):
    prices, _ = two_ticker_prices
    with pytest.raises(ValueError):
        forward_returns(prices, horizon=0)
