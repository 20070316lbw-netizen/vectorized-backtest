"""metrics.py 的单元测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from vectorized_backtest.evaluation.metrics import (
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    turnover,
)


def test_max_drawdown_is_negative_after_a_drop():
    nav = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(nav) == pytest.approx(0.9 / 1.2 - 1.0)


def test_sharpe_ratio_is_nan_for_constant_nav():
    nav = pd.Series([1.0, 1.0, 1.0, 1.0])
    assert pd.isna(sharpe_ratio(nav, periods_per_year=252))


def test_turnover_first_rebalance_is_half_of_gross_exposure():
    date = pd.Timestamp("2024-01-01")
    w = pd.Series({"A": 0.5, "B": -0.5})
    w.index = pd.MultiIndex.from_product([[date], w.index], names=["date", "ticker"])
    t = turnover(w)
    assert t[date] == pytest.approx(0.5)


def test_turnover_between_two_rebalances_matches_manual_calc():
    d0, d1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-22")
    w = pd.Series(
        {
            (d0, "A"): 0.5,
            (d0, "B"): -0.5,
            (d1, "A"): 0.5,
            (d1, "C"): -0.5,
        }
    )
    w.index = pd.MultiIndex.from_tuples(w.index, names=["date", "ticker"])
    t = turnover(w)
    # A 不变(0), B: -0.5 -> 0(0.5), C: 0 -> -0.5(0.5), 单边换手 = (0+0.5+0.5)/2
    assert t[d1] == pytest.approx(0.5)


def test_information_ratio_is_mean_over_std():
    ic = pd.Series([0.05, 0.03, 0.07, 0.02, 0.04])
    assert information_ratio(ic) == pytest.approx(ic.mean() / ic.std())
