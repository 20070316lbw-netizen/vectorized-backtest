"""portfolio.py 的单元测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from vectorized_backtest.portfolio import quantile_long_short


@pytest.fixture
def one_day_score() -> pd.Series:
    date = pd.Timestamp("2024-01-01")
    s = pd.Series({"A": 5.0, "B": 1.0, "C": -1.0, "D": -5.0})
    s.index = pd.MultiIndex.from_product([[date], s.index], names=["date", "ticker"])
    return s


def test_quantile_long_short_is_dollar_neutral_within_each_date(one_day_score):
    w = quantile_long_short(one_day_score, n_quantiles=2)
    assert w.sum() == pytest.approx(0.0)


def test_quantile_long_short_longs_the_top_and_shorts_the_bottom(one_day_score):
    date = one_day_score.index.get_level_values("date")[0]
    w = quantile_long_short(one_day_score, n_quantiles=2)
    assert w[(date, "A")] > 0
    assert w[(date, "D")] < 0


def test_quantile_long_short_rejects_fewer_than_2_quantiles(one_day_score):
    with pytest.raises(ValueError):
        quantile_long_short(one_day_score, n_quantiles=1)
