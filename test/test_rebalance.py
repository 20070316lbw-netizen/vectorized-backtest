"""rebalance.py 的单元测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from vectorized_backtest.rebalance import rebalance_block, rebalance_dates


def test_rebalance_dates_starts_at_earliest_day_and_steps_by_freq():
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    rb = rebalance_dates(dates, freq=3)
    assert list(rb) == [dates[0], dates[3], dates[6]]


def test_rebalance_dates_rejects_non_positive_freq():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    with pytest.raises(ValueError):
        rebalance_dates(dates, freq=0)


def test_rebalance_block_assigns_each_day_to_last_rebalance_on_or_before_it():
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    block = rebalance_block(dates, freq=3)
    expected = {
        dates[0]: dates[0],
        dates[1]: dates[0],
        dates[2]: dates[0],
        dates[3]: dates[3],
        dates[4]: dates[3],
        dates[5]: dates[3],
        dates[6]: dates[6],
    }
    for day, rb_day in expected.items():
        assert block[day] == rb_day
