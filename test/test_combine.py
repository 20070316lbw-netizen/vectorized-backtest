"""combine.py 的单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectorized_backtest.combine import combine_scores, zscore_by_date


@pytest.fixture
def two_day_factors() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "ticker"])
    return pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [10, np.nan, 30, 40, 50, 60],
        },
        index=idx,
    )


def test_zscore_by_date_is_mean_0_std_1_within_each_date(two_day_factors):
    dates = two_day_factors.index.get_level_values("date").unique()
    z = zscore_by_date(two_day_factors)
    assert z.loc[(dates[0], "A"), "f1"] == pytest.approx(-1.0)
    assert z.loc[(dates[0], "B"), "f1"] == pytest.approx(0.0)
    assert z.loc[(dates[0], "C"), "f1"] == pytest.approx(1.0)


def test_combine_scores_ignores_missing_factor_instead_of_treating_as_zero(two_day_factors):
    dates = two_day_factors.index.get_level_values("date").unique()
    z = zscore_by_date(two_day_factors)
    combo = combine_scores(z)
    # (dates[0], "B") 的 f2 是 NaN, 只有 f1 参与, 综合分数应该就等于 f1 的 z-score
    assert combo.loc[(dates[0], "B")] == pytest.approx(z.loc[(dates[0], "B"), "f1"])


def test_combine_scores_rejects_unknown_factor_in_weights(two_day_factors):
    z = zscore_by_date(two_day_factors)
    with pytest.raises(KeyError):
        combine_scores(z, weights={"not_a_factor": 1.0})
