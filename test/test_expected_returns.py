"""expected_returns.py 的单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectorized_backtest.signals.expected_returns import ic_scaled_alpha, rolling_ic, rolling_sigma


@pytest.fixture
def predictive_score_and_forward_return():
    """构造一个"分数确实能预测前瞻收益"的合成数据集, 用来验证 rolling_ic
    算出来的方向和量级是合理的(不追求精确数值, 因子研究里 IC 本身就是
    带噪声的统计量)。"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    tickers = [f"T{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    noise = pd.Series(rng.normal(scale=2.0, size=len(idx)), index=idx)
    fwd_return = 0.5 * score + noise
    return score, fwd_return


def test_rolling_ic_is_positive_when_score_actually_predicts_forward_return(
    predictive_score_and_forward_return,
):
    score, fwd_return = predictive_score_and_forward_return
    ic = rolling_ic(score, fwd_return, window=10)
    assert ic.dropna().mean() > 0


def test_rolling_sigma_is_nonnegative(two_ticker_prices):
    from vectorized_backtest.signals.returns import daily_returns

    prices, _ = two_ticker_prices
    daily_ret = daily_returns(prices, log=False)
    sigma = rolling_sigma(daily_ret, window=3)
    assert (sigma.dropna() >= 0).all()


def test_ic_scaled_alpha_multiplies_the_three_inputs_elementwise():
    date = pd.Timestamp("2024-01-01")
    idx = pd.MultiIndex.from_tuples([(date, "A")], names=["date", "ticker"])
    score = pd.Series([2.0], index=idx)
    sigma = pd.Series([0.1], index=idx)
    ic = pd.Series([0.05], index=[date])

    mu = ic_scaled_alpha(score, sigma, ic)
    assert mu[(date, "A")] == pytest.approx(0.05 * 0.1 * 2.0)
