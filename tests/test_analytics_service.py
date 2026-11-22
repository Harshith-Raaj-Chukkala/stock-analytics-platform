import pandas as pd
import pytest

from app.services.analytics_service import (
    calculate_summary,
    compare_stock_summaries,
    compare_summaries,
)
from app.services.stock_service import calculate_daily_returns


def make_price_df(closes: list[float], volumes: list[int] | None = None) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame indexed by business dates."""
    if volumes is None:
        volumes = [1000] * len(closes)

    dates = pd.date_range(start="2025-01-01", periods=len(closes), freq="B")
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": [float(c) for c in closes],
            "Volume": volumes,
        },
        index=dates,
    )
    df.index.name = "Date"
    return calculate_daily_returns(df)


def test_calculate_summary_core_price_return_and_drawdown_metrics():
    # 5 trading days:
    # Prices: 100 -> 120 (+20%) -> 90 (-25%) -> 90 (0%) -> 110 (+22.2222%)
    # Peak reaches 120 on day 2, drops to 90 on day 3 -> max drawdown = (90 - 120) / 120 = -25.0%
    # Total return = (110 - 100) / 100 = +10.0%
    df = make_price_df(
        closes=[100.0, 120.0, 90.0, 90.0, 110.0],
        volumes=[1000, 2500, 1500, 1200, 1800],
    )

    summary = calculate_summary(df)

    assert summary["latest_price"] == 110.0
    assert summary["highest_price"] == 120.0
    assert summary["lowest_price"] == 90.0
    assert summary["average_close"] == pytest.approx(102.0)
    assert summary["highest_volume"] == 2500

    assert summary["total_return_percent"] == 10.0
    # Mean of [20.0, -25.0, 0.0, 22.22222222222222] = 4.305555... -> rounded to 4.31
    assert summary["average_daily_return"] == 4.31
    # Sample std (ddof=1) of [20.0, -25.0, 0.0, 22.22222222222222] = 21.9444... -> rounded to 21.94
    assert summary["volatility"] == 21.94
    assert summary["max_drawdown"] == -25.0

    # 4 valid daily returns: 2 up, 1 down, 1 unchanged
    assert summary["up_days"] == 2
    assert summary["down_days"] == 1
    assert summary["unchanged_days"] == 1
    assert summary["up_day_percent"] == pytest.approx(50.0)
    assert summary["down_day_percent"] == pytest.approx(25.0)
    assert summary["unchanged_day_percent"] == pytest.approx(25.0)

    # Fewer than 14 daily returns and fewer than 20 closes -> RSI and Bollinger metrics are None
    assert summary["latest_rsi"] is None
    assert summary["rsi_interpretation"] is None
    assert summary["latest_upper_band"] is None
    assert summary["latest_lower_band"] is None
    assert summary["latest_band_width"] is None
    assert summary["bollinger_interpretation"] is None
    assert summary["bandwidth_interpretation"] is None


def test_calculate_summary_single_row_insufficient_data():
    df = make_price_df(closes=[150.0], volumes=[5000])

    summary = calculate_summary(df)

    assert summary["latest_price"] == 150.0
    assert summary["highest_price"] == 150.0
    assert summary["lowest_price"] == 150.0
    assert summary["average_close"] == 150.0
    assert summary["highest_volume"] == 5000
    assert summary["total_return_percent"] == 0.0
    assert summary["max_drawdown"] == 0.0

    # No valid daily returns on a 1-row DataFrame
    assert summary["average_daily_return"] is None
    assert summary["volatility"] is None
    assert summary["up_days"] is None
    assert summary["down_days"] is None
    assert summary["unchanged_days"] is None
    assert summary["up_day_percent"] is None
    assert summary["down_day_percent"] is None
    assert summary["unchanged_day_percent"] is None

    # Windowed indicators requiring 14 returns / 20 closes are None
    assert summary["latest_rsi"] is None
    assert summary["rsi_interpretation"] is None
    assert summary["latest_upper_band"] is None
    assert summary["latest_lower_band"] is None
    assert summary["latest_band_width"] is None
    assert summary["bollinger_interpretation"] is None
    assert summary["bandwidth_interpretation"] is None

    # EWM uses min_periods=0 by default, so EMA 12 == EMA 26 == 150.0 and MACD == 0.0
    assert summary["latest_macd"] == 0.0
    assert summary["latest_signal"] == 0.0
    assert summary["latest_histogram"] == 0.0
    assert summary["macd_interpretation"] == "Neutral - MACD equals signal"


def test_calculate_summary_rsi_all_gains_and_flat_series():
    # 15 closes produce 14 non-NaN Daily Return values (needed for rolling(14).mean())
    rising_closes = [100.0 + i * 2.0 for i in range(15)]
    rising_summary = calculate_summary(make_price_df(rising_closes))

    assert rising_summary["latest_rsi"] == 100.0
    assert rising_summary["rsi_interpretation"] == "Very strong upward momentum"

    # Flat prices -> all 14 daily returns are 0.0 -> average_gain == 0 and average_loss == 0 -> masked to 50
    flat_closes = [100.0] * 15
    flat_summary = calculate_summary(make_price_df(flat_closes))

    assert flat_summary["latest_rsi"] == 50.0
    assert flat_summary["rsi_interpretation"] == "Neutral momentum"


def test_calculate_summary_technical_indicators_on_25_days():
    # 25 days of alternating/trending prices so 20-day Bollinger Bands, 14-day RSI, and MACD are all populated
    closes = [
        100.0, 102.0, 101.0, 104.0, 103.0,
        106.0, 105.0, 108.0, 107.0, 110.0,
        109.0, 112.0, 111.0, 114.0, 113.0,
        116.0, 115.0, 118.0, 117.0, 120.0,
        119.0, 122.0, 121.0, 124.0, 125.0,
    ]
    df = make_price_df(closes)
    summary = calculate_summary(df)

    # Independently calculated Bollinger Bands on last 20 closes:
    # MA20 = 114.60, sample std = 6.0819 -> Upper = 126.76, Lower = 102.44, Band Width = 24.33
    assert summary["latest_upper_band"] == 126.76
    assert summary["latest_lower_band"] == 102.44
    assert summary["latest_band_width"] == 24.33
    assert summary["bollinger_interpretation"] == "Price within bands - within recent volatility range"
    assert summary["bandwidth_interpretation"] == "Bands contracting - volatility decreasing"

    # Independently calculated 14-day RSI (78.81) and MACD (12, 26, 9):
    # MACD = 5.24, Signal = 4.45, Histogram = 0.79
    assert summary["latest_rsi"] == 78.81
    assert summary["rsi_interpretation"] == "Very strong upward momentum"
    assert summary["latest_macd"] == 5.24
    assert summary["latest_signal"] == 4.45
    assert summary["latest_histogram"] == 0.79
    assert summary["macd_interpretation"] == "Bullish - MACD above signal"


def test_compare_summaries_differences_and_none_propagation():
    # Period A: 100 -> 105 -> 110 (+10.0% total return, avg daily return 4.88%, volatility 0.17%, 0.0% max drawdown)
    # Period B: 100 -> 120 -> 90 (-10.0% total return, avg daily return -2.5%, volatility 31.82%, -25.0% max drawdown)
    summary_a = calculate_summary(make_price_df([100.0, 105.0, 110.0]))
    summary_b = calculate_summary(make_price_df([100.0, 120.0, 90.0]))

    comparison = compare_summaries(summary_a, summary_b)

    # Difference is calculated as Period B - Period A
    assert comparison["total_return_difference"] == -20.0
    assert comparison["average_daily_return_difference"] == -7.38
    assert comparison["volatility_difference"] == 31.65
    assert comparison["max_drawdown_difference"] == -25.0

    # When one period has insufficient data (1 row -> average_daily_return and volatility are None),
    # those difference fields must return None while valid fields still compute.
    single_day_summary = calculate_summary(make_price_df([100.0]))
    partial_comparison = compare_summaries(single_day_summary, summary_b)

    assert partial_comparison["total_return_difference"] == -10.0
    assert partial_comparison["max_drawdown_difference"] == -25.0
    assert partial_comparison["average_daily_return_difference"] is None
    assert partial_comparison["volatility_difference"] is None


def test_compare_stock_summaries_rankings_and_missing_metric_fallback():
    # Stock A: steady gain (+10% total, low volatility, 0% drawdown)
    # Stock B: high gain with dip (+25% total, higher volatility, -10% drawdown)
    # Stock C: loss (-15% total, medium volatility, -20% drawdown)
    summaries = {
        "STOCK_A": {
            "total_return_percent": 10.0,
            "average_daily_return": 1.2,
            "volatility": 0.8,
            "max_drawdown": -2.0,
        },
        "STOCK_B": {
            "total_return_percent": 25.0,
            "average_daily_return": 3.1,
            "volatility": 4.5,
            "max_drawdown": -10.0,
        },
        "STOCK_C": {
            "total_return_percent": -15.0,
            "average_daily_return": -1.8,
            "volatility": 2.2,
            "max_drawdown": -20.0,
        },
    }

    result = compare_stock_summaries(summaries)

    assert result["highest_return"] == "STOCK_B"
    assert result["highest_average_daily_return"] == "STOCK_B"
    assert result["lowest_volatility"] == "STOCK_A"
    # best_drawdown uses max(), selecting the least negative drawdown (-2.0)
    assert result["best_drawdown"] == "STOCK_A"
    assert result["interpretation"] == (
        "STOCK_B had the highest return, while STOCK_A had the lowest volatility "
        "and STOCK_A had the best drawdown."
    )

    # If volatility is None for all stocks (e.g. 1-day range), lowest_volatility and interpretation are None
    no_vol_summaries = {
        "STOCK_A": {
            "total_return_percent": 0.0,
            "average_daily_return": None,
            "volatility": None,
            "max_drawdown": 0.0,
        },
        "STOCK_B": {
            "total_return_percent": 0.0,
            "average_daily_return": None,
            "volatility": None,
            "max_drawdown": 0.0,
        },
    }
    no_vol_result = compare_stock_summaries(no_vol_summaries)
    assert no_vol_result["highest_average_daily_return"] is None
    assert no_vol_result["lowest_volatility"] is None
    assert no_vol_result["interpretation"] is None

