import logging
import numpy as np
import pandas as pd
import pytest

from app.services.analytics_service import calculate_summary
from app.services.data_validator import (
    CANONICAL_COLUMNS,
    validate_and_normalize_stock_data,
)
from app.services.stock_service import calculate_daily_returns


def make_ohlcv(
    dates,
    opens,
    highs,
    lows,
    closes,
    volumes,
    dividends=None,
    splits=None,
    as_column=False,
    extra_cols=None,
) -> pd.DataFrame:
    data = {
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }
    if dividends is not None:
        data["Dividends"] = dividends
    if splits is not None:
        data["Stock Splits"] = splits
    if extra_cols:
        data.update(extra_cols)

    if as_column:
        data["Date"] = dates
        return pd.DataFrame(data)

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


def assert_canonical_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None
    assert df.index.name == "Date"
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    for col in ("Open", "High", "Low", "Close", "Dividends", "Stock Splits"):
        assert df[col].dtype == np.dtype("float64")
    assert df["Volume"].dtype == np.dtype("int64")


def test_valid_ohlcv_normalization_column_order_dtypes_and_optional_defaults(caplog):
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    raw = make_ohlcv(
        dates=dates,
        opens=[100, 102, 104],
        highs=[105, 106, 108],
        lows=[99, 101, 103],
        closes=[103, 104, 107],
        volumes=[1000, 1500, 2000],
        extra_cols={"Capital Gains": [0.0, 0.0, 0.0]},
    )

    with caplog.at_level(logging.WARNING, logger="app.services.data_validator"):
        result = validate_and_normalize_stock_data(raw, symbol="RELIANCE.NS")

    assert_canonical_schema(result)
    assert len(result) == 3
    assert (result["Dividends"] == 0.0).all()
    assert (result["Stock Splits"] == 0.0).all()
    assert "Capital Gains" not in result.columns
    assert caplog.records == []


def test_date_provided_as_column_and_timezone_aware_ist_and_us_dates_preserved():
    # IST (UTC+05:30) and US Eastern (UTC-05:00) midnight timestamps must preserve their local calendar dates
    ist_dates = pd.DatetimeIndex(
        ["2025-01-02 00:00:00+05:30", "2025-01-03 00:00:00+05:30"]
    )
    ist_raw = make_ohlcv(
        dates=ist_dates,
        opens=[100.0, 101.0],
        highs=[105.0, 106.0],
        lows=[99.0, 100.0],
        closes=[102.0, 104.0],
        volumes=[1000, 1100],
    )
    ist_result = validate_and_normalize_stock_data(ist_raw, symbol="TCS.NS")
    assert_canonical_schema(ist_result)
    assert list(ist_result.index) == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
    ]

    # String timestamps with mixed timezone offsets provided as a regular Date column
    us_raw = make_ohlcv(
        dates=["2025-01-02 00:00:00-05:00", "2025-03-10 00:00:00-04:00"],
        opens=[200.0, 205.0],
        highs=[210.0, 212.0],
        lows=[198.0, 202.0],
        closes=[205.0, 208.0],
        volumes=[5000, 6000],
        as_column=True,
    )
    us_result = validate_and_normalize_stock_data(us_raw, symbol="AAPL")
    assert_canonical_schema(us_result)
    assert list(us_result.index) == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-03-10"),
    ]


def test_unsorted_rows_and_duplicate_dates_keep_last_occurrence(caplog):
    raw = make_ohlcv(
        dates=pd.DatetimeIndex(
            ["2025-01-03", "2025-01-01", "2025-01-02", "2025-01-03"]
        ),
        opens=[100.0, 90.0, 95.0, 102.0],
        highs=[105.0, 95.0, 98.0, 108.0],
        lows=[99.0, 89.0, 94.0, 101.0],
        closes=[101.0, 92.0, 96.0, 107.0],
        volumes=[1000, 2000, 3000, 4000],
    )

    with caplog.at_level(logging.WARNING, logger="app.services.data_validator"):
        result = validate_and_normalize_stock_data(raw, symbol="INFY.NS")

    assert_canonical_schema(result)
    assert list(result.index) == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
    ]
    # Last occurrence for 2025-01-03 (Close=107.0, Volume=4000) is kept
    assert result.loc[pd.Timestamp("2025-01-03"), "Close"] == 107.0
    assert result.loc[pd.Timestamp("2025-01-03"), "Volume"] == 4000
    assert any("duplicate" in rec.message.lower() and "INFY.NS" in rec.message for rec in caplog.records)


def test_drops_nan_inf_non_positive_prices_negative_and_fractional_volume_but_keeps_zero_volume(caplog):
    dates = pd.date_range("2025-01-01", periods=9, freq="B")
    raw = make_ohlcv(
        dates=dates,
        opens=[100.0, np.nan, np.inf, 0.0, -5.0, 100.0, 100.0, 100.0, 100.0],
        highs=[105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0],
        lows=[99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        closes=[102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0],
        volumes=[1000, 1000, 1000, 1000, 1000, -10, 100.5, 0, 2000],
        dividends=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, -1.0],
        splits=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
    )
    # Row 0: valid (Volume=1000)
    # Row 1: NaN Open -> dropped
    # Row 2: Inf Open -> dropped
    # Row 3: Zero Open -> dropped
    # Row 4: Negative Open -> dropped
    # Row 5: Negative Volume -> dropped
    # Row 6: Fractional Volume (100.5) -> dropped without truncation
    # Row 7: Zero Volume & NaN optional Dividends/Splits -> valid (optional NaNs default to 0.0)
    # Row 8: Negative Dividends (-1.0) -> dropped

    with caplog.at_level(logging.WARNING, logger="app.services.data_validator"):
        result = validate_and_normalize_stock_data(raw, symbol="RELIANCE.NS")

    assert_canonical_schema(result)
    assert list(result.index) == [dates[0], dates[7]]
    assert list(result["Volume"]) == [1000, 0]
    assert result.loc[dates[7], "Dividends"] == 0.0
    assert result.loc[dates[7], "Stock Splits"] == 0.0
    assert len(caplog.records) == 1
    assert "RELIANCE.NS" in caplog.records[0].message


def test_ohlc_envelope_validation_rejects_bad_rows_and_minimally_corrects_within_tolerance(caplog):
    dates = pd.date_range("2025-01-01", periods=7, freq="B")
    raw = make_ohlcv(
        dates=dates,
        opens=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        highs=[
            105.0,               # Row 0: clean valid
            95.0,                # Row 1: High < Low (materially invalid -> dropped)
            100.0 - 2e-6,        # Row 2: High < Open by 2e-6 (just beyond 1e-6 -> dropped)
            105.0,               # Row 3: Low > Open by 1.0 (materially invalid -> dropped)
            102.0 - 5e-7,        # Row 4: High < Close (102.0) by 5e-7 (within 1e-6 -> corrected)
            105.0,               # Row 5: Low > Open (100.0) by 5e-7 (within 1e-6 -> corrected)
            100.0 - 4e-7,        # Row 6: Both High < 100 and Low > 100 by 4e-7 (within 1e-6 -> both minimally clamped to 100.0)
        ],
        lows=[
            98.0,                # Row 0: clean valid
            98.0,                # Row 1
            98.0,                # Row 2
            101.0,               # Row 3: Low (101) > Open (100)
            99.0,                # Row 4: valid Low
            100.0 + 5e-7,        # Row 5: Low slightly above Open (within 1e-6 -> corrected)
            100.0 + 4e-7,        # Row 6: Low slightly above Open/Close (within 1e-6 -> corrected)
        ],
        closes=[102.0, 99.0, 99.0, 100.5, 102.0, 102.0, 100.0],
        volumes=[1000, 1000, 1000, 1000, 1000, 1000, 1000],
    )

    with caplog.at_level(logging.WARNING, logger="app.services.data_validator"):
        result = validate_and_normalize_stock_data(raw, symbol="TCS.NS")

    assert_canonical_schema(result)
    assert list(result.index) == [dates[0], dates[4], dates[5], dates[6]]
    # Row 0 untouched
    assert result.loc[dates[0], "High"] == 105.0
    assert result.loc[dates[0], "Low"] == 98.0
    # Row 4 High minimally corrected to 102.0
    assert result.loc[dates[4], "High"] == 102.0
    # Row 5 Low minimally corrected to 100.0
    assert result.loc[dates[5], "Low"] == 100.0
    # Row 6 both High and Low minimally clamped to 100.0 (no overshoot)
    assert result.loc[dates[6], "High"] == 100.0
    assert result.loc[dates[6], "Low"] == 100.0
    assert len(caplog.records) == 2


def test_empty_and_all_invalid_dataframes_return_empty_canonical_df(caplog):
    # Completely empty DataFrame -> no warning
    with caplog.at_level(logging.WARNING, logger="app.services.data_validator"):
        empty_res = validate_and_normalize_stock_data(pd.DataFrame(), symbol="EMPTY.NS")

    assert empty_res.empty
    assert_canonical_schema(empty_res)
    assert caplog.records == []

    # All rows invalid -> returns empty canonical DataFrame and logs warning
    bad_raw = make_ohlcv(
        dates=pd.DatetimeIndex(["2025-01-01", "2025-01-02"]),
        opens=[-10.0, 0.0],
        highs=[10.0, 10.0],
        lows=[5.0, 5.0],
        closes=[-5.0, 0.0],
        volumes=[100, 100],
    )
    with caplog.at_level(logging.WARNING, logger="app.services.data_validator"):
        all_bad_res = validate_and_normalize_stock_data(bad_raw, symbol="BAD.NS")

    assert all_bad_res.empty
    assert_canonical_schema(all_bad_res)
    assert len(caplog.records) == 1


def test_invalid_structural_inputs_raise_value_error():
    # None and non-DataFrame inputs
    with pytest.raises(ValueError, match="DataFrame"):
        validate_and_normalize_stock_data(None)
    with pytest.raises(ValueError, match="DataFrame"):
        validate_and_normalize_stock_data({"Close": [100.0]})

    # MultiIndex columns
    multi_cols = pd.MultiIndex.from_tuples([("Close", "TCS.NS"), ("Volume", "TCS.NS")])
    multi_df = pd.DataFrame([[100.0, 1000]], columns=multi_cols)
    with pytest.raises(ValueError, match="MultiIndex"):
        validate_and_normalize_stock_data(multi_df)

    # Non-empty DataFrame missing required OHLCV columns
    missing_col_df = pd.DataFrame(
        {"Open": [100.0], "High": [105.0], "Low": [99.0], "Close": [102.0]},
        index=pd.date_range("2025-01-01", periods=1),
    )
    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        validate_and_normalize_stock_data(missing_col_df)


def test_input_dataframe_is_not_mutated():
    raw = make_ohlcv(
        dates=pd.DatetimeIndex(["2025-01-03", "2025-01-01"]),
        opens=[100.0, 90.0],
        highs=[105.0, 95.0],
        lows=[99.0, 89.0],
        closes=[102.0, 92.0],
        volumes=[1000, 2000],
        extra_cols={"Extra": [1, 2]},
    )
    snapshot = raw.copy(deep=True)

    _ = validate_and_normalize_stock_data(raw, symbol="IMMUTABLE.NS")

    pd.testing.assert_frame_equal(raw, snapshot)


def test_downstream_compatibility_with_analytics_service():
    dates = pd.DatetimeIndex(
        [
            "2025-01-01 00:00:00+05:30",
            "2025-01-02 00:00:00+05:30",
            "2025-01-03 00:00:00+05:30",
            "2025-01-06 00:00:00+05:30",
            "2025-01-07 00:00:00+05:30",
        ]
    )
    raw = make_ohlcv(
        dates=dates,
        opens=[100.0, 120.0, 90.0, 90.0, 110.0],
        highs=[105.0, 125.0, 95.0, 95.0, 115.0],
        lows=[95.0, 115.0, 85.0, 85.0, 105.0],
        closes=[100.0, 120.0, 90.0, 90.0, 110.0],
        volumes=[1000, 2500, 1500, 1200, 1800],
    )

    clean = validate_and_normalize_stock_data(raw, symbol="RELIANCE.NS")
    with_returns = calculate_daily_returns(clean)
    summary = calculate_summary(with_returns)

    assert summary["latest_price"] == 110.0
    assert summary["total_return_percent"] == 10.0
    assert summary["max_drawdown"] == -25.0
    assert summary["highest_volume"] == 2500

    # Ensure JSON-ready history serialization matches main.py contract
    records = with_returns.replace({np.nan: None}).reset_index().to_dict(orient="records")
    assert len(records) == 5
    assert "Date" in records[0]
    assert records[0]["Date"] == pd.Timestamp("2025-01-01")


def test_drops_malformed_dates_and_unparseable_numeric_strings(caplog):
    raw = make_ohlcv(
        dates=["2025-01-01", "not-a-date", None, "2025-01-02", "2025-01-03"],
        opens=[100.0, 100.0, 100.0, "bad-num", 105.0],
        highs=[105.0, 105.0, 105.0, 105.0, 110.0],
        lows=[99.0, 99.0, 99.0, 99.0, 104.0],
        closes=[102.0, 102.0, 102.0, 102.0, 108.0],
        volumes=[1000, 1000, 1000, 1000, 1500],
        dividends=[0.0, 0.0, 0.0, 0.0, "bad-div"],
        as_column=True,
    )

    with caplog.at_level(logging.WARNING, logger="app.services.data_validator"):
        result = validate_and_normalize_stock_data(raw, symbol="MALFORMED.NS")

    assert_canonical_schema(result)
    assert list(result.index) == [pd.Timestamp("2025-01-01")]
    assert len(caplog.records) == 2

