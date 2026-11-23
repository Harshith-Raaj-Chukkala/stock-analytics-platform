import logging
import numpy as np
import pandas as pd
import pytest

from app.services.analytics_service import calculate_summary
from app.services.data_validator import (
    CANONICAL_COLUMNS,
    validate_and_normalize_stock_data,
)
from app.services.market_data_repository import MarketDataRepository
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


def test_per_column_price_and_volume_nan_inf_zero_and_negative_rejection():
    dates = pd.date_range("2025-01-01", periods=19, freq="B")
    base_rows = [
        # Row 0: Valid baseline with positive volume
        (100.0, 105.0, 95.0, 102.0, 1000),
        # Row 1: Valid baseline with zero volume
        (100.0, 105.0, 95.0, 102.0, 0),
        # Open invalid: NaN, +inf, -inf, 0.0, -1.0
        (np.nan, 105.0, 95.0, 102.0, 1000),
        (np.inf, 105.0, 95.0, 102.0, 1000),
        (-np.inf, 105.0, 95.0, 102.0, 1000),
        (0.0, 105.0, 95.0, 102.0, 1000),
        (-1.0, 105.0, 95.0, 102.0, 1000),
        # High invalid: NaN, +inf, -inf, 0.0, -1.0
        (100.0, np.nan, 95.0, 102.0, 1000),
        (100.0, np.inf, 95.0, 102.0, 1000),
        (100.0, -np.inf, 95.0, 102.0, 1000),
        (100.0, 0.0, 95.0, 102.0, 1000),
        (100.0, -1.0, 95.0, 102.0, 1000),
        # Low invalid: NaN, +inf, -inf, 0.0, -1.0
        (100.0, 105.0, np.nan, 102.0, 1000),
        (100.0, 105.0, np.inf, 102.0, 1000),
        (100.0, 105.0, -np.inf, 102.0, 1000),
        (100.0, 105.0, 0.0, 102.0, 1000),
        (100.0, 105.0, -1.0, 102.0, 1000),
        # Close invalid: NaN, +inf
        (100.0, 105.0, 95.0, np.nan, 1000),
        (100.0, 105.0, 95.0, np.inf, 1000),
    ]
    raw_part1 = make_ohlcv(
        dates=dates,
        opens=[r[0] for r in base_rows],
        highs=[r[1] for r in base_rows],
        lows=[r[2] for r in base_rows],
        closes=[r[3] for r in base_rows],
        volumes=[r[4] for r in base_rows],
    )
    res1 = validate_and_normalize_stock_data(raw_part1, symbol="PRICE_CHECKS.NS")
    assert_canonical_schema(res1)
    assert list(res1.index) == [dates[0], dates[1]]
    assert list(res1["Volume"]) == [1000, 0]

    dates2 = pd.date_range("2025-02-01", periods=8, freq="B")
    rows2 = [
        # Row 0: Valid
        (100.0, 105.0, 95.0, 102.0, 500),
        # Close invalid: -inf, 0.0, -1.0
        (100.0, 105.0, 95.0, -np.inf, 1000),
        (100.0, 105.0, 95.0, 0.0, 1000),
        (100.0, 105.0, 95.0, -1.0, 1000),
        # Volume invalid: NaN, +inf, -inf, -1
        (100.0, 105.0, 95.0, 102.0, np.nan),
        (100.0, 105.0, 95.0, 102.0, np.inf),
        (100.0, 105.0, 95.0, 102.0, -np.inf),
        (100.0, 105.0, 95.0, 102.0, -1),
    ]
    raw_part2 = make_ohlcv(
        dates=dates2,
        opens=[r[0] for r in rows2],
        highs=[r[1] for r in rows2],
        lows=[r[2] for r in rows2],
        closes=[r[3] for r in rows2],
        volumes=[r[4] for r in rows2],
    )
    res2 = validate_and_normalize_stock_data(raw_part2, symbol="VOL_CHECKS.NS")
    assert_canonical_schema(res2)
    assert list(res2.index) == [dates2[0]]


def test_all_five_ohlc_consistency_violations_explicitly_rejected():
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    raw = make_ohlcv(
        dates=dates,
        opens=[
            100.0,  # Row 0: Valid (Low=95 <= Open=100, Close=102 <= High=105)
            105.0,  # Row 1: High (104) < Open (105) -> rejected
            100.0,  # Row 2: High (103) < Close (104) -> rejected
            100.0,  # Row 3: High (98) < Low (99) -> rejected
            100.0,  # Row 4: Low (101) > Open (100) -> rejected
            105.0,  # Row 5: Low (101) > Close (100) -> rejected
        ],
        highs=[
            105.0,  # Row 0: Valid
            104.0,  # Row 1: High < Open (though High >= Low=95 and High >= Close=102)
            103.0,  # Row 2: High < Close (though High >= Open=100 and High >= Low=95)
            98.0,   # Row 3: High < Low
            105.0,  # Row 4: High valid, but Low > Open
            106.0,  # Row 5: High valid, but Low > Close
        ],
        lows=[
            95.0,   # Row 0: Valid
            95.0,   # Row 1
            95.0,   # Row 2
            99.0,   # Row 3: Low > High
            101.0,  # Row 4: Low > Open (though Low <= Close=103 and Low <= High=105)
            101.0,  # Row 5: Low > Close (though Low <= Open=105 and Low <= High=106)
        ],
        closes=[
            102.0,  # Row 0: Valid
            102.0,  # Row 1
            104.0,  # Row 2
            98.5,   # Row 3
            103.0,  # Row 4
            100.0,  # Row 5
        ],
        volumes=[1000] * 6,
    )

    result = validate_and_normalize_stock_data(raw, symbol="OHLC_5_CASES.NS")
    assert_canonical_schema(result)
    assert list(result.index) == [dates[0]]


def test_rejects_boolean_values_volume_overflow_numeric_date_strings_and_duplicate_columns(tmp_path):
    # 1. Exact duplicate and whitespace-induced duplicate column names raise ValueError
    dup_col_df = pd.DataFrame(
        [[100.0, 101.0, 105.0, 99.0, 102.0, 1000]],
        columns=["Open", "Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex(["2025-01-01"], name="Date"),
    )
    with pytest.raises(ValueError, match="Duplicate column names"):
        validate_and_normalize_stock_data(dup_col_df)

    ws_dup_col_df = pd.DataFrame(
        [[100.0, 101.0, 105.0, 99.0, 102.0, 1000]],
        columns=["Open", " Open ", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex(["2025-01-01"], name="Date"),
    )
    with pytest.raises(ValueError, match="Duplicate column names"):
        validate_and_normalize_stock_data(ws_dup_col_df)

    # 2. Pure bool dtype, NumPy bool_, and pandas nullable boolean Series are rejected
    idx_bool = pd.DatetimeIndex(["2025-01-01", "2025-01-02"], name="Date")
    pure_bool_df = pd.DataFrame(
        {
            "Open": pd.Series([True, False], index=idx_bool, dtype=bool),
            "High": [105.0, 105.0],
            "Low": [95.0, 95.0],
            "Close": [102.0, 102.0],
            "Volume": [1000, 1000],
        },
        index=idx_bool,
    )
    assert validate_and_normalize_stock_data(pure_bool_df).empty

    nullable_bool_df = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "High": [105.0, 105.0],
            "Low": [95.0, 95.0],
            "Close": [102.0, 102.0],
            "Volume": pd.Series([True, False], index=idx_bool, dtype="boolean"),
        },
        index=idx_bool,
    )
    assert validate_and_normalize_stock_data(nullable_bool_df).empty

    # 3. Pure numeric/boolean/whitespace date strings, mixed boolean OHLCV values (bool & np.bool_), and 1e20 overflow
    raw = make_ohlcv(
        dates=[
            "2025-01-01",
            "12345",          # pure numeric string date -> dropped
            "   ",            # whitespace date -> dropped
            True,             # boolean date -> dropped
            np.bool_(False),  # numpy bool_ date -> dropped
            "2025-01-02",     # boolean Open=True -> dropped
            "2025-01-03",     # numpy bool_ Volume=np.bool_(False) -> dropped
            "2025-01-06",     # boolean Dividends=True -> dropped
            "2025-01-07",     # Volume > int64 max (1e20) -> dropped without OverflowError
        ],
        opens=[100.0, 100.0, 100.0, 100.0, 100.0, True, 100.0, 100.0, 100.0],
        highs=[105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0],
        lows=[95.0, 95.0, 95.0, 95.0, 95.0, 1.0, 95.0, 95.0, 95.0],
        closes=[102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0],
        volumes=[1000, 1000, 1000, 1000, 1000, 1000, np.bool_(False), 1000, 1e20],
        dividends=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, 0.0],
        as_column=True,
    )

    result = validate_and_normalize_stock_data(raw, symbol="EDGE_TYPES.NS")
    assert_canonical_schema(result)
    assert list(result.index) == [pd.Timestamp("2025-01-01")]

    # 4. Exact int64 Volume boundary verification (2**63 overflow vs 2**63 - 1 and nextafter(2**63, 0.0))
    repo = MarketDataRepository(db_path=tmp_path / "boundary.db")
    idx_single = pd.DatetimeIndex(["2025-01-01"], name="Date")

    # Rejected boundary cases: float(2**63), uint64(2**63), uint64(2**63 + 512), string "9223372036854775808"
    rejected_volumes = [
        pd.Series([float(2**63)], index=idx_single, dtype="float64"),
        pd.Series(np.array([2**63], dtype=np.uint64), index=idx_single),
        pd.Series(np.array([2**63 + 512], dtype=np.uint64), index=idx_single),
        pd.Series(["9223372036854775808"], index=idx_single, dtype=object),
    ]
    for bad_vol_series in rejected_volumes:
        bad_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [95.0],
                "Close": [102.0],
                "Volume": bad_vol_series,
            },
            index=idx_single,
        )
        validated_bad = validate_and_normalize_stock_data(bad_df, symbol="OVERFLOW.NS")
        assert validated_bad.empty
        assert_canonical_schema(validated_bad)
        assert repo.upsert_stock_data("OVERFLOW.NS", validated_bad) == 0

    # Accepted boundary case A: np.int64(2**63 - 1) exact signed 64-bit maximum
    int64_max_val = np.int64(2**63 - 1)
    df_int64_max = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": pd.Series([int64_max_val], index=idx_single, dtype="int64"),
        },
        index=idx_single,
    )
    res_int64_max = validate_and_normalize_stock_data(df_int64_max, symbol="MAX_I64.NS")
    assert_canonical_schema(res_int64_max)
    assert len(res_int64_max) == 1
    assert res_int64_max["Volume"].dtype == np.dtype("int64")
    assert res_int64_max["Volume"].iloc[0] == int64_max_val
    assert res_int64_max["Volume"].iloc[0] > 0

    # Accepted boundary case B: float(np.nextafter(float(2**63), 0.0)) largest representable float64 < 2**63
    max_safe_float = float(np.nextafter(float(2**63), 0.0))
    df_float_max = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": pd.Series([max_safe_float], index=idx_single, dtype="float64"),
        },
        index=idx_single,
    )
    res_float_max = validate_and_normalize_stock_data(df_float_max, symbol="MAX_F64.NS")
    assert_canonical_schema(res_float_max)
    assert len(res_float_max) == 1
    assert res_float_max["Volume"].dtype == np.dtype("int64")
    assert res_float_max["Volume"].iloc[0] == int(max_safe_float)
    assert res_float_max["Volume"].iloc[0] > 0
    assert repo.upsert_stock_data("MAX_F64.NS", res_float_max) == 1


def test_invalid_records_cannot_enter_sqlite_persistence_via_ingestion_flow(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")

    # Mix of 1 valid row (zero volume) and multiple invalid rows (NaN, inf, zero/negative price,
    # negative volume, High < Open, High < Close, Low > Open, Low > Close, High < Low)
    raw = make_ohlcv(
        dates=pd.date_range("2025-01-01", periods=10, freq="B"),
        opens=[100.0, np.nan, 0.0, -5.0, 100.0, 105.0, 100.0, 100.0, 105.0, 100.0],
        highs=[105.0, 105.0, 105.0, 105.0, 105.0, 104.0, 103.0, 98.0, 106.0, np.inf],
        lows=[95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 99.0, 101.0, 95.0],
        closes=[102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 104.0, 98.5, 100.0, 102.0],
        volumes=[0, 1000, 1000, 1000, -50, 1000, 1000, 1000, 1000, 1000],
    )

    validated = validate_and_normalize_stock_data(raw, symbol="PERSIST_GUARD.NS")
    assert len(validated) == 1

    persisted_count = repo.upsert_stock_data("PERSIST_GUARD.NS", validated)
    assert persisted_count == 1

    stored = repo.get_stock_data("PERSIST_GUARD.NS")
    assert len(stored) == 1
    assert stored.index[0] == pd.Timestamp("2025-01-01")
    assert stored.iloc[0]["Volume"] == 0


