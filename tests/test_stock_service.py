import logging
from datetime import date
import numpy as np
import pandas as pd
import pytest

import app.services.stock_service as stock_service
from app.services.data_validator import (
    CANONICAL_COLUMNS,
    validate_and_normalize_stock_data,
)
from app.services.market_data_repository import MarketDataRepository
from app.services.stock_service import (
    _get_repository,
    calculate_daily_returns,
    download_stock_data,
)


class FakeTicker:
    def __init__(self, df_to_return: pd.DataFrame, call_log: list[dict]):
        self._df = df_to_return
        self._call_log = call_log

    def history(self, **kwargs) -> pd.DataFrame:
        self._call_log.append(kwargs)
        return self._df.copy()


def test_download_stock_data_validates_and_roundtrips_canonical_csv(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    # Raw input has IST timezone timestamps, unsorted/duplicate dates, an invalid negative price row,
    # and an extra column that should be stripped by validation before saving to CSV.
    raw_df = pd.DataFrame(
        {
            "Open": [102.0, 100.0, -5.0, 104.0],
            "High": [106.0, 105.0, 105.0, 108.0],
            "Low": [101.0, 99.0, 99.0, 103.0],
            "Close": [104.0, 103.0, 101.0, 107.0],
            "Volume": [1500, 1000, 1200, 2000],
            "Extra": [1, 2, 3, 4],
        },
        index=pd.DatetimeIndex(
            [
                "2025-01-02 00:00:00+05:30",
                "2025-01-01 00:00:00+05:30",
                "2025-01-03 00:00:00+05:30",  # invalid Open=-5.0 -> dropped
                "2025-01-02 00:00:00+05:30",  # duplicate of 2025-01-02 -> kept last
            ],
            name="Date",
        ),
    )

    history_calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(raw_df, history_calls),
    )

    result = download_stock_data("RELIANCE.NS", period="1mo")

    assert history_calls == [{"period": "1mo"}]
    assert list(result.columns) == list(CANONICAL_COLUMNS)
    assert list(result.index) == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    ]
    assert result.loc[pd.Timestamp("2025-01-02"), "Close"] == 107.0
    assert result.loc[pd.Timestamp("2025-01-02"), "Volume"] == 2000

    # Verify CSV file creation and round-trip compatibility
    csv_path = tmp_path / "data" / "RELIANCE.NS.csv"
    assert csv_path.exists()

    raw_csv_df = pd.read_csv(csv_path)
    assert list(raw_csv_df.columns) == ["Date", *CANONICAL_COLUMNS]
    assert list(raw_csv_df["Date"]) == ["2025-01-01", "2025-01-02"]

    # Re-loading and normalizing the saved CSV reproduces the exact canonical DataFrame
    reloaded = validate_and_normalize_stock_data(raw_csv_df, symbol="RELIANCE.NS")
    pd.testing.assert_frame_equal(reloaded, result)


def test_download_stock_data_forwards_period_vs_start_end_arguments(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    clean_raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [102.0],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2025-01-01"], name="Date"),
    )

    history_calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(clean_raw, history_calls),
    )

    # 1. Predefined period path
    _ = download_stock_data("TCS.NS", period="6mo")
    assert history_calls[-1] == {"period": "6mo"}

    # 2. Custom start/end path (exclusive next calendar day sent to yfinance)
    _ = download_stock_data("TCS.NS", start="2025-01-01", end="2025-01-31")
    assert history_calls[-1] == {"start": "2025-01-01", "end": "2025-02-01"}


def test_download_stock_data_empty_or_all_invalid_skips_csv_and_s3(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", True)
    monkeypatch.setattr(stock_service, "AWS_BUCKET_NAME", "test-bucket")

    def boto_should_not_be_called(*args, **kwargs):
        pytest.fail("boto3.client should not be called when validated data is empty")

    monkeypatch.setattr(stock_service.boto3, "client", boto_should_not_be_called)

    # Case 1: yfinance returns empty DataFrame
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(pd.DataFrame(), []),
    )
    empty_res = download_stock_data("UNKNOWN.NS", period="1mo")
    assert empty_res.empty
    assert list(empty_res.columns) == list(CANONICAL_COLUMNS)
    assert empty_res.index.name == "Date"
    assert not (tmp_path / "data" / "UNKNOWN.NS.csv").exists()

    # Case 2: yfinance returns rows that are all invalid (rejected by validator)
    all_bad_raw = pd.DataFrame(
        {
            "Open": [-10.0, 0.0],
            "High": [5.0, 5.0],
            "Low": [10.0, 10.0],
            "Close": [-1.0, 0.0],
            "Volume": [-100, 100.5],
        },
        index=pd.DatetimeIndex(["2025-01-01", "2025-01-02"], name="Date"),
    )
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(all_bad_raw, []),
    )
    bad_res = download_stock_data("BAD.NS", period="1mo")
    assert bad_res.empty
    assert list(bad_res.columns) == list(CANONICAL_COLUMNS)
    assert not (tmp_path / "data" / "BAD.NS.csv").exists()


def test_download_stock_data_s3_optional_behavior_and_error_isolation(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.chdir(tmp_path)

    clean_raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [102.0],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2025-01-01"], name="Date"),
    )
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(clean_raw, []),
    )

    # 1. S3 disabled -> boto3.client never called
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)
    monkeypatch.setattr(
        stock_service.boto3,
        "client",
        lambda *a, **kw: pytest.fail("boto3.client should not be called when S3 disabled"),
    )
    res_disabled = download_stock_data("INFY.NS", period="1mo")
    assert len(res_disabled) == 1

    # 2. S3 enabled but AWS_BUCKET_NAME missing -> logs warning, skips boto3.client
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", True)
    monkeypatch.setattr(stock_service, "AWS_BUCKET_NAME", None)
    with caplog.at_level(logging.WARNING, logger="app.services.stock_service"):
        res_no_bucket = download_stock_data("INFY.NS", period="1mo")
    assert len(res_no_bucket) == 1
    assert any("AWS_BUCKET_NAME is not set" in r.message for r in caplog.records)

    # 3. S3 enabled with bucket, upload succeeds -> upload_file called with expected args
    caplog.clear()
    upload_calls = []

    class FakeS3Client:
        def upload_file(self, filename, bucket, key):
            upload_calls.append((filename, bucket, key))

    monkeypatch.setattr(stock_service, "AWS_BUCKET_NAME", "my-stock-bucket")
    monkeypatch.setattr(stock_service, "AWS_REGION", "ap-south-1")
    monkeypatch.setattr(
        stock_service.boto3,
        "client",
        lambda service, region_name=None: FakeS3Client(),
    )
    res_uploaded = download_stock_data("INFY.NS", period="1mo")
    assert len(res_uploaded) == 1
    assert upload_calls == [
        ("data/INFY.NS.csv", "my-stock-bucket", "data/INFY.NS.csv")
    ]

    # 4. S3 upload raises exception -> logs warning and still returns validated DataFrame
    class FailingS3Client:
        def upload_file(self, filename, bucket, key):
            raise RuntimeError("Simulated S3 upload failure")

    monkeypatch.setattr(
        stock_service.boto3,
        "client",
        lambda service, region_name=None: FailingS3Client(),
    )
    with caplog.at_level(logging.WARNING, logger="app.services.stock_service"):
        res_failed_s3 = download_stock_data("INFY.NS", period="1mo")
    assert len(res_failed_s3) == 1
    assert any("S3 upload failed" in r.message for r in caplog.records)


def test_calculate_daily_returns_on_canonical_dataframe():
    df = pd.DataFrame(
        {
            "Open": [100.0, 105.0, 110.0],
            "High": [105.0, 112.0, 115.0],
            "Low": [98.0, 104.0, 108.0],
            "Close": [100.0, 110.0, 99.0],
            "Volume": [1000, 1200, 1500],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-03"], name="Date"),
    )

    out = calculate_daily_returns(df.copy())

    assert "Daily Return" in out.columns
    assert np.isnan(out["Daily Return"].iloc[0])
    assert out["Daily Return"].iloc[1] == pytest.approx(10.0)
    assert out["Daily Return"].iloc[2] == pytest.approx(-10.0)


def test_download_stock_data_persists_validated_data_and_metadata_in_sqlite(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, -10.0, 105.0],
            "High": [105.0, 108.0, 105.0, 110.0],
            "Low": [98.0, 101.0, 95.0, 104.0],
            "Close": [103.0, 106.0, -5.0, 109.0],
            "Volume": [1000, 1500, 1200, 2500],
        },
        index=pd.DatetimeIndex(
            [
                "2025-01-01 00:00:00+05:30",
                "2025-01-02 00:00:00+05:30",
                "2025-01-03 00:00:00+05:30",  # invalid row -> dropped
                "2025-01-06 00:00:00+05:30",
            ],
            name="Date",
        ),
    )

    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(raw_df, []),
    )

    returned_df = download_stock_data("RELIANCE.NS", period="1mo")

    db_file = tmp_path / "data" / "market_data.db"
    assert db_file.exists()

    repo = MarketDataRepository(db_path=db_file)
    stored_df = repo.get_stock_data("RELIANCE.NS")
    pd.testing.assert_frame_equal(stored_df, returned_df)

    meta = repo.get_metadata("RELIANCE.NS")
    assert meta is not None
    assert meta.symbol == "RELIANCE.NS"
    assert meta.min_date == "2025-01-01"
    assert meta.max_date == "2025-01-06"
    assert meta.row_count == 3
    assert meta.has_full_history is False


def test_download_stock_data_accumulates_sqlite_history_across_partial_fetches(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    # First call: full history fetch (period="max") with 3 trading days
    history_batch = pd.DataFrame(
        {
            "Open": [100.0, 105.0, 110.0],
            "High": [106.0, 112.0, 115.0],
            "Low": [99.0, 104.0, 108.0],
            "Close": [105.0, 110.0, 112.0],
            "Volume": [1000, 1200, 1400],
        },
        index=pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-03"], name="Date"),
    )
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(history_batch, []),
    )
    _ = download_stock_data("INFY.NS", period="max")

    # Second call: 1-day fetch (e.g. POST /portfolio) updating 2025-01-03 and adding 2025-01-06
    short_batch = pd.DataFrame(
        {
            "Open": [110.0, 114.0],
            "High": [116.0, 120.0],
            "Low": [109.0, 113.0],
            "Close": [114.0, 118.0],
            "Volume": [1600, 2000],
        },
        index=pd.DatetimeIndex(["2025-01-03", "2025-01-06"], name="Date"),
    )
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(short_batch, []),
    )
    short_return = download_stock_data("INFY.NS", period="1d")
    assert len(short_return) == 2

    # SQLite retains all 4 historical dates without truncation and keeps has_full_history=True
    repo = _get_repository()
    stored_all = repo.get_stock_data("INFY.NS")
    assert len(stored_all) == 4
    assert list(stored_all.index) == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
    ]
    assert stored_all.loc[pd.Timestamp("2025-01-03"), "Close"] == 114.0
    assert stored_all.loc[pd.Timestamp("2025-01-06"), "Close"] == 118.0

    meta = repo.get_metadata("INFY.NS")
    assert meta is not None
    assert meta.min_date == "2025-01-01"
    assert meta.max_date == "2025-01-06"
    assert meta.row_count == 4
    assert meta.has_full_history is True


def test_download_stock_data_empty_or_invalid_does_not_persist_or_corrupt_sqlite(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    # Empty fetch for unknown symbol does not create a DB file or metadata
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(pd.DataFrame(), []),
    )
    _ = download_stock_data("EMPTY.NS", period="1mo")
    assert not (tmp_path / "data" / "market_data.db").exists()

    # Seed valid data for TCS.NS, then simulate an all-invalid fetch for TCS.NS
    valid_batch = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [102.0],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2025-01-01"], name="Date"),
    )
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(valid_batch, []),
    )
    _ = download_stock_data("TCS.NS", period="1mo")
    repo = _get_repository()
    meta_before = repo.get_metadata("TCS.NS")

    invalid_batch = pd.DataFrame(
        {
            "Open": [-10.0],
            "High": [5.0],
            "Low": [10.0],
            "Close": [-5.0],
            "Volume": [-100],
        },
        index=pd.DatetimeIndex(["2025-01-02"], name="Date"),
    )
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(invalid_batch, []),
    )
    res_invalid = download_stock_data("TCS.NS", period="1mo")
    assert res_invalid.empty

    # Existing SQLite data and metadata for TCS.NS remain intact
    assert len(repo.get_stock_data("TCS.NS")) == 1
    assert repo.get_metadata("TCS.NS") == meta_before


def test_get_repository_reuses_instance_for_same_db_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    repo1 = _get_repository()
    repo2 = _get_repository()
    assert repo1 is repo2


def test_explicit_date_range_cache_hit_skips_yfinance_and_filters_canonical_data(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    # Seed SQLite via initial fetch covering 2025-01-01 through 2025-01-07
    seed_df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 104.0, 106.0, 108.0],
            "High": [103.0, 105.0, 107.0, 109.0, 111.0],
            "Low": [99.0, 101.0, 103.0, 105.0, 107.0],
            "Close": [101.0, 103.0, 105.0, 107.0, 109.0],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=pd.DatetimeIndex(
            ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
            name="Date",
        ),
    )
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(seed_df, []),
    )
    _ = download_stock_data(
        "RELIANCE.NS", start="2025-01-01", end="2025-01-07"
    )

    # Now forbid any yfinance call on cache-covered explicit date ranges
    def yf_should_not_be_called(*args, **kwargs):
        pytest.fail("yfinance should not be called on a boundary-covered cache hit")

    monkeypatch.setattr(stock_service.yf, "Ticker", yf_should_not_be_called)

    # 1. Exact boundary hit using string dates
    exact_hit = download_stock_data(
        "RELIANCE.NS", start="2025-01-01", end="2025-01-07"
    )
    assert len(exact_hit) == 5

    # 2. Interior sub-range hit using date objects and unnormalized symbol casing
    sub_hit = download_stock_data(
        " reliance.ns ", start=date(2025, 1, 2), end=date(2025, 1, 6)
    )
    assert list(sub_hit.index) == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
    ]
    assert list(sub_hit["Close"]) == [103.0, 105.0, 107.0]

    # Verify canonical schema, index format, and dtypes on cached output
    assert list(sub_hit.columns) == list(CANONICAL_COLUMNS)
    assert isinstance(sub_hit.index, pd.DatetimeIndex)
    assert sub_hit.index.name == "Date"
    assert sub_hit.index.tz is None
    for col in ("Open", "High", "Low", "Close", "Dividends", "Stock Splits"):
        assert sub_hit[col].dtype == np.dtype("float64")
    assert sub_hit["Volume"].dtype == np.dtype("int64")


def test_explicit_date_range_cache_misses_fallback_to_yfinance(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    initial_df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 104.0],
            "High": [103.0, 105.0, 107.0],
            "Low": [99.0, 101.0, 103.0],
            "Close": [101.0, 103.0, 105.0],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.DatetimeIndex(
            ["2025-01-02", "2025-01-03", "2025-01-06"], name="Date"
        ),
    )
    history_calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(initial_df, history_calls),
    )

    # 1. Cold cache miss (database has no data for RELIANCE.NS) -> fetches from yfinance
    res1 = download_stock_data(
        "RELIANCE.NS", start="2025-01-02", end="2025-01-06"
    )
    assert len(res1) == 3
    assert len(history_calls) == 1

    # 2. Unknown symbol (TCS.NS not in database) -> fetches from yfinance
    res_unknown = download_stock_data(
        "TCS.NS", start="2025-01-02", end="2025-01-06"
    )
    assert len(res_unknown) == 3
    assert len(history_calls) == 2

    # 3. Request starts earlier than stored min_date (2025-01-01 < 2025-01-02) -> cache miss
    _ = download_stock_data(
        "RELIANCE.NS", start="2025-01-01", end="2025-01-06"
    )
    assert len(history_calls) == 3

    # 4. Request ends later than stored max_date (2025-01-10 > 2025-01-06) -> cache miss
    _ = download_stock_data(
        "RELIANCE.NS", start="2025-01-02", end="2025-01-10"
    )
    assert len(history_calls) == 4

    # 5. Period-based request on cached symbol still fetches from yfinance
    _ = download_stock_data("RELIANCE.NS", period="1mo")
    assert len(history_calls) == 5
    assert history_calls[-1] == {"period": "1mo"}


class CalendarAwareFakeTicker:
    """
    Fake yfinance Ticker that strictly models yfinance's exclusive `end` date:
    returns rows where `start <= Date < end`.
    """

    def __init__(
        self,
        full_df: pd.DataFrame,
        call_log: list[dict],
        extra_out_of_range_rows: pd.DataFrame | None = None,
    ):
        self._full_df = full_df
        self._call_log = call_log
        self._extra = extra_out_of_range_rows

    def history(self, **kwargs) -> pd.DataFrame:
        self._call_log.append(kwargs)
        if "start" in kwargs and "end" in kwargs:
            s = pd.Timestamp(kwargs["start"])
            e = pd.Timestamp(kwargs["end"])
            sliced = self._full_df.loc[
                (self._full_df.index >= s) & (self._full_df.index < e)
            ].copy()
            if self._extra is not None:
                sliced = pd.concat([sliced, self._extra.copy()])
            return sliced
        return self._full_df.copy()


def _make_calendar_dataset(dates: list[str], base_price: float = 100.0) -> pd.DataFrame:
    n = len(dates)
    closes = [base_price + i * 2.0 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [c - 1.0 for c in closes],
            "High": [c + 2.0 for c in closes],
            "Low": [c - 2.0 for c in closes],
            "Close": closes,
            "Volume": [1000 + i * 100 for i in range(n)],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_disjoint_cached_ranges_do_not_produce_false_cache_hit(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    cal_df = _make_calendar_dataset(
        [
            "2025-01-02",
            "2025-01-15",
            "2025-02-05",
            "2025-02-19",
            "2025-03-04",
            "2025-03-20",
        ]
    )
    calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, calls),
    )

    # 1. Seed January coverage
    jan_df = download_stock_data(
        "RELIANCE.NS", start="2025-01-01", end="2025-01-31"
    )
    assert len(jan_df) == 2
    assert len(calls) == 1

    # 2. Seed March coverage
    mar_df = download_stock_data(
        "RELIANCE.NS", start="2025-03-01", end="2025-03-31"
    )
    assert len(mar_df) == 2
    assert len(calls) == 2

    # 3. Request February (between Jan and Mar, but uncovered) -> MUST call yfinance
    feb_df = download_stock_data(
        "RELIANCE.NS", start="2025-02-01", end="2025-02-28"
    )
    assert len(calls) == 3
    assert calls[-1] == {"start": "2025-02-01", "end": "2025-03-01"}
    assert list(feb_df.index) == [
        pd.Timestamp("2025-02-05"),
        pd.Timestamp("2025-02-19"),
    ]


def test_inclusive_end_date_semantics_one_day_range_and_hit_miss_identity(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    cal_df = _make_calendar_dataset(
        ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    )
    # Stray row beyond requested end to prove post-validation inclusive filtering
    stray_row = _make_calendar_dataset(["2025-01-08"], base_price=999.0)
    calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(
            cal_df, calls, extra_out_of_range_rows=stray_row
        ),
    )

    # 1. One-day range (start == end == 2025-01-02) sends exclusive end 2025-01-03 and includes 2025-01-02
    one_day = download_stock_data(
        "TCS.NS", start="2025-01-02", end="2025-01-02"
    )
    assert calls[-1] == {"start": "2025-01-02", "end": "2025-01-03"}
    assert list(one_day.index) == [pd.Timestamp("2025-01-02")]

    # 2. Multi-day range: cache miss followed by cache hit returns identical DataFrame
    miss_df = download_stock_data(
        "INFY.NS", start="2025-01-01", end="2025-01-06"
    )
    assert calls[-1] == {"start": "2025-01-01", "end": "2025-01-07"}
    assert list(miss_df.index) == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
    ]

    # Forbid provider call on repeat request and verify exact DataFrame equality
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda *a, **kw: pytest.fail("yfinance should not be called on cache hit"),
    )
    hit_df = download_stock_data(
        "INFY.NS", start="2025-01-01", end="2025-01-06"
    )
    pd.testing.assert_frame_equal(hit_df, miss_df)


def test_valid_empty_cache_hit_on_weekend_subrange_inside_covered_interval(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    # Seed 2025-01-01 through 2025-01-07 with weekday bars only (Jan 4-5 is a weekend)
    cal_df = _make_calendar_dataset(
        ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    )
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, []),
    )
    _ = download_stock_data("RELIANCE.NS", start="2025-01-01", end="2025-01-07")

    # Requesting the interior weekend (2025-01-04..2025-01-05) must hit cache and return empty
    # canonical DataFrame without calling yfinance
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda *a, **kw: pytest.fail("yfinance must not be called on weekend sub-range cache hit"),
    )
    weekend_df = download_stock_data(
        "RELIANCE.NS", start="2025-01-04", end="2025-01-05"
    )
    assert weekend_df.empty
    assert list(weekend_df.columns) == list(CANONICAL_COLUMNS)
    assert isinstance(weekend_df.index, pd.DatetimeIndex)
    assert weekend_df.index.name == "Date"


def test_completed_day_utc_clamp_prevents_caching_today_or_future_dates(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)
    # Simulate current UTC date = 2025-01-06 -> last completed UTC date = 2025-01-05
    monkeypatch.setattr(
        stock_service, "_last_completed_date", lambda: "2025-01-05"
    )

    cal_df = _make_calendar_dataset(
        ["2025-01-02", "2025-01-03", "2025-01-06"]
    )
    calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, calls),
    )

    # 1. Request ending "today" (2025-01-06) or in the future (2025-01-10)
    _ = download_stock_data("TCS.NS", start="2025-01-01", end="2025-01-10")
    assert len(calls) == 1

    # Coverage was clamped to 2025-01-05:
    repo = _get_repository()
    assert repo.is_range_covered("TCS.NS", "2025-01-01", "2025-01-05") is True
    assert repo.is_range_covered("TCS.NS", "2025-01-01", "2025-01-06") is False

    # Repeat request for completed past range (2025-01-01..2025-01-05) hits cache
    _ = download_stock_data("TCS.NS", start="2025-01-01", end="2025-01-05")
    assert len(calls) == 1

    # Repeat request including today/future (2025-01-01..2025-01-06) misses cache and calls provider
    _ = download_stock_data("TCS.NS", start="2025-01-01", end="2025-01-06")
    assert len(calls) == 2

    # 2. Request starting today/in future (coverage_end < coverage_start) records NO coverage
    _ = download_stock_data("INFY.NS", start="2025-01-06", end="2025-01-07")
    assert len(calls) == 3
    assert repo.is_range_covered("INFY.NS", "2025-01-06", "2025-01-06") is False


def test_filtering_resulting_in_empty_dataframe_skips_all_persistence(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", True)
    monkeypatch.setattr(stock_service, "AWS_BUCKET_NAME", "test-bucket")
    monkeypatch.setattr(
        stock_service.boto3,
        "client",
        lambda *a, **kw: pytest.fail("S3 should not be called when filtered data is empty"),
    )

    # Provider returns a valid non-empty row, but its date (2025-01-15) is outside [2025-01-01, 2025-01-10]
    out_of_range_only = _make_calendar_dataset(["2025-01-15"])
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(out_of_range_only, []),
    )

    res = download_stock_data(
        "OUTOFRANGE.NS", start="2025-01-01", end="2025-01-10"
    )
    assert res.empty
    assert list(res.columns) == list(CANONICAL_COLUMNS)

    repo = _get_repository()
    assert repo.get_stock_data("OUTOFRANGE.NS").empty
    assert repo.get_metadata("OUTOFRANGE.NS") is None
    assert repo.is_range_covered("OUTOFRANGE.NS", "2025-01-01", "2025-01-10") is False
    assert not (tmp_path / "data" / "OUTOFRANGE.NS.csv").exists()


def test_empty_provider_response_and_provider_exception_do_not_corrupt_coverage(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    # 1. Empty provider response for explicit range writes no coverage, prices, or CSV
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FakeTicker(pd.DataFrame(), []),
    )
    empty_res = download_stock_data(
        "EMPTY.NS", start="2025-01-01", end="2025-01-10"
    )
    assert empty_res.empty
    repo = _get_repository()
    assert repo.is_range_covered("EMPTY.NS", "2025-01-01", "2025-01-10") is False
    assert repo.get_metadata("EMPTY.NS") is None
    assert not (tmp_path / "data" / "EMPTY.NS.csv").exists()

    # 2. Seed valid coverage for RELIANCE.NS, then simulate provider exception on uncovered range
    seed_df = _make_calendar_dataset(["2025-01-02", "2025-01-03"])
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(seed_df, []),
    )
    _ = download_stock_data("RELIANCE.NS", start="2025-01-01", end="2025-01-05")
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-01", "2025-01-05") is True

    class RaisingTicker:
        def history(self, **kwargs):
            raise RuntimeError("Simulated Yahoo Finance network error")

    monkeypatch.setattr(stock_service.yf, "Ticker", lambda symbol: RaisingTicker())

    with pytest.raises(RuntimeError, match="Simulated Yahoo Finance network error"):
        download_stock_data("RELIANCE.NS", start="2025-02-01", end="2025-02-10")

    # Existing Jan coverage remains intact and Feb is not marked covered
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-01", "2025-01-05") is True
    assert repo.is_range_covered("RELIANCE.NS", "2025-02-01", "2025-02-10") is False


def test_reversed_range_and_symbol_normalization_and_csv_path_safety(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    # 1. Reversed date range raises ValueError before calling provider or touching DB
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda *a, **kw: pytest.fail("yfinance should not be called on reversed dates"),
    )
    with pytest.raises(ValueError, match="start date must be before or equal to end date"):
        download_stock_data("RELIANCE.NS", start="2025-01-10", end="2025-01-01")
    assert not (tmp_path / "data" / "market_data.db").exists()

    # 1b. Partially specified date range (start without end, or end without start) raises ValueError
    with pytest.raises(ValueError, match="Both start and end dates must be provided together"):
        download_stock_data("RELIANCE.NS", start="2025-01-01")
    with pytest.raises(ValueError, match="Both start and end dates must be provided together"):
        download_stock_data("RELIANCE.NS", end="2025-01-10")
    assert not (tmp_path / "data" / "market_data.db").exists()
    assert not (tmp_path / "data" / "RELIANCE.NS.csv").exists()

    # 2. Unsafe symbols are rejected with ValueError before calling provider
    for unsafe_symbol in ["../x", "a/b", "a\\b", "FOO..BAR", "", "   ", "../data/EVIL"]:
        with pytest.raises(ValueError):
            download_stock_data(unsafe_symbol, start="2025-01-01", end="2025-01-05")

    # 3. Legitimate Yahoo Finance symbols with special characters normalize and stay inside data/
    sample_df = _make_calendar_dataset(["2025-01-02"])
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(sample_df, []),
    )
    valid_symbols = [" m&m.ns ", "^NSEI", "EURUSD=X", "BTC-USD", "BRK-B"]
    data_dir = (tmp_path / "data").resolve()

    for raw_sym in valid_symbols:
        norm = raw_sym.strip().upper()
        df = download_stock_data(raw_sym, start="2025-01-01", end="2025-01-05")
        assert len(df) == 1
        csv_file = (data_dir / f"{norm}.csv").resolve()
        assert csv_file.exists()
        assert csv_file.parent == data_dir


def test_end_to_end_api_compare_and_stock_disjoint_range_cache_integration(
    monkeypatch, tmp_path
):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    cal_df = _make_calendar_dataset(
        [
            "2025-01-06",
            "2025-01-08",
            "2025-01-10",
            "2025-01-20",
            "2025-02-05",
            "2025-02-12",
            "2025-04-07",
            "2025-04-15",
            "2025-04-25",
        ]
    )
    provider_calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, provider_calls),
    )

    api_client = TestClient(app)

    # Step 1: Populate January and April ranges via GET /compare
    res_compare = api_client.get(
        "/compare",
        params={
            "symbol": "RELIANCE.NS",
            "start_a": "2025-01-01",
            "end_a": "2025-01-31",
            "start_b": "2025-04-01",
            "end_b": "2025-04-30",
        },
    )
    assert res_compare.status_code == 200
    assert len(provider_calls) == 2

    # Step 2: Request a covered January sub-range ending on 2025-01-10 via GET /stock
    res_jan = api_client.get(
        "/stock",
        params={
            "symbol": "RELIANCE.NS",
            "start": "2025-01-06",
            "end": "2025-01-10",
        },
    )
    assert res_jan.status_code == 200
    # Provider is NOT called again (served from SQLite cache)
    assert len(provider_calls) == 2
    jan_history = res_jan.json()["history"]
    returned_jan_dates = [row["Date"][:10] for row in jan_history]
    assert returned_jan_dates == ["2025-01-06", "2025-01-08", "2025-01-10"]

    # Step 3: Request uncovered February range (between Jan and Apr) via GET /stock
    res_feb = api_client.get(
        "/stock",
        params={
            "symbol": "RELIANCE.NS",
            "start": "2025-02-01",
            "end": "2025-02-28",
        },
    )
    assert res_feb.status_code == 200
    # Provider IS called because February was not falsely treated as cached
    assert len(provider_calls) == 3
    feb_history = res_feb.json()["history"]
    returned_feb_dates = [row["Date"][:10] for row in feb_history]
    assert returned_feb_dates == ["2025-02-05", "2025-02-12"]


def test_selective_fetching_backfill_forward_fill_interior_and_multiple_gaps(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    cal_df = _make_calendar_dataset(
        [
            "2025-01-03",
            "2025-01-08",
            "2025-01-13",
            "2025-01-17",
            "2025-01-23",
            "2025-01-29",
        ]
    )
    calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, calls),
    )

    # 1. Seed middle range Jan 10-20
    mid_df = download_stock_data(
        "RELIANCE.NS", start="2025-01-10", end="2025-01-20"
    )
    assert list(mid_df.index) == [
        pd.Timestamp("2025-01-13"),
        pd.Timestamp("2025-01-17"),
    ]
    assert len(calls) == 1
    assert calls[-1] == {"start": "2025-01-10", "end": "2025-01-21"}

    # 2. Backfill (request Jan 1-20): fetches ONLY missing leading gap Jan 1-9 (exclusive end Jan 10)
    backfill_df = download_stock_data(
        "RELIANCE.NS", start="2025-01-01", end="2025-01-20"
    )
    assert len(calls) == 2
    assert calls[-1] == {"start": "2025-01-01", "end": "2025-01-10"}
    assert list(backfill_df.index) == [
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-08"),
        pd.Timestamp("2025-01-13"),
        pd.Timestamp("2025-01-17"),
    ]

    # 3. Forward fill (request Jan 1-31): fetches ONLY missing trailing gap Jan 21-31 (exclusive end Feb 1)
    full_jan_df = download_stock_data(
        "RELIANCE.NS", start="2025-01-01", end="2025-01-31"
    )
    assert len(calls) == 3
    assert calls[-1] == {"start": "2025-01-21", "end": "2025-02-01"}
    assert list(full_jan_df.index) == [
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-08"),
        pd.Timestamp("2025-01-13"),
        pd.Timestamp("2025-01-17"),
        pd.Timestamp("2025-01-23"),
        pd.Timestamp("2025-01-29"),
    ]

    # Verify CSV contains the full assembled range (all 6 rows), not just the trailing gap
    csv_df = pd.read_csv(tmp_path / "data" / "RELIANCE.NS.csv")
    assert len(csv_df) == 6

    # 4. Full cache hit on merged Jan 1-31 makes 0 provider calls
    _ = download_stock_data("RELIANCE.NS", start="2025-01-01", end="2025-01-31")
    assert len(calls) == 3

    # 5. Multiple gaps around interior coverage in one request (for TCS.NS, seed Jan 10-20, request Jan 1-31)
    _ = download_stock_data("TCS.NS", start="2025-01-10", end="2025-01-20")
    assert len(calls) == 4
    tcs_full = download_stock_data("TCS.NS", start="2025-01-01", end="2025-01-31")
    assert len(calls) == 6
    assert calls[-2] == {"start": "2025-01-01", "end": "2025-01-10"}
    assert calls[-1] == {"start": "2025-01-21", "end": "2025-02-01"}
    assert len(tcs_full) == 6


def test_weekend_empty_gap_between_cached_ranges_returns_cached_data_without_false_coverage_or_csv_write(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    # Weekday bars only; Jan 4 (Sat) and Jan 5 (Sun) have no trading rows
    cal_df = _make_calendar_dataset(
        [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ]
    )
    calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, calls),
    )

    # Seed Wed-Fri (Jan 1-3) and Mon-Wed (Jan 6-8) separately so Jan 4-5 is an uncovered weekend gap
    _ = download_stock_data("RELIANCE.NS", start="2025-01-01", end="2025-01-03")
    _ = download_stock_data("RELIANCE.NS", start="2025-01-06", end="2025-01-08")
    assert len(calls) == 2

    # Remove CSV and enable S3 with a failing guard: since the weekend gap returns empty,
    # no new rows are persisted, so neither CSV nor S3 should be triggered.
    csv_path = tmp_path / "data" / "RELIANCE.NS.csv"
    csv_path.unlink()
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", True)
    monkeypatch.setattr(stock_service, "AWS_BUCKET_NAME", "test-bucket")
    monkeypatch.setattr(
        stock_service.boto3,
        "client",
        lambda *a, **kw: pytest.fail("S3 should not be called when gap fetch persists no new data"),
    )

    combined = download_stock_data(
        "RELIANCE.NS", start="2025-01-01", end="2025-01-08"
    )
    # Provider was queried only for the missing weekend gap (Jan 4-5 -> exclusive end Jan 6)
    assert len(calls) == 3
    assert calls[-1] == {"start": "2025-01-04", "end": "2025-01-06"}

    # Full cached trading data (all 6 weekday bars) is returned, NOT an empty DataFrame
    assert len(combined) == 6
    assert not csv_path.exists()

    # Weekend gap was empty, so it must NOT be falsely marked covered
    repo = _get_repository()
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-04", "2025-01-05") is False


def test_partial_failure_across_multiple_gaps_persists_earlier_gap_and_retries_only_remaining(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    cal_df = _make_calendar_dataset(
        [
            "2025-01-10",
            "2025-01-20",
            "2025-02-10",
            "2025-02-20",
            "2025-03-10",
            "2025-03-20",
        ]
    )
    calls = []

    class FailOnMarchTicker(CalendarAwareFakeTicker):
        def history(self, **kwargs):
            if kwargs.get("start") == "2025-03-01":
                self._call_log.append(kwargs)
                raise RuntimeError("Simulated provider failure on March gap")
            return super().history(**kwargs)

    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: FailOnMarchTicker(cal_df, calls),
    )

    # 1. Seed February (2025-02-01..2025-02-28)
    _ = download_stock_data("INFY.NS", start="2025-02-01", end="2025-02-28")
    assert len(calls) == 1

    # 2. Request Jan 1 - Mar 31: Gap 1 (Jan) succeeds and commits; Gap 2 (Mar) raises RuntimeError
    with pytest.raises(RuntimeError, match="Simulated provider failure on March gap"):
        download_stock_data("INFY.NS", start="2025-01-01", end="2025-03-31")

    assert len(calls) == 3
    assert calls[1] == {"start": "2025-01-01", "end": "2025-02-01"}
    assert calls[2] == {"start": "2025-03-01", "end": "2025-04-01"}

    repo = _get_repository()
    # Earlier successful gap (Jan) is persisted and merged with Feb; failed gap (Mar) is not covered
    assert repo.is_range_covered("INFY.NS", "2025-01-01", "2025-02-28") is True
    assert repo.is_range_covered("INFY.NS", "2025-03-01", "2025-03-31") is False
    assert len(repo.get_stock_data("INFY.NS")) == 4

    # 3. Retry with healthy provider fetches ONLY the remaining March gap
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, calls),
    )
    retry_df = download_stock_data("INFY.NS", start="2025-01-01", end="2025-03-31")
    assert len(calls) == 4
    assert calls[-1] == {"start": "2025-03-01", "end": "2025-04-01"}
    assert len(retry_df) == 6
    assert repo.is_range_covered("INFY.NS", "2025-01-01", "2025-03-31") is True


def test_gap_consolidation_threshold_and_per_gap_utc_clamping(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    all_jan_days = [f"2025-01-{d:02d}" for d in range(1, 26)]
    cal_df = _make_calendar_dataset(all_jan_days)
    calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, calls),
    )

    # 1. Seed 3 small disjoint islands inside Jan 1-25, creating 4 uncovered gaps (> MAX_SELECTIVE_GAPS=3):
    # Covered: Jan 4-5, Jan 10-11, Jan 16-17
    # Missing in Jan 1-25: (Jan 1-3), (Jan 6-9), (Jan 12-15), (Jan 18-25) -> 4 gaps
    _ = download_stock_data("RELIANCE.NS", start="2025-01-04", end="2025-01-05")
    _ = download_stock_data("RELIANCE.NS", start="2025-01-10", end="2025-01-11")
    _ = download_stock_data("RELIANCE.NS", start="2025-01-16", end="2025-01-17")
    calls.clear()

    consolidated_df = download_stock_data(
        "RELIANCE.NS", start="2025-01-01", end="2025-01-25"
    )
    # Consolidated into ONE provider fetch spanning first gap start (Jan 1) through last gap end (Jan 25)
    assert len(calls) == 1
    assert calls[0] == {"start": "2025-01-01", "end": "2025-01-26"}
    assert len(consolidated_df) == 25
    repo = _get_repository()
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-01", "2025-01-25") is True

    # 2. Per-gap UTC completed-date clamping across two gaps
    # Simulate last completed UTC date = 2025-01-20
    monkeypatch.setattr(
        stock_service, "_last_completed_date", lambda: "2025-01-20"
    )
    _ = download_stock_data("TCS.NS", start="2025-01-10", end="2025-01-15")
    calls.clear()

    # Request Jan 1-25 -> Gap 1 (Jan 1-9) is before Jan 20 (fully covered),
    # Gap 2 (Jan 16-25) is clamped at Jan 20 (only completed rows Jan 16-20 are persisted and covered;
    # uncompleted rows Jan 21-25 returned by the provider are discarded before persistence)
    tcs_df = download_stock_data("TCS.NS", start="2025-01-01", end="2025-01-25")
    assert len(calls) == 2
    assert len(tcs_df) == 20
    assert tcs_df.index[-1] == pd.Timestamp("2025-01-20")

    # Verify SQLite does not contain any uncompleted rows beyond 2025-01-20
    tcs_stored = repo.get_stock_data("TCS.NS")
    assert len(tcs_stored) == 20
    assert tcs_stored.index[-1] == pd.Timestamp("2025-01-20")

    # Verify exported CSV also excludes uncompleted rows beyond 2025-01-20
    tcs_csv = pd.read_csv(tmp_path / "data" / "TCS.NS.csv")
    assert len(tcs_csv) == 20
    assert tcs_csv["Date"].iloc[-1] == "2025-01-20"

    assert repo.is_range_covered("TCS.NS", "2025-01-01", "2025-01-20") is True
    assert repo.is_range_covered("TCS.NS", "2025-01-01", "2025-01-21") is False


def test_end_to_end_api_selective_gap_fetch_integration(
    monkeypatch, tmp_path
):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stock_service, "S3_SYNC_ENABLED", False)

    cal_df = _make_calendar_dataset(
        [
            "2025-01-10",
            "2025-01-20",
            "2025-02-10",
            "2025-02-20",
            "2025-03-10",
            "2025-03-20",
        ]
    )
    provider_calls = []
    monkeypatch.setattr(
        stock_service.yf,
        "Ticker",
        lambda symbol: CalendarAwareFakeTicker(cal_df, provider_calls),
    )

    api_client = TestClient(app)

    # Step 1: GET /compare seeds January and March ranges (2 provider calls)
    res_compare = api_client.get(
        "/compare",
        params={
            "symbol": "RELIANCE.NS",
            "start_a": "2025-01-01",
            "end_a": "2025-01-31",
            "start_b": "2025-03-01",
            "end_b": "2025-03-31",
        },
    )
    assert res_compare.status_code == 200
    assert len(provider_calls) == 2

    # Step 2: GET /stock for full Q1 (Jan 1 - Mar 31) fetches ONLY the missing February gap
    res_q1 = api_client.get(
        "/stock",
        params={
            "symbol": "RELIANCE.NS",
            "start": "2025-01-01",
            "end": "2025-03-31",
        },
    )
    assert res_q1.status_code == 200
    assert len(provider_calls) == 3
    assert provider_calls[-1] == {"start": "2025-02-01", "end": "2025-03-01"}

    q1_dates = [row["Date"][:10] for row in res_q1.json()["history"]]
    assert q1_dates == [
        "2025-01-10",
        "2025-01-20",
        "2025-02-10",
        "2025-02-20",
        "2025-03-10",
        "2025-03-20",
    ]

    # Step 3: Repeat GET /stock for Q1 makes 0 additional provider calls
    res_q1_cached = api_client.get(
        "/stock",
        params={
            "symbol": "RELIANCE.NS",
            "start": "2025-01-01",
            "end": "2025-03-31",
        },
    )
    assert res_q1_cached.status_code == 200
    assert len(provider_calls) == 3




