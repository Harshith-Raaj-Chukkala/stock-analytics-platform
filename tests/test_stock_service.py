import logging
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

    # 2. Custom start/end path
    _ = download_stock_data("TCS.NS", start="2025-01-01", end="2025-01-31")
    assert history_calls[-1] == {"start": "2025-01-01", "end": "2025-01-31"}


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

