from datetime import date, datetime, timezone
import sqlite3
import numpy as np
import pandas as pd
import pytest

from app.services.analytics_service import calculate_summary
from app.services.data_validator import (
    CANONICAL_COLUMNS,
    validate_and_normalize_stock_data,
)
from app.services.market_data_repository import (
    MarketDataRepository,
    SymbolMetadata,
)
from app.services.stock_service import calculate_daily_returns


def make_canonical_df(
    dates: list[str],
    closes: list[float],
    volumes: list[int] | None = None,
    dividends: list[float] | None = None,
    splits: list[float] | None = None,
) -> pd.DataFrame:
    n = len(dates)
    if volumes is None:
        volumes = [1000 * (i + 1) for i in range(n)]
    if dividends is None:
        dividends = [0.0] * n
    if splits is None:
        splits = [0.0] * n

    raw = pd.DataFrame(
        {
            "Open": [c - 1.0 for c in closes],
            "High": [c + 2.0 for c in closes],
            "Low": [c - 2.0 for c in closes],
            "Close": [float(c) for c in closes],
            "Volume": volumes,
            "Dividends": dividends,
            "Stock Splits": splits,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return validate_and_normalize_stock_data(raw)


def assert_canonical_contract(df: pd.DataFrame) -> None:
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "Date"
    assert df.index.tz is None
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    for col in ("Open", "High", "Low", "Close", "Dividends", "Stock Splits"):
        assert df[col].dtype == np.dtype("float64")
    assert df["Volume"].dtype == np.dtype("int64")


def test_schema_and_parent_directory_auto_creation(tmp_path):
    nested_db_path = tmp_path / "nested" / "storage" / "market_data.db"
    assert not nested_db_path.parent.exists()

    repo = MarketDataRepository(db_path=nested_db_path)
    assert nested_db_path.exists()

    with sqlite3.connect(repo.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        assert {"daily_prices", "ingestion_metadata", "cached_ranges"}.issubset(tables)

        price_cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(daily_prices);").fetchall()
        ]
        assert price_cols == [
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dividends",
            "stock_splits",
        ]

        meta_cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(ingestion_metadata);").fetchall()
        ]
        assert meta_cols == [
            "symbol",
            "min_date",
            "max_date",
            "last_synced_at",
            "has_full_history",
            "row_count",
            "last_refreshed_at",
        ]

        # Confirm redundant explicit index is not created (composite PK provides the autoindex)
        explicit_indexes = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='daily_prices' AND sql IS NOT NULL;"
            ).fetchall()
        ]
        assert explicit_indexes == []


def test_canonical_dataframe_roundtrip_preserves_dtypes_and_date_index(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    df = make_canonical_df(
        dates=["2025-01-02", "2025-01-03", "2025-01-06"],
        closes=[100.5, 102.75, 101.25],
        volumes=[10500, 20300, 15400],
        dividends=[0.0, 5.5, 0.0],
        splits=[0.0, 0.0, 2.0],
    )

    inserted = repo.upsert_stock_data("  reliance.ns ", df)
    assert inserted == 3

    loaded = repo.get_stock_data("RELIANCE.NS")
    assert_canonical_contract(loaded)
    pd.testing.assert_frame_equal(loaded, df)


def test_idempotent_upsert_and_overlapping_data_updates(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    initial_df = make_canonical_df(
        dates=["2025-01-01", "2025-01-02", "2025-01-03"],
        closes=[100.0, 105.0, 110.0],
        volumes=[1000, 2000, 3000],
    )

    repo.upsert_stock_data("TCS.NS", initial_df)
    # Re-upserting the exact same DataFrame is idempotent
    repo.upsert_stock_data("TCS.NS", initial_df)
    assert len(repo.get_stock_data("TCS.NS")) == 3

    # Overlapping batch updates 2025-01-02 and 2025-01-03 and adds 2025-01-06
    overlap_df = make_canonical_df(
        dates=["2025-01-02", "2025-01-03", "2025-01-06"],
        closes=[106.0, 112.0, 115.0],
        volumes=[2500, 3500, 4000],
    )
    repo.upsert_stock_data("TCS.NS", overlap_df)

    result = repo.get_stock_data("TCS.NS")
    assert_canonical_contract(result)
    assert list(result.index) == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
    ]
    assert list(result["Close"]) == [100.0, 106.0, 112.0, 115.0]
    assert list(result["Volume"]) == [1000, 2500, 3500, 4000]


def test_partial_update_does_not_truncate_existing_history(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    ten_days = [f"2025-01-{day:02d}" for day in range(1, 11)]
    full_df = make_canonical_df(
        dates=ten_days,
        closes=[100.0 + i for i in range(10)],
    )
    repo.upsert_stock_data("INFY.NS", full_df)

    # Simulate a 1-day portfolio fetch upserting only the latest day (or a middle day)
    single_day_df = make_canonical_df(
        dates=["2025-01-10"],
        closes=[199.0],
    )
    repo.upsert_stock_data("INFY.NS", single_day_df)

    stored = repo.get_stock_data("INFY.NS")
    assert len(stored) == 10
    assert stored.index[0] == pd.Timestamp("2025-01-01")
    assert stored.index[-1] == pd.Timestamp("2025-01-10")
    assert stored.loc[pd.Timestamp("2025-01-01"), "Close"] == 100.0
    assert stored.loc[pd.Timestamp("2025-01-10"), "Close"] == 199.0

    # Empty DataFrame upsert must also preserve all 10 existing rows
    repo.upsert_stock_data("INFY.NS", pd.DataFrame())
    assert len(repo.get_stock_data("INFY.NS")) == 10


def test_inclusive_date_range_queries_and_symbol_isolation(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    tcs_df = make_canonical_df(
        dates=["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        closes=[100.0, 101.0, 102.0, 103.0, 104.0],
    )
    infy_df = make_canonical_df(
        dates=["2025-01-01", "2025-01-02"],
        closes=[500.0, 510.0],
    )
    repo.upsert_stock_data("TCS.NS", tcs_df)
    repo.upsert_stock_data("INFY.NS", infy_df)

    # Inclusive [start, end] using date objects
    sliced = repo.get_stock_data(
        "TCS.NS", start=date(2025, 1, 2), end=date(2025, 1, 6)
    )
    assert_canonical_contract(sliced)
    assert list(sliced.index) == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
    ]
    assert list(sliced["Close"]) == [101.0, 102.0, 103.0]

    # Open-ended start or end using ISO strings
    from_jan3 = repo.get_stock_data("TCS.NS", start="2025-01-03")
    assert list(from_jan3.index) == [
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2025-01-07"),
    ]
    until_jan2 = repo.get_stock_data("TCS.NS", end="2025-01-02")
    assert list(until_jan2.index) == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    ]

    # Invalid start > end raises ValueError
    with pytest.raises(ValueError, match="start date must be before or equal to end date"):
        repo.get_stock_data("TCS.NS", start="2025-01-07", end="2025-01-02")

    # Symbol isolation: INFY.NS only has its 2 rows
    infy_stored = repo.get_stock_data("INFY.NS")
    assert len(infy_stored) == 2
    assert list(infy_stored["Close"]) == [500.0, 510.0]


def test_empty_query_results_return_canonical_empty_dataframe(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")

    # Unknown symbol
    unknown_df = repo.get_stock_data("UNKNOWN.NS")
    assert unknown_df.empty
    assert_canonical_contract(unknown_df)

    # Known symbol but non-matching date range
    repo.upsert_stock_data(
        "TCS.NS",
        make_canonical_df(dates=["2025-01-01", "2025-01-02"], closes=[100.0, 105.0]),
    )
    out_of_range = repo.get_stock_data(
        "TCS.NS", start="2026-01-01", end="2026-01-31"
    )
    assert out_of_range.empty
    assert_canonical_contract(out_of_range)


def test_metadata_tracking_and_has_full_history_flag_preservation(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")

    assert repo.get_metadata("RELIANCE.NS") is None

    # Empty upsert on unknown symbol does not create empty metadata
    repo.upsert_stock_data("RELIANCE.NS", pd.DataFrame())
    assert repo.get_metadata("RELIANCE.NS") is None

    t1 = datetime(2025, 1, 10, 9, 0, 0, tzinfo=timezone.utc)
    batch1 = make_canonical_df(
        dates=["2025-01-03", "2025-01-06"], closes=[100.0, 102.0]
    )
    repo.upsert_stock_data(
        "reliance.ns", batch1, mark_full_history=False, synced_at=t1
    )

    meta1 = repo.get_metadata("RELIANCE.NS")
    assert meta1 == SymbolMetadata(
        symbol="RELIANCE.NS",
        min_date="2025-01-03",
        max_date="2025-01-06",
        last_synced_at="2025-01-10T09:00:00+00:00",
        has_full_history=False,
        row_count=2,
    )

    # Backfill earlier history and mark_full_history=True
    t2 = datetime(2025, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    batch2 = make_canonical_df(
        dates=["2025-01-01", "2025-01-02"], closes=[96.0, 98.0]
    )
    repo.upsert_stock_data(
        "RELIANCE.NS", batch2, mark_full_history=True, synced_at=t2
    )

    meta2 = repo.get_metadata("RELIANCE.NS")
    assert meta2 == SymbolMetadata(
        symbol="RELIANCE.NS",
        min_date="2025-01-01",
        max_date="2025-01-06",
        last_synced_at="2025-01-10T10:00:00+00:00",
        has_full_history=True,
        row_count=4,
    )

    # Subsequent partial update with mark_full_history=False must NOT downgrade has_full_history
    t3 = datetime(2025, 1, 11, 12, 0, 0, tzinfo=timezone.utc)
    batch3 = make_canonical_df(dates=["2025-01-07"], closes=[105.0])
    repo.upsert_stock_data(
        "RELIANCE.NS", batch3, mark_full_history=False, synced_at=t3
    )

    meta3 = repo.get_metadata("RELIANCE.NS")
    assert meta3 == SymbolMetadata(
        symbol="RELIANCE.NS",
        min_date="2025-01-01",
        max_date="2025-01-07",
        last_synced_at="2025-01-11T12:00:00+00:00",
        has_full_history=True,
        row_count=5,
    )


def test_get_latest_row_returns_most_recent_bar_or_canonical_empty(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")

    empty_latest = repo.get_latest_row("TCS.NS")
    assert empty_latest.empty
    assert_canonical_contract(empty_latest)

    df = make_canonical_df(
        dates=["2025-01-01", "2025-01-02", "2025-01-05"],
        closes=[100.0, 110.0, 125.0],
        volumes=[1000, 1500, 2200],
    )
    repo.upsert_stock_data("TCS.NS", df)

    latest = repo.get_latest_row("TCS.NS")
    assert_canonical_contract(latest)
    assert len(latest) == 1
    assert latest.index[0] == pd.Timestamp("2025-01-05")
    assert latest["Close"].iloc[0] == 125.0
    assert latest["Volume"].iloc[0] == 2200


def test_analytics_compatibility_with_repository_output(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    df = make_canonical_df(
        dates=["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        closes=[100.0, 120.0, 90.0, 90.0, 110.0],
        volumes=[1000, 2500, 1500, 1200, 1800],
    )
    repo.upsert_stock_data("RELIANCE.NS", df)

    stored = repo.get_stock_data("RELIANCE.NS")
    with_returns = calculate_daily_returns(stored)
    summary = calculate_summary(with_returns)

    assert summary["latest_price"] == 110.0
    assert summary["highest_price"] == 120.0
    assert summary["lowest_price"] == 90.0
    assert summary["total_return_percent"] == 10.0
    assert summary["max_drawdown"] == -25.0
    assert summary["highest_volume"] == 2500


def test_transaction_integrity_and_invalid_input_rejection_prevent_partial_writes(
    tmp_path, monkeypatch
):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    initial_df = make_canonical_df(
        dates=["2025-01-01", "2025-01-02"], closes=[100.0, 105.0]
    )
    t_initial = datetime(2025, 1, 2, 16, 0, 0, tzinfo=timezone.utc)
    repo.upsert_stock_data("TCS.NS", initial_df, synced_at=t_initial)

    # 1. Invalid canonical input (e.g., negative Close price) is rejected before touching DB
    corrupted_df = initial_df.copy()
    corrupted_df.loc[pd.Timestamp("2025-01-02"), "Close"] = -50.0
    with pytest.raises(ValueError, match="non-positive"):
        repo.upsert_stock_data("TCS.NS", corrupted_df)

    # 2. Simulate a failure partway through transaction (after daily_prices insert, before metadata commit)
    new_batch = make_canonical_df(
        dates=["2025-01-02", "2025-01-03"], closes=[999.0, 1000.0]
    )
    original_connect = repo._connect

    class FailingMidTransactionConnection:
        def __init__(self, real_conn):
            self._real_conn = real_conn

        def executemany(self, sql, params):
            return self._real_conn.executemany(sql, params)

        def execute(self, sql, params=()):
            if "INSERT INTO ingestion_metadata" in sql:
                raise sqlite3.OperationalError("Simulated failure during metadata update")
            return self._real_conn.execute(sql, params)

    from contextlib import contextmanager

    @contextmanager
    def failing_connect():
        raw_conn = sqlite3.connect(repo.db_path)
        try:
            with raw_conn:
                yield FailingMidTransactionConnection(raw_conn)
        finally:
            raw_conn.close()

    monkeypatch.setattr(repo, "_connect", failing_connect)

    with pytest.raises(sqlite3.OperationalError, match="Simulated failure"):
        repo.upsert_stock_data("TCS.NS", new_batch)

    # Restore connection and verify neither the price update (999.0) nor new row (2025-01-03) was committed
    monkeypatch.setattr(repo, "_connect", original_connect)
    stored = repo.get_stock_data("TCS.NS")
    assert len(stored) == 2
    assert list(stored["Close"]) == [100.0, 105.0]
    meta = repo.get_metadata("TCS.NS")
    assert meta.row_count == 2
    assert meta.max_date == "2025-01-02"
    assert meta.last_synced_at == "2025-01-02T16:00:00+00:00"


def test_rejects_same_day_multiple_timestamps_and_non_midnight_dates_without_collapsing(
    tmp_path,
):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")

    # 1. Valid daily stock data (midnight-normalized unique dates) succeeds
    valid_daily = make_canonical_df(
        dates=["2025-01-01", "2025-01-02"], closes=[100.0, 105.0]
    )
    assert repo.upsert_stock_data("RELIANCE.NS", valid_daily) == 2

    # 2. Multiple timestamps on the same calendar date (e.g. 09:15 and 15:30 on 2025-01-03)
    # must be rejected rather than silently collapsing into one '2025-01-03' row
    same_day_intraday = pd.DataFrame(
        {
            "Open": [105.0, 106.0],
            "High": [108.0, 110.0],
            "Low": [104.0, 105.0],
            "Close": [106.0, 109.0],
            "Volume": [1000, 2000],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=pd.DatetimeIndex(
            ["2025-01-03 09:15:00", "2025-01-03 15:30:00"], name="Date"
        ),
    )
    with pytest.raises(
        ValueError, match="multiple timestamps for the same calendar date"
    ):
        repo.upsert_stock_data("RELIANCE.NS", same_day_intraday)

    # 3. Single non-midnight timestamp is also rejected by the canonical daily contract
    non_midnight = pd.DataFrame(
        {
            "Open": [105.0],
            "High": [108.0],
            "Low": [104.0],
            "Close": [106.0],
            "Volume": [1000],
            "Dividends": [0.0],
            "Stock Splits": [0.0],
        },
        index=pd.DatetimeIndex(["2025-01-03 15:30:00"], name="Date"),
    )
    with pytest.raises(ValueError, match="normalized to midnight"):
        repo.upsert_stock_data("RELIANCE.NS", non_midnight)

    # Existing stored rows for RELIANCE.NS remain untouched
    stored = repo.get_stock_data("RELIANCE.NS")
    assert len(stored) == 2
    assert list(stored.index) == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    ]


def _read_cached_ranges(repo: MarketDataRepository, symbol: str) -> list[tuple[str, str]]:
    with sqlite3.connect(repo.db_path) as conn:
        return conn.execute(
            """
            SELECT start_date, end_date
            FROM cached_ranges
            WHERE symbol = ?
            ORDER BY start_date ASC;
            """,
            (symbol,),
        ).fetchall()


def test_coverage_lookup_and_interval_merging_rules(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    sample_bar = make_canonical_df(dates=["2025-01-02"], closes=[100.0])

    # 1. Seed two disjoint intervals: Jan 1-10 and Jan 20-31
    repo.upsert_stock_data(
        "RELIANCE.NS",
        sample_bar,
        coverage_start="2025-01-01",
        coverage_end="2025-01-10",
    )
    repo.upsert_stock_data(
        "RELIANCE.NS",
        sample_bar,
        coverage_start="2025-01-20",
        coverage_end="2025-01-31",
    )
    assert _read_cached_ranges(repo, "RELIANCE.NS") == [
        ("2025-01-01", "2025-01-10"),
        ("2025-01-20", "2025-01-31"),
    ]

    # Coverage lookup checks:
    assert repo.is_range_covered("reliance.ns", "2025-01-01", "2025-01-10") is True
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-03", "2025-01-08") is True
    # Partially outside covered range -> False
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-05", "2025-01-15") is False
    # Spanning two disjoint intervals -> False
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-05", "2025-01-25") is False
    # Uncovered gap between disjoint intervals -> False
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-12", "2025-01-18") is False
    # Different symbol -> False
    assert repo.is_range_covered("TCS.NS", "2025-01-01", "2025-01-10") is False
    # Reversed range -> raises ValueError
    with pytest.raises(ValueError, match="start date must be before or equal to end date"):
        repo.is_range_covered("RELIANCE.NS", "2025-01-10", "2025-01-01")

    # 2. Overlapping interval (Jan 8-14 overlaps Jan 1-10) -> merges to Jan 1-14
    repo.upsert_stock_data(
        "RELIANCE.NS",
        sample_bar,
        coverage_start="2025-01-08",
        coverage_end="2025-01-14",
    )
    assert _read_cached_ranges(repo, "RELIANCE.NS") == [
        ("2025-01-01", "2025-01-14"),
        ("2025-01-20", "2025-01-31"),
    ]

    # 3. One-calendar-day gap (Jan 16-18 leaves Jan 15 uncovered) -> remains separate
    repo.upsert_stock_data(
        "RELIANCE.NS",
        sample_bar,
        coverage_start="2025-01-16",
        coverage_end="2025-01-18",
    )
    assert _read_cached_ranges(repo, "RELIANCE.NS") == [
        ("2025-01-01", "2025-01-14"),
        ("2025-01-16", "2025-01-18"),
        ("2025-01-20", "2025-01-31"),
    ]

    # 4. Calendar-adjacent interval (Jan 19 touches both Jan 16-18 and Jan 20-31) -> merges both into Jan 16-31
    repo.upsert_stock_data(
        "RELIANCE.NS",
        sample_bar,
        coverage_start="2025-01-19",
        coverage_end="2025-01-19",
    )
    assert _read_cached_ranges(repo, "RELIANCE.NS") == [
        ("2025-01-01", "2025-01-14"),
        ("2025-01-16", "2025-01-31"),
    ]

    # 5. Bridging gap (Jan 15 bridges Jan 1-14 and Jan 16-31) -> merges into a single Jan 1-31 interval
    repo.upsert_stock_data(
        "RELIANCE.NS",
        sample_bar,
        coverage_start="2025-01-15",
        coverage_end="2025-01-15",
    )
    assert _read_cached_ranges(repo, "RELIANCE.NS") == [
        ("2025-01-01", "2025-01-31"),
    ]
    assert repo.is_range_covered("RELIANCE.NS", "2025-01-01", "2025-01-31") is True

    # 6. Fully contained interval (Jan 5-12 inside Jan 1-31) -> no duplicates created
    repo.upsert_stock_data(
        "RELIANCE.NS",
        sample_bar,
        coverage_start="2025-01-05",
        coverage_end="2025-01-12",
    )
    assert _read_cached_ranges(repo, "RELIANCE.NS") == [
        ("2025-01-01", "2025-01-31"),
    ]


def test_empty_upsert_strict_noop_and_invalid_coverage_arguments(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    t1 = datetime(2025, 1, 10, 9, 0, 0, tzinfo=timezone.utc)
    seed_df = make_canonical_df(
        dates=["2025-01-02", "2025-01-03"], closes=[100.0, 105.0]
    )
    repo.upsert_stock_data(
        "TCS.NS",
        seed_df,
        mark_full_history=False,
        synced_at=t1,
        coverage_start="2025-01-01",
        coverage_end="2025-01-05",
    )

    # 1. Invalid coverage arguments must raise ValueError BEFORE empty-DataFrame early return
    with pytest.raises(ValueError, match="provided together"):
        repo.upsert_stock_data(
            "TCS.NS", pd.DataFrame(), coverage_start="2025-01-01"
        )
    with pytest.raises(ValueError, match="provided together"):
        repo.upsert_stock_data(
            "TCS.NS", pd.DataFrame(), coverage_end="2025-01-05"
        )
    with pytest.raises(
        ValueError, match="coverage_start must be before or equal to coverage_end"
    ):
        repo.upsert_stock_data(
            "TCS.NS",
            pd.DataFrame(),
            coverage_start="2025-01-10",
            coverage_end="2025-01-01",
        )
    with pytest.raises(
        ValueError, match="coverage_start must be before or equal to coverage_end"
    ):
        repo.upsert_stock_data(
            "TCS.NS",
            seed_df,
            coverage_start="2025-01-10",
            coverage_end="2025-01-01",
        )

    # 2. Empty DataFrame with valid arguments is a strict no-op (does not promote has_full_history,
    # does not update last_synced_at, and does not write cached_ranges)
    t2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    res = repo.upsert_stock_data(
        "TCS.NS",
        pd.DataFrame(),
        mark_full_history=True,
        synced_at=t2,
        coverage_start="2025-01-01",
        coverage_end="2025-01-31",
    )
    assert res == 0
    assert len(repo.get_stock_data("TCS.NS")) == 2
    meta = repo.get_metadata("TCS.NS")
    assert meta == SymbolMetadata(
        symbol="TCS.NS",
        min_date="2025-01-02",
        max_date="2025-01-03",
        last_synced_at="2025-01-10T09:00:00+00:00",
        has_full_history=False,
        row_count=2,
    )
    assert _read_cached_ranges(repo, "TCS.NS") == [("2025-01-01", "2025-01-05")]


def test_atomicity_rolls_back_prices_and_coverage_when_coverage_write_fails(
    tmp_path, monkeypatch
):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    initial_df = make_canonical_df(dates=["2025-01-02"], closes=[100.0])
    t_initial = datetime(2025, 1, 2, 16, 0, 0, tzinfo=timezone.utc)
    repo.upsert_stock_data(
        "INFY.NS",
        initial_df,
        synced_at=t_initial,
        coverage_start="2025-01-01",
        coverage_end="2025-01-02",
    )

    new_batch = make_canonical_df(
        dates=["2025-01-02", "2025-01-03"], closes=[200.0, 210.0]
    )
    original_connect = repo._connect

    class FailingCoverageConnection:
        def __init__(self, real_conn):
            self._real_conn = real_conn

        def executemany(self, sql, params):
            return self._real_conn.executemany(sql, params)

        def execute(self, sql, params=()):
            if "INSERT INTO cached_ranges" in sql:
                raise sqlite3.OperationalError("Simulated failure during coverage write")
            return self._real_conn.execute(sql, params)

    from contextlib import contextmanager

    @contextmanager
    def failing_connect():
        raw_conn = sqlite3.connect(repo.db_path)
        try:
            with raw_conn:
                yield FailingCoverageConnection(raw_conn)
        finally:
            raw_conn.close()

    monkeypatch.setattr(repo, "_connect", failing_connect)

    with pytest.raises(sqlite3.OperationalError, match="Simulated failure during coverage write"):
        repo.upsert_stock_data(
            "INFY.NS",
            new_batch,
            coverage_start="2025-01-01",
            coverage_end="2025-01-05",
        )

    monkeypatch.setattr(repo, "_connect", original_connect)
    stored = repo.get_stock_data("INFY.NS")
    assert len(stored) == 1
    assert list(stored["Close"]) == [100.0]
    meta = repo.get_metadata("INFY.NS")
    assert meta.row_count == 1
    assert meta.last_synced_at == "2025-01-02T16:00:00+00:00"
    assert _read_cached_ranges(repo, "INFY.NS") == [("2025-01-01", "2025-01-02")]


def test_get_missing_ranges_across_coverage_scenarios(tmp_path):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")
    bar = make_canonical_df(dates=["2025-01-02"], closes=[100.0])

    # 1. No coverage -> full requested interval is missing
    assert repo.get_missing_ranges("RELIANCE.NS", "2025-01-01", "2025-01-31") == [
        ("2025-01-01", "2025-01-31")
    ]

    # 2. Partial coverage at beginning (Jan 1-10 covered, req Jan 1-31) -> Jan 11-31 missing
    repo.upsert_stock_data(
        "RELIANCE.NS",
        bar,
        coverage_start="2025-01-01",
        coverage_end="2025-01-10",
    )
    assert repo.get_missing_ranges("RELIANCE.NS", "2025-01-01", "2025-01-31") == [
        ("2025-01-11", "2025-01-31")
    ]

    # 3. Full coverage (exact boundary and interior sub-range) -> []
    assert repo.get_missing_ranges("RELIANCE.NS", "2025-01-01", "2025-01-10") == []
    assert repo.get_missing_ranges("reliance.ns", "2025-01-03", "2025-01-08") == []

    # 4. Partial coverage at end (add Jan 20-31, query Jan 12-31) -> Jan 12-19 missing
    repo.upsert_stock_data(
        "RELIANCE.NS",
        bar,
        coverage_start="2025-01-20",
        coverage_end="2025-01-31",
    )
    assert repo.get_missing_ranges("RELIANCE.NS", "2025-01-12", "2025-01-31") == [
        ("2025-01-12", "2025-01-19")
    ]

    # 5. Multiple disjoint intervals (Jan 1-10 and Jan 20-31 covered, query Jan 1-31) -> Jan 11-19 missing
    assert repo.get_missing_ranges("RELIANCE.NS", "2025-01-01", "2025-01-31") == [
        ("2025-01-11", "2025-01-19")
    ]

    # 6. Cached interval strictly inside requested range (add Jan 14-16; query Jan 11-19)
    # -> two missing gaps: Jan 11-13 and Jan 17-19
    repo.upsert_stock_data(
        "RELIANCE.NS",
        bar,
        coverage_start="2025-01-14",
        coverage_end="2025-01-16",
    )
    assert repo.get_missing_ranges("RELIANCE.NS", "2025-01-11", "2025-01-19") == [
        ("2025-01-11", "2025-01-13"),
        ("2025-01-17", "2025-01-19"),
    ]
    # Querying full month Jan 1-31 returns both gaps
    assert repo.get_missing_ranges("RELIANCE.NS", "2025-01-01", "2025-01-31") == [
        ("2025-01-11", "2025-01-13"),
        ("2025-01-17", "2025-01-19"),
    ]

    # 7. One-day gap (cover Jan 11-13 and Jan 17-18, leaving only Jan 19 uncovered)
    repo.upsert_stock_data(
        "RELIANCE.NS",
        bar,
        coverage_start="2025-01-11",
        coverage_end="2025-01-13",
    )
    repo.upsert_stock_data(
        "RELIANCE.NS",
        bar,
        coverage_start="2025-01-17",
        coverage_end="2025-01-18",
    )
    assert repo.get_missing_ranges("RELIANCE.NS", "2025-01-01", "2025-01-31") == [
        ("2025-01-19", "2025-01-19")
    ]

    # 8. Leap-day handling (2024 is a leap year: Feb 25-28 and Mar 1-5 covered -> 2024-02-29 missing)
    repo.upsert_stock_data(
        "LEAP.NS",
        bar,
        coverage_start="2024-02-25",
        coverage_end="2024-02-28",
    )
    repo.upsert_stock_data(
        "LEAP.NS",
        bar,
        coverage_start="2024-03-01",
        coverage_end="2024-03-05",
    )
    assert repo.get_missing_ranges("LEAP.NS", "2024-02-25", "2024-03-05") == [
        ("2024-02-29", "2024-02-29")
    ]

    # 9. Year-boundary handling (2025-12-25..2025-12-31 covered, query through 2026-01-05)
    repo.upsert_stock_data(
        "YEAR.NS",
        bar,
        coverage_start="2025-12-25",
        coverage_end="2025-12-31",
    )
    assert repo.get_missing_ranges("YEAR.NS", "2025-12-25", "2026-01-05") == [
        ("2026-01-01", "2026-01-05")
    ]

    # 10. Symbol isolation (OTHER.NS has no coverage)
    assert repo.get_missing_ranges("OTHER.NS", "2025-01-01", "2025-01-10") == [
        ("2025-01-01", "2025-01-10")
    ]

    # 11. Reversed dates raise ValueError
    with pytest.raises(ValueError, match="start date must be before or equal to end date"):
        repo.get_missing_ranges("RELIANCE.NS", "2025-01-31", "2025-01-01")


def test_legacy_database_migration_adds_last_refreshed_at_idempotently(tmp_path):
    db_path = tmp_path / "legacy_market_data.db"

    # Create a pre-M4.4a database without last_refreshed_at in ingestion_metadata
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE daily_prices (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                dividends REAL NOT NULL,
                stock_splits REAL NOT NULL,
                PRIMARY KEY (symbol, trade_date)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE ingestion_metadata (
                symbol TEXT PRIMARY KEY,
                min_date TEXT NOT NULL,
                max_date TEXT NOT NULL,
                last_synced_at TEXT NOT NULL,
                has_full_history INTEGER NOT NULL DEFAULT 0,
                row_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            INSERT INTO ingestion_metadata (
                symbol, min_date, max_date, last_synced_at, has_full_history, row_count
            )
            VALUES ('RELIANCE.NS', '2025-01-01', '2025-01-10', '2025-01-10T10:00:00+00:00', 0, 8);
            """
        )

    # First initialization performs ALTER TABLE; second verifies idempotency
    repo1 = MarketDataRepository(db_path=db_path)
    repo2 = MarketDataRepository(db_path=db_path)

    meta = repo2.get_metadata("RELIANCE.NS")
    assert meta == SymbolMetadata(
        symbol="RELIANCE.NS",
        min_date="2025-01-01",
        max_date="2025-01-10",
        last_synced_at="2025-01-10T10:00:00+00:00",
        has_full_history=False,
        row_count=8,
        last_refreshed_at=None,
    )
    assert repo1.get_metadata("RELIANCE.NS") == meta


def test_last_refreshed_at_qualified_non_qualified_empty_and_rollback(
    tmp_path, monkeypatch
):
    repo = MarketDataRepository(db_path=tmp_path / "market_data.db")

    t1 = datetime(2025, 1, 10, 9, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2025, 1, 11, 8, 30, 0, tzinfo=timezone.utc)
    t4 = datetime(2025, 1, 12, 10, 0, 0, tzinfo=timezone.utc)
    t5 = datetime(2025, 1, 13, 14, 0, 0, tzinfo=timezone.utc)

    # 1. Non-qualified initial historical upsert sets last_synced_at, leaves last_refreshed_at None
    batch1 = make_canonical_df(dates=["2025-01-05"], closes=[100.0])
    repo.upsert_stock_data(
        "TCS.NS", batch1, synced_at=t1, mark_refreshed=False
    )
    meta1 = repo.get_metadata("TCS.NS")
    assert meta1.last_synced_at == "2025-01-10T09:00:00+00:00"
    assert meta1.last_refreshed_at is None

    # 2. Qualified refresh updates both last_synced_at and last_refreshed_at to identical UTC timestamp
    batch2 = make_canonical_df(dates=["2025-01-10"], closes=[105.0])
    repo.upsert_stock_data(
        "TCS.NS", batch2, synced_at=t2, mark_refreshed=True
    )
    meta2 = repo.get_metadata("TCS.NS")
    assert meta2.last_synced_at == "2025-01-10T12:00:00+00:00"
    assert meta2.last_refreshed_at == "2025-01-10T12:00:00+00:00"

    # 3. Non-qualified historical backfill updates last_synced_at but preserves previous last_refreshed_at
    batch3 = make_canonical_df(dates=["2025-01-02"], closes=[98.0])
    repo.upsert_stock_data(
        "TCS.NS", batch3, synced_at=t3, mark_refreshed=False
    )
    meta3 = repo.get_metadata("TCS.NS")
    assert meta3.last_synced_at == "2025-01-11T08:30:00+00:00"
    assert meta3.last_refreshed_at == "2025-01-10T12:00:00+00:00"

    # 4. Empty DataFrame upsert with mark_refreshed=True changes neither timestamp
    repo.upsert_stock_data(
        "TCS.NS", pd.DataFrame(), synced_at=t4, mark_refreshed=True
    )
    meta4 = repo.get_metadata("TCS.NS")
    assert meta4.last_synced_at == "2025-01-11T08:30:00+00:00"
    assert meta4.last_refreshed_at == "2025-01-10T12:00:00+00:00"

    # 5. Failed transaction with mark_refreshed=True rolls back and changes neither timestamp
    original_connect = repo._connect

    class FailingMetadataConnection:
        def __init__(self, real_conn):
            self._real_conn = real_conn

        def executemany(self, sql, params):
            return self._real_conn.executemany(sql, params)

        def execute(self, sql, params=()):
            if "INSERT INTO ingestion_metadata" in sql:
                raise sqlite3.OperationalError("Simulated metadata failure")
            return self._real_conn.execute(sql, params)

    from contextlib import contextmanager

    @contextmanager
    def failing_connect():
        raw_conn = sqlite3.connect(repo.db_path)
        try:
            with raw_conn:
                yield FailingMetadataConnection(raw_conn)
        finally:
            raw_conn.close()

    monkeypatch.setattr(repo, "_connect", failing_connect)
    batch5 = make_canonical_df(dates=["2025-01-13"], closes=[110.0])
    with pytest.raises(sqlite3.OperationalError, match="Simulated metadata failure"):
        repo.upsert_stock_data(
            "TCS.NS", batch5, synced_at=t5, mark_refreshed=True
        )

    monkeypatch.setattr(repo, "_connect", original_connect)
    meta5 = repo.get_metadata("TCS.NS")
    assert meta5.last_synced_at == "2025-01-11T08:30:00+00:00"
    assert meta5.last_refreshed_at == "2025-01-10T12:00:00+00:00"




