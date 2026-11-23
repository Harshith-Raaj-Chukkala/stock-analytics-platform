from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

from app.services.data_validator import (
    CANONICAL_COLUMNS,
    OHLC_TOLERANCE,
    REQUIRED_OHLCV_COLUMNS,
    _empty_canonical_df,
)

DEFAULT_DB_PATH = Path("data/market_data.db")


@dataclass(frozen=True)
class SymbolMetadata:
    """
    Per-symbol ingestion state stored in SQLite.

    Representations:
    - symbol: Uppercase ticker symbol (e.g. "RELIANCE.NS").
    - min_date: Earliest stored trading date as ISO "YYYY-MM-DD".
    - max_date: Latest stored trading date as ISO "YYYY-MM-DD".
    - last_synced_at: ISO-8601 UTC timestamp string (e.g. "2025-01-15T10:30:00+00:00").
    - has_full_history: True if full available history has ever been marked for this symbol.
    - row_count: Total number of stored daily bars for this symbol.
    - last_refreshed_at: ISO-8601 UTC timestamp string of the last recent-edge refresh, or None.
    """

    symbol: str
    min_date: str
    max_date: str
    last_synced_at: str
    has_full_history: bool
    row_count: int
    last_refreshed_at: str | None = None


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Symbol must be a non-empty string.")
    return symbol.strip().upper()


def _normalize_query_date(value: date | datetime | pd.Timestamp | str | None, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError(f"{name} cannot be NaT.")
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} cannot be an empty string.")
        try:
            return date.fromisoformat(stripped).isoformat()
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date for {name}: {value!r}") from exc
    raise ValueError(f"Unsupported date type for {name}: {type(value).__name__}")


def _normalize_synced_at(synced_at: datetime | str | None) -> str:
    if synced_at is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(synced_at, datetime):
        if synced_at.tzinfo is None:
            return synced_at.replace(tzinfo=timezone.utc).isoformat()
        return synced_at.astimezone(timezone.utc).isoformat()
    if isinstance(synced_at, str):
        stripped = synced_at.strip()
        if not stripped:
            raise ValueError("synced_at cannot be an empty string.")
        try:
            parsed = datetime.fromisoformat(stripped)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO timestamp for synced_at: {synced_at!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat()
    raise ValueError(f"Unsupported synced_at type: {type(synced_at).__name__}")


def _verify_canonical_dataframe(df: pd.DataFrame) -> None:
    """
    Verify that a non-empty DataFrame adheres to the canonical contract before writing.
    Raises ValueError if the structure or values would corrupt stored data.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("MultiIndex columns are not supported.")

    if df.empty:
        if len(df.columns) > 0:
            missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required OHLCV columns: {missing}")
        return

    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing canonical columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas DatetimeIndex.")
    if df.index.tz is not None:
        raise ValueError("DataFrame DatetimeIndex must be timezone-naive.")
    if df.index.isna().any():
        raise ValueError("DataFrame index contains NaT values.")

    normalized_dates = df.index.normalize()
    if normalized_dates.duplicated().any():
        raise ValueError(
            "DataFrame index contains multiple timestamps for the same calendar date."
        )
    if (df.index != normalized_dates).any():
        raise ValueError(
            "DataFrame DatetimeIndex must be normalized to midnight (00:00:00)."
        )

    for col in ("Open", "High", "Low", "Close"):
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column {col} must be numeric.")
        if series.isna().any() or not np.isfinite(series).all() or (series <= 0).any():
            raise ValueError(f"Column {col} contains null, non-finite, or non-positive values.")

    vol = df["Volume"]
    if not pd.api.types.is_numeric_dtype(vol):
        raise ValueError("Column Volume must be numeric.")
    if vol.isna().any() or not np.isfinite(vol).all() or (vol < 0).any() or (np.floor(vol) != vol).any():
        raise ValueError("Column Volume contains null, negative, or fractional values.")

    for col in ("Dividends", "Stock Splits"):
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column {col} must be numeric.")
        if series.isna().any() or not np.isfinite(series).all() or (series < 0).any():
            raise ValueError(f"Column {col} contains null, non-finite, or negative values.")

    oc_max = df[["Open", "Close"]].max(axis=1)
    oc_min = df[["Open", "Close"]].min(axis=1)
    if (
        (df["High"] < df["Low"] - OHLC_TOLERANCE).any()
        | (df["High"] < oc_max - OHLC_TOLERANCE).any()
        | (df["Low"] > oc_min + OHLC_TOLERANCE).any()
    ):
        raise ValueError("DataFrame contains rows violating the OHLC price envelope.")


def _rows_to_canonical_df(rows: list[tuple]) -> pd.DataFrame:
    if not rows:
        return _empty_canonical_df()

    records = np.array(rows, dtype=object)
    dates = pd.DatetimeIndex(
        pd.to_datetime(records[:, 0], format="%Y-%m-%d"), name="Date"
    ).normalize()
    dates.name = "Date"

    df = pd.DataFrame(
        {
            "Open": records[:, 1].astype("float64"),
            "High": records[:, 2].astype("float64"),
            "Low": records[:, 3].astype("float64"),
            "Close": records[:, 4].astype("float64"),
            "Volume": records[:, 5].astype("int64"),
            "Dividends": records[:, 6].astype("float64"),
            "Stock Splits": records[:, 7].astype("float64"),
        },
        index=dates,
    )
    return df[list(CANONICAL_COLUMNS)]


class MarketDataRepository:
    """
    SQLite repository for canonical daily stock OHLCV bars and symbol-level sync metadata.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_prices (
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
                CREATE TABLE IF NOT EXISTS ingestion_metadata (
                    symbol TEXT PRIMARY KEY,
                    min_date TEXT NOT NULL,
                    max_date TEXT NOT NULL,
                    last_synced_at TEXT NOT NULL,
                    has_full_history INTEGER NOT NULL DEFAULT 0,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    last_refreshed_at TEXT
                );
                """
            )
            existing_meta_cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(ingestion_metadata);"
                ).fetchall()
            }
            if "last_refreshed_at" not in existing_meta_cols:
                conn.execute(
                    "ALTER TABLE ingestion_metadata ADD COLUMN last_refreshed_at TEXT;"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cached_ranges (
                    symbol TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    PRIMARY KEY (symbol, start_date),
                    CHECK (start_date <= end_date)
                );
                """
            )

    def is_range_covered(
        self,
        symbol: str,
        start: date | datetime | pd.Timestamp | str,
        end: date | datetime | pd.Timestamp | str,
    ) -> bool:
        """
        Return True if a single stored coverage interval in `cached_ranges`
        fully contains the inclusive [start, end] range for `symbol`.
        """
        norm_symbol = _normalize_symbol(symbol)
        start_str = _normalize_query_date(start, "start")
        end_str = _normalize_query_date(end, "end")

        if start_str is None or end_str is None:
            raise ValueError("Both start and end dates are required.")
        if start_str > end_str:
            raise ValueError("start date must be before or equal to end date.")

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM cached_ranges
                WHERE symbol = ?
                  AND start_date <= ?
                  AND end_date >= ?
                LIMIT 1;
                """,
                (norm_symbol, start_str, end_str),
            ).fetchone()

        return row is not None

    def get_missing_ranges(
        self,
        symbol: str,
        start: date | datetime | pd.Timestamp | str,
        end: date | datetime | pd.Timestamp | str,
    ) -> list[tuple[str, str]]:
        """
        Return a chronologically ordered list of uncovered inclusive (start_date, end_date)
        ISO calendar-date intervals within [start, end] for `symbol`.
        """
        norm_symbol = _normalize_symbol(symbol)
        start_str = _normalize_query_date(start, "start")
        end_str = _normalize_query_date(end, "end")

        if start_str is None or end_str is None:
            raise ValueError("Both start and end dates are required.")
        if start_str > end_str:
            raise ValueError("start date must be before or equal to end date.")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT start_date, end_date
                FROM cached_ranges
                WHERE symbol = ?
                  AND start_date <= ?
                  AND end_date >= ?
                ORDER BY start_date ASC;
                """,
                (norm_symbol, end_str, start_str),
            ).fetchall()

        req_start_dt = date.fromisoformat(start_str)
        req_end_dt = date.fromisoformat(end_str)
        cursor = req_start_dt
        missing: list[tuple[str, str]] = []

        for cov_start_str, cov_end_str in rows:
            cov_start_dt = date.fromisoformat(cov_start_str)
            cov_end_dt = date.fromisoformat(cov_end_str)

            if cov_start_dt > cursor:
                gap_end_dt = min(req_end_dt, cov_start_dt - timedelta(days=1))
                missing.append((cursor.isoformat(), gap_end_dt.isoformat()))

            next_day = cov_end_dt + timedelta(days=1)
            if next_day > cursor:
                cursor = next_day
            if cursor > req_end_dt:
                break

        if cursor <= req_end_dt:
            missing.append((cursor.isoformat(), req_end_dt.isoformat()))

        return missing

    def upsert_stock_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        mark_full_history: bool = False,
        synced_at: datetime | str | None = None,
        coverage_start: date | datetime | pd.Timestamp | str | None = None,
        coverage_end: date | datetime | pd.Timestamp | str | None = None,
        mark_refreshed: bool = False,
    ) -> int:
        """
        Atomically upsert canonical daily OHLCV rows, recalculate ingestion metadata,
        and optionally merge a covered date interval into `cached_ranges`.

        Empty DataFrame behavior:
        - After validating arguments, an empty DataFrame is a strict no-op returning `0`:
          it does not insert prices, update metadata, promote `has_full_history`,
          update `last_refreshed_at`, or modify `cached_ranges`.
        """
        norm_symbol = _normalize_symbol(symbol)
        _verify_canonical_dataframe(df)
        norm_synced_at = _normalize_synced_at(synced_at)
        full_history_flag = 1 if bool(mark_full_history) else 0
        refreshed_at_val = norm_synced_at if bool(mark_refreshed) else None

        if (coverage_start is None) != (coverage_end is None):
            raise ValueError(
                "coverage_start and coverage_end must be provided together or both omitted."
            )

        cov_start_str: str | None = None
        cov_end_str: str | None = None
        if coverage_start is not None and coverage_end is not None:
            cov_start_str = _normalize_query_date(coverage_start, "coverage_start")
            cov_end_str = _normalize_query_date(coverage_end, "coverage_end")
            if cov_start_str > cov_end_str:
                raise ValueError(
                    "coverage_start must be before or equal to coverage_end."
                )

        if df.empty:
            return 0

        working = df[list(CANONICAL_COLUMNS)].copy()
        working = working.sort_index(kind="mergesort")

        records = [
            (
                norm_symbol,
                ts.strftime("%Y-%m-%d"),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"]),
                float(row["Dividends"]),
                float(row["Stock Splits"]),
            )
            for ts, row in working.iterrows()
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO daily_prices (
                    symbol, trade_date, open, high, low, close, volume, dividends, stock_splits
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, trade_date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    dividends = excluded.dividends,
                    stock_splits = excluded.stock_splits;
                """,
                records,
            )

            stats_row = conn.execute(
                """
                SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
                FROM daily_prices
                WHERE symbol = ?;
                """,
                (norm_symbol,),
            ).fetchone()

            min_date, max_date, row_count = stats_row

            conn.execute(
                """
                INSERT INTO ingestion_metadata (
                    symbol, min_date, max_date, last_synced_at, has_full_history, row_count, last_refreshed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    min_date = excluded.min_date,
                    max_date = excluded.max_date,
                    last_synced_at = excluded.last_synced_at,
                    has_full_history = MAX(ingestion_metadata.has_full_history, excluded.has_full_history),
                    row_count = excluded.row_count,
                    last_refreshed_at = COALESCE(excluded.last_refreshed_at, ingestion_metadata.last_refreshed_at);
                """,
                (
                    norm_symbol,
                    min_date,
                    max_date,
                    norm_synced_at,
                    full_history_flag,
                    int(row_count),
                    refreshed_at_val,
                ),
            )

            if cov_start_str is not None and cov_end_str is not None:
                lo_date = (
                    date.fromisoformat(cov_start_str) - timedelta(days=1)
                ).isoformat()
                hi_date = (
                    date.fromisoformat(cov_end_str) + timedelta(days=1)
                ).isoformat()

                touching_rows = conn.execute(
                    """
                    SELECT start_date, end_date
                    FROM cached_ranges
                    WHERE symbol = ?
                      AND start_date <= ?
                      AND end_date >= ?;
                    """,
                    (norm_symbol, hi_date, lo_date),
                ).fetchall()

                merged_start = min([cov_start_str, *(r[0] for r in touching_rows)])
                merged_end = max([cov_end_str, *(r[1] for r in touching_rows)])

                if touching_rows:
                    conn.execute(
                        """
                        DELETE FROM cached_ranges
                        WHERE symbol = ?
                          AND start_date <= ?
                          AND end_date >= ?;
                        """,
                        (norm_symbol, hi_date, lo_date),
                    )

                conn.execute(
                    """
                    INSERT INTO cached_ranges (symbol, start_date, end_date)
                    VALUES (?, ?, ?);
                    """,
                    (norm_symbol, merged_start, merged_end),
                )

        return len(records)

    def get_stock_data(
        self,
        symbol: str,
        start: date | datetime | pd.Timestamp | str | None = None,
        end: date | datetime | pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve canonical OHLCV bars for `symbol`, optionally filtered by inclusive [start, end] dates.
        """
        norm_symbol = _normalize_symbol(symbol)
        start_str = _normalize_query_date(start, "start")
        end_str = _normalize_query_date(end, "end")

        if start_str is not None and end_str is not None and start_str > end_str:
            raise ValueError("start date must be before or equal to end date.")

        query = """
            SELECT trade_date, open, high, low, close, volume, dividends, stock_splits
            FROM daily_prices
            WHERE symbol = ?
        """
        params: list[str] = [norm_symbol]

        if start_str is not None:
            query += " AND trade_date >= ?"
            params.append(start_str)
        if end_str is not None:
            query += " AND trade_date <= ?"
            params.append(end_str)

        query += " ORDER BY trade_date ASC;"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return _rows_to_canonical_df(rows)

    def get_metadata(self, symbol: str) -> SymbolMetadata | None:
        """
        Retrieve ingestion metadata for `symbol`, or None if no metadata exists.
        """
        norm_symbol = _normalize_symbol(symbol)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT symbol, min_date, max_date, last_synced_at, has_full_history, row_count, last_refreshed_at
                FROM ingestion_metadata
                WHERE symbol = ?;
                """,
                (norm_symbol,),
            ).fetchone()

        if row is None:
            return None

        return SymbolMetadata(
            symbol=row[0],
            min_date=row[1],
            max_date=row[2],
            last_synced_at=row[3],
            has_full_history=bool(row[4]),
            row_count=int(row[5]),
            last_refreshed_at=row[6],
        )

    def get_latest_row(self, symbol: str) -> pd.DataFrame:
        """
        Retrieve the single most recent daily bar for `symbol` as a 1-row canonical DataFrame,
        or an empty canonical DataFrame if no data exists.
        """
        norm_symbol = _normalize_symbol(symbol)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT trade_date, open, high, low, close, volume, dividends, stock_splits
                FROM daily_prices
                WHERE symbol = ?
                ORDER BY trade_date DESC
                LIMIT 1;
                """,
                (norm_symbol,),
            ).fetchall()

        return _rows_to_canonical_df(rows)
