from datetime import date, datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import re
import threading
import boto3
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from app.services.data_validator import validate_and_normalize_stock_data
from app.services.market_data_repository import (
    DEFAULT_DB_PATH,
    MarketDataRepository,
    _normalize_query_date,
)


load_dotenv()

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
S3_SYNC_ENABLED = os.getenv("S3_SYNC_ENABLED", "false").strip().lower() in (
    "true",
    "1",
    "yes",
)
DB_PATH = os.getenv("MARKET_DATA_DB_PATH", str(DEFAULT_DB_PATH))
MAX_SELECTIVE_GAPS = 3

_SYMBOL_PATTERN = re.compile(r"^\^?[A-Z0-9][A-Z0-9.&=\-]{0,29}$")

_repository: MarketDataRepository | None = None
_repository_path: Path | None = None
_repository_lock = threading.Lock()


def _get_repository() -> MarketDataRepository:
    global _repository, _repository_path
    with _repository_lock:
        target_path = Path(DB_PATH).resolve()
        if _repository is None or _repository_path != target_path:
            _repository = MarketDataRepository(db_path=target_path)
            _repository_path = target_path
        return _repository


def _validate_and_normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Symbol must be a non-empty string.")
    norm_symbol = symbol.strip().upper()
    if (
        ".." in norm_symbol
        or "/" in norm_symbol
        or "\\" in norm_symbol
        or not _SYMBOL_PATTERN.match(norm_symbol)
    ):
        raise ValueError(f"Invalid or unsafe stock symbol: {symbol!r}")
    return norm_symbol


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _last_completed_date() -> str:
    return (_utc_now().date() - timedelta(days=1)).isoformat()


def download_stock_data(symbol, period="1mo", start=None, end=None):
 norm_symbol = _validate_and_normalize_symbol(symbol)

 if (start is None) != (end is None):
    raise ValueError("Both start and end dates must be provided together.")

 if start is not None and end is not None:
    req_start = _normalize_query_date(start, "start")
    req_end = _normalize_query_date(end, "end")
    if req_start > req_end:
        raise ValueError("start date must be before or equal to end date.")

    repo = _get_repository()
    missing_gaps = repo.get_missing_ranges(norm_symbol, req_start, req_end)
    if not missing_gaps:
        return repo.get_stock_data(norm_symbol, start=req_start, end=req_end)

    if len(missing_gaps) > MAX_SELECTIVE_GAPS:
        fetch_gaps = [(missing_gaps[0][0], missing_gaps[-1][1])]
    else:
        fetch_gaps = missing_gaps

    last_completed = _last_completed_date()
    ticker = yf.Ticker(norm_symbol)
    persisted_any = False

    for gap_start, gap_end in fetch_gaps:
        yf_end = (date.fromisoformat(gap_end) + timedelta(days=1)).isoformat()
        gap_data = ticker.history(start=gap_start, end=yf_end)
        gap_data = validate_and_normalize_stock_data(gap_data, symbol=norm_symbol)
        gap_data = gap_data.loc[
            (gap_data.index >= pd.Timestamp(gap_start))
            & (gap_data.index <= pd.Timestamp(gap_end))
        ].copy()

        if gap_data.empty:
            continue

        completed_end = min(gap_end, last_completed)
        if completed_end < gap_start:
            continue

        gap_data = gap_data.loc[
            gap_data.index <= pd.Timestamp(completed_end)
        ].copy()

        if gap_data.empty:
            continue

        gap_data.index = gap_data.index.normalize()
        gap_data.index.name = "Date"

        qualifies_as_refresh = gap_end >= last_completed

        repo.upsert_stock_data(
            symbol=norm_symbol,
            df=gap_data,
            mark_full_history=False,
            synced_at=_utc_now(),
            coverage_start=gap_start,
            coverage_end=completed_end,
            mark_refreshed=qualifies_as_refresh,
        )
        persisted_any = True

    data = repo.get_stock_data(norm_symbol, start=req_start, end=req_end)
    if not persisted_any or data.empty:
        return data
 else:
    ticker = yf.Ticker(norm_symbol)
    data = ticker.history(period=period)
    data = validate_and_normalize_stock_data(data, symbol=norm_symbol)

    if data.empty:
        return data

    mark_full_history = (
        isinstance(period, str) and period.strip().lower() == "max"
    )
    repo = _get_repository()
    repo.upsert_stock_data(
        symbol=norm_symbol,
        df=data,
        mark_full_history=mark_full_history,
        synced_at=_utc_now(),
        mark_refreshed=True,
    )

 data_dir = Path("data").resolve()
 data_dir.mkdir(parents=True, exist_ok=True)
 csv_path = (data_dir / f"{norm_symbol}.csv").resolve()
 if csv_path.parent != data_dir:
     raise ValueError(f"CSV path escapes data directory for symbol: {symbol!r}")

 filename = f"data/{norm_symbol}.csv"
 data.to_csv(csv_path)

 if S3_SYNC_ENABLED:
     if not AWS_BUCKET_NAME:
         logger.warning("S3_SYNC_ENABLED is true, but AWS_BUCKET_NAME is not set. Skipping S3 upload for %s.", filename)
     else:
         try:
             s3 = boto3.client("s3", region_name=AWS_REGION)
             s3.upload_file(filename, AWS_BUCKET_NAME, filename)
         except Exception as exc:
             logger.warning("S3 upload failed for %s: %s", filename, exc)

 return data 
 
def calculate_daily_returns (data): 
     data["Daily Return"] = data["Close"].pct_change() * 100
     return data 
     

 