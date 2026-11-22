import logging
import os
from pathlib import Path
import boto3
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from app.services.data_validator import validate_and_normalize_stock_data
from app.services.market_data_repository import (
    DEFAULT_DB_PATH,
    MarketDataRepository,
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

_repository: MarketDataRepository | None = None
_repository_path: Path | None = None


def _get_repository() -> MarketDataRepository:
    global _repository, _repository_path
    target_path = Path(DB_PATH).resolve()
    if _repository is None or _repository_path != target_path:
        _repository = MarketDataRepository(db_path=target_path)
        _repository_path = target_path
    return _repository


def download_stock_data(symbol, period="1mo", start=None, end=None):
 ticker = yf.Ticker(symbol)
 if start and end:
    data = ticker.history(start=start, end=end)
 else:

    data = ticker.history(period=period)

 data = validate_and_normalize_stock_data(data, symbol=symbol)

 if data.empty: 
     return data

 mark_full_history = (not (start and end)) and (
     isinstance(period, str) and period.strip().lower() == "max"
 )
 repo = _get_repository()
 repo.upsert_stock_data(
     symbol=symbol,
     df=data,
     mark_full_history=mark_full_history,
 )
 
 os.makedirs("data", exist_ok=True)
 filename = f"data/{symbol}.csv"
 data.to_csv(filename)

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
     

 