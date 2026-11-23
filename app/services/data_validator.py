import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
OPTIONAL_COLUMNS = ("Dividends", "Stock Splits")
CANONICAL_COLUMNS = (*REQUIRED_OHLCV_COLUMNS, *OPTIONAL_COLUMNS)
OHLC_TOLERANCE = 1e-6


def _empty_canonical_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": pd.Series(dtype="float64"),
            "High": pd.Series(dtype="float64"),
            "Low": pd.Series(dtype="float64"),
            "Close": pd.Series(dtype="float64"),
            "Volume": pd.Series(dtype="int64"),
            "Dividends": pd.Series(dtype="float64"),
            "Stock Splits": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], name="Date"),
    )


_BOOL_LIKE = (bool, np.bool_)
_INT64_MAX = int(np.iinfo(np.int64).max)
_FLOAT64_INT64_EXCLUSIVE_UPPER = float(2**63)


def _has_bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(True, index=series.index)
    if series.dtype == object:
        return series.map(lambda v: isinstance(v, _BOOL_LIKE))
    return pd.Series(False, index=series.index)


def _parse_local_date(value) -> pd.Timestamp:
    if value is None or isinstance(
        value, (bool, int, float, np.bool_, np.integer, np.floating)
    ):
        return pd.NaT
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return pd.NaT
        try:
            float(stripped)
            return pd.NaT
        except ValueError:
            pass
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _normalize_date_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        if index.tz is not None:
            normalized = index.tz_localize(None).normalize()
        else:
            normalized = index.normalize()
        normalized.name = "Date"
        return normalized

    parsed = [_parse_local_date(val) for val in index]
    return pd.DatetimeIndex(parsed, name="Date")


def validate_and_normalize_stock_data(
    raw_df: pd.DataFrame,
    symbol: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Validate, clean, and normalize a raw OHLCV DataFrame into the canonical
    schema expected by analytics and persistence layers.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if isinstance(raw_df.columns, pd.MultiIndex):
        raise ValueError("MultiIndex columns are not supported.")

    df = raw_df.copy()
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

    if pd.Index(df.columns).duplicated().any():
        raise ValueError("Duplicate column names are not supported.")

    if df.empty and len(df.columns) == 0:
        return _empty_canonical_df()

    if "Date" in df.columns:
        df = df.set_index("Date")

    if df.empty:
        return _empty_canonical_df()

    missing_columns = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    for optional_col in OPTIONAL_COLUMNS:
        if optional_col not in df.columns:
            df[optional_col] = 0.0

    df = df[list(CANONICAL_COLUMNS)].copy()

    # 1. Normalize dates while preserving exchange-local calendar dates
    df.index = _normalize_date_index(df.index)
    invalid_date_mask = df.index.isna()
    dropped_dates = int(invalid_date_mask.sum())
    if dropped_dates > 0:
        df = df.loc[~invalid_date_mask].copy()
        logger.warning(
            "Symbol %s: dropped %d row(s) due to invalid or missing dates (%d remaining).",
            symbol,
            dropped_dates,
            len(df),
        )
        if df.empty:
            return _empty_canonical_df()

    # 2. Coerce numeric columns and validate numeric constraints
    price_cols = ("Open", "High", "Low", "Close")
    valid_numeric_mask = pd.Series(True, index=df.index)

    for col in price_cols:
        bool_mask = _has_bool_mask(df[col])
        coerced = pd.to_numeric(df[col], errors="coerce")
        df[col] = coerced
        valid_numeric_mask &= (
            (~bool_mask) & coerced.notna() & np.isfinite(coerced) & (coerced > 0)
        )

    vol_bool_mask = _has_bool_mask(df["Volume"])
    vol_coerced = pd.to_numeric(df["Volume"], errors="coerce")
    df["Volume"] = vol_coerced
    if pd.api.types.is_integer_dtype(vol_coerced.dtype):
        valid_volume = (
            (~vol_bool_mask)
            & vol_coerced.notna()
            & (vol_coerced >= 0)
            & (vol_coerced <= _INT64_MAX)
        )
    else:
        valid_volume = (
            (~vol_bool_mask)
            & vol_coerced.notna()
            & np.isfinite(vol_coerced)
            & (vol_coerced >= 0)
            & (vol_coerced < _FLOAT64_INT64_EXCLUSIVE_UPPER)
            & (np.floor(vol_coerced) == vol_coerced)
        )
    valid_numeric_mask &= valid_volume

    for col in OPTIONAL_COLUMNS:
        bool_mask = _has_bool_mask(df[col])
        originally_missing = df[col].isna()
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Non-null values that failed numeric conversion are invalid rows
        invalid_unparseable = (~originally_missing) & coerced.isna()
        coerced = coerced.where(~originally_missing, 0.0)
        df[col] = coerced
        valid_numeric_mask &= (
            (~bool_mask)
            & (~invalid_unparseable)
            & coerced.notna()
            & np.isfinite(coerced)
            & (coerced >= 0)
        )

    dropped_numeric = int((~valid_numeric_mask).sum())
    if dropped_numeric > 0:
        df = df.loc[valid_numeric_mask].copy()
        logger.warning(
            "Symbol %s: dropped %d row(s) due to invalid numeric, price, volume, or corporate-action values (%d remaining).",
            symbol,
            dropped_numeric,
            len(df),
        )
        if df.empty:
            return _empty_canonical_df()

    # 3. Validate OHLC envelope (reject beyond tolerance; minimally correct within tolerance)
    oc_max = df[["Open", "Close"]].max(axis=1)
    oc_min = df[["Open", "Close"]].min(axis=1)

    ohlc_invalid_mask = (
        (df["High"] < df["Low"] - OHLC_TOLERANCE)
        | (df["High"] < oc_max - OHLC_TOLERANCE)
        | (df["Low"] > oc_min + OHLC_TOLERANCE)
    )
    dropped_ohlc = int(ohlc_invalid_mask.sum())
    if dropped_ohlc > 0:
        df = df.loc[~ohlc_invalid_mask].copy()
        logger.warning(
            "Symbol %s: dropped %d row(s) due to inconsistent OHLC envelope (%d remaining).",
            symbol,
            dropped_ohlc,
            len(df),
        )
        if df.empty:
            return _empty_canonical_df()

    oc_max = df[["Open", "Close"]].max(axis=1)
    oc_min = df[["Open", "Close"]].min(axis=1)
    needs_high_fix = df["High"] < oc_max
    needs_low_fix = df["Low"] > oc_min
    corrected_rows = int((needs_high_fix | needs_low_fix).sum())

    if corrected_rows > 0:
        df.loc[needs_high_fix, "High"] = oc_max.loc[needs_high_fix]
        df.loc[needs_low_fix, "Low"] = oc_min.loc[needs_low_fix]
        logger.warning(
            "Symbol %s: minimally corrected OHLC rounding discrepancies within tolerance on %d row(s) (%d total).",
            symbol,
            corrected_rows,
            len(df),
        )

    # 4. Sort chronologically (stable) and deduplicate keeping last occurrence
    df = df.sort_index(kind="mergesort")
    duplicate_mask = df.index.duplicated(keep="last")
    dropped_duplicates = int(duplicate_mask.sum())
    if dropped_duplicates > 0:
        df = df.loc[~duplicate_mask].copy()
        logger.warning(
            "Symbol %s: removed %d duplicate trading date row(s), keeping last occurrence (%d remaining).",
            symbol,
            dropped_duplicates,
            len(df),
        )

    # 5. Enforce canonical dtypes and column order
    for col in ("Open", "High", "Low", "Close", "Dividends", "Stock Splits"):
        df[col] = df[col].astype("float64")
    df["Volume"] = df["Volume"].astype("int64")
    df.index.name = "Date"

    return df[list(CANONICAL_COLUMNS)]
