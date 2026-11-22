import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class StartAfterEndAcceptedError(AssertionError):
    """Raised when an endpoint returns 200 instead of 400 for start > end."""


def make_raw_ohlcv_df(closes: list[float], volumes: list[int] | None = None) -> pd.DataFrame:
    """Create a synthetic raw OHLCV DataFrame matching yfinance Ticker.history() output."""
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
    return df


def test_get_stock_success_with_default_and_explicit_period(monkeypatch):
    recorded_calls = []

    def fake_download(symbol, period="1mo", start=None, end=None):
        recorded_calls.append(
            {"symbol": symbol, "period": period, "start": start, "end": end}
        )
        return make_raw_ohlcv_df([100.0, 105.0, 102.0, 108.0, 110.0])

    monkeypatch.setattr("app.main.download_stock_data", fake_download)

    # Omit period -> defaults to "1mo"
    response = client.get("/stock", params={"symbol": "RELIANCE.NS"})

    assert response.status_code == 200
    assert recorded_calls[-1] == {
        "symbol": "RELIANCE.NS",
        "period": "1mo",
        "start": None,
        "end": None,
    }

    body = response.json()
    assert body["stock"] == {"symbol": "RELIANCE.NS"}
    assert body["summary"]["latest_price"] == 110.0
    assert body["summary"]["total_return_percent"] == 10.0
    assert len(body["history"]) == 5
    # First row's Daily Return is NaN in Pandas and must be serialized as None (null in JSON)
    assert body["history"][0]["Daily Return"] is None
    assert body["history"][1]["Daily Return"] == pytest.approx(5.0)


def test_get_stock_validation_errors(monkeypatch):
    def should_not_be_called(*args, **kwargs):
        pytest.fail("download_stock_data should not be called on validation error")

    monkeypatch.setattr("app.main.download_stock_data", should_not_be_called)

    # Invalid period
    res_invalid_period = client.get(
        "/stock", params={"symbol": "RELIANCE.NS", "period": "2mo"}
    )
    assert res_invalid_period.status_code == 400
    assert res_invalid_period.json()["detail"] == "Invalid period"

    # Missing end date when start date is supplied
    res_missing_end = client.get(
        "/stock", params={"symbol": "RELIANCE.NS", "start": "2025-01-01"}
    )
    assert res_missing_end.status_code == 400
    assert (
        res_missing_end.json()["detail"]
        == "Both start and end dates are required."
    )

    # Both period and start/end dates supplied
    res_both = client.get(
        "/stock",
        params={
            "symbol": "RELIANCE.NS",
            "period": "1mo",
            "start": "2025-01-01",
            "end": "2025-01-31",
        },
    )
    assert res_both.status_code == 400
    assert (
        res_both.json()["detail"]
        == "Use either period OR start/end dates, not both."
    )


def test_get_stock_empty_data_returns_404(monkeypatch):
    monkeypatch.setattr(
        "app.main.download_stock_data",
        lambda symbol, period="1mo", start=None, end=None: pd.DataFrame(),
    )

    response = client.get(
        "/stock", params={"symbol": "UNKNOWN.NS", "period": "1mo"}
    )
    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "No trading data available for the requested date range."
    )


def test_post_portfolio_success_and_symbol_aggregation(monkeypatch):
    fetched_symbols = []
    price_map = {
        "TCS.NS": 120.0,
        "INFY.NS": 180.0,
    }

    def fake_download(symbol, period="1mo", start=None, end=None):
        fetched_symbols.append((symbol, period))
        return make_raw_ohlcv_df([price_map[symbol]])

    monkeypatch.setattr("app.main.download_stock_data", fake_download)

    # Two transactions for TCS.NS and one for INFY.NS:
    # Tx 1 (TCS.NS): 5 @ 100 + 0 = 500 -> current 600 (+100, +20%, weight 12.5%, contrib +2.5%)
    # Tx 2 (TCS.NS): 5 @ 100 + 0 = 500 -> current 600 (+100, +20%, weight 12.5%, contrib +2.5%)
    # Tx 3 (INFY.NS): 15 @ 200 + 0 = 3000 -> current 2700 (-300, -10%, weight 75.0%, contrib -7.5%)
    # Total invested: 4000, Total current: 3900, Total P&L: -100 (-2.5%)
    payload = {
        "transactions": [
            {
                "symbol": "TCS.NS",
                "quantity": 5,
                "purchase_price": 100.0,
                "purchase_date": "2025-01-01",
                "charges": 0.0,
            },
            {
                "symbol": "TCS.NS",
                "quantity": 5,
                "purchase_price": 100.0,
                "purchase_date": "2025-02-01",
                "charges": 0.0,
            },
            {
                "symbol": "INFY.NS",
                "quantity": 15,
                "purchase_price": 200.0,
                "purchase_date": "2025-02-15",
                "charges": 0.0,
            },
        ]
    }

    response = client.post("/portfolio", json=payload)
    assert response.status_code == 200

    # Each unique symbol is downloaded once with period="1d"
    assert fetched_symbols == [("TCS.NS", "1d"), ("INFY.NS", "1d")]

    body = response.json()
    assert body["transaction_costs"] == [500.0, 500.0, 3000.0]
    assert body["total_invested"] == 4000.0
    assert body["current_values"] == [600.0, 600.0, 2700.0]
    assert body["profit_losses"] == [100.0, 100.0, -300.0]
    assert body["profit_loss_percentages"] == [20.0, 20.0, -10.0]
    assert body["total_current_value"] == 3900.0
    assert body["total_profit_loss"] == -100.0
    assert body["total_profit_loss_percentage"] == -2.5
    assert body["initial_weights"] == [12.5, 12.5, 75.0]
    assert body["current_weights"] == [15.38, 15.38, 69.23]
    # Contributions are aggregated by symbol across transactions
    assert body["contributions"] == {"TCS.NS": 5.0, "INFY.NS": -7.5}


def test_post_portfolio_validation_error_returns_422(monkeypatch):
    def should_not_be_called(*args, **kwargs):
        pytest.fail("download_stock_data should not be called on invalid payload")

    monkeypatch.setattr("app.main.download_stock_data", should_not_be_called)

    response = client.post("/portfolio", json={"transactions": []})
    assert response.status_code == 422


@pytest.mark.xfail(
    raises=IndexError,
    strict=True,
    reason="POST /portfolio indexes data['Close'].iloc[-1] without checking if data.empty",
)
def test_post_portfolio_empty_stock_data_for_invalid_symbol(monkeypatch):
    monkeypatch.setattr(
        "app.main.download_stock_data",
        lambda symbol, period="1d", start=None, end=None: pd.DataFrame(
            {"Close": pd.Series(dtype=float)}
        ),
    )

    payload = {
        "transactions": [
            {
                "symbol": "INVALID.NS",
                "quantity": 10,
                "purchase_price": 100.0,
                "purchase_date": "2025-01-01",
                "charges": 0.0,
            }
        ]
    }

    response = client.post("/portfolio", json=payload)
    assert response.status_code == 404


def test_get_stock_custom_date_range_and_value_error_handling(monkeypatch):
    recorded_calls = []

    def fake_download(symbol, period="1mo", start=None, end=None):
        recorded_calls.append(
            {"symbol": symbol, "period": period, "start": start, "end": end}
        )
        return make_raw_ohlcv_df([200.0, 210.0, 220.0])

    monkeypatch.setattr("app.main.download_stock_data", fake_download)

    response = client.get(
        "/stock",
        params={
            "symbol": "TCS.NS",
            "start": "2025-01-01",
            "end": "2025-01-15",
        },
    )
    assert response.status_code == 200
    assert recorded_calls[-1] == {
        "symbol": "TCS.NS",
        "period": None,
        "start": "2025-01-01",
        "end": "2025-01-15",
    }
    assert response.json()["summary"]["total_return_percent"] == 10.0

    # ValueError raised by download_stock_data is mapped to HTTP 400
    def raise_value_error(*args, **kwargs):
        raise ValueError("Invalid date format supplied")

    monkeypatch.setattr("app.main.download_stock_data", raise_value_error)
    err_response = client.get(
        "/stock",
        params={"symbol": "TCS.NS", "start": "bad-date", "end": "2025-01-15"},
    )
    assert err_response.status_code == 400
    assert err_response.json()["detail"] == "Invalid date format supplied"


@pytest.mark.xfail(
    raises=StartAfterEndAcceptedError,
    strict=True,
    reason="GET /stock does not validate that start <= end before calling download_stock_data",
)
def test_get_stock_start_after_end_returns_400(monkeypatch):
    monkeypatch.setattr(
        "app.main.download_stock_data",
        lambda *args, **kwargs: make_raw_ohlcv_df([100.0, 105.0]),
    )

    response = client.get(
        "/stock",
        params={
            "symbol": "RELIANCE.NS",
            "start": "2025-06-01",
            "end": "2025-01-01",
        },
    )
    if response.status_code == 200:
        raise StartAfterEndAcceptedError("GET /stock returned 200 for start > end")
    assert response.status_code == 400


def test_get_compare_success(monkeypatch):
    # Period A: 100 -> 110 (+10% total return)
    # Period B: 100 -> 125 (+25% total return)
    # Comparison (Period B - Period A): +15.0% total_return_difference
    def fake_download(symbol, period="1mo", start=None, end=None):
        if str(start) == "2025-01-01":
            return make_raw_ohlcv_df([100.0, 105.0, 110.0])
        return make_raw_ohlcv_df([100.0, 110.0, 125.0])

    monkeypatch.setattr("app.main.download_stock_data", fake_download)

    response = client.get(
        "/compare",
        params={
            "symbol": "RELIANCE.NS",
            "start_a": "2025-01-01",
            "end_a": "2025-01-31",
            "start_b": "2025-02-01",
            "end_b": "2025-02-28",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stock"] == {"symbol": "RELIANCE.NS"}
    assert body["period_a"]["start"] == "2025-01-01"
    assert body["period_a"]["end"] == "2025-01-31"
    assert body["period_a"]["summary"]["total_return_percent"] == 10.0
    assert body["period_b"]["start"] == "2025-02-01"
    assert body["period_b"]["end"] == "2025-02-28"
    assert body["period_b"]["summary"]["total_return_percent"] == 25.0
    assert body["comparison"]["total_return_difference"] == 15.0


def test_get_compare_date_validation_errors(monkeypatch):
    def should_not_be_called(*args, **kwargs):
        pytest.fail("download_stock_data should not be called when date validation fails")

    monkeypatch.setattr("app.main.download_stock_data", should_not_be_called)

    # start_a > end_a
    res_a = client.get(
        "/compare",
        params={
            "symbol": "RELIANCE.NS",
            "start_a": "2025-03-01",
            "end_a": "2025-02-01",
            "start_b": "2025-01-01",
            "end_b": "2025-01-31",
        },
    )
    assert res_a.status_code == 400
    assert (
        res_a.json()["detail"]
        == "Period A start date must be before or equal to the end date."
    )

    # start_b > end_b
    res_b = client.get(
        "/compare",
        params={
            "symbol": "RELIANCE.NS",
            "start_a": "2025-01-01",
            "end_a": "2025-01-31",
            "start_b": "2025-04-01",
            "end_b": "2025-03-01",
        },
    )
    assert res_b.status_code == 400
    assert (
        res_b.json()["detail"]
        == "Period B start date must be before or equal to the end date."
    )

    # Identical Period A and Period B
    res_identical = client.get(
        "/compare",
        params={
            "symbol": "RELIANCE.NS",
            "start_a": "2025-01-01",
            "end_a": "2025-01-31",
            "start_b": "2025-01-01",
            "end_b": "2025-01-31",
        },
    )
    assert res_identical.status_code == 400
    assert (
        res_identical.json()["detail"]
        == "Period A and Period B cannot be identical."
    )


def test_get_compare_empty_data_for_period_a_or_b_returns_404(monkeypatch):
    # Period A empty
    monkeypatch.setattr(
        "app.main.download_stock_data",
        lambda symbol, period="1mo", start=None, end=None: (
            pd.DataFrame()
            if str(start) == "2025-01-01"
            else make_raw_ohlcv_df([100.0, 110.0])
        ),
    )
    res_a_empty = client.get(
        "/compare",
        params={
            "symbol": "RELIANCE.NS",
            "start_a": "2025-01-01",
            "end_a": "2025-01-31",
            "start_b": "2025-02-01",
            "end_b": "2025-02-28",
        },
    )
    assert res_a_empty.status_code == 404
    assert res_a_empty.json()["detail"] == "No trading data available for Period A."

    # Period A non-empty, Period B empty
    monkeypatch.setattr(
        "app.main.download_stock_data",
        lambda symbol, period="1mo", start=None, end=None: (
            make_raw_ohlcv_df([100.0, 110.0])
            if str(start) == "2025-01-01"
            else pd.DataFrame()
        ),
    )
    res_b_empty = client.get(
        "/compare",
        params={
            "symbol": "RELIANCE.NS",
            "start_a": "2025-01-01",
            "end_a": "2025-01-31",
            "start_b": "2025-02-01",
            "end_b": "2025-02-28",
        },
    )
    assert res_b_empty.status_code == 404
    assert res_b_empty.json()["detail"] == "No trading data available for Period B."


def test_get_compare_stocks_success(monkeypatch):
    series_by_symbol = {
        # +10% return, +5.0% and +4.76% daily returns (avg +4.88%, low volatility, 0% drawdown)
        "TCS.NS": [100.0, 105.0, 110.0],
        # +30% return, dip from 120 to 108 (-10% drawdown) then up to 130 (high volatility)
        "INFY.NS": [100.0, 120.0, 108.0, 130.0],
    }

    monkeypatch.setattr(
        "app.main.download_stock_data",
        lambda symbol, period="1mo", start=None, end=None: make_raw_ohlcv_df(
            series_by_symbol[symbol]
        ),
    )

    response = client.get(
        "/compare-stocks",
        params={
            "symbols": ["TCS.NS", "INFY.NS"],
            "start": "2025-01-01",
            "end": "2025-01-31",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body["stocks"].keys()) == {"TCS.NS", "INFY.NS"}
    assert body["stocks"]["TCS.NS"]["total_return_percent"] == 10.0
    assert body["stocks"]["INFY.NS"]["total_return_percent"] == 30.0

    comparison = body["comparison"]
    assert comparison["highest_return"] == "INFY.NS"
    assert comparison["highest_average_daily_return"] == "INFY.NS"
    assert comparison["lowest_volatility"] == "TCS.NS"
    assert comparison["best_drawdown"] == "TCS.NS"
    assert comparison["interpretation"] == (
        "INFY.NS had the highest return, while TCS.NS had the lowest volatility "
        "and TCS.NS had the best drawdown."
    )


def test_get_compare_stocks_symbol_count_validation_and_empty_data(monkeypatch):
    def should_not_be_called(*args, **kwargs):
        pytest.fail("download_stock_data should not be called on invalid symbol count")

    monkeypatch.setattr("app.main.download_stock_data", should_not_be_called)

    # Fewer than 2 symbols -> 400
    res_too_few = client.get(
        "/compare-stocks",
        params={
            "symbols": ["TCS.NS"],
            "start": "2025-01-01",
            "end": "2025-01-31",
        },
    )
    assert res_too_few.status_code == 400
    assert (
        res_too_few.json()["detail"]
        == "Stock comparison requires between 2 and 5 stocks."
    )

    # More than 5 symbols -> 400
    res_too_many = client.get(
        "/compare-stocks",
        params={
            "symbols": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "start": "2025-01-01",
            "end": "2025-01-31",
        },
    )
    assert res_too_many.status_code == 400
    assert (
        res_too_many.json()["detail"]
        == "Stock comparison requires between 2 and 5 stocks."
    )

    # Second symbol has empty data -> 404 identifying that symbol
    monkeypatch.setattr(
        "app.main.download_stock_data",
        lambda symbol, period="1mo", start=None, end=None: (
            make_raw_ohlcv_df([100.0, 105.0])
            if symbol == "TCS.NS"
            else pd.DataFrame()
        ),
    )
    res_empty_symbol = client.get(
        "/compare-stocks",
        params={
            "symbols": ["TCS.NS", "INVALID.NS"],
            "start": "2025-01-01",
            "end": "2025-01-31",
        },
    )
    assert res_empty_symbol.status_code == 404
    assert (
        res_empty_symbol.json()["detail"]
        == "No trading data available for INVALID.NS."
    )


@pytest.mark.xfail(
    raises=StartAfterEndAcceptedError,
    strict=True,
    reason="GET /compare-stocks does not validate that start <= end before calling download_stock_data",
)
def test_get_compare_stocks_start_after_end_returns_400(monkeypatch):
    monkeypatch.setattr(
        "app.main.download_stock_data",
        lambda *args, **kwargs: make_raw_ohlcv_df([100.0, 105.0]),
    )

    response = client.get(
        "/compare-stocks",
        params={
            "symbols": ["TCS.NS", "INFY.NS"],
            "start": "2025-06-01",
            "end": "2025-01-01",
        },
    )
    if response.status_code == 200:
        raise StartAfterEndAcceptedError(
            "GET /compare-stocks returned 200 for start > end"
        )
    assert response.status_code == 400

