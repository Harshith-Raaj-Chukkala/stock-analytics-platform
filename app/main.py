from fastapi import FastAPI , HTTPException
from app.services.stock_service import download_stock_data
from app.services.stock_service import calculate_daily_returns
import numpy as np
allowed_periods = [
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "max"
]
app = FastAPI()

@app.get("/stock")

def get_stock(symbol: str , period: str = "1mo"):

    if period not in allowed_periods:
     raise HTTPException(
        status_code=400,
        detail="Invalid period"
    )

    try:
     

        data = download_stock_data(symbol, period)

        data = calculate_daily_returns(data)

        latest_day = data.iloc[-1]
        latest_price = latest_day["Close"]
        data = data.replace({np.nan: None})

        history = data.to_dict(orient="records")

        return {
            "stock": {
                "symbol": symbol,
                "latest_price": latest_price
            },
            "history": history
        }

    except Exception:

        raise HTTPException(

            status_code=404,

            detail="Invalid stock symbol"

        )