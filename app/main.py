from fastapi import FastAPI , HTTPException
from app.services.stock_service import download_stock_data, calculate_daily_returns

import numpy as np
from app.services.analytics_service import calculate_summary
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
def get_stock(
    symbol: str,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None
):

    if period and period not in allowed_periods:
        raise HTTPException(
            status_code=400,
            detail="Invalid period"
        )

    if (start and not end) or (end and not start):
        raise HTTPException(
            status_code=400,
            detail="Both start and end dates are required."
        )

    if period and start and end:
        raise HTTPException(
            status_code=400,
            detail="Use either period OR start/end dates, not both."
        )

    if not period and not start and not end:
        period = "1mo"

        #Date format: YYYY-MM-DD (ISO 8601)

    try:
        data = download_stock_data(
            symbol=symbol,
            period=period,
            start=start,
            end=end
        )
        print("DATA RECEIVED")

        print(data)
        
        if data.empty:
         raise HTTPException(
         status_code=404,
         detail="No trading data available for the requested date range."
    )
        print(data.tail())
        
        data = calculate_daily_returns(data)
        data = data.dropna(subset=["Close"])
        summary = calculate_summary(data)

        data = data.replace({np.nan: None})
        history = data.to_dict(orient="records")

        return {
    "stock": {
        "symbol": symbol
    },
    "summary": summary,
    "history": history
 }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Invalid stock symbol"
        )
  