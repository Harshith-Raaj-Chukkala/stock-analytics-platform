from fastapi import FastAPI , HTTPException , Query
from app.services.stock_service import download_stock_data, calculate_daily_returns
from app.services.portfolio_service import Portfolio, calculate_current_value, calculate_profit_loss, calculate_profit_loss_percentage, calculate_total_current_value, calculate_total_investment, calculate_total_profit_loss, calculate_total_profit_loss_percentage, calculate_transaction_cost

import numpy as np
from app.services.analytics_service import calculate_summary, compare_stock_summaries , compare_summaries 
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

    print("NEW CODE RUNNING")

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
        data["Moving Average 20"] = data["Close"].rolling(20).mean()
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

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
      print("ACTUAL ERROR:", repr(e))
      raise

from datetime import date
@app.get("/compare")
def compare_stocks(
    symbol: str, 
    start_a: date,
    end_a: date,
    start_b: date,
    end_b: date
): 
    if start_a > end_a:
     raise HTTPException(
        status_code=400,
        detail="Period A start date must be before or equal to the end date."
    )

    if start_b > end_b:
     raise HTTPException(
        status_code=400,
        detail="Period B start date must be before or equal to the end date."
    )
    
    if start_a == start_b and end_a == end_b:
     raise HTTPException(
        status_code=400,
        detail="Period A and Period B cannot be identical."
    )

    try:
        data_a = download_stock_data(symbol, start=start_a, end=end_a)
        data_b = download_stock_data(symbol, start=start_b, end=end_b)

        if data_a.empty:
         raise HTTPException(
        status_code=404,
        detail="No trading data available for Period A."
    )

        if data_b.empty:
         raise HTTPException(
        status_code=404,
        detail="No trading data available for Period B."
    )
        data_a = calculate_daily_returns(data_a)
        data_b = calculate_daily_returns(data_b)

        summary_a = calculate_summary(data_a)
        summary_b = calculate_summary(data_b)

        comparison = compare_summaries(summary_a, summary_b)

        return {
            "stock": {
                "symbol": symbol
            },
            "period_a": {
                "start": start_a,
                "end": end_a,
                "summary": summary_a
            },
            "period_b": {
                "start": start_b,
                "end": end_b,
                "summary": summary_b
            },
            "comparison": comparison
        }

    except HTTPException:
        raise

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
      print("ACTUAL ERROR:", repr(e))
      raise

@app.get("/compare-stocks")
def compare_multiple_stocks(
    symbols: list[str] = Query(...),
    start: date = Query(...),
    end: date = Query(...)
):
    if len(symbols) < 2 or len(symbols) > 5:
        raise HTTPException(
            status_code=400,
            detail="Stock comparison requires between 2 and 5 stocks."
        )

    summaries = {}

    for symbol in symbols:
       data = download_stock_data(symbol, start=start, end=end)
       if data.empty:
         raise HTTPException(
        status_code=404,
        detail=f"No trading data available for {symbol}."
    )
       data = calculate_daily_returns(data)
       summary = calculate_summary(data)

       summaries[symbol] = summary 

    comparison = compare_stock_summaries(summaries)
    return {
    "stocks": summaries,
    "comparison": comparison
}

@app.post("/portfolio")
def calculate_portfolio(portfolio: Portfolio):
     transaction_costs=[]

     for transaction in portfolio.transactions:
         cost = calculate_transaction_cost(transaction)
         transaction_costs.append(cost)

     total_invested = calculate_total_investment(portfolio)

     current_values = []
     for transaction in portfolio.transactions:
         current_value = calculate_current_value(transaction)
         current_values.append(current_value)

     profit_losses = []
     for transaction in portfolio.transactions:
         profit_loss = calculate_profit_loss(transaction)
         profit_losses.append(profit_loss)

     profit_loss_percentages = []
     for transaction in portfolio.transactions:
        profit_loss_percentage = calculate_profit_loss_percentage(transaction)
        profit_loss_percentages.append(profit_loss_percentage)

     total_current_value = calculate_total_current_value(portfolio)

     total_profit_loss = calculate_total_profit_loss(portfolio)

     total_profit_loss_percentage = calculate_total_profit_loss_percentage(portfolio)

     return {
    "transaction_costs": [round(x, 2) for x in transaction_costs],
    "total_invested": round(total_invested, 2),
    "current_values": [round(x, 2) for x in current_values],
    "profit_losses": [round(x, 2) for x in profit_losses],
    "profit_loss_percentages": [round(x, 2) for x in profit_loss_percentages],
    "total_current_value": round(total_current_value, 2),
    "total_profit_loss": round(total_profit_loss, 2),
    "total_profit_loss_percentage": round(total_profit_loss_percentage, 2)
}

