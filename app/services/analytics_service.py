from numpy import average
from yfinance import data


def calculate_summary(data):
    latest_price = data["Close"].iloc[-1]
    highest_price = data["Close"].max()
    lowest_price = data["Close"].min()
    average_price = data["Close"].mean()
    highest_volume = data["Volume"].max()
    first_close = data["Close"].iloc[0]
    last_close = data["Close"].iloc[-1]
    daily_returns = data["Daily Return"].dropna()

    if daily_returns.empty:
     average_daily_return = None
    else:
     average_daily_return = daily_returns.mean()

    if daily_returns.empty:

      volatility = None

    else:

      volatility = daily_returns.std()

    total_return = (last_close - first_close) / first_close * 100

    return {
    "latest_price": float(latest_price),
    "highest_price": float(highest_price),
    "lowest_price": float(lowest_price),
    "average_close": float(average_price),
    "highest_volume": int(highest_volume),
    "total_return_percent": round(float(total_return), 2),
    "average_daily_return": (
        round(float(average_daily_return), 2)
        if average_daily_return is not None
        else None
    ),
    "volatility": (
        round(float(volatility), 2)
        if volatility is not None
        else None
    )
}
 