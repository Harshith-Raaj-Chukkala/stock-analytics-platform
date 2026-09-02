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
    running_peak = data["Close"].cummax()
    drawdown = (data["Close"] - running_peak) / running_peak * 100
    max_drawdown = drawdown.min()
    

    if daily_returns.empty:
     average_daily_return = None
    else:
     average_daily_return = daily_returns.mean()

    

    gains = data["Daily Return"].clip(lower=0)

    losses = data["Daily Return"].clip(upper=0).abs()
    average_gain = gains.rolling(14).mean()      
    average_loss = losses.rolling(14).mean()

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
    (average_loss == 0) & (average_gain > 0),
    100
)

    rsi = rsi.mask(
    (average_loss == 0) & (average_gain == 0),
    50
)


    data["RSI"] = rsi
    valid_rsi = data["RSI"].dropna()

    if valid_rsi.empty:

      latest_rsi = None

      rsi_interpretation = None
    else: 
        latest_rsi = data["RSI"].dropna().iloc[-1]
        if latest_rsi > 70:
         rsi_interpretation = "Very strong upward momentum"

        elif latest_rsi > 55:

         rsi_interpretation = "Strong upward momentum"

        elif latest_rsi >= 45:

         rsi_interpretation = "Neutral momentum"

        elif latest_rsi >= 30:

         rsi_interpretation = "Weak downward momentum"

        else:

         rsi_interpretation = "Strong downward momentum"


    total_return = (last_close - first_close) / first_close * 100
    if len(daily_returns) < 2:

      volatility = None

    else:

     volatility = daily_returns.std()

    print(gains)
    print(losses)
    print("AVERAGE GAIN:")
    print(average_gain)

    print("AVERAGE LOSS:")
    print(average_loss)
    print("RS:")
    print(rs)
    print("RSI:")
    print(rsi)
    
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
    ),
    "max_drawdown": (
        round(float(max_drawdown), 2)
        if max_drawdown is not None
        else None
    ),
   "latest_rsi": (
    round(float(latest_rsi), 2)
    if latest_rsi is not None
    else None
),
"rsi_interpretation": (
    rsi_interpretation
    if rsi_interpretation is not None
    else None
),
}

