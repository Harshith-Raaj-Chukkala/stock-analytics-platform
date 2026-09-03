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
    data["EMA 12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA 26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA 12"] - data["EMA 26"]
    data["Signal Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["Histogram"] = data["MACD"] - data["Signal Line"]
    

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
    
    valid_macd = data.dropna(subset=["MACD", "Signal Line"])
    if valid_macd.empty:
        latest_macd = None
        latest_signal = None
        latest_histogram = None
        macd_interpretation = None
    else:
        latest_macd = valid_macd["MACD"].iloc[-1]
        latest_signal = valid_macd["Signal Line"].iloc[-1]
        latest_histogram = valid_macd["Histogram"].iloc[-1]
        prev_histogram = valid_macd["Histogram"].iloc[-2] if len(valid_macd) > 1 else None

        if latest_macd > latest_signal:
            if prev_histogram is not None and prev_histogram < 0 and latest_histogram > 0:
                macd_interpretation = "Bullish crossover - momentum turning up"
            else:
                macd_interpretation = "Bullish - MACD above signal"
        elif latest_macd < latest_signal:
            if prev_histogram is not None and prev_histogram > 0 and latest_histogram < 0:
                macd_interpretation = "Bearish crossover - momentum turning down"
            else:
                macd_interpretation = "Bearish - MACD below signal"
        else:
            macd_interpretation = "Neutral - MACD equals signal"


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
    print("EMA 12:")
    print(data["EMA 12"])
    print("EMA 26:")
    print(data["EMA 26"])
    print("MACD:")
    print(data["MACD"])
    print("Signal Line:")
    print(data["Signal Line"])
    print("Histogram:")
    print(data["Histogram"])

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
"latest_macd": (
    round(float(latest_macd), 2)
    if latest_macd is not None
    else None
),
"latest_signal": (
    round(float(latest_signal), 2)
    if latest_signal is not None
    else None
),
"latest_histogram": (
    round(float(latest_histogram), 2)
    if latest_histogram is not None
    else None
),
"macd_interpretation": (
    macd_interpretation
    if macd_interpretation is not None
    else None
)

}

