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

    
    data["Standard Deviation 20"] = (
        data["Close"].rolling(window=20).std()
    )

    data["Upper Band"] = (
        data["Moving Average 20"]
        + (data["Standard Deviation 20"] * 2)
    )

    data["Lower Band"] = (
        data["Moving Average 20"]
        - (data["Standard Deviation 20"] * 2)
    )

    data["Band Width"] = (
        data["Upper Band"] - data["Lower Band"]
    )

   
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
        latest_rsi = valid_rsi.iloc[-1]

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

    
    valid_macd = data.dropna(
        subset=["MACD", "Signal Line"]
    )

    if valid_macd.empty:
        latest_macd = None
        latest_signal = None
        latest_histogram = None
        macd_interpretation = None
    else:
        latest_macd = valid_macd["MACD"].iloc[-1]
        latest_signal = valid_macd["Signal Line"].iloc[-1]
        latest_histogram = valid_macd["Histogram"].iloc[-1]

        prev_histogram = (
            valid_macd["Histogram"].iloc[-2]
            if len(valid_macd) > 1
            else None
        )

        if latest_macd > latest_signal:
            if (
                prev_histogram is not None
                and prev_histogram < 0
                and latest_histogram > 0
            ):
                macd_interpretation = (
                    "Bullish crossover - momentum turning up"
                )
            else:
                macd_interpretation = (
                    "Bullish - MACD above signal"
                )

        elif latest_macd < latest_signal:
            if (
                prev_histogram is not None
                and prev_histogram > 0
                and latest_histogram < 0
            ):
                macd_interpretation = (
                    "Bearish crossover - momentum turning down"
                )
            else:
                macd_interpretation = (
                    "Bearish - MACD below signal"
                )

        else:
            macd_interpretation = (
                "Neutral - MACD equals signal"
            )

    # Bollinger interpretation
    valid_bollinger = data.dropna(
        subset=["Upper Band", "Lower Band", "Band Width"]
    )

    if valid_bollinger.empty:
        latest_upper_band = None
        latest_lower_band = None
        latest_band_width = None
        prev_band_width = None
        bollinger_interpretation = None
        bandwidth_interpretation = None

    else:
        latest_upper_band = valid_bollinger["Upper Band"].iloc[-1]
        latest_lower_band = valid_bollinger["Lower Band"].iloc[-1]
        latest_band_width = valid_bollinger["Band Width"].iloc[-1]

        prev_band_width = (
            valid_bollinger["Band Width"].iloc[-2]
            if len(valid_bollinger) > 1
            else None
        )

        if latest_price > latest_upper_band:
            bollinger_interpretation = (
                "Unusually high relative to recent volatility - "
                "potential overbought"
            )

        elif latest_price < latest_lower_band:
            bollinger_interpretation = (
                "Unusually low relative to recent volatility - "
                "potential oversold"
            )

        else:
            bollinger_interpretation = (
                "Price within bands - within recent volatility range"
            )

        if prev_band_width is None:
            bandwidth_interpretation = None

        elif latest_band_width > prev_band_width:
            bandwidth_interpretation = (
                "Bands expanding - volatility increasing"
            )

        elif latest_band_width < prev_band_width:
            bandwidth_interpretation = (
                "Bands contracting - volatility decreasing"
            )

        else:
            bandwidth_interpretation = (
                "Bands stable - volatility unchanged"
            )

    # Total return
    total_return = (
        (last_close - first_close) / first_close * 100
    )

    # Volatility
    if len(daily_returns) < 2:
        volatility = None
    else:
        volatility = daily_returns.std()

    # Debug prints
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

    print("Standard Deviation 20:")
    print(data["Standard Deviation 20"])

    print("Upper Band:")
    print(data["Upper Band"])

    print("Lower Band:")
    print(data["Lower Band"])

    print("Latest Upper Band:")
    print(latest_upper_band)

    print("Latest Lower Band:")
    print(latest_lower_band)

    print("Latest Band Width:")
    print(latest_band_width)

    print("Bollinger Interpretation:")
    print(bollinger_interpretation)

    print("Bandwidth Interpretation:")
    print(bandwidth_interpretation)

    return {
        "latest_price": float(latest_price),
        "highest_price": float(highest_price),
        "lowest_price": float(lowest_price),
        "average_close": float(average_price),
        "highest_volume": int(highest_volume),

        "total_return_percent": round(
            float(total_return), 2
        ),

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
        ),

        "latest_upper_band": (
            round(float(latest_upper_band), 2)
            if latest_upper_band is not None
            else None
        ),

        "latest_lower_band": (
            round(float(latest_lower_band), 2)
            if latest_lower_band is not None
            else None
        ),

        "latest_band_width": (
            round(float(latest_band_width), 2)
            if latest_band_width is not None
            else None
        ),

        "bollinger_interpretation": bollinger_interpretation,

        "bandwidth_interpretation": (
            bandwidth_interpretation
            if bandwidth_interpretation is not None
            else None
        )
    }