# Stock Analytics Backend
A stock market analytics backend I'm building to learn backend development while working with real-world financial data.
The project uses FastAPI to expose REST APIs, `yfinance` to retrieve historical stock data, and Pandas to process the data and calculate financial and technical indicators.
## Why I Built This
I wanted to build something beyond small practice APIs and work on a project where I could learn backend development while working with a domain I'm interested in.
I started with retrieving historical stock data and gradually added financial calculations, technical indicators, input validation, cloud storage, and time-period comparison.
The project is being built step by step, with the goal of eventually expanding it into a larger financial analytics application.
## What It Can Do(Currently)
- Retrieve historical stock market data using Yahoo Finance
- Support predefined periods such as `1mo`, `6mo`, `1y`, `5y`, and `max`
- Support custom start and end dates
- Calculate daily returns
- Calculate a 20-period Moving Average
- Calculate RSI (Relative Strength Index)
- Calculate MACD, Signal Line, and Histogram
- Calculate Bollinger Bands and Band Width
- Calculate Total Return
- Calculate Average Daily Return
- Calculate Volatility
- Calculate Maximum Drawdown
- Provide basic interpretations of technical indicators
- Compare two different time periods for the same stock
- Validate API inputs and date ranges
- Handle ranges where no trading data is available
- Store downloaded historical CSV data in Amazon S3
## Analytics
The backend currently calculates a mix of basic financial metrics and technical indicators.
### Price & Volume
- Latest closing price
- Highest closing price
- Lowest closing price
- Average closing price
- Highest trading volume
### Returns & Risk
- Total Return
- Average Daily Return
- Volatility
- Maximum Drawdown
### Technical Indicators
**20-Period Moving Average**
A rolling average calculated from the previous 20 price observations.
**RSI**
A 14-period momentum indicator based on recent gains and losses. The API also provides a basic interpretation of the latest RSI value.
**MACD**
Calculated using:
- 12-period EMA
- 26-period EMA
- 9-period Signal Line
- MACD Histogram
The API returns the latest MACD values along with a basic interpretation of the current momentum.
**Bollinger Bands**
Calculated using:
- 20-period Moving Average
- 20-period Standard Deviation
- Upper and Lower Bands using 2 standard deviations
Band Width is also calculated to help indicate changes in recent volatility.
## API
### `GET /stock`
Retrieves historical stock data along with the calculated analytics.
Using a predefined period:
```text
/stock?symbol=RELIANCE.NS&period=1mo

Using a custom date range:

/stock?symbol=RELIANCE.NS&start=2025-01-01&end=2025-06-30

The response contains the stock symbol, summary analytics, and processed historical data.

GET /compare

Compares two different time periods for the same stock.

Example:

/compare?symbol=RELIANCE.NS&start_a=2025-04-05&end_a=2026-04-07&start_b=2024-04-05&end_b=2026-04-07

The endpoint calculates the analytics for both periods and compares:

* Total Return
* Average Daily Return
* Volatility
* Maximum Drawdown

The comparison is calculated as:

Period B metric - Period A metric

The endpoint also validates the supplied date ranges and prevents identical comparison periods.

API Documentation

FastAPI provides interactive API documentation through Swagger UI.

After starting the server, open:

http://127.0.0.1:8000/docs

This can be used to try the endpoints, provide parameters, and inspect the JSON responses.

Tech Stack

* Python — Main programming language
* FastAPI — API framework
* Pandas — Data processing and analytics
* NumPy — Numerical operations
* yfinance — Historical stock data
* boto3 — AWS integration
* Amazon S3 — Cloud storage for downloaded historical data

Project Structure

The project is currently kept fairly simple:

app/
├── main.py
└── services/
    ├── stock_service.py
    └── analytics_service.py
data/
└── Local historical CSV files

main.py contains the API endpoints and request validation.

stock_service.py handles retrieving stock data, saving the data as CSV, and uploading it to S3.

analytics_service.py contains the financial calculations, technical indicators, interpretations, summary generation, and time-period comparison logic.

Running Locally

Clone the repository and navigate into the project directory.

Install the dependencies:

pip install -r requirements.txt

Start the development server:

uvicorn app.main:app --reload

The API documentation will then be available at:

http://127.0.0.1:8000/docs

Data Handling

Historical stock data is retrieved through Yahoo Finance using yfinance.

The downloaded data is saved as CSV and uploaded to Amazon S3. At the current stage of the project, S3 is being used mainly for cloud storage rather than as part of a large-scale data pipeline.

Local CSV files used during development are not necessarily committed to the repository.

Validation & Error Handling

The API includes validation for things such as:

* Invalid predefined periods
* Missing start or end dates
* Using both a predefined period and a custom date range
* Start dates occurring after end dates
* Identical comparison periods
* Requested ranges with no available trading data

When there isn’t enough data to meaningfully calculate a metric, the API returns null rather than making up a value.

What’s Next

The project is still evolving.

The next major feature is stock-to-stock comparison, followed by more financial and portfolio analysis. Longer term, I plan to explore feature engineering, machine learning, multiple ML models, deep learning, AI-assisted interpretation, and eventually an Angular frontend and cloud deployment.

The roadmap isn’t completely fixed — I’ll keep expanding it as I learn and as the project grows.

Disclaimer

This project is intended for learning, software development, and financial data analysis.

The analytics and interpretations provided by the application are not financial advice and should not be used as the sole basis for investment decisions.
