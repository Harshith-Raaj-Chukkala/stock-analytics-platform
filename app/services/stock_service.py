import yfinance as yf 
import pandas as pd
import boto3
import os
from dotenv import load_dotenv 


load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

def download_stock_data(symbol, period="1mo"):
 ticker = yf.Ticker(symbol)
 data   = ticker.history(period=period)

 if data.empty: 
     raise Exception("Invalid stock symbol")  
 
 filename = f"data/{symbol}.csv"
 data.to_csv(filename)
 s3.upload_file(
    filename,
    AWS_BUCKET_NAME,
    filename
)
 return data 
 
def calculate_daily_returns (data): 
     data["Daily Return"] = data["Close"].pct_change() * 100
     return data 
     

 