from pydantic import BaseModel, Field
from datetime import date
from app.services.stock_service import download_stock_data


class Transaction(BaseModel):
    symbol: str
    quantity: float = Field(..., gt=0, description="Quantity must be greater than 0")   
    purchase_price: float = Field(..., gt=0, description="Purchase price must be greater than 0")   
    purchase_date: date 
    charges: float = Field(default=0, ge=0, description="Charges must be greater than or equal to 0")

class Portfolio(BaseModel):
    transactions: list[Transaction] = Field(min_length=1)

def calculate_transaction_cost(transaction: Transaction):
    return (transaction.quantity * transaction.purchase_price) + transaction.charges

def calculate_total_investment(portfolio: Portfolio):
   total = 0 

   for transaction in portfolio.transactions:
       total += calculate_transaction_cost(transaction)
   return total

def calculate_current_value(transaction): 
    data = download_stock_data(transaction.symbol,period="1d")  

    latest_price = data["Close"].iloc[-1] 

    return transaction.quantity * latest_price

def calculate_profit_loss(transaction): 
    current_value = calculate_current_value(transaction)
    total_investment = calculate_transaction_cost(transaction)

    return current_value - total_investment

def calculate_profit_loss_percentage(transaction):
    profit_loss = calculate_profit_loss(transaction)
    total_investment = calculate_transaction_cost(transaction)

    if total_investment == 0:
        return 0

    return (profit_loss / total_investment) * 100

def calculate_total_current_value(portfolio: Portfolio):
    total_current_value = 0

    for transaction in portfolio.transactions:
        total_current_value += calculate_current_value(transaction)

    return total_current_value

def calculate_total_profit_loss(portfolio: Portfolio):
    total_profit_loss = 0

    for transaction in portfolio.transactions:
        total_profit_loss += calculate_profit_loss(transaction)

    return total_profit_loss

def calculate_total_profit_loss_percentage(portfolio: Portfolio):
    total_profit_loss = calculate_total_profit_loss(portfolio)
    total_investment = calculate_total_investment(portfolio)

    if total_investment == 0:
        return 0

    return (total_profit_loss / total_investment) * 100