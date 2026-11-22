from datetime import date
import pytest
from pydantic import ValidationError

from app.services.portfolio_service import (
    Portfolio,
    Transaction,
    calculate_contribution,
    calculate_current_value,
    calculate_current_weight,
    calculate_initial_weight,
    calculate_profit_loss,
    calculate_profit_loss_percentage,
    calculate_total_current_value,
    calculate_total_investment,
    calculate_total_profit_loss,
    calculate_total_profit_loss_percentage,
    calculate_transaction_cost,
)


def test_single_transaction_cost_value_and_profit_loss_with_charges():
    # 10 shares @ 100 + 50 charges = 1050 cost basis
    # Current price = 126 -> current value = 1260
    # Profit/Loss = 1260 - 1050 = +210 (+20.0%)
    tx = Transaction(
        symbol="RELIANCE.NS",
        quantity=10,
        purchase_price=100.0,
        purchase_date=date(2025, 1, 10),
        charges=50.0,
    )
    current_prices = {"RELIANCE.NS": 126.0}

    assert calculate_transaction_cost(tx) == 1050.0
    assert calculate_current_value(tx, current_prices) == 1260.0
    assert calculate_profit_loss(tx, current_prices) == 210.0
    assert calculate_profit_loss_percentage(tx, current_prices) == pytest.approx(20.0)


def test_portfolio_totals_weights_and_contribution_invariants():
    # Tx 1: 10 @ 100 + 0 = 1000 invested; current price 120 -> 1200 (+200, +20%)
    # Tx 2: 10 @ 200 + 0 = 2000 invested; current price 180 -> 1800 (-200, -10%)
    # Tx 3: 5 @ 200 + 0 = 1000 invested; current price 240 -> 1200 (+200, +20%)
    # Total invested = 4000; Total current value = 4200; Total P&L = +200 (+5.0%)
    tx1 = Transaction(
        symbol="TCS.NS",
        quantity=10,
        purchase_price=100.0,
        purchase_date=date(2025, 1, 1),
        charges=0.0,
    )
    tx2 = Transaction(
        symbol="INFY.NS",
        quantity=10,
        purchase_price=200.0,
        purchase_date=date(2025, 1, 2),
        charges=0.0,
    )
    tx3 = Transaction(
        symbol="RELIANCE.NS",
        quantity=5,
        purchase_price=200.0,
        purchase_date=date(2025, 1, 3),
        charges=0.0,
    )
    portfolio = Portfolio(transactions=[tx1, tx2, tx3])
    current_prices = {
        "TCS.NS": 120.0,
        "INFY.NS": 180.0,
        "RELIANCE.NS": 240.0,
    }

    assert calculate_total_investment(portfolio) == 4000.0
    assert calculate_total_current_value(portfolio, current_prices) == 4200.0
    assert calculate_total_profit_loss(portfolio, current_prices) == 200.0
    total_pl_pct = calculate_total_profit_loss_percentage(portfolio, current_prices)
    assert total_pl_pct == pytest.approx(5.0)

    initial_weights = [
        calculate_initial_weight(tx, portfolio) for tx in portfolio.transactions
    ]
    current_weights = [
        calculate_current_weight(tx, portfolio, current_prices)
        for tx in portfolio.transactions
    ]
    contributions = [
        calculate_contribution(tx, portfolio, current_prices)
        for tx in portfolio.transactions
    ]

    assert initial_weights == pytest.approx([25.0, 50.0, 25.0])
    assert sum(initial_weights) == pytest.approx(100.0)
    assert sum(current_weights) == pytest.approx(100.0)

    # Mathematical invariant: sum of individual transaction contributions equals total portfolio return %
    assert contributions == pytest.approx([5.0, -5.0, 5.0])
    assert sum(contributions) == pytest.approx(total_pl_pct)


def test_transaction_and_portfolio_validation_constraints():
    # Quantity <= 0 rejected
    with pytest.raises(ValidationError):
        Transaction(
            symbol="TCS.NS",
            quantity=0,
            purchase_price=100.0,
            purchase_date=date(2025, 1, 1),
        )

    # Purchase price <= 0 rejected
    with pytest.raises(ValidationError):
        Transaction(
            symbol="TCS.NS",
            quantity=5,
            purchase_price=-10.0,
            purchase_date=date(2025, 1, 1),
        )

    # Negative charges rejected
    with pytest.raises(ValidationError):
        Transaction(
            symbol="TCS.NS",
            quantity=5,
            purchase_price=100.0,
            purchase_date=date(2025, 1, 1),
            charges=-1.0,
        )

    # Empty portfolio transactions list rejected
    with pytest.raises(ValidationError):
        Portfolio(transactions=[])


@pytest.mark.xfail(
    raises=ZeroDivisionError,
    strict=True,
    reason="calculate_current_weight does not guard against total_current_value == 0",
)
def test_calculate_current_weight_zero_total_current_value():
    tx = Transaction(
        symbol="DELISTED.NS",
        quantity=10,
        purchase_price=50.0,
        purchase_date=date(2025, 1, 1),
    )
    portfolio = Portfolio(transactions=[tx])
    current_prices = {"DELISTED.NS": 0.0}

    weight = calculate_current_weight(tx, portfolio, current_prices)
    assert weight == 0.0
