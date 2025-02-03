"""
Testy dla modułu trade_type.py
"""
import pytest
from src.trading.trade_type import TradeType

def test_trade_type_values():
    """Test wartości typów transakcji."""
    assert TradeType.BUY.value == 'BUY'
    assert TradeType.SELL.value == 'SELL'

def test_trade_type_comparison():
    """Test porównywania typów transakcji."""
    assert TradeType.BUY != TradeType.SELL
    assert TradeType.BUY == TradeType.BUY
    assert TradeType.SELL == TradeType.SELL

def test_trade_type_from_string():
    """Test tworzenia typu z tekstu."""
    assert TradeType('BUY') == TradeType.BUY
    assert TradeType('SELL') == TradeType.SELL

def test_trade_type_invalid_value():
    """Test tworzenia typu z nieprawidłowej wartości."""
    with pytest.raises(ValueError):
        TradeType('INVALID')

def test_trade_type_str_representation():
    """Test reprezentacji tekstowej typu."""
    assert str(TradeType.BUY) == 'TradeType.BUY'
    assert str(TradeType.SELL) == 'TradeType.SELL'

def test_trade_type_name_property():
    """Test właściwości name typu."""
    assert TradeType.BUY.name == 'BUY'
    assert TradeType.SELL.name == 'SELL'

def test_trade_type_opposite():
    """Test sprawdzania przeciwnego typu transakcji."""
    assert TradeType.BUY != TradeType.SELL
    assert TradeType.SELL != TradeType.BUY 