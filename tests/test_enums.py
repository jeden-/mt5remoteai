"""
Testy dla modułu enums.py
"""
import pytest
from src.models.enums import OrderType, OrderStatus, TimeFrame, SignalAction


def test_order_type_values():
    """Test wartości OrderType."""
    assert OrderType.BUY == "BUY"
    assert OrderType.SELL == "SELL"
    assert str(OrderType.BUY) == "BUY"
    assert str(OrderType.SELL) == "SELL"
    assert len(OrderType) == 2


def test_order_status_values():
    """Test wartości OrderStatus."""
    assert OrderStatus.PENDING == "PENDING"
    assert OrderStatus.OPEN == "OPEN"
    assert OrderStatus.CLOSED == "CLOSED"
    assert OrderStatus.CANCELLED == "CANCELLED"
    assert OrderStatus.ERROR == "ERROR"
    assert str(OrderStatus.PENDING) == "PENDING"
    assert str(OrderStatus.ERROR) == "ERROR"
    assert len(OrderStatus) == 5


def test_timeframe_values():
    """Test wartości TimeFrame."""
    assert TimeFrame.M1 == "M1"
    assert TimeFrame.M5 == "M5"
    assert TimeFrame.M15 == "M15"
    assert TimeFrame.M30 == "M30"
    assert TimeFrame.H1 == "H1"
    assert TimeFrame.H4 == "H4"
    assert TimeFrame.D1 == "D1"
    assert TimeFrame.W1 == "W1"
    assert TimeFrame.MN1 == "MN1"
    assert str(TimeFrame.M1) == "M1"
    assert str(TimeFrame.MN1) == "MN1"
    assert len(TimeFrame) == 9


def test_signal_action_values():
    """Test wartości SignalAction."""
    assert SignalAction.BUY == "BUY"
    assert SignalAction.SELL == "SELL"
    assert SignalAction.CLOSE == "CLOSE"
    assert SignalAction.HOLD == "HOLD"
    assert str(SignalAction.BUY) == "BUY"
    assert str(SignalAction.HOLD) == "HOLD"
    assert len(SignalAction) == 4


def test_order_type_comparison():
    """Test porównywania OrderType."""
    assert OrderType.BUY != OrderType.SELL
    assert OrderType.BUY == OrderType.BUY
    assert OrderType.BUY != "SELL"
    assert OrderType.SELL == "SELL"


def test_order_status_comparison():
    """Test porównywania OrderStatus."""
    assert OrderStatus.OPEN != OrderStatus.CLOSED
    assert OrderStatus.PENDING == OrderStatus.PENDING
    assert OrderStatus.ERROR != "CANCELLED"
    assert OrderStatus.CANCELLED == "CANCELLED"


def test_timeframe_comparison():
    """Test porównywania TimeFrame."""
    assert TimeFrame.M1 != TimeFrame.M5
    assert TimeFrame.H1 == TimeFrame.H1
    assert TimeFrame.D1 != "W1"
    assert TimeFrame.W1 == "W1"


def test_signal_action_comparison():
    """Test porównywania SignalAction."""
    assert SignalAction.BUY != SignalAction.SELL
    assert SignalAction.CLOSE == SignalAction.CLOSE
    assert SignalAction.HOLD != "BUY"
    assert SignalAction.BUY == "BUY"


def test_invalid_enum_values():
    """Test obsługi nieprawidłowych wartości."""
    with pytest.raises(ValueError):
        OrderType("INVALID")
    
    with pytest.raises(ValueError):
        OrderStatus("INVALID")
    
    with pytest.raises(ValueError):
        TimeFrame("INVALID")
    
    with pytest.raises(ValueError):
        SignalAction("INVALID") 