"""
Testy dla modułu enums.py
"""
import pytest
from src.models.enums import (
    OrderType,
    OrderStatus,
    TimeFrame,
    SignalAction,
    TradeType,
    PositionStatus
)


def test_order_type_str():
    """Test konwersji OrderType na string."""
    assert str(OrderType.BUY) == "BUY"
    assert str(OrderType.SELL) == "SELL"


def test_order_status_str():
    """Test konwersji OrderStatus na string."""
    assert str(OrderStatus.PENDING) == "PENDING"
    assert str(OrderStatus.OPEN) == "OPEN"
    assert str(OrderStatus.CLOSED) == "CLOSED"
    assert str(OrderStatus.CANCELLED) == "CANCELLED"
    assert str(OrderStatus.ERROR) == "ERROR"


def test_time_frame_str():
    """Test konwersji TimeFrame na string."""
    assert str(TimeFrame.M1) == "M1"
    assert str(TimeFrame.M5) == "M5"
    assert str(TimeFrame.M15) == "M15"
    assert str(TimeFrame.M30) == "M30"
    assert str(TimeFrame.H1) == "H1"
    assert str(TimeFrame.H4) == "H4"
    assert str(TimeFrame.D1) == "D1"
    assert str(TimeFrame.W1) == "W1"
    assert str(TimeFrame.MN1) == "MN1"


def test_signal_action_str():
    """Test konwersji SignalAction na string."""
    assert str(SignalAction.BUY) == "BUY"
    assert str(SignalAction.SELL) == "SELL"
    assert str(SignalAction.CLOSE) == "CLOSE"
    assert str(SignalAction.HOLD) == "HOLD"


def test_trade_type_str():
    """Test konwersji TradeType na string."""
    assert str(TradeType.BUY) == "BUY"
    assert str(TradeType.SELL) == "SELL"


def test_position_status_str():
    """Test konwersji PositionStatus na string."""
    assert str(PositionStatus.PENDING) == "PENDING"
    assert str(PositionStatus.OPEN) == "OPEN"
    assert str(PositionStatus.CLOSED) == "CLOSED"
    assert str(PositionStatus.CANCELLED) == "CANCELLED"
    assert str(PositionStatus.ERROR) == "ERROR"


def test_order_type_values():
    """Test wartości OrderType."""
    assert OrderType.BUY.value == "BUY"
    assert OrderType.SELL.value == "SELL"


def test_order_status_values():
    """Test wartości OrderStatus."""
    assert OrderStatus.PENDING.value == "PENDING"
    assert OrderStatus.OPEN.value == "OPEN"
    assert OrderStatus.CLOSED.value == "CLOSED"
    assert OrderStatus.CANCELLED.value == "CANCELLED"
    assert OrderStatus.ERROR.value == "ERROR"


def test_time_frame_values():
    """Test wartości TimeFrame."""
    assert TimeFrame.M1.value == "M1"
    assert TimeFrame.M5.value == "M5"
    assert TimeFrame.M15.value == "M15"
    assert TimeFrame.M30.value == "M30"
    assert TimeFrame.H1.value == "H1"
    assert TimeFrame.H4.value == "H4"
    assert TimeFrame.D1.value == "D1"
    assert TimeFrame.W1.value == "W1"
    assert TimeFrame.MN1.value == "MN1"


def test_signal_action_values():
    """Test wartości SignalAction."""
    assert SignalAction.BUY.value == "BUY"
    assert SignalAction.SELL.value == "SELL"
    assert SignalAction.CLOSE.value == "CLOSE"
    assert SignalAction.HOLD.value == "HOLD"


def test_trade_type_values():
    """Test wartości TradeType."""
    assert TradeType.BUY.value == "BUY"
    assert TradeType.SELL.value == "SELL"


def test_position_status_values():
    """Test wartości PositionStatus."""
    assert PositionStatus.PENDING.value == "PENDING"
    assert PositionStatus.OPEN.value == "OPEN"
    assert PositionStatus.CLOSED.value == "CLOSED"
    assert PositionStatus.CANCELLED.value == "CANCELLED"
    assert PositionStatus.ERROR.value == "ERROR"


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

    assert OrderType.BUY != "INVALID"
    assert OrderStatus.OPEN != "INVALID"
    assert TimeFrame.H1 != "INVALID"
    assert SignalAction.BUY != "INVALID"
    assert TradeType.BUY != "INVALID"
    assert PositionStatus.OPEN != "INVALID" 