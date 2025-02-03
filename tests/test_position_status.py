"""
Testy dla modułu position_status.py
"""
import pytest
from src.trading.position_status import PositionStatus

def test_position_status_values():
    """Test wartości statusów pozycji."""
    assert PositionStatus.OPEN.value == 'OPEN'
    assert PositionStatus.CLOSED.value == 'CLOSED'
    assert PositionStatus.PENDING.value == 'PENDING'
    assert PositionStatus.CANCELLED.value == 'CANCELLED'
    assert PositionStatus.ERROR.value == 'ERROR'

def test_position_status_comparison():
    """Test porównywania statusów pozycji."""
    assert PositionStatus.OPEN != PositionStatus.CLOSED
    assert PositionStatus.OPEN == PositionStatus.OPEN
    assert PositionStatus.CLOSED == PositionStatus.CLOSED

def test_position_status_from_string():
    """Test tworzenia statusu z tekstu."""
    assert PositionStatus('OPEN') == PositionStatus.OPEN
    assert PositionStatus('CLOSED') == PositionStatus.CLOSED
    assert PositionStatus('PENDING') == PositionStatus.PENDING
    assert PositionStatus('CANCELLED') == PositionStatus.CANCELLED
    assert PositionStatus('ERROR') == PositionStatus.ERROR

def test_position_status_invalid_value():
    """Test tworzenia statusu z nieprawidłowej wartości."""
    with pytest.raises(ValueError):
        PositionStatus('INVALID')

def test_position_status_str_representation():
    """Test reprezentacji tekstowej statusu."""
    assert str(PositionStatus.OPEN) == 'PositionStatus.OPEN'
    assert str(PositionStatus.CLOSED) == 'PositionStatus.CLOSED'

def test_position_status_name_property():
    """Test właściwości name statusu."""
    assert PositionStatus.OPEN.name == 'OPEN'
    assert PositionStatus.CLOSED.name == 'CLOSED' 