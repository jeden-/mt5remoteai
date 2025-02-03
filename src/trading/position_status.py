"""
Moduł zawierający statusy pozycji.
"""
from enum import Enum, auto

class PositionStatus(Enum):
    """Status pozycji."""
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'
    PENDING = 'PENDING'
    CANCELLED = 'CANCELLED'
    ERROR = 'ERROR' 