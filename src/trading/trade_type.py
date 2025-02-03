"""
Moduł zawierający typy transakcji.
"""
from enum import Enum, auto

class TradeType(Enum):
    """Typ transakcji."""
    BUY = 'BUY'
    SELL = 'SELL' 