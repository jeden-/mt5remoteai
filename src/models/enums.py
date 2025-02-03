"""
Enumeracje dla modeli danych.
"""
from enum import Enum


class OrderType(str, Enum):
    """Typy zleceń."""
    
    BUY = "BUY"  # Pozycja długa
    SELL = "SELL"  # Pozycja krótka

    def __str__(self) -> str:
        return self.value


class OrderStatus(str, Enum):
    """Statusy zleceń."""
    
    PENDING = "PENDING"  # Oczekujące
    OPEN = "OPEN"  # Otwarte
    CLOSED = "CLOSED"  # Zamknięte
    CANCELLED = "CANCELLED"  # Anulowane
    ERROR = "ERROR"  # Błąd

    def __str__(self) -> str:
        return self.value


class TimeFrame(str, Enum):
    """Interwały czasowe."""
    
    M1 = "M1"  # 1 minuta
    M5 = "M5"  # 5 minut
    M15 = "M15"  # 15 minut
    M30 = "M30"  # 30 minut
    H1 = "H1"  # 1 godzina
    H4 = "H4"  # 4 godziny
    D1 = "D1"  # 1 dzień
    W1 = "W1"  # 1 tydzień
    MN1 = "MN1"  # 1 miesiąc

    def __str__(self) -> str:
        return self.value


class SignalAction(str, Enum):
    """Akcje sygnałów tradingowych."""
    
    BUY = "BUY"  # Sygnał kupna
    SELL = "SELL"  # Sygnał sprzedaży
    CLOSE = "CLOSE"  # Sygnał zamknięcia
    HOLD = "HOLD"  # Sygnał wstrzymania

    def __str__(self) -> str:
        return self.value


class TradeType(str, Enum):
    """Typy transakcji."""
    
    BUY = "BUY"  # Pozycja długa
    SELL = "SELL"  # Pozycja krótka

    def __str__(self) -> str:
        return self.value


class PositionStatus(str, Enum):
    """Statusy pozycji."""
    
    PENDING = "PENDING"  # Oczekująca
    OPEN = "OPEN"  # Otwarta
    CLOSED = "CLOSED"  # Zamknięta
    CANCELLED = "CANCELLED"  # Anulowana
    ERROR = "ERROR"  # Błąd

    def __str__(self) -> str:
        return self.value 