"""
Moduł zawierający walidatory dla modeli danych.
"""
from decimal import Decimal
from typing import Optional, Union, Any
from datetime import datetime
from pydantic_core import PydanticCustomError


class DecimalPositiveError(ValueError):
    """Błąd ujemnej wartości dziesiętnej."""
    def __init__(self, message: str = 'Wartość musi być dodatnia'):
        super().__init__(message)


class DecimalRangeError(ValueError):
    """Błąd wartości dziesiętnej poza zakresem."""
    def __init__(self, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal]):
        super().__init__(f'Wartość musi być w zakresie {min_val}-{max_val}')


class DatetimePastError(ValueError):
    """Błąd daty z przyszłości."""
    def __init__(self):
        super().__init__('Data nie może być z przyszłości')


class SymbolError(ValueError):
    """Błąd niepoprawnego symbolu."""
    def __init__(self):
        super().__init__('Symbol musi mieć od 3 do 10 znaków')


class ConfidenceError(ValueError):
    """Błąd wartości pewności."""
    def __init__(self):
        super().__init__('Wartość pewności musi być w zakresie 0-1')


class LeverageError(ValueError):
    """Błąd wartości dźwigni."""
    def __init__(self):
        super().__init__('Dźwignia musi być w zakresie 1-500')


class TradeCountError(ValueError):
    """Błąd liczby transakcji."""
    def __init__(self):
        super().__init__('Liczba transakcji nie może być ujemna')


def validate_decimal_positive(value: Decimal) -> Decimal:
    """
    Sprawdza czy wartość dziesiętna jest dodatnia.
    
    Args:
        value: Wartość do sprawdzenia
        
    Returns:
        Decimal: Zwalidowana wartość
        
    Raises:
        DecimalPositiveError: Gdy wartość nie jest dodatnia
    """
    if value is None:
        return value
    if value <= 0:
        raise DecimalPositiveError("Wartość musi być dodatnia")
    return value


def validate_decimal_range(
    value: Decimal,
    min_val: Union[int, float, Decimal],
    max_val: Union[int, float, Decimal]
) -> Decimal:
    """
    Sprawdza czy wartość dziesiętna jest w zadanym zakresie.

    Args:
        value: Wartość do sprawdzenia
        min_val: Minimalna wartość
        max_val: Maksymalna wartość

    Returns:
        Decimal: Zwalidowana wartość

    Raises:
        DecimalRangeError: Gdy wartość jest poza zakresem
    """
    if value < min_val or value > max_val:
        raise DecimalRangeError(min_val, max_val)
    return value


def validate_past_datetime(value: datetime) -> datetime:
    """
    Sprawdza czy data nie jest w przyszłości.
    
    Args:
        value: Data do sprawdzenia
        
    Returns:
        datetime: Zwalidowana data
        
    Raises:
        DatetimePastError: Gdy data jest w przyszłości
    """
    if value > datetime.now():
        raise DatetimePastError()
    return value


def validate_symbol(value: str) -> str:
    """
    Sprawdza poprawność symbolu instrumentu.
    
    Args:
        value: Symbol do sprawdzenia
        
    Returns:
        str: Zwalidowany symbol
        
    Raises:
        SymbolError: Gdy symbol jest nieprawidłowy
    """
    if not value or len(value) < 3 or len(value) > 10:
        raise SymbolError()
    return value


def validate_confidence(value: float) -> float:
    """
    Sprawdza czy wartość pewności jest w zakresie 0-1.
    
    Args:
        value: Wartość do sprawdzenia
        
    Returns:
        float: Zwalidowana wartość
        
    Raises:
        ConfidenceError: Gdy wartość jest poza zakresem
    """
    if value < 0 or value > 1:
        raise ConfidenceError()
    return value


def validate_leverage(value: int) -> int:
    """
    Sprawdza czy dźwignia jest w dozwolonym zakresie.
    
    Args:
        value: Wartość do sprawdzenia
        
    Returns:
        int: Zwalidowana wartość
        
    Raises:
        LeverageError: Gdy dźwignia jest poza zakresem
    """
    if value < 1 or value > 500:
        raise LeverageError()
    return value


def validate_trade_count(value: int) -> int:
    """
    Sprawdza czy liczba transakcji jest nieujemna.
    
    Args:
        value: Wartość do sprawdzenia
        
    Returns:
        int: Zwalidowana wartość
        
    Raises:
        TradeCountError: Gdy liczba jest ujemna
    """
    if value < 0:
        raise TradeCountError()
    return value


class ModelValidator:
    """Klasa zawierająca metody walidacji dla modeli danych."""
    
    @staticmethod
    def validate_decimal_positive(value: Decimal, field_name: str) -> None:
        """
        Sprawdza czy wartość dziesiętna jest dodatnia.
        
        Args:
            value: Wartość do sprawdzenia
            field_name: Nazwa pola (do komunikatu błędu)
            
        Raises:
            DecimalPositiveError: Gdy wartość nie jest dodatnia
        """
        if value <= 0:
            raise DecimalPositiveError()
            
    @staticmethod
    def validate_volume(value: Decimal) -> None:
        """
        Sprawdza czy wolumen jest w prawidłowym zakresie (0-100).
        
        Args:
            value: Wartość wolumenu do sprawdzenia
            
        Raises:
            DecimalPositiveError: Gdy wolumen jest poza zakresem
        """
        if value <= 0 or value > 100:
            raise DecimalPositiveError()
            
    @staticmethod
    def validate_price(value: Decimal) -> None:
        """
        Sprawdza czy cena jest dodatnia.
        
        Args:
            value: Cena do sprawdzenia
            
        Raises:
            DecimalPositiveError: Gdy cena nie jest dodatnia
        """
        if value <= 0:
            raise DecimalPositiveError()
            
    @staticmethod
    def validate_timestamp(value: datetime) -> None:
        """
        Sprawdza czy timestamp nie jest w przyszłości.
        
        Args:
            value: Data do sprawdzenia
            
        Raises:
            DatetimePastError: Gdy data jest w przyszłości
        """
        if value > datetime.now():
            raise DatetimePastError()

    @staticmethod
    def validate_positive_decimal(value: Decimal, field_name: str) -> None:
        """
        Sprawdza czy wartość dziesiętna jest dodatnia.
        
        Args:
            value: Wartość do sprawdzenia
            field_name: Nazwa pola (do komunikatu błędu)
            
        Raises:
            DecimalPositiveError: Gdy wartość nie jest dodatnia
        """
        if value <= 0:
            raise DecimalPositiveError() 