"""
Testy dla modułu validators.py
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from src.models.validators import (
    validate_decimal_positive,
    validate_decimal_range,
    validate_past_datetime,
    validate_symbol,
    validate_confidence,
    validate_leverage,
    validate_trade_count,
    DecimalPositiveError,
    DecimalRangeError,
    DatetimePastError,
    SymbolError,
    ConfidenceError,
    LeverageError,
    TradeCountError,
    ModelValidator
)


def test_validate_decimal_positive():
    """Test walidacji dodatnich wartości dziesiętnych."""
    # Poprawne wartości
    assert validate_decimal_positive(Decimal('1.0')) == Decimal('1.0')
    assert validate_decimal_positive(Decimal('0.1')) == Decimal('0.1')
    assert validate_decimal_positive(Decimal('999.99')) == Decimal('999.99')
    
    # Niepoprawne wartości
    with pytest.raises(DecimalPositiveError):
        validate_decimal_positive(Decimal('0'))
    with pytest.raises(DecimalPositiveError):
        validate_decimal_positive(Decimal('-1.0'))


def test_validate_decimal_range():
    """Test walidacji zakresu wartości dziesiętnych."""
    # Poprawne wartości
    assert validate_decimal_range(Decimal('5.0'), 0, 10) == Decimal('5.0')
    assert validate_decimal_range(Decimal('0'), 0, 10) == Decimal('0')
    assert validate_decimal_range(Decimal('10'), 0, 10) == Decimal('10')
    
    # Niepoprawne wartości
    with pytest.raises(DecimalRangeError):
        validate_decimal_range(Decimal('-1.0'), 0, 10)
    with pytest.raises(DecimalRangeError):
        validate_decimal_range(Decimal('11.0'), 0, 10)


def test_validate_past_datetime():
    """Test walidacji dat z przeszłości."""
    # Poprawne wartości
    past = datetime.now() - timedelta(days=1)
    assert validate_past_datetime(past) == past
    
    # Niepoprawne wartości
    future = datetime.now() + timedelta(days=1)
    with pytest.raises(DatetimePastError):
        validate_past_datetime(future)


def test_validate_symbol():
    """Test walidacji symboli."""
    # Poprawne wartości
    assert validate_symbol('EUR') == 'EUR'
    assert validate_symbol('EURUSD') == 'EURUSD'
    assert validate_symbol('GBPUSD_m') == 'GBPUSD_m'
    
    # Niepoprawne wartości
    with pytest.raises(SymbolError):
        validate_symbol('')
    with pytest.raises(SymbolError):
        validate_symbol('EU')
    with pytest.raises(SymbolError):
        validate_symbol('GBPUSD_micro')  # za długi


def test_validate_confidence():
    """Test walidacji wartości pewności."""
    # Poprawne wartości
    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(0.5) == 0.5
    assert validate_confidence(1.0) == 1.0
    
    # Niepoprawne wartości
    with pytest.raises(ConfidenceError):
        validate_confidence(-0.1)
    with pytest.raises(ConfidenceError):
        validate_confidence(1.1)


def test_validate_leverage():
    """Test walidacji dźwigni."""
    # Poprawne wartości
    assert validate_leverage(1) == 1
    assert validate_leverage(100) == 100
    assert validate_leverage(500) == 500
    
    # Niepoprawne wartości
    with pytest.raises(LeverageError):
        validate_leverage(0)
    with pytest.raises(LeverageError):
        validate_leverage(501)


def test_validate_trade_count():
    """Test walidacji liczby transakcji."""
    # Poprawne wartości
    assert validate_trade_count(0) == 0
    assert validate_trade_count(1) == 1
    assert validate_trade_count(1000) == 1000
    
    # Niepoprawne wartości
    with pytest.raises(TradeCountError):
        validate_trade_count(-1)


def test_model_validator_decimal_positive():
    """Test walidacji dodatnich wartości dziesiętnych w ModelValidator."""
    validator = ModelValidator()
    
    # Poprawne wartości
    validator.validate_decimal_positive(Decimal('1.0'), 'test_field')
    validator.validate_decimal_positive(Decimal('0.1'), 'test_field')
    
    # Niepoprawne wartości
    with pytest.raises(DecimalPositiveError):
        validator.validate_decimal_positive(Decimal('0'), 'test_field')
    with pytest.raises(DecimalPositiveError):
        validator.validate_decimal_positive(Decimal('-1.0'), 'test_field')


def test_model_validator_volume():
    """Test walidacji wolumenu w ModelValidator."""
    validator = ModelValidator()
    
    # Poprawne wartości
    validator.validate_volume(Decimal('0.1'))
    validator.validate_volume(Decimal('50'))
    validator.validate_volume(Decimal('100'))
    
    # Niepoprawne wartości
    with pytest.raises(DecimalPositiveError):
        validator.validate_volume(Decimal('0'))
    with pytest.raises(DecimalPositiveError):
        validator.validate_volume(Decimal('101'))


def test_model_validator_price():
    """Test walidacji ceny w ModelValidator."""
    validator = ModelValidator()
    
    # Poprawne wartości
    validator.validate_price(Decimal('0.1'))
    validator.validate_price(Decimal('1000.50'))
    
    # Niepoprawne wartości
    with pytest.raises(DecimalPositiveError):
        validator.validate_price(Decimal('0'))
    with pytest.raises(DecimalPositiveError):
        validator.validate_price(Decimal('-1.0'))


def test_model_validator_timestamp():
    """Test walidacji timestampu w ModelValidator."""
    validator = ModelValidator()
    
    # Poprawne wartości
    past = datetime.now() - timedelta(days=1)
    validator.validate_timestamp(past)
    
    # Niepoprawne wartości
    future = datetime.now() + timedelta(days=1)
    with pytest.raises(DatetimePastError):
        validator.validate_timestamp(future)


def test_confidence_error_message():
    """Test wiadomości błędu ConfidenceError."""
    error = ConfidenceError()
    assert str(error) == 'Wartość pewności musi być w zakresie 0-1'
    
    # Test tworzenia instancji
    error2 = ConfidenceError()
    assert isinstance(error2, ValueError)
    assert isinstance(error2, ConfidenceError)
    
    # Test wywołania __init__
    error3 = ConfidenceError()
    assert error3.__init__() is None


def test_model_validator_positive_decimal():
    """Test walidacji dodatnich wartości dziesiętnych w ModelValidator."""
    validator = ModelValidator()
    
    # Poprawne wartości
    validator.validate_positive_decimal(Decimal('1.0'), 'test_field')
    validator.validate_positive_decimal(Decimal('0.1'), 'test_field')
    
    # Niepoprawne wartości
    with pytest.raises(DecimalPositiveError):
        validator.validate_positive_decimal(Decimal('0'), 'test_field')
    with pytest.raises(DecimalPositiveError):
        validator.validate_positive_decimal(Decimal('-1.0'), 'test_field') 