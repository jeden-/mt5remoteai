ZADANIE #9 - Implementacja modeli danych

1. W folderze src/models/ utwórz następujące pliki:
- data_models.py (główne modele danych)
- validators.py (walidatory dla modeli)
- enums.py (enumy dla stałych wartości)

2. W pliku src/models/enums.py zdefiniuj:
```python
from enum import Enum, auto

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    
class OrderStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING"
    
class TimeFrame(Enum):
    M1 = "1M"
    M5 = "5M"
    M15 = "15M"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"
    
class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"

3. W pliku src/models/validators.py zdefiniuj:
from decimal import Decimal
from typing import Optional
from datetime import datetime

class ValidationError(Exception):
    pass

class ModelValidator:
    @staticmethod
    def validate_decimal_positive(value: Decimal, field_name: str) -> None:
        if value <= 0:
            raise ValidationError(f"{field_name} must be positive")
            
    @staticmethod
    def validate_volume(value: Decimal) -> None:
        if value <= 0 or value > 100:
            raise ValidationError("Volume must be between 0 and 100")
            
    @staticmethod
    def validate_price(value: Decimal) -> None:
        if value <= 0:
            raise ValidationError("Price must be positive")
            
    @staticmethod
    def validate_timestamp(value: datetime) -> None:
        if value > datetime.now():
            raise ValidationError("Timestamp cannot be in the future")

4. W pliku src/models/data_models.py zdefiniuj:
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict
from decimal import Decimal
from .enums import OrderType, OrderStatus, TimeFrame, SignalAction
from .validators import ModelValidator

@dataclass
class MarketData:
    """Model danych rynkowych"""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timeframe: TimeFrame
    
    def __post_init__(self):
        ModelValidator.validate_timestamp(self.timestamp)
        ModelValidator.validate_price(self.open)
        ModelValidator.validate_price(self.high)
        ModelValidator.validate_price(self.low)
        ModelValidator.validate_price(self.close)
        ModelValidator.validate_decimal_positive(self.volume, "volume")

@dataclass
class Trade:
    """Model transakcji"""
    id: Optional[int]
    symbol: str
    order_type: OrderType
    volume: Decimal
    entry_price: Decimal
    exit_price: Optional[Decimal]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    open_time: datetime
    close_time: Optional[datetime]
    profit: Optional[Decimal]
    status: OrderStatus
    
    def __post_init__(self):
        ModelValidator.validate_volume(self.volume)
        ModelValidator.validate_price(self.entry_price)
        if self.exit_price:
            ModelValidator.validate_price(self.exit_price)
        if self.stop_loss:
            ModelValidator.validate_price(self.stop_loss)
        if self.take_profit:
            ModelValidator.validate_price(self.take_profit)
        ModelValidator.validate_timestamp(self.open_time)
        if self.close_time:
            ModelValidator.validate_timestamp(self.close_time)

@dataclass
class SignalData:
    """Model sygnału handlowego"""
    symbol: str
    timestamp: datetime
    action: SignalAction
    confidence: float
    entry_price: Optional[Decimal]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    indicators: Dict[str, float]
    ai_analysis: Dict[str, str]
    
    def __post_init__(self):
        ModelValidator.validate_timestamp(self.timestamp)
        if self.confidence < 0 or self.confidence > 1:
            raise ValidationError("Confidence must be between 0 and 1")
        if self.entry_price:
            ModelValidator.validate_price(self.entry_price)
        if self.stop_loss:
            ModelValidator.validate_price(self.stop_loss)
        if self.take_profit:
            ModelValidator.validate_price(self.take_profit)

@dataclass
class AccountInfo:
    """Model informacji o koncie"""
    balance: Decimal
    equity: Decimal
    margin: Decimal
    free_margin: Decimal
    margin_level: float
    leverage: int
    currency: str
    
    def __post_init__(self):
        ModelValidator.validate_decimal_positive(self.balance, "balance")
        ModelValidator.validate_decimal_positive(self.equity, "equity")
        ModelValidator.validate_decimal_positive(self.margin, "margin")
        if self.leverage <= 0:
            raise ValidationError("Leverage must be positive")

@dataclass
class BacktestResult:
    """Model wyników backtestu"""
    start_date: datetime
    end_date: datetime
    initial_balance: Decimal
    final_balance: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    max_drawdown: float
    trades: List[Trade]
    equity_curve: List[Dict[str, float]]
    
    def __post_init__(self):
        ModelValidator.validate_timestamp(self.start_date)
        ModelValidator.validate_timestamp(self.end_date)
        if self.start_date >= self.end_date:
            raise ValidationError("Start date must be before end date")
        ModelValidator.validate_decimal_positive(self.initial_balance, "initial_balance")
        ModelValidator.validate_decimal_positive(self.final_balance, "final_balance")
        if self.total_trades < 0:
            raise ValidationError("Total trades cannot be negative")
        if self.winning_trades < 0:
            raise ValidationError("Winning trades cannot be negative")
        if self.losing_trades < 0:
            raise ValidationError("Losing trades cannot be negative")
        if self.winning_trades + self.losing_trades > self.total_trades:
            raise ValidationError("Sum of winning and losing trades cannot exceed total trades")

5. Dodaj testy dla modeli w tests/test_models.py.

6. Zaktualizuj wszystkie istniejące komponenty, aby używały nowych modeli danych.