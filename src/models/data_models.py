"""
Moduł zawierający modele danych używane w aplikacji.
"""
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator, ValidationError, model_validator, ValidationInfo
from pydantic_core import PydanticCustomError
from dataclasses import dataclass

from .enums import OrderType, SignalAction, TradeType, PositionStatus
from .validators import (
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
    TradeCountError
)


class TimeOrderError(ValueError):
    """Błąd kolejności czasowej."""
    def __init__(self):
        super().__init__('Data zamknięcia musi być późniejsza niż data otwarcia')


class StopLossError(ValueError):
    """Błąd poziomu stop loss."""
    def __init__(self, position: str, kierunek: str):
        super().__init__(f'Stop loss dla pozycji {position} musi być {kierunek} ceny wejścia')


class TakeProfitError(ValueError):
    """Błąd poziomu take profit."""
    def __init__(self, position: str, kierunek: str):
        super().__init__(f'Take profit dla pozycji {position} musi być {kierunek} ceny wejścia')


class DateOrderError(ValueError):
    """Błąd kolejności dat."""
    def __init__(self):
        super().__init__('Data końcowa musi być późniejsza niż data początkowa')


class MarketData(BaseModel):
    """Model danych rynkowych."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Walidacja symbolu."""
        if not v or len(v) < 3 or len(v) > 10 or not v.isalnum():
            raise PydanticCustomError('symbol_error', 'Symbol musi mieć od 3 do 10 znaków')
        return v
    
    @field_validator('timestamp')
    def validate_timestamp(cls, v):
        """Walidacja daty."""
        return validate_past_datetime(v)
    
    @field_validator('open', 'high', 'low', 'close')
    def validate_prices(cls, v):
        """Walidacja cen."""
        return validate_decimal_positive(v)
    
    @field_validator('volume')
    def validate_volume(cls, v):
        """Walidacja wolumenu."""
        return validate_decimal_range(v, Decimal('0'), Decimal('1000000'))
    
    @model_validator(mode='after')
    def validate_high_low(self) -> 'MarketData':
        """Walidacja relacji high-low."""
        if self.high < self.low:
            raise PydanticCustomError('high_low_error', 'High musi być wyższe niż low')
        return self


class Trade(BaseModel):
    """Model transakcji."""
    id: int
    symbol: str
    order_type: OrderType
    volume: Decimal
    entry_price: Decimal
    exit_price: Optional[Decimal] = None
    stop_loss: Decimal
    take_profit: Decimal
    open_time: datetime
    close_time: Optional[datetime] = None
    profit: Optional[Decimal] = None
    status: str
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Walidacja symbolu."""
        if not v or len(v) < 3 or len(v) > 10 or not v.isalnum():
            raise PydanticCustomError('symbol_error', 'Symbol musi mieć od 3 do 10 znaków')
        return v
    
    @field_validator('volume', 'entry_price', 'stop_loss', 'take_profit')
    def validate_required_decimals(cls, v):
        """Walidacja wymaganych wartości dziesiętnych."""
        return validate_decimal_positive(v)
    
    @field_validator('exit_price', 'profit')
    def validate_optional_decimals(cls, v):
        """Walidacja opcjonalnych wartości dziesiętnych."""
        if v is not None:
            return validate_decimal_range(v, Decimal('0'), Decimal('1000000'))
        return v
    
    @field_validator('open_time')
    def validate_open_time(cls, v):
        """Walidacja daty otwarcia."""
        return validate_past_datetime(v)
    
    @field_validator('close_time')
    def validate_time_order(cls, v, values):
        """Walidacja kolejności czasowej."""
        if v is not None and values.data.get('open_time') and v <= values.data['open_time']:
            raise TimeOrderError()
        return v


class SignalData(BaseModel):
    """Model sygnału tradingowego."""
    symbol: str
    timestamp: datetime
    action: SignalAction
    confidence: float
    entry_price: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    indicators: Dict[str, Any] = {}
    ai_analysis: Dict[str, Any] = {}
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Walidacja symbolu."""
        if not v or len(v) < 3 or len(v) > 10 or not v.isalnum():
            raise PydanticCustomError('symbol_error', 'Symbol musi mieć od 3 do 10 znaków')
        return v
    
    @field_validator('timestamp')
    def validate_timestamp(cls, v):
        """Walidacja daty."""
        return validate_past_datetime(v)
    
    @field_validator('confidence')
    def validate_confidence(cls, v):
        """Walidacja pewności predykcji."""
        return validate_confidence(v)
    
    @field_validator('entry_price')
    def validate_entry_price(cls, v):
        """Walidacja ceny wejścia."""
        return validate_decimal_positive(v)
    
    @field_validator('stop_loss')
    def validate_stop_loss(cls, v, values):
        """Walidacja poziomu stop loss względem ceny wejścia."""
        if values is None or not hasattr(values, 'data'):
            return v
        if not values.data.get('entry_price') or not values.data.get('action'):
            return v
        
        entry_price = values.data['entry_price']
        action = values.data['action']
        
        if action == SignalAction.BUY and v >= entry_price:
            raise StopLossError('długiej', 'poniżej')
        elif action == SignalAction.SELL and v <= entry_price:
            raise StopLossError('krótkiej', 'powyżej')
        return v
    
    @field_validator('take_profit')
    def validate_take_profit(cls, v, values):
        """Walidacja poziomu take profit względem ceny wejścia."""
        if values is None or not hasattr(values, 'data'):
            return v
        if not values.data.get('entry_price') or not values.data.get('action'):
            return v
        
        entry_price = values.data['entry_price']
        action = values.data['action']
        
        if action == SignalAction.BUY and v <= entry_price:
            raise TakeProfitError('długiej', 'powyżej')
        elif action == SignalAction.SELL and v >= entry_price:
            raise TakeProfitError('krótkiej', 'poniżej')
        return v
    
    @field_validator('volume')
    def validate_volume(cls, v):
        """Walidacja wolumenu."""
        if v is not None:
            return validate_decimal_range(v, Decimal('0'), Decimal('1000000'))
        return v


class AccountInfo(BaseModel):
    """Model informacji o koncie."""
    balance: Decimal
    equity: Decimal
    margin: Decimal
    free_margin: Decimal
    margin_level: Optional[float] = None
    leverage: int
    currency: str
    
    @field_validator('balance', 'equity', 'margin', 'free_margin')
    def validate_decimals(cls, v):
        """Walidacja wartości dziesiętnych."""
        return validate_decimal_range(v, Decimal('0'), Decimal('1000000000'))
    
    @field_validator('leverage')
    def validate_leverage(cls, v):
        """Walidacja dźwigni."""
        return validate_leverage(v)


class BacktestResult(BaseModel):
    """Model wyników backtestingu."""
    start_date: datetime
    end_date: datetime
    initial_balance: Decimal
    final_balance: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    
    @field_validator('initial_balance', 'final_balance')
    def validate_decimals(cls, v):
        """Walidacja wartości dziesiętnych."""
        return validate_decimal_range(v, Decimal('0'), Decimal('1000000000'))
    
    @field_validator('total_trades', 'winning_trades', 'losing_trades')
    def validate_trade_counts(cls, v):
        """Walidacja liczby transakcji."""
        return validate_trade_count(v)
    
    @field_validator('start_date', 'end_date')
    def validate_dates(cls, v):
        """Walidacja dat."""
        return validate_past_datetime(v)
    
    @field_validator('end_date')
    def validate_date_order(cls, v, values):
        """Walidacja kolejności dat."""
        if not values.data.get('start_date'):
            return v
        if v <= values.data['start_date']:
            raise DateOrderError()
        return v
        
    @field_validator('winning_trades', 'losing_trades')
    def validate_trade_sum(cls, v, values):
        """Walidacja sumy transakcji."""
        if not values.data.get('total_trades'):
            return v
            
        total = values.data['total_trades']
        winning = values.data.get('winning_trades', 0)
        losing = values.data.get('losing_trades', 0)
        
        if winning + losing > total:
            raise ValueError('Suma wygranych i przegranych transakcji nie może być większa niż całkowita liczba transakcji')
        return v 

    @model_validator(mode='after')
    def validate_trades_total(self) -> 'BacktestResult':
        """Walidacja sumy transakcji po zwalidowaniu wszystkich pól."""
        if self.winning_trades + self.losing_trades > self.total_trades:
            raise ValueError('Suma wygranych i przegranych transakcji nie może być większa niż całkowita liczba transakcji')
        return self 


class Position(BaseModel):
    """Model pozycji tradingowej."""
    id: str
    timestamp: datetime
    symbol: str
    trade_type: TradeType
    volume: Decimal
    entry_price: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[Decimal] = None
    profit: Optional[Decimal] = None
    pips: Optional[Decimal] = None
    point_value: Decimal = Decimal('0.0001')  # Domyślna wartość punktu dla par walutowych
    
    @field_validator('symbol')
    def validate_symbol(cls, v):
        """Walidacja symbolu."""
        return validate_symbol(v)
    
    @field_validator('timestamp')
    def validate_timestamp(cls, v):
        """Walidacja daty."""
        return validate_past_datetime(v)
    
    @field_validator('entry_price', 'volume', 'stop_loss', 'take_profit')
    def validate_required_decimals(cls, v):
        """Walidacja wymaganych wartości dziesiętnych."""
        return validate_decimal_positive(v)
    
    @field_validator('exit_price', 'profit', 'pips')
    def validate_optional_decimals(cls, v):
        """Walidacja opcjonalnych wartości dziesiętnych."""
        if v is not None:
            return validate_decimal_range(v, Decimal('-1000000'), Decimal('1000000'))
        return v
    
    @model_validator(mode='after')
    def validate_position(self) -> 'Position':
        """Walidacja całej pozycji."""
        # Dla zamkniętych pozycji nie sprawdzamy SL/TP
        if self.status == PositionStatus.CLOSED:
            return self

        # Walidacja stop loss
        if self.trade_type == TradeType.BUY:
            if self.stop_loss is not None and self.stop_loss >= self.entry_price:
                raise ValueError("Stop loss dla pozycji BUY musi być poniżej ceny wejścia")
        else:  # SELL
            if self.stop_loss is not None and self.stop_loss <= self.entry_price:
                raise ValueError("Stop loss dla pozycji SELL musi być powyżej ceny wejścia")

        # Walidacja take profit
        if self.trade_type == TradeType.BUY:
            if self.take_profit is not None and self.take_profit <= self.entry_price:
                raise ValueError("Take profit dla pozycji BUY musi być powyżej ceny wejścia")
        else:  # SELL
            if self.take_profit is not None and self.take_profit >= self.entry_price:
                raise ValueError("Take profit dla pozycji SELL musi być poniżej ceny wejścia")

        return self 