"""
Testy dla modułu data_models.py
"""
from datetime import datetime, timedelta
from decimal import Decimal
import pytest
from pydantic import ValidationError
from src.models.data_models import (
    TimeOrderError,
    StopLossError,
    TakeProfitError,
    DateOrderError,
    MarketData,
    Trade,
    SignalData,
    AccountInfo,
    BacktestResult
)
from src.models.enums import OrderType, SignalAction

# Testy dla klas błędów
def test_time_order_error():
    """Test błędu kolejności czasowej."""
    error = TimeOrderError()
    assert str(error) == 'Data zamknięcia musi być późniejsza niż data otwarcia'

def test_stop_loss_error():
    """Test błędu poziomu stop loss."""
    error = StopLossError('długiej', 'poniżej')
    assert str(error) == 'Stop loss dla pozycji długiej musi być poniżej ceny wejścia'
    
    error = StopLossError('krótkiej', 'powyżej')
    assert str(error) == 'Stop loss dla pozycji krótkiej musi być powyżej ceny wejścia'

def test_take_profit_error():
    """Test błędu poziomu take profit."""
    error = TakeProfitError('długiej', 'powyżej')
    assert str(error) == 'Take profit dla pozycji długiej musi być powyżej ceny wejścia'
    
    error = TakeProfitError('krótkiej', 'poniżej')
    assert str(error) == 'Take profit dla pozycji krótkiej musi być poniżej ceny wejścia'

def test_date_order_error():
    """Test błędu kolejności dat."""
    error = DateOrderError()
    assert str(error) == 'Data końcowa musi być późniejsza niż data początkowa'

# Funkcje pomocnicze do generowania danych testowych
@pytest.fixture
def sample_datetime():
    """Przykładowa data do testów."""
    return datetime.now() - timedelta(days=1)

@pytest.fixture
def sample_market_data(sample_datetime):
    """Przykładowe dane rynkowe do testów."""
    return {
        'symbol': 'EURUSD',
        'timestamp': sample_datetime,
        'open': Decimal('1.1000'),
        'high': Decimal('1.1100'),
        'low': Decimal('1.0900'),
        'close': Decimal('1.1050'),
        'volume': Decimal('1000')
    }

@pytest.fixture
def sample_trade_data(sample_datetime):
    """Przykładowe dane transakcji do testów."""
    return {
        'id': 1,
        'symbol': 'EURUSD',
        'order_type': OrderType.BUY,
        'volume': Decimal('0.1'),
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.0900'),
        'take_profit': Decimal('1.1200'),
        'open_time': sample_datetime,
        'status': 'OPEN'
    }

@pytest.fixture
def sample_signal_data(sample_datetime):
    """Przykładowe dane sygnału do testów."""
    return {
        'symbol': 'EURUSD',
        'timestamp': sample_datetime,
        'action': SignalAction.BUY,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.0900'),
        'take_profit': Decimal('1.1200'),
        'volume': Decimal('0.1')
    }

@pytest.fixture
def sample_account_info():
    """Przykładowe dane konta do testów."""
    return {
        'balance': Decimal('10000'),
        'equity': Decimal('10100'),
        'margin': Decimal('100'),
        'free_margin': Decimal('9900'),
        'margin_level': 101.0,
        'leverage': 100,
        'currency': 'USD'
    }

@pytest.fixture
def sample_backtest_result(sample_datetime):
    """Przykładowe dane wyniku backtestingu do testów."""
    return {
        'start_date': sample_datetime - timedelta(days=30),
        'end_date': sample_datetime,
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 100,
        'winning_trades': 60,
        'losing_trades': 40,
        'profit_factor': 1.5,
        'max_drawdown': 5.0,
        'equity_curve': [],
        'trades': []
    }

@pytest.fixture
def valid_market_data():
    """Fixture z poprawnymi danymi rynkowymi."""
    return {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(minutes=5),
        'open': Decimal('1.1000'),
        'high': Decimal('1.1100'),
        'low': Decimal('1.0900'),
        'close': Decimal('1.1050'),
        'volume': Decimal('1000')
    }

@pytest.fixture
def valid_trade_data():
    """Fixture z poprawnymi danymi transakcji."""
    return {
        'id': 1,
        'symbol': 'EURUSD',
        'order_type': OrderType.BUY,
        'volume': Decimal('0.1'),
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.0900'),
        'take_profit': Decimal('1.1200'),
        'open_time': datetime.now() - timedelta(hours=1),
        'status': 'OPEN'
    }

@pytest.fixture
def valid_signal_data():
    """Fixture z poprawnymi danymi sygnału."""
    return {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(minutes=5),
        'action': SignalAction.BUY,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.0900'),
        'take_profit': Decimal('1.1200'),
        'volume': Decimal('0.1'),
        'indicators': {'RSI': 65, 'MACD': 0.0012},
        'ai_analysis': {'sentiment': 'bullish', 'strength': 'strong'}
    }

@pytest.fixture
def valid_account_info():
    """Fixture z poprawnymi danymi konta."""
    return {
        'balance': Decimal('10000.00'),
        'equity': Decimal('10100.00'),
        'margin': Decimal('1000.00'),
        'free_margin': Decimal('9100.00'),
        'margin_level': 1010.0,
        'leverage': 100,
        'currency': 'USD'
    }

@pytest.fixture
def valid_backtest_result():
    """Fixture z poprawnymi wynikami backtestingu."""
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now() - timedelta(minutes=5)
    return {
        'start_date': start_date,
        'end_date': end_date,
        'initial_balance': Decimal('10000.00'),
        'final_balance': Decimal('11000.00'),
        'total_trades': 50,
        'winning_trades': 30,
        'losing_trades': 20,
        'profit_factor': 1.5,
        'max_drawdown': 5.0,
        'equity_curve': [
            {'timestamp': start_date, 'equity': Decimal('10000.00')},
            {'timestamp': end_date, 'equity': Decimal('11000.00')}
        ],
        'trades': [
            {
                'timestamp': start_date,
                'type': 'BUY',
                'profit': Decimal('100.00')
            }
        ]
    }

def test_market_data_valid(valid_market_data):
    """Test tworzenia poprawnego obiektu MarketData."""
    market_data = MarketData(**valid_market_data)
    assert market_data.symbol == 'EURUSD'
    assert isinstance(market_data.timestamp, datetime)
    assert market_data.open == Decimal('1.1000')
    assert market_data.high == Decimal('1.1100')
    assert market_data.low == Decimal('1.0900')
    assert market_data.close == Decimal('1.1050')
    assert market_data.volume == Decimal('1000')

def test_market_data_invalid():
    """Test walidacji niepoprawnych danych rynkowych."""
    # Niepoprawny symbol (za krótki)
    with pytest.raises(ValidationError) as exc_info:
        MarketData(
            symbol='EU',  # za krótki symbol
            timestamp=datetime.now() - timedelta(minutes=1),
            open=Decimal('1.1000'),
            high=Decimal('1.1100'),
            low=Decimal('1.0900'),
            close=Decimal('1.1050'),
            volume=Decimal('1000')
        )
    assert "Symbol musi mieć od 3 do 10 znaków" in str(exc_info.value)
    
    # Niepoprawna data (przyszła)
    with pytest.raises(ValidationError) as exc_info:
        MarketData(
            symbol='EURUSD',
            timestamp=datetime.now() + timedelta(days=1),
            open=Decimal('1.1000'),
            high=Decimal('1.1100'),
            low=Decimal('1.0900'),
            close=Decimal('1.1050'),
            volume=Decimal('1000')
        )
    assert "Data nie może być z przyszłości" in str(exc_info.value)
    
    # Niepoprawny wolumen (ujemny)
    with pytest.raises(ValidationError) as exc_info:
        MarketData(
            symbol='EURUSD',
            timestamp=datetime.now() - timedelta(minutes=1),
            open=Decimal('1.1000'),
            high=Decimal('1.1100'),
            low=Decimal('1.0900'),
            close=Decimal('1.1050'),
            volume=Decimal('-1000')
        )
    assert "Wartość musi być w zakresie" in str(exc_info.value)

def test_trade_valid(valid_trade_data):
    """Test tworzenia poprawnego obiektu Trade."""
    trade = Trade(**valid_trade_data)
    assert trade.id == 1
    assert trade.symbol == 'EURUSD'
    assert trade.order_type == OrderType.BUY
    assert trade.volume == Decimal('0.1')
    assert trade.entry_price == Decimal('1.1000')
    assert trade.stop_loss == Decimal('1.0900')
    assert trade.take_profit == Decimal('1.1200')
    assert isinstance(trade.open_time, datetime)
    assert trade.status == 'OPEN'

def test_trade_time_order(valid_trade_data):
    """Test walidacji kolejności czasowej w Trade."""
    data = valid_trade_data.copy()
    data['close_time'] = data['open_time'] - timedelta(minutes=5)
    
    with pytest.raises(ValidationError) as exc_info:
        Trade(**data)
    assert "Data zamknięcia musi być późniejsza niż data otwarcia" in str(exc_info.value)

def test_signal_data_valid(valid_signal_data):
    """Test tworzenia poprawnego obiektu SignalData."""
    signal = SignalData(**valid_signal_data)
    assert signal.symbol == 'EURUSD'
    assert isinstance(signal.timestamp, datetime)
    assert signal.action == SignalAction.BUY
    assert signal.confidence == 0.85
    assert signal.entry_price == Decimal('1.1000')
    assert signal.stop_loss == Decimal('1.0900')
    assert signal.take_profit == Decimal('1.1200')
    assert signal.volume == Decimal('0.1')
    assert 'RSI' in signal.indicators
    assert 'sentiment' in signal.ai_analysis

def test_signal_data_stop_loss(valid_signal_data):
    """Test walidacji poziomów stop loss w SignalData."""
    data = valid_signal_data.copy()
    
    # Niepoprawny SL dla pozycji długiej
    data['stop_loss'] = Decimal('1.1100')  # SL powyżej entry price
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**data)
    assert "Stop loss dla pozycji długiej musi być poniżej ceny wejścia" in str(exc_info.value)
    
    # Niepoprawny SL dla pozycji krótkiej
    data['action'] = SignalAction.SELL
    data['stop_loss'] = Decimal('1.0900')  # SL poniżej entry price
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**data)
    assert "Stop loss dla pozycji krótkiej musi być powyżej ceny wejścia" in str(exc_info.value)

def test_signal_data_take_profit(valid_signal_data):
    """Test walidacji poziomów take profit w SignalData."""
    data = valid_signal_data.copy()
    
    # Niepoprawny TP dla pozycji długiej
    data['take_profit'] = Decimal('1.0900')  # TP poniżej entry price
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**data)
    assert "Take profit dla pozycji długiej musi być powyżej ceny wejścia" in str(exc_info.value)
    
    # Niepoprawny TP dla pozycji krótkiej
    data['action'] = SignalAction.SELL
    data['take_profit'] = Decimal('1.1100')  # TP powyżej entry price
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**data)
    assert "Take profit dla pozycji krótkiej musi być poniżej ceny wejścia" in str(exc_info.value)

def test_account_info_valid(valid_account_info):
    """Test tworzenia poprawnego obiektu AccountInfo."""
    account = AccountInfo(**valid_account_info)
    assert account.balance == Decimal('10000.00')
    assert account.equity == Decimal('10100.00')
    assert account.margin == Decimal('1000.00')
    assert account.free_margin == Decimal('9100.00')
    assert account.margin_level == 1010.0
    assert account.leverage == 100
    assert account.currency == 'USD'
    
def test_account_info_invalid(valid_account_info):
    """Test walidacji niepoprawnych danych konta."""
    data = valid_account_info.copy()
    
    # Niepoprawna dźwignia
    data['leverage'] = 0
    with pytest.raises(ValidationError) as exc_info:
        AccountInfo(**data)
    assert "Dźwignia musi być w zakresie" in str(exc_info.value)
    
    # Niepoprawny balans
    data['leverage'] = 100
    data['balance'] = Decimal('-1000.00')
    with pytest.raises(ValidationError) as exc_info:
        AccountInfo(**data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

def test_backtest_result_valid(valid_backtest_result):
    """Test tworzenia poprawnego obiektu BacktestResult."""
    result = BacktestResult(**valid_backtest_result)
    assert result.start_date == valid_backtest_result['start_date']
    assert result.end_date == valid_backtest_result['end_date']
    assert result.initial_balance == Decimal('10000.00')
    assert result.final_balance == Decimal('11000.00')
    assert result.total_trades == 50
    assert result.winning_trades == 30
    assert result.losing_trades == 20
    assert result.profit_factor == 1.5
    assert result.max_drawdown == 5.0
    assert len(result.equity_curve) == 2
    assert len(result.trades) == 1

def test_backtest_result_date_order(valid_backtest_result):
    """Test walidacji kolejności dat w BacktestResult."""
    data = valid_backtest_result.copy()
    data['end_date'] = data['start_date'] - timedelta(days=1)
    
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**data)
    assert "Data końcowa musi być późniejsza niż data początkowa" in str(exc_info.value)

def test_backtest_result_trade_counts(valid_backtest_result):
    """Test walidacji liczby transakcji w BacktestResult."""
    data = valid_backtest_result.copy()
    
    # Suma wygranych i przegranych większa niż total
    data['total_trades'] = 40
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**data)
    assert "Suma wygranych i przegranych transakcji nie może być większa niż całkowita liczba transakcji" in str(exc_info.value)

# Testy dla MarketData
def test_market_data_creation(sample_market_data):
    """Test tworzenia obiektu MarketData z poprawnymi danymi."""
    market_data = MarketData(**sample_market_data)
    assert market_data.symbol == 'EURUSD'
    assert isinstance(market_data.open, Decimal)
    assert isinstance(market_data.high, Decimal)
    assert isinstance(market_data.low, Decimal)
    assert isinstance(market_data.close, Decimal)
    assert isinstance(market_data.volume, Decimal)

def test_market_data_invalid_symbol():
    """Test walidacji nieprawidłowego symbolu."""
    invalid_data = {
        'symbol': 'invalid!@#',
        'timestamp': datetime.now() - timedelta(days=1),
        'open': Decimal('1.1000'),
        'high': Decimal('1.1100'),
        'low': Decimal('1.0900'),
        'close': Decimal('1.1050'),
        'volume': Decimal('1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        MarketData(**invalid_data)
    assert "Symbol musi mieć od 3 do 10 znaków" in str(exc_info.value)

def test_market_data_invalid_timestamp():
    """Test walidacji przyszłej daty."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() + timedelta(days=1),
        'open': Decimal('1.1000'),
        'high': Decimal('1.1100'),
        'low': Decimal('1.0900'),
        'close': Decimal('1.1050'),
        'volume': Decimal('1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        MarketData(**invalid_data)
    assert "Data nie może być z przyszłości" in str(exc_info.value)

def test_market_data_invalid_prices():
    """Test walidacji nieprawidłowych cen."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'open': Decimal('-1.1000'),
        'high': Decimal('1.1100'),
        'low': Decimal('1.0900'),
        'close': Decimal('1.1050'),
        'volume': Decimal('1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        MarketData(**invalid_data)
    assert "Wartość musi być dodatnia" in str(exc_info.value)

def test_market_data_invalid_volume():
    """Test walidacji nieprawidłowego wolumenu."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'open': Decimal('1.1000'),
        'high': Decimal('1.1100'),
        'low': Decimal('1.0900'),
        'close': Decimal('1.1050'),
        'volume': Decimal('-1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        MarketData(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

def test_market_data_high_low_validation():
    """Test walidacji relacji high-low."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'open': Decimal('1.1000'),
        'high': Decimal('1.0800'),  # High niższe niż low
        'low': Decimal('1.0900'),
        'close': Decimal('1.1050'),
        'volume': Decimal('1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        MarketData(**invalid_data)
    assert "High musi być wyższe niż low" in str(exc_info.value)

# Testy dla Trade
def test_trade_creation(sample_trade_data):
    """Test tworzenia obiektu Trade z poprawnymi danymi."""
    trade = Trade(**sample_trade_data)
    assert trade.id == 1
    assert trade.symbol == 'EURUSD'
    assert trade.order_type == OrderType.BUY
    assert isinstance(trade.volume, Decimal)
    assert isinstance(trade.entry_price, Decimal)
    assert isinstance(trade.stop_loss, Decimal)
    assert isinstance(trade.take_profit, Decimal)
    assert trade.exit_price is None
    assert trade.close_time is None
    assert trade.profit is None
    assert trade.status == 'OPEN'

def test_trade_with_exit_data(sample_trade_data, sample_datetime):
    """Test tworzenia obiektu Trade z danymi zamknięcia."""
    sample_trade_data.update({
        'exit_price': Decimal('1.1100'),
        'close_time': sample_datetime + timedelta(hours=1),
        'profit': Decimal('100')
    })
    trade = Trade(**sample_trade_data)
    assert isinstance(trade.exit_price, Decimal)
    assert isinstance(trade.close_time, datetime)
    assert isinstance(trade.profit, Decimal)

def test_trade_invalid_symbol():
    """Test walidacji nieprawidłowego symbolu."""
    invalid_data = {
        'id': 1,
        'symbol': 'invalid!@#',
        'order_type': OrderType.BUY,
        'volume': Decimal('0.1'),
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.0900'),
        'take_profit': Decimal('1.1200'),
        'open_time': datetime.now() - timedelta(days=1),
        'status': 'OPEN'
    }
    with pytest.raises(ValidationError) as exc_info:
        Trade(**invalid_data)
    assert "Symbol musi mieć od 3 do 10 znaków" in str(exc_info.value)

def test_trade_invalid_volume():
    """Test walidacji nieprawidłowego wolumenu."""
    invalid_data = {
        'id': 1,
        'symbol': 'EURUSD',
        'order_type': OrderType.BUY,
        'volume': Decimal('-0.1'),
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.0900'),
        'take_profit': Decimal('1.1200'),
        'open_time': datetime.now() - timedelta(days=1),
        'status': 'OPEN'
    }
    with pytest.raises(ValidationError) as exc_info:
        Trade(**invalid_data)
    assert "Wartość musi być dodatnia" in str(exc_info.value)

def test_trade_invalid_prices():
    """Test walidacji nieprawidłowych cen."""
    invalid_data = {
        'id': 1,
        'symbol': 'EURUSD',
        'order_type': OrderType.BUY,
        'volume': Decimal('0.1'),
        'entry_price': Decimal('-1.1000'),
        'stop_loss': Decimal('1.0900'),
        'take_profit': Decimal('1.1200'),
        'open_time': datetime.now() - timedelta(days=1),
        'status': 'OPEN'
    }
    with pytest.raises(ValidationError) as exc_info:
        Trade(**invalid_data)
    assert "Wartość musi być dodatnia" in str(exc_info.value)

def test_trade_invalid_time_order(sample_trade_data, sample_datetime):
    """Test walidacji nieprawidłowej kolejności czasowej."""
    sample_trade_data['close_time'] = sample_datetime - timedelta(hours=1)
    with pytest.raises(ValidationError) as exc_info:
        Trade(**sample_trade_data)
    assert "Data zamknięcia musi być późniejsza niż data otwarcia" in str(exc_info.value)

def test_trade_invalid_profit():
    """Test walidacji nieprawidłowego zysku."""
    invalid_data = {
        'id': 1,
        'symbol': 'EURUSD',
        'order_type': OrderType.BUY,
        'volume': Decimal('0.1'),
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.0900'),
        'take_profit': Decimal('1.1200'),
        'open_time': datetime.now() - timedelta(days=1),
        'close_time': datetime.now(),
        'profit': Decimal('-1000001'),
        'status': 'CLOSED'
    }
    with pytest.raises(ValidationError) as exc_info:
        Trade(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

# Testy dla SignalData
def test_signal_data_creation(sample_signal_data):
    """Test tworzenia obiektu SignalData z poprawnymi danymi."""
    signal = SignalData(**sample_signal_data)
    assert signal.symbol == 'EURUSD'
    assert signal.action == SignalAction.BUY
    assert isinstance(signal.confidence, float)
    assert isinstance(signal.entry_price, Decimal)
    assert isinstance(signal.stop_loss, Decimal)
    assert isinstance(signal.take_profit, Decimal)
    assert isinstance(signal.volume, Decimal)
    assert isinstance(signal.indicators, dict)
    assert isinstance(signal.ai_analysis, dict)

def test_signal_data_optional_fields():
    """Test tworzenia obiektu SignalData bez opcjonalnych pól."""
    minimal_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.BUY,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000')
    }
    signal = SignalData(**minimal_data)
    assert signal.stop_loss is None
    assert signal.take_profit is None
    assert signal.volume is None
    assert signal.indicators == {}
    assert signal.ai_analysis == {}

def test_signal_data_invalid_symbol():
    """Test walidacji nieprawidłowego symbolu."""
    invalid_data = {
        'symbol': 'invalid!@#',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.BUY,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Symbol musi mieć od 3 do 10 znaków" in str(exc_info.value)

def test_signal_data_invalid_confidence():
    """Test walidacji nieprawidłowej wartości pewności."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.BUY,
        'confidence': 1.5,  # Powyżej 1.0
        'entry_price': Decimal('1.1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Wartość pewności musi być w zakresie 0-1" in str(exc_info.value)

def test_signal_data_invalid_entry_price():
    """Test walidacji nieprawidłowej ceny wejścia."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.BUY,
        'confidence': 0.85,
        'entry_price': Decimal('-1.1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Wartość musi być dodatnia" in str(exc_info.value)

def test_signal_data_invalid_stop_loss_buy():
    """Test walidacji nieprawidłowego stop loss dla pozycji długiej."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.BUY,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.1100')  # SL powyżej ceny wejścia
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Stop loss dla pozycji długiej musi być poniżej ceny wejścia" in str(exc_info.value)

def test_signal_data_invalid_stop_loss_sell():
    """Test walidacji nieprawidłowego stop loss dla pozycji krótkiej."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.SELL,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000'),
        'stop_loss': Decimal('1.0900')  # SL poniżej ceny wejścia
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Stop loss dla pozycji krótkiej musi być powyżej ceny wejścia" in str(exc_info.value)

def test_signal_data_invalid_take_profit_buy():
    """Test walidacji nieprawidłowego take profit dla pozycji długiej."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.BUY,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000'),
        'take_profit': Decimal('1.0900')  # TP poniżej ceny wejścia
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Take profit dla pozycji długiej musi być powyżej ceny wejścia" in str(exc_info.value)

def test_signal_data_invalid_take_profit_sell():
    """Test walidacji nieprawidłowego take profit dla pozycji krótkiej."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.SELL,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000'),
        'take_profit': Decimal('1.1100')  # TP powyżej ceny wejścia
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Take profit dla pozycji krótkiej musi być poniżej ceny wejścia" in str(exc_info.value)

def test_signal_data_invalid_volume():
    """Test walidacji nieprawidłowego wolumenu."""
    invalid_data = {
        'symbol': 'EURUSD',
        'timestamp': datetime.now() - timedelta(days=1),
        'action': SignalAction.BUY,
        'confidence': 0.85,
        'entry_price': Decimal('1.1000'),
        'volume': Decimal('-0.1')
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

# Testy dla AccountInfo
def test_account_info_creation(sample_account_info):
    """Test tworzenia obiektu AccountInfo z poprawnymi danymi."""
    account = AccountInfo(**sample_account_info)
    assert isinstance(account.balance, Decimal)
    assert isinstance(account.equity, Decimal)
    assert isinstance(account.margin, Decimal)
    assert isinstance(account.free_margin, Decimal)
    assert isinstance(account.margin_level, float)
    assert isinstance(account.leverage, int)
    assert isinstance(account.currency, str)

def test_account_info_without_margin_level():
    """Test tworzenia obiektu AccountInfo bez margin_level."""
    data = {
        'balance': Decimal('10000'),
        'equity': Decimal('10100'),
        'margin': Decimal('100'),
        'free_margin': Decimal('9900'),
        'leverage': 100,
        'currency': 'USD'
    }
    account = AccountInfo(**data)
    assert account.margin_level is None

def test_account_info_invalid_balance():
    """Test walidacji nieprawidłowego balansu."""
    invalid_data = {
        'balance': Decimal('-10000'),
        'equity': Decimal('10100'),
        'margin': Decimal('100'),
        'free_margin': Decimal('9900'),
        'margin_level': 101.0,
        'leverage': 100,
        'currency': 'USD'
    }
    with pytest.raises(ValidationError) as exc_info:
        AccountInfo(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

def test_account_info_invalid_equity():
    """Test walidacji nieprawidłowego equity."""
    invalid_data = {
        'balance': Decimal('10000'),
        'equity': Decimal('-10100'),
        'margin': Decimal('100'),
        'free_margin': Decimal('9900'),
        'margin_level': 101.0,
        'leverage': 100,
        'currency': 'USD'
    }
    with pytest.raises(ValidationError) as exc_info:
        AccountInfo(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

def test_account_info_invalid_margin():
    """Test walidacji nieprawidłowego margin."""
    invalid_data = {
        'balance': Decimal('10000'),
        'equity': Decimal('10100'),
        'margin': Decimal('-100'),
        'free_margin': Decimal('9900'),
        'margin_level': 101.0,
        'leverage': 100,
        'currency': 'USD'
    }
    with pytest.raises(ValidationError) as exc_info:
        AccountInfo(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

def test_account_info_invalid_free_margin():
    """Test walidacji nieprawidłowego free margin."""
    invalid_data = {
        'balance': Decimal('10000'),
        'equity': Decimal('10100'),
        'margin': Decimal('100'),
        'free_margin': Decimal('-9900'),
        'margin_level': 101.0,
        'leverage': 100,
        'currency': 'USD'
    }
    with pytest.raises(ValidationError) as exc_info:
        AccountInfo(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

def test_account_info_invalid_leverage():
    """Test walidacji nieprawidłowej dźwigni."""
    invalid_data = {
        'balance': Decimal('10000'),
        'equity': Decimal('10100'),
        'margin': Decimal('100'),
        'free_margin': Decimal('9900'),
        'margin_level': 101.0,
        'leverage': 1001,  # Powyżej maksymalnej wartości
        'currency': 'USD'
    }
    with pytest.raises(ValidationError) as exc_info:
        AccountInfo(**invalid_data)
    assert "Dźwignia musi być w zakresie" in str(exc_info.value)

def test_account_info_too_high_values():
    """Test walidacji zbyt wysokich wartości."""
    invalid_data = {
        'balance': Decimal('2000000000'),  # Powyżej maksymalnej wartości
        'equity': Decimal('10100'),
        'margin': Decimal('100'),
        'free_margin': Decimal('9900'),
        'margin_level': 101.0,
        'leverage': 100,
        'currency': 'USD'
    }
    with pytest.raises(ValidationError) as exc_info:
        AccountInfo(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

# Testy dla BacktestResult
def test_backtest_result_creation(sample_backtest_result):
    """Test tworzenia obiektu BacktestResult z poprawnymi danymi."""
    result = BacktestResult(**sample_backtest_result)
    assert isinstance(result.start_date, datetime)
    assert isinstance(result.end_date, datetime)
    assert isinstance(result.initial_balance, Decimal)
    assert isinstance(result.final_balance, Decimal)
    assert isinstance(result.total_trades, int)
    assert isinstance(result.winning_trades, int)
    assert isinstance(result.losing_trades, int)
    assert isinstance(result.profit_factor, float)
    assert isinstance(result.max_drawdown, float)
    assert isinstance(result.equity_curve, list)
    assert isinstance(result.trades, list)

def test_backtest_result_without_optional_fields():
    """Test tworzenia obiektu BacktestResult bez opcjonalnych pól."""
    data = {
        'start_date': datetime.now() - timedelta(days=30),
        'end_date': datetime.now() - timedelta(days=1),
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 100,
        'winning_trades': 60,
        'losing_trades': 40,
        'equity_curve': [],
        'trades': []
    }
    result = BacktestResult(**data)
    assert result.profit_factor is None
    assert result.max_drawdown is None

def test_backtest_result_invalid_dates():
    """Test walidacji nieprawidłowych dat."""
    invalid_data = {
        'start_date': datetime.now() - timedelta(days=1),
        'end_date': datetime.now() - timedelta(days=30),  # End date przed start date
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 100,
        'winning_trades': 60,
        'losing_trades': 40,
        'equity_curve': [],
        'trades': []
    }
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**invalid_data)
    assert "Data końcowa musi być późniejsza niż data początkowa" in str(exc_info.value)

def test_backtest_result_future_dates():
    """Test walidacji dat z przyszłości."""
    invalid_data = {
        'start_date': datetime.now() + timedelta(days=1),
        'end_date': datetime.now() + timedelta(days=30),
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 100,
        'winning_trades': 60,
        'losing_trades': 40,
        'equity_curve': [],
        'trades': []
    }
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**invalid_data)
    assert "Data nie może być z przyszłości" in str(exc_info.value)

def test_backtest_result_invalid_balances():
    """Test walidacji nieprawidłowych wartości balance."""
    invalid_data = {
        'start_date': datetime.now() - timedelta(days=30),
        'end_date': datetime.now() - timedelta(days=1),
        'initial_balance': Decimal('-10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 100,
        'winning_trades': 60,
        'losing_trades': 40,
        'equity_curve': [],
        'trades': []
    }
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**invalid_data)
    assert "Wartość musi być w zakresie" in str(exc_info.value)

def test_backtest_result_invalid_trade_counts():
    """Test walidacji nieprawidłowej liczby transakcji."""
    invalid_data = {
        'start_date': datetime.now() - timedelta(days=30),
        'end_date': datetime.now() - timedelta(days=1),
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 50,
        'winning_trades': 60,  # Więcej niż total_trades
        'losing_trades': 40,
        'equity_curve': [],
        'trades': []
    }
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**invalid_data)
    assert "Suma wygranych i przegranych transakcji nie może być większa niż całkowita liczba transakcji" in str(exc_info.value)

def test_backtest_result_invalid_trade_sum():
    """Test walidacji sumy wygranych i przegranych transakcji."""
    invalid_data = {
        'start_date': datetime.now() - timedelta(days=30),
        'end_date': datetime.now() - timedelta(days=1),
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 100,
        'winning_trades': 70,
        'losing_trades': 40,  # Suma większa niż total_trades
        'equity_curve': [],
        'trades': []
    }
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**invalid_data)
    assert "Suma wygranych i przegranych" in str(exc_info.value) 