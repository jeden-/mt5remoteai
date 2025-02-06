"""
Testy dla modułu data_models.py
"""
from datetime import datetime, timedelta
from decimal import Decimal
import pytest
from pydantic import ValidationError, ValidationInfo
from src.models.data_models import (
    TimeOrderError,
    StopLossError,
    TakeProfitError,
    DateOrderError,
    MarketData,
    Trade,
    SignalData,
    AccountInfo,
    BacktestResult,
    Position,
    TradeType,
    PositionStatus
)
from src.models.enums import OrderType, SignalAction
from unittest.mock import Mock

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

def test_backtest_result_trade_sum_validation():
    """Test walidacji sumy transakcji."""
    data = {
        'start_date': datetime.now() - timedelta(days=2),
        'end_date': datetime.now() - timedelta(days=1),
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 5,
        'winning_trades': 3,
        'losing_trades': 3,  # 3 + 3 > 5
        'equity_curve': [],
        'trades': []
    }
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**data)
    assert "Suma wygranych i przegranych transakcji nie może być większa niż całkowita liczba transakcji" in str(exc_info.value)

def test_backtest_result_trade_sum_validation_model():
    """Test walidacji sumy transakcji na poziomie modelu."""
    data = {
        'start_date': datetime.now() - timedelta(days=2),
        'end_date': datetime.now() - timedelta(days=1),
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 5,
        'winning_trades': 3,
        'losing_trades': 3,  # 3 + 3 > 5
        'equity_curve': [],
        'trades': []
    }
    with pytest.raises(ValidationError) as exc_info:
        BacktestResult(**data)
    assert "Suma wygranych i przegranych transakcji nie może być większa niż całkowita liczba transakcji" in str(exc_info.value)

@pytest.mark.asyncio
async def test_position_validation():
    """Test walidacji pozycji tradingowej."""
    # Test dla pozycji BUY z poprawnymi danymi
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1050'),
        status=PositionStatus.OPEN
    )
    assert position.validate_position() == position

    # Test dla pozycji SELL z poprawnymi danymi
    position = Position(
        id="test_2",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.SELL,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0950'),
        status=PositionStatus.OPEN
    )
    assert position.validate_position() == position

    # Test dla zamkniętej pozycji (nie sprawdza SL/TP)
    position = Position(
        id="test_3",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        stop_loss=None,
        take_profit=None,
        status=PositionStatus.CLOSED,
        exit_price=Decimal('1.1050'),
        profit=Decimal('50.0'),
        pips=Decimal('50.0')
    )
    assert position.validate_position() == position

@pytest.mark.asyncio
async def test_position_validation_errors():
    """Test błędów walidacji pozycji tradingowej."""
    # Test dla pozycji BUY z nieprawidłowym SL
    with pytest.raises(ValueError, match="Stop loss dla pozycji BUY musi być poniżej ceny wejścia"):
        Position(
            id="test_1",
            timestamp=datetime.now(),
            symbol="EURUSD",
            trade_type=TradeType.BUY,
            volume=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.1050'),  # SL powyżej ceny wejścia
            take_profit=Decimal('1.1100'),
            status=PositionStatus.OPEN
        )

    # Test dla pozycji SELL z nieprawidłowym SL
    with pytest.raises(ValueError, match="Stop loss dla pozycji SELL musi być powyżej ceny wejścia"):
        Position(
            id="test_2",
            timestamp=datetime.now(),
            symbol="EURUSD",
            trade_type=TradeType.SELL,
            volume=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),  # SL poniżej ceny wejścia
            take_profit=Decimal('1.0900'),
            status=PositionStatus.OPEN
        )

    # Test dla pozycji BUY z nieprawidłowym TP
    with pytest.raises(ValueError, match="Take profit dla pozycji BUY musi być powyżej ceny wejścia"):
        Position(
            id="test_3",
            timestamp=datetime.now(),
            symbol="EURUSD",
            trade_type=TradeType.BUY,
            volume=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.0950'),
            take_profit=Decimal('1.0950'),  # TP poniżej ceny wejścia
            status=PositionStatus.OPEN
        )

    # Test dla pozycji SELL z nieprawidłowym TP
    with pytest.raises(ValueError, match="Take profit dla pozycji SELL musi być poniżej ceny wejścia"):
        Position(
            id="test_4",
            timestamp=datetime.now(),
            symbol="EURUSD",
            trade_type=TradeType.SELL,
            volume=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            stop_loss=Decimal('1.1050'),
            take_profit=Decimal('1.1050'),  # TP powyżej ceny wejścia
            status=PositionStatus.OPEN
        )

@pytest.mark.asyncio
async def test_position_optional_fields():
    """Test opcjonalnych pól w pozycji tradingowej."""
    # Test dla pozycji bez SL/TP
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        status=PositionStatus.OPEN
    )
    assert position.stop_loss is None
    assert position.take_profit is None

    # Test dla pozycji z częściowymi danymi zamknięcia
    position = Position(
        id="test_2",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        status=PositionStatus.CLOSED,
        exit_price=Decimal('1.1050')
    )
    assert position.exit_price == Decimal('1.1050')
    assert position.profit is None
    assert position.pips is None

@pytest.mark.asyncio
async def test_position_point_value():
    """Test wartości punktu dla różnych instrumentów."""
    # Test dla pary walutowej (domyślna wartość)
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        status=PositionStatus.OPEN
    )
    assert position.point_value == Decimal('0.0001')

    # Test z niestandardową wartością punktu
    position = Position(
        id="test_2",
        timestamp=datetime.now(),
        symbol="XAUUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1900.00'),
        status=PositionStatus.OPEN,
        point_value=Decimal('0.01')
    )
    assert position.point_value == Decimal('0.01')

@pytest.mark.asyncio
async def test_position_validation_closed():
    """Test walidacji zamkniętej pozycji."""
    # Test dla zamkniętej pozycji - walidacja powinna przejść bez sprawdzania SL/TP
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        status=PositionStatus.CLOSED,
        exit_price=Decimal('1.1050')
    )
    assert position.validate_position() == position

@pytest.mark.asyncio
async def test_position_validation_no_sl_tp():
    """Test walidacji pozycji bez SL/TP."""
    # Test dla pozycji bez SL/TP - powinno przejść dla zamkniętej pozycji
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        status=PositionStatus.CLOSED
    )
    assert position.validate_position() == position

@pytest.mark.asyncio
async def test_position_validation_missing_entry_price():
    """Test walidacji pozycji bez ceny wejścia."""
    # Test dla pozycji bez ceny wejścia - powinno rzucić wyjątek
    with pytest.raises(ValidationError) as exc_info:
        Position(
            id="test_1",
            timestamp=datetime.now(),
            symbol="EURUSD",
            trade_type=TradeType.BUY,
            volume=Decimal('1.0'),
            status=PositionStatus.OPEN
        )
    assert "Field required" in str(exc_info.value)

@pytest.mark.asyncio
async def test_position_validation_missing_volume():
    """Test walidacji pozycji bez wolumenu."""
    # Test dla pozycji bez wolumenu - powinno rzucić wyjątek
    with pytest.raises(ValidationError) as exc_info:
        Position(
            id="test_1",
            timestamp=datetime.now(),
            symbol="EURUSD",
            trade_type=TradeType.BUY,
            entry_price=Decimal('1.1000'),
            status=PositionStatus.OPEN
        )
    assert "Field required" in str(exc_info.value)

@pytest.mark.asyncio
async def test_position_validation_invalid_status():
    """Test walidacji pozycji z nieprawidłowym statusem."""
    # Test dla pozycji z nieprawidłowym statusem - powinno rzucić wyjątek
    with pytest.raises(ValidationError) as exc_info:
        Position(
            id="test_1",
            timestamp=datetime.now(),
            symbol="EURUSD",
            trade_type=TradeType.BUY,
            volume=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            status="INVALID"
        )
    assert "Input should be" in str(exc_info.value)

@pytest.mark.asyncio
async def test_position_validation_invalid_trade_type():
    """Test walidacji pozycji z nieprawidłowym typem transakcji."""
    # Test dla pozycji z nieprawidłowym typem transakcji - powinno rzucić wyjątek
    with pytest.raises(ValidationError) as exc_info:
        Position(
            id="test_1",
            timestamp=datetime.now(),
            symbol="EURUSD",
            trade_type="INVALID",
            volume=Decimal('1.0'),
            entry_price=Decimal('1.1000'),
            status=PositionStatus.OPEN
        )
    assert "Input should be" in str(exc_info.value)

@pytest.mark.asyncio
async def test_position_validation_with_exit_data():
    """Test walidacji pozycji z danymi wyjścia."""
    # Test dla pozycji z danymi wyjścia
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        status=PositionStatus.CLOSED,
        exit_price=Decimal('1.1050'),
        profit=Decimal('50.0'),
        pips=Decimal('50.0')
    )
    assert position.exit_price == Decimal('1.1050')
    assert position.profit == Decimal('50.0')
    assert position.pips == Decimal('50.0')

@pytest.mark.asyncio
async def test_position_validation_partial_exit_data():
    """Test walidacji pozycji z częściowymi danymi wyjścia."""
    # Test dla pozycji tylko z ceną wyjścia
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        status=PositionStatus.CLOSED,
        exit_price=Decimal('1.1050')
    )
    assert position.exit_price == Decimal('1.1050')
    assert position.profit is None
    assert position.pips is None

@pytest.mark.asyncio
async def test_position_validation_open_position():
    """Test walidacji otwartej pozycji."""
    # Test dla otwartej pozycji z SL/TP
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1050'),
        status=PositionStatus.OPEN
    )
    assert position.validate_position() == position

@pytest.mark.asyncio
async def test_market_data_validation_high_low():
    """Test walidacji relacji high-low w MarketData."""
    # Test dla poprawnych danych
    data = MarketData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        open=Decimal('1.1000'),
        high=Decimal('1.1100'),
        low=Decimal('1.0900'),
        close=Decimal('1.1050'),
        volume=Decimal('1000')
    )
    assert data.validate_high_low() == data

    # Test dla niepoprawnej relacji high-low
    with pytest.raises(ValidationError) as exc_info:
        MarketData(
            symbol='EURUSD',
            timestamp=datetime.now() - timedelta(minutes=5),
            open=Decimal('1.1000'),
            high=Decimal('1.0900'),  # High niższe niż low
            low=Decimal('1.1000'),
            close=Decimal('1.1050'),
            volume=Decimal('1000')
        )
    assert "High musi być wyższe niż low" in str(exc_info.value)

@pytest.mark.asyncio
async def test_trade_validation_optional_fields():
    """Test walidacji opcjonalnych pól w Trade."""
    # Test dla opcjonalnego exit_price
    trade = Trade(
        id=1,
        symbol='EURUSD',
        order_type=OrderType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1050'),
        open_time=datetime.now() - timedelta(hours=1),
        status='OPEN'
    )
    assert trade.exit_price is None
    assert trade.validate_optional_decimals(None) is None

    # Test dla poprawnego exit_price
    trade = Trade(
        id=1,
        symbol='EURUSD',
        order_type=OrderType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1050'),
        open_time=datetime.now() - timedelta(hours=1),
        exit_price=Decimal('1.1100'),
        status='CLOSED'
    )
    assert trade.exit_price == Decimal('1.1100')

@pytest.mark.asyncio
async def test_signal_data_validation_without_entry_price():
    """Test walidacji SignalData bez ceny wejścia."""
    # Test walidacji stop_loss bez entry_price
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=Decimal('1.1000'),  # entry_price jest wymagane
        stop_loss=Decimal('1.0950')
    )
    # Symulujemy brak entry_price w values
    values = Mock()
    values.data = {'action': SignalAction.BUY}
    assert signal.validate_stop_loss(Decimal('1.0950'), values) == Decimal('1.0950')

    # Test walidacji take_profit bez entry_price
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=Decimal('1.1000'),  # entry_price jest wymagane
        take_profit=Decimal('1.1050')
    )
    # Symulujemy brak entry_price w values
    values = Mock()
    values.data = {'action': SignalAction.BUY}
    assert signal.validate_take_profit(Decimal('1.1050'), values) == Decimal('1.1050')

@pytest.mark.asyncio
async def test_signal_data_validation_without_action():
    """Test walidacji SignalData bez akcji."""
    # Test walidacji stop_loss bez action
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,  # action jest wymagane
        confidence=0.85,
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950')
    )
    # Symulujemy brak action w values
    values = Mock()
    values.data = {'entry_price': Decimal('1.1000')}
    assert signal.validate_stop_loss(Decimal('1.0950'), values) == Decimal('1.0950')

    # Test walidacji take_profit bez action
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,  # action jest wymagane
        confidence=0.85,
        entry_price=Decimal('1.1000'),
        take_profit=Decimal('1.1050')
    )
    # Symulujemy brak action w values
    values = Mock()
    values.data = {'entry_price': Decimal('1.1000')}
    assert signal.validate_take_profit(Decimal('1.1050'), values) == Decimal('1.1050')

@pytest.mark.asyncio
async def test_trade_validation_time_order():
    """Test walidacji kolejności czasowej w Trade."""
    # Test dla braku open_time w values
    trade = Trade(
        id=1,
        symbol='EURUSD',
        order_type=OrderType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1050'),
        open_time=datetime.now() - timedelta(hours=1),
        close_time=datetime.now(),
        status='CLOSED'
    )
    # Symulujemy brak open_time w values
    values = Mock()
    values.data = {}
    assert trade.validate_time_order(datetime.now(), values) == datetime.now()

@pytest.mark.asyncio
async def test_signal_data_validation_volume():
    """Test walidacji wolumenu w SignalData."""
    # Test dla None volume
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=Decimal('1.1000')
    )
    assert signal.volume is None
    assert signal.validate_volume(None) is None

@pytest.mark.asyncio
async def test_position_validation_none_values():
    """Test walidacji pozycji z None values."""
    # Test dla pozycji z None values
    position = Position(
        id="test_1",
        timestamp=datetime.now(),
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=Decimal('1.0'),
        entry_price=Decimal('1.1000'),
        status=PositionStatus.OPEN
    )
    assert position.validate_optional_decimals(None) is None

@pytest.mark.asyncio
async def test_signal_data_validation_stop_loss_without_values():
    """Test walidacji stop loss w SignalData bez values."""
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950')
    )
    # Test dla braku values
    assert signal.validate_stop_loss(Decimal('1.0950'), None) == Decimal('1.0950')

@pytest.mark.asyncio
async def test_signal_data_validation_take_profit_without_values():
    """Test walidacji take profit w SignalData bez values."""
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=Decimal('1.1000'),
        take_profit=Decimal('1.1050')
    )
    # Test dla braku values
    assert signal.validate_take_profit(Decimal('1.1050'), None) == Decimal('1.1050')

@pytest.mark.asyncio
async def test_signal_data_validation_stop_loss_with_entry_price():
    """Test walidacji stop loss w SignalData z entry_price."""
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950')
    )
    # Test dla entry_price w values
    values = Mock()
    values.data = {'entry_price': Decimal('1.1000'), 'action': SignalAction.BUY}
    assert signal.validate_stop_loss(Decimal('1.0950'), values) == Decimal('1.0950')

@pytest.mark.asyncio
async def test_signal_data_validation_take_profit_with_entry_price():
    """Test walidacji take profit w SignalData z entry_price."""
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now() - timedelta(minutes=5),
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=Decimal('1.1000'),
        take_profit=Decimal('1.1050')
    )
    # Test dla entry_price w values
    values = Mock()
    values.data = {'entry_price': Decimal('1.1000'), 'action': SignalAction.BUY}
    assert signal.validate_take_profit(Decimal('1.1050'), values) == Decimal('1.1050')

def test_signal_data_validation_date_order_without_start_date():
    """Test walidacji kolejności dat bez daty początkowej."""
    data = {
        'symbol': 'EURUSD',
        'action': SignalAction.BUY,
        'confidence': 0.8,
        'timestamp': datetime.now(),
        'entry_price': Decimal('1.1000')
    }
    signal = SignalData(**data)
    assert signal.timestamp == data['timestamp']

def test_signal_data_validation_trade_sum_without_total():
    """Test walidacji sumy transakcji bez całkowitej liczby transakcji."""
    data = {
        'symbol': 'EURUSD',
        'action': SignalAction.BUY,
        'confidence': 0.8,
        'timestamp': datetime.now(),
        'entry_price': Decimal('1.1000'),
        'indicators': {'winning_trades': 5, 'losing_trades': 3}
    }
    signal = SignalData(**data)
    assert signal.indicators['winning_trades'] == 5
    assert signal.indicators['losing_trades'] == 3

def test_backtest_result_date_order_without_start_date():
    """Test walidacji kolejności dat bez daty początkowej w values."""
    data = {
        'start_date': datetime.now() - timedelta(days=2),
        'end_date': datetime.now() - timedelta(days=1),
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 10,
        'winning_trades': 6,
        'losing_trades': 4,
        'equity_curve': [],
        'trades': []
    }
    result = BacktestResult(**data)
    # Symulujemy brak start_date w values
    values = Mock()
    values.data = {}
    assert result.validate_date_order(result.end_date, values) == result.end_date

def test_backtest_result_trade_sum_without_total():
    """Test walidacji sumy transakcji bez total_trades w values."""
    data = {
        'start_date': datetime.now() - timedelta(days=2),
        'end_date': datetime.now() - timedelta(days=1),
        'initial_balance': Decimal('10000'),
        'final_balance': Decimal('11000'),
        'total_trades': 10,
        'winning_trades': 6,
        'losing_trades': 4,
        'equity_curve': [],
        'trades': []
    }
    result = BacktestResult(**data)
    # Symulujemy brak total_trades w values
    values = Mock()
    values.data = {}
    assert result.validate_trade_sum(result.winning_trades, values) == result.winning_trades

def test_trade_invalid_symbol_empty():
    """Test walidacji pustego symbolu."""
    invalid_data = {
        'id': 1,
        'symbol': '',
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

def test_signal_data_invalid_symbol_empty():
    """Test walidacji pustego symbolu."""
    invalid_data = {
        'symbol': '',
        'timestamp': datetime.now(),
        'action': SignalAction.BUY,
        'confidence': 0.8,
        'entry_price': Decimal('1.1000')
    }
    with pytest.raises(ValidationError) as exc_info:
        SignalData(**invalid_data)
    assert "Symbol musi mieć od 3 do 10 znaków" in str(exc_info.value) 