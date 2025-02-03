"""
Testy jednostkowe dla modułu backtester.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from unittest.mock import ANY

from src.backtest.backtester import Backtester
from src.strategies.basic_strategy import BasicStrategy
from src.utils.logger import TradingLogger
from src.backtest.performance_metrics import TradeResult

# Tylko dla testów asynchronicznych
async_tests = pytest.mark.asyncio(loop_scope="session")

@pytest.fixture
def mock_strategy():
    """Mock dla strategii tradingowej."""
    strategy = Mock(spec=BasicStrategy)
    strategy.generate_signals = AsyncMock()
    strategy.config = {'max_position_size': 1.0}
    strategy.name = "TestStrategy"  # Dodane dla loggera
    return strategy

@pytest.fixture
def mock_logger():
    """Mock dla loggera."""
    logger = Mock(spec=TradingLogger)
    logger.log_trade = Mock()
    logger.log_error = Mock()
    logger.info = Mock()  # Dodajemy mock dla metody info
    return logger

@pytest.fixture
def sample_data():
    """Przykładowe dane historyczne dla testów."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Generuj dane cenowe
    base_price = 100.0
    data['open'] = [base_price + i * 0.1 for i in range(len(dates))]
    data['high'] = data['open'] + 0.5
    data['low'] = data['open'] - 0.5
    data['close'] = data['open'] + np.random.uniform(-0.2, 0.2, len(dates))
    data['volume'] = np.random.uniform(1000, 5000, len(dates))
    
    # Dodaj wskaźniki techniczne
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['RSI'] = 50 + np.random.uniform(-20, 20, len(dates))
    data['MACD'] = np.random.uniform(-2, 2, len(dates))
    data['Signal_Line'] = np.random.uniform(-2, 2, len(dates))
    
    return data

@pytest.fixture
def small_sample_data():
    """Małe przykładowe dane dla testów."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Generuj dane cenowe
    data['open'] = [100.0, 101.0, 102.0, 103.0, 104.0]
    data['high'] = [101.0, 102.0, 103.0, 104.0, 105.0]
    data['low'] = [99.0, 100.0, 101.0, 102.0, 103.0]
    data['close'] = [100.5, 101.5, 102.5, 103.5, 104.5]
    data['volume'] = [1000, 1100, 1200, 1300, 1400]
    
    # Dodaj wskaźniki techniczne
    data['SMA_20'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['RSI'] = [50, 55, 60, 65, 70]
    data['MACD'] = [0.1, 0.2, 0.3, 0.4, 0.5]
    data['Signal_Line'] = [0.05, 0.15, 0.25, 0.35, 0.45]
    
    return data

def test_initialization(mock_strategy, mock_logger):
    """Test inicjalizacji Backtestera."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        timeframe='1H',
        initial_capital=10000,
        start_date=datetime(2024, 1, 1),
        logger=mock_logger
    )
    
    assert backtester.strategy == mock_strategy
    assert backtester.symbol == 'EURUSD'
    assert backtester.timeframe == '1H'
    assert backtester.initial_capital == 10000
    assert backtester.start_date == datetime(2024, 1, 1)
    assert backtester.logger == mock_logger
    assert backtester.trades == []
    assert backtester.data is None

@pytest.mark.asyncio
async def test_run_backtest_no_data(mock_strategy, mock_logger):
    """Test backtestingu gdy nie ma danych."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            await backtester.run_backtest()
        assert "Nie udało się załadować danych" in str(exc_info.value)

@pytest.mark.asyncio
async def test_run_backtest_with_trades(mock_strategy, mock_logger, sample_data):
    """Test backtestingu z transakcjami."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Symuluj sygnały
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 1.0},  # Otwórz długą
        {'action': 'HOLD'},  # Trzymaj
        {'action': 'CLOSE'},  # Zamknij
        {'action': 'SELL', 'volume': 1.0},  # Otwórz krótką
        {'action': 'HOLD'},  # Trzymaj
        {'action': 'CLOSE'}  # Zamknij
    ]
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=sample_data):
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert "total_return" in results
    assert "win_rate" in results
    assert "profit_factor" in results
    assert len(backtester.trades) > 0
    assert mock_logger.log_trade.call_count > 0

def test_should_close_position():
    """Test sprawdzania warunków zamknięcia pozycji."""
    mock_strategy = Mock(spec=BasicStrategy)
    mock_strategy.name = "TestStrategy"
    mock_logger = Mock(spec=TradingLogger)
    backtester = Backtester(mock_strategy, 'EURUSD', logger=mock_logger)

    # Test dla pozycji długiej
    position = {
        'direction': 'BUY',
        'stop_loss': 100,
        'take_profit': 110,
        'entry_price': 105,
        'size': 1.0
    }

    # Stop loss hit
    current_bar = pd.Series({'low': 99, 'high': 105, 'open': 105})
    signals = {'action': 'HOLD'}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 100  # Stop loss price

    # Take profit hit
    current_bar = pd.Series({'low': 105, 'high': 111, 'open': 105})
    signals = {'action': 'HOLD'}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 110  # Take profit price

    # No close condition
    current_bar = pd.Series({'low': 102, 'high': 108, 'open': 105})
    signals = {'action': 'HOLD'}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert not should_close

    # Test dla pozycji krótkiej
    position = {
        'direction': 'SELL',
        'stop_loss': 110,
        'take_profit': 100,
        'entry_price': 105,
        'size': 1.0
    }

    # Stop loss hit
    current_bar = pd.Series({'low': 105, 'high': 111, 'open': 105})
    signals = {'action': 'HOLD'}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 110  # Stop loss price

    # Take profit hit
    current_bar = pd.Series({'low': 99, 'high': 105, 'open': 105})
    signals = {'action': 'HOLD'}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 100  # Take profit price

    # No close condition
    current_bar = pd.Series({'low': 102, 'high': 108, 'open': 105})
    signals = {'action': 'HOLD'}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert not should_close

    # Test sygnału zamknięcia
    signals = {'action': 'CLOSE'}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 105  # Cena otwarcia następnej świecy

def test_calculate_profit():
    """Test obliczania zysku/straty."""
    mock_strategy = Mock(spec=BasicStrategy)
    mock_strategy.name = "TestStrategy"
    mock_logger = Mock(spec=TradingLogger)
    backtester = Backtester(mock_strategy, 'EURUSD', logger=mock_logger)
    
    # Test dla pozycji długiej
    position = {
        'direction': 'BUY',
        'entry_price': 100,
        'size': 1.0
    }
    
    # Zysk
    assert backtester._calculate_profit(position, 110) == 10.0
    
    # Strata
    assert backtester._calculate_profit(position, 90) == -10.0
    
    # Test dla pozycji krótkiej
    position['direction'] = 'SELL'
    
    # Zysk
    assert backtester._calculate_profit(position, 90) == 10.0
    
    # Strata
    assert backtester._calculate_profit(position, 110) == -10.0

@pytest.mark.asyncio
async def test_run_backtest_error_handling(mock_strategy, mock_logger, sample_data):
    """Test obsługi błędów podczas backtestingu."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Symuluj błąd w strategii
    mock_strategy.generate_signals.side_effect = Exception("Test error")
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=sample_data):
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert mock_logger.log_error.call_count > 0
    assert "total_return" in results

@pytest.mark.asyncio
async def test_stop_loss_handling(mock_strategy, mock_logger):
    """Test obsługi stop loss."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane z gwarantowanym stop lossem
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Ustaw ceny tak, aby stop loss został trafiony
    entry_price = 100.0
    stop_loss = entry_price - 1.0
    
    data['open'] = [entry_price] * len(dates)
    data['high'] = [entry_price + 0.5] * len(dates)
    data['low'] = [entry_price - 0.5] * len(dates)
    data['close'] = [entry_price] * len(dates)
    data['volume'] = [1000] * len(dates)
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close']
    data['SMA_50'] = data['close']
    data['RSI'] = [50] * len(dates)
    data['MACD'] = [0] * len(dates)
    data['Signal_Line'] = [0] * len(dates)
    
    # Ustaw cenę poniżej stop loss w trzeciej świecy
    data.loc[dates[2], 'low'] = stop_loss - 0.1
    
    # Ustaw sygnały strategii
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 1.0, 'stop_loss': stop_loss},  # Otwórz pozycję
        {'action': 'HOLD'},  # Trzymaj
        {'action': 'HOLD'},  # Stop loss zostanie trafiony
        {'action': 'HOLD'},  # Nie powinno być używane
        {'action': 'HOLD'}   # Nie powinno być używane
    ]
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=data):
        results = await backtester.run_backtest()
    
    assert len(backtester.trades) == 1
    assert backtester.trades[0].profit < 0  # Strata na stop lossie

@pytest.mark.asyncio
async def test_take_profit_handling(mock_strategy, mock_logger):
    """Test obsługi take profit."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane z gwarantowanym take profitem
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Ustaw ceny tak, aby take profit został trafiony
    entry_price = 100.0
    take_profit = entry_price + 1.0
    
    data['open'] = [entry_price] * len(dates)
    data['high'] = [entry_price + 0.5] * len(dates)
    data['low'] = [entry_price - 0.5] * len(dates)
    data['close'] = [entry_price] * len(dates)
    data['volume'] = [1000] * len(dates)
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close']
    data['SMA_50'] = data['close']
    data['RSI'] = [50] * len(dates)
    data['MACD'] = [0] * len(dates)
    data['Signal_Line'] = [0] * len(dates)
    
    # Ustaw cenę powyżej take profit w trzeciej świecy
    data.loc[dates[2], 'high'] = take_profit + 0.1
    
    # Ustaw sygnały strategii
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 1.0, 'take_profit': take_profit},  # Otwórz pozycję
        {'action': 'HOLD'},  # Trzymaj
        {'action': 'HOLD'},  # Take profit zostanie trafiony
        {'action': 'HOLD'},  # Nie powinno być używane
        {'action': 'HOLD'}   # Nie powinno być używane
    ]
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=data):
        results = await backtester.run_backtest()
    
    assert len(backtester.trades) == 1
    assert backtester.trades[0].profit > 0  # Zysk na take proficie

@async_tests
async def test_backtester_initialization(mock_strategy, mock_logger):
    """Test inicjalizacji backtestera"""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol="EURUSD",
        timeframe="1H",
        initial_capital=10000,
        start_date=datetime.now(),
        logger=mock_logger
    )
    
    assert backtester.strategy == mock_strategy
    assert backtester.symbol == "EURUSD"
    assert backtester.timeframe == "1H"
    assert backtester.initial_capital == 10000
    assert isinstance(backtester.trades, list)
    assert len(backtester.trades) == 0

@async_tests
async def test_empty_data_handling(mock_strategy, mock_logger, monkeypatch):
    """Test obsługi pustych danych"""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol="EURUSD",
        logger=mock_logger
    )
    
    # Mockuj HistoricalDataLoader
    mock_loader = AsyncMock()
    mock_loader.load_data = AsyncMock(return_value=pd.DataFrame())
    monkeypatch.setattr("src.backtest.backtester.HistoricalDataLoader", Mock(return_value=mock_loader))
    
    with pytest.raises(RuntimeError, match="Nie udało się załadować danych dla EURUSD"):
        await backtester.run_backtest()

@async_tests
async def test_buy_signal_handling(mock_strategy, mock_logger, sample_data, monkeypatch):
    """Test obsługi sygnału kupna"""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol="EURUSD",
        logger=mock_logger
    )
    
    # Mockuj HistoricalDataLoader
    mock_loader = AsyncMock()
    mock_loader.load_data = AsyncMock(return_value=sample_data)
    monkeypatch.setattr("src.backtest.backtester.HistoricalDataLoader", Mock(return_value=mock_loader))
    
    # Ustaw sygnały strategii
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 1.0},  # Pierwszy sygnał - kupno
        {'action': 'CLOSE'}  # Drugi sygnał - zamknięcie
    ]
    
    results = await backtester.run_backtest()
    
    assert len(backtester.trades) > 0
    assert isinstance(backtester.trades[0], TradeResult)
    assert backtester.trades[0].direction == 'BUY'
    assert mock_logger.log_trade.call_count > 0

@async_tests
async def test_sell_signal_handling(mock_strategy, mock_logger, sample_data, monkeypatch):
    """Test obsługi sygnału sprzedaży"""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol="EURUSD",
        logger=mock_logger
    )
    
    # Mockuj HistoricalDataLoader
    mock_loader = AsyncMock()
    mock_loader.load_data = AsyncMock(return_value=sample_data)
    monkeypatch.setattr("src.backtest.backtester.HistoricalDataLoader", Mock(return_value=mock_loader))
    
    # Ustaw sygnały strategii
    mock_strategy.generate_signals.side_effect = [
        {'action': 'SELL', 'volume': 1.0},  # Pierwszy sygnał - sprzedaż
        {'action': 'CLOSE'}  # Drugi sygnał - zamknięcie
    ]
    
    results = await backtester.run_backtest()
    
    assert len(backtester.trades) > 0
    assert isinstance(backtester.trades[0], TradeResult)
    assert backtester.trades[0].direction == 'SELL'
    assert mock_logger.log_trade.call_count > 0

@async_tests
async def test_error_handling(mock_strategy, mock_logger, sample_data, monkeypatch):
    """Test obsługi błędów"""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol="EURUSD",
        logger=mock_logger
    )
    
    # Mockuj HistoricalDataLoader
    mock_loader = AsyncMock()
    mock_loader.load_data = AsyncMock(return_value=sample_data)
    monkeypatch.setattr("src.backtest.backtester.HistoricalDataLoader", Mock(return_value=mock_loader))
    
    # Symuluj błąd w strategii
    mock_strategy.generate_signals.side_effect = Exception("Test error")
    
    results = await backtester.run_backtest()
    
    assert mock_logger.log_error.call_count > 0
    error_logs = [call.args[0] for call in mock_logger.log_error.call_args_list]
    assert any("Test error" in log for log in error_logs)

@async_tests
async def test_run_backtest_with_error_handling(mock_strategy, mock_logger, small_sample_data):
    """Test obsługi błędów podczas backtestingu."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Mockuj błąd podczas generowania sygnałów
    mock_strategy.generate_signals.side_effect = Exception("Test error")
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=small_sample_data)
        
        results = await backtester.run_backtest()
        
        assert isinstance(results, dict)
        mock_logger.log_error.assert_called()
        assert "Błąd podczas backtestingu" in mock_logger.log_error.call_args[0][0]

@async_tests
async def test_run_backtest_error_during_position_close(mock_strategy, mock_logger, small_sample_data):
    """Test obsługi błędów podczas zamykania pozycji."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Mockuj sygnały - najpierw BUY, potem błąd, potem WAIT
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 1.0},  # Otwórz pozycję
        Exception("Test error during position close"),  # Błąd podczas próby zamknięcia
        {'action': 'WAIT'},  # Ignoruj pozostałe świece
        {'action': 'WAIT'},
        {'action': 'WAIT'}
    ]
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=small_sample_data)
        
        results = await backtester.run_backtest()
        
        assert isinstance(results, dict)
        assert len(backtester.trades) == 1  # Pozycja powinna zostać zamknięta mimo błędu
        
        # Sprawdź oba wywołania log_error
        assert mock_logger.log_error.call_count == 2
        assert "Błąd podczas backtestingu" in mock_logger.log_error.call_args_list[0][0][0]
        assert "Zamykam pozycję z powodu błędu" in mock_logger.log_error.call_args_list[1][0][0] 

@async_tests
async def test_different_timeframes(mock_strategy, mock_logger):
    """Test backtestingu dla różnych interwałów czasowych."""
    timeframes = ["1M", "5M", "15M", "30M", "1H", "4H", "1D"]

    for timeframe in timeframes:
        backtester = Backtester(
            strategy=mock_strategy,
            symbol="EURUSD",
            timeframe=timeframe,
            logger=mock_logger
        )

        # Przygotuj dane testowe dla danego timeframe'a
        freq_map = {
            "1M": "1min",
            "5M": "5min",
            "15M": "15min",
            "30M": "30min",
            "1H": "1h",
            "4H": "4h",
            "1D": "1D"
        }

        dates = pd.date_range(
            start='2024-01-01',
            periods=100,
            freq=freq_map[timeframe]
        )
        data = pd.DataFrame(index=dates)

        # Generuj ceny z trendem wzrostowym
        periods = len(dates)
        base_prices = 1.0 + 0.01 * np.arange(periods) + 0.002 * np.sin(np.arange(periods)/20) + np.random.normal(0, 0.0005, periods)
        data['close'] = base_prices
        
        # Generuj open/high/low
        data.loc[data.index[0], 'open'] = base_prices[0]
        data.loc[data.index[1:], 'open'] = base_prices[:-1]
        
        daily_volatility = np.std(np.diff(base_prices))
        high_offset = np.abs(np.random.normal(0, daily_volatility, periods))
        low_offset = np.abs(np.random.normal(0, daily_volatility, periods))
        
        data['high'] = np.maximum(data['open'], data['close']) + high_offset
        data['low'] = np.minimum(data['open'], data['close']) - low_offset
        data['volume'] = np.random.uniform(1000, 5000, periods) * (1 + 0.5 * np.sin(np.arange(periods)/100))

        # Oblicz wskaźniki techniczne z opóźnieniem
        data['SMA_20'] = data['close'].shift(1).rolling(window=20).mean()
        data['SMA_50'] = data['close'].shift(1).rolling(window=50).mean()
        
        # RSI z opóźnieniem
        delta = data['close'].shift(1).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD z opóźnieniem
        exp1 = data['close'].shift(1).ewm(span=12, adjust=False).mean()
        exp2 = data['close'].shift(1).ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].shift(1).ewm(span=9, adjust=False).mean()

        # Generuj sygnały
        signals = []
        position_open = False
        last_signal = None
        trade_count = 0
        max_trades = 5  # Mniej transakcji dla testu timeframe'ów

        for i in range(periods):
            if i < 50:  # Czekaj na wystarczającą ilość danych
                signals.append({'action': 'WAIT'})
                continue

            if trade_count >= max_trades:
                signals.append({'action': 'WAIT'})
                continue

            # Pobierz wskaźniki z odpowiednim opóźnieniem
            sma20 = data['SMA_20'].iloc[i]
            sma50 = data['SMA_50'].iloc[i]
            rsi = data['RSI'].iloc[i]
            macd = data['MACD'].iloc[i]
            signal_line = data['Signal_Line'].iloc[i]
            current_price = data['close'].iloc[i]

            if np.isnan(sma20) or np.isnan(sma50) or np.isnan(rsi) or np.isnan(macd) or np.isnan(signal_line):
                signals.append({'action': 'WAIT'})
                continue

            if not position_open:
                # Warunki dla BUY
                buy_conditions = (
                    sma20 > sma50 and  # Trend wzrostowy
                    rsi < 70 and rsi > 30 and  # Unikaj ekstremów RSI
                    macd > signal_line and  # Pozytywny MACD
                    (last_signal != 'BUY' or last_signal is None)
                )

                if buy_conditions:
                    signal = {
                        'action': 'BUY',
                        'volume': 1.0,
                        'market_data': {
                            'symbol': 'EURUSD',
                            'current_price': current_price,
                            'sma_20': sma20,
                            'sma_50': sma50,
                            'rsi': rsi,
                            'macd': macd,
                            'signal_line': signal_line
                        }
                    }
                    signals.append(signal)
                    position_open = True
                    last_signal = 'BUY'
                    trade_count += 1
                else:
                    signals.append({'action': 'WAIT'})
            else:
                # Warunki zamknięcia pozycji
                close_conditions = (
                    sma20 < sma50 or  # Zmiana trendu
                    rsi > 80 or  # Wykupienie
                    macd < signal_line  # Zmiana momentum
                )

                if close_conditions:
                    signal = {
                        'action': 'CLOSE',
                        'market_data': {
                            'symbol': 'EURUSD',
                            'current_price': current_price,
                            'sma_20': sma20,
                            'sma_50': sma50,
                            'rsi': rsi,
                            'macd': macd,
                            'signal_line': signal_line
                        }
                    }
                    signals.append(signal)
                    position_open = False
                    last_signal = None
                else:
                    signals.append({'action': 'WAIT'})

        # Ustaw side_effect dla mocka
        mock_strategy.generate_signals.side_effect = signals

        with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
            mock_loader.return_value.load_data = AsyncMock(return_value=data)
            results = await backtester.run_backtest()

            # Sprawdź podstawowe metryki
            assert isinstance(results, dict)
            assert 'total_return' in results
            assert 'max_drawdown' in results
            assert 'sharpe_ratio' in results
            assert 'win_rate' in results

            # Sprawdź czy timeframe jest poprawnie ustawiony
            assert backtester.timeframe == timeframe

            # Sprawdź czy wyniki są sensowne
            assert results['total_return'] > -50  # Maksymalna strata 50%
            assert results['max_drawdown'] < 30  # Maksymalny drawdown 30%
            assert len(results.get('trades', [])) <= max_trades  # Limit transakcji

            # Sprawdź logi
            mock_logger.log_trade.assert_any_call({
                'type': 'INFO',
                'symbol': 'EURUSD',
                'message': f"🥷 Backtest zakończony dla {backtester.symbol}"
            })
            mock_logger.log_trade.assert_any_call({
                'type': 'INFO',
                'symbol': 'EURUSD',
                'message': f"📊 Wyniki: {results}"
            })

            # Zresetuj mocki
            mock_strategy.generate_signals.reset_mock()
            mock_logger.reset_mock() 

@pytest.mark.asyncio
async def test_process_bar_buy_signal(mock_strategy, mock_logger, small_sample_data):
    """Test przetwarzania świecy z sygnałem kupna."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    backtester.data = small_sample_data

    # Ustaw sygnał kupna
    mock_strategy.generate_signals.return_value = {
        'action': 'BUY',
        'volume': 1.0,
        'stop_loss': 99.0,
        'take_profit': 105.0
    }

    # Przetwórz pierwszą świecę
    await backtester.run_backtest()

    # Sprawdź czy pozycja została otwarta
    assert len(backtester.trades) >= 1
    assert mock_logger.log_trade.call_count > 0
    
    # Wyświetl wszystkie wywołania loggera
    print("\nWszystkie wywołania log_trade:")
    for call in mock_logger.log_trade.call_args_list:
        print(f"- {call}")

    # Sprawdź czy zalogowano otwarcie pozycji
    mock_logger.log_trade.assert_any_call({
        'type': 'DEBUG',
        'symbol': 'EURUSD',
        'message': '🥷 Otwieram pozycję BUY na EURUSD'
    })

@pytest.mark.asyncio
async def test_process_bar_sell_signal(mock_strategy, mock_logger, small_sample_data):
    """Test przetwarzania świecy z sygnałem sprzedaży."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    backtester.data = small_sample_data

    # Ustaw sygnał sprzedaży
    mock_strategy.generate_signals.return_value = {
        'action': 'SELL',
        'volume': 1.0,
        'stop_loss': 105.0,
        'take_profit': 95.0
    }

    # Przetwórz pierwszą świecę
    await backtester.run_backtest()

    # Sprawdź czy pozycja została otwarta
    assert len(backtester.trades) >= 1
    assert mock_logger.log_trade.call_count > 0
    
    # Wyświetl wszystkie wywołania loggera
    print("\nWszystkie wywołania log_trade:")
    for call in mock_logger.log_trade.call_args_list:
        print(f"- {call}")

    # Sprawdź czy zalogowano otwarcie pozycji
    mock_logger.log_trade.assert_any_call({
        'type': 'DEBUG',
        'symbol': 'EURUSD',
        'message': '🥷 Otwieram pozycję SELL na EURUSD'
    })

@pytest.mark.asyncio
async def test_process_bar_position_validation(mock_strategy, mock_logger, small_sample_data):
    """Test walidacji wielkości pozycji."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger,
        initial_capital=10000
    )
    backtester.data = small_sample_data

    # Ustaw konfigurację strategii
    mock_strategy.config = {
        'max_position_size': 1.0,
        'risk_per_trade': 0.02,
        'max_trades': 5
    }

    # Ustaw sygnał z nieprawidłową wielkością pozycji
    mock_strategy.generate_signals.return_value = {
        'action': 'BUY',
        'volume': 2.0,  # Większa niż max_position_size w strategii
        'stop_loss': 99.0,
        'take_profit': 105.0
    }

    # Przetwórz świecę
    await backtester.run_backtest()

    # Sprawdź czy użyto wielkości z sygnału (walidacja jest w strategii)
    assert len(backtester.trades) >= 1
    assert backtester.trades[0].size == 2.0

@pytest.mark.asyncio
async def test_process_bar_close_position(mock_strategy, mock_logger, small_sample_data):
    """Test zamykania pozycji."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    backtester.data = small_sample_data

    # Najpierw sygnał kupna, potem zamknięcia
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 1.0, 'stop_loss': 99.0, 'take_profit': 105.0},  # Pierwsza świeca
        {'action': 'CLOSE'},  # Druga świeca
        {'action': 'WAIT'},  # Pozostałe świece
        {'action': 'WAIT'},
        {'action': 'WAIT'}
    ]

    # Przetwórz świece
    await backtester.run_backtest()

    # Sprawdź czy pozycja została zamknięta
    assert len(backtester.trades) == 1
    assert mock_logger.log_trade.call_count > 0
    
    # Wyświetl wszystkie wywołania loggera
    print("\nWszystkie wywołania log_trade:")
    for call in mock_logger.log_trade.call_args_list:
        print(f"- {call}")

    # Sprawdź czy zalogowano zamknięcie pozycji
    mock_logger.log_trade.assert_any_call({
        'type': 'DEBUG',
        'symbol': 'EURUSD',
        'message': '🥷 Zamykam pozycję na EURUSD'
    }) 

@pytest.mark.asyncio
async def test_backtester_with_custom_strategy_config(mock_strategy, mock_logger, small_sample_data):
    """Test backtestera z niestandardową konfiguracją strategii."""
    # Ustaw niestandardową konfigurację
    mock_strategy.config = {
        'max_position_size': 0.5,
        'risk_per_trade': 0.01,
        'max_trades': 3,
        'min_profit_ratio': 1.5
    }
    
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        initial_capital=5000,
        logger=mock_logger
    )
    
    # Symuluj sygnały z uwzględnieniem konfiguracji
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 0.5},  # Maksymalna wielkość pozycji
        {'action': 'CLOSE'},
        {'action': 'SELL', 'volume': 0.3},  # Mniejsza pozycja
        {'action': 'CLOSE'},
        {'action': 'WAIT'}  # Ignoruj pozostałe świece
    ]
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=small_sample_data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) == 2
    assert backtester.trades[0].size == 0.5  # Pierwsza pozycja z max size
    assert backtester.trades[1].size == 0.3  # Druga pozycja z mniejszym size

@pytest.mark.asyncio
async def test_backtester_with_risk_management(mock_strategy, mock_logger, small_sample_data):
    """Test backtestera z zarządzaniem ryzykiem."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        initial_capital=10000,
        logger=mock_logger
    )
    
    # Symuluj sygnały z stop loss i take profit
    mock_strategy.generate_signals.side_effect = [
        {
            'action': 'BUY',
            'volume': 1.0,
            'stop_loss': 99.0,  # 1% stop loss
            'take_profit': 102.0  # 2% take profit
        },
        {'action': 'WAIT'},
        {'action': 'WAIT'},
        {'action': 'WAIT'},
        {'action': 'WAIT'}
    ]
    
    # Modyfikuj dane tak, aby trafić w take profit
    modified_data = small_sample_data.copy()
    modified_data.loc[modified_data.index[2], 'high'] = 102.5  # Take profit zostanie trafiony
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=modified_data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) == 1
    assert backtester.trades[0].profit > 0  # Pozycja zamknięta z zyskiem na take profit
    assert backtester.trades[0].exit_price == 102.0  # Cena wyjścia to take profit

@pytest.mark.asyncio
async def test_backtester_with_position_sizing(mock_strategy, mock_logger):
    """Test backtestera z różnymi wielkościami pozycji."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        initial_capital=10000,
        logger=mock_logger
    )
    
    # Przygotuj dane testowe
    dates = pd.date_range(start='2024-01-01', periods=7, freq='1h')
    data = pd.DataFrame(index=dates)
    data['open'] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    data['high'] = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    data['low'] = [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    data['close'] = [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5]
    data['volume'] = [1000, 1100, 1200, 1300, 1400, 1500, 1600]
    data['SMA_20'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['RSI'] = [50, 55, 60, 65, 70, 75, 80]
    data['MACD'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    data['Signal_Line'] = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    
    # Symuluj sygnały z różnymi wielkościami pozycji
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 0.1},  # Mała pozycja
        {'action': 'CLOSE'},
        {'action': 'BUY', 'volume': 0.5},  # Średnia pozycja
        {'action': 'CLOSE'},
        {'action': 'BUY', 'volume': 1.0},   # Duża pozycja
        {'action': 'CLOSE'},
        {'action': 'WAIT'}
    ]
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) == 3
    assert backtester.trades[0].size == 0.1
    assert backtester.trades[1].size == 0.5
    assert backtester.trades[2].size == 1.0

@pytest.mark.asyncio
async def test_backtester_handle_invalid_signals(mock_strategy, mock_logger):
    """Test obsługi nieprawidłowych sygnałów."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane testowe
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    data['open'] = [100.0, 101.0, 102.0, 103.0, 104.0]
    data['high'] = [101.0, 102.0, 103.0, 104.0, 105.0]
    data['low'] = [99.0, 100.0, 101.0, 102.0, 103.0]
    data['close'] = [100.5, 101.5, 102.5, 103.5, 104.5]
    data['volume'] = [1000, 1100, 1200, 1300, 1400]
    data['SMA_20'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['RSI'] = [50, 55, 60, 65, 70]
    data['MACD'] = [0.1, 0.2, 0.3, 0.4, 0.5]
    data['Signal_Line'] = [0.05, 0.15, 0.25, 0.35, 0.45]
    
    # Symuluj różne nieprawidłowe sygnały
    mock_strategy.generate_signals.side_effect = [
        ValueError("Nieprawidłowa wielkość pozycji"),  # Rzuć wyjątek zamiast zwracać nieprawidłowy sygnał
        KeyError("Brak wymaganego pola action"),  # Rzuć wyjątek dla nieprawidłowej akcji
        TypeError("Nieprawidłowy typ stop loss"),  # Rzuć wyjątek dla nieprawidłowego stop loss
        {'action': 'CLOSE'},
        {'action': 'WAIT'}
    ]
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert mock_logger.log_error.call_count >= 3
    assert len(backtester.trades) == 0  # Żadna pozycja nie powinna zostać otwarta

@pytest.mark.asyncio
async def test_backtester_handle_data_gaps(mock_strategy, mock_logger):
    """Test obsługi luk w danych."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane z lukami (NaN)
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    data['open'] = [100.0, np.nan, 102.0, 103.0, 104.0]
    data['high'] = [101.0, np.nan, 103.0, 104.0, 105.0]
    data['low'] = [99.0, np.nan, 101.0, 102.0, 103.0]
    data['close'] = [100.5, np.nan, 102.5, 103.5, 104.5]
    data['volume'] = [1000, np.nan, 1200, 1300, 1400]
    
    # Dodaj wskaźniki techniczne
    data['SMA_20'] = data['close'].rolling(window=2).mean()
    data['SMA_50'] = data['close'].rolling(window=2).mean()
    data['RSI'] = [50, np.nan, 60, 65, 70]
    data['MACD'] = [0.1, np.nan, 0.3, 0.4, 0.5]
    data['Signal_Line'] = [0.05, np.nan, 0.25, 0.35, 0.45]
    
    # Symuluj sygnały
    def generate_signals_with_error(*args, **kwargs):
        market_data = kwargs.get('market_data', {})
        if pd.isna(market_data.get('current_price')):
            raise ValueError("Brak danych cenowych")
        return {'action': 'WAIT'}
    
    mock_strategy.generate_signals.side_effect = generate_signals_with_error
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert mock_logger.log_error.call_count > 0
    assert "Błąd podczas backtestingu" in mock_logger.log_error.call_args_list[0][0][0]

@pytest.mark.asyncio
async def test_backtester_handle_market_volatility(mock_strategy, mock_logger):
    """Test obsługi wysokiej zmienności rynku."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane z wysoką zmiennością
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Duże skoki cenowe
    data['open'] = [100.0, 120.0, 90.0, 110.0, 85.0]
    data['high'] = [130.0, 125.0, 115.0, 115.0, 100.0]
    data['low'] = [95.0, 85.0, 85.0, 80.0, 80.0]
    data['close'] = [120.0, 90.0, 110.0, 85.0, 95.0]
    data['volume'] = [5000, 8000, 10000, 12000, 15000]
    
    # Dodaj wskaźniki techniczne
    data['SMA_20'] = data['close'].rolling(window=2).mean()
    data['SMA_50'] = data['close'].rolling(window=2).mean()
    data['RSI'] = [80, 20, 75, 25, 60]
    data['MACD'] = [2.0, -2.0, 1.5, -1.5, 0.5]
    data['Signal_Line'] = [1.0, -1.0, 0.8, -0.8, 0.2]
    
    # Symuluj sygnały w warunkach wysokiej zmienności
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 1.0, 'stop_loss': 95.0},  # Pozycja ze stop lossem
        {'action': 'WAIT'},  # Stop loss zostanie trafiony
        {'action': 'SELL', 'volume': 1.0, 'take_profit': 85.0},  # Pozycja z take profitem
        {'action': 'WAIT'},  # Take profit zostanie trafiony
        {'action': 'WAIT'}
    ]
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) == 2  # Powinny być dwie zamknięte pozycje
    
    # Sprawdź czy pierwsza pozycja została zamknięta ze stratą (stop loss)
    assert backtester.trades[0].profit < 0
    # Sprawdź czy druga pozycja została zamknięta z zyskiem (take profit)
    assert backtester.trades[1].profit > 0

@pytest.mark.asyncio
async def test_backtester_edge_case_calculations(mock_strategy, mock_logger):
    """Test kalkulacji zysków/strat w skrajnych przypadkach."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        initial_capital=10000,
        logger=mock_logger
    )
    
    # Przygotuj dane ze skrajnymi wartościami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Bardzo małe zmiany cen
    data['open'] = [1.00001, 1.00002, 1.00003, 1.00004, 1.00005]
    data['high'] = [1.00002, 1.00003, 1.00004, 1.00005, 1.00006]
    data['low'] = [1.00000, 1.00001, 1.00002, 1.00003, 1.00004]
    data['close'] = [1.00001, 1.00002, 1.00003, 1.00004, 1.00005]
    data['volume'] = [1000, 1000, 1000, 1000, 1000]
    
    # Wskaźniki techniczne
    data['SMA_20'] = data['close']
    data['SMA_50'] = data['close']
    data['RSI'] = [50, 50, 50, 50, 50]
    data['MACD'] = [0, 0, 0, 0, 0]
    data['Signal_Line'] = [0, 0, 0, 0, 0]
    
    # Symuluj transakcje z bardzo małymi zmianami
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 0.01},  # Minimalna wielkość
        {'action': 'CLOSE'},
        {'action': 'SELL', 'volume': 0.01},
        {'action': 'CLOSE'},
        {'action': 'WAIT'}
    ]
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) == 2
    # Sprawdź czy zyski/straty są prawidłowo zaokrąglone
    for trade in backtester.trades:
        assert isinstance(trade.profit, float)
        assert abs(trade.profit) < 0.0001  # Bardzo małe zyski/straty

@pytest.mark.asyncio
async def test_backtester_extreme_price_movements(mock_strategy, mock_logger):
    """Test kalkulacji przy ekstremalnych ruchach cenowych."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        initial_capital=10000,
        logger=mock_logger
    )
    
    # Przygotuj dane z ekstremalnymi ruchami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Ekstremalne ruchy cenowe
    data['open'] = [100.0, 200.0, 50.0, 300.0, 25.0]
    data['high'] = [200.0, 250.0, 300.0, 350.0, 300.0]
    data['low'] = [50.0, 50.0, 25.0, 25.0, 20.0]
    data['close'] = [200.0, 50.0, 300.0, 25.0, 250.0]
    data['volume'] = [10000, 20000, 30000, 40000, 50000]
    
    # Wskaźniki techniczne
    data['SMA_20'] = data['close'].rolling(window=2).mean()
    data['SMA_50'] = data['close'].rolling(window=2).mean()
    data['RSI'] = [90, 10, 90, 10, 50]
    data['MACD'] = [5.0, -5.0, 5.0, -5.0, 0]
    data['Signal_Line'] = [2.5, -2.5, 2.5, -2.5, 0]
    
    # Symuluj transakcje w ekstremalnych warunkach
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 1.0, 'stop_loss': 90.0},  # Duży stop loss
        {'action': 'SELL', 'volume': 2.0, 'take_profit': 30.0},  # Duży take profit
        {'action': 'BUY', 'volume': 0.5, 'stop_loss': 20.0},
        {'action': 'CLOSE'},
        {'action': 'WAIT'}
    ]
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) > 0
    # Sprawdź czy są duże zyski i straty
    profits = [trade.profit for trade in backtester.trades]
    assert any(abs(profit) > 100 for profit in profits)  # Powinny być duże zmiany

@pytest.mark.asyncio
async def test_backtester_precision_handling(mock_strategy, mock_logger):
    """Test obsługi precyzji w kalkulacjach."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        initial_capital=10000,
        logger=mock_logger
    )
    
    # Przygotuj dane z różną precyzją
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Ceny z różną precyzją
    data['open'] = [1.23456, 1.23457, 1.23458, 1.23459, 1.23460]
    data['high'] = [1.23457, 1.23458, 1.23459, 1.23460, 1.23461]
    data['low'] = [1.23455, 1.23456, 1.23457, 1.23458, 1.23459]
    data['close'] = [1.23456, 1.23457, 1.23458, 1.23459, 1.23460]
    data['volume'] = [1000.123, 1000.234, 1000.345, 1000.456, 1000.567]
    
    # Wskaźniki techniczne z różną precyzją
    data['SMA_20'] = data['close'].round(5)
    data['SMA_50'] = data['close'].rolling(window=2).mean().round(5)
    data['RSI'] = [50.123, 50.234, 50.345, 50.456, 50.567]
    data['MACD'] = [0.00123, 0.00234, 0.00345, 0.00456, 0.00567]
    data['Signal_Line'] = [0.00100, 0.00200, 0.00300, 0.00400, 0.00500]
    
    # Symuluj transakcje z precyzyjnymi wartościami
    mock_strategy.generate_signals.side_effect = [
        {'action': 'BUY', 'volume': 0.12345},
        {'action': 'CLOSE'},
        {'action': 'SELL', 'volume': 0.23456},
        {'action': 'CLOSE'},
        {'action': 'WAIT'}
    ]
    
    with patch('src.backtest.backtester.HistoricalDataLoader') as mock_loader:
        mock_loader.return_value.load_data = AsyncMock(return_value=data)
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) == 2
    
    # Sprawdź precyzję wyników
    for trade in backtester.trades:
        # Sprawdź czy wartości są typu float
        assert isinstance(trade.entry_price, float)
        assert isinstance(trade.exit_price, float)
        assert isinstance(trade.profit, float)
        
        # Sprawdź czy wartości są w odpowiednim zakresie
        assert 1.23 < trade.entry_price < 1.24
        assert 1.23 < trade.exit_price < 1.24
        
        # Sprawdź czy zysk/strata jest odpowiednio mały (ze względu na małe zmiany cen)
        assert abs(trade.profit) < 0.001 