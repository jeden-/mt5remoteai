"""
Moduł zawierający testy dla backtester.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from src.backtest.backtester import Backtester
from src.backtest.performance_metrics import TradeResult
from src.strategies.basic_strategy import BasicStrategy
from src.utils.logger import TradingLogger

@pytest.fixture
def mock_strategy():
    """Fixture tworzący mock strategii."""
    strategy = Mock(spec=BasicStrategy)
    strategy.name = "TestStrategy"
    strategy.config = {
        'max_position_size': 1.0,
        'stop_loss_pips': 50,
        'take_profit_pips': 100
    }
    return strategy

@pytest.fixture
def mock_logger():
    """Mock dla loggera."""
    logger = Mock(spec=TradingLogger)
    logger.log_trade = Mock()
    logger.log_error = Mock()
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
    data['SMA_20'] = data['close'].rolling(window=20).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=50).mean().fillna(data['close'])
    data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 50 + np.random.normal(0, 10)).fillna(50)
    data['MACD'] = (data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()).fillna(0)
    data['Signal_Line'] = data['MACD'].rolling(window=9).mean().fillna(0)
    
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
        {'market_data': {'action': 'BUY', 'volume': 1.0}},  # Otwórz długą
        {'market_data': {'action': 'HOLD'}},  # Trzymaj
        {'market_data': {'action': 'CLOSE'}},  # Zamknij
        {'market_data': {'action': 'SELL', 'volume': 1.0}},  # Otwórz krótką
        {'market_data': {'action': 'HOLD'}},  # Trzymaj
        {'market_data': {'action': 'CLOSE'}}  # Zamknij
    ]
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=sample_data):
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert "total_return" in results
    assert "win_rate" in results
    assert "profit_factor" in results
    assert len(backtester.trades) > 0
    assert mock_logger.log_trade.call_count > 0

def test_check_close_conditions(mock_strategy, mock_logger):
    """Test sprawdzania warunków zamknięcia pozycji."""
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
    current_bar = pd.Series({'low': 99, 'high': 105, 'open': 105, 'close': 105})
    signals = {'market_data': {'action': 'HOLD'}}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 100  # Stop loss price

    # Take profit hit
    current_bar = pd.Series({'low': 105, 'high': 111, 'open': 105, 'close': 105})
    signals = {'market_data': {'action': 'HOLD'}}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 110  # Take profit price

    # No close condition
    current_bar = pd.Series({'low': 102, 'high': 108, 'open': 105, 'close': 105})
    signals = {'market_data': {'action': 'HOLD'}}
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
    current_bar = pd.Series({'low': 105, 'high': 111, 'open': 105, 'close': 105})
    signals = {'market_data': {'action': 'HOLD'}}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 110  # Stop loss price

    # Take profit hit
    current_bar = pd.Series({'low': 99, 'high': 105, 'open': 105, 'close': 105})
    signals = {'market_data': {'action': 'HOLD'}}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 100  # Take profit price

    # No close condition
    current_bar = pd.Series({'low': 102, 'high': 108, 'open': 105, 'close': 105})
    signals = {'market_data': {'action': 'HOLD'}}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert not should_close

    # Test sygnału zamknięcia
    signals = {'market_data': {'action': 'CLOSE'}}
    should_close, exit_price = backtester._check_close_conditions(position, current_bar, signals)
    assert should_close
    assert exit_price == 105  # Cena otwarcia następnej świecy

def test_calculate_profit(mock_strategy, mock_logger):
    """Test obliczania zysku/straty."""
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
    data['SMA_20'] = data['close'].rolling(window=20).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=50).mean().fillna(data['close'])
    data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 50 + np.random.normal(0, 10)).fillna(50)
    data['MACD'] = (data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()).fillna(0)
    data['Signal_Line'] = data['MACD'].rolling(window=9).mean().fillna(0)
    
    # Ustaw cenę poniżej stop loss w trzeciej świecy
    data.loc[dates[2], 'low'] = stop_loss - 0.1
    
    # Ustaw sygnały strategii
    mock_strategy.generate_signals.side_effect = [
        {'market_data': {'action': 'BUY', 'volume': 1.0, 'stop_loss': stop_loss}},  # Otwórz pozycję
        {'market_data': {'action': 'HOLD'}},  # Trzymaj
        {'market_data': {'action': 'HOLD'}},  # Stop loss zostanie trafiony
        {'market_data': {'action': 'HOLD'}},  # Nie powinno być używane
        {'market_data': {'action': 'HOLD'}}   # Nie powinno być używane
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
    stop_loss = entry_price - 2.0  # Dodaj stop loss poniżej entry price
    
    data['open'] = [entry_price] * len(dates)
    data['high'] = [entry_price + 0.5] * len(dates)
    data['low'] = [entry_price - 0.5] * len(dates)
    data['close'] = [entry_price] * len(dates)
    data['volume'] = [1000] * len(dates)
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=20).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=50).mean().fillna(data['close'])
    data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 50 + np.random.normal(0, 10)).fillna(50)
    data['MACD'] = (data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()).fillna(0)
    data['Signal_Line'] = data['MACD'].rolling(window=9).mean().fillna(0)
    
    # Ustaw cenę powyżej take profit w trzeciej świecy
    data.loc[dates[2], 'high'] = take_profit + 0.1
    
    # Ustaw sygnały strategii
    mock_strategy.generate_signals.side_effect = [
        {'market_data': {'action': 'BUY', 'volume': 1.0, 'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit}},  # Otwórz pozycję
        {'market_data': {'action': 'HOLD'}},  # Trzymaj
        {'market_data': {'action': 'HOLD'}},  # Take profit zostanie trafiony
        {'market_data': {'action': 'HOLD'}},  # Nie powinno być używane
        {'market_data': {'action': 'HOLD'}}   # Nie powinno być używane
    ]
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=data):
        results = await backtester.run_backtest()
    
    assert len(backtester.trades) == 1
    assert backtester.trades[0].profit > 0  # Zysk na take proficie

@pytest.mark.asyncio
async def test_market_regime_detection(mock_strategy, mock_logger):
    """Test wykrywania i reagowania na różne reżimy rynkowe."""
    # Generuj dane z różnymi reżimami rynkowymi
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Trend wzrostowy
    trend_up = np.linspace(1.1000, 1.1200, 50) + np.random.normal(0, 0.0002, 50)
    
    # Trend spadkowy
    trend_down = np.linspace(1.1200, 1.1000, 50) + np.random.normal(0, 0.0002, 50)
    
    # Konsolidacja
    consolidation = np.random.normal(1.1100, 0.0005, 50)
    
    # Wysoka zmienność
    high_volatility = np.random.normal(1.1100, 0.0020, 50)
    
    # Połącz wszystkie reżimy
    data['close'] = pd.Series(
        np.concatenate([trend_up, trend_down, consolidation, high_volatility]),
        index=dates
    )
    data['open'] = data['close'].shift(1).fillna(data['close'])  # Wypełnij NaN pierwszej świecy wartością close
    data['high'] = data['close'] * (1 + np.random.uniform(0, 0.001, len(data)))
    data['low'] = data['close'] * (1 - np.random.uniform(0, 0.001, len(data)))
    data['volume'] = np.random.uniform(1000, 2000, len(data))
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=20).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=50).mean().fillna(data['close'])
    data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 50 + np.random.normal(0, 10)).fillna(50)
    data['MACD'] = (data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()).fillna(0)
    data['Signal_Line'] = data['MACD'].rolling(window=9).mean().fillna(0)
    
    # Symuluj różne sygnały w zależności od reżimu rynkowego
    signals = []
    for i in range(len(data)):
        if i < len(data) - 1:  # Nie generuj sygnału dla ostatniej świecy
            volatility = float(data['close'].iloc[max(0, i-20):i+1].std()) if i >= 20 else 0.0002
            current_price = float(data['close'].iloc[i])
            
            if i < 50:  # Trend wzrostowy
                if i % 2 == 0:  # Co drugą świecę zamiast co piątą
                    signals.append({
                        'market_data': {
                            'action': 'BUY',
                            'volume': 0.1,
                            'stop_loss': current_price - volatility * 2,
                            'take_profit': current_price + volatility * 3
                        }
                    })
                else:
                    signals.append({'market_data': {'action': 'HOLD'}})
            elif i < 100:  # Trend spadkowy
                if len(signals) % 5 == 0:
                    signals.append({
                        'market_data': {
                            'action': 'SELL',
                            'volume': 0.1,
                            'stop_loss': current_price + volatility * 2,
                            'take_profit': current_price - volatility * 3
                        }
                    })
                else:
                    signals.append({'market_data': {'action': 'HOLD'}})
            elif i < 150:  # Konsolidacja
                signals.append({'market_data': {'action': 'HOLD'}})
            else:  # Wysoka zmienność
                if volatility > 0.001:  # Tylko gdy zmienność jest wysoka
                    signals.append({'market_data': {'action': 'CLOSE'}})
                else:
                    signals.append({'market_data': {'action': 'HOLD'}})
        else:
            signals.append({'market_data': {'action': 'CLOSE'}})  # Zamknij wszystkie pozycje na końcu
    
    mock_strategy.generate_signals.side_effect = signals
    
    # Wyświetl pierwsze 10 sygnałów dla debugowania
    print("\nPierwsze 10 sygnałów:")
    for i, signal in enumerate(signals[:10]):
        print(f"Sygnał {i}: {signal}")

    # Inicjalizuj backtester
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        timeframe='1H',
        initial_capital=10000,
        logger=mock_logger
    )
    
    # Podmień dane w data_loader
    backtester.data = data
    
    # Uruchom backtest
    results = await backtester.run_backtest()
    
    # Sprawdź czy wyniki są sensowne
    assert isinstance(results, dict)
    assert 'total_trades' in results
    assert 'win_rate' in results
    assert 'profit_factor' in results
    assert 'max_drawdown' in results
    
    # Sprawdź czy strategia odpowiednio reagowała na różne reżimy
    trades = backtester.trades
    assert len(trades) > 0
    
    # Sprawdź czy były transakcje w trendzie wzrostowym
    trend_up_trades = [t for t in trades if t.entry_time < dates[50]]
    assert len(trend_up_trades) > 0
    assert all(t.direction == 'BUY' for t in trend_up_trades)
    
    # Sprawdź czy były transakcje w trendzie spadkowym
    trend_down_trades = [t for t in trades if dates[50] <= t.entry_time < dates[100]]
    assert len(trend_down_trades) > 0
    assert all(t.direction == 'SELL' for t in trend_down_trades)
    
    # Sprawdź czy w konsolidacji było mniej transakcji
    consolidation_trades = [t for t in trades if dates[100] <= t.entry_time < dates[150]]
    assert len(consolidation_trades) < len(trend_up_trades)
    
    # Sprawdź czy w wysokiej zmienności pozycje były szybciej zamykane
    high_vol_trades = [t for t in trades if t.entry_time >= dates[150]]
    if high_vol_trades:
        high_vol_durations = [(t.exit_time - t.entry_time).total_seconds() for t in high_vol_trades]
        other_durations = [(t.exit_time - t.entry_time).total_seconds() for t in trades if t not in high_vol_trades]
        assert np.mean(high_vol_durations) < np.mean(other_durations)

@pytest.mark.asyncio
async def test_error_handling_with_open_position(mock_strategy, mock_logger):
    """Test obsługi błędów gdy jest otwarta pozycja."""
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
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['RSI'] = [50, 55, 60, 65, 70]
    data['MACD'] = [0.1, 0.2, 0.3, 0.4, 0.5]
    data['Signal_Line'] = [0.05, 0.15, 0.25, 0.35, 0.45]
    
    # Symuluj sygnały - najpierw otwórz pozycję, potem wywołaj błąd
    mock_strategy.generate_signals.side_effect = [
        {'market_data': {'action': 'BUY', 'volume': 1.0}},  # Otwórz pozycję
        Exception("Test error"),  # Wywołaj błąd podczas aktywnej pozycji
        {'market_data': {'action': 'HOLD'}},  # Te sygnały nie powinny być używane
        {'market_data': {'action': 'HOLD'}},
        {'market_data': {'action': 'HOLD'}}
    ]
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=data):
        results = await backtester.run_backtest()
        
    # Sprawdź czy pozycja została zamknięta po błędzie
    assert len(backtester.trades) == 1
    assert mock_logger.log_error.call_count >= 2  # Powinny być dwa błędy: jeden o błędzie, drugi o zamknięciu pozycji
    assert "❌ Błąd podczas backtestingu: Test error" in mock_logger.log_error.call_args_list[0][0][0]
    assert "❌ Zamykam pozycję z powodu błędu na EURUSD" in mock_logger.log_error.call_args_list[1][0][0]

def test_invalid_parameters(mock_strategy, mock_logger):
    """Test walidacji nieprawidłowych parametrów."""
    # Test nieprawidłowego timeframe
    with pytest.raises(ValueError, match="Nieprawidłowy timeframe"):
        Backtester(
            strategy=mock_strategy,
        symbol='EURUSD',
            timeframe='INVALID',
            logger=mock_logger
        )

    # Test nieprawidłowego initial_capital
    with pytest.raises(ValueError, match="Initial capital musi być większy od 0"):
        Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
            initial_capital=0,
            logger=mock_logger
        )
    
    # Test nieprawidłowej daty początkowej
    with pytest.raises(ValueError, match="Data początkowa nie może być z przyszłości"):
        Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
            start_date=datetime.now() + timedelta(days=1),
        logger=mock_logger
    )

@pytest.mark.benchmark
def test_performance_large_dataset(mock_strategy, mock_logger, benchmark):
    """Test wydajności dla dużego zestawu danych."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Generuj duży zestaw danych (rok z interwałem 1h = 8760 świec)
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Generuj realistyczne dane cenowe z trendem i zmiennością
    base_price = 1.1000
    trend = np.linspace(-0.02, 0.02, len(dates))  # Trend ±2%
    volatility = np.random.normal(0, 0.001, len(dates))  # Zmienność ±0.1%
    
    data['close'] = base_price * (1 + trend + volatility)
    data['open'] = data['close'].shift(1).fillna(data['close'])
    data['high'] = data['close'] * (1 + abs(np.random.normal(0, 0.0005, len(dates))))
    data['low'] = data['close'] * (1 - abs(np.random.normal(0, 0.0005, len(dates))))
    data['volume'] = np.random.normal(1000, 100, len(dates))
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=20).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=50).mean().fillna(data['close'])
    data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 50 + np.random.normal(0, 10)).fillna(50)
    data['MACD'] = (data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()).fillna(0)
    data['Signal_Line'] = data['MACD'].rolling(window=9).mean().fillna(0)
    
    # Przygotuj sygnały
    def generate_signals(market_data):
        return {
            'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
            'volume': 0.1,
            'stop_loss': market_data['current_price'] * 0.99,
            'take_profit': market_data['current_price'] * 1.01
        }
    
    mock_strategy.generate_signals.side_effect = generate_signals
    
    def run_backtest():
        with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=data):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(backtester.run_backtest())
    
    # Wykonaj benchmark
    results = benchmark(run_backtest)
    assert results is not None

@pytest.mark.asyncio
async def test_edge_cases(mock_strategy, mock_logger):
    """Test przypadków brzegowych."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane z lukami
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Generuj dane z lukami (co piąta świeca ma ekstremalne wartości)
    base_price = 1.1000
    data['close'] = [base_price * (1 + np.random.normal(0, 0.01 if i % 5 == 0 else 0.0005)) 
                     for i in range(len(dates))]
    
    data['open'] = data['close'].shift(1).fillna(data['close'])
    data['high'] = [price * 1.1 if i % 5 == 0 else price * (1 + abs(np.random.normal(0, 0.0005)))
                    for i, price in enumerate(data['close'])]
    data['low'] = [price * 0.9 if i % 5 == 0 else price * (1 - abs(np.random.normal(0, 0.0005)))
                   for i, price in enumerate(data['close'])]
    data['volume'] = [10000 if i % 5 == 0 else np.random.normal(1000, 100)
                      for i in range(len(dates))]
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=20).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=50).mean().fillna(data['close'])
    data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 50 + np.random.normal(0, 10)).fillna(50)
    data['MACD'] = (data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()).fillna(0)
    data['Signal_Line'] = data['MACD'].rolling(window=9).mean().fillna(0)
    
    # Przygotuj sygnały
    signals = []
    for i in range(len(data)):
        if i % 5 == 0:  # Co piątą świecę generuj sygnał
            signals.append({
                'market_data': {
                    'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
                    'volume': 0.1,
                    'stop_loss': float(data['close'].iloc[i]) * (0.95 if i % 2 == 0 else 1.05),
                    'take_profit': float(data['close'].iloc[i]) * (1.05 if i % 2 == 0 else 0.95)
                }
            })
        else:
            signals.append({'market_data': {'action': 'HOLD'}})
            
    mock_strategy.generate_signals.side_effect = signals
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=data):
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert 'total_trades' in results
    assert 'win_rate' in results
    assert 'profit_factor' in results
    assert 'max_drawdown' in results
    
    # Sprawdź czy były transakcje na świecach z lukami
    trades = backtester.trades
    assert len(trades) > 0
    
    # Sprawdź czy stop loss i take profit były odpowiednio dostosowane do luk
    for trade in trades:
        if trade.profit < 0:
            if trade.direction == 'BUY':
                assert trade.exit_price <= trade.entry_price * 0.95  # Stop loss hit
            else:  # SELL
                assert trade.exit_price >= trade.entry_price  # Stop loss hit
        elif trade.profit > 0:
            if trade.direction == 'BUY':
                assert trade.exit_price >= trade.entry_price  # Take profit hit
            else:  # SELL
                assert trade.exit_price <= trade.entry_price  # Take profit hit

@pytest.mark.asyncio
async def test_low_volatility(mock_strategy, mock_logger):
    """Test zachowania przy bardzo niskiej zmienności."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane z bardzo niską zmiennością
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame(index=dates)
    
    base_price = 1.1000
    data['close'] = [base_price * (1 + np.random.normal(0, 0.0001)) for _ in range(len(dates))]  # Bardzo mała zmienność
    data['open'] = data['close'].shift(1).fillna(data['close'])
    data['high'] = data['close'] * (1 + abs(np.random.normal(0, 0.0001, len(dates))))
    data['low'] = data['close'] * (1 - abs(np.random.normal(0, 0.0001, len(dates))))
    data['volume'] = np.random.normal(1000, 10, len(dates))  # Stabilny wolumen
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=20).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=50).mean().fillna(data['close'])
    data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 50 + np.random.normal(0, 1)).fillna(50)
    data['MACD'] = (data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()).fillna(0)
    data['Signal_Line'] = data['MACD'].rolling(window=9).mean().fillna(0)
    
    # Przygotuj sygnały
    signals = []
    for i in range(len(data)):
        volatility = float(data['close'].iloc[max(0, i-20):i+1].std())
        if volatility < 0.0001:  # Bardzo niska zmienność
            signals.append({
                'action': 'HOLD'  # Nie handluj przy bardzo niskiej zmienności
            })
        else:
            signals.append({
                'action': 'BUY',
                'volume': 0.1,
                'stop_loss': float(data['close'].iloc[i]) * 0.999,  # Bardzo bliski SL
                'take_profit': float(data['close'].iloc[i]) * 1.001  # Bardzo bliski TP
            })
    
    mock_strategy.generate_signals.side_effect = signals
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=data):
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert results['total_trades'] == 0  # Nie powinno być transakcji przy bardzo niskiej zmienności

@pytest.mark.asyncio
async def test_high_volatility(mock_strategy, mock_logger):
    """Test zachowania przy bardzo wysokiej zmienności."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane z bardzo wysoką zmiennością
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame(index=dates)
    
    base_price = 1.1000
    data['close'] = [base_price * (1 + np.random.normal(0, 0.01)) for _ in range(len(dates))]
    data['open'] = data['close'].shift(1).fillna(data['close'])  # Wypełnij NaN pierwszej świecy wartością close
    data['high'] = data['close'] * (1 + abs(np.random.normal(0, 0.01, len(dates))))
    data['low'] = data['close'] * (1 - abs(np.random.normal(0, 0.01, len(dates))))
    data['volume'] = np.random.normal(1000, 500, len(dates))  # Zmienny wolumen
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=20).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=50).mean().fillna(data['close'])
    data['RSI'] = data['close'].rolling(window=14).apply(lambda x: 50 + np.random.normal(0, 20)).fillna(50)
    data['MACD'] = (data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()).fillna(0)
    data['Signal_Line'] = data['MACD'].rolling(window=9).mean().fillna(0)
    
    # Przygotuj sygnały
    signals = []
    for i in range(len(data)):
        volatility = float(data['close'].iloc[max(0, i-20):i+1].std())
        if volatility > 0.01:  # Bardzo wysoka zmienność
            signals.append({
                'market_data': {
                    'action': 'BUY',
                    'volume': 0.01,  # Mniejsza wielkość pozycji
                    'stop_loss': float(data['close'].iloc[i]) * 0.95,  # Szerszy SL
                    'take_profit': float(data['close'].iloc[i]) * 1.05  # Szerszy TP
                }
            })
        else:
            signals.append({
                'market_data': {
                    'action': 'BUY',
                    'volume': 0.1,
                    'stop_loss': float(data['close'].iloc[i]) * 0.99,
                    'take_profit': float(data['close'].iloc[i]) * 1.01
                }
            })
    
    mock_strategy.generate_signals.side_effect = signals
    
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=data):
        results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert 'total_trades' in results
    assert 'win_rate' in results
    assert 'profit_factor' in results
    assert 'max_drawdown' in results
    
    # Sprawdź czy wielkości pozycji były dostosowywane do zmienności
    trades = backtester.trades
    assert len(trades) > 0
    
    high_vol_trades = [t for t in trades if t.size == 0.01]
    low_vol_trades = [t for t in trades if t.size == 0.1]
    
    assert len(high_vol_trades) > 0, "Brak transakcji z małym wolumenem dla wysokiej zmienności"
    assert len(low_vol_trades) > 0, "Brak transakcji z normalnym wolumenem dla niskiej zmienności"

def test_invalid_symbol(mock_strategy, mock_logger):
    """Test walidacji nieprawidłowego symbolu."""
    # Test pustego symbolu
    with pytest.raises(ValueError, match="Symbol musi mieć od 3 do 10 znaków"):
        Backtester(
        strategy=mock_strategy,
            symbol='',
        logger=mock_logger
    )
    
    # Test za krótkiego symbolu
    with pytest.raises(ValueError, match="Symbol musi mieć od 3 do 10 znaków"):
        Backtester(
        strategy=mock_strategy,
            symbol='EU',
        logger=mock_logger
    )
    
    # Test za długiego symbolu
    with pytest.raises(ValueError, match="Symbol musi mieć od 3 do 10 znaków"):
        Backtester(
        strategy=mock_strategy,
            symbol='EURUSDGBPJPY',
        logger=mock_logger
    )
    
def test_invalid_strategy(mock_logger):
    """Test walidacji nieprawidłowej strategii."""
    # Test strategii, która nie jest instancją BasicStrategy
    invalid_strategy = Mock()
    with pytest.raises(ValueError, match="Strategia musi być instancją BasicStrategy"):
        Backtester(
            strategy=invalid_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )

@pytest.mark.asyncio
async def test_error_handling_during_backtest(mock_strategy, mock_logger):
    """Test obsługi błędów podczas backtestingu."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Symuluj brak danych (None)
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=None):
        with pytest.raises(RuntimeError, match="Nie udało się załadować danych"):
            await backtester.run_backtest()
    
    # Symuluj pusty DataFrame
    with patch('src.backtest.data_loader.HistoricalDataLoader.load_data', return_value=pd.DataFrame()):
        with pytest.raises(RuntimeError, match="Nie udało się załadować danych"):
            await backtester.run_backtest()
    
    # Symuluj błąd podczas generowania sygnałów
    mock_strategy.generate_signals.side_effect = Exception("Test error")
    test_data = pd.DataFrame({
        'open': [1.0, 1.0],
        'high': [1.1, 1.1],
        'low': [0.9, 0.9],
        'close': [1.0, 1.0],
        'volume': [100, 100],
        'SMA_20': [1.0, 1.0],
        'SMA_50': [1.0, 1.0],
        'RSI': [50, 50],
        'MACD': [0, 0],
        'Signal_Line': [0, 0]
    }, index=[datetime.now(), datetime.now() + timedelta(hours=1)])
    
    backtester.data = test_data  # Ustawiamy dane bezpośrednio
    results = await backtester.run_backtest()
    assert isinstance(results, dict)
    assert mock_logger.log_error.called

@pytest.mark.asyncio
async def test_entry_conditions_and_gaps(mock_strategy, mock_logger):
    """Test warunków wejścia i obsługi luk cenowych."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane testowe z lukami cenowymi
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Scenariusz 1: Cena otwarcia wyższa niż dozwolona dla BUY
    data['open'] = [100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
    data['high'] = [101.0, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5]
    data['low'] = [99.0, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5]
    data['close'] = [100.5, 102.2, 103.2, 104.2, 105.2, 106.2, 107.2, 108.2, 109.2, 110.2]
    data['volume'] = [1000] * 10
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['RSI'] = [50] * 10
    data['MACD'] = [0.1] * 10
    data['Signal_Line'] = [0.05] * 10
    
    # Ustaw sygnały
    signals = [
        # Sygnał BUY z ceną otwarcia powyżej 1% high (powinien być odrzucony)
        {'market_data': {'action': 'BUY', 'volume': 1.0, 'entry_price': 103.0}},
        # Sygnał SELL z ceną otwarcia poniżej 1% low (powinien być odrzucony)
        {'market_data': {'action': 'SELL', 'volume': 1.0, 'entry_price': 100.0}},
        # Sygnał BUY z luką poniżej stop loss
        {'market_data': {'action': 'BUY', 'volume': 1.0, 'entry_price': 103.0, 'stop_loss': 102.5, 'take_profit': 104.0}},
        # HOLD
        {'market_data': {'action': 'HOLD'}},
        # Sygnał SELL z luką powyżej stop loss
        {'market_data': {'action': 'SELL', 'volume': 1.0, 'entry_price': 105.0, 'stop_loss': 105.5, 'take_profit': 104.0}},
        # HOLD
        {'market_data': {'action': 'HOLD'}},
        # Sygnał BUY z luką powyżej take profit
        {'market_data': {'action': 'BUY', 'volume': 1.0, 'entry_price': 106.0, 'stop_loss': 105.5, 'take_profit': 106.5}},
        # HOLD
        {'market_data': {'action': 'HOLD'}},
        # Sygnał SELL z luką poniżej take profit
        {'market_data': {'action': 'SELL', 'volume': 1.0, 'entry_price': 108.0, 'stop_loss': 109.0, 'take_profit': 107.0}},
        # Zamknij wszystkie pozycje
        {'market_data': {'action': 'CLOSE'}}
    ]
    
    mock_strategy.generate_signals.side_effect = signals
    
    backtester.data = data
    results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) > 0
    
    # Sprawdź czy sygnały z nieodpowiednimi cenami zostały odrzucone
    trades_with_invalid_entry = []
    for trade in backtester.trades:
        # Znajdź świecę dla danej transakcji
        trade_bar = data.loc[trade.entry_time]
        if trade.direction == 'BUY':
            if trade.entry_price > trade_bar['high'] * 1.01:
                trades_with_invalid_entry.append(trade)
        else:  # SELL
            if trade.entry_price < trade_bar['low'] * 0.99:
                trades_with_invalid_entry.append(trade)
    
    assert len(trades_with_invalid_entry) == 0, f"Znaleziono transakcje z niepoprawnymi cenami wejścia: {trades_with_invalid_entry}"
    
    # Sprawdź czy luki cenowe są prawidłowo obsługiwane
    gap_trades = [t for t in backtester.trades if abs(t.exit_price - t.entry_price) > 1.0]
    assert len(gap_trades) > 0

@pytest.mark.asyncio
async def test_opposite_signals_closing(mock_strategy, mock_logger):
    """Test zamykania pozycji przy przeciwnych sygnałach."""
    backtester = Backtester(
        strategy=mock_strategy,
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Przygotuj dane testowe
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Trend wzrostowy, potem spadkowy
    data['open'] = [100.0, 101.0, 102.0, 101.5, 101.0]
    data['high'] = [101.0, 102.0, 103.0, 102.0, 101.5]
    data['low'] = [99.5, 100.5, 101.5, 101.0, 100.5]
    data['close'] = [100.5, 101.5, 102.5, 101.5, 101.0]
    data['volume'] = [1000] * 5
    
    # Dodaj wskaźniki
    data['SMA_20'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['SMA_50'] = data['close'].rolling(window=2).mean().fillna(data['close'])
    data['RSI'] = [50] * 5
    data['MACD'] = [0.1] * 5
    data['Signal_Line'] = [0.05] * 5
    
    # Ustaw sygnały: BUY -> SELL (powinno zamknąć BUY) -> BUY (powinno zamknąć SELL)
    signals = [
        # BUY na początku trendu wzrostowego
        {'market_data': {'action': 'BUY', 'volume': 1.0, 'entry_price': 100.0, 'stop_loss': 99.0, 'take_profit': 102.0}},
        # SELL na szczycie trendu
        {'market_data': {'action': 'SELL', 'volume': 1.0, 'entry_price': 101.5, 'stop_loss': 102.5, 'take_profit': 100.5}},
        # HOLD podczas spadku
        {'market_data': {'action': 'HOLD'}},
        # CLOSE na końcu
        {'market_data': {'action': 'CLOSE'}}
    ]
    
    mock_strategy.generate_signals.side_effect = signals
    
    backtester.data = data
    results = await backtester.run_backtest()
    
    assert isinstance(results, dict)
    assert len(backtester.trades) >= 2  # Powinny być co najmniej 2 transakcje
    
    # Sprawdź czy pozycje są zamykane przez przeciwne sygnały
    trades = backtester.trades
    assert len(trades) >= 2
    
    # Pierwsza transakcja BUY powinna być zamknięta przez SELL
    assert trades[0].direction == 'BUY'
    assert trades[0].entry_price == data.loc[trades[0].entry_time]['open']  # Wejście na otwarciu
    assert trades[0].exit_price == data.loc[trades[0].exit_time]['open']  # Zamknięte na otwarciu następnej świecy
    
    # Druga transakcja SELL powinna być otwarta po zamknięciu BUY
    assert trades[1].direction == 'SELL'
    assert trades[1].entry_price == data.loc[trades[1].entry_time]['open']  # Wejście na otwarciu
    assert trades[1].exit_price == data.loc[trades[1].exit_time]['open']  # Zamknięte na otwarciu następnej świecy