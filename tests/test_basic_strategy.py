"""
Testy jednostkowe dla modułu basic_strategy.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import memory_profiler

from src.strategies.basic_strategy import BasicStrategy
from src.utils.logger import TradingLogger
from src.interfaces.connectors import IMT5Connector, IOllamaConnector, IAnthropicConnector, IDBHandler

@pytest.fixture
def mock_mt5(sample_data):
    """Mock dla MT5 connectora."""
    mock = AsyncMock()
    mock.get_rates.return_value = sample_data
    mock.place_order.return_value = Mock()
    return mock

@pytest.fixture
def mock_ollama():
    """Mock dla konektora Ollama."""
    mock = AsyncMock()
    mock.analyze_market_data = AsyncMock()
    return mock

@pytest.fixture
def mock_claude():
    """Mock dla konektora Claude."""
    mock = AsyncMock()
    mock.analyze_market_conditions = AsyncMock()
    return mock

@pytest.fixture
def mock_db():
    """Mock dla handlera bazy danych."""
    mock = Mock()
    mock.save_trade = Mock()
    return mock

@pytest.fixture
def sample_data():
    """Przygotowuje przykładowe dane rynkowe."""
    # Generuj daty dla ostatnich 100 godzin
    dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
    
    # Generuj losowe ceny
    close_prices = np.random.normal(1.1000, 0.0010, size=100)
    
    # Przygotuj DataFrame
    data = pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'open': close_prices + np.random.normal(0, 0.0002, size=100),
        'high': close_prices + np.abs(np.random.normal(0, 0.0005, size=100)),
        'low': close_prices - np.abs(np.random.normal(0, 0.0005, size=100)),
        'close': close_prices,
        'volume': np.random.randint(100, 1000, size=100),
        'spread': np.random.randint(1, 5, size=100),
        'real_volume': np.random.randint(1000, 10000, size=100)
    })
    
    return data

@pytest.fixture
def mock_logger():
    """Mock dla loggera."""
    return Mock(spec=TradingLogger)

def test_strategy_initialization(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test inicjalizacji strategii."""
    config = {
        'max_position_size': 1.0,
        'stop_loss_pips': 50,
        'take_profit_pips': 100,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )
    
    assert strategy.config == config
    assert strategy.mt5 == mock_mt5
    assert strategy.ollama == mock_ollama
    assert strategy.claude == mock_claude
    assert strategy.db == mock_db

@pytest.mark.asyncio
async def test_generate_signals_no_position(sample_data, mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test generowania sygnałów bez otwartej pozycji."""
    config = {
        'max_position_size': 1.0,
        'stop_loss_pips': 50,
        'take_profit_pips': 100,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'allowed_symbols': ['EURUSD']
    }
    
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )
    
    # Przygotuj dane analizy
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1000,
            'sma_20': 1.0950,
            'sma_50': 1.0900,
            'price_change_24h': 0.5,
            'volume_24h': 10000.0
        },
        'technical_indicators': {
            'sma_20': 1.0950,
            'sma_50': 1.0900
        },
        'ollama_analysis': {
            'recommendation': 'BUY',
            'confidence': 0.8,
            'reason': 'Trend wzrostowy'
        },
        'claude_analysis': {
            'recommendation': 'LONG',
            'stop_loss_pips': 20,
            'take_profit_pips': 60,
            'confidence': 0.9
        }
    }
    
    signals = await strategy.generate_signals(analysis)
    assert isinstance(signals, dict)
    assert 'action' in signals
    assert signals['action'] in ['BUY', 'SELL', 'WAIT']
    
    if signals['action'] in ['BUY', 'SELL']:
        assert 'volume' in signals
        assert signals['volume'] <= config['max_position_size']
        assert 'stop_loss' in signals
        assert 'take_profit' in signals

@pytest.mark.asyncio
async def test_generate_signals_with_position(sample_data, mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test generowania sygnałów z otwartą pozycją."""
    config = {
        'max_position_size': 1.0,
        'stop_loss_pips': 50,
        'take_profit_pips': 100,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'allowed_symbols': ['EURUSD']
    }
    
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )
    
    # Przygotuj dane analizy
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1000,
            'sma_20': 1.0950,
            'sma_50': 1.0900,
            'price_change_24h': 0.5,
            'volume_24h': 10000.0
        },
        'technical_indicators': {
            'sma_20': 1.0950,
            'sma_50': 1.0900
        },
        'ollama_analysis': {
            'recommendation': 'SELL',
            'confidence': 0.8,
            'reason': 'Trend spadkowy'
        },
        'claude_analysis': {
            'recommendation': 'SHORT',
            'stop_loss_pips': 20,
            'take_profit_pips': 60,
            'confidence': 0.9
        }
    }
    
    signals = await strategy.generate_signals(analysis)
    assert isinstance(signals, dict)
    assert 'action' in signals
    assert signals['action'] in ['BUY', 'SELL', 'WAIT']

@pytest.mark.asyncio
async def test_generate_signals_error_handling(sample_data, mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów podczas generowania sygnałów."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )
    
    # Test z nieprawidłowymi danymi
    invalid_analysis = {'market_data': 'invalid'}
    
    with pytest.raises(ValueError):
        await strategy.generate_signals(invalid_analysis)

@pytest.mark.asyncio
async def test_analyze_market(sample_data, mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test analizy rynku."""
    config = {
        'max_position_size': 1.0,
        'stop_loss_pips': 50,
        'take_profit_pips': 100,
        'allowed_symbols': ['EURUSD']
    }

    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )

    # Przygotuj dane testowe
    mock_ollama.analyze_market_data.return_value = {
        'recommendation': 'BUY',
        'confidence': 0.8,
        'reason': 'Trend wzrostowy'
    }
    mock_claude.analyze_market_conditions.return_value = {
        'recommendation': 'LONG',
        'stop_loss_pips': 20,
        'take_profit_pips': 60,
        'confidence': 0.9
    }

    analysis = await strategy.analyze_market('EURUSD')
    assert analysis is not None
    assert 'market_data' in analysis
    assert 'technical_indicators' in analysis
    assert 'ollama_analysis' in analysis
    assert 'claude_analysis' in analysis
    
    # Sprawdź szczegółowo dane rynkowe
    market_data = analysis['market_data']
    assert 'symbol' in market_data
    assert market_data['symbol'] == 'EURUSD'
    assert isinstance(market_data['current_price'], float)
    assert isinstance(market_data['volume'], float)
    assert 'trend' in market_data

@pytest.mark.asyncio
async def test_execute_signals(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test wykonywania sygnałów."""
    config = {
        'max_position_size': 1.0,
        'stop_loss_pips': 50,
        'take_profit_pips': 100,
        'allowed_symbols': ['EURUSD']
    }

    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )

    # Przygotuj sygnały
    signals = {
        'symbol': 'EURUSD',
        'action': 'BUY',
        'volume': 0.1,
        'entry_price': 1.1000,
        'stop_loss': 1.0950,
        'take_profit': 1.1100
    }

    # Przygotuj mock odpowiedzi
    mock_mt5.place_order.return_value = {
        'ticket': 12345,
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 0.1,
        'price': 1.1000,
        'sl': 1.0950,
        'tp': 1.1100
    }
    mock_db.save_trade = AsyncMock()

    result = await strategy.execute_signals(signals)
    assert result is not None
    assert result['ticket'] == 12345
    assert result['symbol'] == 'EURUSD'
    assert result['type'] == 'BUY'
    assert result['volume'] == 0.1
    assert result['price'] == 1.1000
    assert result['sl'] == 1.0950
    assert result['tp'] == 1.1100

def test_calculate_position_size(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obliczania wielkości pozycji."""
    config = {
        'max_position_size': 1.0,
        'risk_per_trade': 0.02,  # 2% kapitału na trade
        'account_balance': 10000.0,
        'allowed_symbols': ['EURUSD']
    }
    
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )
    
    # Przygotuj dane
    entry_price = 1.1000
    stop_loss = 1.0950  # 50 pips SL
    
    # Oblicz wielkość pozycji
    position_size = strategy._calculate_position_size('EURUSD', entry_price, stop_loss)
    
    # Sprawdź czy wielkość pozycji jest w limicie
    assert position_size <= config['max_position_size']
    assert position_size > 0
    
    # Sprawdź czy ryzyko jest prawidłowe
    risk_amount = abs(entry_price - stop_loss) * position_size * 100000  # Dla EURUSD
    expected_risk = config['account_balance'] * config['risk_per_trade']
    assert abs(risk_amount - expected_risk) < 1.0  # Tolerancja na zaokrąglenia

def test_calculate_stop_loss(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obliczania poziomu stop loss."""
    config = {
        'stop_loss_pips': 50,
        'allowed_symbols': ['EURUSD']
    }

    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )

    # Test dla pozycji długiej
    entry_price = 1.1000
    stop_loss = strategy._calculate_stop_loss('EURUSD', 'BUY', entry_price)
    assert stop_loss < entry_price
    assert round(abs(entry_price - stop_loss) / 0.0001) == config['stop_loss_pips']

    # Test dla pozycji krótkiej
    stop_loss = strategy._calculate_stop_loss('EURUSD', 'SELL', entry_price)
    assert stop_loss > entry_price
    assert round(abs(stop_loss - entry_price) / 0.0001) == config['stop_loss_pips']

def test_calculate_take_profit(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obliczania poziomu take profit."""
    config = {
        'take_profit_pips': 100,
        'allowed_symbols': ['EURUSD']
    }

    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )

    # Test dla pozycji długiej
    entry_price = 1.1000
    take_profit = strategy._calculate_take_profit('EURUSD', 'BUY', entry_price)
    assert take_profit > entry_price
    assert round(abs(take_profit - entry_price) / 0.0001) == config['take_profit_pips']

    # Test dla pozycji krótkiej
    take_profit = strategy._calculate_take_profit('EURUSD', 'SELL', entry_price)
    assert take_profit < entry_price
    assert round(abs(entry_price - take_profit) / 0.0001) == config['take_profit_pips']

@pytest.mark.asyncio
async def test_strategy_performance_metrics(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test metryk wydajności strategii."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={'max_position_size': 1.0}
    )
    
    # Symuluj historię tradingu
    trades = [
        {'type': 'BUY', 'profit': 100, 'pips': 50},
        {'type': 'SELL', 'profit': -30, 'pips': -15},
        {'type': 'BUY', 'profit': 80, 'pips': 40}
    ]
    
    metrics = strategy.calculate_performance_metrics(trades)
    assert metrics['win_rate'] > 0.5
    assert metrics['profit_factor'] > 1.0
    assert metrics['average_win'] > abs(metrics['average_loss'])
    assert metrics['max_drawdown'] < 100
    assert metrics['sharpe_ratio'] > 0
    assert metrics['recovery_factor'] > 1.0
    assert metrics['profit_per_trade'] > 0

@pytest.mark.benchmark
def test_strategy_analysis_speed(benchmark, mock_mt5, mock_ollama, mock_claude, mock_db, sample_data):
    """Test wydajności analizy strategii."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )
    
    def analyze_market():
        """Funkcja do testowania wydajności."""
        analysis = strategy.analyze_technical_indicators(sample_data)
        strategy.validate_analysis(analysis)
        return analysis
    
    result = benchmark(analyze_market)
    assert result is not None

@pytest.mark.memory
@pytest.mark.asyncio
async def test_strategy_memory_usage(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test zużycia pamięci podczas analizy strategii."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    @memory_profiler.profile
    async def analyze_large_dataset():
        """Funkcja do profilowania pamięci."""
        # Generuj duży zestaw danych
        dates = pd.date_range(start='2024-01-01', periods=10000, freq='1h')
        data = pd.DataFrame({
            'open': np.random.normal(1.1000, 0.0010, size=10000),
            'high': np.random.normal(1.1010, 0.0010, size=10000),
            'low': np.random.normal(1.0990, 0.0010, size=10000),
            'close': np.random.normal(1.1000, 0.0010, size=10000),
            'volume': np.random.randint(1000, 10000, size=10000)
        }, index=dates)

        # Wykonaj analizę
        analysis = await strategy.analyze_technical_indicators(data)
        await strategy.validate_analysis(analysis)
        return analysis

    mem_usage = memory_profiler.memory_usage((analyze_large_dataset, (), {}))
    assert max(mem_usage) < 200  # MB - zwiększony limit ze względu na duży zestaw danych

@pytest.mark.asyncio
async def test_strategy_recovery(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test odzyskiwania strategii po błędach."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={'allowed_symbols': ['EURUSD']}
    )

    # 1. Test błędu pobierania danych
    mock_mt5.get_rates.side_effect = Exception("Connection error")
    try:
        await strategy.analyze_market('EURUSD')
        assert False, "Powinien wystąpić wyjątek"
    except Exception as e:
        assert str(e) == "Connection error"

    # 2. Test błędu analizy technicznej
    mock_mt5.get_rates.side_effect = None  # Reset błędu
    mock_mt5.get_rates.return_value = pd.DataFrame({
        'time': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    })  # Puste dane
    try:
        await strategy.analyze_market('EURUSD')
        assert False, "Powinien wystąpić wyjątek"
    except Exception as e:
        assert "Brak danych" in str(e)

@pytest.mark.asyncio
async def test_strategy_integration(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test integracji wszystkich komponentów strategii."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={'allowed_symbols': ['EURUSD']}
    )

    # Przygotuj dane testowe - dodaj więcej danych historycznych
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    test_data = pd.DataFrame({
        'time': [int(d.timestamp()) for d in dates],
        'open': np.random.normal(1.1000, 0.0010, size=100),
        'high': np.random.normal(1.1100, 0.0010, size=100),
        'low': np.random.normal(1.0900, 0.0010, size=100),
        'close': np.random.normal(1.1000, 0.0010, size=100),
        'volume': np.random.randint(1000, 2000, size=100),
        'spread': np.random.randint(1, 5, size=100),
        'real_volume': np.random.randint(10000, 20000, size=100)
    })
    mock_mt5.get_rates.return_value = test_data

    # Przygotuj odpowiedzi AI
    mock_ollama.analyze_market_data.return_value = {
        'sentiment': 'bullish',
        'confidence': 0.8,
        'recommendation': 'BUY'
    }
    mock_claude.analyze_market_conditions.return_value = {
        'sentiment': 'bullish',
        'confidence': 0.9,
        'recommendation': 'BUY'
    }

    # 1. Test analizy rynku
    analysis = await strategy.analyze_market('EURUSD')
    assert analysis is not None
    assert isinstance(analysis, dict)

    # Dodaj brakujące wskaźniki techniczne
    analysis['market_data']['technical_indicators'] = {
        'sma_20': 1.1050,
        'sma_50': 1.1000,
        'rsi': 65,
        'macd': 0.0015,
        'macd_signal': 0.0010,
        'bb_upper': 1.1100,
        'bb_lower': 1.0900
    }

    # 2. Test generowania sygnałów
    signals = await strategy.generate_signals(analysis)
    assert isinstance(signals, dict)
    assert 'signals' in signals
    assert len(signals['signals']) > 0
    
    for signal in signals['signals']:
        assert isinstance(signal, dict)
        assert 'symbol' in signal
        assert signal['symbol'] == 'EURUSD'

@pytest.mark.asyncio
async def test_analyze_market_error_handling(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów podczas analizy rynku."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z pustymi danymi
    mock_mt5.get_rates.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="Brak danych"):
        await strategy.analyze_market('EURUSD')

    # Test z błędem połączenia
    mock_mt5.get_rates.side_effect = Exception("Błąd połączenia")
    with pytest.raises(Exception, match="Błąd połączenia"):
        await strategy.analyze_market('EURUSD')

@pytest.mark.asyncio
async def test_execute_signals_error_handling(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów podczas wykonywania sygnałów."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={'allowed_symbols': ['EURUSD']}
    )

    # Test z niedozwolonym symbolem
    signals = {
        'symbol': 'USDJPY',  # Niedozwolony symbol
        'action': 'BUY',
        'volume': 0.1,
        'entry_price': 1.1000,
        'stop_loss': 1.0950,
        'take_profit': 1.1100
    }
    result = await strategy.execute_signals(signals)
    assert result is None

    # Test z błędem podczas składania zlecenia
    signals['symbol'] = 'EURUSD'
    mock_mt5.place_order.side_effect = Exception("Błąd składania zlecenia")
    result = await strategy.execute_signals(signals)
    assert result is None

def test_calculate_position_size_error_handling(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów podczas obliczania wielkości pozycji."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={'max_position_size': 0.5}
    )

    # Test z błędnymi parametrami
    result = strategy._calculate_position_size('EURUSD', None, 1.0950)
    assert result == strategy.config.get('max_position_size', 1.0)

    # Test z błędem obliczeń
    result = strategy._calculate_position_size('EURUSD', 1.1000, 1.1000)  # SL = entry
    assert result == strategy.config.get('max_position_size', 1.0)

def test_calculate_stop_loss_error_handling(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów podczas obliczania stop loss."""
    config = {
        'stop_loss_pips': 50,
        'take_profit_pips': 100
    }
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )

    # Test z błędnymi parametrami
    result = strategy._calculate_stop_loss('EURUSD', None, 1.1000)
    assert result == 1.1000  # Powinno zwrócić entry_price

    # Test z błędem obliczeń
    result = strategy._calculate_stop_loss('EURUSD', 'INVALID', 1.1000)
    assert result == 1.1000  # Powinno zwrócić entry_price

    # Test z prawidłowymi parametrami
    result = strategy._calculate_stop_loss('EURUSD', 'BUY', 1.1000)
    assert result == 1.0950  # 50 pips poniżej entry_price

def test_calculate_take_profit_error_handling(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów podczas obliczania take profit."""
    config = {
        'stop_loss_pips': 50,
        'take_profit_pips': 100
    }
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config=config
    )

    # Test z błędnymi parametrami
    result = strategy._calculate_take_profit('EURUSD', None, 1.1000)
    assert result == 1.1000  # Powinno zwrócić entry_price

    # Test z błędem obliczeń
    result = strategy._calculate_take_profit('EURUSD', 'INVALID', 1.1000)
    assert result == 1.1000  # Powinno zwrócić entry_price

    # Test z prawidłowymi parametrami
    result = strategy._calculate_take_profit('EURUSD', 'BUY', 1.1000)
    assert result == 1.1100  # 100 pips powyżej entry_price

def test_analyze_technical_indicators_error_handling(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów podczas analizy wskaźników technicznych."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z pustym DataFrame
    with pytest.raises(Exception):
        strategy.analyze_technical_indicators(pd.DataFrame())

    # Test z nieprawidłowymi danymi
    invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
    with pytest.raises(Exception):
        strategy.analyze_technical_indicators(invalid_df)

def test_calculate_performance_metrics_error_handling(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów podczas obliczania metryk wydajności."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z pustą listą transakcji
    result = strategy.calculate_performance_metrics([])
    assert result['win_rate'] == 0.0
    assert result['profit_factor'] == 0.0

    # Test z nieprawidłowymi danymi
    with pytest.raises(Exception):
        strategy.calculate_performance_metrics([{'invalid': 'data'}])

def test_validate_analysis_error_handling_extended(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test obsługi błędów w validate_analysis."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z None jako analiza
    assert not strategy.validate_analysis(None)

    # Test z pustym słownikiem
    assert not strategy.validate_analysis({})

    # Test z brakującymi sekcjami
    invalid_analysis = {
        'trend': {},  # Brak wymaganych pól
        'momentum': {
            'rsi': None,  # Nieprawidłowa wartość
            'macd': None
        }
    }
    assert not strategy.validate_analysis(invalid_analysis)

def test_calculate_stop_loss_validation_extended_2(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji w calculate_stop_loss."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={'stop_loss_pips': 50}
    )

    # Test z None jako parametry
    assert strategy._calculate_stop_loss(None, 'BUY', 1.1000) == 1.1000
    assert strategy._calculate_stop_loss('EURUSD', None, 1.1000) == 1.1000
    assert strategy._calculate_stop_loss('EURUSD', 'BUY', None) is None

    # Test z JPY
    jpy_result = strategy._calculate_stop_loss('USDJPY', 'BUY', 150.000)
    assert abs(jpy_result - 149.500) < 0.001  # 50 pips dla JPY to 0.500

def test_calculate_take_profit_validation_extended_2(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji w calculate_take_profit."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={'take_profit_pips': 100}
    )

    # Test z None jako parametry
    assert strategy._calculate_take_profit(None, 'BUY', 1.1000) == 1.1000
    assert strategy._calculate_take_profit('EURUSD', None, 1.1000) == 1.1000
    assert strategy._calculate_take_profit('EURUSD', 'BUY', None) is None

    # Test z JPY
    jpy_result = strategy._calculate_take_profit('USDJPY', 'BUY', 150.000)
    assert abs(jpy_result - 151.000) < 0.001  # 100 pips dla JPY to 1.000

def test_validate_analysis_validation_extended(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji w validate_analysis."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z None jako analiza
    assert not strategy.validate_analysis(None)

    # Test z nieprawidłowymi wartościami
    invalid_analysis = {
        'trend': {
            'sma_20': None,
            'sma_50': None,
            'sma_200': None,
            'trend_direction': 'INVALID',
            'trend_strength': -1
        },
        'momentum': {
            'rsi': 101,  # Nieprawidłowa wartość
            'macd': None,
            'macd_signal': None,
            'macd_hist': None,
            'momentum': None
        },
        'volatility': {
            'bb_upper': 1.0,
            'bb_middle': 1.1,  # Nieprawidłowa kolejność
            'bb_lower': 1.2,  # Nieprawidłowa kolejność
            'bb_width': -1,
            'atr': 0
        }
    }
    assert not strategy.validate_analysis(invalid_analysis)

@pytest.mark.asyncio
async def test_analyze_market_data_validation_extended(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji danych w analyze_market."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z błędem w danych historycznych
    mock_mt5.get_rates.return_value = pd.DataFrame({
        'close': [None] * 100,  # Nieprawidłowe wartości
        'volume': [None] * 100
    })

    with pytest.raises(Exception):
        await strategy.analyze_market('EURUSD')

@pytest.mark.asyncio
async def test_generate_signals_validation_extended_2(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji w generate_signals."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z nieprawidłowymi danymi market_data
    analysis = {
        'market_data': {
            'symbol': None,  # Nieprawidłowa wartość
            'current_price': None  # Nieprawidłowa wartość
        },
        'ollama_analysis': {'recommendation': 'BUY'},
        'claude_analysis': {'recommendation': 'BUY'}
    }

    result = await strategy.generate_signals(analysis)
    assert result['action'] == 'WAIT'
    assert len(result['signals']) == 0

@pytest.mark.asyncio
async def test_execute_signals_validation_extended(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji w execute_signals."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={'allowed_symbols': ['EURUSD']}
    )

    # Test z błędem podczas zapisu do bazy
    signals = {
        'symbol': 'EURUSD',
        'action': 'BUY',
        'volume': 0.1,
        'entry_price': 1.1000,
        'stop_loss': 1.0950,
        'take_profit': 1.1100
    }

    # Symuluj udane złożenie zlecenia
    mock_mt5.place_order.return_value = {
        'ticket': 12345,
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 0.1,
        'price': 1.1000,
        'sl': 1.0950,
        'tp': 1.1100
    }

    # Symuluj błąd zapisu do bazy
    mock_db.save_trade.side_effect = Exception("Database error")
    result = await strategy.execute_signals(signals)
    assert result is None

    # Test z błędem podczas składania zlecenia
    mock_mt5.place_order.side_effect = Exception("Order error")
    result = await strategy.execute_signals(signals)
    assert result is None

    # Test z nieprawidłowym symbolem
    signals['symbol'] = 'INVALID'
    result = await strategy.execute_signals(signals)
    assert result is None

def test_validate_analysis_validation_extended_2(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji w validate_analysis."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z nieprawidłowymi wartościami
    invalid_analysis = {
        'trend': {
            'sma_20': None,
            'sma_50': None,
            'sma_200': None,
            'trend_direction': None,
            'trend_strength': None
        },
        'momentum': {
            'rsi': None,
            'macd': None,
            'macd_signal': None,
            'macd_hist': None,
            'momentum': None
        },
        'volatility': {
            'bb_upper': None,
            'bb_middle': None,
            'bb_lower': None,
            'bb_width': None,
            'atr': None
        }
    }
    assert not strategy.validate_analysis(invalid_analysis)

@pytest.mark.asyncio
async def test_analyze_market_missing_data(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test analizy rynku gdy brakuje danych."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Symuluj brak danych z MT5
    mock_mt5.get_rates.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="Brak danych"):
        await strategy.analyze_market('EURUSD')

    # Symuluj błąd w analizie AI
    mock_mt5.get_rates.return_value = sample_data
    mock_ollama.analyze_market_data.side_effect = Exception("Błąd analizy")
    mock_claude.analyze_market_conditions.side_effect = Exception("Błąd analizy")

    with pytest.raises(Exception):
        await strategy.analyze_market('EURUSD')

@pytest.mark.asyncio
async def test_generate_signals_invalid_data(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test generowania sygnałów z nieprawidłowymi danymi."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z brakującymi sekcjami
    invalid_analysis = {
        'market_data': {}
    }
    with pytest.raises(ValueError, match="Brak wymaganego pola: ollama_analysis"):
        await strategy.generate_signals(invalid_analysis)

    # Test z nieprawidłowymi rekomendacjami AI
    invalid_analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1000,
            'technical_indicators': {
                'sma_20': 1.0950,
                'sma_50': 1.0900
            }
        },
        'ollama_analysis': {
            'recommendation': 'INVALID',
            'confidence': 0.8
        },
        'claude_analysis': {
            'recommendation': 'INVALID',
            'confidence': 0.9
        }
    }
    result = await strategy.generate_signals(invalid_analysis)
    assert result['action'] == 'WAIT'
    assert len(result['signals']) == 0

def test_calculate_performance_metrics_validation_extended_3(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji w calculate_performance_metrics."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z pustą listą
    metrics = strategy.calculate_performance_metrics([])
    assert metrics['win_rate'] == 0.0
    assert metrics['profit_factor'] == 0.0
    assert metrics['average_win'] == 0.0
    assert metrics['average_loss'] == 0.0
    assert metrics['max_drawdown'] == 0.0
    assert metrics['sharpe_ratio'] == 0.0
    assert metrics['recovery_factor'] == 0.0
    assert metrics['profit_per_trade'] == 0.0

    # Test z samymi stratami
    trades = [
        {'type': 'BUY', 'profit': -10},
        {'type': 'SELL', 'profit': -20}
    ]
    metrics = strategy.calculate_performance_metrics(trades)
    assert metrics['win_rate'] == 0.0
    assert metrics['profit_factor'] == 0.0
    assert metrics['average_win'] == 0.0
    assert metrics['average_loss'] == -15.0

    # Test z samymi zyskami
    trades = [
        {'type': 'BUY', 'profit': 10},
        {'type': 'SELL', 'profit': 20}
    ]
    metrics = strategy.calculate_performance_metrics(trades)
    assert metrics['win_rate'] == 1.0
    assert metrics['average_win'] == 15.0
    assert metrics['average_loss'] == 0.0

def test_validate_analysis_validation_extended_3(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test rozszerzonej walidacji w validate_analysis."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={}
    )

    # Test z nieprawidłowymi wartościami wskaźników
    invalid_analysis = {
        'trend': {
            'sma_20': 1.1000,
            'sma_50': 1.0950,
            'sma_200': 1.0900,
            'trend_direction': 'UP',
            'trend_strength': 0.005
        },
        'momentum': {
            'rsi': 150,  # Nieprawidłowa wartość RSI
            'macd': 0.0015,
            'macd_signal': 0.0010,
            'macd_hist': 0.0005,
            'momentum': 0.01
        },
        'volatility': {
            'bb_upper': 1.1100,
            'bb_middle': 1.1000,
            'bb_lower': 1.0900,
            'bb_width': 0.02,
            'atr': 0.0015
        }
    }
    assert not strategy.validate_analysis(invalid_analysis)

    # Test z nieprawidłową kolejnością Bollinger Bands
    invalid_analysis['momentum']['rsi'] = 65  # Popraw RSI
    invalid_analysis['volatility']['bb_upper'] = 1.0900  # Nieprawidłowa kolejność
    invalid_analysis['volatility']['bb_lower'] = 1.1100
    assert not strategy.validate_analysis(invalid_analysis)

    # Test z ujemną wartością ATR
    invalid_analysis['volatility']['bb_upper'] = 1.1100  # Przywróć prawidłową kolejność
    invalid_analysis['volatility']['bb_lower'] = 1.0900
    invalid_analysis['volatility']['atr'] = -0.0015  # Nieprawidłowa wartość ATR
    assert not strategy.validate_analysis(invalid_analysis)

@pytest.mark.asyncio
async def test_edge_cases_coverage(mock_mt5, mock_ollama, mock_claude, mock_db):
    """Test pokrywający brzegowe przypadki w różnych metodach."""
    strategy = BasicStrategy(
        mt5_connector=mock_mt5,
        ollama_connector=mock_ollama,
        anthropic_connector=mock_claude,
        db_handler=mock_db,
        config={
            'risk_per_trade': 0.02,
            'account_balance': 10000.0,
            'max_position_size': 1.0,
            'stop_loss_pips': 50,
            'take_profit_pips': 100
        }
    )

    # Test dla analyze_market
    mock_mt5.get_rates.return_value = pd.DataFrame({
        'close': [1.1000] * 10,
        'volume': [1000.0] * 10
    })
    with pytest.raises(ValueError):
        await strategy.analyze_market('EURUSD')

    # Test dla generate_signals
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1000,
            'technical_indicators': None  # Brak wskaźników technicznych
        },
        'ollama_analysis': {
            'recommendation': 'BUY',
            'confidence': 0.8
        },
        'claude_analysis': {
            'recommendation': 'BUY',
            'confidence': 0.9
        }
    }
    result = await strategy.generate_signals(analysis)
    assert result['action'] == 'WAIT'

    # Test dla _calculate_position_size
    position_size = strategy._calculate_position_size('USDJPY', 150.000, 149.500)
    assert position_size <= strategy.max_position_size

    # Test dla _calculate_stop_loss i _calculate_take_profit
    sl = strategy._calculate_stop_loss('USDJPY', 'BUY', 150.000)
    tp = strategy._calculate_take_profit('USDJPY', 'BUY', 150.000)
    assert sl < 150.000
    assert tp > 150.000

    # Test dla calculate_performance_metrics
    trades = [
        {'type': 'BUY', 'profit': 0},  # Neutralna transakcja
        {'type': 'SELL', 'profit': 0}  # Neutralna transakcja
    ]
    metrics = strategy.calculate_performance_metrics(trades)
    assert metrics['win_rate'] == 0.0
    assert metrics['profit_factor'] == 0.0
    assert metrics['average_win'] == 0.0
    assert metrics['average_loss'] == 0.0

    # Test dla validate_analysis
    invalid_analysis = {
        'trend': {
            'sma_20': None,  # Brak wartości
            'sma_50': None,
            'sma_200': None,
            'trend_direction': 'UP',
            'trend_strength': 0.005
        },
        'momentum': {
            'rsi': 65,
            'macd': None,  # Brak wartości
            'macd_signal': None,
            'macd_hist': None,
            'momentum': 0.01
        },
        'volatility': {
            'bb_upper': 1.1100,
            'bb_middle': 1.1000,
            'bb_lower': 1.0900,
            'bb_width': 0.02,
            'atr': 0.0015
        }
    }
    assert not strategy.validate_analysis(invalid_analysis) 