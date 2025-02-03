"""
Testy jednostkowe dla modułu basic_strategy.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
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
        'ollama_analysis': 'RECOMMENDATION: BUY',
        'claude_analysis': 'Rekomendacja: long\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips'
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
        'ollama_analysis': 'RECOMMENDATION: SELL',
        'claude_analysis': 'Rekomendacja: short\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips'
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
    mock_ollama.analyze_market_data.return_value = "RECOMMENDATION: BUY"
    mock_claude.analyze_market_conditions.return_value = "Rekomendacja: long"

    analysis = await strategy.analyze_market('EURUSD')
    assert analysis is not None
    assert 'market_data' in analysis
    assert 'technical_indicators' in analysis
    assert 'ollama_analysis' in analysis
    assert 'claude_analysis' in analysis
    
    # Sprawdź szczegółowo dane rynkowe
    market_data = analysis['market_data']
    assert market_data['symbol'] == 'EURUSD'
    assert isinstance(market_data['current_price'], float)
    assert isinstance(market_data['sma_20'], float)
    assert isinstance(market_data['sma_50'], float)
    assert isinstance(market_data['price_change_24h'], float)
    assert isinstance(market_data['volume_24h'], float)

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
    assert isinstance(analysis, dict)  # Upewnij się, że analiza jest słownikiem

    # 2. Test generowania sygnałów
    signals = await strategy.generate_signals(analysis)  # Przekaż wynik analizy zamiast stringa
    assert len(signals) > 0
    for signal in signals:
        assert signal.symbol == 'EURUSD'
        assert signal.confidence > 0
        assert signal.stop_loss is not None
        assert signal.take_profit is not None

    # 3. Test wykonania sygnałów
    for signal in signals:
        result = await strategy.execute_signal(signal)
        assert result is True  # Sygnały powinny być wykonane pomyślnie

    # 4. Test metryki wydajności
    metrics = strategy.calculate_performance_metrics()
    assert metrics is not None
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics
    assert 'max_drawdown' in metrics 