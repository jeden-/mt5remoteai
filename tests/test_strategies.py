"""
Moduł zawierający testy dla strategii tradingowych.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from src.strategies.base_strategy import BaseStrategy
from src.strategies.basic_strategy import BasicStrategy


class TestStrategy(BaseStrategy):
    """Testowa implementacja strategii bazowej."""
    async def analyze_market(self, symbol):
        return {'test': 'data'}
        
    async def generate_signals(self, analysis):
        return {'action': 'BUY'}
        
    async def execute_signals(self, signals):
        return {'ticket': 123}


@pytest.fixture
def mock_connectors():
    """Fixture tworzący mocki konektorów."""
    mt5_mock = Mock()
    mt5_mock.get_rates.return_value = [
        {'time': '2024-01-01', 'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.15, 'volume': 1000}
        for _ in range(100)
    ]
    mt5_mock.place_order.return_value = {'ticket': 123, 'symbol': 'EURUSD', 'type': 'BUY', 'volume': 0.1, 'price': 1.1, 'sl': 1.09, 'tp': 1.12}
    
    ollama_mock = Mock()
    ollama_mock.analyze_market_data = AsyncMock(return_value="RECOMMENDATION: BUY\nStrength: High")
    
    claude_mock = Mock()
    claude_mock.analyze_market_conditions = AsyncMock(return_value="Rekomendacja: long\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips")
    
    db_mock = Mock()
    db_mock.save_trade.return_value = None
    
    return mt5_mock, ollama_mock, claude_mock, db_mock


@pytest.fixture
def config():
    """Fixture tworzący konfigurację."""
    return {
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'allowed_symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'ollama_prompt_template': 'Analyze {symbol}',
        'claude_prompt_template': 'Analyze {symbol}'
    }


@pytest.fixture
def test_strategy(mock_connectors, config):
    """Fixture tworzący testową strategię."""
    mt5_mock, ollama_mock, claude_mock, db_mock = mock_connectors
    return TestStrategy(mt5_mock, ollama_mock, claude_mock, db_mock, config)


@pytest.fixture
def basic_strategy(mock_connectors, config):
    """Fixture tworzący podstawową strategię."""
    mt5_mock, ollama_mock, claude_mock, db_mock = mock_connectors
    return BasicStrategy(mt5_mock, ollama_mock, claude_mock, db_mock, config)


@pytest.mark.asyncio
async def test_base_strategy_initialization(test_strategy, config):
    """Test inicjalizacji strategii bazowej."""
    assert test_strategy.max_position_size == config['max_position_size']
    assert test_strategy.max_risk_per_trade == config['max_risk_per_trade']
    assert test_strategy.allowed_symbols == config['allowed_symbols']


@pytest.mark.asyncio
async def test_base_strategy_run(test_strategy):
    """Test metody run strategii bazowej."""
    result = await test_strategy.run('EURUSD')
    assert result == {'ticket': 123}


@pytest.mark.asyncio
async def test_base_strategy_run_error(test_strategy):
    """Test obsługi błędów w metodzie run."""
    test_strategy.analyze_market = AsyncMock(side_effect=Exception("Test error"))
    result = await test_strategy.run('EURUSD')
    assert result is None


@pytest.mark.asyncio
async def test_basic_strategy_analyze_market(basic_strategy, mock_connectors):
    """Test analizy rynku w strategii podstawowej."""
    mt5_mock, _, _, _ = mock_connectors
    
    result = await basic_strategy.analyze_market('EURUSD')
    
    assert 'market_data' in result
    assert 'technical_indicators' in result
    assert 'ollama_analysis' in result
    assert 'claude_analysis' in result
    
    mt5_mock.get_rates.assert_called_once_with('EURUSD', 'H1', 100)


@pytest.mark.asyncio
async def test_basic_strategy_analyze_market_error(basic_strategy, mock_connectors):
    """Test obsługi błędów w analizie rynku."""
    mt5_mock, _, _, _ = mock_connectors
    mt5_mock.get_rates.side_effect = Exception("Test error")
    
    with pytest.raises(Exception):
        await basic_strategy.analyze_market('EURUSD')


@pytest.mark.asyncio
async def test_basic_strategy_generate_signals(basic_strategy):
    """Test generowania sygnałów w strategii podstawowej."""
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1
        },
        'ollama_analysis': 'RECOMMENDATION: BUY\nStrength: High',
        'claude_analysis': 'Rekomendacja: long\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips'
    }
    
    signals = await basic_strategy.generate_signals(analysis)
    
    assert signals['action'] == 'BUY'
    assert signals['symbol'] == 'EURUSD'
    assert signals['entry_price'] == 1.1
    assert signals['stop_loss'] < signals['entry_price']
    assert signals['take_profit'] > signals['entry_price']


@pytest.mark.asyncio
async def test_basic_strategy_generate_signals_wait(basic_strategy):
    """Test generowania sygnału WAIT."""
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1
        },
        'ollama_analysis': 'RECOMMENDATION: WAIT\nStrength: Low',
        'claude_analysis': 'Rekomendacja: neutral\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips'
    }
    
    signals = await basic_strategy.generate_signals(analysis)
    assert signals['action'] == 'WAIT'


@pytest.mark.asyncio
async def test_basic_strategy_execute_signals(basic_strategy, mock_connectors):
    """Test wykonywania sygnałów w strategii podstawowej."""
    _, _, _, db_mock = mock_connectors
    
    signals = {
        'symbol': 'EURUSD',
        'action': 'BUY',
        'entry_price': 1.1,
        'stop_loss': 1.09,
        'take_profit': 1.12,
        'volume': 0.1
    }
    
    result = await basic_strategy.execute_signals(signals)
    
    assert result is not None
    assert result['ticket'] == 123
    db_mock.save_trade.assert_called_once()


@pytest.mark.asyncio
async def test_basic_strategy_execute_signals_wait(basic_strategy, mock_connectors):
    """Test wykonywania sygnału WAIT."""
    signals = {
        'symbol': 'EURUSD',
        'action': 'WAIT',
        'entry_price': 1.1,
        'stop_loss': 1.09,
        'take_profit': 1.12,
        'volume': 0.1
    }
    
    result = await basic_strategy.execute_signals(signals)
    assert result is None


@pytest.mark.asyncio
async def test_basic_strategy_execute_signals_invalid_symbol(basic_strategy, mock_connectors):
    """Test wykonywania sygnałów dla niedozwolonego symbolu."""
    signals = {
        'symbol': 'INVALID',
        'action': 'BUY',
        'entry_price': 1.1,
        'stop_loss': 1.09,
        'take_profit': 1.12,
        'volume': 0.1
    }
    
    result = await basic_strategy.execute_signals(signals)
    assert result is None


@pytest.mark.asyncio
async def test_basic_strategy_generate_signals_sell(basic_strategy):
    """Test generowania sygnału SELL."""
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1
        },
        'ollama_analysis': 'RECOMMENDATION: SELL\nStrength: High',
        'claude_analysis': 'Rekomendacja: short\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips'
    }
    
    signals = await basic_strategy.generate_signals(analysis)
    
    assert signals['action'] == 'SELL'
    assert signals['symbol'] == 'EURUSD'
    assert signals['entry_price'] == 1.1
    assert signals['stop_loss'] > signals['entry_price']
    assert signals['take_profit'] < signals['entry_price']


@pytest.mark.asyncio
async def test_basic_strategy_generate_signals_parsing_error(basic_strategy):
    """Test obsługi błędu podczas parsowania wyników analizy."""
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1
        },
        'ollama_analysis': 'Invalid format',
        'claude_analysis': 'Also invalid format'
    }
    
    with pytest.raises(Exception):
        await basic_strategy.generate_signals(analysis)


@pytest.mark.asyncio
async def test_basic_strategy_execute_signals_error(basic_strategy, mock_connectors):
    """Test obsługi błędu podczas wykonywania zlecenia."""
    mt5_mock, _, _, _ = mock_connectors
    mt5_mock.place_order.side_effect = Exception("Test error")
    
    signals = {
        'symbol': 'EURUSD',
        'action': 'BUY',
        'entry_price': 1.1,
        'stop_loss': 1.09,
        'take_profit': 1.12,
        'volume': 0.1
    }
    
    result = await basic_strategy.execute_signals(signals)
    assert result is None


@pytest.mark.asyncio
async def test_basic_strategy_generate_signals_jpy(basic_strategy):
    """Test generowania sygnałów dla par z JPY."""
    analysis = {
        'market_data': {
            'symbol': 'USDJPY',
            'current_price': 110.0
        },
        'ollama_analysis': 'RECOMMENDATION: BUY\nStrength: High',
        'claude_analysis': 'Rekomendacja: long\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips'
    }
    
    signals = await basic_strategy.generate_signals(analysis)
    
    assert signals['action'] == 'BUY'
    assert signals['symbol'] == 'USDJPY'
    assert signals['entry_price'] == 110.0
    # Dla JPY pip to 0.01
    assert signals['stop_loss'] == 110.0 - (20 * 0.01)
    assert signals['take_profit'] == 110.0 + (60 * 0.01)


@pytest.mark.asyncio
async def test_basic_strategy_generate_signals_empty_analysis(basic_strategy):
    """Test generowania sygnałów dla pustych wyników analizy."""
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1
        },
        'ollama_analysis': '',  # Pusty string
        'claude_analysis': '\n'  # Tylko znak nowej linii
    }
    
    with pytest.raises(ValueError, match="Nie udało się sparsować wyników analizy AI"):
        await basic_strategy.generate_signals(analysis) 