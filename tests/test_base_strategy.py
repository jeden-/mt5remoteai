"""
Testy dla modułu base_strategy.py
"""
import pytest
from unittest.mock import AsyncMock, Mock
from src.strategies.base_strategy import BaseStrategy
from src.interfaces.connectors import IMT5Connector, IOllamaConnector, IAnthropicConnector, IDBHandler


class TestStrategy(BaseStrategy):
    """Klasa testowa implementująca BaseStrategy."""
    
    async def analyze_market(self, symbol: str):
        """Implementacja testowa."""
        return {'status': 'analyzed', 'symbol': symbol}
        
    async def generate_signals(self, analysis):
        """Implementacja testowa."""
        return {'action': 'BUY', 'symbol': analysis['symbol']}
        
    async def execute_signals(self, signals):
        """Implementacja testowa."""
        return {'status': 'executed', 'action': signals['action']}


@pytest.fixture
def mock_connectors():
    """Fixture tworzący mocki konektorów."""
    return {
        'mt5': AsyncMock(spec=IMT5Connector),
        'ollama': AsyncMock(spec=IOllamaConnector),
        'claude': AsyncMock(spec=IAnthropicConnector),
        'db': Mock(spec=IDBHandler)
    }


@pytest.fixture
def test_strategy(mock_connectors):
    """Fixture tworzący instancję strategii testowej."""
    config = {
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'allowed_symbols': ['EURUSD', 'GBPUSD']
    }
    return TestStrategy(
        mt5_connector=mock_connectors['mt5'],
        ollama_connector=mock_connectors['ollama'],
        anthropic_connector=mock_connectors['claude'],
        db_handler=mock_connectors['db'],
        config=config
    )


def test_strategy_initialization(mock_connectors):
    """Test inicjalizacji strategii."""
    config = {
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'allowed_symbols': ['EURUSD', 'GBPUSD']
    }
    
    strategy = TestStrategy(
        mt5_connector=mock_connectors['mt5'],
        ollama_connector=mock_connectors['ollama'],
        anthropic_connector=mock_connectors['claude'],
        db_handler=mock_connectors['db'],
        config=config
    )
    
    assert strategy.mt5 == mock_connectors['mt5']
    assert strategy.ollama == mock_connectors['ollama']
    assert strategy.claude == mock_connectors['claude']
    assert strategy.db == mock_connectors['db']
    assert strategy.config == config
    assert strategy.max_position_size == config['max_position_size']
    assert strategy.max_risk_per_trade == config['max_risk_per_trade']
    assert strategy.allowed_symbols == config['allowed_symbols']


def test_strategy_initialization_default_values(mock_connectors):
    """Test inicjalizacji strategii z domyślnymi wartościami."""
    strategy = TestStrategy(
        mt5_connector=mock_connectors['mt5'],
        ollama_connector=mock_connectors['ollama'],
        anthropic_connector=mock_connectors['claude'],
        db_handler=mock_connectors['db'],
        config={}
    )
    
    assert strategy.max_position_size == 0.1
    assert strategy.max_risk_per_trade == 0.02
    assert strategy.allowed_symbols == ['EURUSD', 'GBPUSD', 'USDJPY']


@pytest.mark.asyncio
async def test_strategy_run_success(test_strategy):
    """Test pomyślnego wykonania pełnego cyklu strategii."""
    result = await test_strategy.run('EURUSD')
    
    assert result is not None
    assert result['status'] == 'executed'
    assert result['action'] == 'BUY'


@pytest.mark.asyncio
async def test_strategy_run_error_handling(test_strategy):
    """Test obsługi błędów w metodzie run."""
    # Symuluj błąd w analyze_market
    async def analyze_error(symbol):
        raise Exception("Analysis error")
    test_strategy.analyze_market = analyze_error
    
    result = await test_strategy.run('EURUSD')
    assert result is None


def test_abstract_methods():
    """Test czy metody abstrakcyjne są wymagane."""
    class IncompleteStrategy(BaseStrategy):
        """Klasa testowa bez implementacji metod abstrakcyjnych."""
        pass
    
    with pytest.raises(TypeError):
        IncompleteStrategy(
            mt5_connector=AsyncMock(),
            ollama_connector=AsyncMock(),
            anthropic_connector=AsyncMock(),
            db_handler=Mock(),
            config={}
        )


@pytest.mark.asyncio
async def test_strategy_run_with_none_values(test_strategy):
    """Test wykonania strategii gdy metody zwracają None."""
    # Nadpisz metody aby zwracały None
    async def return_none(*args, **kwargs):
        return None
    
    test_strategy.analyze_market = return_none
    result = await test_strategy.run('EURUSD')
    assert result is None


@pytest.mark.asyncio
async def test_strategy_run_with_invalid_symbol(test_strategy):
    """Test wykonania strategii z nieprawidłowym symbolem."""
    result = await test_strategy.run('INVALID')
    assert result is not None  # Strategia testowa nie sprawdza symbolu
    
    # Nadpisz metodę analyze_market aby sprawdzała symbol
    async def check_symbol(symbol):
        if symbol not in test_strategy.allowed_symbols:
            raise ValueError("Invalid symbol")
        return {'status': 'analyzed', 'symbol': symbol}
    
    test_strategy.analyze_market = check_symbol
    result = await test_strategy.run('INVALID')
    assert result is None 