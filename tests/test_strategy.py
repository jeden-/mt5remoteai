"""
Moduł zawierający testy dla strategii tradingowych.
"""
import pytest
import asyncio
from src.strategies.basic_strategy import BasicStrategy
from src.interfaces.connectors import IMT5Connector, IOllamaConnector, IAnthropicConnector, IDBHandler
from typing import Dict, Any, List


class MockMT5Connector(IMT5Connector):
    """Mock dla konektora MT5."""
    
    def get_rates(self, symbol: str, timeframe: str, count: int) -> List[Dict[str, Any]]:
        """Zwraca przykładowe dane historyczne."""
        return [
            {'close': 1.1000, 'volume': 100} for _ in range(count)
        ]
    
    def get_account_info(self) -> Dict[str, Any]:
        """Zwraca przykładowe informacje o koncie."""
        return {
            'balance': 10000,
            'equity': 10000,
            'margin': 0,
            'margin_level': 0,
            'profit': 0
        }
    
    def place_order(self, **kwargs) -> Dict[str, Any]:
        """Symuluje składanie zlecenia."""
        return {
            'ticket': 12345,
            'symbol': kwargs['symbol'],
            'type': kwargs['order_type'],
            'volume': kwargs['volume'],
            'price': kwargs['price'],
            'sl': kwargs['sl'],
            'tp': kwargs['tp']
        }


class MockOllamaConnector(IOllamaConnector):
    """Mock dla konektora Ollama."""
    
    async def analyze_market_data(self, market_data: Dict[str, Any], prompt_template: str) -> str:
        """Zwraca przykładową analizę rynku."""
        return "TREND_DIRECTION: UP\nTREND_STRENGTH: 8\nRECOMMENDATION: BUY"


class MockAnthropicConnector(IAnthropicConnector):
    """Mock dla konektora Anthropic."""
    
    async def analyze_market_conditions(self, market_data: Dict[str, Any], prompt_template: str) -> str:
        """Zwraca przykładową analizę rynku."""
        return "Rekomendacja: long\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips"


class MockDBHandler(IDBHandler):
    """Mock dla handlera bazy danych."""
    
    def save_trade(self, trade_info: Dict[str, Any]) -> None:
        """Symuluje zapisywanie transakcji."""
        pass


@pytest.fixture
async def setup_strategy():
    """
    Fixture przygotowujący strategię do testów.
    
    Returns:
        BasicStrategy: Skonfigurowana strategia z mockami
    """
    # Przygotuj mocki
    mt5_connector = MockMT5Connector()
    ollama_connector = MockOllamaConnector()
    anthropic_connector = MockAnthropicConnector()
    db_handler = MockDBHandler()
    
    # Konfiguracja strategii
    strategy_config = {
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'allowed_symbols': ['EURUSD']
    }
    
    # Utwórz i zwróć strategię
    strategy = BasicStrategy(
        mt5_connector=mt5_connector,
        ollama_connector=ollama_connector,
        anthropic_connector=anthropic_connector,
        db_handler=db_handler,
        config=strategy_config
    )
    
    return strategy


@pytest.mark.asyncio
async def test_market_analysis(setup_strategy):
    """Test analizy rynku."""
    strategy = await setup_strategy
    analysis = await strategy.analyze_market('EURUSD')
    
    assert 'market_data' in analysis
    assert 'technical_indicators' in analysis
    assert 'ollama_analysis' in analysis
    assert 'claude_analysis' in analysis


@pytest.mark.asyncio
async def test_signal_generation(setup_strategy):
    """Test generowania sygnałów."""
    strategy = await setup_strategy
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1000,
            'sma_20': 1.0990,
            'sma_50': 1.0980,
            'price_change_24h': 0.5,
            'volume_24h': 10000
        },
        'technical_indicators': {
            'sma_20': 1.0990,
            'sma_50': 1.0980
        },
        'ollama_analysis': 'TREND_DIRECTION: UP\nTREND_STRENGTH: 8\nRECOMMENDATION: BUY',
        'claude_analysis': 'Rekomendacja: long\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips'
    }
    
    signals = await strategy.generate_signals(analysis)
    
    assert 'action' in signals
    assert 'entry_price' in signals
    assert 'stop_loss' in signals
    assert 'take_profit' in signals


@pytest.mark.asyncio
async def test_execute_signals(setup_strategy):
    """Test wykonywania sygnałów."""
    strategy = await setup_strategy
    signals = {
        'symbol': 'EURUSD',
        'action': 'BUY',
        'entry_price': 1.1000,
        'stop_loss': 1.0950,
        'take_profit': 1.1100,
        'volume': 0.1
    }
    
    result = await strategy.execute_signals(signals)
    
    assert result is not None
    assert result['ticket'] == 12345
    assert result['symbol'] == 'EURUSD'
    assert result['type'] == 'BUY' 