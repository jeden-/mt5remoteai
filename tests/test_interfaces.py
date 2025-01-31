"""
Moduł zawierający testy dla interfejsów konektorów.
"""
import pytest
from typing import Dict, Any, List
from src.interfaces.connectors import IMT5Connector, IOllamaConnector, IAnthropicConnector, IDBHandler


def test_imt5_connector_interface():
    """Test interfejsu IMT5Connector."""
    # Test tworzenia abstrakcyjnej klasy
    with pytest.raises(TypeError):
        IMT5Connector()
    
    # Test metody get_rates
    assert hasattr(IMT5Connector, 'get_rates')
    assert IMT5Connector.get_rates.__isabstractmethod__
    
    # Test metody get_account_info
    assert hasattr(IMT5Connector, 'get_account_info')
    assert IMT5Connector.get_account_info.__isabstractmethod__
    
    # Test metody place_order
    assert hasattr(IMT5Connector, 'place_order')
    assert IMT5Connector.place_order.__isabstractmethod__
    
    # Test implementacji
    class TestImpl(IMT5Connector):
        def get_rates(self, symbol: str, timeframe: str, count: int) -> List[Dict[str, Any]]:
            super().get_rates(symbol, timeframe, count)  # Wywołanie metody abstrakcyjnej
            return []
        
        def get_account_info(self) -> Dict[str, Any]:
            super().get_account_info()  # Wywołanie metody abstrakcyjnej
            return {}
        
        def place_order(self, **kwargs) -> Dict[str, Any]:
            super().place_order(**kwargs)  # Wywołanie metody abstrakcyjnej
            return {}
    
    impl = TestImpl()
    assert isinstance(impl, IMT5Connector)


def test_iollama_connector_interface():
    """Test interfejsu IOllamaConnector."""
    # Test tworzenia abstrakcyjnej klasy
    with pytest.raises(TypeError):
        IOllamaConnector()
    
    # Test metody analyze_market_data
    assert hasattr(IOllamaConnector, 'analyze_market_data')
    assert IOllamaConnector.analyze_market_data.__isabstractmethod__
    
    # Test implementacji
    class TestImpl(IOllamaConnector):
        async def analyze_market_data(self, market_data: Dict[str, Any], prompt_template: str) -> str:
            await super().analyze_market_data(market_data, prompt_template)  # Wywołanie metody abstrakcyjnej
            return ""
    
    impl = TestImpl()
    assert isinstance(impl, IOllamaConnector)


def test_ianthropicconnector_interface():
    """Test interfejsu IAnthropicConnector."""
    # Test tworzenia abstrakcyjnej klasy
    with pytest.raises(TypeError):
        IAnthropicConnector()
    
    # Test metody analyze_market_conditions
    assert hasattr(IAnthropicConnector, 'analyze_market_conditions')
    assert IAnthropicConnector.analyze_market_conditions.__isabstractmethod__
    
    # Test implementacji
    class TestImpl(IAnthropicConnector):
        async def analyze_market_conditions(self, market_data: Dict[str, Any], prompt_template: str) -> str:
            await super().analyze_market_conditions(market_data, prompt_template)  # Wywołanie metody abstrakcyjnej
            return ""
    
    impl = TestImpl()
    assert isinstance(impl, IAnthropicConnector)


def test_idbhandler_interface():
    """Test interfejsu IDBHandler."""
    # Test tworzenia abstrakcyjnej klasy
    with pytest.raises(TypeError):
        IDBHandler()
    
    # Test metody save_trade
    assert hasattr(IDBHandler, 'save_trade')
    assert IDBHandler.save_trade.__isabstractmethod__
    
    # Test implementacji
    class TestImpl(IDBHandler):
        def save_trade(self, trade_info: Dict[str, Any]) -> None:
            super().save_trade(trade_info)  # Wywołanie metody abstrakcyjnej
            pass
    
    impl = TestImpl()
    assert isinstance(impl, IDBHandler)


def test_mt5_connector_implementation():
    """Test implementacji MT5Connector."""
    class TestMT5ConnectorImpl(IMT5Connector):
        def get_rates(self, symbol: str, timeframe: str, count: int) -> List[Dict[str, Any]]:
            if not isinstance(symbol, str) or not isinstance(timeframe, str) or not isinstance(count, int):
                raise TypeError("Nieprawidłowe typy parametrów")
            if not symbol or not timeframe or count <= 0:
                raise ValueError("Nieprawidłowe parametry")
            return [{"time": 1234567890, "open": 1.1234, "high": 1.1235, "low": 1.1233, "close": 1.1234}]
        
        def get_account_info(self) -> Dict[str, Any]:
            return {"balance": 10000.0, "equity": 10000.0, "margin": 0.0}
        
        def place_order(self, **kwargs) -> Dict[str, Any]:
            required_fields = ["symbol", "volume", "type"]
            if not all(field in kwargs for field in required_fields):
                raise ValueError("Brak wymaganych parametrów zlecenia")
            if not isinstance(kwargs["symbol"], str) or not isinstance(kwargs["volume"], (int, float)):
                raise TypeError("Nieprawidłowe typy parametrów zlecenia")
            return {"ticket": 123, "volume": kwargs["volume"], "price": 1.1234}
    
    connector = TestMT5ConnectorImpl()
    
    # Test get_rates
    rates = connector.get_rates("EURUSD", "M1", 100)
    assert isinstance(rates, list)
    assert len(rates) > 0
    assert all(isinstance(rate, dict) for rate in rates)
    assert all(key in rates[0] for key in ["time", "open", "high", "low", "close"])
    
    # Test get_rates - błędne wartości
    with pytest.raises(ValueError):
        connector.get_rates("", "M1", 100)
    with pytest.raises(ValueError):
        connector.get_rates("EURUSD", "", 100)
    with pytest.raises(ValueError):
        connector.get_rates("EURUSD", "M1", 0)
    
    # Test get_rates - błędne typy
    with pytest.raises(TypeError):
        connector.get_rates(123, "M1", 100)
    with pytest.raises(TypeError):
        connector.get_rates("EURUSD", 123, 100)
    with pytest.raises(TypeError):
        connector.get_rates("EURUSD", "M1", "100")
    
    # Test get_account_info
    account_info = connector.get_account_info()
    assert isinstance(account_info, dict)
    assert all(key in account_info for key in ["balance", "equity", "margin"])
    assert all(isinstance(value, float) for value in account_info.values())
    
    # Test place_order
    order = connector.place_order(symbol="EURUSD", volume=0.1, type="BUY")
    assert isinstance(order, dict)
    assert all(key in order for key in ["ticket", "volume", "price"])
    
    # Test place_order - błędne parametry
    with pytest.raises(ValueError):
        connector.place_order(symbol="EURUSD")
    with pytest.raises(ValueError):
        connector.place_order(volume=0.1, type="BUY")
    
    # Test place_order - błędne typy
    with pytest.raises(TypeError):
        connector.place_order(symbol=123, volume=0.1, type="BUY")
    with pytest.raises(TypeError):
        connector.place_order(symbol="EURUSD", volume="0.1", type="BUY")


@pytest.mark.asyncio
async def test_ollama_connector_implementation():
    """Test implementacji OllamaConnector."""
    class TestOllamaConnectorImpl(IOllamaConnector):
        async def analyze_market_data(self, market_data: Dict[str, Any], prompt_template: str) -> str:
            if not isinstance(market_data, dict) or not isinstance(prompt_template, str):
                raise TypeError("Nieprawidłowe typy danych")
            if not market_data or not prompt_template:
                raise ValueError("Brak wymaganych danych")
            return "Analiza rynku: Trend wzrostowy"
    
    connector = TestOllamaConnectorImpl()
    
    # Test analyze_market_data
    market_data = {
        "symbol": "EURUSD",
        "price": 1.1234,
        "indicators": {"sma": 1.1230, "rsi": 65}
    }
    prompt = "Przeanalizuj {symbol} przy cenie {price}"
    result = await connector.analyze_market_data(market_data, prompt)
    assert isinstance(result, str)
    assert "Analiza rynku" in result
    
    # Test analyze_market_data - błędne dane
    with pytest.raises(ValueError):
        await connector.analyze_market_data({}, prompt)
    with pytest.raises(ValueError):
        await connector.analyze_market_data(market_data, "")
    
    # Test analyze_market_data - błędne typy
    with pytest.raises(TypeError):
        await connector.analyze_market_data([], prompt)
    with pytest.raises(TypeError):
        await connector.analyze_market_data(market_data, 123)
    with pytest.raises(TypeError):
        await connector.analyze_market_data(None, prompt)


@pytest.mark.asyncio
async def test_anthropic_connector_implementation():
    """Test implementacji AnthropicConnector."""
    class TestAnthropicConnectorImpl(IAnthropicConnector):
        async def analyze_market_conditions(self, market_data: Dict[str, Any], prompt_template: str) -> str:
            if not isinstance(market_data, dict) or not isinstance(prompt_template, str):
                raise TypeError("Nieprawidłowe typy danych")
            if not market_data or not prompt_template:
                raise ValueError("Brak wymaganych danych")
            return "Analiza warunków: Silny sygnał kupna"
    
    connector = TestAnthropicConnectorImpl()
    
    # Test analyze_market_conditions
    market_data = {
        "symbol": "EURUSD",
        "price": 1.1234,
        "volume": 1000000,
        "sentiment": "bullish"
    }
    prompt = "Oceń warunki dla {symbol}"
    result = await connector.analyze_market_conditions(market_data, prompt)
    assert isinstance(result, str)
    assert "Analiza warunków" in result
    
    # Test analyze_market_conditions - błędne dane
    with pytest.raises(ValueError):
        await connector.analyze_market_conditions({}, prompt)
    with pytest.raises(ValueError):
        await connector.analyze_market_conditions(market_data, "")
    
    # Test analyze_market_conditions - błędne typy
    with pytest.raises(TypeError):
        await connector.analyze_market_conditions([], prompt)
    with pytest.raises(TypeError):
        await connector.analyze_market_conditions(market_data, 123)
    with pytest.raises(TypeError):
        await connector.analyze_market_conditions(None, prompt)


def test_db_handler_implementation():
    """Test implementacji DBHandler."""
    class TestDBHandlerImpl(IDBHandler):
        def __init__(self):
            self.trades = []
        
        def save_trade(self, trade_info: Dict[str, Any]) -> None:
            if not isinstance(trade_info, dict):
                raise TypeError("trade_info musi być słownikiem")
            
            required_fields = ["ticket", "symbol", "type", "volume"]
            if not all(field in trade_info for field in required_fields):
                raise ValueError("Brak wymaganych pól w trade_info")
                
            self.trades.append(trade_info)
    
    handler = TestDBHandlerImpl()
    
    # Test save_trade
    trade_info = {
        "ticket": 123,
        "symbol": "EURUSD",
        "type": "BUY",
        "volume": 0.1,
        "price": 1.1234,
        "profit": 50.0
    }
    handler.save_trade(trade_info)
    assert len(handler.trades) == 1
    assert handler.trades[0] == trade_info
    
    # Test save_trade - błędne typy
    with pytest.raises(TypeError):
        handler.save_trade([])
    with pytest.raises(TypeError):
        handler.save_trade(None)
    with pytest.raises(TypeError):
        handler.save_trade(123)
    
    # Test save_trade - błędne dane
    with pytest.raises(ValueError):
        handler.save_trade({"ticket": 123})
    with pytest.raises(ValueError):
        handler.save_trade({"symbol": "EURUSD", "type": "BUY"})
    with pytest.raises(ValueError):
        handler.save_trade({}) 