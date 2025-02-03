"""
Testy dla modułu MT5Connector.
"""
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.connectors.mt5_connector import MT5Connector
from src.utils.config import Config

# Przykładowe dane testowe
SAMPLE_ACCOUNT_INFO = Mock(
    balance=10000.0,
    equity=10500.0,
    profit=500.0,
    margin=1000.0,
    margin_level=1050.0
)

# Tworzenie mocków dla symboli z właściwością name
EURUSD_MOCK = Mock()
EURUSD_MOCK.name = "EURUSD"
GBPUSD_MOCK = Mock()
GBPUSD_MOCK.name = "GBPUSD"
USDJPY_MOCK = Mock()
USDJPY_MOCK.name = "USDJPY"

SAMPLE_SYMBOLS = [EURUSD_MOCK, GBPUSD_MOCK, USDJPY_MOCK]

@pytest.fixture
def config() -> Config:
    """Fixture dostarczający przykładową konfigurację."""
    return Config(
        MT5_LOGIN=12345,
        MT5_PASSWORD="test_password",
        MT5_SERVER="TestServer"
    )

@pytest.fixture
def mt5_connector(config: Config) -> MT5Connector:
    """Fixture dostarczający instancję MT5Connector."""
    return MT5Connector(config)

@pytest.mark.asyncio
async def test_connect_success(mt5_connector: MT5Connector) -> None:
    """Test udanego połączenia z MT5."""
    with patch("MetaTrader5.initialize", return_value=True), \
         patch("MetaTrader5.login", return_value=True):
        assert mt5_connector.connect() is True
        assert mt5_connector.connected is True

@pytest.mark.asyncio
async def test_connect_initialize_failure(mt5_connector: MT5Connector) -> None:
    """Test nieudanej inicjalizacji MT5."""
    with patch("MetaTrader5.initialize", return_value=False):
        assert mt5_connector.connect() is False
        assert mt5_connector.connected is False

@pytest.mark.asyncio
async def test_connect_login_failure(mt5_connector: MT5Connector) -> None:
    """Test nieudanego logowania do MT5."""
    with patch("MetaTrader5.initialize", return_value=True), \
         patch("MetaTrader5.login", return_value=False):
        assert mt5_connector.connect() is False
        assert mt5_connector.connected is False

@pytest.mark.asyncio
async def test_disconnect(mt5_connector: MT5Connector) -> None:
    """Test rozłączania z MT5."""
    with patch("MetaTrader5.shutdown") as mock_shutdown:
        mt5_connector.connected = True
        mt5_connector.disconnect()
        assert mt5_connector.connected is False
        mock_shutdown.assert_called_once()

@pytest.mark.asyncio
async def test_get_account_info_success(mt5_connector: MT5Connector) -> None:
    """Test pobierania informacji o koncie."""
    with patch("MetaTrader5.account_info", return_value=SAMPLE_ACCOUNT_INFO):
        mt5_connector.connected = True
        info = mt5_connector.get_account_info()
        assert isinstance(info, dict)
        assert info["balance"] == 10000.0
        assert info["equity"] == 10500.0
        assert info["profit"] == 500.0
        assert info["margin"] == 1000.0
        assert info["margin_level"] == 1050.0

@pytest.mark.asyncio
async def test_get_account_info_not_connected(mt5_connector: MT5Connector) -> None:
    """Test pobierania informacji o koncie bez połączenia."""
    with pytest.raises(RuntimeError, match="Brak połączenia z MT5"):
        mt5_connector.get_account_info()

@pytest.mark.asyncio
async def test_get_account_info_failure(mt5_connector: MT5Connector) -> None:
    """Test nieudanego pobierania informacji o koncie."""
    with patch("MetaTrader5.account_info", return_value=None), \
         pytest.raises(RuntimeError, match="Nie można pobrać informacji o koncie"):
        mt5_connector.connected = True
        mt5_connector.get_account_info()

@pytest.mark.asyncio
async def test_get_symbols_success(mt5_connector: MT5Connector) -> None:
    """Test pobierania listy symboli."""
    with patch("MetaTrader5.symbols_get", return_value=SAMPLE_SYMBOLS):
        mt5_connector.connected = True
        symbols = mt5_connector.get_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) == 3
        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols
        assert "USDJPY" in symbols

@pytest.mark.asyncio
async def test_get_symbols_not_connected(mt5_connector: MT5Connector) -> None:
    """Test pobierania listy symboli bez połączenia."""
    with pytest.raises(RuntimeError, match="Brak połączenia z MT5"):
        mt5_connector.get_symbols()

@pytest.mark.asyncio
async def test_get_symbols_failure(mt5_connector: MT5Connector) -> None:
    """Test nieudanego pobierania listy symboli."""
    with patch("MetaTrader5.symbols_get", return_value=None), \
         pytest.raises(RuntimeError, match="Nie można pobrać listy symboli"):
        mt5_connector.connected = True
        mt5_connector.get_symbols() 