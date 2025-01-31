"""
Moduł zawierający testy dla konektora MT5.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.utils.config import Config
from src.connectors.mt5_connector import MT5Connector


@pytest.fixture
def config():
    """Fixture tworzący przykładową konfigurację."""
    return Config(
        MT5_LOGIN='12345',
        MT5_PASSWORD='password',
        MT5_SERVER='TestServer',
        ANTHROPIC_API_KEY='test_key',
        POSTGRES_USER='test_user',
        POSTGRES_PASSWORD='test_password'
    )


@pytest.fixture
def connector(config):
    """Fixture tworzący instancję konektora."""
    return MT5Connector(config)


def test_initialization(connector):
    """Test inicjalizacji konektora."""
    assert connector.config is not None
    assert connector.connected is False


@patch('MetaTrader5.initialize')
@patch('MetaTrader5.login')
def test_connect_success(mock_login, mock_initialize, connector):
    """Test udanego połączenia z MT5."""
    mock_initialize.return_value = True
    mock_login.return_value = True
    
    result = connector.connect()
    
    assert result is True
    assert connector.connected is True
    mock_initialize.assert_called_once()
    mock_login.assert_called_once_with(
        login=connector.config.MT5_LOGIN,
        password=connector.config.MT5_PASSWORD,
        server=connector.config.MT5_SERVER
    )


@patch('MetaTrader5.initialize')
def test_connect_initialize_failure(mock_initialize, connector):
    """Test nieudanej inicjalizacji MT5."""
    mock_initialize.return_value = False
    
    result = connector.connect()
    
    assert result is False
    assert connector.connected is False
    mock_initialize.assert_called_once()


@patch('MetaTrader5.initialize')
@patch('MetaTrader5.login')
def test_connect_login_failure(mock_login, mock_initialize, connector):
    """Test nieudanego logowania do MT5."""
    mock_initialize.return_value = True
    mock_login.return_value = False
    
    result = connector.connect()
    
    assert result is False
    assert connector.connected is False
    mock_initialize.assert_called_once()
    mock_login.assert_called_once()


@patch('MetaTrader5.shutdown')
def test_disconnect(mock_shutdown, connector):
    """Test rozłączania z MT5."""
    connector.connected = True
    
    connector.disconnect()
    
    assert connector.connected is False
    mock_shutdown.assert_called_once()


@patch('MetaTrader5.shutdown')
def test_disconnect_when_not_connected(mock_shutdown, connector):
    """Test rozłączania gdy nie było połączenia."""
    connector.connected = False
    
    connector.disconnect()
    
    assert connector.connected is False
    mock_shutdown.assert_not_called()


@patch('MetaTrader5.account_info')
def test_get_account_info_success(mock_account_info, connector):
    """Test pobierania informacji o koncie."""
    mock_info = MagicMock()
    mock_info.balance = 1000.0
    mock_info.equity = 1100.0
    mock_info.profit = 100.0
    mock_info.margin = 200.0
    mock_info.margin_level = 550.0
    mock_account_info.return_value = mock_info
    
    connector.connected = True
    result = connector.get_account_info()
    
    assert result['balance'] == 1000.0
    assert result['equity'] == 1100.0
    assert result['profit'] == 100.0
    assert result['margin'] == 200.0
    assert result['margin_level'] == 550.0
    mock_account_info.assert_called_once()


def test_get_account_info_not_connected(connector):
    """Test pobierania informacji o koncie bez połączenia."""
    connector.connected = False
    
    with pytest.raises(RuntimeError, match="Brak połączenia z MT5"):
        connector.get_account_info()


@patch('MetaTrader5.account_info')
def test_get_account_info_failure(mock_account_info, connector):
    """Test błędu podczas pobierania informacji o koncie."""
    mock_account_info.return_value = None
    connector.connected = True
    
    with pytest.raises(RuntimeError, match="Nie można pobrać informacji o koncie"):
        connector.get_account_info()


@patch('MetaTrader5.symbols_get')
def test_get_symbols_success(mock_symbols_get, connector):
    """Test pobierania listy symboli."""
    mock_symbol1 = MagicMock()
    mock_symbol1.name = 'EURUSD'
    mock_symbol2 = MagicMock()
    mock_symbol2.name = 'GBPUSD'
    mock_symbols_get.return_value = [mock_symbol1, mock_symbol2]
    
    connector.connected = True
    result = connector.get_symbols()
    
    assert result == ['EURUSD', 'GBPUSD']
    mock_symbols_get.assert_called_once()


def test_get_symbols_not_connected(connector):
    """Test pobierania listy symboli bez połączenia."""
    connector.connected = False
    
    with pytest.raises(RuntimeError, match="Brak połączenia z MT5"):
        connector.get_symbols()


@patch('MetaTrader5.symbols_get')
def test_get_symbols_failure(mock_symbols_get, connector):
    """Test błędu podczas pobierania listy symboli."""
    mock_symbols_get.return_value = None
    connector.connected = True
    
    with pytest.raises(RuntimeError, match="Nie można pobrać listy symboli"):
        connector.get_symbols() 