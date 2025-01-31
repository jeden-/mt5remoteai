"""
Moduł zawierający testy dla handlera PostgreSQL.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.database.postgres_handler import PostgresHandler
from src.utils.config import Config


@pytest.fixture
def mock_psycopg2():
    """
    Fixture tworzący zamockowany moduł psycopg2.
    
    Returns:
        Mock: Zamockowany moduł psycopg2
    """
    with patch('psycopg2.connect') as mock_connect:
        # Mock dla połączenia
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Mock dla kursora
        mock_cursor = Mock()
        mock_cursor_cm = Mock()
        mock_cursor_cm.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor_cm.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor_cm
        
        yield {
            'connect': mock_connect,
            'connection': mock_conn,
            'cursor': mock_cursor
        }


@pytest.fixture
def config():
    """
    Fixture tworzący obiekt konfiguracyjny.
    
    Returns:
        Mock: Zamockowany obiekt Config
    """
    mock_config = Mock(spec=Config)
    mock_config.DB_HOST = "localhost"
    mock_config.DB_PORT = 5432
    mock_config.DB_NAME = "test_db"
    mock_config.DB_USER = "test_user"
    mock_config.DB_PASSWORD = "test_pass"
    return mock_config


@pytest.fixture
def postgres_handler(config):
    """
    Fixture tworzący handler PostgreSQL.
    
    Args:
        config: Zamockowany obiekt konfiguracyjny
        
    Returns:
        PostgresHandler: Skonfigurowany handler
    """
    return PostgresHandler(config)


def test_connect_success(postgres_handler, mock_psycopg2, config):
    """Test udanego połączenia z bazą danych."""
    # Wywołaj metodę
    postgres_handler.connect()
    
    # Sprawdź czy psycopg2.connect został wywołany z poprawnymi parametrami
    mock_psycopg2['connect'].assert_called_once_with(
        host=config.DB_HOST,
        port=config.DB_PORT,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD
    )
    
    # Sprawdź czy połączenie zostało zapisane
    assert postgres_handler.conn == mock_psycopg2['connection']


def test_connect_failure(postgres_handler, mock_psycopg2):
    """Test nieudanego połączenia z bazą danych."""
    # Skonfiguruj mock aby connect rzucał wyjątek
    mock_psycopg2['connect'].side_effect = Exception("Connection error")
    
    # Sprawdź czy metoda rzuca wyjątek
    with pytest.raises(Exception) as exc_info:
        postgres_handler.connect()
    
    assert "Connection error" in str(exc_info.value)
    assert postgres_handler.conn is None


def test_disconnect(postgres_handler, mock_psycopg2):
    """Test rozłączania z bazą danych."""
    # Najpierw połącz
    postgres_handler.connect()
    
    # Teraz rozłącz
    postgres_handler.disconnect()
    
    # Sprawdź czy połączenie zostało zamknięte
    mock_psycopg2['connection'].close.assert_called_once()
    assert postgres_handler.conn is None


def test_create_tables_success(postgres_handler, mock_psycopg2):
    """Test tworzenia tabel w bazie danych."""
    # Najpierw połącz
    postgres_handler.connect()
    
    # Utwórz tabele
    postgres_handler.create_tables()
    
    # Sprawdź czy wykonano odpowiednie zapytania SQL
    calls = mock_psycopg2['cursor'].execute.call_args_list
    assert len(calls) == 2  # Powinny być dwa wywołania execute
    
    # Sprawdź czy commit został wywołany
    mock_psycopg2['connection'].commit.assert_called_once()


def test_create_tables_not_connected(postgres_handler):
    """Test próby utworzenia tabel bez połączenia."""
    with pytest.raises(RuntimeError) as exc_info:
        postgres_handler.create_tables()
    
    assert "Brak połączenia z bazą danych" in str(exc_info.value)


def test_save_market_data_success(postgres_handler, mock_psycopg2):
    """Test zapisywania danych rynkowych."""
    # Najpierw połącz
    postgres_handler.connect()
    
    # Przygotuj przykładowe dane
    market_data = {
        "symbol": "EURUSD",
        "timestamp": datetime.now(),
        "open": 1.1000,
        "high": 1.1100,
        "low": 1.0900,
        "close": 1.1050,
        "volume": 1000.0
    }
    
    # Zapisz dane
    result = postgres_handler.save_market_data(market_data)
    
    # Sprawdź czy dane zostały zapisane poprawnie
    assert result is True
    mock_psycopg2['cursor'].execute.assert_called_once()
    mock_psycopg2['connection'].commit.assert_called_once()


def test_save_market_data_not_connected(postgres_handler):
    """Test próby zapisania danych bez połączenia."""
    result = postgres_handler.save_market_data({})
    assert result is False


def test_save_market_data_error(postgres_handler, mock_psycopg2):
    """Test błędu podczas zapisywania danych."""
    # Najpierw połącz
    postgres_handler.connect()
    
    # Skonfiguruj mock aby execute rzucał wyjątek
    mock_psycopg2['cursor'].execute.side_effect = Exception("Database error")
    
    # Próba zapisu danych
    result = postgres_handler.save_market_data({
        "symbol": "EURUSD",
        "timestamp": datetime.now(),
        "open": 1.1000,
        "high": 1.1100,
        "low": 1.0900,
        "close": 1.1050,
        "volume": 1000.0
    })
    
    # Sprawdź wyniki
    assert result is False
    mock_psycopg2['connection'].rollback.assert_called_once() 