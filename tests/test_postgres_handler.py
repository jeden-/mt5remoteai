"""
Moduł zawierający testy dla handlera PostgreSQL.
"""
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime
from src.database.postgres_handler import PostgresHandler, retry_on_error
import asyncpg
from typing import Dict, Any


@pytest_asyncio.fixture
async def mock_asyncpg():
    """
    Fixture tworzący zamockowany moduł asyncpg.
    
    Returns:
        Mock: Zamockowany moduł asyncpg
    """
    # Mock dla połączenia
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock()
    mock_conn.executemany = AsyncMock()
    mock_conn.fetch = AsyncMock()
    mock_conn.fetchrow = AsyncMock()
    mock_conn.fetchval = AsyncMock()
    
    # Mock dla transakcji
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_transaction.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_transaction)
    
    # Mock dla puli połączeń
    mock_pool = MagicMock()
    mock_pool_context = MagicMock()
    mock_pool_context.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool_context.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire = MagicMock(return_value=mock_pool_context)
    mock_pool.close = AsyncMock()
    
    # Mock dla create_pool
    mock_create_pool = AsyncMock(return_value=mock_pool)
    
    with patch('asyncpg.create_pool', mock_create_pool):
        yield {
            'create_pool': mock_create_pool,
            'pool': mock_pool,
            'connection': mock_conn
        }


@pytest.fixture
def config() -> Dict[str, Any]:
    """
    Fixture tworzący obiekt konfiguracyjny.
    
    Returns:
        Dict[str, Any]: Słownik z konfiguracją
    """
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "mt5remotetest",
            "user": "mt5remote",
            "password": "mt5remote",
            "pool": {
                "min_size": 5,
                "max_size": 20,
                "command_timeout": 60,
                "connect_timeout": 30
            }
        }
    }


@pytest_asyncio.fixture
async def postgres_handler(config):
    """
    Fixture tworzący handler PostgreSQL.
    
    Args:
        config: Słownik z konfiguracją
        
    Returns:
        PostgresHandler: Skonfigurowany handler
    """
    handler = PostgresHandler(config)
    yield handler
    await handler.close_pool()


@pytest.mark.asyncio
async def test_create_pool_success(postgres_handler, mock_asyncpg):
    """Test udanego tworzenia puli połączeń."""
    await postgres_handler.create_pool()
    assert postgres_handler.pool is not None


@pytest.mark.asyncio
async def test_create_pool_failure(postgres_handler, mock_asyncpg):
    """Test nieudanego tworzenia puli połączeń."""
    mock_asyncpg['create_pool'].side_effect = asyncpg.PostgresError("Connection error")
    
    with pytest.raises(asyncpg.PostgresError):
        await postgres_handler.create_pool()
    assert postgres_handler.pool is None


@pytest.mark.asyncio
async def test_close_pool(postgres_handler, mock_asyncpg):
    """Test zamykania puli połączeń."""
    await postgres_handler.create_pool()
    assert postgres_handler.pool is not None
    
    await postgres_handler.close_pool()
    assert postgres_handler.pool is None


@pytest.mark.asyncio
async def test_create_tables_success(postgres_handler, mock_asyncpg):
    """Test tworzenia tabel w bazie danych."""
    await postgres_handler.create_pool()
    await postgres_handler.create_tables()


@pytest.mark.asyncio
async def test_create_tables_no_pool(postgres_handler):
    """Test tworzenia tabel bez puli połączeń."""
    with pytest.raises(RuntimeError):
        await postgres_handler.create_tables()


@pytest.mark.asyncio
async def test_save_market_data_success(postgres_handler, mock_asyncpg):
    """Test zapisywania danych rynkowych."""
    await postgres_handler.create_pool()

    market_data = {
        "symbol": "EURUSD",
        "timestamp": datetime.now(),
        "open": 1.1000,
        "high": 1.1100,
        "low": 1.0900,
        "close": 1.1050,
        "volume": 1000.0
    }

    result = await postgres_handler.save_market_data(market_data)
    assert result is True


@pytest.mark.asyncio
async def test_save_market_data_no_pool(postgres_handler):
    """Test zapisywania danych rynkowych bez puli połączeń."""
    market_data = {
        "symbol": "EURUSD",
        "timestamp": datetime.now(),
        "open": 1.1000,
        "high": 1.1100,
        "low": 1.0900,
        "close": 1.1050,
        "volume": 1000.0
    }

    result = await postgres_handler.save_market_data(market_data)
    assert result is False


@pytest.mark.asyncio
async def test_save_market_data_error(postgres_handler, mock_asyncpg):
    """Test błędu podczas zapisywania danych rynkowych."""
    await postgres_handler.create_pool()
    mock_asyncpg['connection'].execute.side_effect = asyncpg.PostgresError("Insert error")

    market_data = {
        "symbol": "EURUSD",
        "timestamp": datetime.now(),
        "open": 1.1000,
        "high": 1.1100,
        "low": 1.0900,
        "close": 1.1050,
        "volume": 1000.0
    }

    result = await postgres_handler.save_market_data(market_data)
    assert result is False


@pytest.mark.asyncio
async def test_execute_success(postgres_handler, mock_asyncpg):
    """Test udanego wykonania zapytania."""
    await postgres_handler.create_pool()

    query = "SELECT * FROM market_data"
    await postgres_handler.execute(query)


@pytest.mark.asyncio
async def test_execute_many_success(postgres_handler, mock_asyncpg):
    """Test udanego wykonania wielu zapytań."""
    await postgres_handler.create_pool()

    query = "INSERT INTO market_data (symbol, timestamp) VALUES ($1, $2)"
    args = [(1, datetime.now()), (2, datetime.now())]

    await postgres_handler.execute_many(query, args)


@pytest.mark.asyncio
async def test_fetch_all_success(postgres_handler, mock_asyncpg):
    """Test pobierania wszystkich wyników."""
    await postgres_handler.create_pool()

    query = "SELECT * FROM market_data"
    expected_result = [{'id': 1}, {'id': 2}]
    mock_asyncpg['connection'].fetch.return_value = expected_result

    result = await postgres_handler.fetch_all(query)
    assert result == expected_result


@pytest.mark.asyncio
async def test_fetch_one_success(postgres_handler, mock_asyncpg):
    """Test pobierania pojedynczego wyniku."""
    await postgres_handler.create_pool()

    query = "SELECT * FROM market_data LIMIT 1"
    expected_result = {'id': 1}
    mock_asyncpg['connection'].fetchrow.return_value = expected_result

    result = await postgres_handler.fetch_one(query)
    assert result == expected_result


@pytest.mark.asyncio
async def test_fetch_val_success(postgres_handler, mock_asyncpg):
    """Test pobierania pojedynczej wartości."""
    await postgres_handler.create_pool()

    query = "SELECT count(*) FROM market_data"
    expected_result = 42
    mock_asyncpg['connection'].fetchval.return_value = expected_result

    result = await postgres_handler.fetch_val(query)
    assert result == expected_result


@pytest.mark.asyncio
async def test_retry_mechanism(postgres_handler, mock_asyncpg):
    """Test mechanizmu ponawiania prób."""
    await postgres_handler.create_pool()

    # Symuluj błąd połączenia
    mock_asyncpg['connection'].execute.side_effect = [
        asyncpg.PostgresError("Connection lost"),
        None  # Drugie wywołanie się powiedzie
    ]

    query = "SELECT 1"
    await postgres_handler.execute(query)


@pytest.mark.asyncio
async def test_timeout_handling(postgres_handler, mock_asyncpg):
    """Test obsługi timeout."""
    await postgres_handler.create_pool()

    mock_asyncpg['connection'].execute.side_effect = asyncio.TimeoutError()

    with pytest.raises(asyncio.TimeoutError):
        await postgres_handler.execute("SELECT 1")


@pytest.mark.asyncio
async def test_fetch_all_error(postgres_handler, mock_asyncpg):
    """Test błędu podczas pobierania wszystkich wyników."""
    await postgres_handler.create_pool()
    mock_asyncpg['connection'].fetch.side_effect = asyncpg.PostgresError("Fetch error")

    with pytest.raises(asyncpg.PostgresError):
        await postgres_handler.fetch_all("SELECT 1")


@pytest.mark.asyncio
async def test_fetch_one_error(postgres_handler, mock_asyncpg):
    """Test błędu podczas pobierania pojedynczego wyniku."""
    await postgres_handler.create_pool()
    mock_asyncpg['connection'].fetchrow.side_effect = asyncpg.PostgresError("Fetch error")

    with pytest.raises(asyncpg.PostgresError):
        await postgres_handler.fetch_one("SELECT 1")


@pytest.mark.asyncio
async def test_fetch_val_error(postgres_handler, mock_asyncpg):
    """Test błędu podczas pobierania pojedynczej wartości."""
    await postgres_handler.create_pool()
    mock_asyncpg['connection'].fetchval.side_effect = asyncpg.PostgresError("Fetch error")

    with pytest.raises(asyncpg.PostgresError):
        await postgres_handler.fetch_val("SELECT 1")


@pytest.mark.asyncio
async def test_execute_error(postgres_handler, mock_asyncpg):
    """Test błędu podczas wykonywania zapytania."""
    await postgres_handler.create_pool()
    mock_asyncpg['connection'].execute.side_effect = asyncpg.PostgresError("Execute error")

    with pytest.raises(asyncpg.PostgresError):
        await postgres_handler.execute("SELECT 1")


@pytest.mark.asyncio
async def test_execute_many_error(postgres_handler, mock_asyncpg):
    """Test błędu podczas wykonywania wielu zapytań."""
    await postgres_handler.create_pool()
    mock_asyncpg['connection'].executemany.side_effect = asyncpg.PostgresError("Execute error")

    with pytest.raises(asyncpg.PostgresError):
        await postgres_handler.execute_many("INSERT INTO test VALUES ($1)", [(1,), (2,)])


@pytest.mark.asyncio
async def test_retry_mechanism_exhausted(postgres_handler, mock_asyncpg):
    """Test wyczerpania wszystkich prób w mechanizmie ponowień."""
    await postgres_handler.create_pool()

    # Symuluj ciągłe błędy połączenia
    error = asyncpg.PostgresError("Connection lost")
    mock_asyncpg['connection'].execute.side_effect = [error, error, error]

    with pytest.raises(asyncpg.PostgresError) as exc_info:
        await postgres_handler.execute("SELECT 1")
    assert "Connection lost" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retry_mechanism_mixed_errors(postgres_handler, mock_asyncpg):
    """Test mechanizmu ponowień dla różnych typów błędów."""
    await postgres_handler.create_pool()

    # Symuluj różne typy błędów
    mock_asyncpg['connection'].execute.side_effect = [
        asyncpg.PostgresError("Connection lost"),  # Pierwszy błąd
        asyncio.TimeoutError(),  # Drugi błąd
        asyncpg.PostgresError("Query error")  # Trzeci błąd
    ]

    with pytest.raises(asyncpg.PostgresError) as exc_info:
        await postgres_handler.execute("SELECT 1")
    assert "Query error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retry_mechanism_custom_params():
    """Test mechanizmu ponowień z niestandardowymi parametrami."""
    # Utwórz funkcję testową
    mock_func = AsyncMock()
    mock_func.side_effect = [
        asyncpg.PostgresError("Error 1"),
        asyncpg.PostgresError("Error 2"),
        asyncpg.PostgresError("Error 3"),
        asyncpg.PostgresError("Error 4"),
        asyncpg.PostgresError("Error 5")
    ]

    # Udekoruj funkcję testową
    @retry_on_error(retries=5, delay=0.1)
    async def test_func():
        await mock_func()

    # Wywołaj funkcję i sprawdź, czy wszystkie próby zostały wykorzystane
    with pytest.raises(asyncpg.PostgresError) as exc_info:
        await test_func()

    assert mock_func.call_count == 5
    assert "Error 5" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retry_mechanism_success_after_errors():
    """Test mechanizmu ponowień z sukcesem po kilku błędach."""
    # Utwórz funkcję testową
    mock_func = AsyncMock()
    mock_func.side_effect = [
        asyncpg.PostgresError("Error 1"),
        asyncio.TimeoutError(),
        None  # Sukces po dwóch błędach
    ]

    # Udekoruj funkcję testową
    @retry_on_error(retries=3, delay=0.1)
    async def test_func():
        await mock_func()

    # Wywołaj funkcję i sprawdź, czy zakończyła się sukcesem
    await test_func()
    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_retry_mechanism_all_methods(postgres_handler, mock_asyncpg):
    """Test mechanizmu ponowień dla wszystkich metod."""
    await postgres_handler.create_pool()

    # Symuluj błędy dla różnych metod
    mock_asyncpg['connection'].fetch.side_effect = [
        asyncpg.PostgresError("Error 1"),
        [{'id': 1}]  # Sukces po jednym błędzie
    ]
    mock_asyncpg['connection'].fetchrow.side_effect = [
        asyncpg.PostgresError("Error 1"),
        {'id': 1}  # Sukces po jednym błędzie
    ]
    mock_asyncpg['connection'].fetchval.side_effect = [
        asyncpg.PostgresError("Error 1"),
        42  # Sukces po jednym błędzie
    ]
    mock_asyncpg['connection'].execute.side_effect = [
        asyncpg.PostgresError("Error 1"),
        None  # Sukces po jednym błędzie
    ]
    mock_asyncpg['connection'].executemany.side_effect = [
        asyncpg.PostgresError("Error 1"),
        None  # Sukces po jednym błędzie
    ]

    # Testuj wszystkie metody
    result = await postgres_handler.fetch_all("SELECT 1")
    assert result == [{'id': 1}]

    result = await postgres_handler.fetch_one("SELECT 1")
    assert result == {'id': 1}

    result = await postgres_handler.fetch_val("SELECT 1")
    assert result == 42

    await postgres_handler.execute("SELECT 1")

    await postgres_handler.execute_many("INSERT INTO test VALUES ($1)", [(1,), (2,)])


@pytest.mark.asyncio
async def test_retry_mechanism_with_transaction(postgres_handler, mock_asyncpg):
    """Test mechanizmu ponowień z transakcją."""
    await postgres_handler.create_pool()

    # Symuluj błędy dla transakcji
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock(side_effect=[
        asyncpg.PostgresError("Transaction error"),
        mock_asyncpg['connection']  # Sukces po jednym błędzie
    ])
    mock_transaction.__aexit__ = AsyncMock(return_value=None)
    mock_asyncpg['connection'].transaction.return_value = mock_transaction

    # Testuj wykonanie zapytania w transakcji
    await postgres_handler.execute("SELECT 1")

    # Sprawdź, czy transakcja została utworzona dwukrotnie
    assert mock_asyncpg['connection'].transaction.call_count == 2 