"""
Moduł zawierający testy integracyjne dla bazy danych.
"""
import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
import asyncpg
from src.database.postgres_handler import PostgresHandler


# Konfiguracja event_loop dla wszystkich testów w module
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture
async def clean_test_db(test_db_handler):
    """Fixture czyszczący bazę danych przed i po testach."""
    async with test_db_handler.pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("TRUNCATE market_data, trades, historical_data RESTART IDENTITY CASCADE")
    yield
    async with test_db_handler.pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("TRUNCATE market_data, trades, historical_data RESTART IDENTITY CASCADE")


@pytest.mark.asyncio
async def test_db_connection(test_db_handler):
    """Test połączenia z bazą danych."""
    # Sprawdź czy handler ma aktywną pulę połączeń
    assert test_db_handler.pool is not None
    
    # Spróbuj wykonać proste zapytanie
    result = await test_db_handler.fetch_val("SELECT 1")
    assert result == 1


@pytest.mark.asyncio
async def test_create_tables(test_db_handler):
    """Test tworzenia tabel w bazie danych."""
    # Utwórz tabele
    await test_db_handler.create_tables()
    
    # Sprawdź czy tabele istnieją
    async with test_db_handler.pool.acquire() as conn:
        result = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'market_data'
            )
        """)
        assert result is True


@pytest.mark.asyncio
async def test_market_data_operations(test_db_handler, clean_test_db):
    """Test operacji na danych rynkowych."""
    # Przygotuj dane testowe
    test_data = {
        "symbol": "EURUSD",
        "timestamp": datetime.now(),
        "open": 1.1000,
        "high": 1.1100,
        "low": 1.0900,
        "close": 1.1050,
        "volume": 1000.0
    }
    
    async with test_db_handler.pool.acquire() as conn:
        async with conn.transaction():
            # Zapisz dane
            await conn.execute("""
                INSERT INTO market_data (
                    symbol, timestamp, open, high, low, close, volume
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7
                )
            """, test_data["symbol"], test_data["timestamp"], test_data["open"], 
                test_data["high"], test_data["low"], test_data["close"], test_data["volume"])
            
            # Pobierz dane
            result = await conn.fetch("""
                SELECT * FROM market_data WHERE symbol = $1
            """, test_data["symbol"])
    
    # Sprawdź czy dane są poprawne
    assert len(result) == 1
    assert result[0]["symbol"] == test_data['symbol']
    assert result[0]["open"] == test_data['open']
    assert result[0]["high"] == test_data['high']
    assert result[0]["low"] == test_data['low']
    assert result[0]["close"] == test_data['close']
    assert result[0]["volume"] == test_data['volume']


@pytest.mark.asyncio
async def test_trade_operations(test_db_handler, clean_test_db):
    """Test operacji na transakcjach."""
    # Dodaj transakcję
    await test_db_handler.execute("""
        INSERT INTO trades (
            symbol, order_type, volume, price, sl, tp,
            open_time, status
        ) VALUES (
            'EURUSD', 'BUY', 0.1, 1.1000, 1.0950, 1.1100,
            $1, 'OPEN'
        )
    """, datetime.now())
    
    # Sprawdź czy transakcja została dodana
    result = await test_db_handler.fetch_all("""
        SELECT * FROM trades WHERE symbol = 'EURUSD'
    """)
    
    assert len(result) == 1
    assert result[0]["symbol"] == "EURUSD"
    assert result[0]["order_type"] == "BUY"
    assert float(result[0]["volume"]) == 0.1
    assert float(result[0]["price"]) == 1.1000
    assert float(result[0]["sl"]) == 1.0950
    assert float(result[0]["tp"]) == 1.1100
    assert result[0]["status"] == "OPEN"


@pytest.mark.asyncio
async def test_historical_data_operations(test_db_handler, clean_test_db):
    """Test operacji na danych historycznych."""
    # Dodaj dane historyczne
    base_time = datetime.now().replace(microsecond=0)
    start_time = base_time - timedelta(hours=24)
    
    for i in range(24):
        timestamp = start_time + timedelta(hours=i)
        await test_db_handler.execute("""
            INSERT INTO historical_data (
                symbol, timeframe, timestamp,
                open, high, low, close, volume
            ) VALUES (
                'EURUSD', '1H', $1,
                1.1000, 1.1100, 1.0900, 1.1050, 1000
            )
        """, timestamp)
    
    # Sprawdź czy dane zostały dodane
    result = await test_db_handler.fetch_all("""
        SELECT * FROM historical_data 
        WHERE symbol = 'EURUSD' AND timeframe = '1H'
        ORDER BY timestamp
    """)
    
    assert len(result) == 24
    assert result[0]["timestamp"] == start_time
    assert result[-1]["timestamp"] == base_time - timedelta(hours=1)


@pytest.mark.asyncio
async def test_concurrent_operations(test_db_handler, clean_test_db):
    """Test współbieżnych operacji na bazie."""
    async def insert_data(symbol: str, count: int):
        for i in range(count):
            await test_db_handler.execute("""
                INSERT INTO market_data (
                    symbol, timestamp, open, high, low, close, volume
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7
                )
            """, symbol, 
                datetime.now() + timedelta(seconds=i),
                1.1000 + i * 0.0001,
                1.1100 + i * 0.0001,
                1.0900 + i * 0.0001,
                1.1050 + i * 0.0001,
                1000.0 + i)
    
    # Uruchom współbieżne operacje
    await asyncio.gather(
        insert_data("EURUSD", 100),
        insert_data("GBPUSD", 100),
        insert_data("USDJPY", 100)
    )
    
    # Sprawdź czy wszystkie dane zostały zapisane
    result = await test_db_handler.fetch_all("""
        SELECT symbol, COUNT(*) as count 
        FROM market_data 
        GROUP BY symbol
    """)
    
    assert len(result) == 3
    for row in result:
        assert row["count"] == 100


@pytest.mark.asyncio
async def test_error_handling(test_db_handler):
    """Test obsługi błędów bazy danych."""
    # Próba duplikacji unikalnego klucza
    test_data = {
        "symbol": "EURUSD",
        "timestamp": datetime.now(),
        "open": 1.1000,
        "high": 1.1100,
        "low": 1.0900,
        "close": 1.1050,
        "volume": 1000.0
    }
    
    # Pierwszy zapis powinien się udać
    await test_db_handler.execute("""
        INSERT INTO market_data (
            symbol, timestamp, open, high, low, close, volume
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7
        )
    """, test_data["symbol"], test_data["timestamp"], test_data["open"],
        test_data["high"], test_data["low"], test_data["close"], test_data["volume"])
    
    # Drugi zapis z tymi samymi danymi powinien się nie udać
    with pytest.raises(asyncpg.UniqueViolationError):
        await test_db_handler.execute("""
            INSERT INTO market_data (
                symbol, timestamp, open, high, low, close, volume
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7
            )
        """, test_data["symbol"], test_data["timestamp"], test_data["open"],
            test_data["high"], test_data["low"], test_data["close"], test_data["volume"])


@pytest.mark.asyncio
async def test_transaction_rollback(test_db_handler, clean_test_db):
    """Test wycofywania transakcji."""
    async with test_db_handler.pool.acquire() as conn:
        # Rozpocznij transakcję
        async with conn.transaction():
            # Wykonaj kilka operacji
            await conn.execute("""
                INSERT INTO market_data (
                    symbol, timestamp, open, high, low, close, volume
                ) VALUES (
                    'EURUSD', NOW(), 1.1000, 1.1100, 1.0900, 1.1050, 1000
                )
            """)
            
            # Zasymuluj błąd
            with pytest.raises(asyncpg.UndefinedTableError):
                await conn.execute("SELECT * FROM nieistniejaca_tabela")
    
    # Sprawdź czy dane zostały wycofane
    result = await test_db_handler.fetch_all("SELECT * FROM market_data")
    assert len(result) == 0 