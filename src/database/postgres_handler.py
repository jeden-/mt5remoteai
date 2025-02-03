"""
Moduł zawierający klasę do obsługi bazy danych PostgreSQL.
"""
from typing import Optional, Dict, List, Any, Union, Callable
import asyncpg
from asyncpg import Pool
from loguru import logger
from datetime import datetime
import asyncio
from functools import wraps


def retry_on_error(retries: int = 3, delay: float = 1.0):
    """
    Dekorator implementujący mechanizm ponownych prób dla operacji bazodanowych.
    
    Args:
        retries: Liczba prób ponowienia operacji
        delay: Opóźnienie między próbami w sekundach
        
    Returns:
        Callable: Udekorowana funkcja
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (asyncpg.PostgresError, asyncio.TimeoutError) as e:
                    last_error = e
                    if attempt < retries - 1:
                        logger.warning(f"⚠️ Próba {attempt + 1}/{retries} nie powiodła się: {str(e)}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"❌ Wszystkie próby nie powiodły się: {str(e)}")
                        raise last_error
            return None
        return wrapper
    return decorator


class PostgresHandler:
    """Handler do operacji na bazie PostgreSQL."""

    def __init__(
        self,
        user: str = 'postgres',
        password: str = 'mt5remote',
        database: str = 'mt5remotetest',
        host: str = 'localhost',
        port: int = 5432,
        min_size: int = 1,
        max_size: int = 5,
        command_timeout: int = 30,
        ssl: bool = False
    ) -> None:
        """
        Inicjalizuje handler bazy danych.

        Args:
            user: Nazwa użytkownika bazy danych
            password: Hasło do bazy danych
            database: Nazwa bazy danych
            host: Host bazy danych
            port: Port bazy danych
            min_size: Minimalna liczba połączeń w puli
            max_size: Maksymalna liczba połączeń w puli
            command_timeout: Timeout dla komend w sekundach
            ssl: Czy używać SSL
        """
        self.user = user
        self.password = password
        self.database = database
        self.host = host
        self.port = port
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.ssl = ssl
        self.pool: Optional[asyncpg.Pool] = None
        self._loop = None
        self._initialized = False

    async def initialize(self) -> None:
        """Inicjalizuje połączenie z bazą danych i tworzy wymagane tabele."""
        try:
            self._loop = asyncio.get_running_loop()
            await self.create_pool()
            await self.create_tables()
        except Exception as e:
            logger.error(f"❌ Błąd inicjalizacji: {e}")
            if self.pool:
                await self.close()
            raise

    async def create_pool(self) -> None:
        """Tworzy pulę połączeń do bazy danych."""
        try:
            self.pool = await asyncpg.create_pool(
                user=self.user,
                password=self.password,
                database=self.database,
                host=self.host,
                port=self.port,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout,
                ssl=self.ssl
            )
            logger.info("✅ Pomyślnie utworzono pulę połączeń do bazy danych")
        except Exception as e:
            logger.error(f"❌ Błąd podczas tworzenia puli połączeń: {str(e)}")
            raise

    async def close(self) -> None:
        """Zamyka połączenie z bazą danych."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("🥷 Zamknięto pulę połączeń do bazy")

    @retry_on_error()
    async def execute(self, query: str, *args) -> None:
        """
        Wykonuje zapytanie SQL.

        Args:
            query: Zapytanie SQL
            *args: Parametry zapytania
        """
        if not self.pool:
            raise RuntimeError("❌ Brak puli połączeń do bazy")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(query, *args)

    @retry_on_error()
    async def fetch_val(self, query: str, *args) -> Any:
        """
        Pobiera pojedynczą wartość z zapytania.

        Args:
            query: Zapytanie SQL
            *args: Parametry zapytania

        Returns:
            Pojedyncza wartość
        """
        if not self.pool:
            raise RuntimeError("❌ Brak puli połączeń do bazy")

        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    @retry_on_error()
    async def fetch_all(self, query: str, *args) -> List[asyncpg.Record]:
        """
        Pobiera wszystkie wiersze z zapytania.

        Args:
            query: Zapytanie SQL
            *args: Parametry zapytania

        Returns:
            Lista wierszy
        """
        if not self.pool:
            raise RuntimeError("❌ Brak puli połączeń do bazy")

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @retry_on_error()
    async def create_tables(self) -> None:
        """Tworzy wymagane tabele w bazie danych."""
        if not self.pool:
            raise RuntimeError("⚠️ Brak puli połączeń do bazy")

        async with self.pool.acquire() as conn:
            # Tabela dla danych rynkowych
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DECIMAL NOT NULL,
                    high DECIMAL NOT NULL,
                    low DECIMAL NOT NULL,
                    close DECIMAL NOT NULL,
                    volume DECIMAL NOT NULL,
                    UNIQUE (symbol, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_market_data_lookup
                ON market_data (symbol, timestamp);
            """)

            # Tabela dla transakcji
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    order_type VARCHAR(10) NOT NULL,
                    volume DECIMAL NOT NULL,
                    price DECIMAL NOT NULL,
                    sl DECIMAL,
                    tp DECIMAL,
                    open_time TIMESTAMP NOT NULL,
                    close_time TIMESTAMP,
                    status VARCHAR(10) NOT NULL,
                    profit DECIMAL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_lookup
                ON trades (symbol, open_time);
            """)

            # Tabela dla danych historycznych
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DECIMAL NOT NULL,
                    high DECIMAL NOT NULL,
                    low DECIMAL NOT NULL,
                    close DECIMAL NOT NULL,
                    volume DECIMAL NOT NULL,
                    UNIQUE (symbol, timeframe, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_historical_data_lookup
                ON historical_data (symbol, timeframe, timestamp);
            """)

            logger.info("🥷 Utworzono tabele w bazie danych")

    @retry_on_error()
    async def save_market_data(self, data: Dict[str, Any]) -> bool:
        """
        Zapisywanie danych rynkowych do bazy.

        Args:
            data: Słownik zawierający dane rynkowe (symbol, timestamp, OHLCV)

        Returns:
            bool: True jeśli zapis się powiódł, False w przeciwnym razie
        """
        if not self.pool:
            logger.warning("⚠️ Brak puli połączeń do bazy")
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, timestamp) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, 
                    data["symbol"],
                    data["timestamp"],
                    data["open"],
                    data["high"],
                    data["low"],
                    data["close"],
                    data["volume"]
                )
            
            logger.info(f"🥷 Zapisano dane rynkowe dla {data['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas zapisywania danych rynkowych: {str(e)}")
            return False

    @retry_on_error()
    async def execute_many(self, query: str, args: List[tuple]) -> None:
        """
        Wykonuje wiele zapytań SQL.
        
        Args:
            query: Zapytanie SQL
            args: Lista krotek z parametrami
        """
        if not self.pool:
            raise RuntimeError("❌ Brak puli połączeń do bazy")
            
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(query, args) 