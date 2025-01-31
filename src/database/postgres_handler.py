"""
Moduł zawierający klasę do obsługi bazy danych PostgreSQL.
"""
import psycopg2
from psycopg2.extensions import connection
from typing import Optional, Dict
from loguru import logger

from ..utils.config import Config


class PostgresHandler:
    """Klasa odpowiedzialna za komunikację z bazą danych PostgreSQL."""

    def __init__(self, config: Config):
        """
        Inicjalizacja handlera bazy danych.

        Args:
            config: Obiekt konfiguracyjny z parametrami połączenia
        """
        self.config = config
        self.conn: Optional[connection] = None
        
    def connect(self) -> None:
        """Nawiązuje połączenie z bazą danych."""
        try:
            self.conn = psycopg2.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                database=self.config.DB_NAME,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD
            )
            logger.info("🥷 Połączono z bazą danych PostgreSQL")
        except Exception as e:
            logger.error(f"❌ Błąd połączenia z bazą danych: {e}")
            raise
            
    def disconnect(self) -> None:
        """Zamyka połączenie z bazą danych."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("🥷 Rozłączono z bazą danych")
            
    def create_tables(self) -> None:
        """Tworzy wymagane tabele w bazie danych."""
        if not self.conn:
            raise RuntimeError("⚠️ Brak połączenia z bazą danych")
            
        with self.conn.cursor() as cur:
            # Tabela dla danych rynkowych
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DECIMAL NOT NULL,
                    high DECIMAL NOT NULL,
                    low DECIMAL NOT NULL,
                    close DECIMAL NOT NULL,
                    volume DECIMAL NOT NULL
                )
            """)
            
            # Tabela dla transakcji
            cur.execute("""
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
                    profit DECIMAL,
                    status VARCHAR(10) NOT NULL
                )
            """)
            
        self.conn.commit()
        logger.info("🥷 Utworzono tabele w bazie danych")

    def save_market_data(self, data: Dict) -> bool:
        """
        Zapisywanie danych rynkowych do bazy.

        Args:
            data: Słownik zawierający dane rynkowe (symbol, timestamp, OHLCV)

        Returns:
            bool: True jeśli zapis się powiódł, False w przeciwnym razie
        """
        if not self.conn:
            logger.warning("⚠️ Brak połączenia z bazą PostgreSQL")
            return False

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    data["symbol"],
                    data["timestamp"],
                    data["open"],
                    data["high"],
                    data["low"],
                    data["close"],
                    data["volume"]
                ))
            
            self.conn.commit()
            logger.info(f"🥷 Zapisano dane rynkowe dla {data['symbol']}")
            return True
        except Exception as e:
            logger.error(f"❌ Błąd podczas zapisywania danych rynkowych: {str(e)}")
            self.conn.rollback()
            return False 