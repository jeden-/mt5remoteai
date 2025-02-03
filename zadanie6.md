ZADANIE #6 - Implementacja modułu RAG

1. Dodaj nowe zależności do requirements.txt:
```txt
chromadb==0.4.22
sentence-transformers==2.5.1

2. Utwórz nowy folder src/rag/ z plikami:
src/rag/
├── __init__.py
├── embeddings_handler.py
├── market_memory.py
└── context_provider.py


DODATKOWO
# W pliku src/backtest/data_loader.py dodaj:

from typing import Optional, List
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5
from ..database.postgres_handler import PostgresHandler

class HistoricalDataLoader:
    def __init__(
        self,
        symbol: str,
        timeframe: str = "1H",
        start_date: Optional[datetime] = None,
        db_handler: Optional[PostgresHandler] = None
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_map = {
            "1M": mt5.TIMEFRAME_M1,
            "5M": mt5.TIMEFRAME_M5,
            "15M": mt5.TIMEFRAME_M15,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1,
        }
        self.start_date = start_date or (datetime.now() - timedelta(days=30))
        self.db_handler = db_handler

    async def load_data(self) -> pd.DataFrame:
        """Ładuje dane historyczne, najpierw próbując z bazy, potem z MT5"""
        if self.db_handler:
            # Próbuj załadować z bazy
            df = await self.load_from_database()
            if df is not None and not df.empty:
                print(f"Załadowano dane z bazy dla {self.symbol}")
                return self.add_indicators(df)

        # Jeśli nie ma w bazie lub brak handlera, pobierz z MT5
        df = await self.load_from_mt5()
        
        # Zapisz do bazy jeśli handler jest dostępny
        if self.db_handler and not df.empty:
            await self.save_to_database(df)
            
        return df

    async def load_from_database(self) -> Optional[pd.DataFrame]:
        """Ładuje dane historyczne z bazy PostgreSQL"""
        query = """
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM historical_data
        WHERE 
            symbol = %s 
            AND timeframe = %s
            AND timestamp >= %s
        ORDER BY timestamp ASC
        """
        
        try:
            result = await self.db_handler.fetch_all(
                query,
                (self.symbol, self.timeframe, self.start_date)
            )
            
            if result:
                df = pd.DataFrame(result, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            print(f"Błąd podczas ładowania danych z bazy: {e}")
            
        return None

    async def load_from_mt5(self) -> pd.DataFrame:
        """Ładuje dane historyczne z MT5"""
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MT5")
            
        try:
            rates = mt5.copy_rates_from(
                self.symbol,
                self.timeframe_map[self.timeframe],
                self.start_date,
                10000
            )
            
            if rates is None:
                raise RuntimeError(f"Failed to get historical data for {self.symbol}")
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return self.add_indicators(df)
            
        finally:
            mt5.shutdown()

    async def save_to_database(self, df: pd.DataFrame) -> None:
        """Zapisuje dane historyczne do bazy"""
        query = """
        INSERT INTO historical_data 
        (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timeframe, timestamp) 
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
        """
        
        data = [
            (
                self.symbol,
                self.timeframe,
                index,
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            )
            for index, row in df.iterrows()
        ]
        
        try:
            await self.db_handler.execute_many(query, data)
            print(f"Zapisano {len(data)} rekordów do bazy dla {self.symbol}")
        except Exception as e:
            print(f"Błąd podczas zapisywania do bazy: {e}")

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje wskaźniki techniczne do danych"""
        # (pozostała implementacja bez zmian)

Dodaj też nową tabelę w bazie danych. W pliku src/database/postgres_handler.py dodaj:
async def create_historical_data_table(self) -> None:
    """Tworzy tabelę dla danych historycznych"""
    query = """
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
    """
    await self.execute(query)

async def execute(self, query: str, params: tuple = None) -> None:
    """Wykonuje zapytanie SQL"""
    if not self.conn:
        raise RuntimeError("Brak połączenia z bazą danych")
        
    async with self.conn.cursor() as cur:
        await cur.execute(query, params)
    await self.conn.commit()

async def execute_many(self, query: str, params: List[tuple]) -> None:
    """Wykonuje wiele zapytań SQL"""
    if not self.conn:
        raise RuntimeError("Brak połączenia z bazą danych")
        
    async with self.conn.cursor() as cur:
        await cur.executemany(query, params)
    await self.conn.commit()

async def fetch_all(self, query: str, params: tuple = None) -> List[tuple]:
    """Pobiera wszystkie wyniki zapytania"""
    if not self.conn:
        raise RuntimeError("Brak połączenia z bazą danych")
        
    async with self.conn.cursor() as cur:
        await cur.execute(query, params)
        return await cur.fetchall()

Zaktualizuj też główną funkcję backtestingu, aby używała bazy danych:
async def run_backtest():
    logger = TradingLogger()
    config = Config.load_config()
    
    # Inicjalizacja połączenia z bazą
    db_handler = PostgresHandler(config)
    await db_handler.connect()
    await db_handler.create_historical_data_table()
    
    try:
        # Inicjalizacja strategii
        strategy = BasicStrategy(...)  # jak wcześniej
        
        # Utworzenie backtestera z obsługą bazy
        backtester = Backtester(
            strategy=strategy,
            symbol='EURUSD',
            timeframe='1H',
            initial_capital=10000,
            start_date=datetime.now() - timedelta(days=30),
            logger=logger,
            db_handler=db_handler  # dodaj handler bazy
        )
        
        # Uruchomienie backtestu
        results = await backtester.run_backtest()
        
        # Wizualizacja wyników
        visualizer = BacktestVisualizer(backtester.data, backtester.trades)
        visualizer.save_dashboard('backtest_results.html')
        
        print("\nWYNIKI BACKTESTU:")
        for metric, value in results.items():
            print(f"{metric}: {value}")
            
    finally:
        await db_handler.disconnect()
