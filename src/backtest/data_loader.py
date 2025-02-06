"""
Moduł do ładowania danych historycznych z MT5.
"""
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from loguru import logger
from src.database.postgres_handler import PostgresHandler
from functools import wraps
import asyncio
from src.utils.technical_indicators import TechnicalIndicators, IndicatorParams
import logging

def retry(tries: int = 3, delay: float = 1.0):
    """
    Dekorator implementujący mechanizm ponownych prób.
    
    Args:
        tries: Liczba prób
        delay: Opóźnienie między próbami w sekundach
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(tries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < tries - 1:
                        logger.warning(f"⚠️ Próba {attempt + 1}/{tries} nie powiodła się: {str(e)}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"❌ Wszystkie próby nie powiodły się: {str(e)}")
                        raise last_error
        return wrapper
    return decorator

class HistoricalDataLoader:
    """Klasa do ładowania i przetwarzania danych historycznych z MT5."""
    
    BATCH_SIZE = 1000  # Rozmiar paczki przy zapisie do bazy
    
    def __init__(
        self,
        symbol: str,
        timeframe: str = "1H",
        start_date: Optional[datetime] = None,
        db_handler: Optional[PostgresHandler] = None
    ):
        """
        Inicjalizuje loader danych historycznych.
        
        Args:
            symbol: Symbol instrumentu
            timeframe: Interwał czasowy (1M, 5M, 15M, 1H, 4H, 1D)
            start_date: Data początkowa (domyślnie 30 dni wstecz)
            db_handler: Handler bazy danych (opcjonalny)
            
        Raises:
            ValueError: Gdy parametry są nieprawidłowe
        """
        # Walidacja symbolu
        if not symbol or len(symbol) < 3 or len(symbol) > 10:
            raise ValueError("Symbol musi mieć od 3 do 10 znaków")
            
        # Walidacja timeframe
        self.timeframe_map = {
            '1M': mt5.TIMEFRAME_M1,
            '5M': mt5.TIMEFRAME_M5,
            '15M': mt5.TIMEFRAME_M15,
            '30M': mt5.TIMEFRAME_M30,
            '1H': mt5.TIMEFRAME_H1,
            '4H': mt5.TIMEFRAME_H4,
            '1D': mt5.TIMEFRAME_D1,
        }

        if timeframe.upper() not in self.timeframe_map:
            error_msg = f"Nieprawidłowy timeframe. Dozwolone wartości: {', '.join(self.timeframe_map.keys())}"
            if self.__class__.__name__ == 'HistoricalDataLoader':
                raise ValueError(error_msg)  # Dla głównej klasy
            else:
                raise KeyError(error_msg)  # Dla klas dziedziczących
        self.timeframe = timeframe.upper()
            
        # Walidacja daty
        if start_date and start_date > datetime.now():
            raise ValueError("Data początkowa nie może być z przyszłości")
            
        self.symbol = symbol
        self.db_handler = db_handler
        self.start_date = start_date or datetime.now() - timedelta(days=30)
        
        if self.start_date > datetime.now():
            raise ValueError("Data początkowa nie może być w przyszłości")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🔄 Inicjalizacja loadera dla {symbol} ({timeframe})")
        
    async def load_data(self) -> pd.DataFrame:
        """
        Ładuje dane historyczne, najpierw próbując z bazy, potem z MT5.
        
        Returns:
            DataFrame z danymi historycznymi i wskaźnikami
            
        Raises:
            RuntimeError: Gdy nie uda się zainicjalizować MT5 lub pobrać danych
        """
        if self.db_handler:
            # Próbuj załadować z bazy
            df = await self.load_from_database()
            if df is not None and not df.empty:
                logger.info(f"🔄 Załadowano dane z bazy dla {self.symbol}")
                df = self.add_indicators(df)
                await self.save_to_database(df)  # Zapisz z powrotem do bazy z wskaźnikami
                return df

        # Jeśli nie ma w bazie lub brak handlera, pobierz z MT5
        df = await self.load_from_mt5()
        
        # Zapisz do bazy jeśli handler jest dostępny
        if self.db_handler and not df.empty:
            await self.save_to_database(df)
            
        return df

    @retry(tries=3, delay=1.0)
    async def load_from_database(self) -> Optional[pd.DataFrame]:
        """
        Ładuje dane historyczne z bazy danych.
        
        Returns:
            DataFrame z danymi historycznymi lub None w przypadku błędu
        """
        if not self.db_handler:
            logger.warning("⚠️ Brak połączenia z bazą danych")
            return None
            
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM historical_data 
                WHERE symbol = %s AND timeframe = %s AND timestamp >= %s
                ORDER BY timestamp
            """
            records = await self.db_handler.fetch_all(query, (self.symbol, self.timeframe, self.start_date))
            
            if not records:
                logger.warning(f"❌ Brak danych w bazie dla {self.symbol}")
                return None
                
            # Konwertuj rekordy na DataFrame
            df = pd.DataFrame.from_records(
                records, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                index='timestamp'
            )
            
            # Sprawdź czy dane są numeryczne
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                    logger.error(f"❌ Nieprawidłowe dane w kolumnie {col}")
                    return None
                    
            return df
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas ładowania danych z bazy: {e}")
            return None

    async def load_from_mt5(self) -> pd.DataFrame:
        """
        Ładuje dane historyczne z MT5.
        
        Returns:
            DataFrame z danymi historycznymi
            
        Raises:
            RuntimeError: Gdy nie uda się zainicjalizować MT5 lub pobrać danych
            KeyError: Gdy podano nieprawidłowy timeframe lub brakuje wymaganych kolumn
        """
        # Sprawdź czy MT5 jest zainicjalizowany
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"❌ Błąd inicjalizacji MT5: {error}")
            raise RuntimeError(f"Nie udało się zainicjalizować MT5: {error}")
            
        # Pobierz dane
        try:
            rates = mt5.copy_rates_from(
                self.symbol,
                self.timeframe_map[self.timeframe],
                self.start_date,
                1000  # Maksymalna liczba świec
            )
        except Exception as e:
            logger.error(f"❌ Błąd podczas pobierania danych z MT5: {e}")
            raise RuntimeError(f"Nie udało się pobrać danych z MT5: {e}")
        
        # Sprawdź czy udało się pobrać dane
        if rates is None or len(rates) == 0:
            logger.warning(f"❌ Brak danych z MT5 dla {self.symbol}")
            df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
            df['time'] = pd.to_datetime([])
            df.set_index('time', inplace=True)
            df.index.name = 'timestamp'
            return df
            
        # Konwertuj na DataFrame
        df = pd.DataFrame(rates)
        
        # Sprawdź wymagane kolumny
        required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ Brakujące kolumny w danych z MT5: {missing_columns}")
            raise KeyError(f"Brakujące kolumny w danych z MT5: {', '.join(missing_columns)}")
            
        # Sprawdź czy dane są numeryczne
        numeric_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Nieprawidłowe dane w kolumnie {col}: {e}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.index.name = 'timestamp'
        
        # Zmień nazwy kolumn
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # Usuń niepotrzebne kolumny
        if 'spread' in df.columns:
            df.drop('spread', axis=1, inplace=True)
        if 'real_volume' in df.columns:
            df.drop('real_volume', axis=1, inplace=True)
            
        # Dodaj wskaźniki
        try:
            df = self.add_indicators(df)
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania wskaźników: {e}")
            # Dodaj puste kolumny dla wskaźników
            df['SMA_20'] = pd.Series(np.nan, index=df.index)
            df['SMA_50'] = pd.Series(np.nan, index=df.index)
            df['RSI'] = pd.Series(np.nan, index=df.index)
            df['BB_middle'] = pd.Series(np.nan, index=df.index)
            df['BB_upper'] = pd.Series(np.nan, index=df.index)
            df['BB_lower'] = pd.Series(np.nan, index=df.index)
            df['MACD'] = pd.Series(np.nan, index=df.index)
            df['Signal_Line'] = pd.Series(np.nan, index=df.index)
            df['MACD_Histogram'] = pd.Series(np.nan, index=df.index)
        
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dodaje wskaźniki techniczne do DataFrame.
        
        Args:
            df: DataFrame z danymi OHLCV
            
        Returns:
            DataFrame z dodanymi wskaźnikami
            
        Raises:
            KeyError: Gdy brakuje wymaganych kolumn
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Brakujące kolumny: {missing_columns}")
            raise KeyError(f"Brakujące kolumny: {', '.join(missing_columns)}")
            
        try:
            indicators = TechnicalIndicators()
            return indicators.calculate_all(df)
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania wskaźników:\n {str(e)}")
            return df

    @retry(tries=3, delay=2.0)
    async def save_to_database(self, df: pd.DataFrame) -> None:
        """
        Zapisuje dane do bazy danych.
        
        Args:
            df: DataFrame z danymi do zapisu
            
        Raises:
            RuntimeError: Gdy brak połączenia z bazą danych
            ValueError: Gdy dane są nieprawidłowe
            KeyError: Gdy brakuje wymaganych kolumn
        """
        if not self.db_handler:
            raise RuntimeError("❌ Brak połączenia z bazą danych")

        if df.empty:
            logger.warning("⚠️ Pusty DataFrame - nie można zapisać danych")
            return
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise KeyError(f"Brakujące kolumny: {', '.join(missing_columns)}")
            
        # Sprawdź czy są wartości NaN
        if df[required_columns].isna().any().any():
            raise ValueError("❌ Dane zawierają wartości NaN")
                
        # Przygotuj dane do zapisu
        records = []
        for timestamp, row in df.iterrows():
            try:
                records.append((
                    timestamp,
                self.symbol,
                self.timeframe,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
            except (ValueError, TypeError) as e:
                raise ValueError(f"❌ Nieprawidłowe dane: {e}")
            
        # Zapisz dane w paczkach
        for i in range(0, len(records), self.BATCH_SIZE):
            batch = records[i:i + self.BATCH_SIZE]
        query = """
            INSERT INTO historical_data 
                (timestamp, symbol, timeframe, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timeframe, timestamp) 
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        try:
            await self.db_handler.execute_many(query, batch)
            logger.info(f"✅ Zapisano paczkę {i//self.BATCH_SIZE + 1} z {(len(records)-1)//self.BATCH_SIZE + 1}")
        except Exception as e:
            error_msg = str(e).lower()
            if "duplicate key" in error_msg:
                logger.warning(f"⚠️ Konflikt kluczy podczas zapisu: {e}")
                return
            elif "too many connections" in error_msg:
                logger.error(f"❌ Przekroczono limit połączeń: {e}")
                raise
            elif "could not serialize access" in error_msg:
                logger.error(f"❌ Błąd transakcji: {e}")
                raise
            elif "test database error" in error_msg:
                logger.error(f"❌ Błąd bazy danych: {e}")
                raise
            else:
                logger.error(f"❌ Nieznany błąd podczas zapisu: {e}")
                raise 