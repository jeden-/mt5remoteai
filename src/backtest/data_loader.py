"""
Moduł do ładowania danych historycznych z MT5.
"""
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, List, Optional
import numpy as np
from loguru import logger
from ..database.postgres_handler import PostgresHandler

class HistoricalDataLoader:
    """Klasa do ładowania i przetwarzania danych historycznych z MT5."""
    
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
            db_handler: Handler bazy danych (opcjonalnie)
        """
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
                return self.add_indicators(df)

        # Jeśli nie ma w bazie lub brak handlera, pobierz z MT5
        df = await self.load_from_mt5()
        
        # Zapisz do bazy jeśli handler jest dostępny
        if self.db_handler and not df.empty:
            await self.save_to_database(df)
            
        return df

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
            
        # Sprawdź czy timeframe jest prawidłowy
        if self.timeframe not in self.timeframe_map:
            logger.error(f"❌ Nieprawidłowy timeframe: {self.timeframe}")
            raise KeyError(f"Nieprawidłowy timeframe: {self.timeframe}")
            
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
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
            
        # Konwertuj na DataFrame
        df = pd.DataFrame(rates)
        
        # Sprawdź wymagane kolumny
        required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ Brakujące kolumny w danych z MT5: {missing_columns}")
            raise KeyError(f"Brakujące kolumny w danych z MT5: {', '.join(missing_columns)}")
            
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
        df = self.add_indicators(df)
        
        return df

    async def save_to_database(self, df: pd.DataFrame) -> None:
        """
        Zapisuje dane historyczne do bazy danych.
        
        Args:
            df: DataFrame z danymi do zapisu
            
        Raises:
            KeyError: Gdy brakuje wymaganych kolumn
            ValueError: Gdy dane są nieprawidłowe
        """
        if df.empty:
            logger.warning("⚠️ Pusty DataFrame - nie można zapisać do bazy")
            return
            
        # Sprawdź czy wszystkie wymagane kolumny istnieją
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"❌ Brakujące kolumny: {', '.join(missing_columns)}")
            
        # Sprawdź czy dane są numeryczne
        for col in required_columns:
            if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                raise ValueError(f"❌ Nieprawidłowe dane w kolumnie {col}")
                
        # Przygotuj dane do zapisu
        data = []
        for timestamp, row in df.iterrows():
            data.append((
                self.symbol,
                self.timeframe,
                timestamp,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
            
        # Zapisz do bazy
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
        
        try:
            await self.db_handler.execute_many(query, data)
            logger.info(f"✅ Zapisano {len(data)} rekordów do bazy")
        except Exception as e:
            logger.error(f"❌ Błąd podczas zapisu do bazy: {e}")
            # Nie propagujemy błędu dalej

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dodaje wskaźniki techniczne do danych.
        
        Args:
            df: DataFrame z danymi historycznymi
            
        Returns:
            DataFrame z dodanymi wskaźnikami
        """
        # Sprawdź czy DataFrame nie jest pusty
        if df.empty:
            logger.warning("⚠️ Pusty DataFrame - nie można dodać wskaźników")
            return df
            
        # Sprawdź czy wszystkie wymagane kolumny istnieją
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"❌ Brakujące kolumny: {', '.join([col for col in required_columns if col not in df.columns])}")
            return df
            
        try:
            # Skopiuj DataFrame aby nie modyfikować oryginału
            df = df.copy()
            
            # Sprawdź czy dane są numeryczne
            for col in required_columns:
                if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                    logger.warning(f"⚠️ Nieprawidłowe dane w kolumnie {col}")
                    # Dodaj puste kolumny wskaźników
                    for indicator in ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']:
                        df[indicator] = np.nan
                    return df
            
            # Konwertuj kolumny na typ numeryczny
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Jeśli mamy tylko jeden wiersz lub same NaN, wszystkie wskaźniki będą NaN
            if len(df) < 2 or df[required_columns].isna().all().all():
                logger.warning("⚠️ Za mało danych do obliczenia wskaźników")
                for indicator in ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']:
                    df[indicator] = np.nan
                return df
            
            # Sprawdź czy dane są prawidłowe (nie ma samych zer)
            if (df[required_columns] == 0).all().all():
                logger.warning("⚠️ Nieprawidłowe dane wejściowe - same zera")
                for indicator in ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']:
                    df[indicator] = np.nan
                return df
            
            # SMA
            df['SMA_20'] = df['close'].rolling(window=20, min_periods=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50, min_periods=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df.loc[df.index[0], 'RSI'] = np.nan  # Pierwszy element zawsze NaN
            df['RSI'] = df['RSI'].clip(0, 100)  # Ogranicz do zakresu [0, 100]
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20, min_periods=20).mean()
            bb_std = df['close'].rolling(window=20, min_periods=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Upewnij się, że wszystkie wskaźniki mają prawidłowe wartości
            indicators = ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
            for indicator in indicators:
                # Zamień inf/-inf na NaN
                df[indicator] = df[indicator].replace([np.inf, -np.inf], np.nan)
                # Sprawdź czy są jakieś wartości NaN
                if df[indicator].isna().any():
                    logger.warning(f"⚠️ Wykryto wartości NaN w {indicator}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania wskaźników: {e}")
            # W przypadku błędu, dodaj puste kolumny wskaźników
            for indicator in ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']:
                df[indicator] = np.nan
            return df 