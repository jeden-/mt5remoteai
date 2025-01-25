"""Moduł zawierający funkcje analizy technicznej."""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class AnalizaTechniczna:
    """Klasa zawierająca metody analizy technicznej."""

    def __init__(self):
        """Inicjalizacja obiektu analizy technicznej."""
        pass

    def oblicz_rsi(self, df: pd.DataFrame, okres: int = 14) -> pd.Series:
        """Oblicza wskaźnik RSI dla danego szeregu czasowego.
        
        Args:
            df: DataFrame z danymi cenowymi
            okres: Okres RSI (domyślnie 14)
            
        Returns:
            Series z wartościami RSI
        """
        try:
            delta = df['close'].diff()
            zyski = delta.copy()
            straty = delta.copy()
            zyski[zyski < 0] = 0
            straty[straty > 0] = 0
            straty = abs(straty)
            
            avg_zyski = zyski.rolling(window=okres).mean()
            avg_straty = straty.rolling(window=okres).mean()
            
            rs = avg_zyski / avg_straty
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania RSI: {str(e)}")
            return pd.Series(index=df.index)

    def oblicz_atr(self, df: pd.DataFrame, okres: int = 14) -> pd.Series:
        """Oblicza wskaźnik Average True Range (ATR).
        
        Args:
            df: DataFrame z danymi cenowymi (high, low, close)
            okres: Okres ATR (domyślnie 14)
            
        Returns:
            Series z wartościami ATR
        """
        try:
            # Obliczanie True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # Obliczanie ATR jako średniej kroczącej z True Range
            atr = true_range.rolling(window=okres).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania ATR: {str(e)}")
            return pd.Series(index=df.index)

    def oblicz_momentum(self, df: pd.DataFrame, okres: int = 10) -> pd.Series:
        """Oblicza momentum dla danego szeregu czasowego.
        
        Args:
            df: DataFrame z danymi cenowymi
            okres: Okres momentum (domyślnie 10)
            
        Returns:
            Series z wartościami momentum
        """
        try:
            momentum = df['close'].diff(okres)
            return momentum
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania momentum: {str(e)}")
            return pd.Series(index=df.index)

    def oblicz_sma(self, df: pd.DataFrame, okres: int = 20) -> pd.Series:
        """Oblicza prostą średnią kroczącą (SMA).
        
        Args:
            df: DataFrame z danymi cenowymi
            okres: Okres średniej (domyślnie 20)
            
        Returns:
            Series z wartościami SMA
        """
        try:
            sma = df['close'].rolling(window=okres).mean()
            return sma
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania SMA: {str(e)}")
            return pd.Series(index=df.index)

    def oblicz_volatility(self, df: pd.DataFrame, okres: int = 20) -> pd.Series:
        """Oblicza zmienność jako odchylenie standardowe zwrotów.
        
        Args:
            df: DataFrame z danymi cenowymi
            okres: Okres zmienności (domyślnie 20)
            
        Returns:
            Series z wartościami zmienności
        """
        try:
            zwroty = df['close'].pct_change()
            volatility = zwroty.rolling(window=okres).std()
            return volatility
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania zmienności: {str(e)}")
            return pd.Series(index=df.index)

    def oblicz_relative_volume(self, df: pd.DataFrame, okres: int = 20) -> pd.Series:
        """Oblicza względny wolumen jako stosunek bieżącego wolumenu do średniej.
        
        Args:
            df: DataFrame z danymi cenowymi i wolumenem
            okres: Okres średniej (domyślnie 20)
            
        Returns:
            Series z wartościami względnego wolumenu
        """
        try:
            avg_volume = df['volume'].rolling(window=okres).mean()
            relative_volume = df['volume'] / avg_volume
            return relative_volume
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania względnego wolumenu: {str(e)}")
            return pd.Series(index=df.index) 