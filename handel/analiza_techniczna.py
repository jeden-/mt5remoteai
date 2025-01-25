"""
Moduł do analizy technicznej.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
import logging
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from scipy import stats
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)

class AnalizaTechniczna:
    """Klasa do przeprowadzania analizy technicznej."""
    
    def __init__(self):
        """Inicjalizacja analizy technicznej."""
        self.logger = logger
        
    def oblicz_rsi(self, ceny: np.ndarray, okres: int = 14) -> np.ndarray:
        """
        Oblicza wskaźnik RSI.
        
        Args:
            ceny: Tablica cen zamknięcia
            okres: Okres RSI (domyślnie 14)
            
        Returns:
            np.ndarray: Wartości RSI
        """
        try:
            df = pd.DataFrame({'close': ceny})
            rsi = RSIIndicator(close=df['close'], window=okres)
            return rsi.rsi().fillna(50.0).to_numpy()
        except Exception as e:
            self.logger.error(f"❌ Błąd obliczania RSI: {str(e)}")
            return np.full_like(ceny, 50.0)
            
    def oblicz_macd(
        self,
        ceny: np.ndarray,
        szybki_okres: int = 12,
        wolny_okres: int = 26,
        sygnal_okres: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Oblicza wskaźnik MACD.
        
        Args:
            ceny: Tablica cen zamknięcia
            szybki_okres: Okres szybkiej średniej (domyślnie 12)
            wolny_okres: Okres wolnej średniej (domyślnie 26)
            sygnal_okres: Okres linii sygnału (domyślnie 9)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (MACD, Sygnał, Histogram)
        """
        try:
            df = pd.DataFrame({'close': ceny})
            macd = MACD(
                close=df['close'],
                window_slow=min(wolny_okres, len(ceny)-1),
                window_fast=min(szybki_okres, len(ceny)-1),
                window_sign=min(sygnal_okres, len(ceny)-1)
            )
            return (
                macd.macd().fillna(0.0).to_numpy(),
                macd.macd_signal().fillna(0.0).to_numpy(),
                macd.macd_diff().fillna(0.0).to_numpy()
            )
        except Exception as e:
            self.logger.error(f"❌ Błąd obliczania MACD: {str(e)}")
            return np.array([]), np.array([]), np.array([])
            
    def oblicz_sma(self, ceny: np.ndarray, okres: int = 20) -> np.ndarray:
        """
        Oblicza średnią kroczącą (SMA).
        
        Args:
            ceny: Tablica cen zamknięcia
            okres: Okres średniej (domyślnie 20)
            
        Returns:
            np.ndarray: Wartości SMA
        """
        try:
            df = pd.DataFrame({'close': ceny})
            sma = SMAIndicator(close=df['close'], window=min(okres, len(ceny)-1))
            return sma.sma_indicator().bfill().to_numpy()
        except Exception as e:
            self.logger.error(f"❌ Błąd obliczania SMA: {str(e)}")
            return np.array([])
            
    def oblicz_ema(self, ceny: np.ndarray, okres: int = 20) -> np.ndarray:
        """
        Oblicza wykładniczą średnią kroczącą (EMA).
        
        Args:
            ceny: Tablica cen zamknięcia
            okres: Okres średniej (domyślnie 20)
            
        Returns:
            np.ndarray: Wartości EMA
        """
        try:
            df = pd.DataFrame({'close': ceny})
            ema = EMAIndicator(close=df['close'], window=min(okres, len(ceny)-1))
            return ema.ema_indicator().bfill().to_numpy()
        except Exception as e:
            self.logger.error(f"❌ Błąd obliczania EMA: {str(e)}")
            return np.array([])
            
    def wykryj_formacje_swiecowe(
        self,
        ceny: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_price: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Wykrywa formacje świecowe.
        
        Args:
            ceny: Tablica cen zamknięcia
            high: Tablica cen najwyższych
            low: Tablica cen najniższych
            open_price: Tablica cen otwarcia (opcjonalnie)
            
        Returns:
            Dict[str, np.ndarray]: Słownik z wykrytymi formacjami
        """
        try:
            if open_price is None:
                open_price = np.roll(ceny, 1)
                open_price[0] = ceny[0]
                
            df = pd.DataFrame({
                'open': open_price,
                'high': high,
                'low': low,
                'close': ceny
            })
            
            formacje = {}
            
            body = np.abs(ceny - open_price)
            upper_shadow = high - np.maximum(ceny, open_price)
            lower_shadow = np.minimum(ceny, open_price) - low
            
            formacje['hammer'] = (lower_shadow > 2 * body) & (upper_shadow < 0.5 * body)
            formacje['shooting_star'] = (upper_shadow > 2 * body) & (lower_shadow < 0.5 * body)
            
            formacje['doji'] = body < 0.1 * (high - low)
            
            prev_body = np.roll(body, 1)
            prev_direction = np.roll(ceny > open_price, 1)
            current_direction = ceny > open_price
            
            formacje['bullish_engulfing'] = (
                (body > prev_body) & 
                (current_direction) & 
                (~prev_direction)
            )
            
            formacje['bearish_engulfing'] = (
                (body > prev_body) & 
                (~current_direction) & 
                (prev_direction)
            )
            
            return {k: v.astype(int) for k, v in formacje.items()}
            
        except Exception as e:
            self.logger.error(f"❌ Błąd wykrywania formacji świecowych: {str(e)}")
            return {}
            
    def analizuj_wolumen(
        self,
        ceny: np.ndarray,
        wolumen: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        okres: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Analizuje wskaźniki wolumenu.
        
        Args:
            ceny: Tablica cen zamknięcia
            wolumen: Tablica wolumenu
            high: Tablica cen najwyższych
            low: Tablica cen najniższych
            okres: Okres analizy (domyślnie 20)
            
        Returns:
            Dict[str, np.ndarray]: Słownik ze wskaźnikami wolumenu
        """
        try:
            df = pd.DataFrame({
                'close': ceny,
                'volume': wolumen,
                'high': high,
                'low': low
            })
            
            obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
            adi = AccDistIndexIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            cmf = ChaikinMoneyFlowIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=min(okres, len(ceny)-1)
            )
            
            wynik = {
                'obv': obv.on_balance_volume().fillna(0.0).to_numpy(),
                'adi': adi.acc_dist_index().fillna(0.0).to_numpy(),
                'cmf': cmf.chaikin_money_flow().fillna(0.0).to_numpy()
            }
            return wynik
        except Exception as e:
            self.logger.error(f"❌ Błąd analizy wolumenu: {str(e)}")
            return {}
            
    def oblicz_vwap(self, df_or_high, low=None, close=None, volume=None) -> np.ndarray:
        """Oblicza VWAP (Volume Weighted Average Price).

        Args:
            df_or_high (Union[pd.DataFrame, np.ndarray]): DataFrame z danymi OHLCV lub tablica high
            low (np.ndarray, optional): Tablica low. Defaults to None.
            close (np.ndarray, optional): Tablica close. Defaults to None.
            volume (np.ndarray, optional): Tablica volume. Defaults to None.

        Returns:
            np.ndarray: VWAP dla każdego punktu
        """
        try:
            if isinstance(df_or_high, pd.DataFrame):
                high = df_or_high['high'].values
                low = df_or_high['low'].values
                close = df_or_high['close'].values
                volume = df_or_high['volume'].values
            else:
                high = df_or_high
                if any(x is None for x in [low, close, volume]):
                    raise ValueError("Jeśli nie podano DataFrame, wymagane są wszystkie tablice: high, low, close, volume")

            typical_price = (high + low + close) / 3
            vwap = np.zeros_like(typical_price)
            cumulative_tp_vol = np.zeros_like(typical_price)
            cumulative_vol = np.zeros_like(typical_price)

            for i in range(len(typical_price)):
                if i == 0:
                    cumulative_tp_vol[i] = typical_price[i] * volume[i]
                    cumulative_vol[i] = volume[i]
                else:
                    cumulative_tp_vol[i] = cumulative_tp_vol[i-1] + typical_price[i] * volume[i]
                    cumulative_vol[i] = cumulative_vol[i-1] + volume[i]
                vwap[i] = cumulative_tp_vol[i] / cumulative_vol[i] if cumulative_vol[i] > 0 else typical_price[i]

            return vwap

        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania VWAP: {str(e)}")
            return np.full_like(df_or_high['close'].values if isinstance(df_or_high, pd.DataFrame) else df_or_high, np.nan)
            
    def generuj_sygnaly(
        self,
        ceny: np.ndarray,
        wolumen: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        rsi_okres: int = 14,
        macd_szybki: int = 12,
        macd_wolny: int = 26,
        macd_sygnal: int = 9,
        vwap_okres: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Generuje sygnały handlowe na podstawie wskaźników.
        
        Args:
            ceny: Tablica cen zamknięcia
            wolumen: Tablica wolumenu
            high: Tablica cen najwyższych
            low: Tablica cen najniższych
            rsi_okres: Okres RSI
            macd_szybki: Okres szybkiej średniej MACD
            macd_wolny: Okres wolnej średniej MACD
            macd_sygnal: Okres linii sygnału MACD
            vwap_okres: Okres VWAP (domyślnie: cała historia)
            
        Returns:
            Dict[str, np.ndarray]: Słownik z sygnałami
        """
        try:
            rsi = self.oblicz_rsi(ceny, rsi_okres)
            macd, signal, hist = self.oblicz_macd(ceny, macd_szybki, macd_wolny, macd_sygnal)
            sma20 = self.oblicz_sma(ceny, 20)
            sma50 = self.oblicz_sma(ceny, 50)
            vwap = self.oblicz_vwap(ceny, high, ceny, wolumen)
            
            macd_przeciecie = np.zeros_like(ceny)
            macd_przeciecie[1:] = np.diff(np.signbit(hist))
            
            sygnaly = {
                'rsi_wykupienie': rsi > 70,
                'rsi_wyprzedanie': rsi < 30,
                'macd_przeciecie': macd_przeciecie,
                'trend_wzrostowy': sma20 > sma50,
                'trend_spadkowy': sma20 < sma50,
                'ponad_vwap': ceny > vwap,
                'ponizej_vwap': ceny < vwap
            }
            
            return sygnaly
            
        except Exception as e:
            self.logger.error(f"❌ Błąd generowania sygnałów: {str(e)}")
            return {}
            
    def oblicz_atr(self,
                   df_or_high: Union[pd.DataFrame, np.ndarray],
                   low: Optional[np.ndarray] = None,
                   close: Optional[np.ndarray] = None,
                   okres: int = 14) -> np.ndarray:
        """
        Oblicza wskaźnik ATR (Average True Range).
        
        Args:
            df_or_high: DataFrame z danymi OHLCV lub tablica high
            low: Tablica low (opcjonalnie, jeśli podano DataFrame)
            close: Tablica close (opcjonalnie, jeśli podano DataFrame)
            okres: Okres ATR (domyślnie 14)
            
        Returns:
            Tablica wartości ATR
        """
        try:
            if isinstance(df_or_high, pd.DataFrame):
                high = df_or_high['high'].values
                low = df_or_high['low'].values
                close = df_or_high['close'].values
            else:
                high = df_or_high
                if any(x is None for x in [low, close]):
                    raise ValueError("Jeśli nie podano DataFrame, wymagane są wszystkie tablice: high, low, close")

            if len(high) < 2:
                return np.zeros_like(high)
                
            # True Range
            tr = np.zeros_like(high)
            tr[0] = high[0] - low[0]  # Dla pierwszej świecy tylko zakres
            
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i],  # Zakres bieżącej świecy
                    abs(high[i] - close[i-1]),  # Różnica między high a poprzednim close
                    abs(low[i] - close[i-1])  # Różnica między low a poprzednim close
                )
            
            # ATR jako średnia krocząca z TR
            atr = np.zeros_like(tr)
            atr[0] = tr[0]  # Pierwsza wartość to TR
            
            # Jeśli mamy mniej świec niż okres, używamy dostępnych danych
            okres_efektywny = min(okres, len(tr))
            
            for i in range(1, len(tr)):
                if i < okres_efektywny:
                    # Średnia prosta dla pierwszych n świec
                    atr[i] = np.mean(tr[:i+1])
                else:
                    # Średnia wykładnicza dla kolejnych świec
                    atr[i] = (atr[i-1] * (okres_efektywny-1) + tr[i]) / okres_efektywny
            
            return atr
            
        except Exception as e:
            logger.error("❌ Błąd podczas obliczania ATR: %s", str(e))
            return np.zeros_like(high)  # Zwracamy same zera w przypadku błędu
            
    def wykryj_trend(self, 
                     ceny: np.ndarray,
                     okno: int = 20) -> Tuple[float, float]:
        """
        Wykrywa trend na podstawie regresji liniowej.
        
        Args:
            ceny: Tablica cen zamknięcia
            okno: Okno analizy (domyślnie 20 świec)
            
        Returns:
            Krotka (nachylenie, r_kwadrat)
        """
        try:
            if len(ceny) < okno:
                return 0.0, 0.0
                
            x = np.arange(okno)
            y = ceny[-okno:]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            return slope, r_squared
            
        except Exception as e:
            self.logger.error("❌ Błąd podczas wykrywania trendu: %s", str(e))
            return 0.0, 0.0
            
    def znajdz_poziomy(self,
                       high: np.ndarray,
                       low: np.ndarray,
                       min_odleglosc: int = 10,
                       min_dotkniec: int = 3) -> Tuple[List[float], List[float]]:
        """
        Znajduje poziomy wsparcia i oporu.
        
        Args:
            high: Tablica cen najwyższych
            low: Tablica cen najniższych
            min_odleglosc: Minimalna odległość między poziomami (w świecach)
            min_dotkniec: Minimalna liczba dotknięć poziomu
            
        Returns:
            Krotka (poziomy_wsparcia, poziomy_oporu)
        """
        try:
            # Szukamy lokalnych minimów i maksimów
            lokalne_min = []
            lokalne_max = []
            
            for i in range(1, len(high)-1):
                # Lokalne minimum
                if low[i] < low[i-1] and low[i] < low[i+1]:
                    lokalne_min.append((i, low[i]))
                    
                # Lokalne maksimum
                if high[i] > high[i-1] and high[i] > high[i+1]:
                    lokalne_max.append((i, high[i]))
                    
            # Grupowanie bliskich poziomów
            def grupuj_poziomy(punkty: List[Tuple[int, float]], 
                             tolerancja: float = 0.001) -> List[float]:
                if not punkty:
                    return []
                    
                grupy = []
                aktualna_grupa = [punkty[0]]
                
                for i in range(1, len(punkty)):
                    if (abs(punkty[i][1] - aktualna_grupa[0][1]) / aktualna_grupa[0][1] 
                        <= tolerancja):
                        aktualna_grupa.append(punkty[i])
                    else:
                        if len(aktualna_grupa) >= min_dotkniec:
                            grupy.append(sum(p[1] for p in aktualna_grupa) / len(aktualna_grupa))
                        aktualna_grupa = [punkty[i]]
                        
                if len(aktualna_grupa) >= min_dotkniec:
                    grupy.append(sum(p[1] for p in aktualna_grupa) / len(aktualna_grupa))
                    
                return grupy
                
            wsparcia = grupuj_poziomy(lokalne_min)
            opory = grupuj_poziomy(lokalne_max)
            
            # Filtrowanie zbyt bliskich poziomów
            def filtruj_bliskie(poziomy: List[float], 
                              min_odleglosc_proc: float = 0.01) -> List[float]:
                if not poziomy:
                    return []
                    
                poziomy = sorted(poziomy)
                wynik = [poziomy[0]]
                
                for p in poziomy[1:]:
                    if (p - wynik[-1]) / wynik[-1] >= min_odleglosc_proc:
                        wynik.append(p)
                        
                return wynik
                
            wsparcia = filtruj_bliskie(wsparcia)
            opory = filtruj_bliskie(opory)
            
            return wsparcia, opory
            
        except Exception as e:
            self.logger.error("❌ Błąd podczas znajdowania poziomów: %s", str(e))
            return [], []
            
    def oblicz_momentum(self, ceny: np.ndarray, okres: int = 14) -> np.ndarray:
        """
        Oblicza wskaźnik momentum.
        
        Args:
            ceny: Tablica cen zamknięcia
            okres: Okres momentum (domyślnie 14)
            
        Returns:
            np.ndarray: Wartości momentum
        """
        try:
            momentum = np.zeros_like(ceny)
            momentum[okres:] = ceny[okres:] - ceny[:-okres]
            momentum[:okres] = momentum[okres]  # Wypełnij początkowe wartości
            return momentum
        except Exception as e:
            self.logger.error(f"❌ Błąd obliczania momentum: {str(e)}")
            return np.zeros_like(ceny)
            
    def oblicz_volatility(self, ceny: np.ndarray, okres: int = 20) -> np.ndarray:
        """
        Oblicza zmienność cen.
        
        Args:
            ceny: Tablica cen zamknięcia
            okres: Okres zmienności (domyślnie 20)
            
        Returns:
            np.ndarray: Wartości zmienności
        """
        try:
            volatility = np.zeros_like(ceny)
            for i in range(okres, len(ceny)):
                volatility[i] = np.std(ceny[i-okres:i])
            volatility[:okres] = volatility[okres]  # Wypełnij początkowe wartości
            return volatility
        except Exception as e:
            self.logger.error(f"❌ Błąd obliczania zmienności: {str(e)}")
            return np.zeros_like(ceny)
            
    def oblicz_relative_volume(self, wolumen: np.ndarray, okres: int = 20) -> np.ndarray:
        """
        Oblicza względny wolumen.
        
        Args:
            wolumen: Tablica wolumenu
            okres: Okres średniej (domyślnie 20)
            
        Returns:
            np.ndarray: Wartości względnego wolumenu
        """
        try:
            avg_volume = np.zeros_like(wolumen)
            for i in range(okres, len(wolumen)):
                avg_volume[i] = np.mean(wolumen[i-okres:i])
            avg_volume[:okres] = avg_volume[okres]  # Wypełnij początkowe wartości
            
            relative_volume = np.zeros_like(wolumen)
            relative_volume[avg_volume != 0] = wolumen[avg_volume != 0] / avg_volume[avg_volume != 0]
            return relative_volume
        except Exception as e:
            self.logger.error(f"❌ Błąd obliczania względnego wolumenu: {str(e)}")
            return np.ones_like(wolumen)
            
    def oblicz_stochastic(self, df: pd.DataFrame, okres_k: int = 14, okres_d: int = 3, wygładzanie: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Oblicza oscylator Stochastic.

        Args:
            df (pd.DataFrame): DataFrame z danymi OHLCV
            okres_k (int, optional): Okres dla %K. Defaults to 14.
            okres_d (int, optional): Okres dla %D. Defaults to 3.
            wygładzanie (int, optional): Okres wygładzania. Defaults to 3.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Wartości %K i %D
        """
        try:
            stoch = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=okres_k,
                smooth_window=wygładzanie
            )
            stoch_k = stoch.stoch()
            stoch_d = stoch.stoch_signal()

            return stoch_k.values, stoch_d.values

        except Exception as e:
            logger.error(f"❌ Błąd obliczania Stochastic: {str(e)}")
            return np.zeros(len(df)), np.zeros(len(df))
            
    def oblicz_linie_livermore(self, df: pd.DataFrame, min_zmiana: float = 0.0001) -> Dict[str, Any]:
        """Analizuje linię najmniejszego oporu wg teorii Livermore'a.

        Args:
            df (pd.DataFrame): DataFrame z danymi OHLCV
            min_zmiana (float, optional): Minimalna zmiana procentowa. Defaults to 0.0001.

        Returns:
            Dict[str, Any]: Słownik z wynikami analizy zawierający:
                - kierunek: 1 (wzrost), -1 (spadek), 0 (trend boczny)
                - sila: wartość od 0 do 1 określająca siłę trendu
                - punkty_zwrotne: lista punktów zwrotnych
                - opor: najbliższy poziom oporu
                - wsparcie: najbliższy poziom wsparcia
        """
        try:
            if len(df) < 2:
                return {
                    'kierunek': 0,
                    'sila': 0.0,
                    'punkty_zwrotne': [],
                    'opor': df['high'].iloc[-1],
                    'wsparcie': df['low'].iloc[-1]
                }

            # Obliczamy nachylenie linii trendu
            x = np.arange(len(df))
            y = df['close'].values
            avg_price = np.mean(y)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            # Normalizujemy nachylenie względem średniej ceny
            norm_slope = slope / avg_price
            norm_slope_strength = np.clip(abs(norm_slope) / (min_zmiana * 10), 0, 1)
            
            # Obliczamy momentum i wolumen względny
            momentum = (y[-1] - y[0]) / y[0] if len(y) > 1 else 0
            norm_momentum = np.clip((momentum + 1) / 2, 0, 1)  # Normalizujemy do przedziału [0,1]
            
            volume = df['volume'].values
            avg_volume = np.mean(volume)
            rel_volume = volume[-1] / avg_volume if avg_volume > 0 else 1.0
            norm_volume = np.clip(rel_volume / 2, 0, 1)  # Normalizujemy do przedziału [0,1]
            
            # Obliczamy siłę trendu jako ważoną sumę składowych
            sila = 0.3 * norm_slope_strength + 0.3 * r_squared + 0.2 * norm_momentum + 0.2 * norm_volume
            sila = np.clip(sila, 0, 1)
            
            # Określamy kierunek trendu
            if r_squared < 0.3 or abs(norm_slope) < min_zmiana * 15:
                kierunek = 0  # Trend boczny
            else:
                kierunek = 1 if norm_slope > 0 else -1
            
            # Znajdujemy punkty zwrotne (lokalne maksima i minima)
            punkty_zwrotne = []
            for i in range(2, len(df) - 2):
                if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                    punkty_zwrotne.append(('szczyt', df.index[i], df['high'].iloc[i]))
                elif df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                    punkty_zwrotne.append(('dołek', df.index[i], df['low'].iloc[i]))
            
            # Znajdujemy najbliższe poziomy wsparcia i oporu
            if len(punkty_zwrotne) > 0:
                cena = df['close'].iloc[-1]
                opory = [p[2] for p in punkty_zwrotne if p[0] == 'szczyt' and p[2] > cena]
                wsparcia = [p[2] for p in punkty_zwrotne if p[0] == 'dołek' and p[2] < cena]
                
                opor = min(opory) if opory else df['high'].quantile(0.75)
                wsparcie = max(wsparcia) if wsparcia else df['low'].quantile(0.25)
            else:
                opor = df['high'].quantile(0.75)
                wsparcie = df['low'].quantile(0.25)
            
            return {
                'kierunek': kierunek,
                'sila': sila,
                'punkty_zwrotne': punkty_zwrotne,
                'opor': opor,
                'wsparcie': wsparcie
            }
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy Livermore'a: {str(e)}")
            return {
                'kierunek': 0,
                'sila': 0.0,
                'punkty_zwrotne': [],
                'opor': df['high'].iloc[-1],
                'wsparcie': df['low'].iloc[-1]
            } 