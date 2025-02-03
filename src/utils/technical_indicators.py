import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Tuple, List
from dataclasses import dataclass, field

@dataclass
class IndicatorParams:
    """Parametry dla wskaźników technicznych"""
    sma_periods: Dict[str, int] = field(default_factory=lambda: {'fast': 20, 'slow': 50})
    ema_periods: Dict[str, int] = field(default_factory=lambda: {'fast': 12, 'slow': 26})
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    macd_params: Dict[str, int] = field(default_factory=lambda: {
        'fast': 12,
        'slow': 26,
        'signal': 9
    })
    stoch_params: Dict[str, int] = field(default_factory=lambda: {
        'k_period': 14,
        'k_smooth': 3,
        'd_period': 3
    })

class TechnicalIndicators:
    """Klasa do obliczania wskaźników technicznych"""
    
    def __init__(self, params: Optional[IndicatorParams] = None):
        self.params = params or IndicatorParams()
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza wszystkie wskaźniki techniczne dla danego DataFrame"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Dane wejściowe muszą być typu pandas DataFrame")
        
        if len(df) == 0:
            return df.copy()
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise KeyError(f"DataFrame musi zawierać kolumny: {required_columns}")

        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_bollinger_bands(df)
        df = self.add_macd(df)
        df = self.add_stochastic(df)
        df = self.add_atr(df)
        df = self.add_momentum(df)
        return df

    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Oblicza Simple Moving Average dla danej serii danych."""
        if not isinstance(series, pd.Series):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if period <= 0:
            raise ValueError("Okres musi być większy od 0")
        return series.rolling(window=period, min_periods=period).mean()

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Oblicza Exponential Moving Average dla danej serii danych."""
        if not isinstance(series, pd.Series):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if period <= 0:
            raise ValueError("Okres musi być większy od 0")
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje różne średnie ruchome"""
        df = df.copy()
        
        # SMA
        for name, period in self.params.sma_periods.items():
            df[f'SMA_{period}'] = self.calculate_sma(df['close'], period)
        
        # EMA
        for name, period in self.params.ema_periods.items():
            df[f'EMA_{period}'] = self.calculate_ema(df['close'], period)
        
        return df

    def calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Oblicza Relative Strength Index dla danej serii danych."""
        if not isinstance(series, pd.Series):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if period <= 0:
            raise ValueError("Okres musi być większy od 0")
            
        delta = series.diff()
        delta = delta[1:]  # Usuwamy pierwszą wartość (NaN)
        
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Dodajemy NaN na początku dla zachowania długości serii
        return pd.concat([pd.Series([np.nan]), rsi])

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje wskaźnik RSI"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['close'], self.params.rsi_period)
        return df

    def calculate_bollinger_bands(
        self, 
        series: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Oblicza Bollinger Bands dla danej serii danych."""
        if not isinstance(series, pd.Series):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if period <= 0:
            raise ValueError("Okres musi być większy od 0")
        if std_dev <= 0:
            raise ValueError("Odchylenie standardowe musi być większe od 0")
            
        # Obliczamy SMA i odchylenie standardowe
        middle = series.rolling(window=period, min_periods=period).mean()
        rolling_std = series.rolling(window=period, min_periods=period).std()
        
        # Upewniamy się, że pierwsze period-1 wartości to NaN
        middle.iloc[:period-1] = np.nan
        rolling_std.iloc[:period-1] = np.nan
        
        # Obliczamy górne i dolne pasmo
        deviation = rolling_std * std_dev
        upper = middle + deviation
        lower = middle - deviation
        
        # Upewniamy się, że wszystkie wartości są typu float
        upper = upper.astype(float)
        middle = middle.astype(float)
        lower = lower.astype(float)
        
        # Zachowujemy nazwy kolumn
        upper.name = series.name
        middle.name = series.name
        lower.name = series.name
        
        # Upewniamy się, że NaN są zachowane
        mask = pd.isna(series) | pd.isna(middle) | pd.isna(rolling_std)
        upper[mask] = np.nan
        middle[mask] = np.nan
        lower[mask] = np.nan
        
        return upper, middle, lower

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje Bollinger Bands"""
        df = df.copy()
        upper, middle, lower = self.calculate_bollinger_bands(
            df['close'], 
            self.params.bb_period, 
            self.params.bb_std
        )
        df['BB_middle'] = middle
        df['BB_upper'] = upper
        df['BB_lower'] = lower
        return df

    def calculate_macd(
        self, 
        series: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Oblicza MACD dla danej serii danych."""
        if not isinstance(series, pd.Series):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("Okresy muszą być większe od 0")
        if fast_period >= slow_period:
            raise ValueError("Okres szybki musi być mniejszy niż okres wolny")
            
        exp1 = self.calculate_ema(series, fast_period)
        exp2 = self.calculate_ema(series, slow_period)
        macd = exp1 - exp2
        signal = self.calculate_ema(macd, signal_period)
        hist = macd - signal
        return macd, signal, hist

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje MACD"""
        df = df.copy()
        macd, signal, hist = self.calculate_macd(
            df['close'],
            self.params.macd_params['fast'],
            self.params.macd_params['slow'],
            self.params.macd_params['signal']
        )
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = hist
        return df

    def calculate_stochastic(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Oblicza oscylator stochastyczny dla danych serii."""
        if not all(isinstance(x, pd.Series) for x in [high, low, close]):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if k_period <= 0 or d_period <= 0:
            raise ValueError("Okresy muszą być większe od 0")
            
        low_min = low.rolling(window=k_period, min_periods=k_period).min()
        high_max = high.rolling(window=k_period, min_periods=k_period).max()
        
        k = 100 * (close - low_min) / (high_max - low_min)
        k = k.rolling(window=d_period, min_periods=d_period).mean()
        d = k.rolling(window=d_period, min_periods=d_period).mean()
        return k, d

    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje oscylator stochastyczny"""
        df = df.copy()
        k, d = self.calculate_stochastic(
            df['high'],
            df['low'],
            df['close'],
            self.params.stoch_params['k_period'],
            self.params.stoch_params['d_period']
        )
        df['Stoch_K'] = k
        df['Stoch_D'] = d
        return df

    def calculate_atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Oblicza Average True Range dla danych serii."""
        if not all(isinstance(x, pd.Series) for x in [high, low, close]):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if period <= 0:
            raise ValueError("Okres musi być większy od 0")
            
        # Obliczamy True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Obliczamy ATR
        atr = pd.Series(index=true_range.index, dtype=float)
        
        # Pierwsze period wartości to NaN
        atr.iloc[:period] = np.nan
        
        if len(true_range) > period:
            # Pierwsza wartość ATR (dla indeksu period) to średnia TR z pierwszych period wartości
            atr.iloc[period] = true_range.iloc[:period].mean()
            
            # Kolejne wartości obliczamy według wzoru: ATR = ((period-1) * prev_ATR + TR) / period
            for i in range(period + 1, len(true_range)):
                atr.iloc[i] = ((period-1) * atr.iloc[i-1] + true_range.iloc[i]) / period
        
        # Zachowujemy nazwę kolumny
        atr.name = 'ATR'
        
        return atr

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Dodaje Average True Range"""
        df = df.copy()
        df['ATR'] = self.calculate_atr(
            df['high'],
            df['low'],
            df['close'],
            period
        )
        return df

    def calculate_adx(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Oblicza Average Directional Index dla danych serii."""
        if not all(isinstance(x, pd.Series) for x in [high, low, close]):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if period <= 0:
            raise ValueError("Okres musi być większy od 0")
            
        # True Range
        tr = self.calculate_atr(high, low, close, period)
        
        # Plus Directional Movement (+DM)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        
        # Minus Directional Movement (-DM)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smoothed +DM and -DM
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / tr)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period, min_periods=period).mean()
        
        return adx

    def add_momentum(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Dodaje wskaźniki momentum"""
        df = df.copy()
        
        # ROC (Rate of Change)
        df['ROC'] = ((df['close'] - df['close'].shift(period)) / 
                     df['close'].shift(period)) * 100
        
        # Momentum
        df['Momentum'] = df['close'] - df['close'].shift(period)
        
        return df

    def calculate_support_resistance(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> Tuple[float, float]:
        """Oblicza poziomy wsparcia i oporu"""
        if not all(isinstance(x, pd.Series) for x in [high, low, close]):
            raise ValueError("Dane wejściowe muszą być typu pandas Series")
        if period <= 0:
            raise ValueError("Okres musi być większy od 0")
            
        recent_high = high.tail(period)
        recent_low = low.tail(period)
        recent_close = close.tail(period)
        
        # Metoda 1: Pivot Points
        pivot = (recent_high.iloc[-1] + recent_low.iloc[-1] + recent_close.iloc[-1]) / 3
        s1 = 2 * pivot - recent_high.iloc[-1]
        r1 = 2 * pivot - recent_low.iloc[-1]
        
        # Metoda 2: Local extrema
        local_min = recent_low.min()
        local_max = recent_high.max()
        
        # Zwracamy średnią z obu metod
        support = float((s1 + local_min) / 2)
        resistance = float((r1 + local_max) / 2)
        
        return support, resistance

    def calculate_pivot_points(
        self, 
        high: float, 
        low: float, 
        close: float
    ) -> Tuple[float, float, float, float, float]:
        """Oblicza punkty pivota."""
        if any(pd.isna([high, low, close])):
            raise ValueError("Wartości wejściowe nie mogą być NaN")
        if high < low:
            raise ValueError("Wartość high nie może być mniejsza niż low")
            
        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        s1 = (2 * pp) - high
        s2 = pp - (high - low)
        return pp, r1, r2, s1, s2

    @staticmethod
    def detect_divergence(
        df: pd.DataFrame, 
        price_col: str = 'close', 
        indicator_col: str = 'RSI', 
        window: int = 20
    ) -> Dict[str, bool]:
        """Wykrywa dywergencje między ceną a wskaźnikiem"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Dane wejściowe muszą być typu pandas DataFrame")
        if window <= 0:
            raise ValueError("Okno musi być większe od 0")
        if price_col not in df.columns or indicator_col not in df.columns:
            raise KeyError(f"DataFrame musi zawierać kolumny: {price_col} i {indicator_col}")
            
        recent_data = df.tail(window)
        
        price_trend = recent_data[price_col].iloc[-1] > recent_data[price_col].iloc[0]
        indicator_trend = recent_data[indicator_col].iloc[-1] > recent_data[indicator_col].iloc[0]
        
        return {
            'bullish_divergence': not price_trend and indicator_trend,
            'bearish_divergence': price_trend and not indicator_trend
        }

    @staticmethod
    def detect_patterns(df: pd.DataFrame, window: int = 5) -> Dict[str, bool]:
        """Wykrywa podstawowe formacje świecowe"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Dane wejściowe muszą być typu pandas DataFrame")
        if window <= 0:
            raise ValueError("Okno musi być większe od 0")
            
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise KeyError(f"DataFrame musi zawierać kolumny: {required_columns}")
            
        recent = df.tail(window)
        
        patterns = {
            'doji': False,
            'hammer': False,
            'shooting_star': False,
            'engulfing_bullish': False,
            'engulfing_bearish': False
        }
        
        # Doji
        latest = recent.iloc[-1]
        body_size = abs(latest['open'] - latest['close'])
        wick_size = latest['high'] - latest['low']
        patterns['doji'] = bool(body_size <= (wick_size * 0.1))
        
        # Hammer
        if latest['low'] < latest['open'] and latest['low'] < latest['close']:
            lower_wick = min(latest['open'], latest['close']) - latest['low']
            upper_wick = latest['high'] - max(latest['open'], latest['close'])
            patterns['hammer'] = bool(lower_wick > (body_size * 2) and upper_wick < body_size)
            
        # Shooting Star
        if latest['high'] > latest['open'] and latest['high'] > latest['close']:
            upper_wick = latest['high'] - max(latest['open'], latest['close'])
            lower_wick = min(latest['open'], latest['close']) - latest['low']
            patterns['shooting_star'] = bool(upper_wick > (body_size * 2) and lower_wick < body_size)
            
        # Engulfing
        if len(recent) >= 2:
            prev = recent.iloc[-2]
            curr = recent.iloc[-1]
            
            patterns['engulfing_bullish'] = bool(
                prev['close'] < prev['open'] and  # Previous red candle
                curr['close'] > curr['open'] and  # Current green candle
                curr['open'] < prev['close'] and  # Opens below previous close
                curr['close'] > prev['open']      # Closes above previous open
            )
            
            patterns['engulfing_bearish'] = bool(
                prev['close'] > prev['open'] and  # Previous green candle
                curr['close'] < curr['open'] and  # Current red candle
                curr['open'] > prev['close'] and  # Opens above previous close
                curr['close'] < prev['open']      # Closes below previous open
            )
            
        return patterns 

def calculate_rsi(data: Union[pd.Series, List[float]], periods: int = 14) -> pd.Series:
    """
    Oblicza wskaźnik RSI (Relative Strength Index).

    Args:
        data: Seria danych cenowych
        periods: Liczba okresów do obliczenia RSI

    Returns:
        Seria z wartościami RSI
    """
    if isinstance(data, list):
        data = pd.Series(data)
        
    # Oblicz zmiany cen
    delta = data.diff()
    
    # Rozdziel zmiany na dodatnie i ujemne
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Oblicz średnie kroczące
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    # Oblicz RS i RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data: Union[pd.Series, List[float]], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Oblicza wskaźnik MACD (Moving Average Convergence Divergence).

    Args:
        data: Seria danych cenowych
        fast: Okres szybkiej średniej
        slow: Okres wolnej średniej
        signal: Okres linii sygnału

    Returns:
        Dict z seriami MACD, sygnału i histogramu
    """
    if isinstance(data, list):
        data = pd.Series(data)
        
    # Oblicz średnie wykładnicze
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    
    # Oblicz MACD i sygnał
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    # Oblicz histogram
    hist = macd - signal_line
    
    return {
        'macd': macd,
        'signal': signal_line,
        'hist': hist
    }

def calculate_bollinger_bands(data: Union[pd.Series, List[float]], window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
    """
    Oblicza wstęgi Bollingera.

    Args:
        data: Seria danych cenowych
        window: Okres średniej kroczącej
        num_std: Liczba odchyleń standardowych

    Returns:
        Dict z seriami górnej, środkowej i dolnej wstęgi
    """
    if isinstance(data, list):
        data = pd.Series(data)
        
    # Oblicz średnią kroczącą
    middle = data.rolling(window=window).mean()
    
    # Oblicz odchylenie standardowe
    std = data.rolling(window=window).std()
    
    # Oblicz górną i dolną wstęgę
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    } 