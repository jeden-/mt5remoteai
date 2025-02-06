"""
Testy jednostkowe dla modułu technical_indicators.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from src.utils.technical_indicators import TechnicalIndicators, IndicatorParams

@pytest.fixture
def sample_data():
    """Generuje przykładowe dane rynkowe do testów."""
    dates = pd.date_range(start='2025-01-29', periods=100, freq='h')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': np.random.normal(1.15, 0.03, 100),
        'high': np.random.normal(1.20, 0.03, 100),
        'low': np.random.normal(1.10, 0.03, 100),
        'close': np.random.normal(1.15, 0.03, 100),
        'volume': np.random.normal(3000, 1000, 100)
    }, index=dates)
    
    # Upewniamy się, że high jest zawsze większe od low
    data['high'] = data[['high', 'low']].max(axis=1) + 0.01
    data['low'] = data[['high', 'low']].min(axis=1)
    
    return data

@pytest.fixture
def indicators():
    """Tworzy instancję klasy TechnicalIndicators z domyślnymi parametrami."""
    return TechnicalIndicators()

def test_init_default_params():
    """Test inicjalizacji z domyślnymi parametrami."""
    ti = TechnicalIndicators()
    assert ti.params is not None
    assert ti.params.sma_periods['fast'] == 20
    assert ti.params.sma_periods['slow'] == 50

def test_init_custom_params():
    """Test inicjalizacji z własnymi parametrami."""
    custom_params = IndicatorParams(
        sma_periods={'fast': 10, 'slow': 30},
        rsi_period=10
    )
    ti = TechnicalIndicators(custom_params)
    assert ti.params.sma_periods['fast'] == 10
    assert ti.params.sma_periods['slow'] == 30
    assert ti.params.rsi_period == 10

def test_calculate_all_invalid_input():
    """Test obsługi nieprawidłowych danych wejściowych."""
    ti = TechnicalIndicators()
    with pytest.raises(ValueError):
        ti.calculate_all([1, 2, 3])  # Lista zamiast DataFrame

def test_calculate_all_missing_columns():
    """Test obsługi brakujących kolumn."""
    ti = TechnicalIndicators()
    df = pd.DataFrame({'close': [1, 2, 3]})
    with pytest.raises(KeyError):
        ti.calculate_all(df)

def test_calculate_all_empty_dataframe():
    """Test obsługi pustego DataFrame."""
    ti = TechnicalIndicators()
    df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    result = ti.calculate_all(df)
    assert len(result) == 0

def test_moving_averages(indicators, sample_data):
    """Test obliczania średnich ruchomych."""
    result = indicators.add_moving_averages(sample_data)
    assert 'SMA_20' in result.columns
    assert 'SMA_50' in result.columns
    assert 'EMA_12' in result.columns
    assert 'EMA_26' in result.columns

    # Sprawdzamy czy wartości są obliczone poprawnie
    sma20 = result['close'].rolling(window=20).mean()
    assert np.allclose(result['SMA_20'].dropna(), sma20.dropna(), rtol=1e-10, atol=0)

def test_rsi(indicators, sample_data):
    """Test obliczania RSI."""
    result = indicators.add_rsi(sample_data)
    assert 'RSI' in result.columns
    assert result['RSI'].min() >= 0
    assert result['RSI'].max() <= 100

def test_bollinger_bands(indicators, sample_data):
    """Test obliczania Bollinger Bands."""
    result = indicators.add_bollinger_bands(sample_data)
    assert 'BB_middle' in result.columns
    assert 'BB_upper' in result.columns
    assert 'BB_lower' in result.columns

    # Sprawdzamy czy górne pasmo jest zawsze wyższe niż środkowe
    assert (result['BB_upper'].dropna() >= result['BB_middle'].dropna()).all()
    # Sprawdzamy czy dolne pasmo jest zawsze niższe niż środkowe
    assert (result['BB_lower'].dropna() <= result['BB_middle'].dropna()).all()

def test_macd(indicators, sample_data):
    """Test obliczania MACD."""
    result = indicators.add_macd(sample_data)
    assert 'MACD' in result.columns
    assert 'MACD_Signal' in result.columns
    assert 'MACD_Histogram' in result.columns

    # Sprawdzamy czy histogram jest różnicą MACD i sygnału
    pd.testing.assert_series_equal(
        result['MACD_Histogram'].dropna(),
        (result['MACD'] - result['MACD_Signal']).dropna(),
        check_names=False
    )

def test_stochastic(indicators, sample_data):
    """Test obliczania oscylatora stochastycznego."""
    result = indicators.add_stochastic(sample_data)
    assert 'Stoch_K' in result.columns
    assert 'Stoch_D' in result.columns
    assert result['Stoch_K'].min() >= 0
    assert result['Stoch_K'].max() <= 100

def test_atr(indicators, sample_data):
    """Test obliczania Average True Range."""
    result = indicators.add_atr(sample_data)
    assert 'ATR' in result.columns
    assert (result['ATR'].dropna() >= 0).all()

def test_momentum(indicators, sample_data):
    """Test obliczania wskaźników momentum."""
    result = indicators.add_momentum(sample_data)
    assert 'ROC' in result.columns
    assert 'Momentum' in result.columns

def test_support_resistance(indicators, sample_data):
    """Test obliczania poziomów wsparcia i oporu."""
    # Test standardowego przypadku
    support, resistance = indicators.calculate_support_resistance(
        high=sample_data['high'],
        low=sample_data['low'],
        close=sample_data['close'],
        period=20
    )
    assert isinstance(support, float)
    assert isinstance(resistance, float)
    assert support <= resistance  # Poziom wsparcia powinien być niższy niż oporu
    
    # Test z nieprawidłowymi parametrami
    with pytest.raises(ValueError):
        indicators.calculate_support_resistance(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            period=0
        )

def test_detect_patterns(indicators, sample_data):
    """Test wykrywania formacji świecowych."""
    # Tworzymy dane testowe z konkretną formacją świecową
    test_data = pd.DataFrame({
        'open': [1.15, 1.16],
        'high': [1.17, 1.18],
        'low': [1.13, 1.14],
        'close': [1.14, 1.17],
        'volume': [3000, 3100]
    }, index=pd.date_range(start='2025-01-29', periods=2, freq='h'))
    
    result = indicators.detect_patterns(test_data)
    assert 'doji' in result
    assert 'hammer' in result
    assert 'shooting_star' in result
    assert 'engulfing_bullish' in result
    assert 'engulfing_bearish' in result
    assert all(isinstance(v, bool) for v in result.values())

def test_detect_divergence(indicators, sample_data):
    """Test wykrywania dywergencji."""
    # Najpierw dodajemy RSI do danych
    sample_data = indicators.add_rsi(sample_data)
    result = indicators.detect_divergence(sample_data)
    assert 'bullish_divergence' in result
    assert 'bearish_divergence' in result
    assert all(isinstance(v, bool) for v in result.values())

def test_calculate_sma(indicators, sample_data):
    """Test obliczania Simple Moving Average."""
    # Test standardowego przypadku
    sma = indicators.calculate_sma(sample_data['close'], period=20)
    assert isinstance(sma, pd.Series)
    assert len(sma) == len(sample_data)
    assert pd.isna(sma[:19]).all()  # Pierwsze 19 wartości powinny być NaN
    assert not pd.isna(sma[19:]).any()  # Pozostałe wartości nie powinny być NaN
    
    # Test z różnymi okresami
    sma_5 = indicators.calculate_sma(sample_data['close'], period=5)
    sma_10 = indicators.calculate_sma(sample_data['close'], period=10)
    assert pd.isna(sma_5[:4]).all()
    assert pd.isna(sma_10[:9]).all()
    
    # Test z nieprawidłowym okresem
    with pytest.raises(ValueError):
        indicators.calculate_sma(sample_data['close'], period=0)
    with pytest.raises(ValueError):
        indicators.calculate_sma(sample_data['close'], period=-1)

def test_calculate_ema(indicators, sample_data):
    """Test obliczania Exponential Moving Average."""
    # Test standardowego przypadku
    ema = indicators.calculate_ema(sample_data['close'], period=20)
    assert isinstance(ema, pd.Series)
    assert len(ema) == len(sample_data)
    assert pd.isna(ema[:19]).all()
    assert not pd.isna(ema[19:]).any()
    
    # Test z różnymi okresami
    ema_5 = indicators.calculate_ema(sample_data['close'], period=5)
    ema_10 = indicators.calculate_ema(sample_data['close'], period=10)
    assert pd.isna(ema_5[:4]).all()
    assert pd.isna(ema_10[:9]).all()
    
    # Test z nieprawidłowym okresem
    with pytest.raises(ValueError):
        indicators.calculate_ema(sample_data['close'], period=0)

def test_calculate_rsi(indicators, sample_data):
    """Test obliczania Relative Strength Index."""
    # Test standardowego przypadku
    rsi = indicators.calculate_rsi(sample_data['close'], period=14)
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(sample_data)
    assert pd.isna(rsi[:14]).all()
    assert not pd.isna(rsi[14:]).any()
    
    # Sprawdź zakres wartości (RSI zawsze między 0 a 100)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    # Test z różnymi okresami
    rsi_7 = indicators.calculate_rsi(sample_data['close'], period=7)
    rsi_21 = indicators.calculate_rsi(sample_data['close'], period=21)
    assert pd.isna(rsi_7[:7]).all()
    assert pd.isna(rsi_21[:21]).all()
    
    # Test z nieprawidłowym okresem
    with pytest.raises(ValueError):
        indicators.calculate_rsi(sample_data['close'], period=0)

def test_calculate_macd(indicators, sample_data):
    """Test obliczania MACD."""
    # Test standardowego przypadku
    macd, signal, hist = indicators.calculate_macd(
        sample_data['close'],
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert isinstance(hist, pd.Series)
    assert len(macd) == len(sample_data)
    assert len(signal) == len(sample_data)
    assert len(hist) == len(sample_data)
    
    # Sprawdź czy histogram to różnica między MACD a linią sygnału
    pd.testing.assert_series_equal(hist, macd - signal, check_names=False)
    
    # Test z nieprawidłowymi parametrami
    with pytest.raises(ValueError):
        indicators.calculate_macd(sample_data['close'], fast_period=26, slow_period=12)

def test_calculate_bollinger_bands(indicators, sample_data):
    """Test obliczania Bollinger Bands."""
    # Test standardowego przypadku
    upper, middle, lower = indicators.calculate_bollinger_bands(
        sample_data['close'],
        period=20,
        std_dev=2
    )
    
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    assert len(upper) == len(sample_data)
    
    # Sprawdź relacje między pasmami
    valid_data = pd.concat([upper, middle, lower], axis=1).dropna()
    assert (valid_data.iloc[:,0] >= valid_data.iloc[:,1]).all()  # upper >= middle
    assert (valid_data.iloc[:,1] >= valid_data.iloc[:,2]).all()  # middle >= lower
    
    # Test z różnymi odchyleniami standardowymi
    upper_1, middle_1, lower_1 = indicators.calculate_bollinger_bands(
        sample_data['close'],
        period=20,
        std_dev=1
    )
    
    # Porównujemy tylko wartości niezerowe
    valid_mask = ~pd.isna(upper) & ~pd.isna(upper_1)
    assert (upper[valid_mask] >= upper_1[valid_mask]).all()
    assert (lower[valid_mask] <= lower_1[valid_mask]).all()
    
    # Test z nieprawidłowymi parametrami
    with pytest.raises(ValueError):
        indicators.calculate_bollinger_bands(sample_data['close'], period=0)
    with pytest.raises(ValueError):
        indicators.calculate_bollinger_bands(sample_data['close'], std_dev=0)

def test_calculate_stochastic(indicators, sample_data):
    """Test obliczania oscylatora stochastycznego."""
    # Test standardowego przypadku
    k, d = indicators.calculate_stochastic(
        high=sample_data['high'],
        low=sample_data['low'],
        close=sample_data['close'],
        k_period=14,
        d_period=3
    )
    
    assert isinstance(k, pd.Series)
    assert isinstance(d, pd.Series)
    assert len(k) == len(sample_data)
    assert len(d) == len(sample_data)
    
    # Sprawdź zakres wartości (zawsze między 0 a 100)
    valid_k = k.dropna()
    valid_d = d.dropna()
    assert (valid_k >= 0).all() and (valid_k <= 100).all()
    assert (valid_d >= 0).all() and (valid_d <= 100).all()
    
    # Test z nieprawidłowymi parametrami
    with pytest.raises(ValueError):
        indicators.calculate_stochastic(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            k_period=0
        )

def test_calculate_atr(indicators, sample_data):
    """Test obliczania Average True Range."""
    # Test standardowego przypadku
    atr = indicators.calculate_atr(
        high=sample_data['high'],
        low=sample_data['low'],
        close=sample_data['close'],
        period=14
    )
    
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(sample_data)
    assert pd.isna(atr[:14]).all()
    assert not pd.isna(atr[14:]).any()
    
    # ATR powinien być zawsze dodatni
    valid_atr = atr.dropna()
    assert (valid_atr >= 0).all()
    
    # Test z nieprawidłowymi parametrami
    with pytest.raises(ValueError):
        indicators.calculate_atr(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            period=0
        )

def test_calculate_adx(indicators, sample_data):
    """Test obliczania Average Directional Index."""
    # Test standardowego przypadku
    adx = indicators.calculate_adx(
        high=sample_data['high'],
        low=sample_data['low'],
        close=sample_data['close'],
        period=14
    )
    
    assert isinstance(adx, pd.Series)
    assert len(adx) == len(sample_data)
    
    # ADX powinien być między 0 a 100
    valid_adx = adx.dropna()
    assert (valid_adx >= 0).all() and (valid_adx <= 100).all()
    
    # Test z nieprawidłowymi parametrami
    with pytest.raises(ValueError):
        indicators.calculate_adx(
            high=sample_data['high'],
            low=sample_data['low'],
            close=sample_data['close'],
            period=0
        )

def test_calculate_pivot_points(indicators, sample_data):
    """Test obliczania punktów pivota."""
    # Test standardowego przypadku
    pp, r1, r2, s1, s2 = indicators.calculate_pivot_points(
        high=sample_data['high'].iloc[-1],
        low=sample_data['low'].iloc[-1],
        close=sample_data['close'].iloc[-1]
    )
    
    assert isinstance(pp, float)
    assert isinstance(r1, float)
    assert isinstance(r2, float)
    assert isinstance(s1, float)
    assert isinstance(s2, float)
    
    # Sprawdź relacje między poziomami
    assert s2 < s1 < pp < r1 < r2
    
    # Test z nieprawidłowymi danymi
    with pytest.raises(ValueError):
        indicators.calculate_pivot_points(
            high=np.nan,
            low=sample_data['low'].iloc[-1],
            close=sample_data['close'].iloc[-1]
        ) 

def test_detect_patterns_detailed(indicators):
    """Test szczegółowy wykrywania formacji świecowych."""
    # Test Doji - bardzo mały korpus w porównaniu do cieni
    doji_data = pd.DataFrame({
        'open': [1.1500, 1.1501],
        'high': [1.1520, 1.1520],
        'low': [1.1480, 1.1480],
        'close': [1.1501, 1.1500],  # Korpus 0.0001 vs cienie 0.004
        'volume': [1000, 1000]
    }, index=pd.date_range(start='2025-01-29', periods=2, freq='h'))
    
    patterns = indicators.detect_patterns(doji_data)
    assert patterns['doji'] == True
    
    # Test Hammer - długi dolny cień, mały korpus i bardzo mały górny cień
    hammer_data = pd.DataFrame({
        'open': [1.1500, 1.1500],
        'high': [1.1520, 1.1510],  # Mały górny cień
        'low': [1.1400, 1.1400],   # Długi dolny cień
        'close': [1.1510, 1.1505], # Mały korpus
        'volume': [1000, 1000]
    }, index=pd.date_range(start='2025-01-29', periods=2, freq='h'))
    
    patterns = indicators.detect_patterns(hammer_data)
    assert patterns['hammer'] == True
    
    # Test Shooting Star - długi górny cień, mały korpus i bardzo mały dolny cień
    shooting_star_data = pd.DataFrame({
        'open': [1.1500, 1.1520],  # Korpus = 0.002 (1.1520 - 1.1500)
        'high': [1.1520, 1.1600],  # Górny cień = 0.008 (1.1600 - 1.1520) > 2 * korpus
        'low': [1.1480, 1.1499],   # Dolny cień = 0.001 (1.1500 - 1.1499) < korpus
        'close': [1.1510, 1.1500], # Świeca spadkowa
        'volume': [1000, 1000]
    }, index=pd.date_range(start='2025-01-29', periods=2, freq='h'))
    
    patterns = indicators.detect_patterns(shooting_star_data)
    assert patterns['shooting_star'] == True
    
    # Test Bullish Engulfing - druga świeca całkowicie pochłania pierwszą
    bullish_engulfing_data = pd.DataFrame({
        'open': [1.1520, 1.1480],  # Druga świeca otwiera się niżej
        'high': [1.1530, 1.1540],
        'low': [1.1490, 1.1470],
        'close': [1.1500, 1.1530], # Druga świeca zamyka się wyżej
        'volume': [1000, 1000]
    }, index=pd.date_range(start='2025-01-29', periods=2, freq='h'))
    
    patterns = indicators.detect_patterns(bullish_engulfing_data)
    assert patterns['engulfing_bullish'] == True
    
    # Test Bearish Engulfing - druga świeca całkowicie pochłania pierwszą
    bearish_engulfing_data = pd.DataFrame({
        'open': [1.1480, 1.1520],  # Druga świeca otwiera się wyżej
        'high': [1.1500, 1.1530],
        'low': [1.1470, 1.1460],
        'close': [1.1490, 1.1470], # Druga świeca zamyka się niżej
        'volume': [1000, 1000]
    }, index=pd.date_range(start='2025-01-29', periods=2, freq='h'))
    
    patterns = indicators.detect_patterns(bearish_engulfing_data)
    assert patterns['engulfing_bearish'] == True

def test_detect_patterns_validation(indicators):
    """Test walidacji danych wejściowych dla wykrywania wzorców."""
    # Test nieprawidłowego typu danych
    with pytest.raises(ValueError, match="Dane wejściowe muszą być typu pandas DataFrame"):
        indicators.detect_patterns([1, 2, 3])
    
    # Test nieprawidłowego okna
    with pytest.raises(ValueError, match="Okno musi być większe od 0"):
        indicators.detect_patterns(pd.DataFrame(), window=0)
    
    # Test brakujących kolumn
    with pytest.raises(KeyError, match="DataFrame musi zawierać kolumny"):
        indicators.detect_patterns(pd.DataFrame({'close': [1, 2, 3]}))

def test_detect_patterns_edge_cases(indicators):
    """Test przypadków brzegowych dla wykrywania wzorców."""
    # Test z jedną świecą
    single_candle = pd.DataFrame({
        'open': [1.1500],
        'high': [1.1520],
        'low': [1.1480],
        'close': [1.1510],
        'volume': [1000]
    }, index=pd.date_range(start='2025-01-29', periods=1, freq='h'))
    
    patterns = indicators.detect_patterns(single_candle)
    assert isinstance(patterns, dict)
    assert len(patterns) == 5
    assert not any(patterns.values())  # Żaden wzorzec nie powinien być wykryty
    
    # Test z identycznymi cenami
    flat_data = pd.DataFrame({
        'open': [1.1500, 1.1500],
        'high': [1.1500, 1.1500],
        'low': [1.1500, 1.1500],
        'close': [1.1500, 1.1500],
        'volume': [1000, 1000]
    }, index=pd.date_range(start='2025-01-29', periods=2, freq='h'))
    
    patterns = indicators.detect_patterns(flat_data)
    assert patterns['doji'] == True
    assert not any(v for k, v in patterns.items() if k != 'doji') 

def test_input_validation_edge_cases(indicators):
    """Test walidacji danych wejściowych dla skrajnych przypadków."""
    # Test dla pustego DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        indicators.calculate_all(empty_df)
    
    # Test dla DataFrame bez wymaganych kolumn
    invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    with pytest.raises(KeyError):
        indicators.calculate_all(invalid_df)
    
    # Test dla DataFrame z nieprawidłowymi typami danych
    invalid_types_df = pd.DataFrame({
        'open': ['invalid', 'data'],
        'high': [1.1, 1.2],
        'low': [1.0, 1.1],
        'close': [1.1, 1.2],
        'volume': [1000, 1000]
    })
    with pytest.raises(ValueError):
        indicators.calculate_sma(pd.Series(['invalid', 'data']), period=5)

def test_pivot_points_detailed(indicators):
    """Test szczegółowy obliczania punktów pivot."""
    # Przygotowanie danych testowych
    high = 1.1580
    low = 1.1450
    close = 1.1500
    
    # Obliczenie punktów pivot
    pp, r1, r2, s1, s2 = indicators.calculate_pivot_points(high, low, close)
    
    # Sprawdzenie czy wszystkie poziomy są obliczone
    assert isinstance(pp, float)
    assert isinstance(r1, float)
    assert isinstance(r2, float)
    assert isinstance(s1, float)
    assert isinstance(s2, float)
    
    # Sprawdzenie poprawności obliczeń
    expected_pp = (high + low + close) / 3
    assert abs(pp - expected_pp) < 0.0001
    
    # Sprawdzenie relacji między poziomami
    assert s2 < s1 < pp < r1 < r2

def test_divergence_detection(indicators):
    """Test wykrywania dywergencji."""
    # Przygotowanie danych testowych dla dywergencji byczej
    bullish_data = pd.DataFrame({
        'open': [1.1500, 1.1480, 1.1460, 1.1440, 1.1420],
        'high': [1.1520, 1.1500, 1.1480, 1.1460, 1.1440],
        'low': [1.1480, 1.1460, 1.1440, 1.1420, 1.1400],
        'close': [1.1510, 1.1470, 1.1450, 1.1430, 1.1410],
        'volume': [1000] * 5
    }, index=pd.date_range(start='2025-01-29', periods=5, freq='h'))
    
    # Obliczenie RSI dla danych
    rsi_values = indicators.calculate_rsi(bullish_data['close'], period=14)
    bullish_data['RSI'] = rsi_values
    
    # Test wykrywania dywergencji
    divergence = indicators.detect_divergence(
        df=bullish_data,
        price_col='close',
        indicator_col='RSI',
        window=5
    )
    
    assert isinstance(divergence, dict)
    assert 'bullish_divergence' in divergence
    assert 'bearish_divergence' in divergence
    
    # Test dla nieprawidłowych danych wejściowych
    with pytest.raises(ValueError):
        indicators.detect_divergence(
            df=pd.DataFrame(),
            price_col='close',
            indicator_col='RSI',
            window=5
        )
    
    # Test dla brakujących kolumn
    with pytest.raises(KeyError):
        indicators.detect_divergence(
            df=pd.DataFrame({'wrong_col': [1, 2, 3]}),
            price_col='close',
            indicator_col='RSI',
            window=5
        )

def test_helper_functions_edge_cases(indicators):
    """Test funkcji pomocniczych dla skrajnych przypadków."""
    # Test dla pustej serii
    empty_series = pd.Series([])
    with pytest.raises(ValueError):
        indicators.calculate_sma(empty_series, period=5)
    
    # Test dla serii z wartościami NaN
    nan_series = pd.Series([1.0, np.nan, 3.0])
    result = indicators.calculate_sma(nan_series, period=2)
    assert pd.isna(result.iloc[0])
    
    # Test dla nieprawidłowego okresu
    data = pd.Series([1.1550, 1.1500])
    
    with pytest.raises(ValueError):
        indicators.calculate_sma(data, period=0)
    
    with pytest.raises(ValueError):
        indicators.calculate_sma(data, period=-1)
    
    with pytest.raises(ValueError):
        indicators.calculate_sma(data, period=len(data) + 1) 