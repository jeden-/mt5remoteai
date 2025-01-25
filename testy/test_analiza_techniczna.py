"""
Testy modułu analizy technicznej.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from handel.analiza_techniczna import AnalizaTechniczna

@pytest.fixture
def przykladowe_dane():
    """Fixture dostarczający przykładowe dane do testów."""
    return {
        'close': np.array([100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5]),
        'high': np.array([101.0, 102.0, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5]),
        'low': np.array([99.0, 100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5]),
        'volume': np.array([1000, 1100, 1200, 1150, 1300, 1250, 1400, 1350, 1500, 1450])
    }

@pytest.fixture
def analiza():
    """Fixture dostarczający instancję AnalizaTechniczna."""
    return AnalizaTechniczna()

def test_oblicz_rsi(analiza, przykladowe_dane):
    """Test obliczania RSI."""
    rsi = analiza.oblicz_rsi(przykladowe_dane['close'])
    assert isinstance(rsi, np.ndarray)
    assert len(rsi) == len(przykladowe_dane['close'])
    assert not np.all(np.isnan(rsi))  # Sprawdź czy nie wszystkie wartości to NaN

def test_oblicz_rsi_blad(analiza):
    """Test błędu podczas obliczania RSI."""
    with patch('ta.momentum.RSIIndicator', side_effect=Exception("Test error")):
        rsi = analiza.oblicz_rsi(np.array([]))
        assert len(rsi) == 0

def test_oblicz_macd(analiza, przykladowe_dane):
    """Test obliczania MACD."""
    macd, signal, hist = analiza.oblicz_macd(przykladowe_dane['close'])
    assert all(isinstance(x, np.ndarray) for x in [macd, signal, hist])
    assert all(len(x) == len(przykladowe_dane['close']) for x in [macd, signal, hist])
    assert not np.all(np.isnan(macd))  # Sprawdź czy nie wszystkie wartości to NaN

def test_oblicz_macd_blad(analiza):
    """Test błędu podczas obliczania MACD."""
    with patch('ta.trend.MACD', side_effect=Exception("Test error")):
        macd, signal, hist = analiza.oblicz_macd(np.array([]))
        assert all(len(x) == 0 for x in [macd, signal, hist])

def test_oblicz_sma(analiza, przykladowe_dane):
    """Test obliczania SMA."""
    sma = analiza.oblicz_sma(przykladowe_dane['close'])
    assert isinstance(sma, np.ndarray)
    assert len(sma) == len(przykladowe_dane['close'])
    assert not np.all(np.isnan(sma))  # Sprawdź czy nie wszystkie wartości to NaN

def test_oblicz_sma_blad(analiza):
    """Test błędu podczas obliczania SMA."""
    with patch('ta.trend.SMAIndicator', side_effect=Exception("Test error")):
        sma = analiza.oblicz_sma(np.array([]))
        assert len(sma) == 0

def test_oblicz_ema(analiza, przykladowe_dane):
    """Test obliczania EMA."""
    ema = analiza.oblicz_ema(przykladowe_dane['close'])
    assert isinstance(ema, np.ndarray)
    assert len(ema) == len(przykladowe_dane['close'])
    assert not np.all(np.isnan(ema))  # Sprawdź czy nie wszystkie wartości to NaN

def test_oblicz_ema_blad(analiza):
    """Test błędu podczas obliczania EMA."""
    with patch('ta.trend.EMAIndicator', side_effect=Exception("Test error")):
        ema = analiza.oblicz_ema(np.array([]))
        assert len(ema) == 0

def test_wykryj_formacje_swiecowe(analiza, przykladowe_dane):
    """Test wykrywania formacji świecowych."""
    formacje = analiza.wykryj_formacje_swiecowe(
        przykladowe_dane['close'],
        przykladowe_dane['high'],
        przykladowe_dane['low']
    )
    assert isinstance(formacje, dict)
    assert len(formacje) > 0
    assert all(isinstance(v, np.ndarray) for v in formacje.values())
    assert all(len(v) == len(przykladowe_dane['close']) for v in formacje.values())
    assert all(np.all((v == 0) | (v == 1)) for v in formacje.values())  # Wartości binarne (0 lub 1)

def test_wykryj_formacje_swiecowe_blad(analiza):
    """Test błędu podczas wykrywania formacji świecowych."""
    formacje = analiza.wykryj_formacje_swiecowe(
        np.array([]),
        np.array([]),
        np.array([])
    )
    assert len(formacje) == 0

def test_analizuj_wolumen(analiza, przykladowe_dane):
    """Test analizy wolumenu."""
    wynik = analiza.analizuj_wolumen(
        przykladowe_dane['close'],
        przykladowe_dane['volume'],
        przykladowe_dane['high'],
        przykladowe_dane['low']
    )
    assert isinstance(wynik, dict)
    assert all(k in wynik for k in ['obv', 'adi', 'cmf'])
    assert all(isinstance(v, np.ndarray) for v in wynik.values())
    assert all(len(v) == len(przykladowe_dane['close']) for v in wynik.values())
    assert not any(np.all(np.isnan(v)) for v in wynik.values())  # Sprawdź czy nie wszystkie wartości to NaN

def test_analizuj_wolumen_blad(analiza):
    """Test błędu podczas analizy wolumenu."""
    with patch('ta.volume.OnBalanceVolumeIndicator', side_effect=Exception("Test error")), \
         patch('ta.volume.AccDistIndexIndicator', side_effect=Exception("Test error")), \
         patch('ta.volume.ChaikinMoneyFlowIndicator', side_effect=Exception("Test error")):
        wynik = analiza.analizuj_wolumen(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )
        assert len(wynik) == 0

def test_generuj_sygnaly(analiza, przykladowe_dane):
    """Test generowania sygnałów."""
    mock_rsi = np.array([30.0] * len(przykladowe_dane['close']))
    mock_macd = (
        np.array([1.0] * len(przykladowe_dane['close'])),
        np.array([0.5] * len(przykladowe_dane['close'])),
        np.array([-0.5] * len(przykladowe_dane['close']))
    )
    mock_sma = np.array([100.0] * len(przykladowe_dane['close']))
    mock_vwap = np.array([99.0] * len(przykladowe_dane['close']))
    
    with patch.object(analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(analiza, 'oblicz_macd', return_value=mock_macd), \
         patch.object(analiza, 'oblicz_sma', return_value=mock_sma), \
         patch.object(analiza, 'oblicz_vwap', return_value=mock_vwap):
        
        sygnaly = analiza.generuj_sygnaly(
            przykladowe_dane['close'],
            przykladowe_dane['volume'],
            przykladowe_dane['high'],
            przykladowe_dane['low']
        )
        
        assert len(sygnaly) == 7
        assert all(isinstance(v, np.ndarray) for v in sygnaly.values())
        assert all(len(v) == len(przykladowe_dane['close']) for v in sygnaly.values())

def test_generuj_sygnaly_blad(analiza):
    """Test błędu podczas generowania sygnałów."""
    with patch.object(analiza, 'oblicz_rsi', side_effect=Exception("Test error")):
        sygnaly = analiza.generuj_sygnaly(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )
        assert len(sygnaly) == 0

def test_oblicz_vwap(analiza, przykladowe_dane):
    """Test obliczania VWAP."""
    # Obliczenie oczekiwanego wyniku
    typical_price = (przykladowe_dane['high'] + przykladowe_dane['low'] + przykladowe_dane['close']) / 3
    price_volume = typical_price * przykladowe_dane['volume']
    expected_vwap = np.cumsum(price_volume) / np.cumsum(przykladowe_dane['volume'])
    
    vwap = analiza.oblicz_vwap(
        przykladowe_dane['high'],
        przykladowe_dane['low'],
        przykladowe_dane['close'],
        przykladowe_dane['volume']
    )
    
    assert len(vwap) == len(przykladowe_dane['close'])
    assert isinstance(vwap, np.ndarray)
    np.testing.assert_array_almost_equal(vwap, expected_vwap)

def test_oblicz_vwap_z_okresem(analiza, przykladowe_dane):
    """Test obliczania VWAP z zadanym okresem."""
    okres = 3
    vwap = analiza.oblicz_vwap(
        przykladowe_dane['high'],
        przykladowe_dane['low'],
        przykladowe_dane['close'],
        przykladowe_dane['volume'],
        okres=okres
    )
    
    assert len(vwap) == len(przykladowe_dane['close'])
    assert isinstance(vwap, np.ndarray)
    assert np.isnan(vwap[0])  # Pierwszy element powinien być nan
    assert not np.isnan(vwap[-1])  # Ostatni element nie powinien być nan

def test_oblicz_vwap_blad(analiza):
    """Test błędu podczas obliczania VWAP."""
    # Przygotowanie danych, które spowodują błąd
    bledne_dane = np.array([])
    
    vwap = analiza.oblicz_vwap(bledne_dane, bledne_dane, bledne_dane, bledne_dane)
    assert len(vwap) == 0
    assert isinstance(vwap, np.ndarray)

def test_generuj_sygnaly_z_vwap(analiza, przykladowe_dane):
    """Test generowania sygnałów z uwzględnieniem VWAP."""
    mock_rsi = np.array([30.0] * len(przykladowe_dane['close']))
    mock_macd = (
        np.array([1.0] * len(przykladowe_dane['close'])),
        np.array([0.5] * len(przykladowe_dane['close'])),
        np.array([-0.5] * len(przykladowe_dane['close']))
    )
    mock_sma = np.array([100.0] * len(przykladowe_dane['close']))
    mock_vwap = np.array([99.0] * len(przykladowe_dane['close']))
    
    with patch.object(analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(analiza, 'oblicz_macd', return_value=mock_macd), \
         patch.object(analiza, 'oblicz_sma', return_value=mock_sma), \
         patch.object(analiza, 'oblicz_vwap', return_value=mock_vwap):
        
        sygnaly = analiza.generuj_sygnaly(
            przykladowe_dane['close'],
            przykladowe_dane['volume'],
            przykladowe_dane['high'],
            przykladowe_dane['low']
        )
        
        assert len(sygnaly) == 7  # 5 poprzednich + 2 nowe sygnały VWAP
        assert 'ponad_vwap' in sygnaly
        assert 'ponizej_vwap' in sygnaly
        assert all(isinstance(v, np.ndarray) for v in sygnaly.values())
        assert all(len(v) == len(przykladowe_dane['close']) for v in sygnaly.values()) 

def test_oblicz_linie_livermore_trend_wzrostowy():
    """Test analizy linii Livermore'a dla trendu wzrostowego."""
    analiza = AnalizaTechniczna()
    
    # Przygotowanie danych - trend wzrostowy
    daty = pd.date_range(start='2024-01-01', periods=50, freq='1H')
    df = pd.DataFrame({
        'open': np.linspace(100, 120, 50),  # Trend wzrostowy
        'high': np.linspace(101, 122, 50),
        'low': np.linspace(99, 119, 50),
        'close': np.linspace(100, 120, 50),
        'volume': np.random.normal(1000, 100, 50)  # Stabilny wolumen
    }, index=daty)
    
    wynik = analiza.oblicz_linie_livermore(df)
    
    assert wynik['kierunek'] == 1  # Trend wzrostowy
    assert wynik['sila'] > 0.5  # Silny trend
    assert wynik['wsparcie'] is not None
    assert wynik['opor'] is not None
    assert len(wynik['punkty_zwrotne']) >= 0

def test_oblicz_linie_livermore_trend_spadkowy():
    """Test analizy linii Livermore'a dla trendu spadkowego."""
    analiza = AnalizaTechniczna()
    
    # Przygotowanie danych - trend spadkowy
    daty = pd.date_range(start='2024-01-01', periods=50, freq='1H')
    df = pd.DataFrame({
        'open': np.linspace(120, 100, 50),  # Trend spadkowy
        'high': np.linspace(122, 101, 50),
        'low': np.linspace(119, 99, 50),
        'close': np.linspace(120, 100, 50),
        'volume': np.random.normal(1000, 100, 50) * np.linspace(1.2, 0.8, 50)  # Malejący wolumen
    }, index=daty)
    
    wynik = analiza.oblicz_linie_livermore(df)
    
    assert wynik['kierunek'] == -1  # Trend spadkowy
    assert wynik['sila'] > 0.5  # Silny trend
    assert wynik['wsparcie'] is not None
    assert wynik['opor'] is not None
    assert len(wynik['punkty_zwrotne']) >= 0

def test_oblicz_linie_livermore_trend_boczny():
    """Test analizy linii Livermore'a dla trendu bocznego."""
    analiza = AnalizaTechniczna()
    
    # Przygotowanie danych - trend boczny
    daty = pd.date_range(start='2024-01-01', periods=50, freq='1H')
    ceny = np.random.normal(100, 1, 50)  # Losowe wahania wokół 100
    df = pd.DataFrame({
        'open': ceny,
        'high': ceny + 1,
        'low': ceny - 1,
        'close': ceny,
        'volume': np.random.normal(1000, 100, 50)
    }, index=daty)
    
    wynik = analiza.oblicz_linie_livermore(df)
    
    assert wynik['kierunek'] == 0  # Trend boczny
    assert wynik['sila'] < 0.5  # Słaby trend
    assert wynik['wsparcie'] is not None
    assert wynik['opor'] is not None

def test_oblicz_linie_livermore_punkty_zwrotne():
    """Test wykrywania punktów zwrotnych w analizie Livermore'a."""
    analiza = AnalizaTechniczna()
    
    # Przygotowanie danych z wyraźnymi punktami zwrotnymi
    daty = pd.date_range(start='2024-01-01', periods=20, freq='1H')
    df = pd.DataFrame({
        'open': [100] * 20,
        'high': [100, 101, 102, 105, 104, 103, 102, 101, 100, 99,
                98, 97, 96, 95, 96, 97, 98, 99, 100, 101],
        'low':  [98, 99, 100, 103, 102, 101, 100, 99, 98, 97,
                96, 95, 94, 93, 94, 95, 96, 97, 98, 99],
        'close': [99] * 20,
        'volume': [1000] * 20
    }, index=daty)
    
    wynik = analiza.oblicz_linie_livermore(df)
    
    punkty_zwrotne = wynik['punkty_zwrotne']
    assert len(punkty_zwrotne) >= 2  # Powinny być co najmniej 2 punkty zwrotne
    
    # Sprawdzamy czy są szczyty i dołki
    szczyty = [p for p in punkty_zwrotne if p['typ'] == 'szczyt']
    dolki = [p for p in punkty_zwrotne if p['typ'] == 'dołek']
    
    assert len(szczyty) >= 1  # Powinien być co najmniej 1 szczyt
    assert len(dolki) >= 1  # Powinien być co najmniej 1 dołek

def test_oblicz_linie_livermore_za_malo_danych():
    """Test analizy Livermore'a gdy jest za mało danych."""
    analiza = AnalizaTechniczna()
    
    # Przygotowanie małego zbioru danych
    daty = pd.date_range(start='2024-01-01', periods=5, freq='1H')
    df = pd.DataFrame({
        'open': [100] * 5,
        'high': [101] * 5,
        'low': [99] * 5,
        'close': [100] * 5,
        'volume': [1000] * 5
    }, index=daty)
    
    wynik = analiza.oblicz_linie_livermore(df)
    
    assert wynik['kierunek'] == 0
    assert wynik['sila'] == 0.0
    assert len(wynik['punkty_zwrotne']) == 0
    assert wynik['wsparcie'] is None
    assert wynik['opor'] is None 