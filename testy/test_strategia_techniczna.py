"""
Testy dla strategii technicznej.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from strategie.techniczna import StrategiaTechniczna
from strategie.interfejs import KierunekTransakcji

@pytest.fixture
def dane_testowe():
    """Przygotowuje testowe dane OHLCV."""
    daty = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    dane = pd.DataFrame({
        'open': np.random.normal(100, 2, 100),
        'high': np.random.normal(102, 2, 100),
        'low': np.random.normal(98, 2, 100),
        'close': np.random.normal(101, 2, 100),
        'volume': np.random.normal(1000, 200, 100),
        'symbol': ['NKY'] * 100  # Dodajemy kolumnę symbol
    }, index=daty)
    
    # Korekta high/low
    dane['high'] = np.maximum(dane[['open', 'close', 'high']].max(axis=1), dane['high'])
    dane['low'] = np.minimum(dane[['open', 'close', 'low']].min(axis=1), dane['low'])
    
    return dane

@pytest.fixture
def strategia():
    """Tworzy instancję strategii z domyślnymi parametrami."""
    strategia = StrategiaTechniczna()
    strategia.inicjalizuj({})
    return strategia

def test_inicjalizacja_parametrow(strategia):
    """Test inicjalizacji z różnymi parametrami."""
    parametry = {
        'rsi_okres': 10,
        'rsi_wykupienie': 75,
        'rsi_wyprzedanie': 25,
        'macd_szybki': 8,
        'macd_wolny': 17,
        'macd_sygnalowy': 7,
        'sl_atr': 2.5,
        'tp_atr': 3.5
    }
    
    strategia.inicjalizuj(parametry)
    
    for nazwa, wartosc in parametry.items():
        assert strategia.parametry[nazwa] == wartosc

def test_analizuj_sygnaly_long(strategia, dane_testowe):
    """Test generowania sygnałów LONG."""
    # Mock dla RSI (wyprzedanie)
    mock_rsi = np.array([25.0] * len(dane_testowe))
    # Mock dla MACD (rosnący)
    mock_hist = np.array([0.1] * len(dane_testowe))
    mock_hist[-1] = 0.2
    # Mock dla VWAP (cena powyżej)
    mock_vwap = np.array([95.0] * len(dane_testowe))
    # Mock dla Stochastic (wyprzedanie i przecięcie)
    mock_stoch_k = np.array([15.0] * len(dane_testowe))  # Poniżej wyprzedania (20)
    mock_stoch_d = np.array([10.0] * len(dane_testowe))  # %K > %D (sygnał kupna)
    
    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)), \
         patch.object(strategia.analiza, 'oblicz_vwap', return_value=mock_vwap), \
         patch.object(strategia.analiza, 'oblicz_stochastic', return_value=(mock_stoch_k, mock_stoch_d)):
        
        sygnaly = strategia.analizuj(dane_testowe)
        
        assert len(sygnaly) > 0
        assert all(s.kierunek == KierunekTransakcji.LONG for s in sygnaly)
        assert all('RSI(25.0) wyprzedany' in s.opis for s in sygnaly)
        assert all('Stoch(15.0) wyprzedany' in s.opis for s in sygnaly)
        assert all(s.stop_loss < s.cena_wejscia for s in sygnaly)
        assert all(s.take_profit > s.cena_wejscia for s in sygnaly)

def test_analizuj_sygnaly_short(strategia, dane_testowe):
    """Test generowania sygnałów SHORT."""
    # Mock dla RSI (wykupienie)
    mock_rsi = np.array([75.0] * len(dane_testowe))
    # Mock dla MACD (malejący)
    mock_hist = np.array([0.2] * len(dane_testowe))
    mock_hist[-1] = 0.1
    # Mock dla VWAP (cena powyżej)
    mock_vwap = np.array([95.0] * len(dane_testowe))
    # Mock dla Stochastic (wykupienie i przecięcie)
    mock_stoch_k = np.array([85.0] * len(dane_testowe))  # Powyżej wykupienia (80)
    mock_stoch_d = np.array([90.0] * len(dane_testowe))  # %K < %D (sygnał sprzedaży)
    
    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)), \
         patch.object(strategia.analiza, 'oblicz_vwap', return_value=mock_vwap), \
         patch.object(strategia.analiza, 'oblicz_stochastic', return_value=(mock_stoch_k, mock_stoch_d)):
        
        sygnaly = strategia.analizuj(dane_testowe)
        
        assert len(sygnaly) > 0
        assert all(s.kierunek == KierunekTransakcji.SHORT for s in sygnaly)
        assert all('RSI(75.0) wykupiony' in s.opis for s in sygnaly)
        assert all('Stoch(85.0) wykupiony' in s.opis for s in sygnaly)
        assert all(s.stop_loss > s.cena_wejscia for s in sygnaly)
        assert all(s.take_profit < s.cena_wejscia for s in sygnaly)

def test_aktualizuj_pozycje_long(strategia, dane_testowe):
    """Test aktualizacji pozycji LONG."""
    # Mock dla RSI (wykupienie - sygnał zamknięcia)
    mock_rsi = np.array([75.0] * len(dane_testowe))
    # Mock dla MACD (malejący - sygnał zamknięcia)
    mock_hist = np.array([0.2] * len(dane_testowe))
    mock_hist[-1] = 0.1
    
    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)):
        
        sygnaly = strategia.aktualizuj(
            dane_testowe,
            [('NKY', KierunekTransakcji.LONG, 100.0)]
        )
        
        assert len(sygnaly) == 1
        assert sygnaly[0].kierunek == KierunekTransakcji.SHORT  # Zamknięcie LONG
        assert 'RSI(75.0) wykupiony' in sygnaly[0].opis
        assert 'zysk_procent' in sygnaly[0].metadane

def test_aktualizuj_pozycje_short(strategia, dane_testowe):
    """Test aktualizacji pozycji SHORT."""
    # Mock dla RSI (wyprzedanie - sygnał zamknięcia)
    mock_rsi = np.array([25.0] * len(dane_testowe))
    # Mock dla MACD (rosnący - sygnał zamknięcia)
    mock_hist = np.array([0.1] * len(dane_testowe))
    mock_hist[-1] = 0.2
    
    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)):
        
        sygnaly = strategia.aktualizuj(
            dane_testowe,
            [('NKY', KierunekTransakcji.SHORT, 100.0)]
        )
        
        assert len(sygnaly) == 1
        assert sygnaly[0].kierunek == KierunekTransakcji.LONG  # Zamknięcie SHORT
        assert 'RSI(25.0) wyprzedany' in sygnaly[0].opis
        assert 'zysk_procent' in sygnaly[0].metadane

def test_optymalizacja_parametrow(strategia, dane_testowe):
    """Test optymalizacji parametrów."""
    # Przygotowanie danych testowych z wyraźnym trendem
    daty = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    dane = pd.DataFrame({
        'open': np.linspace(100, 150, 100),  # Trend wzrostowy
        'high': np.linspace(102, 155, 100),
        'low': np.linspace(98, 145, 100),
        'close': np.linspace(101, 152, 100),
        'volume': np.random.normal(1000, 200, 100),
        'symbol': ['NKY'] * 100  # Dodajemy kolumnę symbol
    }, index=daty)
    
    # Parametry do optymalizacji
    parametry_zakres = {
        'rsi_okres': (10, 20, 5),
        'rsi_wykupienie': (65, 75, 5),
        'rsi_wyprzedanie': (25, 35, 5),
        'macd_szybki': (8, 12, 2),
        'macd_wolny': (17, 26, 3),
        'macd_sygnalowy': (7, 9, 1)
    }
    
    # Mock dla wskaźników technicznych - generujemy sygnały LONG i SHORT
    def mock_rsi(dane):
        n = len(dane)
        return np.array([25.0] * (n//2) + [75.0] * (n - n//2))  # Najpierw wyprzedanie, potem wykupienie
    
    def mock_macd(dane, szybki, wolny, sygnał):
        n = len(dane)
        hist = np.zeros(n)
        # Pierwsza połowa - rosnący trend
        hist[1:n//2] = np.linspace(0.1, 0.3, n//2-1)
        # Druga połowa - malejący trend
        hist[n//2:] = np.linspace(0.3, 0.1, n - n//2)
        return None, None, hist
    
    def mock_vwap(high, low, close, volume):
        # VWAP powyżej ceny dla pierwszej połowy (sygnały LONG)
        # i poniżej ceny dla drugiej połowy (sygnały SHORT)
        n = len(close)
        vwap = np.zeros(n)
        vwap[:n//2] = np.array(close[:n//2]) * 1.05  # 5% powyżej ceny
        vwap[n//2:] = np.array(close[n//2:]) * 0.95  # 5% poniżej ceny
        return vwap

    def mock_stochastic(df, okres_k=14, okres_d=3, wygładzanie=3):
        n = len(df)
        # %K poniżej 20 dla pierwszej połowy (wyprzedanie)
        # %K powyżej 80 dla drugiej połowy (wykupienie)
        k = np.array([15.0] * (n//2) + [85.0] * (n - n//2))
        # %D opóźniony względem %K
        d = np.array([10.0] * (n//2) + [90.0] * (n - n//2))
        return k, d

    with patch.object(strategia.analiza, 'oblicz_rsi', side_effect=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', side_effect=mock_macd), \
         patch.object(strategia.analiza, 'oblicz_vwap', side_effect=mock_vwap), \
         patch.object(strategia.analiza, 'oblicz_stochastic', side_effect=mock_stochastic):
        
        wynik = strategia.optymalizuj(dane, parametry_zakres)
        
        # Sprawdzenie podstawowe
        assert isinstance(wynik, dict)
        for param in parametry_zakres.keys():
            assert param in wynik
            start, stop, step = parametry_zakres[param]
            assert start <= wynik[param] <= stop
        
        # Sprawdzenie czy parametry są optymalne dla danych warunków
        strategia.inicjalizuj(wynik)
        sygnaly = strategia.analizuj(dane, {'tryb': 'optymalizacja'})
        statystyki = strategia.generuj_statystyki(sygnaly)
        
        assert len(sygnaly) > 0
        assert statystyki['profit_factor'] > 1.0  # Strategia powinna być zyskowna
        assert statystyki['win_rate'] > 0.5  # Win rate powyżej 50%

def test_optymalizacja_parametrow_puste_dane(strategia):
    """Test optymalizacji dla pustych danych."""
    dane_puste = pd.DataFrame()
    parametry_zakres = {
        'rsi_okres': (10, 20, 5),
        'rsi_wykupienie': (65, 75, 5)
    }
    
    wynik = strategia.optymalizuj(dane_puste, parametry_zakres)
    assert wynik == strategia.parametry  # Powinny zostać domyślne parametry

def test_optymalizacja_parametrow_bledne_zakresy(strategia, dane_testowe):
    """Test optymalizacji dla błędnych zakresów parametrów."""
    parametry_zakres = {
        'rsi_okres': (20, 10, 5),  # Błędny zakres (start > stop)
        'rsi_wykupienie': (65, 75, -5)  # Ujemny krok
    }
    
    wynik = strategia.optymalizuj(dane_testowe, parametry_zakres)
    assert wynik == strategia.parametry  # Powinny zostać domyślne parametry

def test_generowanie_statystyk(strategia):
    """Test generowania statystyk."""
    historia = [
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={'zysk_procent': 2.5}
        ),
        Mock(
            kierunek=KierunekTransakcji.SHORT,
            metadane={'zysk_procent': -1.5}
        ),
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={'zysk_procent': 3.0}
        )
    ]
    
    statystyki = strategia.generuj_statystyki(historia)
    
    assert isinstance(statystyki, dict)
    assert statystyki['liczba_transakcji'] == 3
    assert statystyki['win_rate'] == pytest.approx(0.67, rel=0.01)
    assert statystyki['liczba_long'] == 2
    assert statystyki['liczba_short'] == 1
    assert statystyki['zysk_sredni'] > 0
    assert statystyki['win_rate_long'] == 1.0
    assert statystyki['win_rate_short'] == 0.0
    assert statystyki['profit_factor'] > 1.0

def test_generowanie_statystyk_pusta_historia(strategia):
    """Test generowania statystyk dla pustej historii transakcji."""
    statystyki = strategia.generuj_statystyki([])
    assert isinstance(statystyki, dict)
    assert len(statystyki) == 0

def test_generowanie_statystyk_brak_zysku(strategia):
    """Test generowania statystyk dla transakcji bez zysku w metadanych."""
    historia = [
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={}  # Brak zysk_procent
        ),
        Mock(
            kierunek=KierunekTransakcji.SHORT,
            metadane={'inna_wartosc': 123}  # Brak zysk_procent
        )
    ]
    
    statystyki = strategia.generuj_statystyki(historia)
    assert isinstance(statystyki, dict)
    assert len(statystyki) == 0

def test_obsluga_bledow_analizy(strategia, dane_testowe):
    """Test obsługi błędów podczas analizy."""
    with patch.object(strategia.analiza, 'oblicz_rsi', side_effect=Exception("Test error")):
        sygnaly = strategia.analizuj(dane_testowe)
        assert len(sygnaly) == 0

def test_obsluga_bledow_aktualizacji(strategia, dane_testowe):
    """Test obsługi błędów podczas aktualizacji."""
    with patch.object(strategia.analiza, 'oblicz_rsi', side_effect=Exception("Test error")):
        sygnaly = strategia.aktualizuj(dane_testowe, [('NKY', KierunekTransakcji.LONG, 100.0)])
        assert len(sygnaly) == 0

def test_brak_sygnalow_neutralny_rynek(strategia, dane_testowe):
    """Test braku sygnałów przy neutralnym rynku."""
    # Mock dla RSI (neutralny)
    mock_rsi = np.array([50.0] * len(dane_testowe))
    # Mock dla MACD (bez trendu)
    mock_hist = np.array([0.1] * len(dane_testowe))
    # Mock dla VWAP (cena na poziomie)
    mock_vwap = np.array([100.0] * len(dane_testowe))
    
    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)), \
         patch.object(strategia.analiza, 'oblicz_vwap', return_value=mock_vwap):
        
        sygnaly = strategia.analizuj(dane_testowe)
        assert len(sygnaly) == 0

def test_brak_kolumny_symbol(strategia, dane_testowe):
    """Test generowania sygnałów gdy DataFrame nie ma kolumny 'symbol'."""
    # Mock dla RSI (wyprzedanie)
    mock_rsi = np.array([25.0] * len(dane_testowe))
    # Mock dla MACD (rosnący)
    mock_hist = np.array([0.1] * len(dane_testowe))
    mock_hist[-1] = 0.2
    # Mock dla VWAP
    mock_vwap = np.array([95.0] * len(dane_testowe))  # Zmienione ze 105.0 na 95.0
    
    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)), \
         patch.object(strategia.analiza, 'oblicz_vwap', return_value=mock_vwap):
        
        # Usuwamy kolumnę 'symbol'
        df = dane_testowe.copy()
        if 'symbol' in df.columns:
            df = df.drop('symbol', axis=1)
        
        sygnaly = strategia.analizuj(df)
        
        assert len(sygnaly) == 0  # Teraz oczekujemy 0 sygnałów gdy brak kolumny 'symbol'

def test_aktualizuj_brak_pozycji(strategia, dane_testowe):
    """Test aktualizacji gdy nie ma aktywnych pozycji."""
    sygnaly = strategia.aktualizuj(dane_testowe, [])
    assert isinstance(sygnaly, list)
    assert len(sygnaly) == 0

def test_analizuj_za_malo_danych(strategia):
    """Test analizy gdy jest za mało danych."""
    df = pd.DataFrame({
        'open': [100],
        'high': [101],
        'low': [99],
        'close': [100],
        'volume': [1000]
    }, index=[pd.Timestamp('2024-01-01')])
    
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 0

def test_analizuj_nieprawidlowe_dane(strategia):
    """Test analizy gdy dane zawierają nieprawidłowe wartości."""
    df = pd.DataFrame({
        'open': [100, np.nan, 102],
        'high': [101, 103, np.inf],
        'low': [99, -np.inf, 101],
        'close': [100, None, 102],
        'volume': [1000, -1000, 0]
    }, index=pd.date_range(start='2024-01-01', periods=3))
    
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 0

def test_aktualizuj_brak_danych_historycznych(strategia):
    """Test aktualizacji gdy brak danych historycznych."""
    df_pusty = pd.DataFrame()
    sygnaly = strategia.aktualizuj(df_pusty, [('NKY', KierunekTransakcji.LONG, 100.0)])
    assert len(sygnaly) == 0

def test_optymalizacja_jeden_parametr(strategia, dane_testowe):
    """Test optymalizacji gdy optymalizowany jest tylko jeden parametr."""
    parametry_zakres = {
        'rsi_okres': (10, 20, 5)
    }
    
    wynik = strategia.optymalizuj(dane_testowe, parametry_zakres)
    assert isinstance(wynik, dict)
    assert 'rsi_okres' in wynik
    assert 10 <= wynik['rsi_okres'] <= 20

def test_generowanie_statystyk_jedna_transakcja(strategia):
    """Test generowania statystyk dla pojedynczej transakcji."""
    historia = [
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={'zysk_procent': 2.5}
        )
    ]
    
    statystyki = strategia.generuj_statystyki(historia)
    assert statystyki['liczba_transakcji'] == 1
    assert statystyki['liczba_long'] == 1
    assert statystyki['liczba_short'] == 0
    assert statystyki['win_rate'] == 1.0
    assert statystyki['win_rate_long'] == 1.0
    assert 'win_rate_short' not in statystyki

def test_generowanie_statystyk_same_straty(strategia):
    """Test generowania statystyk gdy wszystkie transakcje są stratne."""
    historia = [
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={'zysk_procent': -1.5}
        ),
        Mock(
            kierunek=KierunekTransakcji.SHORT,
            metadane={'zysk_procent': -2.0}
        )
    ]
    
    statystyki = strategia.generuj_statystyki(historia)
    assert statystyki['liczba_transakcji'] == 2
    assert statystyki['win_rate'] == 0.0
    assert statystyki['win_rate_long'] == 0.0
    assert statystyki['win_rate_short'] == 0.0
    assert statystyki['profit_factor'] == 0.0

def test_analizuj_z_dodatkowymi_danymi(strategia, dane_testowe):
    """Test analizy z dodatkowymi danymi w trybie optymalizacji."""
    mock_rsi = np.array([25.0] * len(dane_testowe))
    mock_hist = np.array([0.1] * len(dane_testowe))
    mock_hist[-1] = 0.2
    mock_vwap = np.array([95.0] * len(dane_testowe))  # Zmienione ze 105.0 na 95.0
    mock_stoch_k = np.array([15.0] * len(dane_testowe))  # Wyprzedanie
    mock_stoch_d = np.array([10.0] * len(dane_testowe))  # %K > %D (sygnał kupna)

    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)), \
         patch.object(strategia.analiza, 'oblicz_vwap', return_value=mock_vwap), \
         patch.object(strategia.analiza, 'oblicz_stochastic', return_value=(mock_stoch_k, mock_stoch_d)):

        # Test w trybie normalnym
        sygnaly_normalne = strategia.analizuj(dane_testowe)
        assert len(sygnaly_normalne) > 0

        # Test w trybie optymalizacji
        sygnaly_optymalizacja = strategia.analizuj(dane_testowe, {'tryb': 'optymalizacja'})
        assert len(sygnaly_optymalizacja) > 0

def test_aktualizuj_wiele_pozycji(strategia, dane_testowe):
    """Test aktualizacji wielu pozycji jednocześnie."""
    mock_rsi = np.array([75.0] * len(dane_testowe))  # Sygnał zamknięcia dla LONG
    mock_hist = np.array([0.2] * len(dane_testowe))
    mock_hist[-1] = 0.1  # Sygnał zamknięcia dla LONG
    
    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)):
        
        sygnaly = strategia.aktualizuj(
            dane_testowe,
            [
                ('NKY', KierunekTransakcji.LONG, 100.0),
                ('NKY', KierunekTransakcji.LONG, 101.0),
                ('NKY', KierunekTransakcji.SHORT, 102.0)
            ]
        )
        
        assert len(sygnaly) == 2  # Powinny być 2 sygnały zamknięcia dla pozycji LONG
        assert all(s.kierunek == KierunekTransakcji.SHORT for s in sygnaly[:2])  # Zamknięcie LONG

def test_optymalizacja_parametrow_rozne_kroki(strategia, dane_testowe):
    """Test optymalizacji z różnymi krokami dla parametrów."""
    parametry_zakres = {
        'rsi_okres': (10, 20, 2),  # Mały krok
        'rsi_wykupienie': (65, 75, 5),  # Średni krok
        'macd_szybki': (8, 12, 4)  # Duży krok
    }
    
    wynik = strategia.optymalizuj(dane_testowe, parametry_zakres)
    assert isinstance(wynik, dict)
    for param in parametry_zakres:
        assert param in wynik
        start, stop, step = parametry_zakres[param]
        assert start <= wynik[param] <= stop

def test_analizuj_atr_zero(strategia, dane_testowe):
    """Test analizy gdy ATR wynosi zero."""
    # Ustawiamy wszystkie ceny na tę samą wartość, co spowoduje ATR = 0
    df = dane_testowe.copy()
    df['high'] = 100.0
    df['low'] = 100.0
    df['close'] = 100.0
    df['open'] = 100.0

    mock_rsi = np.array([25.0] * len(df))  # Sygnał LONG
    mock_hist = np.array([0.1] * len(df))
    mock_hist[-1] = 0.2  # Rosnący MACD
    mock_vwap = np.array([95.0] * len(df))  # Zmienione ze 105.0 na 95.0
    mock_stoch_k = np.array([15.0] * len(df))  # Wyprzedanie
    mock_stoch_d = np.array([10.0] * len(df))  # %K > %D (sygnał kupna)

    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)), \
         patch.object(strategia.analiza, 'oblicz_vwap', return_value=mock_vwap), \
         patch.object(strategia.analiza, 'oblicz_stochastic', return_value=(mock_stoch_k, mock_stoch_d)):
        
        sygnaly = strategia.analizuj(df)
        assert len(sygnaly) > 0
        assert all(s.stop_loss < s.cena_wejscia for s in sygnaly)  # SL poniżej ceny wejścia dla LONG
        assert all(s.take_profit > s.cena_wejscia for s in sygnaly)  # TP powyżej ceny wejścia dla LONG

def test_stochastic_parametry(strategia):
    """Test inicjalizacji parametrów Stochastic."""
    parametry = {
        'stoch_k': 21,
        'stoch_d': 5,
        'stoch_smooth': 4,
        'stoch_wykupienie': 85,
        'stoch_wyprzedanie': 15
    }
    
    strategia.inicjalizuj(parametry)
    
    for nazwa, wartosc in parametry.items():
        assert strategia.parametry[nazwa] == wartosc

def test_brak_sygnalu_stochastic_niezgodny(strategia, dane_testowe):
    """Test braku sygnału gdy Stochastic nie potwierdza RSI."""
    # Mock dla RSI (wyprzedanie)
    mock_rsi = np.array([25.0] * len(dane_testowe))
    # Mock dla MACD (rosnący)
    mock_hist = np.array([0.1] * len(dane_testowe))
    mock_hist[-1] = 0.2
    # Mock dla VWAP (cena powyżej)
    mock_vwap = np.array([95.0] * len(dane_testowe))
    # Mock dla Stochastic (brak potwierdzenia - %K < %D)
    mock_stoch_k = np.array([15.0] * len(dane_testowe))
    mock_stoch_d = np.array([20.0] * len(dane_testowe))  # %K < %D (brak sygnału)
    
    with patch.object(strategia.analiza, 'oblicz_rsi', return_value=mock_rsi), \
         patch.object(strategia.analiza, 'oblicz_macd', return_value=(None, None, mock_hist)), \
         patch.object(strategia.analiza, 'oblicz_vwap', return_value=mock_vwap), \
         patch.object(strategia.analiza, 'oblicz_stochastic', return_value=(mock_stoch_k, mock_stoch_d)):
        
        sygnaly = strategia.analizuj(dane_testowe)
        assert len(sygnaly) == 0  # Brak sygnałów gdy Stochastic nie potwierdza 