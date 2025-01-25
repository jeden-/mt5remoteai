"""
Testy dla analizy linii najmniejszego oporu Livermore'a.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from handel.analiza_techniczna import AnalizaTechniczna
from backtesting.symulator import SymulatorRynku
from strategie.techniczna import StrategiaTechniczna

def generuj_dane_trend_wzrostowy(n_punktow: int = 100) -> pd.DataFrame:
    """Generuje dane z trendem wzrostowym."""
    np.random.seed(42)  # Dla powtarzalności
    
    # Generujemy trend liniowy z szumem
    x = np.linspace(0, 1, n_punktow)
    trend = 100 + 20 * x  # Silniejszy trend wzrostowy
    szum = np.random.normal(0, 1, n_punktow)
    ceny = trend + szum
    
    # Dodajemy korekty co 20 punktów
    for i in range(20, n_punktow, 20):
        if i + 5 <= n_punktow:
            ceny[i:i+5] -= np.linspace(2, 0, 5)  # Korekta w dół
    
    # Generujemy pozostałe kolumny
    df = pd.DataFrame({
        'open': ceny - 0.5,
        'high': ceny + 1.0,
        'low': ceny - 1.0,
        'close': ceny,
        'volume': np.random.normal(1000, 200, n_punktow),
        'symbol': ['NKY'] * n_punktow
    })
    
    # Zwiększamy wolumen podczas korekt
    for i in range(20, n_punktow, 20):
        if i + 5 <= n_punktow:
            df.loc[df.index[i:i+5], 'volume'] *= 1.5
    
    df.index = pd.date_range(start='2024-01-01', periods=n_punktow, freq='H')
    return df

def generuj_dane_trend_spadkowy(n_punktow: int = 100) -> pd.DataFrame:
    """Generuje dane z trendem spadkowym."""
    np.random.seed(43)  # Dla powtarzalności
    
    # Generujemy trend liniowy z szumem
    x = np.linspace(0, 1, n_punktow)
    trend = 100 - 20 * x  # Silniejszy trend spadkowy
    szum = np.random.normal(0, 1, n_punktow)
    ceny = trend + szum
    
    # Dodajemy odbicia co 20 punktów
    for i in range(20, n_punktow, 20):
        if i + 5 <= n_punktow:
            ceny[i:i+5] += np.linspace(2, 0, 5)  # Odbicie w górę
    
    # Generujemy pozostałe kolumny
    df = pd.DataFrame({
        'open': ceny - 0.5,
        'high': ceny + 1.0,
        'low': ceny - 1.0,
        'close': ceny,
        'volume': np.random.normal(1000, 200, n_punktow),
        'symbol': ['NKY'] * n_punktow
    })
    
    # Zwiększamy wolumen podczas odbić
    for i in range(20, n_punktow, 20):
        if i + 5 <= n_punktow:
            df.loc[df.index[i:i+5], 'volume'] *= 1.5
    
    df.index = pd.date_range(start='2024-01-01', periods=n_punktow, freq='H')
    return df

def generuj_dane_trend_boczny(n_punktow: int = 100) -> pd.DataFrame:
    """Generuje dane z trendem bocznym."""
    np.random.seed(44)  # Dla powtarzalności
    
    # Generujemy oscylacje wokół średniej
    x = np.linspace(0, 4*np.pi, n_punktow)
    oscylacje = 2 * np.sin(x)  # Mniejsza amplituda
    szum = np.random.normal(0, 0.5, n_punktow)  # Mniejszy szum
    ceny = 100 + oscylacje + szum
    
    # Generujemy pozostałe kolumny
    df = pd.DataFrame({
        'open': ceny - 0.3,
        'high': ceny + 0.5,
        'low': ceny - 0.5,
        'close': ceny,
        'volume': np.random.normal(1000, 100, n_punktow),
        'symbol': ['NKY'] * n_punktow
    })
    
    # Zwiększamy wolumen przy ekstremach (powyżej 75 percentyla i poniżej 25 percentyla)
    kwartyle = np.percentile(ceny, [25, 75])
    df.loc[df['close'] < kwartyle[0], 'volume'] *= 1.3  # Większy wolumen przy dołkach
    df.loc[df['close'] > kwartyle[1], 'volume'] *= 1.3  # Większy wolumen przy szczytach
    
    df.index = pd.date_range(start='2024-01-01', periods=n_punktow, freq='H')
    return df

def test_livermore_trend_wzrostowy():
    """Test skuteczności analizy Livermore'a w trendzie wzrostowym."""
    df = generuj_dane_trend_wzrostowy()
    analiza = AnalizaTechniczna()
    
    # Analiza każdego punktu w czasie
    wyniki = []
    for i in range(20, len(df)):  # Zaczynamy po 20 świecach dla stabilności
        window = df.iloc[:i+1]
        wynik = analiza.oblicz_linie_livermore(window)
        wyniki.append(wynik)
    
    # Sprawdzenie wyników
    kierunki = [w['kierunek'] for w in wyniki]
    sily = [w['sila'] for w in wyniki]
    
    # W trendzie wzrostowym powinno być więcej sygnałów wzrostowych
    assert sum(k == 1 for k in kierunki) > sum(k == -1 for k in kierunki)
    # Średnia siła trendu powinna być wysoka
    assert np.mean(sily) > 0.6
    # Powinny być wykryte punkty zwrotne
    assert any(len(w['punkty_zwrotne']) > 0 for w in wyniki)

def test_livermore_trend_spadkowy():
    """Test skuteczności analizy Livermore'a w trendzie spadkowym."""
    df = generuj_dane_trend_spadkowy()
    analiza = AnalizaTechniczna()
    
    # Analiza każdego punktu w czasie
    wyniki = []
    for i in range(20, len(df)):
        window = df.iloc[:i+1]
        wynik = analiza.oblicz_linie_livermore(window)
        wyniki.append(wynik)
    
    # Sprawdzenie wyników
    kierunki = [w['kierunek'] for w in wyniki]
    sily = [w['sila'] for w in wyniki]
    
    # W trendzie spadkowym powinno być więcej sygnałów spadkowych
    assert sum(k == -1 for k in kierunki) > sum(k == 1 for k in kierunki)
    # Średnia siła trendu powinna być wysoka
    assert np.mean(sily) > 0.6
    # Powinny być wykryte punkty zwrotne
    assert any(len(w['punkty_zwrotne']) > 0 for w in wyniki)

def test_livermore_trend_boczny():
    """Test skuteczności analizy Livermore'a w trendzie bocznym."""
    df = generuj_dane_trend_boczny()
    analiza = AnalizaTechniczna()
    
    # Analiza każdego punktu w czasie
    wyniki = []
    for i in range(20, len(df)):
        window = df.iloc[:i+1]
        wynik = analiza.oblicz_linie_livermore(window)
        wyniki.append(wynik)
    
    # Sprawdzenie wyników
    kierunki = [w['kierunek'] for w in wyniki]
    sily = [w['sila'] for w in wyniki]
    
    # W trendzie bocznym powinno być więcej sygnałów neutralnych
    assert sum(k == 0 for k in kierunki) > sum(k != 0 for k in kierunki)
    # Średnia siła trendu powinna być niska
    assert np.mean(sily) < 0.5
    # Powinno być więcej punktów zwrotnych niż w trendach kierunkowych
    punkty_zwrotne = [len(w['punkty_zwrotne']) for w in wyniki]
    assert np.mean(punkty_zwrotne) > 2

def test_livermore_backtest():
    """Test skuteczności strategii opartej na analizie Livermore'a."""
    # Połączenie różnych warunków rynkowych z sekwencyjnymi datami
    start_date = pd.Timestamp('2024-01-01')
    df1 = generuj_dane_trend_wzrostowy(100)  # Trend wzrostowy
    df1.index = pd.date_range(start=start_date, periods=100, freq='H')
    
    start_date2 = df1.index[-1] + pd.Timedelta(hours=1)
    df2 = generuj_dane_trend_boczny(60)  # Trend boczny
    df2.index = pd.date_range(start=start_date2, periods=60, freq='H')
    
    start_date3 = df2.index[-1] + pd.Timedelta(hours=1)
    df3 = generuj_dane_trend_spadkowy(100)  # Trend spadkowy
    df3.index = pd.date_range(start=start_date3, periods=100, freq='H')
    
    df = pd.concat([df1, df2, df3])
    df = df.sort_index()  # Upewniamy się, że dane są posortowane po czasie
    
    # Inicjalizacja strategii i symulatora
    strategia = StrategiaTechniczna()
    strategia.inicjalizuj({
        'min_zmiana': 0.0001,  # Minimalna zmiana dla linii Livermore'a
        'okres_rsi': 14,
        'rsi_wyprzedanie': 35,  # Łagodniejsze warunki
        'rsi_wykupienie': 65,  # Łagodniejsze warunki
        'macd_szybki': 12,
        'macd_wolny': 26,
        'macd_sygnalowy': 9,
        'stoch_k': 14,
        'stoch_d': 3,
        'stoch_smooth': 3,
        'stoch_wyprzedanie': 25,  # Łagodniejsze warunki
        'stoch_wykupienie': 75,  # Łagodniejsze warunki
        'atr_mnoznik': 2.0,
        'min_spread': 0.01
    })
    
    symulator = SymulatorRynku(
        kapital_poczatkowy=100000,
        prowizja=0.001  # 0.1%
    )
    
    # Wykonanie backtestu
    wynik = symulator.testuj_strategie(
        strategia=strategia,
        dane=df,
        generuj_wykresy=True,
        sciezka_raportu=Path("raporty/livermore_backtest.html")
    )
    
    # Sprawdzenie wyników
    assert wynik.liczba_transakcji > 0, "Brak transakcji"
    assert wynik.zysk_procent > -10.0, "Zbyt duża strata"
    assert wynik.max_drawdown < 15.0, "Zbyt duży drawdown" 