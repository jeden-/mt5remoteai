"""
Testy backtestingu strategii Wyckoffa w systemie NikkeiNinja.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from backtesting.symulator import SymulatorRynku
from strategie.wyckoff import StrategiaWyckoff, FazaWyckoff

logger = logging.getLogger(__name__)


def generuj_dane_testowe_akumulacja() -> pd.DataFrame:
    """Generuje dane testowe symulujące fazę akumulacji."""
    # Generujemy 200 świec
    daty = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    
    # Tworzymy 4 fazy akumulacji:
    # 1. Preliminary Support (PS) - zatrzymanie spadków
    ps = np.linspace(100, 85, 50)  # Trend spadkowy
    
    # 2. Secondary Test (ST) - test wsparcia
    st = np.array([85, 84, 83, 84, 85] * 10)  # Konsolidacja przy wsparciu
    
    # 3. Spring - test dołka i odbicie
    spring_base = np.array([84, 82, 81, 83, 85, 86, 87])
    spring = np.repeat(spring_base, 5)  # Powtarzamy wzorzec dla lepszej widoczności
    
    # 4. Sign of Strength (SOS) - wybicie
    sos = np.linspace(87, 95, 65)  # Trend wzrostowy
    
    # Łączymy wszystkie fazy
    ceny = np.concatenate([ps, st, spring, sos])
    
    # Sprawdzamy rozmiar
    assert len(ceny) == 200, f"Nieprawidłowy rozmiar tablicy cen: {len(ceny)}"
    
    # Generujemy OHLC z charakterystycznym wzorcem
    df = pd.DataFrame({
        'open': ceny + np.random.normal(0, 0.1, 200),
        'high': ceny + np.random.normal(0.3, 0.1, 200),
        'low': ceny - np.random.normal(0.3, 0.1, 200),
        'close': ceny + np.random.normal(0, 0.1, 200)
    }, index=daty)
    
    # Generujemy charakterystyczny wzorzec wolumenu
    base_volume = 2000
    volume = np.zeros(200)
    
    # 1. PS - wysoki wolumen na dołku
    volume[:50] = np.random.normal(base_volume * 1.5, base_volume * 0.1, 50)
    volume[45:50] = np.random.normal(base_volume * 2, base_volume * 0.1, 5)  # Szczyt na końcu PS
    
    # 2. ST - malejący wolumen w konsolidacji
    volume[50:100] = np.random.normal(base_volume * 0.8, base_volume * 0.1, 50)
    
    # 3. Spring - bardzo wysoki wolumen przy dołku
    volume[100:135] = np.random.normal(base_volume * 0.7, base_volume * 0.1, 35)
    volume[130:135] = np.random.normal(base_volume * 2.5, base_volume * 0.1, 5)  # Szczyt na spring
    
    # 4. SOS - rosnący wolumen przy wybiciu
    volume[135:] = np.linspace(base_volume, base_volume * 2, 65)
    
    df['volume'] = volume
    
    return df


def generuj_dane_testowe_dystrybucja() -> pd.DataFrame:
    """Generuje dane testowe symulujące fazę dystrybucji."""
    # Generujemy 200 świec
    daty = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    
    # Tworzymy 4 fazy dystrybucji:
    # 1. Preliminary Supply (PSY) - zatrzymanie wzrostów
    psy = np.linspace(100, 120, 50)  # Zwiększony zakres wzrostu
    
    # 2. Buying Climax (BC) - szczyt z wysokim wolumenem
    bc = np.array([120, 122, 121, 120, 119] * 10)  # Wyższy szczyt
    
    # 3. Upthrust - test szczytu i odrzucenie
    upthrust_base = np.array([120, 123, 124, 121, 119, 117, 115])  # Bardziej wyraźny upthrust
    upthrust = np.repeat(upthrust_base, 5)  # Powtarzamy wzorzec
    
    # 4. Sign of Weakness (SOW) - załamanie
    sow = np.linspace(115, 100, 65)  # Większy zakres spadku
    
    # Łączymy wszystkie fazy
    ceny = np.concatenate([psy, bc, upthrust, sow])
    
    # Sprawdzamy rozmiar
    assert len(ceny) == 200, f"Nieprawidłowy rozmiar tablicy cen: {len(ceny)}"
    
    # Generujemy OHLC z charakterystycznym wzorcem
    df = pd.DataFrame({
        'open': ceny + np.random.normal(0, 0.2, 200),  # Zwiększona zmienność
        'high': ceny + np.random.normal(0.5, 0.2, 200),  # Zwiększona zmienność
        'low': ceny - np.random.normal(0.5, 0.2, 200),  # Zwiększona zmienność
        'close': ceny + np.random.normal(0, 0.2, 200)  # Zwiększona zmienność
    }, index=daty)
    
    # Generujemy charakterystyczny wzorzec wolumenu
    base_volume = 2000
    volume = np.zeros(200)
    
    # 1. PSY - rosnący wolumen w trendzie
    volume[:50] = np.linspace(base_volume, base_volume * 3, 50)  # Większy wzrost wolumenu
    
    # 2. BC - bardzo wysoki wolumen na szczycie
    volume[50:100] = np.random.normal(base_volume * 2, base_volume * 0.2, 50)  # Wyższy wolumen
    volume[95:100] = np.random.normal(base_volume * 4, base_volume * 0.2, 5)  # Ekstremalne wartości na BC
    
    # 3. Upthrust - malejący wolumen przy testach
    volume[100:135] = np.random.normal(base_volume * 0.6, base_volume * 0.1, 35)  # Niższy wolumen
    volume[130:135] = np.random.normal(base_volume * 1.5, base_volume * 0.1, 5)  # Wyższy wzrost na upthrust
    
    # 4. SOW - rosnący wolumen przy spadkach
    volume[135:] = np.linspace(base_volume * 1.5, base_volume * 3, 65)  # Większy wzrost wolumenu
    
    df['volume'] = volume
    
    return df


def test_backtest_akumulacja():
    """Test backtestingu strategii w fazie akumulacji."""
    # Przygotowanie danych
    df = generuj_dane_testowe_akumulacja()
    
    # Inicjalizacja strategii i symulatora
    strategia = StrategiaWyckoff()
    strategia.inicjalizuj({
        'okres_ma': 20,
        'min_spread_mult': 0.5,
        'min_vol_mult': 1.1,
        'sl_atr': 2.0,
        'tp_atr': 3.0,
        'vol_std_mult': 1.5,
        'min_swing_candles': 3,
        'rsi_okres': 14,
        'rsi_min': 30,
        'rsi_max': 70,
        'trend_momentum': 0.1
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
        sciezka_raportu=Path("raporty/wyckoff_akumulacja.html")
    )
    
    # Sprawdzenie wyników
    assert wynik.liczba_transakcji > 0, "Brak transakcji w fazie akumulacji"
    assert wynik.zysk_procent > -5, "Zbyt duża strata w fazie akumulacji"
    assert len(wynik.sygnaly_long) > len(wynik.sygnaly_short), "Nieprawidłowe sygnały w akumulacji"


def test_backtest_dystrybucja():
    """Test backtestingu strategii w fazie dystrybucji."""
    # Przygotowanie danych
    df = generuj_dane_testowe_dystrybucja()
    
    # Inicjalizacja strategii i symulatora
    strategia = StrategiaWyckoff()
    strategia.inicjalizuj({
        'okres_ma': 20,
        'min_spread_mult': 0.5,
        'min_vol_mult': 1.1,
        'sl_atr': 2.0,
        'tp_atr': 3.0,
        'vol_std_mult': 1.5,
        'min_swing_candles': 3,
        'rsi_okres': 14,
        'rsi_min': 30,
        'rsi_max': 70,
        'trend_momentum': 0.1
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
        sciezka_raportu=Path("raporty/wyckoff_dystrybucja.html")
    )
    
    # Sprawdzenie wyników
    assert wynik.liczba_transakcji > 0, "Brak transakcji w fazie dystrybucji"
    assert wynik.zysk_procent > -5, "Zbyt duża strata w fazie dystrybucji"
    assert len(wynik.sygnaly_short) > len(wynik.sygnaly_long), "Nieprawidłowe sygnały w dystrybucji"


def test_optymalizacja_parametrow():
    """Test optymalizacji parametrów strategii."""
    # Przygotowanie danych
    df_akumulacja = generuj_dane_testowe_akumulacja()
    df_dystrybucja = generuj_dane_testowe_dystrybucja()
    df = pd.concat([df_akumulacja, df_dystrybucja])
    
    # Zakres parametrów do optymalizacji
    parametry_zakres = {
        'min_vol_mult': np.linspace(1.0, 1.5, 6),
        'vol_std_mult': np.linspace(1.0, 2.0, 6),
        'rsi_min': np.array([25, 30, 35]),
        'rsi_max': np.array([65, 70, 75]),
        'trend_momentum': np.linspace(0.05, 0.15, 3)
    }
    
    # Inicjalizacja strategii
    strategia = StrategiaWyckoff()
    
    # Wykonanie optymalizacji
    najlepsze_parametry = strategia.optymalizuj(df, parametry_zakres)
    
    # Sprawdzenie wyników
    assert isinstance(najlepsze_parametry, dict), "Nieprawidłowy format najlepszych parametrów"
    assert all(k in najlepsze_parametry for k in parametry_zakres.keys()), "Brak wszystkich parametrów"
    
    # Test z optymalnymi parametrami
    strategia.inicjalizuj(najlepsze_parametry)
    symulator = SymulatorRynku(kapital_poczatkowy=100000, prowizja=0.001)
    
    wynik = symulator.testuj_strategie(
        strategia=strategia,
        dane=df,
        generuj_wykresy=True,
        sciezka_raportu=Path("raporty/wyckoff_optymalny.html")
    )
    
    assert wynik.zysk_procent > -5, "Strategia ze zoptymalizowanymi parametrami przynosi zbyt duże straty" 