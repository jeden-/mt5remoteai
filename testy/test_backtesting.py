"""
Testy dla modułu backtestingu w systemie NikkeiNinja.
"""
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from backtesting.metryki import (oblicz_max_drawdown, oblicz_metryki_ryzyka,
                                oblicz_profit_factor, oblicz_sharpe_ratio)
from backtesting.symulator import (KierunekTransakcji, SymulatorRynku,
                                 WynikBacktestu, WynikTransakcji)
from strategie.interfejs import IStrategia, SygnalTransakcyjny


class TestowaStrategia(IStrategia):
    """Prosta strategia testowa do weryfikacji backtestingu."""
    
    def __init__(self):
        """Inicjalizacja strategii testowej."""
        self.parametry = {}
    
    def inicjalizuj(self, parametry: Dict) -> None:
        """Inicjalizacja parametrów."""
        self.parametry = parametry
    
    def analizuj(self, df: pd.DataFrame) -> List[SygnalTransakcyjny]:
        """
        Generuje sygnały testowe:
        - LONG gdy RSI < 30
        - SHORT gdy RSI > 70
        """
        if len(df) < 14:
            return []
            
        # Obliczanie RSI
        close = df['close'].values
        delta = np.diff(close)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        sygnaly = []
        if rsi < 30:  # Wykupienie - sygnał long
            sygnaly.append(SygnalTransakcyjny(
                timestamp=df.index[-1],
                symbol='TEST',
                kierunek=KierunekTransakcji.LONG,
                cena_wejscia=float(df['close'].iloc[-1]),
                stop_loss=float(df['close'].iloc[-1] * 0.95),
                take_profit=float(df['close'].iloc[-1] * 1.1),
                wolumen=1.0,
                opis="RSI < 30",
                metadane={'rsi': rsi}
            ))
        elif rsi > 70:  # Wyprzedanie - sygnał short
            sygnaly.append(SygnalTransakcyjny(
                timestamp=df.index[-1],
                symbol='TEST',
                kierunek=KierunekTransakcji.SHORT,
                cena_wejscia=float(df['close'].iloc[-1]),
                stop_loss=float(df['close'].iloc[-1] * 1.05),
                take_profit=float(df['close'].iloc[-1] * 0.9),
                wolumen=1.0,
                opis="RSI > 70",
                metadane={'rsi': rsi}
            ))
        
        return sygnaly
    
    def aktualizuj(self, 
                   df: pd.DataFrame,
                   aktywne_pozycje: List[tuple]) -> List[SygnalTransakcyjny]:
        """Aktualizacja pozycji."""
        return []


def test_inicjalizacja_symulatora():
    """Test inicjalizacji symulatora rynku."""
    symulator = SymulatorRynku(kapital_poczatkowy=100000.0, prowizja_procent=0.1)
    
    assert symulator.kapital_poczatkowy == Decimal('100000.0')
    assert symulator.prowizja_procent == Decimal('0.1')
    assert symulator.kapital_aktualny == symulator.kapital_poczatkowy
    assert len(symulator.pozycje) == 0
    assert len(symulator.historia_transakcji) == 0


def test_otwarcie_pozycji():
    """Test otwierania pozycji."""
    symulator = SymulatorRynku(kapital_poczatkowy=100000.0)
    
    # Otwarcie pozycji LONG
    sukces = symulator.otworz_pozycje(
        timestamp=datetime.now(),
        symbol="TEST",
        kierunek=KierunekTransakcji.LONG,
        cena=1000.0,
        wolumen=1.0
    )
    
    assert sukces
    assert len(symulator.pozycje) == 1
    assert symulator.pozycje[0][0] == "TEST"
    assert symulator.pozycje[0][1] == KierunekTransakcji.LONG
    assert float(symulator.pozycje[0][2]) == 1000.0
    assert float(symulator.pozycje[0][3]) == 1.0


def test_zamkniecie_pozycji():
    """Test zamykania pozycji."""
    symulator = SymulatorRynku(kapital_poczatkowy=100000.0)
    
    # Otwarcie i zamknięcie pozycji LONG z zyskiem
    symulator.otworz_pozycje(
        timestamp=datetime.now(),
        symbol="TEST",
        kierunek=KierunekTransakcji.LONG,
        cena=1000.0,
        wolumen=1.0
    )
    
    sukces = symulator.zamknij_pozycje(
        timestamp=datetime.now(),
        symbol="TEST",
        cena=1100.0
    )
    
    assert sukces
    assert len(symulator.pozycje) == 0
    assert len(symulator.historia_transakcji) == 1
    assert symulator.historia_transakcji[0].zysk_procent > 0


def test_obliczanie_profit_factor():
    """Test obliczania profit factor."""
    transakcje = [
        WynikTransakcji(
            timestamp_wejscia=datetime.now(),
            timestamp_wyjscia=datetime.now(),
            kierunek=KierunekTransakcji.LONG,
            cena_wejscia=Decimal('1000.0'),
            cena_wyjscia=Decimal('1100.0'),
            wolumen=Decimal('1.0'),
            zysk_procent=10.0,
            powod_wyjscia="Test",
            metadane={}
        ),
        WynikTransakcji(
            timestamp_wejscia=datetime.now(),
            timestamp_wyjscia=datetime.now(),
            kierunek=KierunekTransakcji.LONG,
            cena_wejscia=Decimal('1000.0'),
            cena_wyjscia=Decimal('950.0'),
            wolumen=Decimal('1.0'),
            zysk_procent=-5.0,
            powod_wyjscia="Test",
            metadane={}
        )
    ]
    
    pf = oblicz_profit_factor(transakcje)
    assert pf == 2.0  # 10.0 / 5.0 = 2.0
    
    # Test dla samych zyskownych transakcji
    pf = oblicz_profit_factor([transakcje[0]])
    assert pf == float('inf')
    
    # Test dla pustej listy
    pf = oblicz_profit_factor([])
    assert pf == 0.0


def test_obliczanie_max_drawdown():
    """Test obliczania maksymalnego drawdown."""
    kapital = [100.0, 110.0, 105.0, 95.0, 100.0]
    
    dd, start, end = oblicz_max_drawdown(kapital)
    assert dd == pytest.approx(13.64, rel=0.01)  # (110-95)/110 * 100
    assert start == 1  # Szczyt na indeksie 1
    assert end == 3  # Dołek na indeksie 3
    
    # Test dla rosnącego kapitału
    kapital = [100.0, 110.0, 120.0]
    dd, start, end = oblicz_max_drawdown(kapital)
    assert dd == 0.0
    assert start == 0
    assert end == 0
    
    # Test dla pustej listy
    dd, start, end = oblicz_max_drawdown([])
    assert dd == 0.0
    assert start == 0
    assert end == 0


def test_obliczanie_sharpe_ratio():
    """Test obliczania wskaźnika Sharpe'a."""
    # Test dla stabilnych dodatnich zwrotów
    zwroty = [0.1, 0.2, 0.15, 0.1, 0.2]
    sharpe = oblicz_sharpe_ratio(zwroty)
    assert sharpe > 0
    
    # Test dla zmiennych zwrotów
    zwroty = [0.1, -0.2, 0.15, -0.1, 0.2]
    sharpe = oblicz_sharpe_ratio(zwroty)
    assert sharpe < oblicz_sharpe_ratio([0.1, 0.2, 0.15, 0.1, 0.2])  # Mniejszy Sharpe dla bardziej zmiennych zwrotów
    
    # Test dla samych strat
    zwroty = [-0.1, -0.2, -0.15]
    sharpe = oblicz_sharpe_ratio(zwroty)
    assert sharpe < 0
    
    # Test dla pustej listy
    sharpe = oblicz_sharpe_ratio([])
    assert sharpe == 0.0
    
    # Test z niezerową stopą wolną od ryzyka
    zwroty = [0.1, 0.2, 0.15, 0.1, 0.2]
    sharpe_z_rfr = oblicz_sharpe_ratio(zwroty, stopa_wolna_od_ryzyka=2.0)  # 2% rocznie
    assert sharpe_z_rfr < oblicz_sharpe_ratio(zwroty)  # Mniejszy Sharpe z uwzględnieniem RFR


def test_metryki_ryzyka():
    """Test obliczania kompletu metryk ryzyka."""
    transakcje = [
        WynikTransakcji(
            timestamp_wejscia=datetime.now(),
            timestamp_wyjscia=datetime.now(),
            kierunek=KierunekTransakcji.LONG,
            cena_wejscia=Decimal('1000.0'),
            cena_wyjscia=Decimal('1100.0'),
            wolumen=Decimal('1.0'),
            zysk_procent=10.0,
            powod_wyjscia="Test",
            metadane={}
        ),
        WynikTransakcji(
            timestamp_wejscia=datetime.now(),
            timestamp_wyjscia=datetime.now(),
            kierunek=KierunekTransakcji.SHORT,
            cena_wejscia=Decimal('1100.0'),
            cena_wyjscia=Decimal('1000.0'),
            wolumen=Decimal('1.0'),
            zysk_procent=9.09,
            powod_wyjscia="Test",
            metadane={}
        )
    ]
    
    historia_kapitalu = [100000.0, 110000.0, 120000.0]
    
    metryki = oblicz_metryki_ryzyka(transakcje, historia_kapitalu)
    
    assert 'profit_factor' in metryki
    assert 'max_drawdown' in metryki
    assert 'sharpe_ratio' in metryki
    assert 'win_rate' in metryki
    assert metryki['win_rate'] == 1.0  # Wszystkie transakcje zyskowne


def test_backtest_strategii():
    """Test pełnego procesu backtestingu strategii."""
    # Przygotowanie danych testowych
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = {
        'open': np.random.uniform(1000, 1100, len(dates)),
        'high': np.random.uniform(1050, 1150, len(dates)),
        'low': np.random.uniform(950, 1050, len(dates)),
        'close': np.random.uniform(1000, 1100, len(dates)),
        'volume': np.random.uniform(1000, 2000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    
    # Inicjalizacja symulatora i strategii
    symulator = SymulatorRynku(kapital_poczatkowy=100000.0)
    strategia = TestowaStrategia()
    
    # Przeprowadzenie backtestu
    wynik = symulator.testuj_strategie(strategia, df, "Test")
    
    assert isinstance(wynik, WynikBacktestu)
    assert wynik.nazwa_strategii == "Test"
    assert wynik.data_rozpoczecia == dates[0]
    assert wynik.data_zakonczenia == dates[-1]


def test_generowanie_raportow(tmp_path):
    """Test generowania raportów z wyników backtestingu."""
    # Przygotowanie danych testowych
    transakcje = [
        WynikTransakcji(
            timestamp_wejscia=datetime.now(),
            timestamp_wyjscia=datetime.now() + timedelta(days=1),
            kierunek=KierunekTransakcji.LONG,
            cena_wejscia=Decimal('1000.0'),
            cena_wyjscia=Decimal('1100.0'),
            wolumen=Decimal('1.0'),
            zysk_procent=10.0,
            powod_wyjscia="Test",
            metadane={}
        )
    ]
    
    wynik = WynikBacktestu(
        nazwa_strategii="Test",
        data_rozpoczecia=datetime.now(),
        data_zakonczenia=datetime.now() + timedelta(days=1),
        liczba_transakcji=1,
        zysk_calkowity=10.0,
        win_rate=1.0,
        profit_factor=float('inf'),
        max_drawdown=0.0,
        sharpe_ratio=2.0,
        transakcje=transakcje,
        metadane={}
    )
    
    # Test generowania raportów
    from backtesting.raporty import generuj_raport_html, generuj_raport_json
    
    # Utworzenie katalogu tymczasowego
    raporty_dir = tmp_path / "raporty"
    raporty_dir.mkdir()
    
    # Generowanie raportów
    generuj_raport_html(wynik, str(raporty_dir))
    generuj_raport_json(wynik, str(raporty_dir))
    
    # Sprawdzenie czy pliki zostały utworzone
    pliki = list(raporty_dir.glob("*"))
    assert len(pliki) >= 2  # HTML + JSON + wykresy
    
    # Sprawdzenie rozszerzeń plików
    rozszerzenia = {p.suffix for p in pliki}
    assert ".html" in rozszerzenia
    assert ".json" in rozszerzenia


def test_backtest_z_metrykami():
    """Test pełnego procesu backtestingu z nowymi metrykami."""
    # Przygotowanie danych testowych
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = {
        'open': np.random.uniform(1000, 1100, len(dates)),
        'high': np.random.uniform(1050, 1150, len(dates)),
        'low': np.random.uniform(950, 1050, len(dates)),
        'close': np.random.uniform(1000, 1100, len(dates)),
        'volume': np.random.uniform(1000, 2000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    
    # Inicjalizacja symulatora i strategii
    symulator = SymulatorRynku(kapital_poczatkowy=100000.0)
    strategia = TestowaStrategia()
    
    # Przeprowadzenie backtestu
    wynik = symulator.testuj_strategie(strategia, df)
    
    # Sprawdzenie nowych metryk
    assert isinstance(wynik.profit_factor, float)
    assert isinstance(wynik.max_drawdown, float)
    assert isinstance(wynik.sharpe_ratio, float)
    assert 'historia_kapitalu' in wynik.metadane
    assert 'zwroty_dzienne' in wynik.metadane
    assert 'szczyt_dd' in wynik.metadane
    assert 'dolek_dd' in wynik.metadane
    
    # Sprawdzenie spójności metryk
    if wynik.liczba_transakcji > 0:
        assert wynik.profit_factor >= 0.0
        assert wynik.max_drawdown >= 0.0
        assert len(wynik.metadane['historia_kapitalu']) == len(df) + 1  # +1 dla kapitału początkowego
        assert len(wynik.metadane['zwroty_dzienne']) == len(df) - 1  # -1 bo pierwszy dzień nie ma zwrotu 