"""
Moduł do obliczania metryk wydajności strategii w systemie NikkeiNinja.
"""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from backtesting.symulator import WynikTransakcji

logger = logging.getLogger(__name__)


def oblicz_profit_factor(transakcje: List[WynikTransakcji]) -> float:
    """
    Oblicza profit factor (stosunek zysków do strat).
    
    Profit Factor = Suma zysków / |Suma strat|
    Wartość > 1 oznacza strategię zyskowną.
    """
    if not transakcje:
        return 0.0
        
    zyski = sum(t.zysk_procent for t in transakcje if t.zysk_procent > 0)
    straty = abs(sum(t.zysk_procent for t in transakcje if t.zysk_procent < 0))
    
    return zyski / straty if straty > 0 else float('inf')


def oblicz_max_drawdown(kapital: List[float]) -> Tuple[float, int, int]:
    """
    Oblicza maksymalny drawdown i jego okres.
    
    Zwraca:
    - Maksymalny drawdown w procentach
    - Indeks początku drawdown
    - Indeks końca drawdown
    """
    if not kapital:
        return 0.0, 0, 0
        
    # Konwersja na array numpy dla wydajności
    kapital_arr = np.array(kapital)
    
    # Obliczanie szczytów
    max_kapital = np.maximum.accumulate(kapital_arr)
    drawdowns = (kapital_arr - max_kapital) / max_kapital * 100
    
    # Znajdowanie największego drawdown
    max_dd = np.min(drawdowns)
    end_idx = np.argmin(drawdowns)
    
    # Znajdowanie początku drawdown
    start_idx = np.argmax(kapital_arr[:end_idx])
    
    return float(abs(max_dd)), int(start_idx), int(end_idx)


def oblicz_sharpe_ratio(zwroty: List[float], 
                       stopa_wolna_od_ryzyka: float = 0.0) -> float:
    """
    Oblicza wskaźnik Sharpe'a dla strategii.
    
    Sharpe Ratio = (Średni zwrot - Stopa wolna od ryzyka) / Odchylenie standardowe zwrotów
    Wartość > 1 oznacza dobrą relację zysku do ryzyka.
    """
    if not zwroty:
        return 0.0
        
    zwroty_arr = np.array(zwroty)
    sredni_zwrot = np.mean(zwroty_arr)
    odch_std = np.std(zwroty_arr)
    
    if odch_std == 0:
        return 0.0
        
    return (sredni_zwrot - stopa_wolna_od_ryzyka) / odch_std


def oblicz_metryki_ryzyka(transakcje: List[WynikTransakcji],
                         historia_kapitalu: List[float]) -> dict:
    """
    Oblicza komplet metryk ryzyka dla strategii.
    
    Zwraca słownik z metrykami:
    - Profit Factor
    - Maximum Drawdown
    - Sharpe Ratio
    - Win Rate
    - Średni zysk
    - Średnia strata
    - Liczba transakcji
    """
    try:
        if not transakcje or not historia_kapitalu:
            return {
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_start': 0,
                'max_drawdown_end': 0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'liczba_transakcji': 0
            }
        
        # Podstawowe statystyki
        zyski = [t.zysk_procent for t in transakcje if t.zysk_procent > 0]
        straty = [t.zysk_procent for t in transakcje if t.zysk_procent < 0]
        
        # Obliczanie metryk
        profit_factor = oblicz_profit_factor(transakcje)
        max_dd, dd_start, dd_end = oblicz_max_drawdown(historia_kapitalu)
        
        # Obliczanie zwrotów dziennych
        zwroty = np.diff(historia_kapitalu) / historia_kapitalu[:-1] * 100
        sharpe = oblicz_sharpe_ratio(zwroty.tolist())
        
        return {
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'max_drawdown_start': dd_start,
            'max_drawdown_end': dd_end,
            'sharpe_ratio': sharpe,
            'win_rate': len(zyski) / len(transakcje) if transakcje else 0.0,
            'avg_win': np.mean(zyski) if zyski else 0.0,
            'avg_loss': np.mean(straty) if straty else 0.0,
            'liczba_transakcji': len(transakcje)
        }
        
    except Exception as e:
        logger.error("❌ Błąd podczas obliczania metryk: %s", str(e))
        return {
            'error': str(e)
        }


def generuj_raport_dzienny(transakcje: List[WynikTransakcji]) -> pd.DataFrame:
    """
    Generuje dzienny raport wyników.
    
    Zwraca DataFrame z kolumnami:
    - Data
    - Liczba transakcji
    - Zysk/strata
    - Skumulowany wynik
    """
    try:
        if not transakcje:
            return pd.DataFrame()
            
        # Konwersja na DataFrame
        df = pd.DataFrame([
            {
                'data': t.timestamp_wyjscia.date(),
                'zysk': t.zysk_procent
            }
            for t in transakcje
        ])
        
        # Grupowanie po dniach
        daily = df.groupby('data').agg({
            'zysk': ['count', 'sum']
        }).reset_index()
        
        daily.columns = ['data', 'liczba_transakcji', 'zysk']
        daily['skumulowany_wynik'] = daily['zysk'].cumsum()
        
        return daily
        
    except Exception as e:
        logger.error("❌ Błąd podczas generowania raportu dziennego: %s", str(e))
        return pd.DataFrame() 