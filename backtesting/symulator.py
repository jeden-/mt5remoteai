"""
Symulator rynku do backtestingu strategii w systemie NikkeiNinja.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from handel.analiza_techniczna import AnalizaTechniczna
from strategie.interfejs import IStrategia, KierunekTransakcji, SygnalTransakcyjny
from raporty.generator import generuj_raport_html

logger = logging.getLogger(__name__)


@dataclass
class WynikTransakcji:
    """Wynik pojedynczej transakcji w backteście."""
    timestamp_wejscia: datetime
    timestamp_wyjscia: datetime
    kierunek: KierunekTransakcji
    cena_wejscia: Decimal
    cena_wyjscia: Decimal
    wolumen: Decimal
    zysk_procent: float
    powod_wyjscia: str
    metadane: Dict


@dataclass
class WynikBacktestu:
    """Wynik całego backtestu dla strategii."""
    nazwa_strategii: str
    data_rozpoczecia: datetime
    data_zakonczenia: datetime
    liczba_transakcji: int
    zysk_calkowity: float
    zysk_procent: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    transakcje: List[WynikTransakcji]
    metadane: Dict
    sygnaly_long: List[SygnalTransakcyjny] = None  # Lista sygnałów kupna
    sygnaly_short: List[SygnalTransakcyjny] = None  # Lista sygnałów sprzedaży

    def __post_init__(self):
        """Inicjalizacja po utworzeniu obiektu."""
        if self.sygnaly_long is None:
            self.sygnaly_long = []
        if self.sygnaly_short is None:
            self.sygnaly_short = []

    def generuj_statystyki(self) -> Dict:
        """Generuje słownik ze statystykami backtestingu."""
        return {
            'nazwa_strategii': self.nazwa_strategii,
            'okres': f"{self.data_rozpoczecia.strftime('%Y-%m-%d')} - {self.data_zakonczenia.strftime('%Y-%m-%d')}",
            'liczba_transakcji': self.liczba_transakcji,
            'zysk_calkowity': f"{self.zysk_calkowity:.2f} JPY",
            'zysk_procent': f"{self.zysk_procent:.2f}%",
            'win_rate': f"{self.win_rate * 100:.1f}%",
            'profit_factor': f"{self.profit_factor:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'transakcje': [
                {
                    'data_wejscia': t.timestamp_wejscia.strftime('%Y-%m-%d %H:%M'),
                    'data_wyjscia': t.timestamp_wyjscia.strftime('%Y-%m-%d %H:%M'),
                    'kierunek': t.kierunek.value,
                    'cena_wejscia': f"{float(t.cena_wejscia):.2f}",
                    'cena_wyjscia': f"{float(t.cena_wyjscia):.2f}",
                    'wolumen': f"{float(t.wolumen):.2f}",
                    'zysk_procent': f"{t.zysk_procent:.2f}%",
                    'powod_wyjscia': t.powod_wyjscia
                }
                for t in self.transakcje
            ]
        }


class SymulatorRynku:
    """
    Symulator rynku do testowania strategii na danych historycznych.
    
    Funkcjonalności:
    - Symulacja wykonania zleceń
    - Śledzenie pozycji i kapitału
    - Obliczanie kosztów transakcyjnych
    - Generowanie statystyk
    """
    
    def __init__(self, 
                 kapital_poczatkowy: float = 100000.0,
                 prowizja: float = 0.001):
        """
        Inicjalizacja symulatora.
        
        Parametry:
        - kapital_poczatkowy: Kapitał początkowy w JPY
        - prowizja: Prowizja od transakcji w procentach
        """
        self.kapital_poczatkowy = Decimal(str(kapital_poczatkowy))
        self.prowizja = Decimal(str(prowizja))
        self.kapital_aktualny = self.kapital_poczatkowy
        self.pozycje: List[Tuple[str, KierunekTransakcji, Decimal, Decimal]] = []  # (symbol, kierunek, cena, wolumen)
        self.historia_transakcji: List[WynikTransakcji] = []
        self.wszystkie_sygnaly_long: List[SygnalTransakcyjny] = []  # Nowe
        self.wszystkie_sygnaly_short: List[SygnalTransakcyjny] = []  # Nowe
        logger.info("🥷 Zainicjalizowano symulator rynku (kapitał=%.2f JPY, prowizja=%.3f%%)", 
                   kapital_poczatkowy, prowizja * 100)
    
    def _oblicz_prowizje(self, cena: Decimal, wolumen: Decimal) -> Decimal:
        """Oblicza prowizję dla transakcji."""
        return (cena * wolumen * self.prowizja) / Decimal('100.0')
    
    def _aktualizuj_kapital(self, zmiana: Decimal) -> None:
        """Aktualizuje stan kapitału."""
        self.kapital_aktualny += zmiana
        logger.debug("🥷 Aktualizacja kapitału: %.2f JPY (zmiana: %.2f JPY)", 
                    self.kapital_aktualny, zmiana)
    
    def otworz_pozycje(self, 
                       timestamp: datetime,
                       symbol: str,
                       kierunek: KierunekTransakcji,
                       cena: float,
                       wolumen: float,
                       metadane: Optional[Dict] = None) -> bool:
        """
        Otwiera nową pozycję w symulacji.
        
        Parametry:
        - timestamp: Znacznik czasowy otwarcia
        - symbol: Symbol instrumentu
        - kierunek: Kierunek transakcji (LONG/SHORT)
        - cena: Cena wejścia
        - wolumen: Wielkość pozycji
        - metadane: Dodatkowe informacje o transakcji
        
        Zwraca:
        - True jeśli udało się otworzyć pozycję
        """
        try:
            # Sprawdzenie czy już mamy pozycję dla tego symbolu
            for sym, kier, _, _ in self.pozycje:
                if sym == symbol:
                    logger.debug("⚠️ Pozycja dla %s już istnieje", symbol)
                    return False
            
            cena_dec = Decimal(str(cena))
            wolumen_dec = Decimal(str(wolumen))
            wartosc = cena_dec * wolumen_dec
            prowizja = self._oblicz_prowizje(cena_dec, wolumen_dec)
            
            # Sprawdzenie dostępnego kapitału
            if wartosc + prowizja > self.kapital_aktualny:
                logger.warning("⚠️ Brak wystarczającego kapitału na otwarcie pozycji")
                return False
            
            # Otwarcie pozycji
            self.pozycje.append((symbol, kierunek, cena_dec, wolumen_dec))
            self._aktualizuj_kapital(-prowizja)  # Pobranie prowizji
            
            # Zapisanie wyniku otwarcia
            self.historia_transakcji.append(WynikTransakcji(
                timestamp_wejscia=timestamp,
                timestamp_wyjscia=timestamp,  # Tymczasowo to samo
                kierunek=kierunek,
                cena_wejscia=cena_dec,
                cena_wyjscia=cena_dec,  # Tymczasowo to samo
                wolumen=wolumen_dec,
                zysk_procent=0.0,  # Na razie 0
                powod_wyjscia="Pozycja otwarta",
                metadane=metadane or {}
            ))
            
            logger.info("🥷 Otwarto pozycję: %s %s %.2f @ %.2f", 
                       kierunek.value, symbol, float(wolumen), cena)
            return True
            
        except Exception as e:
            logger.error("❌ Błąd podczas otwierania pozycji: %s", str(e))
            return False
    
    def zamknij_pozycje(self,
                        timestamp: datetime,
                        symbol: str,
                        cena: float,
                        metadane: Optional[Dict] = None) -> bool:
        """
        Zamyka istniejącą pozycję w symulacji.
        
        Parametry:
        - timestamp: Znacznik czasowy zamknięcia
        - symbol: Symbol instrumentu
        - cena: Cena wyjścia
        - metadane: Dodatkowe informacje o transakcji
        
        Zwraca:
        - True jeśli udało się zamknąć pozycję
        """
        try:
            # Szukanie pozycji do zamknięcia
            for i, (sym, kier, cena_wej, wol) in enumerate(self.pozycje):
                if sym == symbol:
                    cena_wyj = Decimal(str(cena))
                    prowizja = self._oblicz_prowizje(cena_wyj, wol)
                    
                    # Obliczanie zysku/straty
                    if kier == KierunekTransakcji.LONG:
                        zysk = (cena_wyj - cena_wej) * wol
                    else:  # SHORT
                        zysk = (cena_wej - cena_wyj) * wol
                    
                    # Aktualizacja kapitału
                    self._aktualizuj_kapital(zysk - prowizja)
                    
                    # Aktualizacja wyniku w historii
                    for j, wynik in enumerate(self.historia_transakcji):
                        if (wynik.timestamp_wejscia == wynik.timestamp_wyjscia and  # Pozycja otwarta
                            wynik.kierunek == kier and
                            wynik.cena_wejscia == cena_wej and
                            wynik.wolumen == wol):
                            # Aktualizacja wyniku
                            self.historia_transakcji[j] = WynikTransakcji(
                                timestamp_wejscia=wynik.timestamp_wejscia,
                                timestamp_wyjscia=timestamp,
                                kierunek=kier,
                                cena_wejscia=cena_wej,
                                cena_wyjscia=cena_wyj,
                                wolumen=wol,
                                zysk_procent=float((zysk / (cena_wej * wol)) * Decimal('100.0')),
                                powod_wyjscia="Zamknięcie manualne",
                                metadane=metadane or {}
                            )
                            break
                    
                    # Usunięcie pozycji
                    self.pozycje.pop(i)
                    
                    logger.info("🥷 Zamknięto pozycję: %s %s %.2f @ %.2f (zysk: %.2f JPY)", 
                               kier.value, symbol, float(wol), cena, float(zysk))
                    return True
            
            logger.warning("⚠️ Nie znaleziono pozycji do zamknięcia: %s", symbol)
            return False
            
        except Exception as e:
            logger.error("❌ Błąd podczas zamykania pozycji: %s", str(e))
            return False
    
    def testuj_strategie(self,
                        strategia: IStrategia,
                        dane: pd.DataFrame,
                        generuj_wykresy: bool = False,
                        sciezka_raportu: Optional[Path] = None) -> WynikBacktestu:
        """
        Przeprowadza backtest strategii na danych historycznych.
        
        Parametry:
        - strategia: Instancja strategii do przetestowania
        - dane: DataFrame z danymi historycznymi
        - generuj_wykresy: Czy generować wykresy
        - sciezka_raportu: Ścieżka do zapisu raportu HTML
        
        Zwraca:
        - Wynik backtestu z metrykami
        """
        try:
            logger.info("🥷 Rozpoczynam backtest strategii")
            
            # Reset stanu symulatora
            self.kapital_aktualny = self.kapital_poczatkowy
            self.pozycje = []
            self.historia_transakcji = []
            self.wszystkie_sygnaly_long = []
            self.wszystkie_sygnaly_short = []
            
            # Historia kapitału do obliczania drawdown i zwrotów
            historia_kapitalu = [float(self.kapital_poczatkowy)]
            zwroty_dzienne = []
            
            # Iteracja po danych
            for i in range(len(dane)):
                window = dane.iloc[:i+1]
                
                # Analiza strategii
                sygnaly = strategia.analizuj(window)
                
                # Segregacja sygnałów
                for sygnal in sygnaly:
                    if sygnal.get('typ') == 'LONG':
                        self.wszystkie_sygnaly_long.append(sygnal)
                    else:
                        self.wszystkie_sygnaly_short.append(sygnal)
                    
                    # Próba otwarcia pozycji
                    self.otworz_pozycje(
                        timestamp=sygnal['timestamp'],
                        symbol=sygnal['symbol'],
                        kierunek=KierunekTransakcji.LONG if sygnal['typ'] == 'LONG' else KierunekTransakcji.SHORT,
                        cena=float(sygnal['cena']),
                        wolumen=1.0,  # Stała wielkość pozycji
                        metadane=sygnal.get('metadane', {})
                    )
                
                # Aktualizacja otwartych pozycji
                for pozycja in list(self.pozycje):  # Kopia listy, bo będziemy modyfikować
                    symbol, kierunek, cena_wejscia, wolumen = pozycja
                    sygnaly_zamkniecia = strategia.aktualizuj(
                        df=window,
                        symbol=symbol,
                        kierunek=kierunek,
                        cena_wejscia=float(cena_wejscia)
                    )
                    
                    # Zamykanie pozycji
                    for sygnal in sygnaly_zamkniecia:
                        self.zamknij_pozycje(
                            timestamp=window.index[-1],  # Aktualny timestamp
                            symbol=sygnal['symbol'],
                            cena=float(sygnal['cena']),
                            metadane=sygnal.get('metadane', {})
                        )
                
                # Aktualizacja historii kapitału
                historia_kapitalu.append(float(self.kapital_aktualny))
                if i > 0:
                    zwrot = (historia_kapitalu[-1] - historia_kapitalu[-2]) / historia_kapitalu[-2] * 100
                    zwroty_dzienne.append(zwrot)
            
            # Obliczanie metryk
            zysk_calkowity = float(self.kapital_aktualny - self.kapital_poczatkowy)
            zysk_procent = (zysk_calkowity / float(self.kapital_poczatkowy)) * 100
            
            if self.historia_transakcji:
                win_rate = len([t for t in self.historia_transakcji if t.zysk_procent > 0]) / len(self.historia_transakcji)
                profit_factor = oblicz_profit_factor(self.historia_transakcji)
            else:
                win_rate = 0.0
                profit_factor = 0.0
            
            max_dd, szczyt_idx, dolek_idx = oblicz_max_drawdown(historia_kapitalu)
            sharpe = oblicz_sharpe_ratio(zwroty_dzienne)
            
            wynik = WynikBacktestu(
                nazwa_strategii=strategia.__class__.__name__,
                data_rozpoczecia=dane.index[0],
                data_zakonczenia=dane.index[-1],
                liczba_transakcji=len(self.historia_transakcji),
                zysk_calkowity=zysk_calkowity,
                zysk_procent=zysk_procent,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                transakcje=self.historia_transakcji,
                metadane={
                    'historia_kapitalu': historia_kapitalu,
                    'zwroty_dzienne': zwroty_dzienne,
                    'szczyt_dd': dane.index[szczyt_idx],
                    'dolek_dd': dane.index[dolek_idx]
                },
                sygnaly_long=self.wszystkie_sygnaly_long,
                sygnaly_short=self.wszystkie_sygnaly_short
            )
            
            # Generowanie raportu
            if generuj_wykresy and sciezka_raportu:
                generuj_raport_html(wynik, dane, sciezka_raportu)
            
            return wynik
            
        except Exception as e:
            logger.error("❌ Błąd podczas backtestingu: %s", str(e))
            return WynikBacktestu(
                nazwa_strategii=strategia.__class__.__name__,
                data_rozpoczecia=dane.index[0],
                data_zakonczenia=dane.index[-1],
                liczba_transakcji=0,
                zysk_calkowity=0.0,
                zysk_procent=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                transakcje=[],
                metadane={},
                sygnaly_long=self.wszystkie_sygnaly_long,
                sygnaly_short=self.wszystkie_sygnaly_short
            )

def oblicz_profit_factor(transakcje: List['WynikTransakcji']) -> float:
    """
    Oblicza profit factor na podstawie historii transakcji.
    
    Args:
        transakcje: Lista transakcji
        
    Returns:
        float: Profit factor (suma zysków / suma strat)
    """
    if not transakcje:
        return 0.0
        
    zyski = sum(t.zysk_procent for t in transakcje if t.zysk_procent > 0)
    straty = abs(sum(t.zysk_procent for t in transakcje if t.zysk_procent < 0))
    
    return zyski / straty if straty > 0 else float('inf')

def oblicz_max_drawdown(kapital: List[float]) -> Tuple[float, int, int]:
    """
    Oblicza maksymalny drawdown z historii kapitału.
    
    Args:
        kapital: Lista wartości kapitału
        
    Returns:
        Tuple[float, int, int]: (drawdown w procentach, indeks szczytu, indeks dołka)
    """
    if not kapital:
        return 0.0, 0, 0
        
    max_dd = 0.0
    szczyt_idx = 0
    dolek_idx = 0
    
    for i in range(len(kapital)):
        for j in range(i + 1, len(kapital)):
            dd = (kapital[i] - kapital[j]) / kapital[i] * 100
            if dd > max_dd:
                max_dd = dd
                szczyt_idx = i
                dolek_idx = j
                
    return max_dd, szczyt_idx, dolek_idx

def oblicz_sharpe_ratio(zwroty: List[float], stopa_wolna_od_ryzyka: float = 0.0) -> float:
    """
    Oblicza wskaźnik Sharpe'a dla serii zwrotów.
    
    Args:
        zwroty: Lista zwrotów procentowych
        stopa_wolna_od_ryzyka: Roczna stopa wolna od ryzyka (domyślnie 0%)
        
    Returns:
        float: Wskaźnik Sharpe'a
    """
    if not zwroty:
        return 0.0
        
    zwroty = np.array(zwroty)
    nadwyzka = zwroty - (stopa_wolna_od_ryzyka / 252)  # Dzienna stopa
    
    if len(nadwyzka) < 2:
        return 0.0
        
    return np.mean(nadwyzka) / np.std(nadwyzka, ddof=1) * np.sqrt(252)  # Annualizacja 