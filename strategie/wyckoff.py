"""
Strategia oparta na teorii Wyckoffa w systemie NikkeiNinja.
"""
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from handel.analiza_techniczna import AnalizaTechniczna
from strategie.interfejs import IStrategia, KierunekTransakcji, SygnalTransakcyjny

logger = logging.getLogger(__name__)


class FazaWyckoff(Enum):
    """Fazy rynku według teorii Wyckoffa."""
    AKUMULACJA = "AKUMULACJA"
    WZROST = "WZROST"
    DYSTRYBUCJA = "DYSTRYBUCJA"
    SPADEK = "SPADEK"
    NIEZNANA = "NIEZNANA"


class StrategiaWyckoff(IStrategia):
    """
    Strategia wykorzystująca teorię Wyckoffa:
    - Analiza faz rynku (akumulacja, wzrost, dystrybucja, spadek)
    - Analiza wolumenu i price spread
    - Identyfikacja punktów zwrotnych (spring, upthrust)
    - Potwierdzenie trendu przez wolumen
    """
    
    def __init__(self, parametry: Dict[str, Any] = None):
        """Inicjalizacja strategii Wyckoffa."""
        self.logger = logging.getLogger(__name__)
        self.faza_rynku = FazaWyckoff.NIEZNANA
        self.analiza = AnalizaTechniczna()
        self.parametry = {}
        self.dane_historyczne = pd.DataFrame()
        self.aktywne_sygnaly = []  # Lista aktywnych sygnałów
        self.logger.info("🥷 Zainicjalizowano strategię Wyckoffa")
        if parametry:
            self.inicjalizuj(parametry)
    
    def inicjalizuj(self, parametry: Dict) -> None:
        """
        Inicjalizuje parametry strategii.
        
        Parametry:
        - okres_ma: okres średniej kroczącej (domyślnie 20)
        - min_spread_mult: minimalny spread jako mnożnik ATR (domyślnie 0.5)
        - min_vol_mult: minimalny wolumen jako mnożnik średniego wolumenu (domyślnie 1.1)
        - sl_atr: mnożnik ATR dla stop loss (domyślnie 2)
        - tp_atr: mnożnik ATR dla take profit (domyślnie 3)
        
        Nowe parametry:
        - vol_std_mult: mnożnik odchylenia standardowego wolumenu (domyślnie 1.5)
        - min_swing_candles: minimalna liczba świec dla potwierdzenia formacji (domyślnie 3)
        - rsi_okres: okres dla RSI (domyślnie 14)
        - rsi_min: minimalny poziom RSI dla spring (domyślnie 30)
        - rsi_max: maksymalny poziom RSI dla upthrust (domyślnie 70)
        - trend_momentum: minimalna wartość momentum dla trendu (domyślnie 0.1)
        """
        self.parametry = {
            'okres_ma': 20,
            'min_spread_mult': 0.5,
            'min_vol_mult': 1.1,
            'sl_atr': 2.0,
            'tp_atr': 3.0,
            # Nowe parametry
            'vol_std_mult': 1.5,
            'min_swing_candles': 3,
            'rsi_okres': 14,
            'rsi_min': 30,
            'rsi_max': 70,
            'trend_momentum': 0.1,
            **parametry
        }
        logger.info("🥷 Zaktualizowano parametry strategii: %s", self.parametry)
    
    def _identyfikuj_faze(self, df: pd.DataFrame) -> FazaWyckoff:
        """
        Identyfikuje aktualną fazę rynku według teorii Wyckoffa.
        
        Logika:
        1. Obliczanie trendów dla cen, wolumenu i RSI
        2. Przyznawanie punktów dla każdej fazy na podstawie warunków
        3. Wybór fazy z najwyższą liczbą punktów
        """
        try:
            # Sprawdzamy czy mamy wystarczająco danych
            if len(df) < 20:
                return FazaWyckoff.NIEZNANA
                
            # Obliczamy trendy
            ceny_trend = np.polyfit(range(len(df)), df['close'], 1)[0]
            wolumen_trend = np.polyfit(range(len(df)), df['volume'], 1)[0]
            momentum = ceny_trend / df['close'].mean()
            
            # Obliczamy RSI
            ostatni_rsi = self.analiza.oblicz_rsi(df['close'])[-1]
            rsi_trend = np.polyfit(range(len(df)), self.analiza.oblicz_rsi(df['close']), 1)[0]
            
            # Inicjalizacja punktów dla każdej fazy
            punkty = {
                FazaWyckoff.AKUMULACJA: 0,
                FazaWyckoff.WZROST: 0,
                FazaWyckoff.DYSTRYBUCJA: 0,
                FazaWyckoff.SPADEK: 0
            }
            
            # Punkty dla fazy akumulacji
            if abs(ceny_trend) < self.parametry['trend_momentum']:  # Trend boczny
                punkty[FazaWyckoff.AKUMULACJA] += 2
                self.logger.debug("🥷 Akumulacja +2: Trend boczny")
            if ceny_trend < 0:  # Trend spadkowy lub boczny
                punkty[FazaWyckoff.AKUMULACJA] += 1
                self.logger.debug("🥷 Akumulacja +1: Trend spadkowy")
            if ostatni_rsi <= 35:  # Niski RSI
                punkty[FazaWyckoff.AKUMULACJA] += 1
                self.logger.debug("🥷 Akumulacja +1: Niski RSI")
            if rsi_trend > 0:  # Rosnący RSI
                punkty[FazaWyckoff.AKUMULACJA] += 1
                self.logger.debug("🥷 Akumulacja +1: Rosnący RSI")
            if wolumen_trend > 0:  # Rosnący wolumen
                punkty[FazaWyckoff.AKUMULACJA] += 1
                self.logger.debug("🥷 Akumulacja +1: Rosnący wolumen")
            
            # Punkty dla fazy wzrostu
            if ceny_trend > self.parametry['trend_momentum']:  # Trend wzrostowy
                punkty[FazaWyckoff.WZROST] += 3
                self.logger.debug("🥷 Wzrost +3: Trend wzrostowy")
            if wolumen_trend > 0:  # Rosnący wolumen
                punkty[FazaWyckoff.WZROST] += 2
                self.logger.debug("🥷 Wzrost +2: Rosnący wolumen")
            if ostatni_rsi > 55:  # RSI powyżej 55
                punkty[FazaWyckoff.WZROST] += 2
                self.logger.debug("🥷 Wzrost +2: RSI powyżej 55")
            
            # Punkty dla fazy dystrybucji
            if abs(ceny_trend) < self.parametry['trend_momentum']:  # Trend boczny
                punkty[FazaWyckoff.DYSTRYBUCJA] += 2
                self.logger.debug("🥷 Dystrybucja +2: Trend boczny")
            if wolumen_trend < 0:  # Spadający wolumen
                punkty[FazaWyckoff.DYSTRYBUCJA] += 3
                self.logger.debug("🥷 Dystrybucja +3: Spadający wolumen")
            if ostatni_rsi > 70:  # Wysoki RSI
                punkty[FazaWyckoff.DYSTRYBUCJA] += 2
                self.logger.debug("🥷 Dystrybucja +2: Wysoki RSI")
            if rsi_trend < 0:  # Spadający RSI
                punkty[FazaWyckoff.DYSTRYBUCJA] += 2
                self.logger.debug("🥷 Dystrybucja +2: Spadający RSI")
            
            # Punkty dla fazy spadku
            if ceny_trend < -self.parametry['trend_momentum']:  # Trend spadkowy
                punkty[FazaWyckoff.SPADEK] += 3
                self.logger.debug("🥷 Spadek +3: Trend spadkowy")
            if wolumen_trend < 0:  # Spadający wolumen
                punkty[FazaWyckoff.SPADEK] += 2
                self.logger.debug("🥷 Spadek +2: Spadający wolumen")
            if ostatni_rsi < 35:  # Niski RSI
                punkty[FazaWyckoff.SPADEK] += 2
                self.logger.debug("🥷 Spadek +2: Niski RSI")
            
            # Znajdujemy fazy z maksymalną liczbą punktów
            max_punkty = max(punkty.values())
            max_fazy = [f for f, p in punkty.items() if p == max_punkty]
            
            # Inicjalizujemy słownik siły warunków dla każdej fazy
            sila_warunkow = {
                FazaWyckoff.SPADEK: abs(ceny_trend) if ceny_trend < 0 else 0,
                FazaWyckoff.WZROST: abs(ceny_trend) if ceny_trend > 0 else 0,
                FazaWyckoff.DYSTRYBUCJA: (abs(wolumen_trend) if wolumen_trend < 0 else 0) + (abs(rsi_trend) if rsi_trend < 0 and ostatni_rsi > 70 else 0),
                FazaWyckoff.AKUMULACJA: (abs(wolumen_trend) if wolumen_trend > 0 else 0) + (abs(rsi_trend) if rsi_trend > 0 and ostatni_rsi < 35 else 0)
            }
            
            # Jeśli jest tylko jedna faza z max punktami, wybieramy ją
            if len(max_fazy) == 1:
                faza = max_fazy[0]
            else:
                # W przypadku remisu, wybieramy fazę z najsilniejszymi warunkami
                faza = max(max_fazy, key=lambda f: sila_warunkow[f])
            
            self.logger.debug(f"🥷 Punkty dla faz: {punkty}")
            self.logger.debug(f"🥷 Siła warunków: {sila_warunkow}")
            self.logger.debug(f"🥷 Wybrana faza: {faza}")
            
            return faza
            
        except Exception as e:
            self.logger.error(f"❌ Błąd identyfikacji fazy: {str(e)}")
            return FazaWyckoff.NIEZNANA
    
    def _wykryj_spring(self, df: pd.DataFrame, i: int) -> bool:
        """Wykrywa formację spring."""
        try:
            # Sprawdzamy czy mamy wystarczająco danych
            if i < 19 or i >= len(df):
                self.logger.debug("🥷 Spring: Niewystarczająco danych")
                return False

            # Pobieramy dane
            last_low = df['low'].iloc[i]
            last_close = df['close'].iloc[i]
            last_volume = df['volume'].iloc[i]
            last_open = df['open'].iloc[i]
            
            # Analiza poprzednich świec
            min_low_10 = df['low'].iloc[i-10:i].min()
            min_low_20 = df['low'].iloc[i-20:i].min()
            avg_volume = df['volume'].iloc[i-10:i].mean()
            avg_volume_std = df['volume'].iloc[i-10:i].std()
            
            # Obliczamy trend przed formacją (ostatnie 15 świec)
            ceny_przed = df['close'].iloc[i-15:i].values
            trend_przed = np.polyfit(np.arange(len(ceny_przed)), ceny_przed, 1)[0]
            
            # Sprawdzamy RSI
            rsi = df['rsi'].iloc[i] if 'rsi' in df.columns else self.analiza.oblicz_rsi(df['close'])[-1]
            rsi_wartosci = df['rsi'].iloc[i-5:i+1].values
            rsi_trend = np.polyfit(np.arange(len(rsi_wartosci)), rsi_wartosci, 1)[0]
            
            # Logowanie wartości
            self.logger.debug(f"🥷 Spring: last_low={last_low:.2f} min_low_10={min_low_10:.2f}")
            self.logger.debug(f"🥷 Spring: last_close={last_close:.2f} last_open={last_open:.2f}")
            self.logger.debug(f"🥷 Spring: last_volume={last_volume:.0f} avg_volume={avg_volume:.0f}")
            self.logger.debug(f"🥷 Spring: trend_przed={trend_przed:.3f}")
            self.logger.debug(f"🥷 Spring: rsi={rsi:.1f} rsi_trend={rsi_trend:.3f}")
            
            # Warunki dla spring:
            if (last_low <= min_low_10 and  # Nowe minimum w ostatnich 10 świecach (lub równe)
                last_close > last_low * 1.01 and  # Zamknięcie wyraźnie powyżej minimum
                last_close >= last_open and  # Biała świeca (lub równa)
                last_volume > avg_volume * 1.2 and  # Podwyższony wolumen
                trend_przed < -0.05 and  # Wyraźny trend spadkowy przed formacją
                rsi <= 35 and  # Niski RSI
                rsi_trend > 0):  # Rosnący RSI
                
                self.logger.debug("🥷 Spring: Wszystkie warunki spełnione")
                return True
                
            self.logger.debug("🥷 Spring: Warunki nie spełnione")
            return False

        except Exception as e:
            self.logger.error(f"❌ Błąd wykrywania spring: {str(e)}")
            return False
    
    def _wykryj_upthrust(self, df: pd.DataFrame, i: int) -> bool:
        """Wykrywa formację upthrust."""
        try:
            # Sprawdzamy czy mamy wystarczająco danych
            if i < 19 or i >= len(df):
                self.logger.debug("🥷 Upthrust: Niewystarczająco danych")
                return False

            # Pobieramy dane
            last_high = df['high'].iloc[i]
            last_close = df['close'].iloc[i]
            last_volume = df['volume'].iloc[i]
            last_open = df['open'].iloc[i]
            
            # Analiza poprzednich świec
            max_high_10 = df['high'].iloc[i-10:i].max()
            max_high_20 = df['high'].iloc[i-20:i].max()
            avg_volume = df['volume'].iloc[i-10:i].mean()
            avg_volume_std = df['volume'].iloc[i-10:i].std()
            avg_close = df['close'].iloc[i-10:i].mean()
            
            # Obliczamy trend przed formacją (ostatnie 15 świec)
            ceny_przed = df['close'].iloc[i-15:i].values
            trend_przed = np.polyfit(np.arange(len(ceny_przed)), ceny_przed, 1)[0]
            
            # Sprawdzamy RSI
            rsi = df['rsi'].iloc[i] if 'rsi' in df.columns else self.analiza.oblicz_rsi(df['close'])[-1]
            rsi_wartosci = df['rsi'].iloc[i-5:i+1].values
            rsi_trend = np.polyfit(np.arange(len(rsi_wartosci)), rsi_wartosci, 1)[0]
            
            # Logowanie wartości
            self.logger.debug(f"🥷 Upthrust: last_high={last_high:.2f} max_high_10={max_high_10:.2f}")
            self.logger.debug(f"🥷 Upthrust: last_close={last_close:.2f} avg_close={avg_close:.2f}")
            self.logger.debug(f"🥷 Upthrust: last_volume={last_volume:.0f} avg_volume={avg_volume:.0f}")
            self.logger.debug(f"🥷 Upthrust: trend_przed={trend_przed:.3f}")
            self.logger.debug(f"🥷 Upthrust: rsi={rsi:.1f} rsi_trend={rsi_trend:.3f}")
            
            # Warunki dla upthrust:
            if (last_high >= max_high_10 and  # Nowe maksimum w ostatnich 10 świecach (lub równe)
                last_close < last_high * 0.99 and  # Zamknięcie wyraźnie poniżej maksimum
                last_close <= last_open and  # Czarna świeca (lub równa)
                last_volume > avg_volume * 1.2 and  # Podwyższony wolumen
                trend_przed > 0.05 and  # Wyraźny trend wzrostowy przed formacją
                rsi >= 70 and  # Wysoki RSI
                rsi_trend < 0):  # Spadający RSI
                
                self.logger.debug("🥷 Upthrust: Wszystkie warunki spełnione")
                return True
                
            self.logger.debug("🥷 Upthrust: Warunki nie spełnione")
            return False

        except Exception as e:
            self.logger.error(f"❌ Błąd wykrywania upthrust: {str(e)}")
            return False
    
    def analizuj(self, df: pd.DataFrame) -> List[SygnalTransakcyjny]:
        try:
            if len(df) < 20:
                self.logger.debug("🥷 Analizuj: Niewystarczająco danych")
                return []

            sygnaly = self.aktywne_sygnaly.copy()  # Zaczynamy od aktywnych sygnałów
            self.dane_historyczne = df.copy()
            
            # Identyfikacja fazy rynku
            self.faza_rynku = self._identyfikuj_faze(df)
            self.logger.info("🥷 Analizuj: Zidentyfikowana faza: %s", self.faza_rynku)
            
            # Wykrywanie formacji dla ostatniej świecy
            i = len(df) - 1
            spring = self._wykryj_spring(df, i)
            upthrust = self._wykryj_upthrust(df, i)
            
            self.logger.debug(f"🥷 Analizuj: spring={spring} upthrust={upthrust}")
            
            # Generowanie sygnałów na podstawie fazy i formacji
            nowy_sygnal = None
            
            if self.faza_rynku == FazaWyckoff.AKUMULACJA and spring:
                # Sprawdź czy nie mamy już aktywnego sygnału LONG
                if not any(s.kierunek == KierunekTransakcji.LONG for s in sygnaly):
                    self.logger.debug("🥷 Analizuj: Generuję sygnał LONG (spring w akumulacji)")
                    nowy_sygnal = SygnalTransakcyjny(
                        kierunek=KierunekTransakcji.LONG,
                        cena_wejscia=df['close'].iloc[i],
                        stop_loss=df['low'].iloc[i:i+1].min(),
                        take_profit=df['close'].iloc[i] * (1 + self.parametry['tp_atr'] * self.analiza.oblicz_atr(
                            df['high'].values,
                            df['low'].values,
                            df['close'].values
                        )[-1]),
                        symbol=df['symbol'].iloc[i],
                        opis="Sygnał LONG - wykryto formację spring w fazie akumulacji",
                        timestamp=datetime.now(),
                        wolumen=1.0,
                        metadane={
                            'faza': FazaWyckoff.AKUMULACJA.value,
                            'formacja': 'spring'
                        }
                    )
                    self.logger.info("🥷 Analizuj: Wygenerowano sygnał LONG (spring w akumulacji)")
                
            elif self.faza_rynku == FazaWyckoff.DYSTRYBUCJA and upthrust:
                # Sprawdź czy nie mamy już aktywnego sygnału SHORT
                if not any(s.kierunek == KierunekTransakcji.SHORT for s in sygnaly):
                    self.logger.debug("🥷 Analizuj: Generuję sygnał SHORT (upthrust w dystrybucji)")
                    nowy_sygnal = SygnalTransakcyjny(
                        kierunek=KierunekTransakcji.SHORT,
                        cena_wejscia=df['close'].iloc[i],
                        stop_loss=df['high'].iloc[i:i+1].max(),
                        take_profit=df['close'].iloc[i] * (1 - self.parametry['tp_atr'] * self.analiza.oblicz_atr(
                            df['high'].values,
                            df['low'].values,
                            df['close'].values
                        )[-1]),
                        symbol=df['symbol'].iloc[i],
                        opis="Sygnał SHORT - wykryto formację upthrust w fazie dystrybucji",
                        timestamp=datetime.now(),
                        wolumen=1.0,
                        metadane={
                            'faza': FazaWyckoff.DYSTRYBUCJA.value,
                            'formacja': 'upthrust'
                        }
                    )
                    self.logger.info("🥷 Analizuj: Wygenerowano sygnał SHORT (upthrust w dystrybucji)")
            else:
                self.logger.debug(f"🥷 Analizuj: Brak nowego sygnału (faza={self.faza_rynku}, spring={spring}, upthrust={upthrust})")

            # Dodaj nowy sygnał jeśli został wygenerowany
            if nowy_sygnal:
                sygnaly.append(nowy_sygnal)
                self.aktywne_sygnaly = sygnaly  # Aktualizuj listę aktywnych sygnałów

            return sygnaly
            
        except Exception as e:
            self.logger.error(f"❌ Błąd analizy: {str(e)}")
            return []
            
    def aktualizuj(self,
                   df: pd.DataFrame,
                   aktywne_pozycje: List[Tuple[str, KierunekTransakcji, float]]) -> List[SygnalTransakcyjny]:
        """
        Aktualizuje stan strategii i generuje sygnały na podstawie nowych danych.
        
        Parametry:
        - df: DataFrame z danymi OHLCV
        - aktywne_pozycje: Lista krotek (symbol, kierunek, cena_wejscia)
        
        Zwraca:
        - Lista sygnałów transakcyjnych
        """
        try:
            sygnaly = []
            
            # Identyfikacja fazy rynku
            self.faza_rynku = self._identyfikuj_faze(df)
            self.logger.info("🥷 Zidentyfikowana faza rynku: %s", self.faza_rynku)
            
            # Jeśli faza jest nieznana, nie generujemy sygnałów
            if self.faza_rynku == FazaWyckoff.NIEZNANA:
                return []
            
            # Obliczanie ATR
            atr = self.analiza.oblicz_atr(
                df['high'].values,
                df['low'].values,
                df['close'].values
            )[-1]
            
            # Sprawdzamy każdą aktywną pozycję
            for symbol, kierunek, cena_wejscia in aktywne_pozycje:
                cena_aktualna = df['close'].iloc[-1]
                
                # Obliczenie zysku z aktualnej pozycji
                if kierunek == KierunekTransakcji.LONG:
                    zysk_procent = (cena_aktualna - cena_wejscia) / cena_wejscia * 100
                else:
                    zysk_procent = (cena_wejscia - cena_aktualna) / cena_wejscia * 100
                
                # Zamykanie pozycji w przeciwnych fazach
                if (kierunek == KierunekTransakcji.LONG and 
                    self.faza_rynku in [FazaWyckoff.DYSTRYBUCJA, FazaWyckoff.SPADEK]):
                    sl = cena_aktualna + (atr * self.parametry['sl_atr'])
                    tp = cena_aktualna - (atr * self.parametry['tp_atr'])
                    sygnaly.append(SygnalTransakcyjny(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        kierunek=KierunekTransakcji.SHORT,  # Zamknięcie LONG
                        cena_wejscia=cena_aktualna,
                        stop_loss=sl,
                        take_profit=tp,
                        wolumen=1.0,
                        opis=f"Wyckoff: Zamknięcie LONG w fazie {self.faza_rynku.value}",
                        metadane={
                            'faza': self.faza_rynku.value,
                            'zysk_procent': zysk_procent
                        }
                    ))
                
                elif (kierunek == KierunekTransakcji.SHORT and 
                      self.faza_rynku in [FazaWyckoff.AKUMULACJA, FazaWyckoff.WZROST]):
                    sl = cena_aktualna - (atr * self.parametry['sl_atr'])
                    tp = cena_aktualna + (atr * self.parametry['tp_atr'])
                    sygnaly.append(SygnalTransakcyjny(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        kierunek=KierunekTransakcji.LONG,  # Zamknięcie SHORT
                        cena_wejscia=cena_aktualna,
                        stop_loss=sl,
                        take_profit=tp,
                        wolumen=1.0,
                        opis=f"Wyckoff: Zamknięcie SHORT w fazie {self.faza_rynku.value}",
                        metadane={
                            'faza': self.faza_rynku.value,
                            'zysk_procent': zysk_procent
                        }
                    ))
            
            return sygnaly
                    
        except Exception as e:
            self.logger.error("❌ Błąd podczas aktualizacji: %s", str(e))
            return []
            
    def optymalizuj(self,
                    dane_historyczne: pd.DataFrame,
                    parametry_zakres: Dict) -> Dict:
        """
        Optymalizuje parametry strategii na podstawie danych historycznych.
        
        Parametry:
        - dane_historyczne: DataFrame z danymi do optymalizacji
        - parametry_zakres: Słownik z zakresami parametrów do optymalizacji
        
        Zwraca:
        - Słownik z optymalnymi parametrami
        """
        try:
            self.logger.info("🥷 Rozpoczynam optymalizację parametrów")
            
            # Zachowujemy obecne parametry
            obecne_parametry = self.parametry.copy()
            
            najlepszy_wynik = None
            najlepsze_parametry = None
            
            # Grid search po wszystkich kombinacjach parametrów
            from itertools import product
            
            # Przygotowanie wartości do testowania
            parametry_wartosci = []
            parametry_nazwy = []
            for nazwa, wartosci in parametry_zakres.items():
                parametry_wartosci.append(wartosci)
                parametry_nazwy.append(nazwa)
            
            # Testowanie wszystkich kombinacji
            for wartosci in product(*parametry_wartosci):
                # Ustawienie parametrów
                parametry_testowe = obecne_parametry.copy()
                for nazwa, wartosc in zip(parametry_nazwy, wartosci):
                    parametry_testowe[nazwa] = wartosc
                
                self.inicjalizuj(parametry_testowe)
                
                # Backtest
                symulator = SymulatorRynku(kapital_poczatkowy=100000.0)
                wynik = symulator.testuj_strategie(self, dane_historyczne)
                
                # Ocena wyniku (przykładowa metryka)
                score = wynik.zysk_procent * wynik.win_rate
                
                # Aktualizacja najlepszego wyniku
                if najlepszy_wynik is None or score > najlepszy_wynik:
                    najlepszy_wynik = score
                    najlepsze_parametry = parametry_testowe
                    
                    self.logger.info("🥷 Znaleziono lepsze parametry:")
                    for nazwa, wartosc in parametry_testowe.items():
                        if nazwa in parametry_zakres:
                            self.logger.info("  %s: %.3f", nazwa, wartosc)
                    self.logger.info("  Score: %.2f", score)
            
            # Przywracamy poprzednie parametry
            self.inicjalizuj(obecne_parametry)
            
            return najlepsze_parametry if najlepsze_parametry else self.parametry
            
        except Exception as e:
            self.logger.error("❌ Błąd podczas optymalizacji: %s", str(e))
            return self.parametry
            
    def generuj_statystyki(self, historia: List[SygnalTransakcyjny]) -> Dict:
        """
        Generuje statystyki dla strategii na podstawie historii transakcji.
        
        Parametry:
        - historia: Lista sygnałów transakcyjnych z historii
        
        Zwraca:
        - Słownik ze statystykami (win rate, zysk średni, itp.)
        """
        try:
            if not historia:
                return {}
                
            # Podstawowe statystyki
            liczba_transakcji = len(historia)
            zyskowne = sum(1 for s in historia if s.metadane.get('zysk_procent', 0) > 0)
            win_rate = zyskowne / liczba_transakcji if liczba_transakcji > 0 else 0
            
            # Statystyki per faza
            statystyki = {
                'liczba_transakcji': liczba_transakcji,
                'win_rate': win_rate
            }
            
            # Grupowanie po fazach
            for faza in FazaWyckoff:
                transakcje_w_fazie = [s for s in historia if s.metadane.get('faza') == faza.value]
                if transakcje_w_fazie:
                    liczba = len(transakcje_w_fazie)
                    zyskowne = sum(1 for s in transakcje_w_fazie if s.metadane.get('zysk_procent', 0) > 0)
                    zyski = [s.metadane.get('zysk_procent', 0) for s in transakcje_w_fazie]
                    
                    statystyki[f'liczba_transakcji_{faza.value}'] = liczba
                    statystyki[f'win_rate_{faza.value}'] = zyskowne / liczba
                    statystyki[f'zysk_sredni_{faza.value}'] = sum(zyski) / liczba
            
            return statystyki
                    
        except Exception as e:
            self.logger.error("❌ Błąd podczas generowania statystyk: %s", str(e))
            return {}

    def _generuj_sygnal_long(self, df: pd.DataFrame, indeks: int) -> SygnalTransakcyjny:
        """Generuje sygnał kupna."""
        cena = df['close'].iloc[indeks]
        atr = self.analiza.oblicz_atr(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )[-1]
        
        return SygnalTransakcyjny(
            timestamp=df.index[indeks],
            symbol=df['symbol'].iloc[indeks],
            kierunek=KierunekTransakcji.LONG,
            cena_wejscia=cena,
            stop_loss=cena - (atr * self.parametry['sl_atr']),
            take_profit=cena + (atr * self.parametry['tp_atr']),
            wolumen=1.0,
            opis="Wyckoff: Spring w fazie akumulacji",
            metadane={
                'faza': FazaWyckoff.AKUMULACJA.value,
                'formacja': 'spring'
            }
        )

    def _generuj_sygnal_short(self, df: pd.DataFrame, indeks: int) -> SygnalTransakcyjny:
        """Generuje sygnał sprzedaży."""
        cena = df['close'].iloc[indeks]
        atr = self.analiza.oblicz_atr(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )[-1]
        
        return SygnalTransakcyjny(
            timestamp=df.index[indeks],
            symbol=df['symbol'].iloc[indeks],
            kierunek=KierunekTransakcji.SHORT,
            cena_wejscia=cena,
            stop_loss=cena + (atr * self.parametry['sl_atr']),
            take_profit=cena - (atr * self.parametry['tp_atr']),
            wolumen=1.0,
            opis="Wyckoff: Upthrust w fazie dystrybucji",
            metadane={
                'faza': FazaWyckoff.DYSTRYBUCJA.value,
                'formacja': 'upthrust'
            }
        ) 