"""
Strategia oparta na analizie technicznej w systemie NikkeiNinja.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from handel.analiza_techniczna import AnalizaTechniczna
from strategie.interfejs import IStrategia, KierunekTransakcji, SygnalTransakcyjny

logger = logging.getLogger(__name__)


class StrategiaTechniczna(IStrategia):
    """
    Strategia wykorzystująca wskaźniki analizy technicznej:
    - RSI do określania wykupienia/wyprzedania
    - MACD do potwierdzenia trendu
    - VWAP do określania poziomu ceny względem średniej ważonej wolumenem
    """
    
    def __init__(self):
        """Inicjalizacja strategii."""
        self.analiza = AnalizaTechniczna()
        self.parametry = {
            'rsi_okres': 14,
            'rsi_wyprzedanie': 30,
            'rsi_wykupienie': 70,
            'macd_wolny': 26,
            'macd_szybki': 12,
            'macd_sygnalowy': 9,
            'atr_okres': 14,
            'atr_mnoznik': 2.0,
            'stoch_k': 14,  # Okres dla %K
            'stoch_d': 3,   # Okres dla %D
            'stoch_smooth': 3,  # Wygładzanie
            'stoch_wyprzedanie': 20,  # Poziom wyprzedania
            'stoch_wykupienie': 80,    # Poziom wykupienia
            'min_spread': 0.01,  # Minimalny spread dla przypadku gdy ATR=0
            'min_zmiana': 0.01  # Minimalna zmiana do analizy trendu
        }
        self.dane_historyczne = pd.DataFrame()
        logger.info("🥷 Zainicjalizowano strategię techniczną")
    
    def inicjalizuj(self, parametry: Dict) -> None:
        """
        Inicjalizuje parametry strategii.
        
        Parametry:
        - rsi_okres: okres dla RSI (domyślnie 14)
        - rsi_wykupienie: poziom wykupienia (domyślnie 70)
        - rsi_wyprzedanie: poziom wyprzedania (domyślnie 30)
        - macd_szybki: okres szybkiej średniej MACD (domyślnie 12)
        - macd_wolny: okres wolnej średniej MACD (domyślnie 26)
        - macd_sygnał: okres średniej sygnałowej (domyślnie 9)
        - sl_atr: mnożnik ATR dla stop loss (domyślnie 2)
        - tp_atr: mnożnik ATR dla take profit (domyślnie 3)
        - stoch_k: okres dla %K Stochastic (domyślnie 14)
        - stoch_d: okres dla %D Stochastic (domyślnie 3)
        - stoch_smooth: okres wygładzania %K (domyślnie 3)
        - stoch_wykupienie: poziom wykupienia (domyślnie 80)
        - stoch_wyprzedanie: poziom wyprzedania (domyślnie 20)
        - min_spread: minimalny spread dla przypadku gdy ATR=0 (domyślnie 0.01)
        - min_zmiana: minimalna zmiana do analizy trendu (domyślnie 0.01)
        """
        self.parametry.update(parametry)
        logger.info("🥷 Zaktualizowano parametry strategii: %s", self.parametry)
    
    def analizuj(self, df: pd.DataFrame, metadane: Dict = None) -> List[Dict]:
        """Analizuje dane i generuje sygnały transakcyjne."""
        try:
            if len(df) < 14:  # Zmniejszamy minimalną liczbę świec do okresu RSI
                logger.warning("⚠️ Za mało danych do analizy (minimum 14 świec)")
                return []

            if 'symbol' not in df.columns:
                logger.warning("⚠️ Brak kolumny 'symbol' w danych")
                return []

            logger.info(f"🥷 Analizuję {len(df)} świec dla {df['symbol'].iloc[0]}")

            # Obliczamy wskaźniki techniczne
            rsi = self.analiza.oblicz_rsi(df['close'].values, self.parametry['rsi_okres'])
            macd, signal, hist = self.analiza.oblicz_macd(
                df['close'].values,
                self.parametry['macd_szybki'],
                self.parametry['macd_wolny'],
                self.parametry['macd_sygnalowy']
            )
            vwap = self.analiza.oblicz_vwap(df)
            stoch_k, stoch_d = self.analiza.oblicz_stochastic(
                df,
                self.parametry['stoch_k'],
                self.parametry['stoch_d'],
                self.parametry['stoch_smooth']
            )
            
            # Obliczamy linię Livermore'a
            livermore = self.analiza.oblicz_linie_livermore(df, self.parametry['min_zmiana'])
            
            sygnaly = []
            
            for i in range(len(df)-1, max(len(df)-6, -1), -1):  # Sprawdzamy ostatnie 5 świec
                cena = df['close'].iloc[i]
                
                # Warunki dla pozycji LONG
                if ((rsi[i] < self.parametry['rsi_wyprzedanie'] or  # Wyprzedanie wg RSI
                     stoch_k[i] < self.parametry['stoch_wyprzedanie']) and  # Wyprzedanie wg Stochastic
                    hist[i] > 0 and  # MACD powyżej linii sygnalowej
                    stoch_k[i] > stoch_d[i] and  # %K przecina %D od dołu
                    (livermore.get('trend_direction', 0) >= 0 or livermore.get('trend_strength', 1.0) < 0.3)):  # Trend wzrostowy lub słaby trend
                    
                    # Obliczamy poziomy SL i TP
                    atr = self.analiza.oblicz_atr(df)[-1]
                    if atr == 0:
                        atr = cena * self.parametry['min_spread']
                    
                    sl = cena - (atr * self.parametry['atr_mnoznik'])
                    tp = cena + (atr * self.parametry['atr_mnoznik'] * 1.5)
                    
                    logger.info(f"🥷 Generuję sygnał LONG: RSI={rsi[i]:.1f}, MACD={hist[i]:.2f}, "
                              f"Stoch %K={stoch_k[i]:.1f}, %D={stoch_d[i]:.1f}")
                    
                    sygnaly.append({
                        'timestamp': df.index[i],
                        'symbol': df['symbol'].iloc[i],
                        'typ': 'LONG',
                        'cena': cena,
                        'sl': sl,
                        'tp': tp,
                        'opis': f"🥷 LONG: RSI={rsi[i]:.1f}, MACD hist={hist[i]:.2f}, "
                               f"Stoch %K={stoch_k[i]:.1f} > %D={stoch_d[i]:.1f}, "
                               f"Livermore: trend={livermore.get('trend_direction', 0)}, siła={livermore.get('trend_strength', 0.0):.2f}"
                    })
                
                # Warunki dla pozycji SHORT
                elif ((rsi[i] > self.parametry['rsi_wykupienie'] or  # Wykupienie wg RSI
                       stoch_k[i] > self.parametry['stoch_wykupienie']) and  # Wykupienie wg Stochastic
                      hist[i] < 0 and  # MACD poniżej linii sygnalowej
                      stoch_k[i] < stoch_d[i] and  # %K przecina %D od góry
                      (livermore.get('trend_direction', 0) <= 0 or livermore.get('trend_strength', 1.0) < 0.3)):  # Trend spadkowy lub słaby trend
                    
                    # Obliczamy poziomy SL i TP
                    atr = self.analiza.oblicz_atr(df)[-1]
                    if atr == 0:
                        atr = cena * self.parametry['min_spread']
                    
                    sl = cena + (atr * self.parametry['atr_mnoznik'])
                    tp = cena - (atr * self.parametry['atr_mnoznik'] * 1.5)
                    
                    logger.info(f"🥷 Generuję sygnał SHORT: RSI={rsi[i]:.1f}, MACD={hist[i]:.2f}, "
                              f"Stoch %K={stoch_k[i]:.1f}, %D={stoch_d[i]:.1f}")
                    
                    sygnaly.append({
                        'timestamp': df.index[i],
                        'symbol': df['symbol'].iloc[i],
                        'typ': 'SHORT',
                        'cena': cena,
                        'sl': sl,
                        'tp': tp,
                        'opis': f"🥷 SHORT: RSI={rsi[i]:.1f}, MACD hist={hist[i]:.2f}, "
                               f"Stoch %K={stoch_k[i]:.1f} < %D={stoch_d[i]:.1f}, "
                               f"Livermore: trend={livermore.get('trend_direction', 0)}, siła={livermore.get('trend_strength', 0.0):.2f}"
                    })
            
            return sygnaly

        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy: {str(e)}")
            return []
    
    def aktualizuj(self, df: pd.DataFrame, symbol: str, kierunek: KierunekTransakcji, cena_wejscia: float) -> List[Dict]:
        """Aktualizuje stan pozycji i generuje sygnały zamknięcia."""
        try:
            if len(df) < 14:  # Minimalny okres dla analizy
                return []
                
            # Obliczanie wskaźników
            rsi = self.analiza.oblicz_rsi(df['close'].values)
            _, _, hist = self.analiza.oblicz_macd(df['close'].values)
            vwap = self.analiza.oblicz_vwap(df)
            stoch_k, stoch_d = self.analiza.oblicz_stochastic(df)
            livermore = self.analiza.oblicz_linie_livermore(df)
            
            sygnaly = []
            i = -1  # Ostatnia świeca
            
            # Warunki zamknięcia dla pozycji LONG
            if kierunek == KierunekTransakcji.LONG:
                if (rsi[i] > self.parametry['rsi_wykupienie'] or  # RSI wykupiony
                    hist[i] < hist[i-1] or  # MACD spada
                    stoch_k[i] > self.parametry['stoch_wykupienie'] or  # Stochastic wykupiony
                    (livermore.get('trend_direction', 0) < 0 and livermore.get('trend_strength', 0.0) > 0.6)):  # Silny trend spadkowy
                    
                    zysk_procent = (df['close'].iloc[i] - cena_wejscia) / cena_wejscia * 100
                    
                    sygnaly.append({
                        'timestamp': df.index[i],
                        'symbol': symbol,
                        'typ': 'SHORT',  # Sygnał zamknięcia LONG
                        'cena': float(df['close'].iloc[i]),
                        'sl': None,
                        'tp': None,
                        'opis': f"Zamknięcie LONG: RSI({rsi[i]:.1f}), MACD spadający, "
                               f"Stoch({stoch_k[i]:.1f}), Livermore: kierunek={livermore.get('trend_direction', 0)}, siła={livermore.get('trend_strength', 0.0):.2f}",
                        'metadane': {
                            'zysk_procent': float(zysk_procent)
                        }
                    })
            
            # Warunki zamknięcia dla pozycji SHORT
            elif kierunek == KierunekTransakcji.SHORT:
                if (rsi[i] < self.parametry['rsi_wyprzedanie'] or  # RSI wyprzedany
                    hist[i] > hist[i-1] or  # MACD rośnie
                    stoch_k[i] < self.parametry['stoch_wyprzedanie'] or  # Stochastic wyprzedany
                    (livermore.get('trend_direction', 0) > 0 and livermore.get('trend_strength', 0.0) > 0.6)):  # Silny trend wzrostowy
                    
                    zysk_procent = (cena_wejscia - df['close'].iloc[i]) / cena_wejscia * 100
                    
                    sygnaly.append({
                        'timestamp': df.index[i],
                        'symbol': symbol,
                        'typ': 'LONG',  # Sygnał zamknięcia SHORT
                        'cena': float(df['close'].iloc[i]),
                        'sl': None,
                        'tp': None,
                        'opis': f"Zamknięcie SHORT: RSI({rsi[i]:.1f}), MACD rosnący, "
                               f"Stoch({stoch_k[i]:.1f}), Livermore: kierunek={livermore.get('trend_direction', 0)}, siła={livermore.get('trend_strength', 0.0):.2f}",
                        'metadane': {
                            'zysk_procent': float(zysk_procent)
                        }
                    })
            
            return sygnaly
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas aktualizacji: {str(e)}")
            return []
    
    def optymalizuj(self, 
                    dane_historyczne: pd.DataFrame,
                    parametry_zakres: Dict[str, Tuple[float, float, float]]) -> Dict:
        """
        Optymalizuje parametry strategii na danych historycznych używając grid search.
        
        Args:
            dane_historyczne: DataFrame z danymi OHLCV
            parametry_zakres: Słownik z zakresami parametrów w formacie:
                            {nazwa_parametru: (start, stop, step)}
        
        Returns:
            Dict z optymalnymi parametrami maksymalizującymi profit factor
        """
        if dane_historyczne.empty:
            return self.parametry.copy()
            
        # Zachowujemy oryginalne parametry
        parametry_oryginalne = self.parametry.copy()
        najlepsze_parametry = parametry_oryginalne.copy()
        najlepszy_profit_factor = 0.0
        
        try:
            # Generowanie wszystkich kombinacji parametrów
            zakresy = {}
            for param, (start, stop, step) in parametry_zakres.items():
                if step <= 0 or start > stop:
                    logger.warning("⚠️ Błędny zakres dla %s: start=%.2f, stop=%.2f, step=%.2f", 
                                 param, start, stop, step)
                    continue
                zakresy[param] = np.arange(start, stop + step, step)
            
            if not zakresy:
                logger.warning("⚠️ Brak poprawnych zakresów parametrów")
                return parametry_oryginalne
            
            import itertools
            kombinacje = [dict(zip(zakresy.keys(), wartosci)) 
                         for wartosci in itertools.product(*zakresy.values())]
            
            logger.info("🥷 Rozpoczynam optymalizację na %d kombinacjach", len(kombinacje))
            
            # Testowanie każdej kombinacji
            for parametry in kombinacje:
                # Ustawienie parametrów
                self.parametry.update(parametry)
                
                # Generowanie sygnałów na danych historycznych
                sygnaly = self.analizuj(dane_historyczne, {'tryb': 'optymalizacja'})
                
                # Obliczanie statystyk
                if sygnaly:
                    statystyki = self.generuj_statystyki(sygnaly)
                    if statystyki and 'profit_factor' in statystyki:
                        profit_factor = statystyki['profit_factor']
                        if profit_factor > najlepszy_profit_factor:
                            najlepszy_profit_factor = profit_factor
                            najlepsze_parametry = parametry.copy()
                            logger.info("🥷 Znaleziono lepsze parametry: PF=%.2f, %s", 
                                      profit_factor, parametry)
            
            logger.info("🥷 Zakończono optymalizację. Najlepszy PF=%.2f", najlepszy_profit_factor)
            
        except Exception as e:
            logger.error("❌ Błąd podczas optymalizacji: %s", str(e))
            self.parametry = parametry_oryginalne
            return parametry_oryginalne
        
        # Przywracamy oryginalne parametry
        self.parametry = parametry_oryginalne
        return najlepsze_parametry
    
    def generuj_statystyki(self, 
                          historia_transakcji: List[Dict]) -> Dict:
        """
        Generuje statystyki skuteczności strategii.
        
        Returns:
            Dict zawierający statystyki:
            - liczba_transakcji: całkowita liczba transakcji
            - liczba_long: liczba pozycji długich
            - liczba_short: liczba pozycji krótkich
            - win_rate: procent zyskownych transakcji
            - win_rate_long: procent zyskownych pozycji długich
            - win_rate_short: procent zyskownych pozycji krótkich
            - zysk_sredni: średni zysk procentowy
            - zysk_mediana: mediana zysków
            - zysk_std: odchylenie standardowe zysków
            - profit_factor: stosunek zysków do strat
        """
        if not historia_transakcji:
            return {}
        
        # Podział na transakcje LONG i SHORT
        long_transakcje = [t for t in historia_transakcji if t['typ'] == 'LONG']
        short_transakcje = [t for t in historia_transakcji if t['typ'] == 'SHORT']
        
        # Zbieranie zysków
        zyski = []
        zyski_long = []
        zyski_short = []
        
        for sygnal in historia_transakcji:
            if 'zysk_procent' in sygnal:
                zysk = sygnal['zysk_procent']
                zyski.append(zysk)
                if sygnal['typ'] == 'LONG':
                    zyski_long.append(zysk)
                else:
                    zyski_short.append(zysk)
        
        if not zyski:
            return {}
        
        # Konwersja na numpy arrays dla obliczeń
        zyski = np.array(zyski)
        zyski_long = np.array(zyski_long) if zyski_long else np.array([])
        zyski_short = np.array(zyski_short) if zyski_short else np.array([])
        
        # Obliczanie statystyk
        statystyki = {
            'liczba_transakcji': len(zyski),
            'liczba_long': len(zyski_long),
            'liczba_short': len(zyski_short),
            'zysk_sredni': float(np.mean(zyski)),
            'zysk_mediana': float(np.median(zyski)),
            'zysk_std': float(np.std(zyski)),
            'win_rate': float(np.sum(zyski > 0) / len(zyski))
        }
        
        # Win rate dla LONG i SHORT
        if len(zyski_long) > 0:
            statystyki['win_rate_long'] = float(np.sum(zyski_long > 0) / len(zyski_long))
        if len(zyski_short) > 0:
            statystyki['win_rate_short'] = float(np.sum(zyski_short > 0) / len(zyski_short))
            
        # Profit factor
        zyski_dodatnie = np.sum(zyski[zyski > 0])
        straty = abs(np.sum(zyski[zyski < 0]))
        statystyki['profit_factor'] = float(zyski_dodatnie / straty) if straty != 0 else float('inf')
        
        return statystyki 