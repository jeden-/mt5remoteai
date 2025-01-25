"""
Strategia hybrydowa łącząca analizę techniczną, teorię Wyckoffa i sentyment rynku.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import and_

from .interfejs import IStrategia, KierunekTransakcji, SygnalTransakcyjny
from .wyckoff import StrategiaWyckoff, FazaWyckoff
from .techniczna import StrategiaTechniczna
from ai.rag_rynkowy import RAGRynkowy
from ai.system_rag import SystemRAG
from baza_danych.modele import KalendarzEkonomiczny

logger = logging.getLogger(__name__)


class StrategiaHybrydowa(IStrategia):
    """
    Strategia hybrydowa łącząca:
    - Analizę techniczną (RSI, MACD, VWAP)
    - Teorię Wyckoffa (fazy rynku, formacje)
    - Sentyment rynku (analiza wzmianek)
    - Podobne wzorce historyczne (RAG)
    """
    
    def __init__(
        self,
        baza,
        techniczna: StrategiaTechniczna = None,
        wyckoff: StrategiaWyckoff = None,
        rag: RAGRynkowy = None
    ):
        """
        Inicjalizacja strategii hybrydowej.
        
        Args:
            baza: Instancja bazy danych
            techniczna: Instancja strategii technicznej (opcjonalnie)
            wyckoff: Instancja strategii Wyckoffa (opcjonalnie)
            rag: Instancja systemu RAG (opcjonalnie)
        """
        self.baza = baza
        self.techniczna = techniczna or StrategiaTechniczna()
        self.wyckoff = wyckoff or StrategiaWyckoff()
        self.rag = rag or RAGRynkowy(SystemRAG())
        
        # Domyślne parametry
        self.parametry = {
            'waga_techniczna': 0.4,
            'waga_wyckoff': 0.3,
            'waga_sentyment': 0.2,
            'waga_wydarzenia': 0.1,
            'min_pewnosc_sentymentu': 0.7,
            'min_podobienstwo_wzorca': 0.8,
            'min_sila_sygnalu': 0.3,  # Minimalny poziom łącznego sygnału
            'sl_atr': 2.0,  # Stop loss jako wielokrotność ATR
            'tp_atr': 3.0   # Take profit jako wielokrotność ATR
        }
        
        logger.info("🥷 Zainicjalizowano strategię hybrydową")
    
    def inicjalizuj(self, parametry: Dict) -> None:
        """
        Inicjalizuje parametry strategii.
        
        Parametry:
        - waga_techniczna: waga dla sygnałów analizy technicznej (0-1)
        - waga_wyckoff: waga dla sygnałów Wyckoffa (0-1)
        - waga_sentyment: waga dla sygnałów sentymentu (0-1)
        - min_pewnosc_sentymentu: minimalna pewność sentymentu (0-1)
        - min_podobienstwo_wzorca: minimalne podobieństwo wzorca (0-1)
        - sl_atr: mnożnik ATR dla stop loss (domyślnie 2)
        - tp_atr: mnożnik ATR dla take profit (domyślnie 3)
        """
        self.parametry = {
            'waga_techniczna': 0.4,
            'waga_wyckoff': 0.4,
            'waga_sentyment': 0.2,
            'min_pewnosc_sentymentu': 0.6,
            'min_podobienstwo_wzorca': 0.7,
            'sl_atr': 2.0,
            'tp_atr': 3.0,
            **parametry
        }
        
        # Inicjalizacja strategii składowych
        self.techniczna.inicjalizuj(parametry)
        self.wyckoff.inicjalizuj(parametry)
        logger.info("🥷 Zaktualizowano parametry strategii: %s", self.parametry)
    
    async def _analizuj_sentyment(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analizuje sentyment rynku i szuka podobnych wzorców.
        
        Args:
            df: DataFrame z danymi OHLCV
            
        Returns:
            Dict z wynikami analizy sentymentu i podobieństwa wzorców
        """
        try:
            # Analizuj sentyment dla ostatnich świec
            sentyment = await self.rag.agreguj_sentyment(df)
            
            # Znajdź podobne wzorce
            podobne = await self.rag.znajdz_podobne_wzorce(
                df,
                limit=5,
                filtry={
                    'sentyment_pewnosc': {'$gte': self.parametry['min_pewnosc_sentymentu']}
                }
            )
            
            # Oblicz średnią zmianę dla podobnych wzorców
            if podobne:
                srednia_zmiana = np.mean([
                    w['metadata']['zmiana_proc'] 
                    for w in podobne
                ])
            else:
                srednia_zmiana = 0.0
            
            return {
                'sentyment': sentyment['sredni_sentyment'],
                'pewnosc': sentyment['zmiana_sentymentu'],
                'srednia_zmiana': srednia_zmiana,
                'liczba_wzmianek': sentyment['liczba_wzmianek']
            }
            
        except Exception as e:
            logger.error("❌ Błąd podczas analizy sentymentu: %s", str(e))
            return {
                'sentyment': 0.0,
                'pewnosc': 0.0,
                'srednia_zmiana': 0.0,
                'liczba_wzmianek': 0
            }
    
    async def analizuj(self, 
                      df: pd.DataFrame,
                      symbol: str,
                      dodatkowe_dane: Optional[Dict] = None) -> List[SygnalTransakcyjny]:
        """
        Analizuje dane i generuje sygnały transakcyjne.
        
        Args:
            df: DataFrame z danymi OHLCV
            symbol: Symbol instrumentu
            dodatkowe_dane: Opcjonalne dodatkowe dane
        
        Returns:
            Lista sygnałów transakcyjnych
        """
        try:
            sygnaly = []
            
            # Pobierz sygnały ze strategii składowych
            sygnaly_tech = self.techniczna.analizuj(df, dodatkowe_dane)
            sygnaly_wyckoff = self.wyckoff.analizuj(df, dodatkowe_dane)
            
            # Analizuj sentyment
            sentyment = await self._analizuj_sentyment(df)
            
            # Jeśli brak sygnałów, zwróć pustą listę
            if not (sygnaly_tech or sygnaly_wyckoff):
                return []
            
            # Pobierz analizę wydarzeń z kalendarza
            wydarzenia = await self._analizuj_wydarzenia_kalendarz(df.index[-1].strftime('%Y-%m-%d'), df)
            
            # Oblicz łączny sygnał
            sygnal_tech = np.mean([1 if s.kierunek == KierunekTransakcji.LONG else -1 for s in sygnaly_tech])
            sygnal_wyckoff = np.mean([1 if s.kierunek == KierunekTransakcji.LONG else -1 for s in sygnaly_wyckoff]) if sygnaly_wyckoff else 0
            sygnal_sentyment = sentyment['sentyment'] * sentyment['pewnosc']
            sygnal_wydarzenia = wydarzenia['wplyw'] * min(1, wydarzenia['waznosc'] / 2)
            
            # Normalizuj wagi
            suma_wag = (self.parametry['waga_techniczna'] + self.parametry['waga_wyckoff'] + 
                       self.parametry['waga_sentyment'] + self.parametry['waga_wydarzenia'])
            
            waga_tech = self.parametry['waga_techniczna'] / suma_wag
            waga_wyckoff = self.parametry['waga_wyckoff'] / suma_wag
            waga_sentyment = self.parametry['waga_sentyment'] / suma_wag
            waga_wydarzenia = self.parametry['waga_wydarzenia'] / suma_wag
            
            # Oblicz łączny sygnał ważony
            sygnal = (
                sygnal_tech * waga_tech +
                sygnal_wyckoff * waga_wyckoff +
                sygnal_sentyment * waga_sentyment +
                sygnal_wydarzenia * waga_wydarzenia
            )
            
            # Sprawdź czy sygnał jest wystarczająco silny
            if abs(sygnal) < self.parametry['min_sila_sygnalu']:
                return []
                
            # Przygotuj opis sygnału
            opis = f"Sygnał łączony (siła: {sygnal:.2f}):\n"
            opis += f"• Analiza techniczna ({sygnal_tech:.2f})\n"
            if sygnaly_wyckoff:
                opis += f"• Analiza Wyckoffa ({sygnal_wyckoff:.2f})\n"
            if sentyment:
                opis += f"• Sentyment rynku: {sentyment['opis']}\n"
            if wydarzenia and wydarzenia['liczba'] > 0:
                opis += f"• Wydarzenia ekonomiczne:\n{wydarzenia['opis']}\n"
            
            # Wygeneruj sygnał transakcyjny
            kierunek = KierunekTransakcji.LONG if sygnal > 0 else KierunekTransakcji.SHORT
            cena = df['close'].iloc[-1]
            
            # Oblicz poziomy SL i TP na podstawie ATR
            atr = self.techniczna.oblicz_atr(df)[-1]
            stop_loss = cena - (atr * self.parametry['sl_atr']) if kierunek == KierunekTransakcji.LONG else cena + (atr * self.parametry['sl_atr'])
            take_profit = cena + (atr * self.parametry['tp_atr']) if kierunek == KierunekTransakcji.LONG else cena - (atr * self.parametry['tp_atr'])
            
            return [SygnalTransakcyjny(
                timestamp=df.index[-1],
                kierunek=kierunek,
                symbol=symbol,
                cena_wejscia=cena,
                stop_loss=stop_loss,
                take_profit=take_profit,
                wolumen=1.0,
                opis=opis,
                metadane={
                    'symbol': symbol,
                    'sentyment': float(sentyment['sentyment']),
                    'sentyment_pewnosc': float(sentyment['pewnosc']),
                    'sentyment_wzmianki': int(sentyment['liczba_wzmianek']),
                    'wydarzenia': wydarzenia
                }
            )]
            
        except Exception as e:
            logger.error("❌ Błąd podczas analizy: %s", str(e))
            return []
    
    async def aktualizuj(self,
                        df: pd.DataFrame,
                        symbol: str,
                        kierunek: Optional[KierunekTransakcji] = None,
                        cena_wejscia: Optional[float] = None) -> List[SygnalTransakcyjny]:
        """
        Aktualizuje stan strategii i sprawdza aktywne pozycje.
        
        Args:
            df: DataFrame z danymi OHLCV
            symbol: Symbol instrumentu
            kierunek: Kierunek aktywnej pozycji
            cena_wejscia: Cena wejścia aktywnej pozycji
            
        Returns:
            Lista sygnałów transakcyjnych
        """
        try:
            sygnaly = []
            
            # Pobierz sygnały ze strategii składowych
            sygnaly_tech = self.techniczna.aktualizuj(df, [(symbol, kierunek, cena_wejscia)])
            sygnaly_wyckoff = self.wyckoff.aktualizuj(df, symbol, kierunek, cena_wejscia)
            
            # Analizuj sentyment
            sentyment = await self._analizuj_sentyment(df)
            
            # Jeśli brak aktywnej pozycji, zwróć pustą listę
            if not kierunek or not cena_wejscia:
                return []
            
            # Sprawdź warunki zamknięcia
            cena_aktualna = df['close'].iloc[-1]
            zysk_procent = ((cena_aktualna - cena_wejscia) / cena_wejscia * 100 
                           if kierunek == KierunekTransakcji.LONG
                           else (cena_wejscia - cena_aktualna) / cena_wejscia * 100)
            
            # Zamknij pozycję jeśli:
            # 1. Sygnały techniczne lub Wyckoffa sugerują zamknięcie
            # 2. Sentyment jest przeciwny do pozycji
            if ((sygnaly_tech or sygnaly_wyckoff) or
                (kierunek == KierunekTransakcji.LONG and sentyment['sentyment'] < -0.3) or
                (kierunek == KierunekTransakcji.SHORT and sentyment['sentyment'] > 0.3)):
                
                sygnaly.append(SygnalTransakcyjny(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    kierunek=KierunekTransakcji.SHORT if kierunek == KierunekTransakcji.LONG else KierunekTransakcji.LONG,
                    cena_wejscia=cena_aktualna,
                    stop_loss=None,
                    take_profit=None,
                    wolumen=1.0,
                    opis=f"Zamknięcie {kierunek.value} - zmiana sentymentu/sygnały przeciwne",
                    metadane={
                        'symbol': symbol,
                        'zysk_procent': float(zysk_procent),
                        'sentyment': float(sentyment['sentyment']),
                        'sentyment_pewnosc': float(sentyment['pewnosc']),
                        'sentyment_wzmianki': int(sentyment['liczba_wzmianek'])
                    }
                ))
            
            return sygnaly
            
        except Exception as e:
            logger.error("❌ Błąd podczas aktualizacji: %s", str(e))
            return []
    
    def optymalizuj(self,
                    dane_historyczne: pd.DataFrame,
                    parametry_zakres: Optional[Dict] = None) -> Dict:
        """
        Optymalizuje parametry strategii na danych historycznych.
        
        Args:
            dane_historyczne: DataFrame z danymi do optymalizacji
            parametry_zakres: Słownik z zakresami parametrów (opcjonalnie)
            
        Returns:
            Dict z optymalnymi parametrami
        """
        try:
            logger.info("🥷 Rozpoczynam optymalizację strategii hybrydowej")
            
            # Zachowaj obecne parametry
            obecne_parametry = self.parametry.copy()
            najlepsze_parametry = None
            najlepszy_wynik = float('-inf')
            
            # Przygotuj zakresy dla wag
            wagi_zakres = {
                'waga_techniczna': np.linspace(0.2, 0.6, 5),
                'waga_wyckoff': np.linspace(0.2, 0.6, 5),
                'waga_sentyment': np.linspace(0.1, 0.4, 4)
            }
            
            # Przygotuj zakresy dla progów
            progi_zakres = {
                'min_pewnosc_sentymentu': np.linspace(0.4, 0.8, 5),
                'min_podobienstwo_wzorca': np.linspace(0.5, 0.9, 5)
            }
            
            # Użyj domyślnych zakresów jeśli nie podano innych
            if parametry_zakres is None:
                parametry_zakres = {}
            
            # Optymalizuj parametry składowych strategii
            parametry_tech = self.techniczna.optymalizuj(dane_historyczne, parametry_zakres)
            parametry_wyckoff = self.wyckoff.optymalizuj(dane_historyczne, parametry_zakres)
            
            # Połącz wszystkie zakresy
            wszystkie_zakresy = {
                **wagi_zakres,
                **progi_zakres,
                **parametry_zakres
            }
            
            # Przygotuj wartości do testowania
            parametry_wartosci = []
            parametry_nazwy = []
            for nazwa, wartosci in wszystkie_zakresy.items():
                parametry_wartosci.append(wartosci)
                parametry_nazwy.append(nazwa)
            
            # Grid search po kombinacjach parametrów
            from itertools import product
            symulator = SymulatorRynku(kapital_poczatkowy=100000.0)
            
            for wartosci in product(*parametry_wartosci):
                # Normalizacja wag
                wt, ww, ws = wartosci[:3]  # wagi techniczna, wyckoff, sentyment
                suma = wt + ww + ws
                wt, ww, ws = wt/suma, ww/suma, ws/suma
                
                # Ustawienie parametrów
                parametry_testowe = {
                    **parametry_tech,
                    **parametry_wyckoff,
                    'waga_techniczna': float(wt),
                    'waga_wyckoff': float(ww),
                    'waga_sentyment': float(ws),
                    'min_pewnosc_sentymentu': float(wartosci[3]),
                    'min_podobienstwo_wzorca': float(wartosci[4]),
                    'sl_atr': obecne_parametry['sl_atr'],
                    'tp_atr': obecne_parametry['tp_atr']
                }
                
                self.inicjalizuj(parametry_testowe)
                
                # Backtest
                wynik = symulator.testuj_strategie(self, dane_historyczne)
                
                # Ocena wyniku (złożona metryka)
                score = (
                    wynik.zysk_procent * 
                    wynik.win_rate * 
                    wynik.profit_factor * 
                    (1 - wynik.max_drawdown/100) * 
                    max(0, wynik.sharpe_ratio)
                )
                
                # Aktualizacja najlepszego wyniku
                if score > najlepszy_wynik:
                    najlepszy_wynik = score
                    najlepsze_parametry = parametry_testowe.copy()
                    
                    logger.info("🥷 Znaleziono lepsze parametry (score: %.2f):", score)
                    for nazwa, wartosc in parametry_testowe.items():
                        logger.info("  %s: %.3f", nazwa, wartosc)
            
            # Przywróć poprzednie parametry
            self.inicjalizuj(obecne_parametry)
            
            return najlepsze_parametry if najlepsze_parametry else self.parametry
            
        except Exception as e:
            logger.error("❌ Błąd podczas optymalizacji: %s", str(e))
            return self.parametry
    
    def generuj_statystyki(self, historia: List[SygnalTransakcyjny]) -> Dict:
        """
        Generuje statystyki dla strategii na podstawie historii transakcji.
        
        Args:
            historia: Lista sygnałów transakcyjnych
            
        Returns:
            Dict ze statystykami
        """
        try:
            if not historia:
                return {}
                
            # Podstawowe statystyki
            liczba_transakcji = len(historia)
            zyskowne = sum(1 for s in historia if s.metadane.get('zysk_procent', 0) > 0)
            win_rate = zyskowne / liczba_transakcji if liczba_transakcji > 0 else 0
            
            # Statystyki per kierunek
            long_transakcje = [s for s in historia if s.kierunek == KierunekTransakcji.LONG]
            short_transakcje = [s for s in historia if s.kierunek == KierunekTransakcji.SHORT]
            
            # Statystyki sentymentu
            sredni_sentyment = np.mean([
                s.metadane.get('sentyment', 0) 
                for s in historia
            ])
            
            srednia_pewnosc = np.mean([
                s.metadane.get('sentyment_pewnosc', 0) 
                for s in historia
            ])
            
            return {
                'liczba_transakcji': liczba_transakcji,
                'win_rate': win_rate,
                'liczba_long': len(long_transakcje),
                'liczba_short': len(short_transakcje),
                'win_rate_long': (sum(1 for s in long_transakcje if s.metadane.get('zysk_procent', 0) > 0) 
                                 / len(long_transakcje) if long_transakcje else 0),
                'win_rate_short': (sum(1 for s in short_transakcje if s.metadane.get('zysk_procent', 0) > 0) 
                                  / len(short_transakcje) if short_transakcje else 0),
                'sredni_sentyment': float(sredni_sentyment),
                'srednia_pewnosc_sentymentu': float(srednia_pewnosc),
                'sredni_zysk': float(np.mean([s.metadane.get('zysk_procent', 0) for s in historia])),
                'max_zysk': float(max([s.metadane.get('zysk_procent', 0) for s in historia])),
                'max_strata': float(min([s.metadane.get('zysk_procent', 0) for s in historia]))
            }
            
        except Exception as e:
            logger.error("❌ Błąd podczas generowania statystyk: %s", str(e))
            return {}
    
    async def _analizuj_wydarzenia_kalendarz(
            self,
            symbol: str,
            dane: pd.DataFrame,
            okno_dni: int = 7
        ) -> Dict[str, Any]:
            """
            Analizuje wpływ wydarzeń z kalendarza ekonomicznego na rynek.
            
            Args:
                symbol: Symbol do analizy
                dane: DataFrame z danymi OHLCV
                okno_dni: Liczba dni do analizy wstecz
                
            Returns:
                Dict z wynikami analizy:
                - wplyw: Łączny wpływ wydarzeń (-1 do 1)
                - waznosc: Średnia ważność wydarzeń
                - liczba: Liczba znalezionych wydarzeń
                - opis: Tekstowy opis najważniejszych wydarzeń
            """
            try:
                # Pobierz walutę dla symbolu
                waluta = self._pobierz_walute_dla_symbolu(symbol)
                if not waluta:
                    return {
                        'wplyw': 0,
                        'waznosc': 0,
                        'liczba': 0,
                        'opis': "Brak danych o walucie"
                    }
                
                # Pobierz wydarzenia z ostatnich X dni
                od = dane.index[-1] - pd.Timedelta(days=okno_dni)
                wydarzenia = self.baza.session.query(KalendarzEkonomiczny)\
                    .filter(
                        and_(
                            KalendarzEkonomiczny.timestamp >= od,
                            KalendarzEkonomiczny.timestamp <= dane.index[-1],
                            KalendarzEkonomiczny.waluta == waluta,
                            KalendarzEkonomiczny.waznosc >= 2  # Tylko ważne wydarzenia
                        )
                    )\
                    .order_by(KalendarzEkonomiczny.waznosc.desc())\
                    .all()
                
                if not wydarzenia:
                    return {
                        'wplyw': 0,
                        'waznosc': 0,
                        'liczba': 0,
                        'opis': "Brak ważnych wydarzeń"
                    }
                
                # Oblicz wpływ każdego wydarzenia
                wplywy = []
                opisy = []
                for w in wydarzenia:
                    if pd.notna(w.wartosc_aktualna) and pd.notna(w.wartosc_prognoza):
                        # Normalizuj różnicę do zakresu -1 do 1
                        roznica = w.wartosc_aktualna - w.wartosc_prognoza
                        if abs(roznica) > 0:
                            wplyw = np.tanh(roznica)  # Normalizacja do -1,1
                            waga = w.waznosc / 3  # Normalizacja ważności
                            wplywy.append(wplyw * waga)
                            
                            # Dodaj opis dla ważnych wydarzeń
                            if w.waznosc >= 2:
                                opis = f"{w.nazwa}: "
                                if wplyw > 0:
                                    opis += f"Lepiej niż oczekiwano (+{roznica:.2f}) 📈"
                                else:
                                    opis += f"Gorzej niż oczekiwano ({roznica:.2f}) 📉"
                                opisy.append(opis)
                
                if not wplywy:
                    return {
                        'wplyw': 0,
                        'waznosc': sum(w.waznosc for w in wydarzenia) / len(wydarzenia),
                        'liczba': len(wydarzenia),
                        'opis': "Brak danych o wpływie wydarzeń"
                    }
                
                return {
                    'wplyw': np.mean(wplywy),  # Średni wpływ ważony ważnością
                    'waznosc': sum(w.waznosc for w in wydarzenia) / len(wydarzenia),
                    'liczba': len(wydarzenia),
                    'opis': "\n".join(opisy[:3])  # Top 3 najważniejsze wydarzenia
                }
                
            except Exception as e:
                logger.error(f"❌ Błąd analizy wydarzeń: {str(e)}")
                return {
                    'wplyw': 0,
                    'waznosc': 0,
                    'liczba': 0,
                    'opis': f"Błąd analizy: {str(e)}"
                }
        
    def _pobierz_walute_dla_symbolu(self, symbol: str) -> Optional[str]:
        """Pobiera kod waluty dla danego symbolu."""
        # Mapowanie symboli na waluty
        MAPA_WALUT = {
            'JP225': 'JPY',
            'NKY': 'JPY',
            'NI225': 'JPY',
            'USDJPY': 'JPY',
            'EURJPY': 'JPY',
            'GBPJPY': 'JPY'
        }
        return MAPA_WALUT.get(symbol)

    async def analizuj_sygnaly(
            self,
            symbol: str,
            dane: pd.DataFrame,
            limit_pozycji: float = 1.0
        ) -> List[SygnalTransakcyjny]:
        """
        Analizuje sygnały transakcyjne na podstawie wszystkich dostępnych źródeł.
        
        Args:
            symbol: Symbol do analizy
            dane: DataFrame z danymi OHLCV
            limit_pozycji: Maksymalny rozmiar pozycji (0-1)
            
        Returns:
            Lista sygnałów transakcyjnych
        """
        try:
            # Pobierz sygnały z analizy technicznej
            sygnaly_tech = await self.techniczna.generuj_sygnaly(dane)
            if not sygnaly_tech:
                return []
                
            # Pobierz sygnały z analizy Wyckoffa
            sygnaly_wyckoff = await self.wyckoff.analizuj_sygnaly(symbol, dane)
            
            # Pobierz analizę sentymentu
            sentyment = await self._analizuj_sentyment(dane)
            
            # Pobierz analizę wydarzeń z kalendarza
            wydarzenia = await self._analizuj_wydarzenia_kalendarz(symbol, dane)
            
            # Znajdź podobne wzorce historyczne
            wzorce = await self.rag.znajdz_podobne_wzorce(
                symbol=symbol,
                dane=dane,
                limit=5,
                filtry={
                    'min_podobienstwo': self.parametry['min_podobienstwo_wzorca']
                }
            )
            
            # Oblicz łączny sygnał
            sygnal_tech = np.mean([1 if s.kierunek == KierunekTransakcji.LONG else -1 for s in sygnaly_tech])
            sygnal_wyckoff = np.mean([1 if s.kierunek == KierunekTransakcji.LONG else -1 for s in sygnaly_wyckoff]) if sygnaly_wyckoff else 0
            sygnal_sentyment = sentyment['sentyment'] * sentyment['pewnosc']
            sygnal_wydarzenia = wydarzenia['wplyw'] * min(1, wydarzenia['waznosc'] / 2)
            
            # Normalizuj wagi
            suma_wag = (self.parametry['waga_techniczna'] + self.parametry['waga_wyckoff'] + 
                       self.parametry['waga_sentyment'] + self.parametry['waga_wydarzenia'])
            
            waga_tech = self.parametry['waga_techniczna'] / suma_wag
            waga_wyckoff = self.parametry['waga_wyckoff'] / suma_wag
            waga_sentyment = self.parametry['waga_sentyment'] / suma_wag
            waga_wydarzenia = self.parametry['waga_wydarzenia'] / suma_wag
            
            # Oblicz łączny sygnał ważony
            sygnal = (
                sygnal_tech * waga_tech +
                sygnal_wyckoff * waga_wyckoff +
                sygnal_sentyment * waga_sentyment +
                sygnal_wydarzenia * waga_wydarzenia
            )
            
            # Sprawdź czy sygnał jest wystarczająco silny
            if abs(sygnal) < self.parametry['min_sila_sygnalu']:
                return []
                
            # Przygotuj opis sygnału
            opis = f"Sygnał łączony (siła: {sygnal:.2f}):\n"
            opis += f"• Analiza techniczna ({sygnal_tech:.2f})\n"
            if sygnaly_wyckoff:
                opis += f"• Analiza Wyckoffa ({sygnal_wyckoff:.2f})\n"
            if sentyment:
                opis += f"• Sentyment rynku: {sentyment['opis']}\n"
            if wydarzenia and wydarzenia['liczba'] > 0:
                opis += f"• Wydarzenia ekonomiczne:\n{wydarzenia['opis']}\n"
            if wzorce:
                opis += f"• Znaleziono {len(wzorce)} podobnych wzorców\n"
            
            # Wygeneruj sygnał transakcyjny
            kierunek = KierunekTransakcji.LONG if sygnal > 0 else KierunekTransakcji.SHORT
            cena = dane.iloc[-1]['close']
            
            # Oblicz poziomy SL i TP na podstawie ATR
            atr = self.techniczna.oblicz_atr(dane)[-1]
            stop_loss = cena - (atr * self.parametry['sl_atr']) if kierunek == KierunekTransakcji.LONG else cena + (atr * self.parametry['sl_atr'])
            take_profit = cena + (atr * self.parametry['tp_atr']) if kierunek == KierunekTransakcji.LONG else cena - (atr * self.parametry['tp_atr'])
            
            return [SygnalTransakcyjny(
                timestamp=dane.index[-1],
                kierunek=kierunek,
                symbol=symbol,
                cena_wejscia=cena,
                stop_loss=stop_loss,
                take_profit=take_profit,
                wolumen=abs(sygnal) * limit_pozycji,
                opis=opis
            )]
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy sygnałów: {str(e)}")
            return [] 