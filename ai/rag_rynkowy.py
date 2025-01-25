"""
Moduł integrujący system RAG z danymi rynkowymi.
Umożliwia indeksowanie i wyszukiwanie wzorców cenowych.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .system_rag import SystemRAG
from handel.analiza_techniczna import AnalizaTechniczna
from strategie.wyckoff import StrategiaWyckoff, FazaWyckoff
from .analiza_sentymentu import AnalizatorSentymentu
from .scraper_social import ScraperSocial, WzmiankaSocial

logger = logging.getLogger(__name__)

class RAGRynkowy:
    """Klasa integrująca system RAG z danymi rynkowymi."""
    
    def __init__(self, system_rag: SystemRAG):
        """
        Inicjalizacja systemu RAG dla danych rynkowych.
        
        Args:
            system_rag: Instancja systemu RAG do wykorzystania
        """
        self.logger = logger
        self.rag = system_rag
        self.analiza = AnalizaTechniczna()
        self.wyckoff = StrategiaWyckoff()
        self.analizator_sentymentu = AnalizatorSentymentu()
        self.scraper = ScraperSocial()
        self._cache = {}  # Cache dla często wyszukiwanych wzorców
        
    def _oblicz_wskazniki(
        self,
        dane: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> Dict[str, float]:
        """
        Oblicza wskaźniki techniczne dla fragmentu danych.
        
        Args:
            dane: DataFrame z danymi OHLCV
            start_idx: Początkowy indeks
            end_idx: Końcowy indeks
            
        Returns:
            Dict[str, float]: Słownik ze wskaźnikami
        """
        fragment = dane.iloc[start_idx:end_idx + 1]
        
        return {
            'rsi': float(self.analiza.oblicz_rsi(fragment['close'].values)[-1]),
            'atr': float(self.analiza.oblicz_atr(
                fragment['high'].values,
                fragment['low'].values,
                fragment['close'].values
            )[-1]),
            'momentum': float(self.analiza.oblicz_momentum(fragment['close'].values)[-1]),
            'volatility': float(self.analiza.oblicz_volatility(fragment['close'].values)[-1]),
            'rel_volume': float(self.analiza.oblicz_relative_volume(fragment['volume'].values)[-1])
        }

    def _wykryj_formacje(
        self,
        dane: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> List[str]:
        """
        Wykrywa formacje świecowe w danych.
        
        Args:
            dane: DataFrame z danymi OHLCV
            start_idx: Początkowy indeks
            end_idx: Końcowy indeks
            
        Returns:
            List[str]: Lista wykrytych formacji
        """
        fragment = dane.iloc[start_idx:end_idx + 1]
        formacje = []
        
        # Oblicz podstawowe parametry świec
        body = fragment['close'] - fragment['open']
        upper_shadow = fragment['high'] - fragment[['open', 'close']].max(axis=1)
        lower_shadow = fragment[['open', 'close']].min(axis=1) - fragment['low']
        
        # Doji
        doji_mask = (abs(body) <= 0.1 * (fragment['high'] - fragment['low']))
        if doji_mask.iloc[-1]:
            formacje.append("Doji")
            
        # Młot/Wisielec
        hammer_mask = (
            (lower_shadow >= 2 * abs(body)) & 
            (upper_shadow <= 0.1 * (fragment['high'] - fragment['low']))
        )
        if hammer_mask.iloc[-1]:
            if body.iloc[-1] > 0:
                formacje.append("Młot")
            else:
                formacje.append("Wisielec")
                
        # Gwiazda poranna/wieczorna
        if len(fragment) >= 3:
            last_3_days = fragment.iloc[-3:]
            if (body.iloc[-3] < 0 and  # Pierwszy dzień czerwony
                abs(body.iloc[-2]) <= 0.1 * (last_3_days['high'].max() - last_3_days['low'].min()) and  # Drugi dzień doji
                body.iloc[-1] > 0):  # Trzeci dzień zielony
                formacje.append("Gwiazda poranna")
            elif (body.iloc[-3] > 0 and  # Pierwszy dzień zielony
                  abs(body.iloc[-2]) <= 0.1 * (last_3_days['high'].max() - last_3_days['low'].min()) and  # Drugi dzień doji
                  body.iloc[-1] < 0):  # Trzeci dzień czerwony
                formacje.append("Gwiazda wieczorna")
                
        # Pochłonięcie hossy/bessy
        if len(fragment) >= 2:
            last_2_days = fragment.iloc[-2:]
            body_1 = body.iloc[-2]  # Pierwszy dzień
            body_2 = body.iloc[-1]  # Drugi dzień
            
            if (body_1 < 0 and  # Pierwszy dzień czerwony
                body_2 > 0 and  # Drugi dzień zielony
                last_2_days['open'].iloc[-1] <= last_2_days['close'].iloc[-2] and  # Otwarcie poniżej zamknięcia
                last_2_days['close'].iloc[-1] >= last_2_days['open'].iloc[-2]):  # Zamknięcie powyżej otwarcia
                formacje.append("Pochłonięcie hossy")
            elif (body_1 > 0 and  # Pierwszy dzień zielony
                  body_2 < 0 and  # Drugi dzień czerwony
                  last_2_days['open'].iloc[-1] >= last_2_days['close'].iloc[-2] and  # Otwarcie powyżej zamknięcia
                  last_2_days['close'].iloc[-1] <= last_2_days['open'].iloc[-2]):  # Zamknięcie poniżej otwarcia
                formacje.append("Pochłonięcie bessy")
        
        return formacje

    def _identyfikuj_faze_wyckoff(
        self,
        dane: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> Tuple[FazaWyckoff, str]:
        """
        Identyfikuje fazę rynku według teorii Wyckoffa.
        
        Args:
            dane: DataFrame z danymi OHLCV
            start_idx: Początkowy indeks
            end_idx: Końcowy indeks
            
        Returns:
            Tuple[FazaWyckoff, str]: Faza i jej opis
        """
        fragment = dane.iloc[start_idx:end_idx + 1]
        faza = self.wyckoff._identyfikuj_faze(fragment)
        
        opis_faz = {
            FazaWyckoff.AKUMULACJA: "Faza akumulacji - budowanie bazy przed wzrostami",
            FazaWyckoff.WZROST: "Faza wzrostu - dominacja kupujących",
            FazaWyckoff.DYSTRYBUCJA: "Faza dystrybucji - wyczerpywanie się wzrostów",
            FazaWyckoff.SPADEK: "Faza spadku - dominacja sprzedających",
            FazaWyckoff.NIEZNANA: "Faza niejednoznaczna"
        }
        
        return faza, opis_faz[faza]
        
    async def _przygotuj_opis_wzorca(
        self,
        dane: pd.DataFrame,
        indeks: int,
        okno: int = 20
    ) -> str:
        """
        Przygotowuje tekstowy opis wzorca cenowego.
        
        Args:
            dane: DataFrame z danymi OHLCV
            indeks: Indeks końcowy wzorca
            okno: Rozmiar okna do analizy
            
        Returns:
            str: Tekstowy opis wzorca
        """
        start_idx = max(0, indeks - okno + 1)
        fragment = dane.iloc[start_idx:indeks + 1]
        
        # Oblicz podstawowe statystyki
        zmiana = ((fragment['close'].iloc[-1] - fragment['close'].iloc[0]) 
                 / fragment['close'].iloc[0] * 100)
        wolumen_sredni = fragment['volume'].mean()
        zmiennosc = (fragment['high'] - fragment['low']).std()
        
        # Wykryj formacje świecowe
        formacje = self._wykryj_formacje(dane, start_idx, indeks)
        
        # Zidentyfikuj fazę Wyckoffa
        faza, opis_fazy = self._identyfikuj_faze_wyckoff(dane, start_idx, indeks)
        
        # Oblicz wskaźniki techniczne
        wskazniki = self._oblicz_wskazniki(dane, start_idx, indeks)
        
        # Analizuj sentyment
        sentyment = await self._analizuj_sentyment_rynku(dane, start_idx, indeks)
        
        # Przygotuj opis
        opis = f"""Wzorzec cenowy z okresu {dane.index[start_idx].strftime('%Y-%m-%d %H:%M')} do {dane.index[indeks].strftime('%Y-%m-%d %H:%M')}

Zmiana ceny: {zmiana:.2f}%
Średni wolumen: {wolumen_sredni:.0f}
Zmienność (STD H-L): {zmiennosc:.2f}

Faza rynku: {opis_fazy}

Wskaźniki techniczne:
- RSI: {wskazniki['rsi']:.2f}
- ATR: {wskazniki['atr']:.2f}
- Momentum: {wskazniki['momentum']:.2f}
- Zmienność: {wskazniki['volatility']:.2f}
- Względny wolumen: {wskazniki['rel_volume']:.2f}

Sentyment rynku:
- Ogólny sentyment: {sentyment['sentyment']}
- Pewność: {sentyment['pewnosc']:.2%}
- Liczba wzmianek: {sentyment['liczba_wzmianek']}
- Rozkład sentymentu:
  * Pozytywne: {sentyment.get('pozytywne_proc', 0):.1f}%
  * Neutralne: {sentyment.get('neutralne_proc', 0):.1f}%
  * Negatywne: {sentyment.get('negatywne_proc', 0):.1f}%

Ceny:
Open: {fragment['open'].iloc[-1]:.2f}
High: {fragment['high'].iloc[-1]:.2f}
Low: {fragment['low'].iloc[-1]:.2f}
Close: {fragment['close'].iloc[-1]:.2f}
Wolumen: {fragment['volume'].iloc[-1]:.0f}"""

        if formacje:
            opis += f"\n\nWykryte formacje świecowe:\n- " + "\n- ".join(formacje)
            
        return opis
        
    async def indeksuj_dane_historyczne(
        self,
        dane: pd.DataFrame,
        okno: int = 20,
        krok: int = 5
    ) -> bool:
        """
        Indeksuje dane historyczne w systemie RAG.
        
        Args:
            dane: DataFrame z danymi rynkowymi (OHLCV)
            okno: Rozmiar okna do analizy wzorców
            krok: Co ile świec zapisywać wzorzec
            
        Returns:
            bool: True jeśli indeksowanie się powiodło
        """
        try:
            if len(dane) < okno:
                self.logger.warning("⚠️ Za mało danych do indeksowania")
                return False
                
            if not all(col in dane.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                self.logger.error("❌ Brak wymaganych kolumn w danych")
                return False
                
            for i in range(okno, len(dane), krok):
                try:
                    # Generuj opis wzorca
                    opis = await self._przygotuj_opis_wzorca(dane, i, okno)
                    
                    # Analizuj sentyment
                    sentyment = await self._analizuj_sentyment_rynku(dane, i-okno, i)
                    
                    # Przygotuj metadane
                    metadata = {
                        'id': f"wzorzec_{dane.index[i].strftime('%Y%m%d_%H%M')}",
                        'data': dane.index[i].isoformat(),
                        'zmiana_proc': ((dane['close'].iloc[i] - dane['close'].iloc[i-okno])
                                      / dane['close'].iloc[i-okno] * 100),
                        'wolumen_sredni': dane['volume'].iloc[i-okno:i+1].mean(),
                        'zmiennosc': (dane['high'].iloc[i-okno:i+1] - 
                                     dane['low'].iloc[i-okno:i+1]).std(),
                        'sentyment': sentyment['sentyment'],
                        'sentyment_pewnosc': sentyment['pewnosc'],
                        'sentyment_wzmianki': sentyment['liczba_wzmianek']
                    }
                    
                    # Dodaj do bazy RAG
                    if not self.rag.dodaj_dokument(opis, metadata):
                        self.logger.warning(f"⚠️ Nie udało się dodać wzorca {metadata['id']}")
                        continue
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Błąd podczas przetwarzania wzorca: {str(e)}")
                    continue
                    
            self.logger.info(f"🥷 Zindeksowano {len(dane) // krok} wzorców")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas indeksowania: {str(e)}")
            return False

    async def aktualizuj_dane(
        self,
        dane: pd.DataFrame,
        okno: int = 20
    ) -> bool:
        """
        Aktualizuje bazę o najnowszy wzorzec.
        
        Args:
            dane: DataFrame z danymi rynkowymi (OHLCV)
            okno: Rozmiar okna do analizy wzorców
            
        Returns:
            bool: True jeśli aktualizacja się powiodła
        """
        try:
            if len(dane) < okno:
                return False
                
            # Generuj opis dla ostatniego wzorca
            opis = await self._przygotuj_opis_wzorca(dane, len(dane)-1, okno)
            
            # Analizuj sentyment
            sentyment = await self._analizuj_sentyment_rynku(dane, len(dane)-okno-1, len(dane)-1)
            
            # Przygotuj metadane
            metadata = {
                'id': f"wzorzec_{dane.index[-1].strftime('%Y%m%d_%H%M')}",
                'data': dane.index[-1].isoformat(),
                'zmiana_proc': ((dane['close'].iloc[-1] - dane['close'].iloc[-okno])
                              / dane['close'].iloc[-okno] * 100),
                'wolumen_sredni': dane['volume'].iloc[-okno:].mean(),
                'zmiennosc': (dane['high'].iloc[-okno:] - 
                             dane['low'].iloc[-okno:]).std(),
                'sentyment': sentyment['sentyment'],
                'sentyment_pewnosc': sentyment['pewnosc'],
                'sentyment_wzmianki': sentyment['liczba_wzmianek']
            }
            
            # Dodaj/zaktualizuj w bazie RAG
            if not self.rag.dodaj_dokument(opis, metadata, True):
                self.logger.error("❌ Nie udało się zaktualizować bazy")
                return False
            
            self.logger.info("🥷 Zaktualizowano bazę o najnowszy wzorzec")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas aktualizacji: {str(e)}")
            return False
            
    def _cache_key(self, dane: pd.DataFrame, okno: int) -> str:
        """
        Generuje klucz cache'a dla danych wzorców.
        
        Args:
            dane: DataFrame z danymi
            okno: Rozmiar okna
            
        Returns:
            str: Klucz cache'a
        """
        ostatnia_data = dane.index[-1].strftime('%Y%m%d_%H%M')
        return f"wzorzec_{ostatnia_data}_{okno}"
        
    def _cache_wynik(self, klucz: str, wynik: List[Dict[str, Any]]) -> None:
        """
        Zapisuje wynik w cache'u.
        
        Args:
            klucz: Klucz cache'a
            wynik: Wynik do zapisania
        """
        self._cache[klucz] = {
            'wynik': wynik,
            'timestamp': datetime.now(),
            'ttl': timedelta(minutes=5)  # Cache ważny przez 5 minut
        }
        
    def _wyczysc_cache(self) -> None:
        """Czyści wygasłe wpisy z cache'a."""
        teraz = datetime.now()
        do_usuniecia = []
        
        for klucz, wpis in self._cache.items():
            if teraz - wpis['timestamp'] > wpis['ttl']:
                do_usuniecia.append(klucz)
                
        for klucz in do_usuniecia:
            del self._cache[klucz]
            
    async def uruchom_automatyczne_aktualizacje(
        self,
        dane: pd.DataFrame,
        okno: int = 20,
        interwal: int = 60  # sekundy
    ) -> None:
        """
        Uruchamia automatyczne aktualizacje bazy wzorców.
        
        Args:
            dane: DataFrame z danymi rynkowymi
            okno: Rozmiar okna do analizy
            interwal: Częstotliwość aktualizacji w sekundach
        """
        import asyncio
        
        while True:
            try:
                # Aktualizuj bazę
                await self.aktualizuj_dane(dane, okno)
                
                # Wyczyść cache
                self._wyczysc_cache()
                
                self.logger.info("🥷 Wykonano automatyczną aktualizację bazy wzorców")
                
            except Exception as e:
                self.logger.error(f"❌ Błąd podczas automatycznej aktualizacji: {str(e)}")
                
            await asyncio.sleep(interwal)

    async def znajdz_podobne_wzorce(
        self,
        dane: pd.DataFrame,
        okno: int = 20,
        limit: int = 5,
        filtry: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Znajduje podobne wzorce cenowe w bazie.
        
        Args:
            dane: DataFrame z danymi rynkowymi (OHLCV)
            okno: Rozmiar okna do analizy wzorców
            limit: Maksymalna liczba podobnych wzorców do zwrócenia
            filtry: Opcjonalne filtry metadanych
            
        Returns:
            Lista słowników z podobnymi wzorcami i ich metadanymi
        """
        try:
            if len(dane) < okno:
                return []
                
            # Generuj opis dla aktualnego wzorca
            opis = await self._przygotuj_opis_wzorca(dane, len(dane)-1, okno)
            
            # Przygotuj filtry
            if filtry is None:
                filtry = {}
                
            # Znajdź podobne wzorce
            wyniki = self.rag.znajdz_podobne(opis, limit, filtry)
            
            if not wyniki:
                self.logger.warning("⚠️ Nie znaleziono podobnych wzorców")
                return []
                
            self.logger.info(f"🥷 Znaleziono {len(wyniki)} podobnych wzorców")
            return wyniki
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas wyszukiwania: {str(e)}")
            return []

    async def _analizuj_sentyment_rynku(
        self,
        dane: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> Dict[str, Any]:
        """
        Analizuje sentyment rynku na podstawie danych społecznościowych.
        
        Args:
            dane: DataFrame z danymi OHLCV
            start_idx: Początkowy indeks
            end_idx: Końcowy indeks
            
        Returns:
            Dict[str, Any]: Wyniki analizy sentymentu
        """
        try:
            # Pobierz datę dla analizowanego okresu
            data = dane.index[end_idx]
            
            # Pobierz wzmianki z mediów społecznościowych
            wzmianki = await self.scraper.aktualizuj_dane()
            
            # Filtruj wzmianki z odpowiedniego okresu
            okres_start = data - timedelta(hours=24)
            wzmianki_okresu = [
                w for w in wzmianki 
                if okres_start <= w.data <= data
            ]
            
            if not wzmianki_okresu:
                return {
                    "sentyment": "NEUTRALNY",
                    "pewnosc": 0.0,
                    "liczba_wzmianek": 0
                }
            
            # Analizuj sentyment wzmianek
            wyniki = await self.analizator_sentymentu.analizuj_wzmianki(wzmianki_okresu)
            
            return {
                "sentyment": "POZYTYWNY" if wyniki["sredni_sentyment"] > 0.2 
                            else "NEGATYWNY" if wyniki["sredni_sentyment"] < -0.2 
                            else "NEUTRALNY",
                "pewnosc": abs(wyniki["sredni_sentyment"]),
                "liczba_wzmianek": len(wzmianki_okresu),
                "pozytywne_proc": wyniki["pozytywne_proc"],
                "neutralne_proc": wyniki["neutralne_proc"],
                "negatywne_proc": wyniki["negatywne_proc"]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas analizy sentymentu: {str(e)}")
            return {
                "sentyment": "NEUTRALNY",
                "pewnosc": 0.0,
                "liczba_wzmianek": 0
            }

    async def filtruj_po_sentymencie(
        self,
        sentyment: str,
        min_pewnosc: float = 0.2,
        min_wzmianki: int = 5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Filtruje wzorce po określonym sentymencie.
        
        Args:
            sentyment: Oczekiwany sentyment (POZYTYWNY/NEUTRALNY/NEGATYWNY)
            min_pewnosc: Minimalna pewność sentymentu (0-1)
            min_wzmianki: Minimalna liczba wzmianek
            limit: Maksymalna liczba wzorców do zwrócenia
            
        Returns:
            Lista wzorców spełniających kryteria
        """
        filtry = {
            'sentyment': sentyment,
            'sentyment_pewnosc': {'$gte': min_pewnosc},
            'sentyment_wzmianki': {'$gte': min_wzmianki}
        }
        
        wyniki = self.rag.znajdz_podobne("", limit, filtry)
        
        if not wyniki:
            self.logger.warning(f"⚠️ Nie znaleziono wzorców o sentymencie {sentyment}")
            return []
            
        self.logger.info(f"🥷 Znaleziono {len(wyniki)} wzorców o sentymencie {sentyment}")
        return wyniki
        
    async def agreguj_sentyment(
        self,
        dane: pd.DataFrame,
        okno_dni: int = 7
    ) -> Dict[str, Any]:
        """
        Agreguje sentyment z ostatnich dni.
        
        Args:
            dane: DataFrame z danymi OHLCV
            okno_dni: Liczba dni do analizy
            
        Returns:
            Dict z zagregowanymi statystykami sentymentu
        """
        try:
            data_koncowa = dane.index[-1]
            data_poczatkowa = data_koncowa - timedelta(days=okno_dni)
            
            # Pobierz wszystkie wzmianki z okresu
            wzmianki = await self.scraper.pobierz_wzmianki(
                data_poczatkowa,
                data_koncowa
            )
            
            if not wzmianki:
                return {
                    "sredni_sentyment": 0.0,
                    "zmiana_sentymentu": 0.0,
                    "liczba_wzmianek": 0,
                    "rozklad_sentymentu": {
                        "pozytywne": 0,
                        "neutralne": 0,
                        "negatywne": 0
                    },
                    "trendy": []
                }
                
            # Analizuj sentyment dla wszystkich wzmianek
            wyniki = await self.analizator_sentymentu.analizuj_wzmianki(wzmianki)
            
            # Oblicz zmianę sentymentu
            wzmianki_sorted = sorted(wzmianki, key=lambda x: x.data)
            polowa = len(wzmianki_sorted) // 2
            
            wyniki_pierwsze = await self.analizator_sentymentu.analizuj_wzmianki(
                wzmianki_sorted[:polowa]
            )
            wyniki_drugie = await self.analizator_sentymentu.analizuj_wzmianki(
                wzmianki_sorted[polowa:]
            )
            
            zmiana_sentymentu = wyniki_drugie["sredni_sentyment"] - wyniki_pierwsze["sredni_sentyment"]
            
            # Wykryj trendy w sentymencie
            trendy = []
            if zmiana_sentymentu > 0.1:
                trendy.append("Rosnący optymizm")
            elif zmiana_sentymentu < -0.1:
                trendy.append("Rosnący pesymizm")
                
            if wyniki["sredni_sentyment"] > 0.3:
                trendy.append("Silny optymizm")
            elif wyniki["sredni_sentyment"] < -0.3:
                trendy.append("Silny pesymizm")
                
            return {
                "sredni_sentyment": wyniki["sredni_sentyment"],
                "zmiana_sentymentu": zmiana_sentymentu,
                "liczba_wzmianek": len(wzmianki),
                "rozklad_sentymentu": {
                    "pozytywne": wyniki["pozytywne_proc"],
                    "neutralne": wyniki["neutralne_proc"],
                    "negatywne": wyniki["negatywne_proc"]
                },
                "trendy": trendy
            }
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas agregacji sentymentu: {str(e)}")
            return {
                "sredni_sentyment": 0.0,
                "zmiana_sentymentu": 0.0,
                "liczba_wzmianek": 0,
                "rozklad_sentymentu": {
                    "pozytywne": 0,
                    "neutralne": 0,
                    "negatywne": 0
                },
                "trendy": []
            }
            
    async def generuj_raport_sentymentu(
        self,
        dane: pd.DataFrame,
        okno_dni: int = 7
    ) -> str:
        """
        Generuje raport tekstowy o sentymencie rynku.
        
        Args:
            dane: DataFrame z danymi OHLCV
            okno_dni: Liczba dni do analizy
            
        Returns:
            str: Raport tekstowy
        """
        try:
            # Pobierz zagregowane dane
            agregacja = await self.agreguj_sentyment(dane, okno_dni)
            
            # Znajdź wzorce o podobnym sentymencie
            sentyment = ("POZYTYWNY" if agregacja["sredni_sentyment"] > 0.2
                        else "NEGATYWNY" if agregacja["sredni_sentyment"] < -0.2
                        else "NEUTRALNY")
            
            podobne = await self.filtruj_po_sentymencie(
                sentyment,
                min_pewnosc=abs(agregacja["sredni_sentyment"]),
                limit=3
            )
            
            # Generuj raport
            raport = f"""📊 Raport sentymentu rynku ({dane.index[-okno_dni].strftime('%Y-%m-%d')} - {dane.index[-1].strftime('%Y-%m-%d')})

🔍 Analiza ogólna:
- Średni sentyment: {agregacja["sredni_sentyment"]:.2f}
- Zmiana sentymentu: {agregacja["zmiana_sentymentu"]:.2f}
- Liczba przeanalizowanych wzmianek: {agregacja["liczba_wzmianek"]}

📈 Rozkład sentymentu:
- Pozytywne: {agregacja["rozklad_sentymentu"]["pozytywne"]:.1f}%
- Neutralne: {agregacja["rozklad_sentymentu"]["neutralne"]:.1f}%
- Negatywne: {agregacja["rozklad_sentymentu"]["negatywne"]:.1f}%

🎯 Wykryte trendy:"""

            if agregacja["trendy"]:
                for trend in agregacja["trendy"]:
                    raport += f"\n- {trend}"
            else:
                raport += "\n- Brak wyraźnych trendów"
                
            if podobne:
                raport += "\n\n🔄 Podobne wzorce historyczne:"
                for wzorzec in podobne:
                    raport += f"\n- {wzorzec['metadata']['data']}: {wzorzec['metadata']['zmiana_proc']:.1f}% zmiana"
                    
            return raport
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas generowania raportu: {str(e)}")
            return "❌ Nie udało się wygenerować raportu sentymentu"

    async def indeksuj_wydarzenia_kalendarz(
        self,
        wydarzenia: pd.DataFrame,
        aktualizuj: bool = False
    ) -> bool:
        """
        Indeksuje wydarzenia z kalendarza ekonomicznego w systemie RAG.
        
        Args:
            wydarzenia: DataFrame z wydarzeniami
            aktualizuj: Czy aktualizować istniejące dokumenty
            
        Returns:
            bool: True jeśli indeksowanie się powiodło
        """
        try:
            for _, wydarzenie in wydarzenia.iterrows():
                # Przygotuj opis wydarzenia
                opis = f"Wydarzenie ekonomiczne: {wydarzenie['nazwa']}\n"
                opis += f"Data: {wydarzenie['time'].strftime('%Y-%m-%d %H:%M')}\n"
                opis += f"Kraj: {wydarzenie['kraj']}, Waluta: {wydarzenie['waluta']}\n"
                opis += f"Ważność: {'❗' * wydarzenie['waznosc']}\n\n"
                
                if pd.notna(wydarzenie['wartosc_aktualna']):
                    opis += f"Wartość aktualna: {wydarzenie['wartosc_aktualna']}\n"
                if pd.notna(wydarzenie['wartosc_prognoza']):
                    opis += f"Prognoza: {wydarzenie['wartosc_prognoza']}\n"
                if pd.notna(wydarzenie['wartosc_poprzednia']):
                    opis += f"Poprzednia wartość: {wydarzenie['wartosc_poprzednia']}\n"
                if pd.notna(wydarzenie['rewizja']):
                    opis += f"Rewizja: {wydarzenie['rewizja']}\n"
                
                # Oblicz wpływ wydarzenia
                if pd.notna(wydarzenie['wartosc_aktualna']) and pd.notna(wydarzenie['wartosc_prognoza']):
                    roznica = wydarzenie['wartosc_aktualna'] - wydarzenie['wartosc_prognoza']
                    if abs(roznica) > 0:
                        opis += f"\nRóżnica od prognozy: {roznica:+.2f}\n"
                        if roznica > 0:
                            opis += "Lepszy wynik niż oczekiwano 📈\n"
                        else:
                            opis += "Gorszy wynik niż oczekiwano 📉\n"
                
                # Dodaj do systemu RAG
                await self.rag.dodaj_dokument(
                    tekst=opis,
                    metadata={
                        'id': f"kalendarz_{wydarzenie['event_id']}_{wydarzenie['time'].strftime('%Y%m%d%H%M')}",
                        'typ': 'kalendarz',
                        'timestamp': wydarzenie['time'].isoformat(),
                        'event_id': wydarzenie['event_id'],
                        'waluta': wydarzenie['waluta'],
                        'kraj': wydarzenie['kraj'],
                        'waznosc': int(wydarzenie['waznosc'])
                    },
                    aktualizuj=aktualizuj
                )
            
            self.logger.info(f"🥷 Zaindeksowano {len(wydarzenia)} wydarzeń w systemie RAG")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd indeksowania wydarzeń: {str(e)}")
            return False 