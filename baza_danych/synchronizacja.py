"""
Moduł do synchronizacji danych między MT5 a bazą danych.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
import json
import MetaTrader5 as mt5
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_

from baza_danych.modele import (
    HistoriaCen, SynchronizacjaDanych, ZadanieAktualizacji, 
    StatusSynchronizacji, Cache, Aktyw, StatusRynku, KalendarzEkonomiczny
)
from baza_danych.cache import MenedzerCache

logger = logging.getLogger(__name__)

# Mapowanie timeframes na stałe MT5
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}

# Lista obsługiwanych symboli
SYMBOLS = {
    "JP225",     # Nikkei 225
    "USDJPY",    # Potrzebny do analizy korelacji
    "JPN225",    # Alternatywny symbol dla Nikkei
    "NI225",     # Jeszcze jeden wariant Nikkei
}

class SynchronizatorMT5:
    """Klasa odpowiedzialna za synchronizację danych z MT5."""
    
    def __init__(self, session: Session):
        """
        Inicjalizacja synchronizatora.
        
        Args:
            session: Sesja SQLAlchemy
        """
        self.session = session
        self.logger = logger
        self.cache = MenedzerCache(session)
        
    @staticmethod
    def dostepne_timeframes() -> Set[str]:
        """
        Zwraca listę dostępnych timeframes.
        
        Returns:
            Set[str]: Lista dostępnych timeframes
        """
        return set(TIMEFRAMES.keys())
        
    @staticmethod
    def dostepne_symbole() -> Set[str]:
        """
        Zwraca listę obsługiwanych symboli.
        
        Returns:
            Set[str]: Lista obsługiwanych symboli
        """
        return SYMBOLS
        
    async def sprawdz_dostepnosc_symbolu(self, symbol: str) -> bool:
        """
        Sprawdza czy symbol jest dostępny w MT5.
        
        Args:
            symbol: Symbol do sprawdzenia
            
        Returns:
            bool: True jeśli symbol jest dostępny
        """
        # Najpierw sprawdź w cache
        klucz_cache = f"symbol_dostepny_{symbol}"
        dostepnosc = await self.cache.pobierz(klucz_cache)
        if dostepnosc is not None:
            return dostepnosc
            
        # Jeśli nie ma w cache, sprawdź w MT5
        if symbol not in self.dostepne_symbole():
            self.logger.warning(f"⚠️ Symbol {symbol} nie jest obsługiwany")
            await self.cache.dodaj(klucz_cache, False, timedelta(hours=1))
            return False
            
        info = mt5.symbol_info(symbol)
        dostepny = info is not None
        
        if not dostepny:
            self.logger.error(f"❌ Symbol {symbol} nie jest dostępny w MT5")
            
        # Zapisz w cache na godzinę
        await self.cache.dodaj(klucz_cache, dostepny, timedelta(hours=1))
        return dostepny
        
    async def synchronizuj_historie(
        self,
        symbol: str,
        timeframe: str,
        od: datetime,
        do: Optional[datetime] = None
    ) -> bool:
        """
        Synchronizuje historię cen dla danego symbolu i timeframe'u.
        
        Args:
            symbol: Symbol (np. "JP225")
            timeframe: Timeframe (np. "M1", "H1")
            od: Data początkowa
            do: Data końcowa (domyślnie: teraz)
            
        Returns:
            bool: True jeśli synchronizacja się powiodła
        """
        sync = None
        try:
            # Walidacja parametrów
            if timeframe not in TIMEFRAMES:
                raise ValueError(f"Nieobsługiwany timeframe: {timeframe}")
                
            if not await self.sprawdz_dostepnosc_symbolu(symbol):
                return False
                
            # Sprawdź czy mamy dane w cache
            klucz_cache = f"historia_{symbol}_{timeframe}_{od.isoformat()}_{do.isoformat() if do else 'now'}"
            dane_cache = await self.cache.pobierz(klucz_cache)
            if dane_cache:
                self.logger.info(f"🥷 Użyto cache dla {symbol} {timeframe}")
                return True
                
            # Ustawienie parametrów
            do = do or datetime.utcnow()
            
            # Utworzenie rekordu synchronizacji
            sync = SynchronizacjaDanych(
                symbol=symbol,
                timeframe=timeframe,
                zakres_od=od,
                zakres_do=do,
                status=StatusSynchronizacji.W_TOKU
            )
            self.session.add(sync)
            self.session.commit()
            
            # Pobranie danych z MT5
            rates = mt5.copy_rates_range(
                symbol,
                TIMEFRAMES[timeframe],
                od,
                do
            )
            
            if rates is None:
                raise Exception(f"Nie udało się pobrać danych dla {symbol} {timeframe}")
            
            # Konwersja na rekordy bazy danych
            rekordy = []
            for rate in rates:
                rekord = HistoriaCen(
                    timestamp=datetime.fromtimestamp(rate['time']),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=rate['open'],
                    high=rate['high'],
                    low=rate['low'],
                    close=rate['close'],
                    volume=rate['real_volume']
                )
                rekordy.append(rekord)
            
            # Zapis do bazy
            self.session.bulk_save_objects(rekordy)
            
            # Aktualizacja statusu synchronizacji
            sync.status = StatusSynchronizacji.ZAKONCZONA
            sync.liczba_rekordow = len(rekordy)
            sync.timestamp_koniec = datetime.utcnow()
            
            # Zapisz w cache na 5 minut
            await self.cache.dodaj(
                klucz_cache,
                {
                    'liczba_rekordow': len(rekordy),
                    'timestamp': datetime.utcnow().isoformat()
                },
                timedelta(minutes=5)
            )
            
            self.session.commit()
            self.logger.info(f"🥷 Zsynchronizowano {len(rekordy)} rekordów dla {symbol} {timeframe}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd synchronizacji {symbol} {timeframe}: {str(e)}")
            if sync:
                sync.status = StatusSynchronizacji.BLAD
                sync.blad = str(e)
                sync.timestamp_koniec = datetime.utcnow()
                self.session.commit()
            return False
            
    async def przetworz_zadania(self) -> None:
        """Przetwarza kolejkę zadań aktualizacji."""
        try:
            # Pobierz niezakończone zadania posortowane po priorytecie
            zadania = self.session.execute(
                select(ZadanieAktualizacji)
                .where(ZadanieAktualizacji.wykonane == False)
                .order_by(ZadanieAktualizacji.priorytet.desc())
            ).scalars().all()
            
            for zadanie in zadania:
                try:
                    if zadanie.typ == 'historia_cen':
                        # Dla historii cen synchronizujemy ostatnie 24h
                        od = datetime.utcnow() - timedelta(days=1)
                        sukces = await self.synchronizuj_historie(
                            zadanie.symbol,
                            zadanie.timeframe,
                            od
                        )
                    elif zadanie.typ == 'status':
                        # Aktualizacja statusu rynku
                        sukces = await self.aktualizuj_status_rynku(zadanie.symbol)
                    else:
                        self.logger.warning(f"⚠️ Nieznany typ zadania: {zadanie.typ}")
                        sukces = False
                        
                    zadanie.wykonane = sukces
                    if not sukces:
                        zadanie.blad = "Nie udało się wykonać zadania"
                        
                except Exception as e:
                    zadanie.wykonane = False
                    zadanie.blad = str(e)
                    
                self.session.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Błąd przetwarzania zadań: {str(e)}")
            
    async def aktualizuj_status_rynku(self, symbol: str) -> bool:
        """
        Aktualizuje status rynku dla danego symbolu.
        
        Args:
            symbol: Symbol do sprawdzenia
            
        Returns:
            bool: True jeśli aktualizacja się powiodła
        """
        try:
            # Sprawdź w cache
            klucz_cache = f"status_rynku_{symbol}"
            status_cache = await self.cache.pobierz(klucz_cache)
            if status_cache:
                self.logger.info(f"🥷 Użyto cache dla statusu {symbol}")
                return True
                
            if not await self.sprawdz_dostepnosc_symbolu(symbol):
                return False
                
            # Pobranie informacji o symbolu
            info = mt5.symbol_info(symbol)
            if info is None:
                raise Exception(f"Nie znaleziono symbolu {symbol}")
                
            # Aktualizacja statusu
            status = StatusRynku(
                symbol=symbol,
                otwarty=bool(info.trade_mode != 0),  # 0 oznacza wyłączony handel
                powod=None if info.trade_mode != 0 else "Handel wyłączony",
                nastepne_otwarcie=None  # TODO: Dodać logikę dla następnego otwarcia
            )
            
            self.session.add(status)
            
            # Zapisz w cache na minutę
            await self.cache.dodaj(
                klucz_cache,
                {
                    'otwarty': status.otwarty,
                    'timestamp': datetime.utcnow().isoformat()
                },
                timedelta(minutes=1)
            )
            
            self.session.commit()
            
            self.logger.info(f"🥷 Zaktualizowano status rynku dla {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd aktualizacji statusu {symbol}: {str(e)}")
            return False 

class AktualizatorKalendarza:
    """Klasa odpowiedzialna za aktualizację kalendarza ekonomicznego."""
    
    def __init__(self, session: Session):
        """
        Inicjalizacja aktualizatora.
        
        Args:
            session: Sesja SQLAlchemy
        """
        self.session = session
        self.logger = logging.getLogger(__name__)
        self.cache = MenedzerCache(session)
        
    async def aktualizuj_kalendarz(
        self,
        waluty: Optional[List[str]] = None,
        kraje: Optional[List[str]] = None,
        min_waznosc: int = 1
    ) -> bool:
        """
        Aktualizuje dane kalendarza ekonomicznego.
        
        Args:
            waluty: Lista kodów walut do aktualizacji
            kraje: Lista kodów krajów do aktualizacji
            min_waznosc: Minimalna ważność wydarzeń (1-3)
            
        Returns:
            bool: True jeśli aktualizacja się powiodła
        """
        try:
            # Sprawdź cache
            klucz_cache = f"kalendarz_last_update_{','.join(waluty or [])}_{','.join(kraje or [])}_{min_waznosc}"
            if await self.cache.pobierz(klucz_cache):
                return True
                
            # Pobierz ostatnią aktualizację z bazy
            ostatnia = self.session.query(KalendarzEkonomiczny.timestamp)\
                .order_by(KalendarzEkonomiczny.timestamp.desc())\
                .first()
                
            start_date = ostatnia[0] if ostatnia else datetime.utcnow() - timedelta(days=1)
            
            # Utwórz zadanie aktualizacji
            zadanie = ZadanieAktualizacji(
                typ='kalendarz',
                priorytet=2,  # Wysoki priorytet dla danych ekonomicznych
                timestamp=start_date
            )
            self.session.add(zadanie)
            
            # Pobierz wydarzenia dla każdej waluty/kraju
            events = []
            if waluty:
                for waluta in waluty:
                    waluta_events = mt5.calendar_event_by_currency(waluta)
                    if waluta_events:
                        events.extend(waluta_events)
                        
            if kraje:
                for kraj in kraje:
                    kraj_events = mt5.calendar_event_by_country(kraj)
                    if kraj_events:
                        events.extend(kraj_events)
                        
            if not waluty and not kraje:
                # Pobierz wszystkie wydarzenia
                countries = mt5.calendar_countries()
                for country in countries:
                    kraj_events = mt5.calendar_event_by_country(country['code'])
                    if kraj_events:
                        events.extend(kraj_events)
            
            # Filtruj po ważności
            events = [e for e in events if e['importance'] >= min_waznosc]
            
            if not events:
                self.logger.warning("⚠️ Brak nowych wydarzeń do aktualizacji")
                return True
                
            # Pobierz wartości dla wydarzeń
            values = mt5.calendar_value_history(
                datetime_from=int(start_date.timestamp()),
                datetime_to=int(datetime.utcnow().timestamp()),
                event_ids=[e['id'] for e in events]
            )
            
            if not values:
                self.logger.warning("⚠️ Brak nowych wartości do aktualizacji")
                return True
                
            # Przygotuj słownik wydarzeń
            events_dict = {e['id']: e for e in events}
            
            # Aktualizuj bazę danych
            for value in values:
                event = events_dict[value['event_id']]
                self.session.merge(KalendarzEkonomiczny(
                    timestamp=datetime.fromtimestamp(value['time']),
                    event_id=value['event_id'],
                    nazwa=event['name'],
                    waluta=event['currency'],
                    kraj=event['country'],
                    waznosc=event['importance'],
                    wartosc_aktualna=value.get('actual'),
                    wartosc_prognoza=value.get('forecast'),
                    wartosc_poprzednia=value.get('prev'),
                    rewizja=value.get('revised')
                ))
            
            # Oznacz zadanie jako wykonane
            zadanie.wykonane = True
            self.session.commit()
            
            # Zapisz w cache na 5 minut
            await self.cache.dodaj(
                klucz_cache,
                {'timestamp': datetime.utcnow().isoformat()},
                timedelta(minutes=5)
            )
            
            self.logger.info(f"🥷 Zaktualizowano {len(values)} wydarzeń w kalendarzu")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd aktualizacji kalendarza: {str(e)}")
            if zadanie:
                zadanie.blad = str(e)
                self.session.commit()
            return False 