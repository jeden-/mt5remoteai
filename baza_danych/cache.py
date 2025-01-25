"""
Moduł do zarządzania cache'm danych.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_

from baza_danych.modele import Cache

logger = logging.getLogger(__name__)

class MenedzerCache:
    """Klasa do zarządzania cache'm danych."""
    
    def __init__(self, session: Session):
        """
        Inicjalizacja menedżera cache.
        
        Args:
            session: Sesja SQLAlchemy
        """
        self.session = session
        self.logger = logger
        
    async def dodaj(
        self,
        klucz: str,
        wartosc: Any,
        czas_wygasniecia: Optional[timedelta] = None
    ) -> bool:
        """
        Dodaje wartość do cache'u.
        
        Args:
            klucz: Klucz cache'u
            wartosc: Wartość do zapisania
            czas_wygasniecia: Opcjonalny czas wygaśnięcia
            
        Returns:
            bool: True jeśli dodano pomyślnie
        """
        try:
            # Usunięcie starego wpisu jeśli istnieje
            await self.usun(klucz)
            
            # Serializacja wartości do JSON
            wartosc_json = json.dumps(wartosc)
            
            # Obliczenie czasu wygaśnięcia
            wygasa = None
            if czas_wygasniecia:
                wygasa = datetime.utcnow() + czas_wygasniecia
            
            # Utworzenie nowego wpisu
            cache = Cache(
                klucz=klucz,
                wartosc=wartosc_json,
                timestamp=datetime.utcnow(),
                wygasa=wygasa
            )
            
            self.session.add(cache)
            self.session.commit()
            
            self.logger.info(f"🥷 Dodano do cache: {klucz}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd dodawania do cache: {str(e)}")
            return False
            
    async def pobierz(self, klucz: str) -> Optional[Any]:
        """
        Pobiera wartość z cache'u.
        
        Args:
            klucz: Klucz cache'u
            
        Returns:
            Any: Wartość z cache'u lub None jeśli nie znaleziono/wygasła
        """
        try:
            # Pobranie wpisu z bazy
            cache = self.session.execute(
                select(Cache)
                .where(
                    and_(
                        Cache.klucz == klucz,
                        or_(
                            Cache.wygasa.is_(None),
                            Cache.wygasa > datetime.utcnow()
                        )
                    )
                )
            ).scalar_one_or_none()
            
            if cache is None:
                return None
                
            # Deserializacja wartości z JSON
            wartosc = json.loads(cache.wartosc)
            
            self.logger.info(f"🥷 Pobrano z cache: {klucz}")
            return wartosc
            
        except Exception as e:
            self.logger.error(f"❌ Błąd pobierania z cache: {str(e)}")
            return None
            
    async def usun(self, klucz: str) -> bool:
        """
        Usuwa wartość z cache'u.
        
        Args:
            klucz: Klucz cache'u
            
        Returns:
            bool: True jeśli usunięto pomyślnie
        """
        try:
            self.session.execute(
                Cache.__table__.delete().where(Cache.klucz == klucz)
            )
            self.session.commit()
            
            self.logger.info(f"🥷 Usunięto z cache: {klucz}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd usuwania z cache: {str(e)}")
            return False
            
    async def wyczysc_wygasle(self) -> int:
        """
        Usuwa wygasłe wpisy z cache'u.
        
        Returns:
            int: Liczba usuniętych wpisów
        """
        try:
            teraz = datetime.utcnow()
            wynik = self.session.execute(
                Cache.__table__.delete().where(
                    and_(
                        Cache.wygasa.isnot(None),
                        Cache.wygasa <= teraz
                    )
                )
            )
            self.session.commit()
            
            liczba = wynik.rowcount
            if liczba > 0:
                self.logger.info(f"🥷 Usunięto {liczba} wygasłych wpisów z cache")
            return liczba
            
        except Exception as e:
            self.logger.error(f"❌ Błąd czyszczenia cache: {str(e)}")
            return 0
            
    async def pobierz_wiele(self, klucze: List[str]) -> Dict[str, Any]:
        """
        Pobiera wiele wartości z cache'u.
        
        Args:
            klucze: Lista kluczy do pobrania
            
        Returns:
            Dict[str, Any]: Słownik z pobranymi wartościami
        """
        try:
            # Pobranie wpisów z bazy
            cache_wpisy = self.session.execute(
                select(Cache)
                .where(
                    and_(
                        Cache.klucz.in_(klucze),
                        or_(
                            Cache.wygasa.is_(None),
                            Cache.wygasa > datetime.utcnow()
                        )
                    )
                )
            ).scalars().all()
            
            # Deserializacja wartości
            wynik = {}
            for wpis in cache_wpisy:
                try:
                    wynik[wpis.klucz] = json.loads(wpis.wartosc)
                except Exception as e:
                    self.logger.error(f"❌ Błąd deserializacji {wpis.klucz}: {str(e)}")
                    
            self.logger.info(f"🥷 Pobrano {len(wynik)} wpisów z cache")
            return wynik
            
        except Exception as e:
            self.logger.error(f"❌ Błąd pobierania wielu z cache: {str(e)}")
            return {} 