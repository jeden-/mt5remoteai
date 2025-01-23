"""
Moduł obsługujący połączenie z bazą danych.
"""

import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from .modele import (
    Base, Transakcja, Pozycja, HistoriaCen, 
    Wiadomosc, SesjaHandlowa, StatusRynku, 
    Aktyw, MetrykiHandlu
)

logger = logging.getLogger(__name__)

class BazaDanych:
    """Klasa obsługująca połączenie z bazą danych."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicjalizacja połączenia z bazą danych.
        
        Args:
            config: Konfiguracja połączenia z bazą
        """
        self.logger = logger
        self.config = config
        self.engine = None
        self.Session = None
        
    def inicjalizuj(self) -> bool:
        """
        Inicjalizuje połączenie z bazą danych.
        
        Returns:
            bool: True jeśli połączenie udane
        """
        try:
            # Tworzenie URL połączenia
            url = f"postgresql://{self.config['DB_USER']}:{self.config['DB_PASSWORD']}@{self.config['DB_HOST']}:{self.config['DB_PORT']}/{self.config['DB_NAME']}"
            
            # Tworzenie silnika bazy danych
            self.engine = create_engine(url)
            
            # Tworzenie wszystkich tabel
            Base.metadata.create_all(self.engine)
            
            # Tworzenie fabryki sesji
            self.Session = sessionmaker(bind=self.engine)
            
            self.logger.info("🥷 Połączono z bazą danych")
            return True
            
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd połączenia z bazą: {str(e)}")
            return False
            
    def dodaj_transakcje(self, transakcja: Dict[str, Any]) -> Optional[Transakcja]:
        """
        Dodaje nową transakcję do bazy.
        
        Args:
            transakcja: Dane transakcji
            
        Returns:
            Optional[Transakcja]: Utworzona transakcja lub None w przypadku błędu
        """
        try:
            with self.Session() as sesja:
                nowa_transakcja = Transakcja(**transakcja)
                sesja.add(nowa_transakcja)
                sesja.commit()
                return nowa_transakcja
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd dodawania transakcji: {str(e)}")
            return None
            
    def dodaj_pozycje(self, pozycja: Dict[str, Any]) -> Optional[Pozycja]:
        """
        Dodaje nową pozycję do bazy.
        
        Args:
            pozycja: Dane pozycji
            
        Returns:
            Optional[Pozycja]: Utworzona pozycja lub None w przypadku błędu
        """
        try:
            with self.Session() as sesja:
                nowa_pozycja = Pozycja(**pozycja)
                sesja.add(nowa_pozycja)
                sesja.commit()
                return nowa_pozycja
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd dodawania pozycji: {str(e)}")
            return None
            
    def dodaj_cene(self, cena: Dict[str, Any]) -> Optional[HistoriaCen]:
        """
        Dodaje nowy rekord ceny do historii.
        
        Args:
            cena: Dane cenowe
            
        Returns:
            Optional[HistoriaCen]: Utworzony rekord lub None w przypadku błędu
        """
        try:
            with self.Session() as sesja:
                nowa_cena = HistoriaCen(**cena)
                sesja.add(nowa_cena)
                sesja.commit()
                return nowa_cena
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd dodawania ceny: {str(e)}")
            return None
            
    def dodaj_wiadomosc(self, wiadomosc: Dict[str, Any]) -> Optional[Wiadomosc]:
        """
        Dodaje nową wiadomość do bazy.
        
        Args:
            wiadomosc: Dane wiadomości
            
        Returns:
            Optional[Wiadomosc]: Utworzona wiadomość lub None w przypadku błędu
        """
        try:
            with self.Session() as sesja:
                nowa_wiadomosc = Wiadomosc(**wiadomosc)
                sesja.add(nowa_wiadomosc)
                sesja.commit()
                return nowa_wiadomosc
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd dodawania wiadomości: {str(e)}")
            return None
            
    def dodaj_sesje_handlowa(self, sesja_handlowa: Dict[str, Any]) -> Optional[SesjaHandlowa]:
        """
        Dodaje nową sesję handlową do bazy.
        
        Args:
            sesja_handlowa: Dane sesji
            
        Returns:
            Optional[SesjaHandlowa]: Utworzona sesja lub None w przypadku błędu
        """
        try:
            with self.Session() as sesja:
                nowa_sesja = SesjaHandlowa(**sesja_handlowa)
                sesja.add(nowa_sesja)
                sesja.commit()
                return nowa_sesja
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd dodawania sesji handlowej: {str(e)}")
            return None
            
    def aktualizuj_status_rynku(self, symbol: str, status: Dict[str, Any]) -> Optional[StatusRynku]:
        """
        Aktualizuje status rynku dla danego symbolu.
        
        Args:
            symbol: Symbol (np. "EURUSD")
            status: Dane statusu
            
        Returns:
            Optional[StatusRynku]: Zaktualizowany status lub None w przypadku błędu
        """
        try:
            with self.Session() as sesja:
                # Sprawdzenie czy istnieje aktualny status
                aktualny_status = sesja.query(StatusRynku).filter(
                    StatusRynku.symbol == symbol
                ).order_by(StatusRynku.timestamp.desc()).first()
                
                # Jeśli status się zmienił lub nie istnieje, dodaj nowy
                if not aktualny_status or \
                   aktualny_status.otwarty != status['otwarty'] or \
                   aktualny_status.powod != status.get('powod'):
                    nowy_status = StatusRynku(symbol=symbol, **status)
                    sesja.add(nowy_status)
                    sesja.commit()
                    return nowy_status
                    
                return aktualny_status
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd aktualizacji statusu rynku: {str(e)}")
            return None
            
    def aktualizuj_aktyw(self, nazwa: str, dane: Dict[str, Any]) -> Optional[Aktyw]:
        """
        Aktualizuje informacje o aktywie.
        
        Args:
            nazwa: Nazwa aktywa (np. "EURUSD")
            dane: Dane aktywa
            
        Returns:
            Optional[Aktyw]: Zaktualizowany aktyw lub None w przypadku błędu
        """
        try:
            with self.Session() as sesja:
                # Sprawdzenie czy aktyw istnieje
                aktyw = sesja.query(Aktyw).filter(
                    Aktyw.nazwa == nazwa
                ).order_by(Aktyw.timestamp.desc()).first()
                
                # Jeśli dane się zmieniły lub aktyw nie istnieje, dodaj nowy rekord
                if not aktyw or any(
                    getattr(aktyw, k) != v 
                    for k, v in dane.items() 
                    if hasattr(aktyw, k)
                ):
                    nowy_aktyw = Aktyw(nazwa=nazwa, **dane)
                    sesja.add(nowy_aktyw)
                    sesja.commit()
                    return nowy_aktyw
                    
                return aktyw
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd aktualizacji aktywa: {str(e)}")
            return None
            
    def dodaj_metryki(self, metryki: Dict[str, Any]) -> Optional[MetrykiHandlu]:
        """
        Dodaje nowe metryki handlowe do bazy.
        
        Args:
            metryki: Dane metryki
            
        Returns:
            Optional[MetrykiHandlu]: Utworzone metryki lub None w przypadku błędu
        """
        try:
            with self.Session() as sesja:
                nowe_metryki = MetrykiHandlu(**metryki)
                sesja.add(nowe_metryki)
                sesja.commit()
                return nowe_metryki
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd dodawania metryk: {str(e)}")
            return None
            
    def pobierz_historie_cen(
        self,
        symbol: str,
        timeframe: str,
        od: datetime,
        do: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[HistoriaCen]:
        """
        Pobiera historię cen dla danego symbolu i timeframe'u.
        
        Args:
            symbol: Symbol (np. "EURUSD")
            timeframe: Interwał czasowy
            od: Data początkowa
            do: Data końcowa (opcjonalna)
            limit: Maksymalna liczba rekordów (opcjonalna)
            
        Returns:
            List[HistoriaCen]: Lista rekordów historii cen
        """
        try:
            with self.Session() as sesja:
                query = sesja.query(HistoriaCen).filter(
                    and_(
                        HistoriaCen.symbol == symbol,
                        HistoriaCen.timeframe == timeframe,
                        HistoriaCen.timestamp >= od
                    )
                )
                
                if do:
                    query = query.filter(HistoriaCen.timestamp <= do)
                    
                query = query.order_by(HistoriaCen.timestamp.desc())
                
                if limit:
                    query = query.limit(limit)
                    
                return query.all()
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd pobierania historii cen: {str(e)}")
            return []
            
    def pobierz_wiadomosci(
        self,
        od: datetime,
        do: Optional[datetime] = None,
        kategoria: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Wiadomosc]:
        """
        Pobiera wiadomości z określonego zakresu dat.
        
        Args:
            od: Data początkowa
            do: Data końcowa (opcjonalna)
            kategoria: Kategoria wiadomości (opcjonalna)
            limit: Maksymalna liczba wiadomości (opcjonalna)
            
        Returns:
            List[Wiadomosc]: Lista wiadomości
        """
        try:
            with self.Session() as sesja:
                query = sesja.query(Wiadomosc).filter(Wiadomosc.timestamp >= od)
                
                if do:
                    query = query.filter(Wiadomosc.timestamp <= do)
                    
                if kategoria:
                    query = query.filter(Wiadomosc.kategoria == kategoria)
                    
                query = query.order_by(Wiadomosc.timestamp.desc())
                
                if limit:
                    query = query.limit(limit)
                    
                return query.all()
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd pobierania wiadomości: {str(e)}")
            return []
            
    def pobierz_metryki(
        self,
        symbol: str,
        od: datetime,
        do: Optional[datetime] = None
    ) -> List[MetrykiHandlu]:
        """
        Pobiera metryki handlowe dla danego symbolu.
        
        Args:
            symbol: Symbol (np. "EURUSD")
            od: Data początkowa
            do: Data końcowa (opcjonalna)
            
        Returns:
            List[MetrykiHandlu]: Lista metryk
        """
        try:
            with self.Session() as sesja:
                query = sesja.query(MetrykiHandlu).filter(
                    and_(
                        MetrykiHandlu.symbol == symbol,
                        MetrykiHandlu.timestamp >= od
                    )
                )
                
                if do:
                    query = query.filter(MetrykiHandlu.timestamp <= do)
                    
                return query.order_by(MetrykiHandlu.timestamp.desc()).all()
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Błąd pobierania metryk: {str(e)}")
            return [] 