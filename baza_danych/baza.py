"""
Moduł obsługujący połączenie z bazą danych.
"""

import logging
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from .modele import Base, Transakcja, Pozycja, HistoriaCen

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
            url = f"postgresql://{self.config['uzytkownik']}:{self.config['haslo']}@{self.config['host']}:{self.config['port']}/{self.config['nazwa']}"
            
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