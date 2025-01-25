"""
Daemon do synchronizacji danych z MT5 w tle.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import MetaTrader5 as mt5
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

from baza_danych.modele import Base
from baza_danych.synchronizacja import SynchronizatorMT5
from baza_danych.modele import ZadanieAktualizacji

# Konfiguracja logowania
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Ładowanie zmiennych środowiskowych
load_dotenv()

class DaemonSynchronizacji:
    """Daemon do synchronizacji danych w tle."""
    
    def __init__(self):
        """Inicjalizacja daemona."""
        self.logger = logger
        self.running = False
        self.engine = None
        self.Session = None
        
    async def inicjalizuj(self) -> bool:
        """
        Inicjalizacja połączeń z bazą danych i MT5.
        
        Returns:
            bool: True jeśli inicjalizacja się powiodła
        """
        try:
            # Połączenie z bazą danych
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                raise Exception("Brak URL bazy danych w zmiennych środowiskowych")
                
            self.engine = create_engine(db_url)
            self.Session = sessionmaker(bind=self.engine)
            
            # Inicjalizacja MT5
            if not mt5.initialize():
                raise Exception("Nie udało się zainicjalizować MT5")
                
            self.logger.info("🥷 Daemon zainicjalizowany")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd inicjalizacji: {str(e)}")
            return False
            
    async def dodaj_zadania_synchronizacji(
        self,
        symbol: str,
        timeframes: List[str] = ["M1", "M5", "H1"]
    ) -> None:
        """
        Dodaje zadania synchronizacji dla symbolu.
        
        Args:
            symbol: Symbol do synchronizacji
            timeframes: Lista timeframes do synchronizacji
        """
        try:
            session = self.Session()
            
            # Dodanie zadań dla historii cen
            for tf in timeframes:
                zadanie = ZadanieAktualizacji(
                    symbol=symbol,
                    timeframe=tf,
                    typ="historia_cen",
                    priorytet=1
                )
                session.add(zadanie)
            
            # Dodanie zadania dla statusu rynku
            zadanie_status = ZadanieAktualizacji(
                symbol=symbol,
                timeframe="",  # Nie dotyczy
                typ="status",
                priorytet=2
            )
            session.add(zadanie_status)
            
            session.commit()
            self.logger.info(f"🥷 Dodano zadania synchronizacji dla {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Błąd dodawania zadań: {str(e)}")
            
        finally:
            session.close()
            
    async def uruchom(self) -> None:
        """Uruchamia daemon synchronizacji."""
        try:
            if not await self.inicjalizuj():
                return
                
            self.running = True
            self.logger.info("🥷 Daemon uruchomiony")
            
            while self.running:
                try:
                    session = self.Session()
                    synchronizator = SynchronizatorMT5(session)
                    
                    # Przetworzenie zadań
                    await synchronizator.przetworz_zadania()
                    
                    # Dodanie nowych zadań dla JP225
                    await self.dodaj_zadania_synchronizacji("JP225")
                    
                    # Czekamy 5 minut przed następnym cyklem
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    self.logger.error(f"❌ Błąd w cyklu synchronizacji: {str(e)}")
                    await asyncio.sleep(60)  # Krótsza przerwa przy błędzie
                    
                finally:
                    session.close()
                    
        except Exception as e:
            self.logger.error(f"❌ Krytyczny błąd daemona: {str(e)}")
            self.running = False
            
    def zatrzymaj(self) -> None:
        """Zatrzymuje daemon synchronizacji."""
        self.running = False
        self.logger.info("🥷 Daemon zatrzymany")
        
async def main():
    """Główna funkcja uruchamiająca daemon."""
    daemon = DaemonSynchronizacji()
    
    try:
        await daemon.uruchom()
    except KeyboardInterrupt:
        daemon.zatrzymaj()
        
if __name__ == "__main__":
    asyncio.run(main()) 