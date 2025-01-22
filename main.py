"""
Główny moduł systemu NikkeiNinja.
"""

import asyncio
import logging
from typing import Dict, Any

from rdzen.polaczenie_mt5 import PolaczenieMT5
from narzedzia.logowanie import skonfiguruj_logger
from narzedzia.konfiguracja import wczytaj_config
from interfejs.dashboard import DashboardApp
from baza_danych.baza import BazaDanych

async def main():
    """Główna funkcja systemu."""
    try:
        # Konfiguracja
        print("🥷 Wczytuję konfigurację...")
        config = wczytaj_config()
        
        print("🥷 Konfiguruję logger...")
        logger = skonfiguruj_logger(
            "NikkeiNinja",
            poziom="INFO",
            plik_logow="logs/nikkeininja.log"
        )
        
        logger.info("🥷 Rozpoczynam inicjalizację systemu NikkeiNinja")
        
        # Inicjalizacja bazy danych
        logger.info("🥷 Łączę z bazą danych PostgreSQL...")
        baza = BazaDanych(config["baza_danych"])
        if not baza.inicjalizuj():
            logger.error("❌ Nie udało się zainicjalizować bazy danych")
            return
        logger.info("🥷 Połączono z bazą danych")
        
        # Inicjalizacja połączenia z MT5
        logger.info("🥷 Łączę z platformą MetaTrader 5...")
        mt5_connector = PolaczenieMT5()
        wynik = mt5_connector.inicjalizuj()
        
        if not wynik["status"]:
            logger.error(f"❌ Nie udało się połączyć z MT5: {wynik['blad']}")
            return
            
        logger.info("🥷 System NikkeiNinja uruchomiony pomyślnie")
        
        # Uruchomienie dashboardu
        logger.info("🥷 Uruchamiam dashboard na http://localhost:8050")
        dashboard = DashboardApp()
        dashboard.uruchom()
        
    except Exception as e:
        print(f"❌ Błąd krytyczny: {str(e)}")
        if 'logger' in locals():
            logger.error(f"❌ Błąd systemu: {str(e)}")
    finally:
        if 'mt5_connector' in locals():
            mt5_connector.zakoncz()

if __name__ == "__main__":
    print("🥷 Uruchamiam system NikkeiNinja...")
    asyncio.run(main()) 