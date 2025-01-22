"""
Moduł odpowiedzialny za połączenie z platformą MetaTrader 5.
Zapewnia podstawowe funkcje do komunikacji z terminalem MT5.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Konfiguracja loggera
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Flaga określająca czy MT5 jest dostępny
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ MetaTrader5 nie jest dostępny - działam w trybie testowym")

class PolaczenieMT5:
    """Klasa zarządzająca połączeniem z platformą MetaTrader 5."""
    
    def __init__(self):
        """Inicjalizacja obiektu połączenia."""
        self.polaczony = False
        self.handel_dozwolony = False
        self.sciezka_terminala = None
        self.tryb_testowy = not MT5_AVAILABLE
        load_dotenv()
        
    def inicjalizuj(self) -> Dict[str, Any]:
        """
        Inicjalizuje połączenie z platformą MetaTrader 5.
        W trybie testowym zwraca symulowane dane.
        
        Returns:
            Dict[str, Any]: Słownik zawierający informacje o statusie połączenia
        """
        wynik = {
            "status": False,
            "wersja": None,
            "polaczony": False,
            "handel_dozwolony": False,
            "sciezka_terminala": None,
            "blad": None,
            "tryb_testowy": self.tryb_testowy
        }
        
        if self.tryb_testowy:
            logger.info("🥷 Uruchamiam w trybie testowym")
            wynik.update({
                "status": True,
                "wersja": "TEST",
                "polaczony": True,
                "handel_dozwolony": True,
                "sciezka_terminala": "/test/path"
            })
            return wynik
            
        try:
            # Standardowa inicjalizacja MT5
            if not mt5.initialize():
                blad = mt5.last_error()
                logger.error(f"❌ Nie udało się zainicjalizować MT5: {blad}")
                wynik["blad"] = str(blad)
                return wynik
                
            # Logowanie do konta
            login = int(os.getenv('MT5_LOGIN'))
            haslo = os.getenv('MT5_PASSWORD')
            serwer = os.getenv('MT5_SERVER')
            
            if not mt5.login(login, haslo, serwer):
                blad = mt5.last_error()
                logger.error(f"❌ Nie udało się zalogować do MT5: {blad}")
                wynik["blad"] = f"Błąd logowania: {blad}"
                return wynik
                
            # Sprawdzenie wersji
            wersja = mt5.version()
            logger.info(f"🥷 MetaTrader5 wersja: {wersja}")
            wynik["wersja"] = wersja
            
            # Sprawdzenie informacji o terminalu
            info_terminal = mt5.terminal_info()
            if info_terminal is None:
                logger.error("❌ Nie udało się pobrać informacji o terminalu")
                wynik["blad"] = "Brak dostępu do informacji o terminalu"
                return wynik
                
            # Sprawdzenie informacji o koncie
            info_konto = mt5.account_info()
            if info_konto is None:
                logger.error("❌ Nie udało się pobrać informacji o koncie")
                wynik["blad"] = "Brak dostępu do informacji o koncie"
                return wynik
                
            logger.info(f"🥷 Zalogowano do konta: {info_konto.login} ({info_konto.server})")
            logger.info(f"🥷 Saldo: {info_konto.balance} {info_konto.currency}")
                
            # Aktualizacja statusu
            self.polaczony = info_terminal.connected
            self.handel_dozwolony = info_terminal.trade_allowed
            self.sciezka_terminala = info_terminal.path
            
            wynik.update({
                "status": True,
                "polaczony": self.polaczony,
                "handel_dozwolony": self.handel_dozwolony,
                "sciezka_terminala": self.sciezka_terminala
            })
            
            logger.info(f"🥷 Ścieżka terminala: {self.sciezka_terminala}")
            logger.info(f"🥷 Status połączenia: {self.polaczony}")
            logger.info(f"🥷 Handel dozwolony: {self.handel_dozwolony}")
            
            return wynik
            
        except Exception as e:
            logger.error(f"❌ Wystąpił nieoczekiwany błąd: {str(e)}")
            wynik["blad"] = str(e)
            return wynik
    
    def zakoncz(self) -> None:
        """Zamyka połączenie z platformą MT5."""
        if not self.tryb_testowy and MT5_AVAILABLE:
            mt5.shutdown()
        self.polaczony = False
        self.handel_dozwolony = False
        logger.info("🥷 Połączenie z MT5 zakończone")

if __name__ == "__main__":
    logger.info("🥷 Rozpoczynam test połączenia z MT5...")
    mt5_connector = PolaczenieMT5()
    wynik = mt5_connector.inicjalizuj()
    
    if wynik["status"]:
        logger.info("🥷 Test połączenia zakończony sukcesem!")
    else:
        logger.error(f"❌ Test połączenia zakończony niepowodzeniem: {wynik['blad']}")
    
    mt5_connector.zakoncz() 