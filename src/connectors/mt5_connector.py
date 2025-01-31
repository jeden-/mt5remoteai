"""
Moduł zawierający klasę do komunikacji z platformą MetaTrader 5.
"""
import MetaTrader5 as mt5
from typing import Optional, List, Dict, Any
from loguru import logger

from ..utils.config import Config


class MT5Connector:
    """Klasa odpowiedzialna za komunikację z platformą MetaTrader 5."""

    def __init__(self, config: Config):
        """
        Inicjalizacja połączenia z MT5.

        Args:
            config: Obiekt konfiguracyjny z parametrami połączenia
        """
        self.config = config
        self.connected = False
        
    def connect(self) -> bool:
        """
        Nawiązuje połączenie z MT5.
        
        Returns:
            bool: True jeśli połączenie zostało nawiązane pomyślnie, False w przeciwnym razie
        """
        if not mt5.initialize():
            logger.error("❌ Inicjalizacja MT5 nie powiodła się")
            return False
            
        authorized = mt5.login(
            login=self.config.MT5_LOGIN,
            password=self.config.MT5_PASSWORD,
            server=self.config.MT5_SERVER
        )
        
        if authorized:
            logger.info("🥷 Połączono z MT5")
            self.connected = True
            return True
        else:
            logger.error("❌ Autoryzacja MT5 nie powiodła się")
            return False
            
    def disconnect(self) -> None:
        """Zamyka połączenie z MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("🥷 Rozłączono z MT5")
            
    def get_account_info(self) -> Dict[str, Any]:
        """
        Pobiera informacje o koncie.
        
        Returns:
            Dict[str, Any]: Słownik z informacjami o koncie
            
        Raises:
            RuntimeError: Gdy brak połączenia lub nie można pobrać informacji
        """
        if not self.connected:
            logger.error("❌ Brak połączenia z MT5")
            raise RuntimeError("Brak połączenia z MT5")
            
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("❌ Nie można pobrać informacji o koncie")
            raise RuntimeError("Nie można pobrać informacji o koncie")
            
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'margin_level': account_info.margin_level,
        }
        
    def get_symbols(self) -> List[str]:
        """
        Pobiera listę dostępnych instrumentów.
        
        Returns:
            List[str]: Lista dostępnych symboli
            
        Raises:
            RuntimeError: Gdy brak połączenia z MT5
        """
        if not self.connected:
            logger.error("❌ Brak połączenia z MT5")
            raise RuntimeError("Brak połączenia z MT5")
            
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error("❌ Nie można pobrać listy symboli")
            raise RuntimeError("Nie można pobrać listy symboli")
            
        return [symbol.name for symbol in symbols] 