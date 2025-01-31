"""
Moduł zawierający klasy konfiguracyjne.
"""
import os
import json
from typing import Dict, Any


class Config:
    """Klasa przechowująca konfigurację aplikacji."""
    
    def __init__(
        self,
        MT5_LOGIN: str = '',
        MT5_PASSWORD: str = '',
        MT5_SERVER: str = '',
        ANTHROPIC_API_KEY: str = '',
        POSTGRES_HOST: str = 'localhost',
        POSTGRES_PORT: str = '5432',
        POSTGRES_DB: str = 'trading_db',
        POSTGRES_USER: str = '',
        POSTGRES_PASSWORD: str = '',
        DEBUG: bool = False
    ):
        """
        Inicjalizacja konfiguracji.
        
        Args:
            MT5_LOGIN: Login do MT5
            MT5_PASSWORD: Hasło do MT5
            MT5_SERVER: Serwer MT5
            ANTHROPIC_API_KEY: Klucz API do Anthropic
            POSTGRES_HOST: Host bazy danych
            POSTGRES_PORT: Port bazy danych
            POSTGRES_DB: Nazwa bazy danych
            POSTGRES_USER: Użytkownik bazy danych
            POSTGRES_PASSWORD: Hasło do bazy danych
            DEBUG: Tryb debug
            
        Raises:
            ValueError: Gdy podano nieprawidłowe wartości
        """
        # Walidacja wartości
        if not isinstance(MT5_LOGIN, str):
            raise ValueError("MT5_LOGIN musi być typu str")
        if not isinstance(MT5_PASSWORD, str):
            raise ValueError("MT5_PASSWORD musi być typu str")
        if not isinstance(MT5_SERVER, str):
            raise ValueError("MT5_SERVER musi być typu str")
        if not isinstance(ANTHROPIC_API_KEY, str):
            raise ValueError("ANTHROPIC_API_KEY musi być typu str")
        if not isinstance(POSTGRES_HOST, str):
            raise ValueError("POSTGRES_HOST musi być typu str")
        if not isinstance(POSTGRES_PORT, str):
            raise ValueError("POSTGRES_PORT musi być typu str")
        if not isinstance(POSTGRES_DB, str):
            raise ValueError("POSTGRES_DB musi być typu str")
        if not isinstance(POSTGRES_USER, str):
            raise ValueError("POSTGRES_USER musi być typu str")
        if not isinstance(POSTGRES_PASSWORD, str):
            raise ValueError("POSTGRES_PASSWORD musi być typu str")
        if not isinstance(DEBUG, bool):
            raise ValueError("DEBUG musi być typu bool")
            
        self.MT5_LOGIN = MT5_LOGIN
        self.MT5_PASSWORD = MT5_PASSWORD
        self.MT5_SERVER = MT5_SERVER
        self.ANTHROPIC_API_KEY = ANTHROPIC_API_KEY
        self.POSTGRES_HOST = POSTGRES_HOST
        self.POSTGRES_PORT = POSTGRES_PORT
        self.POSTGRES_DB = POSTGRES_DB
        self.POSTGRES_USER = POSTGRES_USER
        self.POSTGRES_PASSWORD = POSTGRES_PASSWORD
        self.DEBUG = DEBUG
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Tworzy konfigurację ze słownika.
        
        Args:
            config_dict: Słownik z konfiguracją
            
        Returns:
            Config: Obiekt konfiguracji
            
        Raises:
            ValueError: Gdy brakuje wymaganych pól lub podano nieprawidłowe wartości
        """
        required_fields = [
            'MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER',
            'ANTHROPIC_API_KEY', 'POSTGRES_USER', 'POSTGRES_PASSWORD'
        ]
        
        # Sprawdź czy są wszystkie wymagane pola
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Brak wymaganego pola: {field}")
            if not config_dict[field]:
                raise ValueError(f"Pole {field} nie może być puste")
                
        return cls(**config_dict)
        
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Tworzy konfigurację ze zmiennych środowiskowych.
        
        Returns:
            Config: Obiekt konfiguracji
            
        Raises:
            ValueError: Gdy brakuje wymaganych zmiennych środowiskowych
        """
        config_dict = {
            'MT5_LOGIN': os.getenv('MT5_LOGIN', ''),
            'MT5_PASSWORD': os.getenv('MT5_PASSWORD', ''),
            'MT5_SERVER': os.getenv('MT5_SERVER', ''),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY', ''),
            'POSTGRES_HOST': os.getenv('POSTGRES_HOST', 'localhost'),
            'POSTGRES_PORT': os.getenv('POSTGRES_PORT', '5432'),
            'POSTGRES_DB': os.getenv('POSTGRES_DB', 'trading_db'),
            'POSTGRES_USER': os.getenv('POSTGRES_USER', ''),
            'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD', ''),
            'DEBUG': os.getenv('DEBUG', 'false').lower() == 'true'
        }
        
        # Sprawdź czy są wszystkie wymagane zmienne
        required_vars = [
            'MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER',
            'ANTHROPIC_API_KEY', 'POSTGRES_USER', 'POSTGRES_PASSWORD'
        ]
        
        for var in required_vars:
            if not config_dict[var]:
                raise ValueError(f"Brak wymaganej zmiennej środowiskowej: {var}")
                
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Konwertuje konfigurację do słownika.
        
        Returns:
            Dict[str, Any]: Słownik z konfiguracją
        """
        return {
            'MT5_LOGIN': self.MT5_LOGIN,
            'MT5_PASSWORD': self.MT5_PASSWORD,
            'MT5_SERVER': self.MT5_SERVER,
            'ANTHROPIC_API_KEY': self.ANTHROPIC_API_KEY,
            'POSTGRES_HOST': self.POSTGRES_HOST,
            'POSTGRES_PORT': self.POSTGRES_PORT,
            'POSTGRES_DB': self.POSTGRES_DB,
            'POSTGRES_USER': self.POSTGRES_USER,
            'POSTGRES_PASSWORD': self.POSTGRES_PASSWORD,
            'DEBUG': self.DEBUG
        }
        
    @classmethod
    def load_config(cls, config_path: str = 'config.json') -> 'Config':
        """
        Wczytuje konfigurację z pliku JSON.
        
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego
            
        Returns:
            Config: Obiekt konfiguracji
            
        Raises:
            FileNotFoundError: Gdy plik nie istnieje
            ValueError: Gdy plik zawiera nieprawidłowe dane
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            return cls.from_env()
        except json.JSONDecodeError as e:
            raise ValueError(f"Nieprawidłowy format pliku JSON: {str(e)}")
            
    def save_config(self, config_path: str = 'config.json'):
        """
        Zapisuje konfigurację do pliku JSON.
        
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego
            
        Raises:
            IOError: Gdy nie można zapisać pliku
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4)
        except IOError as e:
            raise IOError(f"Nie można zapisać pliku konfiguracyjnego: {str(e)}")
            
    def validate(self) -> bool:
        """
        Sprawdza poprawność konfiguracji.
        
        Returns:
            bool: True jeśli konfiguracja jest poprawna
            
        Raises:
            ValueError: Gdy konfiguracja jest nieprawidłowa
        """
        # Sprawdź wymagane pola MT5
        if not self.MT5_LOGIN:
            raise ValueError("Brak wymaganego pola: MT5_LOGIN")
        if not self.MT5_PASSWORD:
            raise ValueError("Brak wymaganego pola: MT5_PASSWORD")
        if not self.MT5_SERVER:
            raise ValueError("Brak wymaganego pola: MT5_SERVER")
            
        # Sprawdź wymagane pola bazy danych
        if not self.POSTGRES_USER:
            raise ValueError("Brak wymaganego pola: POSTGRES_USER")
        if not self.POSTGRES_PASSWORD:
            raise ValueError("Brak wymaganego pola: POSTGRES_PASSWORD")
            
        # Sprawdź klucz API
        if not self.ANTHROPIC_API_KEY:
            raise ValueError("Brak wymaganego pola: ANTHROPIC_API_KEY")
            
        return True
        
    def update(self, config_dict: Dict[str, Any]):
        """
        Aktualizuje konfigurację.
        
        Args:
            config_dict: Słownik z nowymi wartościami
            
        Raises:
            ValueError: Gdy podano nieprawidłowe wartości
        """
        # Sprawdź typy danych
        for key, value in config_dict.items():
            if hasattr(self, key):
                if key == 'DEBUG' and not isinstance(value, bool):
                    raise ValueError("DEBUG musi być typu bool")
                elif key != 'DEBUG' and not isinstance(value, str):
                    raise ValueError(f"{key} musi być typu str")
                    
        # Aktualizuj pola
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value) 