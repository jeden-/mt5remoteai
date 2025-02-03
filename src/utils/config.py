"""
Moduł zawierający klasę konfiguracyjną.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

class Config:
    """Klasa przechowująca konfigurację aplikacji."""

    def __init__(self, **kwargs):
        """
        Inicjalizacja konfiguracji.

        Args:
            **kwargs: Parametry konfiguracyjne
        """
        # MT5 credentials
        self.MT5_LOGIN = kwargs.get('MT5_LOGIN') if 'MT5_LOGIN' in kwargs else int(os.getenv('MT5_LOGIN', 0))
        self.MT5_PASSWORD = kwargs.get('MT5_PASSWORD') if 'MT5_PASSWORD' in kwargs else os.getenv('MT5_PASSWORD', '')
        self.MT5_SERVER = kwargs.get('MT5_SERVER') if 'MT5_SERVER' in kwargs else os.getenv('MT5_SERVER', '')

        # Database credentials
        self.DB_HOST = kwargs.get('DB_HOST') if 'DB_HOST' in kwargs else os.getenv('DB_HOST', 'localhost')
        self.DB_PORT = kwargs.get('DB_PORT') if 'DB_PORT' in kwargs else int(os.getenv('DB_PORT', 5432))
        self.DB_NAME = kwargs.get('DB_NAME') if 'DB_NAME' in kwargs else os.getenv('DB_NAME', 'mt5remotetest')
        self.DB_USER = kwargs.get('DB_USER') if 'DB_USER' in kwargs else os.getenv('DB_USER', 'mt5remote')
        self.DB_PASSWORD = kwargs.get('DB_PASSWORD') if 'DB_PASSWORD' in kwargs else os.getenv('DB_PASSWORD', '')

        # Trading parameters
        self.SYMBOL = kwargs.get('SYMBOL') if 'SYMBOL' in kwargs else os.getenv('SYMBOL', 'EURUSD')
        self.TIMEFRAME = kwargs.get('TIMEFRAME') if 'TIMEFRAME' in kwargs else os.getenv('TIMEFRAME', 'H1')
        self.MAX_POSITION_SIZE = float(kwargs.get('MAX_POSITION_SIZE') if 'MAX_POSITION_SIZE' in kwargs else os.getenv('MAX_POSITION_SIZE', 1.0))
        self.STOP_LOSS_PIPS = float(kwargs.get('STOP_LOSS_PIPS') if 'STOP_LOSS_PIPS' in kwargs else os.getenv('STOP_LOSS_PIPS', 50))
        self.TAKE_PROFIT_PIPS = float(kwargs.get('TAKE_PROFIT_PIPS') if 'TAKE_PROFIT_PIPS' in kwargs else os.getenv('TAKE_PROFIT_PIPS', 100))

        # Strategy parameters
        self.RSI_PERIOD = int(kwargs.get('RSI_PERIOD') if 'RSI_PERIOD' in kwargs else os.getenv('RSI_PERIOD', 14))
        self.RSI_OVERSOLD = float(kwargs.get('RSI_OVERSOLD') if 'RSI_OVERSOLD' in kwargs else os.getenv('RSI_OVERSOLD', 30))
        self.RSI_OVERBOUGHT = float(kwargs.get('RSI_OVERBOUGHT') if 'RSI_OVERBOUGHT' in kwargs else os.getenv('RSI_OVERBOUGHT', 70))
        self.SMA_FAST = int(kwargs.get('SMA_FAST') if 'SMA_FAST' in kwargs else os.getenv('SMA_FAST', 20))
        self.SMA_SLOW = int(kwargs.get('SMA_SLOW') if 'SMA_SLOW' in kwargs else os.getenv('SMA_SLOW', 50))

        # Logging parameters
        self.LOG_LEVEL = kwargs.get('LOG_LEVEL') if 'LOG_LEVEL' in kwargs else os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = kwargs.get('LOG_FILE') if 'LOG_FILE' in kwargs else os.getenv('LOG_FILE', 'trading.log')
        self.LOG_FORMAT = kwargs.get('LOG_FORMAT') if 'LOG_FORMAT' in kwargs else os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s')

    def validate(self) -> None:
        """
        Waliduje konfigurację.

        Raises:
            ValueError: Gdy konfiguracja jest nieprawidłowa
        """
        # Walidacja MT5
        if not self.MT5_LOGIN or not self.MT5_PASSWORD or not self.MT5_SERVER:
            raise ValueError("Brak wymaganych danych dostępowych do MT5")

        # Walidacja bazy danych
        if not self.DB_HOST or not self.DB_PORT or not self.DB_NAME or not self.DB_USER or not self.DB_PASSWORD:
            raise ValueError("Brak wymaganych danych dostępowych do bazy danych")

        # Walidacja parametrów tradingowych
        if self.MAX_POSITION_SIZE <= 0:
            raise ValueError("MAX_POSITION_SIZE musi być większe od 0")
        if self.STOP_LOSS_PIPS <= 0:
            raise ValueError("STOP_LOSS_PIPS musi być większe od 0")
        if self.TAKE_PROFIT_PIPS <= 0:
            raise ValueError("TAKE_PROFIT_PIPS musi być większe od 0")

        # Walidacja parametrów strategii
        if self.RSI_OVERSOLD >= 50:
            raise ValueError("RSI_OVERSOLD musi być mniejsze od 50")
        if self.RSI_OVERBOUGHT <= 50:
            raise ValueError("RSI_OVERBOUGHT musi być większe od 50")
        if self.SMA_FAST >= self.SMA_SLOW:
            raise ValueError("SMA_FAST musi być mniejsze od SMA_SLOW")

        # Walidacja logowania
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL not in valid_levels:
            raise ValueError(f"Nieprawidłowy poziom logowania. Dozwolone: {', '.join(valid_levels)}")

class ConfigLoader:
    """Klasa do ładowania i walidacji konfiguracji."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Ładuje konfigurację z pliku JSON lub YAML.

        Args:
            config_path: Ścieżka do pliku konfiguracyjnego

        Returns:
            Dict zawierający konfigurację

        Raises:
            FileNotFoundError: Gdy plik nie istnieje
            ValueError: Gdy format pliku jest nieobsługiwany
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError("Nie znaleziono pliku konfiguracyjnego")

        file_ext = Path(config_path).suffix.lower()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if file_ext == '.json':
                    return json.load(f)
                elif file_ext in ['.yml', '.yaml']:
                    return yaml.safe_load(f)
                else:
                    raise ValueError("Nieobsługiwany format pliku")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Błąd parsowania pliku: {str(e)}")

    @staticmethod
    def validate_trading_config(config: Dict[str, Any]) -> None:
        """
        Waliduje konfigurację tradingową.

        Args:
            config: Słownik z konfiguracją tradingową

        Raises:
            ValueError: Gdy konfiguracja jest nieprawidłowa
        """
        required_fields = ['symbol', 'timeframe', 'max_position_size', 'stop_loss_pips', 'take_profit_pips']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Brak wymaganego pola: {field}")

        if config['max_position_size'] <= 0:
            raise ValueError("Nieprawidłowa wartość: max_position_size musi być większe od 0")
        if config['stop_loss_pips'] <= 0:
            raise ValueError("Nieprawidłowa wartość: stop_loss_pips musi być większe od 0")
        if config['take_profit_pips'] <= 0:
            raise ValueError("Nieprawidłowa wartość: take_profit_pips musi być większe od 0")

    @staticmethod
    def validate_strategy_config(config: Dict[str, Any]) -> None:
        """
        Waliduje konfigurację strategii.

        Args:
            config: Słownik z konfiguracją strategii

        Raises:
            ValueError: Gdy konfiguracja jest nieprawidłowa
        """
        required_fields = ['rsi_period', 'rsi_oversold', 'rsi_overbought', 'sma_fast', 'sma_slow']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Brak wymaganego pola: {field}")

        if config['rsi_oversold'] >= 50:
            raise ValueError("Nieprawidłowa wartość: rsi_oversold musi być mniejsze od 50")
        if config['rsi_overbought'] <= 50:
            raise ValueError("Nieprawidłowa wartość: rsi_overbought musi być większe od 50")
        if config['sma_fast'] >= config['sma_slow']:
            raise ValueError("Nieprawidłowa wartość: sma_fast musi być mniejsze od sma_slow")

    @staticmethod
    def validate_database_config(config: Dict[str, Any]) -> None:
        """
        Waliduje konfigurację bazy danych.

        Args:
            config: Słownik z konfiguracją bazy danych

        Raises:
            ValueError: Gdy konfiguracja jest nieprawidłowa
        """
        required_fields = ['host', 'port', 'name', 'user', 'password']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Brak wymaganego pola: {field}")

        try:
            port = int(config['port'])
            if port <= 0 or port > 65535:
                raise ValueError
        except ValueError:
            raise ValueError("Nieprawidłowa wartość: port musi być liczbą z zakresu 1-65535")

    @staticmethod
    def validate_logging_config(config: Dict[str, Any]) -> None:
        """
        Waliduje konfigurację logowania.

        Args:
            config: Słownik z konfiguracją logowania

        Raises:
            ValueError: Gdy konfiguracja jest nieprawidłowa
        """
        required_fields = ['level', 'file', 'format']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Brak wymaganego pola: {field}")

        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config['level'] not in valid_levels:
            raise ValueError(f"Nieprawidłowy poziom logowania. Dozwolone: {', '.join(valid_levels)}")

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Łączy dwie konfiguracje, nadpisując wartości z base_config wartościami z override_config.

        Args:
            base_config: Podstawowa konfiguracja
            override_config: Konfiguracja nadpisująca

        Returns:
            Dict zawierający połączoną konfigurację
        """
        merged = base_config.copy()
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def load_env_variables(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nadpisuje wartości w konfiguracji zmiennymi środowiskowymi.

        Args:
            config: Słownik z konfiguracją

        Returns:
            Dict zawierający zaktualizowaną konfigurację
        """
        env_mapping = {
            'TRADING_SYMBOL': ('trading', 'symbol'),
            'DB_PASSWORD': ('database', 'password'),
            'LOG_LEVEL': ('logging', 'level')
        }

        for env_var, (section, key) in env_mapping.items():
            if env_var in os.environ:
                if section not in config:
                    config[section] = {}
                config[section][key] = os.environ[env_var]

        return config

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        Zapisuje konfigurację do pliku.

        Args:
            config: Słownik z konfiguracją
            config_path: Ścieżka do pliku

        Raises:
            ValueError: Gdy format pliku jest nieobsługiwany lub wystąpił błąd zapisu
        """
        file_ext = Path(config_path).suffix.lower()
        
        try:
            # Upewnij się, że katalog istnieje
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if file_ext == '.json':
                    try:
                        json.dump(config, f, indent=4)
                    except (TypeError, ValueError) as e:
                        raise ValueError(f"Błąd zapisu pliku: {str(e)}")
                elif file_ext in ['.yml', '.yaml']:
                    try:
                        yaml.dump(config, f, default_flow_style=False)
                    except (TypeError, ValueError) as e:
                        raise ValueError(f"Błąd zapisu pliku: {str(e)}")
                else:
                    raise ValueError("Nieobsługiwany format pliku")
        except (OSError, IOError) as e:
            raise ValueError(f"Błąd zapisu pliku: {str(e)}")

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Zwraca domyślną konfigurację.

        Returns:
            Dict zawierający domyślną konfigurację
        """
        return {
            "trading": {
                "symbol": "EURUSD",
                "timeframe": "1H",
                "max_position_size": 1.0,
                "stop_loss_pips": 50,
                "take_profit_pips": 100
            },
            "strategy": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "sma_fast": 20,
                "sma_slow": 50
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_db",
                "user": "trader",
                "password": "default_password"
            },
            "logging": {
                "level": "INFO",
                "file": "trading.log",
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            }
        }

    @staticmethod
    def validate_config_dependencies(config: Dict[str, Any]) -> None:
        """
        Waliduje zależności między sekcjami konfiguracji.

        Args:
            config: Słownik z konfiguracją

        Raises:
            ValueError: Gdy zależności są nieprawidłowe
        """
        # Sprawdź czy wszystkie wymagane sekcje są obecne
        required_sections = ['trading', 'strategy', 'database', 'logging']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Brak wymaganej sekcji: {section}")

        # Sprawdź zależności między stop loss i take profit
        trading = config['trading']
        if trading['stop_loss_pips'] >= trading['take_profit_pips']:
            raise ValueError("Nieprawidłowe zależności: take_profit_pips musi być większe od stop_loss_pips")

        # Sprawdź zależności w strategii
        strategy = config['strategy']
        if strategy['sma_fast'] >= strategy['sma_slow']:
            raise ValueError("Nieprawidłowe zależności: sma_fast musi być mniejsze od sma_slow")
        
        if strategy['rsi_oversold'] >= strategy['rsi_overbought']:
            raise ValueError("Nieprawidłowe zależności: rsi_oversold musi być mniejsze od rsi_overbought")
        
        # Sprawdź minimalną różnicę między poziomami RSI
        if (strategy['rsi_overbought'] - strategy['rsi_oversold']) < 40:
            raise ValueError("Nieprawidłowe zależności: Różnica między rsi_overbought i rsi_oversold musi być co najmniej 40") 