"""
Testy dla modułu config.py
"""
import os
import pytest
from pathlib import Path
from src.utils.config import Config, ConfigLoader
import json
import yaml

def test_config_initialization(monkeypatch):
    """Test inicjalizacji konfiguracji z domyślnymi wartościami."""
    # Czyszczenie zmiennych środowiskowych
    monkeypatch.delenv('MT5_LOGIN', raising=False)
    monkeypatch.delenv('MT5_PASSWORD', raising=False)
    monkeypatch.delenv('MT5_SERVER', raising=False)
    monkeypatch.delenv('DB_HOST', raising=False)
    monkeypatch.delenv('DB_PORT', raising=False)
    monkeypatch.delenv('DB_NAME', raising=False)
    monkeypatch.delenv('DB_USER', raising=False)
    monkeypatch.delenv('DB_PASSWORD', raising=False)
    monkeypatch.delenv('SYMBOL', raising=False)
    monkeypatch.delenv('TIMEFRAME', raising=False)
    monkeypatch.delenv('MAX_POSITION_SIZE', raising=False)
    monkeypatch.delenv('STOP_LOSS_PIPS', raising=False)
    monkeypatch.delenv('TAKE_PROFIT_PIPS', raising=False)
    monkeypatch.delenv('RSI_PERIOD', raising=False)
    monkeypatch.delenv('RSI_OVERSOLD', raising=False)
    monkeypatch.delenv('RSI_OVERBOUGHT', raising=False)
    monkeypatch.delenv('SMA_FAST', raising=False)
    monkeypatch.delenv('SMA_SLOW', raising=False)
    monkeypatch.delenv('LOG_LEVEL', raising=False)
    monkeypatch.delenv('LOG_FILE', raising=False)
    monkeypatch.delenv('LOG_FORMAT', raising=False)

    config = Config()
    assert config.MT5_LOGIN == 0
    assert config.MT5_PASSWORD == ''
    assert config.MT5_SERVER == ''
    assert config.DB_HOST == 'localhost'
    assert config.DB_PORT == 5432
    assert config.DB_NAME == 'mt5remotetest'
    assert config.DB_USER == 'mt5remote'
    assert config.DB_PASSWORD == ''
    assert config.SYMBOL == 'EURUSD'
    assert config.TIMEFRAME == 'H1'
    assert config.MAX_POSITION_SIZE == 1.0
    assert config.STOP_LOSS_PIPS == 50
    assert config.TAKE_PROFIT_PIPS == 100
    assert config.RSI_PERIOD == 14
    assert config.RSI_OVERSOLD == 30
    assert config.RSI_OVERBOUGHT == 70
    assert config.SMA_FAST == 20
    assert config.SMA_SLOW == 50
    assert config.LOG_LEVEL == 'INFO'
    assert config.LOG_FILE == 'trading.log'
    assert config.LOG_FORMAT == '%(asctime)s - %(levelname)s - %(message)s'

def test_config_initialization_with_kwargs():
    """Test inicjalizacji konfiguracji z przekazanymi parametrami."""
    config = Config(
        MT5_LOGIN=12345,
        MT5_PASSWORD='test_pass',
        MT5_SERVER='test_server',
        DB_HOST='test_host',
        DB_PORT=5433,
        DB_NAME='test_db',
        DB_USER='test_user',
        DB_PASSWORD='test_db_pass',
        SYMBOL='GBPUSD',
        TIMEFRAME='M15',
        MAX_POSITION_SIZE=0.5,
        STOP_LOSS_PIPS=30,
        TAKE_PROFIT_PIPS=60,
        RSI_PERIOD=21,
        RSI_OVERSOLD=25,
        RSI_OVERBOUGHT=75,
        SMA_FAST=10,
        SMA_SLOW=30,
        LOG_LEVEL='DEBUG',
        LOG_FILE='test.log',
        LOG_FORMAT='%(message)s'
    )
    assert config.MT5_LOGIN == 12345
    assert config.MT5_PASSWORD == 'test_pass'
    assert config.MT5_SERVER == 'test_server'
    assert config.DB_HOST == 'test_host'
    assert config.DB_PORT == 5433
    assert config.DB_NAME == 'test_db'
    assert config.DB_USER == 'test_user'
    assert config.DB_PASSWORD == 'test_db_pass'
    assert config.SYMBOL == 'GBPUSD'
    assert config.TIMEFRAME == 'M15'
    assert config.MAX_POSITION_SIZE == 0.5
    assert config.STOP_LOSS_PIPS == 30
    assert config.TAKE_PROFIT_PIPS == 60
    assert config.RSI_PERIOD == 21
    assert config.RSI_OVERSOLD == 25
    assert config.RSI_OVERBOUGHT == 75
    assert config.SMA_FAST == 10
    assert config.SMA_SLOW == 30
    assert config.LOG_LEVEL == 'DEBUG'
    assert config.LOG_FILE == 'test.log'
    assert config.LOG_FORMAT == '%(message)s'

def test_config_validation_mt5_credentials(monkeypatch):
    """Test walidacji danych dostępowych do MT5."""
    # Czyszczenie zmiennych środowiskowych
    monkeypatch.delenv('MT5_LOGIN', raising=False)
    monkeypatch.delenv('MT5_PASSWORD', raising=False)
    monkeypatch.delenv('MT5_SERVER', raising=False)
    
    config = Config()
    with pytest.raises(ValueError, match="Brak wymaganych danych dostępowych do MT5"):
        config.validate()

def test_config_validation_db_credentials(monkeypatch):
    """Test walidacji danych dostępowych do bazy danych."""
    # Czyszczenie zmiennych środowiskowych
    monkeypatch.delenv('DB_HOST', raising=False)
    monkeypatch.delenv('DB_PORT', raising=False)
    monkeypatch.delenv('DB_NAME', raising=False)
    monkeypatch.delenv('DB_USER', raising=False)
    monkeypatch.delenv('DB_PASSWORD', raising=False)
    
    config = Config(
        MT5_LOGIN=12345,
        MT5_PASSWORD='test',
        MT5_SERVER='test'
    )
    with pytest.raises(ValueError, match="Brak wymaganych danych dostępowych do bazy danych"):
        config.validate()

def test_config_validation_trading_params(monkeypatch):
    """Test walidacji parametrów tradingowych."""
    # Czyszczenie wszystkich zmiennych środowiskowych
    monkeypatch.delenv('MT5_LOGIN', raising=False)
    monkeypatch.delenv('MT5_PASSWORD', raising=False)
    monkeypatch.delenv('MT5_SERVER', raising=False)
    monkeypatch.delenv('DB_HOST', raising=False)
    monkeypatch.delenv('DB_PORT', raising=False)
    monkeypatch.delenv('DB_NAME', raising=False)
    monkeypatch.delenv('DB_USER', raising=False)
    monkeypatch.delenv('DB_PASSWORD', raising=False)
    monkeypatch.delenv('SYMBOL', raising=False)
    monkeypatch.delenv('TIMEFRAME', raising=False)
    monkeypatch.delenv('MAX_POSITION_SIZE', raising=False)
    monkeypatch.delenv('STOP_LOSS_PIPS', raising=False)
    monkeypatch.delenv('TAKE_PROFIT_PIPS', raising=False)
    monkeypatch.delenv('RSI_PERIOD', raising=False)
    monkeypatch.delenv('RSI_OVERSOLD', raising=False)
    monkeypatch.delenv('RSI_OVERBOUGHT', raising=False)
    monkeypatch.delenv('SMA_FAST', raising=False)
    monkeypatch.delenv('SMA_SLOW', raising=False)
    monkeypatch.delenv('LOG_LEVEL', raising=False)
    monkeypatch.delenv('LOG_FILE', raising=False)
    monkeypatch.delenv('LOG_FORMAT', raising=False)
    
    config = Config(
        MT5_LOGIN=12345,
        MT5_PASSWORD='test',
        MT5_SERVER='test',
        DB_HOST='test',
        DB_PORT=5432,
        DB_NAME='test',
        DB_USER='test',
        DB_PASSWORD='test',
        MAX_POSITION_SIZE=0
    )
    with pytest.raises(ValueError, match="MAX_POSITION_SIZE musi być większe od 0"):
        config.validate()

    config.MAX_POSITION_SIZE = 1.0
    config.STOP_LOSS_PIPS = 0
    with pytest.raises(ValueError, match="STOP_LOSS_PIPS musi być większe od 0"):
        config.validate()

    config.STOP_LOSS_PIPS = 50
    config.TAKE_PROFIT_PIPS = 0
    with pytest.raises(ValueError, match="TAKE_PROFIT_PIPS musi być większe od 0"):
        config.validate()

def test_config_validation_strategy_params():
    """Test walidacji parametrów strategii."""
    config = Config(
        MT5_LOGIN=12345,
        MT5_PASSWORD='test',
        MT5_SERVER='test',
        DB_HOST='test',
        DB_PORT=5432,
        DB_NAME='test',
        DB_USER='test',
        DB_PASSWORD='test',
        RSI_OVERSOLD=50
    )
    with pytest.raises(ValueError, match="RSI_OVERSOLD musi być mniejsze od 50"):
        config.validate()

    config.RSI_OVERSOLD = 30
    config.RSI_OVERBOUGHT = 50
    with pytest.raises(ValueError, match="RSI_OVERBOUGHT musi być większe od 50"):
        config.validate()

    config.RSI_OVERBOUGHT = 70
    config.SMA_FAST = 50
    config.SMA_SLOW = 50
    with pytest.raises(ValueError, match="SMA_FAST musi być mniejsze od SMA_SLOW"):
        config.validate()

def test_config_validation_log_level():
    """Test walidacji poziomu logowania."""
    config = Config(
        MT5_LOGIN=12345,
        MT5_PASSWORD='test',
        MT5_SERVER='test',
        DB_HOST='test',
        DB_PORT=5432,
        DB_NAME='test',
        DB_USER='test',
        DB_PASSWORD='test',
        LOG_LEVEL='INVALID'
    )
    with pytest.raises(ValueError, match="Nieprawidłowy poziom logowania"):
        config.validate()

def test_config_loader_load_config(tmp_path):
    """Test ładowania konfiguracji z pliku."""
    # Test JSON
    json_config = {
        'test_key': 'test_value'
    }
    json_path = tmp_path / 'config.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write('{"test_key": "test_value"}')
    
    loaded_json = ConfigLoader.load_config(str(json_path))
    assert loaded_json == json_config

    # Test YAML
    yaml_path = tmp_path / 'config.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write('test_key: test_value')
    
    loaded_yaml = ConfigLoader.load_config(str(yaml_path))
    assert loaded_yaml == json_config

    # Test nieistniejący plik
    with pytest.raises(FileNotFoundError):
        ConfigLoader.load_config('nonexistent.json')

    # Test nieprawidłowy format
    invalid_path = tmp_path / 'config.txt'
    invalid_path.touch()
    with pytest.raises(ValueError, match="Nieobsługiwany format pliku"):
        ConfigLoader.load_config(str(invalid_path))

def test_config_loader_validate_trading_config():
    """Test walidacji konfiguracji tradingowej."""
    valid_config = {
        'symbol': 'EURUSD',
        'timeframe': 'H1',
        'max_position_size': 1.0,
        'stop_loss_pips': 50,
        'take_profit_pips': 100
    }
    ConfigLoader.validate_trading_config(valid_config)

    # Test brakującego pola
    invalid_config = valid_config.copy()
    del invalid_config['symbol']
    with pytest.raises(ValueError, match="Brak wymaganego pola: symbol"):
        ConfigLoader.validate_trading_config(invalid_config)

    # Test nieprawidłowych wartości
    invalid_config = valid_config.copy()
    invalid_config['max_position_size'] = 0
    with pytest.raises(ValueError, match="Nieprawidłowa wartość: max_position_size musi być większe od 0"):
        ConfigLoader.validate_trading_config(invalid_config)

def test_config_loader_validate_strategy_config():
    """Test walidacji konfiguracji strategii."""
    valid_config = {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sma_fast': 20,
        'sma_slow': 50
    }
    ConfigLoader.validate_strategy_config(valid_config)

    # Test brakującego pola
    invalid_config = valid_config.copy()
    del invalid_config['rsi_period']
    with pytest.raises(ValueError, match="Brak wymaganego pola: rsi_period"):
        ConfigLoader.validate_strategy_config(invalid_config)

    # Test nieprawidłowych wartości
    invalid_config = valid_config.copy()
    invalid_config['rsi_oversold'] = 50
    with pytest.raises(ValueError, match="Nieprawidłowa wartość: rsi_oversold musi być mniejsze od 50"):
        ConfigLoader.validate_strategy_config(invalid_config)

def test_config_loader_validate_database_config():
    """Test walidacji konfiguracji bazy danych."""
    valid_config = {
        'host': 'localhost',
        'port': 5432,
        'name': 'test_db',
        'user': 'test_user',
        'password': 'test_pass'
    }
    ConfigLoader.validate_database_config(valid_config)

    # Test brakującego pola
    invalid_config = valid_config.copy()
    del invalid_config['host']
    with pytest.raises(ValueError, match="Brak wymaganego pola: host"):
        ConfigLoader.validate_database_config(invalid_config)

    # Test nieprawidłowego portu
    invalid_config = valid_config.copy()
    invalid_config['port'] = 0
    with pytest.raises(ValueError, match="Nieprawidłowa wartość: port musi być liczbą z zakresu 1-65535"):
        ConfigLoader.validate_database_config(invalid_config)

def test_config_loader_validate_logging_config():
    """Test walidacji konfiguracji logowania."""
    valid_config = {
        'level': 'INFO',
        'file': 'test.log',
        'format': '%(message)s'
    }
    ConfigLoader.validate_logging_config(valid_config)

    # Test brakującego pola
    invalid_config = valid_config.copy()
    del invalid_config['level']
    with pytest.raises(ValueError, match="Brak wymaganego pola: level"):
        ConfigLoader.validate_logging_config(invalid_config)

    # Test nieprawidłowego poziomu
    invalid_config = valid_config.copy()
    invalid_config['level'] = 'INVALID'
    with pytest.raises(ValueError, match="Nieprawidłowy poziom logowania"):
        ConfigLoader.validate_logging_config(invalid_config)

def test_config_loader_merge_configs():
    """Test łączenia konfiguracji."""
    base_config = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3
        }
    }
    override_config = {
        'b': {
            'c': 4
        },
        'e': 5
    }
    merged = ConfigLoader.merge_configs(base_config, override_config)
    assert merged == {
        'a': 1,
        'b': {
            'c': 4,
            'd': 3
        },
        'e': 5
    }

def test_config_loader_load_env_variables(monkeypatch):
    """Test ładowania zmiennych środowiskowych."""
    config = {
        'trading': {'symbol': 'EURUSD'},
        'database': {'password': 'old_pass'},
        'logging': {'level': 'INFO'}
    }
    
    monkeypatch.setenv('TRADING_SYMBOL', 'GBPUSD')
    monkeypatch.setenv('DB_PASSWORD', 'new_pass')
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    
    updated = ConfigLoader.load_env_variables(config)
    assert updated['trading']['symbol'] == 'GBPUSD'
    assert updated['database']['password'] == 'new_pass'
    assert updated['logging']['level'] == 'DEBUG'

def test_config_loader_parse_error(tmp_path):
    """Test obsługi błędów parsowania plików konfiguracyjnych."""
    # Test dla JSON
    json_path = tmp_path / "invalid.json"
    json_path.write_text("{invalid json")
    with pytest.raises(ValueError, match="Błąd parsowania pliku"):
        ConfigLoader.load_config(str(json_path))

    # Test dla YAML
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text("invalid: yaml: [")
    with pytest.raises(ValueError, match="Błąd parsowania pliku"):
        ConfigLoader.load_config(str(yaml_path))

def test_config_loader_save_config(tmp_path):
    """Test zapisywania konfiguracji do pliku."""
    config = {
        "test": "value",
        "nested": {"key": "value"}
    }

    # Test zapisu do JSON
    json_path = tmp_path / "config.json"
    ConfigLoader.save_config(config, str(json_path))
    assert json_path.exists()
    loaded_json = ConfigLoader.load_config(str(json_path))
    assert loaded_json == config

    # Test zapisu do YAML
    yaml_path = tmp_path / "config.yaml"
    ConfigLoader.save_config(config, str(yaml_path))
    assert yaml_path.exists()
    loaded_yaml = ConfigLoader.load_config(str(yaml_path))
    assert loaded_yaml == config

    # Test nieobsługiwanego formatu
    invalid_path = tmp_path / "config.txt"
    with pytest.raises(ValueError, match="Nieobsługiwany format pliku"):
        ConfigLoader.save_config(config, str(invalid_path))

def test_config_loader_validate_dependencies():
    """Test walidacji zależności między sekcjami konfiguracji."""
    # Test poprawnej konfiguracji
    valid_config = ConfigLoader.get_default_config()
    ConfigLoader.validate_config_dependencies(valid_config)

    # Test brakującej sekcji
    invalid_config = valid_config.copy()
    del invalid_config['trading']
    with pytest.raises(ValueError, match="Brak wymaganej sekcji: trading"):
        ConfigLoader.validate_config_dependencies(invalid_config)

    # Test nieprawidłowych zależności stop loss i take profit
    invalid_config = valid_config.copy()
    invalid_config['trading']['stop_loss_pips'] = 100
    invalid_config['trading']['take_profit_pips'] = 50
    with pytest.raises(ValueError, match="Nieprawidłowe zależności: take_profit_pips musi być większe od stop_loss_pips"):
        ConfigLoader.validate_config_dependencies(invalid_config)

    # Test nieprawidłowych zależności SMA
    invalid_config = valid_config.copy()
    invalid_config['strategy']['sma_fast'] = 50
    invalid_config['strategy']['sma_slow'] = 20
    # Upewniamy się, że take_profit jest większe od stop_loss
    invalid_config['trading']['stop_loss_pips'] = 50
    invalid_config['trading']['take_profit_pips'] = 100
    with pytest.raises(ValueError, match="Nieprawidłowe zależności: sma_fast musi być mniejsze od sma_slow"):
        ConfigLoader.validate_config_dependencies(invalid_config)

    # Test nieprawidłowych zależności RSI
    invalid_config = valid_config.copy()
    invalid_config['strategy']['rsi_oversold'] = 60
    invalid_config['strategy']['rsi_overbought'] = 40
    # Upewniamy się, że pozostałe zależności są poprawne
    invalid_config['trading']['stop_loss_pips'] = 50
    invalid_config['trading']['take_profit_pips'] = 100
    invalid_config['strategy']['sma_fast'] = 20
    invalid_config['strategy']['sma_slow'] = 50
    with pytest.raises(ValueError, match="Nieprawidłowe zależności: rsi_oversold musi być mniejsze od rsi_overbought"):
        ConfigLoader.validate_config_dependencies(invalid_config)

    # Test minimalnej różnicy RSI
    invalid_config = valid_config.copy()
    invalid_config['strategy']['rsi_oversold'] = 30
    invalid_config['strategy']['rsi_overbought'] = 60
    # Upewniamy się, że pozostałe zależności są poprawne
    invalid_config['trading']['stop_loss_pips'] = 50
    invalid_config['trading']['take_profit_pips'] = 100
    invalid_config['strategy']['sma_fast'] = 20
    invalid_config['strategy']['sma_slow'] = 50
    with pytest.raises(ValueError, match="Nieprawidłowe zależności: Różnica między rsi_overbought i rsi_oversold musi być co najmniej 40"):
        ConfigLoader.validate_config_dependencies(invalid_config)

def test_config_loader_parse_error_details(tmp_path):
    """Test szczegółowych błędów parsowania plików konfiguracyjnych."""
    # Test dla JSON z komunikatem błędu
    json_path = tmp_path / "invalid.json"
    json_path.write_text("{invalid: json}")
    with pytest.raises(ValueError) as exc_info:
        ConfigLoader.load_config(str(json_path))
    assert "Błąd parsowania pliku:" in str(exc_info.value)

    # Test dla YAML z komunikatem błędu
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text("- invalid:\n  yaml: [")
    with pytest.raises(ValueError) as exc_info:
        ConfigLoader.load_config(str(yaml_path))
    assert "Błąd parsowania pliku:" in str(exc_info.value)

def test_config_loader_save_error(tmp_path, monkeypatch):
    """Test błędów przy zapisywaniu konfiguracji."""
    config = {"test": "value"}
    
    # Test nieobsługiwanego formatu
    invalid_format = tmp_path / "config.txt"
    with pytest.raises(ValueError, match="Nieobsługiwany format pliku"):
        ConfigLoader.save_config(config, str(invalid_format))
    
    # Symuluj błąd zapisu przez zablokowanie dostępu do katalogu
    def mock_mkdir(*args, **kwargs):
        raise PermissionError("Brak uprawnień")
    
    monkeypatch.setattr(Path, "mkdir", mock_mkdir)
    
    # Test zapisu do katalogu bez uprawnień
    invalid_path = tmp_path / "nonexistent" / "config.json"
    with pytest.raises(ValueError, match="Błąd zapisu pliku: Brak uprawnień"):
        ConfigLoader.save_config(config, str(invalid_path))

def test_config_loader_get_default_config():
    """Test domyślnej konfiguracji."""
    config = ConfigLoader.get_default_config()
    
    # Sprawdź czy wszystkie wymagane sekcje są obecne
    assert all(section in config for section in ['trading', 'strategy', 'database', 'logging'])
    
    # Sprawdź wartości domyślne dla tradingu
    assert config['trading']['symbol'] == 'EURUSD'
    assert config['trading']['timeframe'] == '1H'
    assert config['trading']['max_position_size'] == 1.0
    assert config['trading']['stop_loss_pips'] == 50
    assert config['trading']['take_profit_pips'] == 100
    
    # Sprawdź wartości domyślne dla strategii
    assert config['strategy']['rsi_period'] == 14
    assert config['strategy']['rsi_oversold'] == 30
    assert config['strategy']['rsi_overbought'] == 70
    assert config['strategy']['sma_fast'] == 20
    assert config['strategy']['sma_slow'] == 50
    
    # Sprawdź wartości domyślne dla bazy danych
    assert config['database']['host'] == 'localhost'
    assert config['database']['port'] == 5432
    assert config['database']['name'] == 'trading_db'
    assert config['database']['user'] == 'trader'
    assert config['database']['password'] == 'default_password'
    
    # Sprawdź wartości domyślne dla logowania
    assert config['logging']['level'] == 'INFO'
    assert config['logging']['file'] == 'trading.log'
    assert config['logging']['format'] == '%(asctime)s - %(levelname)s - %(message)s'

def test_config_loader_file_operations(tmp_path, monkeypatch):
    """Test operacji na plikach konfiguracyjnych."""
    config = {"test": "value"}
    
    # Test błędu parsowania JSON
    def mock_json_load(*args, **kwargs):
        raise json.JSONDecodeError("Invalid JSON", "", 0)
    
    monkeypatch.setattr(json, "load", mock_json_load)
    
    json_path = tmp_path / "config.json"
    json_path.write_text("{")
    with pytest.raises(ValueError, match="Błąd parsowania pliku:"):
        ConfigLoader.load_config(str(json_path))
    
    # Test błędu parsowania YAML
    def mock_yaml_load(*args, **kwargs):
        raise yaml.YAMLError("Invalid YAML")
    
    monkeypatch.setattr(yaml, "safe_load", mock_yaml_load)
    
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("[")
    with pytest.raises(ValueError, match="Błąd parsowania pliku:"):
        ConfigLoader.load_config(str(yaml_path))
    
    # Test błędu zapisu JSON
    def mock_json_dump(*args, **kwargs):
        raise TypeError("Invalid type")
    
    monkeypatch.setattr(json, "dump", mock_json_dump)
    
    with pytest.raises(ValueError, match="Błąd zapisu pliku:"):
        ConfigLoader.save_config(config, str(json_path))
    
    # Test błędu zapisu YAML
    def mock_yaml_dump(*args, **kwargs):
        raise TypeError("Invalid type")
    
    monkeypatch.setattr(yaml, "dump", mock_yaml_dump)
    
    with pytest.raises(ValueError, match="Błąd zapisu pliku:"):
        ConfigLoader.save_config(config, str(yaml_path))

def test_config_loader_load_error(tmp_path, monkeypatch):
    """Test błędów przy wczytywaniu plików konfiguracyjnych."""
    # Test błędu otwarcia pliku
    def mock_open(*args, **kwargs):
        raise IOError("Błąd otwarcia pliku")
    
    monkeypatch.setattr("builtins.open", mock_open)
    
    # Test dla JSON
    json_path = tmp_path / "config.json"
    with pytest.raises(ValueError, match="Błąd parsowania pliku: Błąd otwarcia pliku"):
        ConfigLoader.load_config(str(json_path))
    
    # Test dla YAML
    yaml_path = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="Błąd parsowania pliku: Błąd otwarcia pliku"):
        ConfigLoader.load_config(str(yaml_path))
    
    # Test dla nieobsługiwanego formatu
    txt_path = tmp_path / "config.txt"
    with pytest.raises(ValueError, match="Nieobsługiwany format pliku"):
        ConfigLoader.load_config(str(txt_path)) 