"""
Moduł zawierający testy dla konfiguracji.
"""
import pytest
import os
import json
from unittest.mock import patch, mock_open
from src.utils.config import Config


@pytest.fixture
def sample_config():
    """Fixture tworzący przykładową konfigurację."""
    return {
        'MT5_LOGIN': 'test_login',
        'MT5_PASSWORD': 'test_password',
        'MT5_SERVER': 'test_server',
        'ANTHROPIC_API_KEY': 'test_key',
        'POSTGRES_HOST': 'test_host',
        'POSTGRES_PORT': '5433',
        'POSTGRES_DB': 'test_db',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_password',
        'DEBUG': True
    }


def test_config_initialization():
    """Test inicjalizacji konfiguracji z domyślnymi wartościami."""
    config = Config()
    assert config.MT5_LOGIN == ''
    assert config.MT5_PASSWORD == ''
    assert config.MT5_SERVER == ''
    assert config.ANTHROPIC_API_KEY == ''
    assert config.POSTGRES_HOST == 'localhost'
    assert config.POSTGRES_PORT == '5432'
    assert config.POSTGRES_DB == 'trading_db'
    assert config.POSTGRES_USER == ''
    assert config.POSTGRES_PASSWORD == ''
    assert config.DEBUG is False


def test_config_initialization_type_validation():
    """Test walidacji typów podczas inicjalizacji."""
    with pytest.raises(ValueError, match="MT5_LOGIN musi być typu str"):
        Config(MT5_LOGIN=123)
        
    with pytest.raises(ValueError, match="MT5_PASSWORD musi być typu str"):
        Config(MT5_PASSWORD=123)
        
    with pytest.raises(ValueError, match="MT5_SERVER musi być typu str"):
        Config(MT5_SERVER=123)
        
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY musi być typu str"):
        Config(ANTHROPIC_API_KEY=123)
        
    with pytest.raises(ValueError, match="POSTGRES_HOST musi być typu str"):
        Config(POSTGRES_HOST=123)
        
    with pytest.raises(ValueError, match="POSTGRES_PORT musi być typu str"):
        Config(POSTGRES_PORT=5432)  # int zamiast str
        
    with pytest.raises(ValueError, match="POSTGRES_DB musi być typu str"):
        Config(POSTGRES_DB=123)
        
    with pytest.raises(ValueError, match="POSTGRES_USER musi być typu str"):
        Config(POSTGRES_USER=123)
        
    with pytest.raises(ValueError, match="POSTGRES_PASSWORD musi być typu str"):
        Config(POSTGRES_PASSWORD=123)
        
    with pytest.raises(ValueError, match="DEBUG musi być typu bool"):
        Config(DEBUG="true")  # str zamiast bool


def test_config_from_dict(sample_config):
    """Test tworzenia konfiguracji ze słownika."""
    config = Config.from_dict(sample_config)
    
    assert config.MT5_LOGIN == 'test_login'
    assert config.MT5_PASSWORD == 'test_password'
    assert config.MT5_SERVER == 'test_server'
    assert config.ANTHROPIC_API_KEY == 'test_key'
    assert config.POSTGRES_HOST == 'test_host'
    assert config.POSTGRES_PORT == '5433'
    assert config.POSTGRES_DB == 'test_db'
    assert config.POSTGRES_USER == 'test_user'
    assert config.POSTGRES_PASSWORD == 'test_password'
    assert config.DEBUG is True


def test_config_from_dict_missing_fields():
    """Test tworzenia konfiguracji z brakującymi polami."""
    with pytest.raises(ValueError, match="Brak wymaganego pola: MT5_LOGIN"):
        Config.from_dict({})
        
    with pytest.raises(ValueError, match="Pole MT5_LOGIN nie może być puste"):
        Config.from_dict({'MT5_LOGIN': ''})


def test_config_to_dict(sample_config):
    """Test konwersji konfiguracji do słownika."""
    config = Config.from_dict(sample_config)
    config_dict = config.to_dict()
    
    assert config_dict['MT5_LOGIN'] == 'test_login'
    assert config_dict['MT5_PASSWORD'] == 'test_password'
    assert config_dict['MT5_SERVER'] == 'test_server'
    assert config_dict['ANTHROPIC_API_KEY'] == 'test_key'
    assert config_dict['POSTGRES_HOST'] == 'test_host'
    assert config_dict['POSTGRES_PORT'] == '5433'
    assert config_dict['POSTGRES_DB'] == 'test_db'
    assert config_dict['POSTGRES_USER'] == 'test_user'
    assert config_dict['POSTGRES_PASSWORD'] == 'test_password'
    assert config_dict['DEBUG'] is True


def test_load_config_from_file(tmp_path, sample_config):
    """Test wczytywania konfiguracji z pliku."""
    config_path = tmp_path / "config.json"
    
    # Zapisz przykładową konfigurację do pliku
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f)
    
    # Wczytaj konfigurację
    config = Config.load_config(str(config_path))
    
    assert config.MT5_LOGIN == 'test_login'
    assert config.MT5_PASSWORD == 'test_password'
    assert config.MT5_SERVER == 'test_server'
    assert config.ANTHROPIC_API_KEY == 'test_key'
    assert config.POSTGRES_USER == 'test_user'
    assert config.POSTGRES_PASSWORD == 'test_password'


def test_load_config_file_not_found():
    """Test wczytywania konfiguracji z nieistniejącego pliku."""
    with patch('builtins.open', side_effect=FileNotFoundError):
        with patch.dict('os.environ', {
            'MT5_LOGIN': 'env_login',
            'MT5_PASSWORD': 'env_password',
            'MT5_SERVER': 'env_server',
            'ANTHROPIC_API_KEY': 'env_key',
            'POSTGRES_USER': 'env_user',
            'POSTGRES_PASSWORD': 'env_password'
        }):
            config = Config.load_config('nonexistent.json')
            assert config.MT5_LOGIN == 'env_login'


def test_load_config_invalid_json():
    """Test wczytywania konfiguracji z nieprawidłowego pliku JSON."""
    with patch('builtins.open', mock_open(read_data='invalid json')):
        with pytest.raises(ValueError, match="Nieprawidłowy format pliku JSON"):
            Config.load_config('config.json')


def test_save_config_io_error(sample_config):
    """Test zapisywania konfiguracji z błędem IO."""
    config = Config.from_dict(sample_config)
    with patch('builtins.open', side_effect=IOError):
        with pytest.raises(IOError, match="Nie można zapisać pliku konfiguracyjnego"):
            config.save_config('config.json')


def test_save_config_to_file(tmp_path, sample_config):
    """Test zapisywania konfiguracji do pliku."""
    config_path = tmp_path / "config.json"
    config = Config.from_dict(sample_config)
    
    # Zapisz konfigurację
    config.save_config(str(config_path))
    
    # Sprawdź czy plik istnieje
    assert os.path.exists(config_path)
    
    # Wczytaj zapisaną konfigurację
    with open(config_path, 'r', encoding='utf-8') as f:
        saved_config = json.load(f)
        
    assert saved_config == sample_config


def test_config_from_env(monkeypatch):
    """Test tworzenia konfiguracji ze zmiennych środowiskowych."""
    env_vars = {
        'MT5_LOGIN': 'env_login',
        'MT5_PASSWORD': 'env_password',
        'MT5_SERVER': 'env_server',
        'ANTHROPIC_API_KEY': 'env_key',
        'POSTGRES_HOST': 'env_host',
        'POSTGRES_PORT': '5434',
        'POSTGRES_DB': 'env_db',
        'POSTGRES_USER': 'env_user',
        'POSTGRES_PASSWORD': 'env_password',
        'DEBUG': 'true'
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
        
    config = Config.from_env()
    
    assert config.MT5_LOGIN == 'env_login'
    assert config.MT5_PASSWORD == 'env_password'
    assert config.MT5_SERVER == 'env_server'
    assert config.ANTHROPIC_API_KEY == 'env_key'
    assert config.POSTGRES_HOST == 'env_host'
    assert config.POSTGRES_PORT == '5434'
    assert config.POSTGRES_DB == 'env_db'
    assert config.POSTGRES_USER == 'env_user'
    assert config.POSTGRES_PASSWORD == 'env_password'
    assert config.DEBUG is True


def test_config_from_env_missing_vars():
    """Test tworzenia konfiguracji z brakującymi zmiennymi środowiskowymi."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="Brak wymaganej zmiennej środowiskowej: MT5_LOGIN"):
            Config.from_env()


def test_config_from_env_empty_vars():
    """Test tworzenia konfiguracji z pustymi zmiennymi środowiskowymi."""
    with patch.dict('os.environ', {
        'MT5_LOGIN': '',
        'MT5_PASSWORD': '',
        'MT5_SERVER': '',
        'ANTHROPIC_API_KEY': '',
        'POSTGRES_USER': '',
        'POSTGRES_PASSWORD': ''
    }):
        with pytest.raises(ValueError, match="Brak wymaganej zmiennej środowiskowej: MT5_LOGIN"):
            Config.from_env()


def test_config_save_and_load(tmp_path, sample_config):
    """Test zapisywania i wczytywania konfiguracji z pliku."""
    config_path = tmp_path / "test_config.json"
    
    # Zapisz konfigurację
    config = Config.from_dict(sample_config)
    config.save_config(str(config_path))
    
    # Wczytaj konfigurację
    loaded_config = Config.load_config(str(config_path))
    
    assert loaded_config.MT5_LOGIN == 'test_login'
    assert loaded_config.MT5_PASSWORD == 'test_password'
    assert loaded_config.MT5_SERVER == 'test_server'
    assert loaded_config.ANTHROPIC_API_KEY == 'test_key'
    assert loaded_config.POSTGRES_USER == 'test_user'
    assert loaded_config.POSTGRES_PASSWORD == 'test_password'


def test_config_validate():
    """Test walidacji konfiguracji."""
    # Niepoprawna konfiguracja - brak MT5_LOGIN
    with pytest.raises(ValueError, match="Brak wymaganego pola: MT5_LOGIN"):
        config = Config()
        config.validate()
    
    # Niepoprawna konfiguracja - brak MT5_PASSWORD
    with pytest.raises(ValueError, match="Brak wymaganego pola: MT5_PASSWORD"):
        config = Config(MT5_LOGIN='test')
        config.validate()
        
    # Niepoprawna konfiguracja - brak MT5_SERVER
    with pytest.raises(ValueError, match="Brak wymaganego pola: MT5_SERVER"):
        config = Config(MT5_LOGIN='test', MT5_PASSWORD='test')
        config.validate()
        
    # Niepoprawna konfiguracja - brak POSTGRES_USER
    with pytest.raises(ValueError, match="Brak wymaganego pola: POSTGRES_USER"):
        config = Config(MT5_LOGIN='test', MT5_PASSWORD='test', MT5_SERVER='test')
        config.validate()
        
    # Niepoprawna konfiguracja - brak POSTGRES_PASSWORD
    with pytest.raises(ValueError, match="Brak wymaganego pola: POSTGRES_PASSWORD"):
        config = Config(
            MT5_LOGIN='test',
            MT5_PASSWORD='test',
            MT5_SERVER='test',
            POSTGRES_USER='test'
        )
        config.validate()
        
    # Niepoprawna konfiguracja - brak ANTHROPIC_API_KEY
    with pytest.raises(ValueError, match="Brak wymaganego pola: ANTHROPIC_API_KEY"):
        config = Config(
            MT5_LOGIN='test',
            MT5_PASSWORD='test',
            MT5_SERVER='test',
            POSTGRES_USER='test',
            POSTGRES_PASSWORD='test'
        )
        config.validate()
        
    # Poprawna konfiguracja
    config = Config(
        MT5_LOGIN='test',
        MT5_PASSWORD='test',
        MT5_SERVER='test',
        POSTGRES_USER='test',
        POSTGRES_PASSWORD='test',
        ANTHROPIC_API_KEY='test'
    )
    assert config.validate() is True


def test_config_update():
    """Test aktualizacji konfiguracji."""
    config = Config()
    
    # Test poprawnej aktualizacji
    update_dict = {
        'MT5_LOGIN': 'new_login',
        'MT5_PASSWORD': 'new_password',
        'DEBUG': True
    }
    config.update(update_dict)
    assert config.MT5_LOGIN == 'new_login'
    assert config.MT5_PASSWORD == 'new_password'
    assert config.DEBUG is True
    
    # Test aktualizacji z nieprawidłowym typem
    with pytest.raises(ValueError, match="MT5_LOGIN musi być typu str"):
        config.update({'MT5_LOGIN': 123})
    
    with pytest.raises(ValueError, match="DEBUG musi być typu bool"):
        config.update({'DEBUG': 'true'})
    
    # Test aktualizacji z nieistniejącym polem
    config.update({'nonexistent_field': 'value'})  # powinno być zignorowane
    assert not hasattr(config, 'nonexistent_field')


def test_config_defaults():
    """Test wartości domyślnych konfiguracji."""
    config = Config()
    assert config.POSTGRES_HOST == 'localhost'
    assert config.POSTGRES_PORT == '5432'
    assert config.POSTGRES_DB == 'trading_db'
    assert config.DEBUG is False


def test_config_invalid_values():
    """Test nieprawidłowych wartości w konfiguracji."""
    # Test nieprawidłowych typów w słowniku
    invalid_config = {
        'MT5_LOGIN': 123,  # int zamiast str
        'MT5_PASSWORD': 'test_password',
        'MT5_SERVER': 'test_server',
        'ANTHROPIC_API_KEY': 'test_key',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_password'
    }
    with pytest.raises(ValueError, match="MT5_LOGIN musi być typu str"):
        Config.from_dict(invalid_config)
    
    # Test nieprawidłowego typu DEBUG w słowniku
    invalid_debug = {
        'MT5_LOGIN': 'test_login',
        'MT5_PASSWORD': 'test_password',
        'MT5_SERVER': 'test_server',
        'ANTHROPIC_API_KEY': 'test_key',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_password',
        'DEBUG': 'true'  # str zamiast bool
    }
    with pytest.raises(ValueError, match="DEBUG musi być typu bool"):
        Config.from_dict(invalid_debug) 