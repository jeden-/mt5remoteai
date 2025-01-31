"""
Moduł zawierający testy dla loggera.
"""
import os
import pytest
import logging
import shutil
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock
from src.utils.logger import TradingLogger


@pytest.fixture
def temp_logger(tmp_path):
    """Fixture tworzący tymczasowy logger."""
    logs_dir = tmp_path / "logs"
    logger = TradingLogger(str(logs_dir))
    yield logger
    # Cleanup po testach
    if os.path.exists(logs_dir):
        # Zamknij wszystkie handlery przed usunięciem plików
        logger.trades_logger.handlers.clear()
        logger.error_logger.handlers.clear()
        logger.ai_logger.handlers.clear()
        shutil.rmtree(logs_dir)


@pytest.fixture
def sample_trade_info():
    """Fixture tworzący przykładowe dane o transakcji."""
    return {
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 0.1,
        'price': 1.1234,
        'sl': 1.1200,
        'tp': 1.1300,
        'profit': 50.0
    }


@pytest.fixture
def sample_ai_analysis():
    """Fixture tworzący przykładowe dane analizy AI."""
    return {
        'symbol': 'EURUSD',
        'signal': 'BUY',
        'confidence': 85,
        'reasoning': 'Silny trend wzrostowy'
    }


def test_logger_initialization(temp_logger):
    """Test inicjalizacji loggera."""
    assert os.path.exists(temp_logger.logs_dir)
    assert os.path.exists(os.path.join(temp_logger.logs_dir, 'trades.log'))
    assert os.path.exists(os.path.join(temp_logger.logs_dir, 'errors.log'))
    assert os.path.exists(os.path.join(temp_logger.logs_dir, 'ai_analysis.log'))


def test_log_trade(temp_logger, sample_trade_info):
    """Test logowania transakcji."""
    temp_logger.log_trade(sample_trade_info)
    
    log_path = temp_logger.get_logs_path('trades')
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    assert '🥷 Symbol: EURUSD' in log_content
    assert 'Typ: BUY' in log_content
    assert 'Wolumen: 0.1' in log_content
    assert 'Cena: 1.1234' in log_content
    assert 'SL: 1.12' in log_content
    assert 'TP: 1.13' in log_content
    assert 'Profit: 50.0' in log_content


def test_log_trade_with_missing_data(temp_logger):
    """Test logowania transakcji z brakującymi danymi."""
    temp_logger.log_trade({})
    
    log_path = temp_logger.get_logs_path('trades')
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    assert '🥷 Symbol: N/A' in log_content
    assert 'Typ: N/A' in log_content
    assert 'Wolumen: 0' in log_content
    assert 'Cena: 0' in log_content
    assert 'SL: N/A' in log_content
    assert 'TP: N/A' in log_content
    assert 'Profit: 0' in log_content


def test_log_error(temp_logger):
    """Test logowania błędu."""
    error_msg = "Test error message"
    exception = ValueError("Test exception")
    temp_logger.log_error(error_msg, exception)
    
    log_path = temp_logger.get_logs_path('errors')
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    assert '❌ Test error message' in log_content
    assert 'Stacktrace: Test exception' in log_content


def test_log_error_without_exception(temp_logger):
    """Test logowania błędu bez wyjątku."""
    error_msg = "Test error message"
    temp_logger.log_error(error_msg)
    
    log_path = temp_logger.get_logs_path('errors')
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    assert '❌ Test error message' in log_content
    assert 'Stacktrace' not in log_content


def test_log_warning(temp_logger):
    """Test logowania ostrzeżenia."""
    warning_msg = "Test warning message"
    temp_logger.log_warning(warning_msg)
    
    log_path = temp_logger.get_logs_path('errors')
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    assert '⚠️ Test warning message' in log_content


def test_log_ai_analysis(temp_logger, sample_ai_analysis):
    """Test logowania analizy AI."""
    temp_logger.log_ai_analysis(sample_ai_analysis)
    
    log_path = temp_logger.get_logs_path('ai')
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    assert '🤖 Symbol: EURUSD' in log_content
    assert 'Sygnał: BUY' in log_content
    assert 'Pewność: 85%' in log_content
    assert 'Uzasadnienie: Silny trend wzrostowy' in log_content


def test_log_ai_analysis_with_missing_data(temp_logger):
    """Test logowania analizy AI z brakującymi danymi."""
    temp_logger.log_ai_analysis({})
    
    log_path = temp_logger.get_logs_path('ai')
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    assert '🤖 Symbol: N/A' in log_content
    assert 'Sygnał: N/A' in log_content
    assert 'Pewność: 0%' in log_content
    assert 'Uzasadnienie: N/A' in log_content


def test_get_logs_path(temp_logger):
    """Test pobierania ścieżek do plików logów."""
    assert temp_logger.get_logs_path('trades').endswith('trades.log')
    assert temp_logger.get_logs_path('errors').endswith('errors.log')
    assert temp_logger.get_logs_path('ai').endswith('ai_analysis.log')
    
    with pytest.raises(ValueError, match="Nieznany typ logów: invalid"):
        temp_logger.get_logs_path('invalid')


def test_clear_logs(temp_logger, sample_trade_info, sample_ai_analysis):
    """Test czyszczenia logów."""
    # Dodaj przykładowe logi
    temp_logger.log_trade(sample_trade_info)
    temp_logger.log_error("Test error")
    temp_logger.log_ai_analysis(sample_ai_analysis)
    
    # Wyczyść tylko logi transakcji
    temp_logger.clear_logs('trades')
    assert os.path.getsize(temp_logger.get_logs_path('trades')) == 0
    assert os.path.getsize(temp_logger.get_logs_path('errors')) > 0
    assert os.path.getsize(temp_logger.get_logs_path('ai')) > 0
    
    # Wyczyść wszystkie logi
    temp_logger.clear_logs()
    assert os.path.getsize(temp_logger.get_logs_path('trades')) == 0
    assert os.path.getsize(temp_logger.get_logs_path('errors')) == 0
    assert os.path.getsize(temp_logger.get_logs_path('ai')) == 0
    
    # Test nieprawidłowego typu logów
    with pytest.raises(ValueError, match="Nieznany typ logów: invalid"):
        temp_logger.clear_logs('invalid')


def test_archive_logs(temp_logger, sample_trade_info, sample_ai_analysis):
    """Test archiwizacji logów."""
    # Dodaj przykładowe logi
    temp_logger.log_trade(sample_trade_info)
    temp_logger.log_error("Test error")
    temp_logger.log_ai_analysis(sample_ai_analysis)
    
    # Archiwizuj logi
    temp_logger.archive_logs()
    
    # Sprawdź czy pliki zostały zarchiwizowane
    archive_dir = os.path.join(temp_logger.logs_dir, 'archive')
    assert os.path.exists(archive_dir)
    assert len(os.listdir(archive_dir)) == 1  # jeden katalog z timestampem
    
    # Sprawdź czy oryginalne pliki są puste
    assert os.path.getsize(temp_logger.get_logs_path('trades')) == 0
    assert os.path.getsize(temp_logger.get_logs_path('errors')) == 0
    assert os.path.getsize(temp_logger.get_logs_path('ai')) == 0


def test_archive_logs_with_error(temp_logger):
    """Test archiwizacji logów z błędem."""
    with patch('os.makedirs', side_effect=OSError("Test error")):
        with pytest.raises(IOError, match="Błąd podczas archiwizacji logów"):
            temp_logger.archive_logs()


def test_log_trade_with_exception(temp_logger):
    """Test logowania transakcji z wyjątkiem."""
    with patch.object(temp_logger.trades_logger, 'info', side_effect=Exception("Test error")):
        temp_logger.log_trade({})
        
        log_path = temp_logger.get_logs_path('errors')
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
        assert '❌ Błąd podczas logowania transakcji: Test error' in log_content


def test_log_ai_analysis_with_exception(temp_logger):
    """Test logowania analizy AI z wyjątkiem."""
    with patch.object(temp_logger.ai_logger, 'info', side_effect=Exception("Test error")):
        temp_logger.log_ai_analysis({})
        
        log_path = temp_logger.get_logs_path('errors')
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
        assert '❌ Błąd podczas logowania analizy AI: Test error' in log_content


def test_logger_custom_log_files(tmp_path):
    """Test loggera z niestandardowymi nazwami plików."""
    logs_dir = tmp_path / "logs"
    logger = TradingLogger(
        str(logs_dir),
        trades_log='custom_trades.log',
        errors_log='custom_errors.log',
        ai_log='custom_ai.log'
    )
    
    assert os.path.exists(os.path.join(logs_dir, 'custom_trades.log'))
    assert os.path.exists(os.path.join(logs_dir, 'custom_errors.log'))
    assert os.path.exists(os.path.join(logs_dir, 'custom_ai.log')) 