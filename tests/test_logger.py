"""
Testy jednostkowe dla modułu logger.py
"""
import os
import json
import pytest
import logging
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock
from src.utils.logger import TradingLogger, get_logger


@pytest.fixture
def sample_trade_info():
    """Przykładowe dane o transakcji do testów."""
    return {
        "symbol": "EURUSD",
        "type": "BUY",
        "volume": 0.1,
        "price": 1.1234,
        "sl": 1.1200,
        "tp": 1.1300,
        "profit": 100.0
    }


@pytest.fixture
def sample_ai_analysis():
    """Przykładowe dane analizy AI do testów."""
    return {
        "symbol": "EURUSD",
        "signal": "BUY",
        "confidence": 85,
        "reasoning": "Trend wzrostowy na RSI"
    }


@pytest.fixture
def sample_market_data():
    """Przykładowe dane rynkowe do testów."""
    return {
        "symbol": "EURUSD",
        "timeframe": "1H",
        "open": 1.1234,
        "high": 1.1256,
        "low": 1.1212,
        "close": 1.1245,
        "volume": 1000
    }


@pytest.fixture
def sample_performance_metrics():
    """Przykładowe metryki wydajności do testów."""
    return {
        "execution_time": 0.123,
        "memory_usage": 156.7,
        "cpu_usage": 25.4
    }


@pytest.fixture
def sample_strategy_stats():
    """Przykładowe statystyki strategii do testów."""
    return {
        "total_trades": 100,
        "winning_trades": 60,
        "losing_trades": 40,
        "win_rate": 0.6,
        "profit_factor": 1.5,
        "max_drawdown": 0.1,
        "sharpe_ratio": 1.8,
        "total_profit": 5000.0
    }


@pytest.fixture
def logger(tmp_path):
    """Fixture tworzący instancję TradingLogger z tymczasowym katalogiem na logi."""
    logs_dir = tmp_path / "logs"
    return TradingLogger(
        strategy_name="test_strategy",
        logs_dir=str(logs_dir),
        trades_log="trades.log",
        errors_log="errors.log",
        ai_log="ai.log",
        log_file="main.log",
        json_log_file="trades.json"
    )


def test_logger_initialization(tmp_path):
    """Test inicjalizacji loggera."""
    logs_dir = tmp_path / "logs"
    logger = TradingLogger(
        strategy_name="test_strategy",
        logs_dir=str(logs_dir)
    )
    
    # Sprawdź czy katalog został utworzony
    assert os.path.exists(logs_dir)
    
    # Sprawdź czy pliki logów zostały utworzone
    assert os.path.exists(logs_dir / "trades.log")
    assert os.path.exists(logs_dir / "errors.log")
    assert os.path.exists(logs_dir / "ai.log")
    assert os.path.exists(logs_dir / "main.log")


def test_log_trade(logger, sample_trade_info):
    """Test logowania transakcji."""
    logger.log_trade(sample_trade_info)
    
    # Sprawdź czy plik z logami transakcji istnieje i zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.trades_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "EURUSD" in log_content
        assert "BUY" in log_content
        assert "0.1" in log_content


def test_log_error(logger):
    """Test logowania błędu."""
    test_message = "Test error message"
    test_exception = Exception("Test exception")
    
    logger.log_error(test_message, test_exception)
    
    # Sprawdź czy plik z logami błędów istnieje i zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.errors_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert test_message in log_content
        assert "Test exception" in log_content


def test_log_warning(logger):
    """Test logowania ostrzeżenia."""
    test_message = "Test warning message"
    
    logger.log_warning(test_message)
    
    # Sprawdź czy plik z logami błędów istnieje i zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.errors_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert test_message in log_content
        assert "WARNING" in log_content


def test_log_ai_analysis(logger, sample_ai_analysis):
    """Test logowania analizy AI."""
    logger.log_ai_analysis(sample_ai_analysis)
    
    # Sprawdź czy plik z logami AI istnieje i zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.ai_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "EURUSD" in log_content
        assert "BUY" in log_content
        assert "85%" in log_content
        assert "Trend wzrostowy na RSI" in log_content


def test_get_logs_path(logger):
    """Test pobierania ścieżki do plików logów."""
    # Test dla prawidłowych typów
    assert logger.get_logs_path('trades') == os.path.join(logger.logs_dir, logger.trades_log)
    assert logger.get_logs_path('errors') == os.path.join(logger.logs_dir, logger.errors_log)
    assert logger.get_logs_path('ai') == os.path.join(logger.logs_dir, logger.ai_log)
    
    # Test dla nieprawidłowego typu
    with pytest.raises(ValueError, match="Nieznany typ logów"):
        logger.get_logs_path('invalid')


def test_clear_logs(logger):
    """Test czyszczenia logów."""
    # Najpierw dodaj jakieś logi
    logger.log_warning("Test warning")
    logger.log_trade({"symbol": "EURUSD", "type": "BUY"})
    logger.log_ai_analysis({"symbol": "EURUSD", "signal": "SELL"})
    
    # Wyczyść konkretny typ logów
    logger.clear_logs('trades')
    trades_log_path = os.path.join(logger.logs_dir, logger.trades_log)
    with open(trades_log_path, 'r', encoding='utf-8') as f:
        assert f.read() == ''
    
    # Wyczyść wszystkie logi
    logger.clear_logs()
    for log_type in ['trades', 'errors', 'ai']:
        log_path = logger.get_logs_path(log_type)
        with open(log_path, 'r', encoding='utf-8') as f:
            assert f.read() == ''


def test_archive_logs(logger, sample_trade_info):
    """Test archiwizacji logów."""
    # Dodaj przykładowe logi
    logger.log_trade(sample_trade_info)
    logger.log_warning("Test warning")
    logger.log_ai_analysis({"symbol": "EURUSD", "signal": "SELL"})
    
    # Archiwizuj logi
    logger.archive_logs()
    
    # Sprawdź czy katalog archiwum został utworzony
    archive_dir = os.path.join(logger.logs_dir, 'archive')
    assert os.path.exists(archive_dir)
    
    # Sprawdź czy pliki zostały zarchiwizowane
    archived_files = os.listdir(os.path.join(archive_dir, os.listdir(archive_dir)[0]))
    assert 'trades.log' in archived_files
    assert 'errors.log' in archived_files
    assert 'ai.log' in archived_files
    
    # Sprawdź czy oryginalne pliki są puste
    for log_type in ['trades', 'errors', 'ai']:
        log_path = logger.get_logs_path(log_type)
        with open(log_path, 'r', encoding='utf-8') as f:
            assert f.read() == ''


def test_log_strategy_stats(logger, sample_strategy_stats):
    """Test logowania statystyk strategii."""
    logger.log_strategy_stats(sample_strategy_stats)
    
    # Sprawdź czy plik z logami zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.log_file)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "Statystyki strategii" in log_content
        assert "Liczba transakcji: 100" in log_content
        assert "Win rate: 60.00%" in log_content
        assert "Profit factor: 1.50" in log_content


def test_log_market_data(logger, sample_market_data):
    """Test logowania danych rynkowych."""
    logger.log_market_data(sample_market_data)
    
    # Sprawdź czy plik z logami zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.log_file)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "EURUSD" in log_content
        assert "1H" in log_content
        assert "1.1234" in log_content


def test_log_performance(logger, sample_performance_metrics):
    """Test logowania metryk wydajności."""
    logger.log_performance(sample_performance_metrics)
    
    # Sprawdź czy plik z logami zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.log_file)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "Metryki wydajności" in log_content
        assert "0.123s" in log_content
        assert "156.7MB" in log_content
        assert "25.4%" in log_content


def test_log_trade_to_json_success(logger, sample_trade_info):
    """Test zapisywania transakcji do pliku JSON."""
    logger.log_trade_to_json(sample_trade_info)
    
    # Sprawdź czy plik JSON istnieje i zawiera odpowiednie dane
    json_path = logger._get_json_file_path()
    assert os.path.exists(json_path)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        trades = json.load(f)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "EURUSD"
        assert trades[0]["type"] == "BUY"
        assert trades[0]["volume"] == 0.1
        assert "timestamp" in trades[0]


def test_log_trade_to_json_error(logger, sample_trade_info):
    """Test obsługi błędu podczas zapisu do pliku JSON."""
    # Symuluj błąd podczas zapisu do pliku
    with patch('builtins.open', side_effect=IOError("Test IO Error")):
        with pytest.raises(IOError, match="Błąd podczas zapisu do pliku JSON"):
            logger.log_trade_to_json(sample_trade_info)


def test_get_log_file_path(logger):
    """Test pobierania ścieżki do pliku logów."""
    path = logger._get_log_file_path("test.log")
    assert path == os.path.join(logger.logs_dir, "test.log")


def test_get_json_file_path(logger):
    """Test pobierania ścieżki do pliku JSON."""
    path = logger._get_json_file_path()
    assert path == os.path.join(logger.logs_dir, logger.json_log_file)


def test_get_archive_dir_path(logger):
    """Test pobierania ścieżki do katalogu archiwum."""
    path = logger._get_archive_dir_path()
    assert path == os.path.join(logger.logs_dir, "archive")


def test_log_trade_error(logger):
    """Test obsługi błędu podczas logowania transakcji."""
    with patch.object(logger.trades_logger, 'info', side_effect=Exception("Test error")):
        logger.log_trade({"symbol": "EURUSD", "type": "BUY"})
    
    # Sprawdź czy błąd został zalogowany
    log_path = os.path.join(logger.logs_dir, logger.errors_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "Błąd podczas logowania transakcji" in log_content


def test_log_ai_analysis_error(logger):
    """Test obsługi błędu podczas logowania analizy AI."""
    with patch.object(logger.ai_logger, 'info', side_effect=Exception("Test error")):
        logger.log_ai_analysis({"symbol": "EURUSD", "signal": "BUY"})
    
    # Sprawdź czy błąd został zalogowany
    log_path = os.path.join(logger.logs_dir, logger.errors_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "Błąd podczas logowania analizy AI" in log_content


def test_archive_logs_file_not_found(logger):
    """Test obsługi błędu gdy katalog logów nie istnieje."""
    # Zamknij wszystkie handlery logów
    logger.logger.handlers = []
    logger.trades_logger.handlers = []
    logger.error_logger.handlers = []
    logger.ai_logger.handlers = []
    
    # Usuń katalog logów
    import shutil
    shutil.rmtree(logger.logs_dir)
    
    with pytest.raises(FileNotFoundError, match="Katalog logów nie istnieje"):
        logger.archive_logs()


def test_archive_logs_permission_error(logger):
    """Test obsługi błędu braku uprawnień podczas archiwizacji."""
    with patch('os.makedirs', side_effect=PermissionError("Test error")):
        with pytest.raises(PermissionError, match="Brak uprawnień do zapisu w katalogu archiwum"):
            logger.archive_logs()


def test_archive_logs_io_error(logger):
    """Test obsługi błędu IO podczas archiwizacji."""
    with patch('shutil.copy2', side_effect=IOError("Test error")):
        with pytest.raises(IOError, match="Błąd podczas archiwizacji logów"):
            logger.archive_logs()


def test_log_signal(logger):
    """Test logowania sygnału tradingowego."""
    signal = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "strength": 0.85,
        "timestamp": "2024-02-03T12:00:00",
        "indicators": {
            "rsi": 30,
            "sma_fast": 1.1200,
            "sma_slow": 1.1150
        }
    }
    
    logger.log_signal(signal)
    
    # Sprawdź czy plik z logami zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.log_file)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "EURUSD" in log_content
        assert "BUY" in log_content
        assert "85%" in log_content
        assert "RSI: 30" in log_content


def test_log_trade_to_json_file_not_exists(logger, sample_trade_info):
    """Test logowania transakcji do JSON gdy plik nie istnieje."""
    json_path = logger._get_json_file_path()
    
    # Upewnij się, że plik nie istnieje
    if os.path.exists(json_path):
        os.remove(json_path)
    
    logger.log_trade_to_json(sample_trade_info)
    
    # Sprawdź czy plik został utworzony i zawiera poprawne dane
    assert os.path.exists(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["symbol"] == sample_trade_info["symbol"]


def test_log_trade_json_write_error(logger, sample_trade_info):
    """Test obsługi błędu podczas zapisu do pliku JSON w metodzie log_trade."""
    # Symuluj błąd podczas zapisu do pliku
    with patch('json.dump', side_effect=OSError("Test error")):
        logger.log_trade(sample_trade_info)
        
    # Sprawdź czy błąd został zalogowany
    log_path = os.path.join(logger.logs_dir, logger.errors_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "Błąd podczas zapisu do pliku JSON" in log_content


def test_log_trade_to_json_write_error(logger, sample_trade_info):
    """Test obsługi błędu podczas zapisu do pliku JSON w metodzie log_trade_to_json."""
    # Symuluj błąd podczas zapisu do pliku
    with patch('json.dump', side_effect=OSError("Test error")):
        with pytest.raises(OSError, match="Błąd podczas zapisu do pliku JSON"):
            logger.log_trade_to_json(sample_trade_info)


def test_archive_logs_copy_error(logger):
    """Test obsługi błędu podczas kopiowania plików."""
    with patch('shutil.copy2', side_effect=PermissionError("Test error")):
        with pytest.raises(PermissionError, match="Brak uprawnień do skopiowania pliku"):
            logger.archive_logs()


def test_archive_logs_create_archive_dir_error(logger):
    """Test obsługi błędu podczas tworzenia katalogu archiwum."""
    with patch('os.makedirs', side_effect=PermissionError("Test error")):
        with pytest.raises(PermissionError, match="Brak uprawnień do zapisu w katalogu archiwum: Test error"):
            logger.archive_logs()


def test_get_logger():
    """Test funkcji get_logger."""
    logger = get_logger("test_module")
    assert logger.name == "test_module"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2  # console_handler i file_handler
    
    # Sprawdź czy drugi wywołanie nie dodaje nowych handlerów
    logger2 = get_logger("test_module")
    assert len(logger2.handlers) == 2


def test_basic_logging_methods(logger):
    """Test podstawowych metod logowania."""
    test_message = "Test message"
    
    logger.info(test_message)
    logger.error(test_message)
    logger.warning(test_message)
    logger.debug(test_message)
    
    # Sprawdź czy plik z logami zawiera odpowiednie dane
    log_path = os.path.join(logger.logs_dir, logger.log_file)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "INFO" in log_content
        assert "ERROR" in log_content
        assert "WARNING" in log_content
        assert "DEBUG" in log_content
        assert test_message in log_content


def test_private_path_methods(logger):
    """Test prywatnych metod zwracających ścieżki."""
    # Test _get_log_file_path
    log_path = logger._get_log_file_path("test.log")
    assert log_path == os.path.join(logger.logs_dir, "test.log")
    
    # Test _get_json_file_path
    json_path = logger._get_json_file_path()
    assert json_path == os.path.join(logger.logs_dir, logger.json_log_file)
    
    # Test _get_archive_dir_path
    archive_path = logger._get_archive_dir_path()
    assert archive_path == os.path.join(logger.logs_dir, "archive")


@pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_logger_levels(tmp_path, log_level):
    """Test różnych poziomów logowania."""
    logger = TradingLogger(
        strategy_name="test_strategy",
        logs_dir=str(tmp_path)
    )
    
    logger.logger.setLevel(getattr(logging, log_level))
    test_message = f"Test {log_level} message"
    
    getattr(logger, log_level.lower())(test_message)
    
    log_path = os.path.join(logger.logs_dir, logger.log_file)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert test_message in log_content
        assert log_level in log_content


@patch('builtins.open', side_effect=PermissionError("Brak uprawnień"))
def test_log_trade_to_json_permission_error(mock_open, logger, sample_trade_info):
    """Test obsługi błędu uprawnień przy zapisie do JSON."""
    with pytest.raises(IOError, match="Błąd podczas zapisu do pliku JSON: Brak uprawnień"):
        logger.log_trade_to_json(sample_trade_info)


@patch('json.dump', side_effect=TypeError("Invalid type"))
def test_log_trade_to_json_type_error(mock_dump, logger, sample_trade_info):
    """Test obsługi błędu typu przy zapisie do JSON."""
    with pytest.raises(IOError, match="Błąd podczas zapisu do pliku JSON: Invalid type"):
        logger.log_trade_to_json(sample_trade_info)


def test_log_trade_with_missing_fields(logger):
    """Test logowania transakcji z brakującymi polami."""
    trade_info = {}  # Pusty słownik
    logger.log_trade(trade_info)
    
    # Sprawdź czy plik z logami zawiera wartości domyślne
    log_path = os.path.join(logger.logs_dir, logger.trades_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "Symbol: N/A" in log_content
        assert "Typ: N/A" in log_content
        assert "Wolumen: 0" in log_content


def test_log_ai_analysis_with_missing_fields(logger):
    """Test logowania analizy AI z brakującymi polami."""
    analysis = {}  # Pusty słownik
    logger.log_ai_analysis(analysis)
    
    # Sprawdź czy plik z logami zawiera wartości domyślne
    log_path = os.path.join(logger.logs_dir, logger.ai_log)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "Symbol: N/A" in log_content
        assert "Sygnał: N/A" in log_content
        assert "Pewność: 0%" in log_content


def test_log_trade_with_invalid_json(logger, sample_trade_info):
    """Test logowania transakcji z nieprawidłowym formatem JSON."""
    # Dodaj pole, które nie może być zserializowane do JSON
    sample_trade_info['invalid'] = set()  # set nie może być zserializowany do JSON
    
    logger.log_trade(sample_trade_info)
    
    # Sprawdź czy błąd został zalogowany
    error_log_path = os.path.join(logger.logs_dir, logger.errors_log)
    with open(error_log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "Błąd podczas zapisu do pliku JSON" in log_content


def test_archive_logs_with_existing_archive(logger, sample_trade_info):
    """Test archiwizacji logów gdy katalog archiwum już istnieje."""
    # Najpierw utwórz jakieś logi
    logger.log_trade(sample_trade_info)
    logger.log_warning("Test warning")
    
    # Wykonaj pierwszą archiwizację
    logger.archive_logs()
    
    # Dodaj nowe logi
    logger.log_trade(sample_trade_info)
    logger.log_warning("Another warning")
    
    # Wykonaj drugą archiwizację
    logger.archive_logs()
    
    # Sprawdź czy istnieją dwa katalogi archiwum
    archive_dir = logger._get_archive_dir_path()
    archive_dirs = [d for d in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, d))]
    assert len(archive_dirs) == 2


def test_log_trade_to_json_with_existing_data(logger, sample_trade_info):
    """Test logowania transakcji do JSON gdy plik już zawiera dane."""
    # Najpierw zapisz jedną transakcję
    logger.log_trade_to_json(sample_trade_info)
    
    # Zmodyfikuj dane i zapisz drugą transakcję
    modified_trade = sample_trade_info.copy()
    modified_trade['price'] = 1.2345
    logger.log_trade_to_json(modified_trade)
    
    # Sprawdź czy plik zawiera obie transakcje
    json_path = logger._get_json_file_path()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["price"] != data[1]["price"] 