"""
Moduł zawierający klasę do logowania operacji tradingowych.
"""
import os
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional

def get_logger(name: str) -> logging.Logger:
    """
    Tworzy i zwraca logger dla podanego modułu.
    
    Args:
        name: Nazwa modułu
        
    Returns:
        logging.Logger: Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Handler dla konsoli
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Handler dla pliku
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(logs_dir, "app.log"), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

class TradingLogger:
    """Klasa do logowania operacji tradingowych."""
    
    def __init__(
        self,
        strategy_name: str,
        logs_dir: str = "logs",
        trades_log: str = "trades.log",
        errors_log: str = "errors.log",
        ai_log: str = "ai.log",
        log_file: str = "main.log",
        json_log_file: str = "trades.json"
    ):
        """
        Inicjalizuje logger.
        
        Args:
            strategy_name: Nazwa strategii
            logs_dir: Katalog na logi
            trades_log: Nazwa pliku z logami transakcji
            errors_log: Nazwa pliku z logami błędów
            ai_log: Nazwa pliku z logami AI
            log_file: Nazwa głównego pliku logów
            json_log_file: Nazwa pliku JSON z transakcjami
        """
        self.strategy_name = strategy_name
        self.logs_dir = logs_dir
        self.trades_log = trades_log
        self.errors_log = errors_log
        self.ai_log = ai_log
        self.log_file = log_file
        self.json_log_file = json_log_file
        
        # Utwórz katalog na logi jeśli nie istnieje
        os.makedirs(logs_dir, exist_ok=True)
        
        # Konfiguracja głównego loggera
        self.logger = logging.getLogger(f"{strategy_name}_main")
        self.logger.setLevel(logging.DEBUG)
        
        # Handler dla głównego pliku logów
        main_handler = logging.FileHandler(os.path.join(logs_dir, log_file), encoding='utf-8')
        main_handler.setLevel(logging.DEBUG)
        main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        main_handler.setFormatter(main_formatter)
        self.logger.addHandler(main_handler)
        
        # Logger dla transakcji
        self.trades_logger = logging.getLogger(f"{strategy_name}_trades")
        self.trades_logger.setLevel(logging.DEBUG)
        trades_handler = logging.FileHandler(os.path.join(logs_dir, trades_log), encoding='utf-8')
        trades_handler.setLevel(logging.DEBUG)
        trades_formatter = logging.Formatter('%(asctime)s - %(message)s')
        trades_handler.setFormatter(trades_formatter)
        self.trades_logger.addHandler(trades_handler)
        
        # Logger dla błędów
        self.error_logger = logging.getLogger(f"{strategy_name}_errors")
        self.error_logger.setLevel(logging.DEBUG)
        errors_handler = logging.FileHandler(os.path.join(logs_dir, errors_log), encoding='utf-8')
        errors_handler.setLevel(logging.DEBUG)
        errors_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        errors_handler.setFormatter(errors_formatter)
        self.error_logger.addHandler(errors_handler)
        
        # Logger dla AI
        self.ai_logger = logging.getLogger(f"{strategy_name}_ai")
        self.ai_logger.setLevel(logging.DEBUG)
        ai_handler = logging.FileHandler(os.path.join(logs_dir, ai_log), encoding='utf-8')
        ai_handler.setLevel(logging.DEBUG)
        ai_formatter = logging.Formatter('%(asctime)s - %(message)s')
        ai_handler.setFormatter(ai_formatter)
        self.ai_logger.addHandler(ai_handler)
        
    def log_trade(self, trade_info: Dict[str, Any]):
        """
        Loguje informacje o transakcji.
        
        Args:
            trade_info: Słownik z informacjami o transakcji
        """
        try:
            # Formatowanie wiadomości
            msg = f"🥷 Symbol: {trade_info.get('symbol', 'N/A')} | "
            msg += f"Typ: {trade_info.get('type', 'N/A')} | "
            msg += f"Wolumen: {trade_info.get('volume', 0)} | "
            msg += f"Cena: {trade_info.get('price', 0)} | "
            msg += f"SL: {trade_info.get('sl', 'N/A')} | "
            msg += f"TP: {trade_info.get('tp', 'N/A')} | "
            msg += f"Profit: {trade_info.get('profit', 0)}"
            
            self.trades_logger.info(msg)
            
            # Logowanie do pliku JSON
            if self.json_log_file and trade_info:
                self.log_trade_to_json(trade_info)
                    
        except Exception as e:
            self.log_error(f"Błąd podczas logowania transakcji: {str(e)}")
            
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """
        Loguje błąd.
        
        Args:
            message: Wiadomość błędu
            exception: Opcjonalny obiekt wyjątku
        """
        error_msg = f"❌ {message}"
        if exception:
            error_msg += f"\nStacktrace: {str(exception)}"
        self.error_logger.error(error_msg)
        
    def log_warning(self, message: str):
        """
        Loguje ostrzeżenie.
        
        Args:
            message: Treść ostrzeżenia
        """
        self.error_logger.warning(f"⚠️ {message}")
        
    def log_ai_analysis(self, analysis: Dict[str, Any]):
        """
        Loguje wyniki analizy AI.
        
        Args:
            analysis: Słownik z wynikami analizy
        """
        try:
            # Formatowanie wiadomości
            msg = f"🤖 Symbol: {analysis.get('symbol', 'N/A')} | "
            msg += f"Sygnał: {analysis.get('signal', 'N/A')} | "
            msg += f"Pewność: {analysis.get('confidence', 0)}% | "
            msg += f"Uzasadnienie: {analysis.get('reasoning', 'N/A')}"
            
            self.ai_logger.info(msg)
        except Exception as e:
            self.log_error(f"Błąd podczas logowania analizy AI: {str(e)}")
            
    def get_logs_path(self, log_type: str) -> str:
        """
        Zwraca ścieżkę do pliku z logami.
        
        Args:
            log_type: Typ logów (trades/errors/ai)
            
        Returns:
            str: Ścieżka do pliku z logami
            
        Raises:
            ValueError: Gdy podano nieprawidłowy typ logów
        """
        if log_type == 'trades':
            return os.path.join(self.logs_dir, self.trades_log)
        elif log_type == 'errors':
            return os.path.join(self.logs_dir, self.errors_log)
        elif log_type == 'ai':
            return os.path.join(self.logs_dir, self.ai_log)
        else:
            raise ValueError(f"Nieznany typ logów: {log_type}")
            
    def clear_logs(self, log_type: Optional[str] = None):
        """
        Czyści pliki z logami.
        
        Args:
            log_type: Opcjonalny typ logów do wyczyszczenia (trades/errors/ai)
            
        Raises:
            ValueError: Gdy podano nieprawidłowy typ logów
        """
        if log_type:
            # Wyczyść tylko określony plik
            log_path = self.get_logs_path(log_type)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('')
        else:
            # Wyczyść wszystkie pliki
            for lt in ['trades', 'errors', 'ai']:
                log_path = self.get_logs_path(lt)
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write('')
                    
    def archive_logs(self):
        """
        Archiwizuje pliki z logami.
        
        Raises:
            FileNotFoundError: Gdy katalog logów nie istnieje
            PermissionError: Gdy brak uprawnień do zapisu w katalogu archiwum
            IOError: Gdy wystąpi inny błąd podczas archiwizacji
        """
        try:
            # Sprawdź czy katalog główny istnieje
            if not os.path.exists(self.logs_dir):
                raise FileNotFoundError(f"Katalog logów nie istnieje: {self.logs_dir}")
                
            # Utwórz katalog archiwum
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_base = os.path.join(self.logs_dir, 'archive')
            archive_dir = os.path.join(archive_base, timestamp)
            
            # Sprawdź czy mamy uprawnienia do zapisu w katalogu archiwum
            test_file = os.path.join(archive_base, '.test')
            try:
                if not os.path.exists(archive_base):
                    os.makedirs(archive_base)
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (IOError, OSError) as e:
                raise PermissionError(f"Brak uprawnień do zapisu w katalogu archiwum: {str(e)}")
                
            # Utwórz katalog z datą
            try:
                os.makedirs(archive_dir)
            except FileExistsError:
                # Jeśli katalog już istnieje, dodaj unikalny sufiks
                for i in range(1, 100):
                    new_archive_dir = f"{archive_dir}_{i}"
                    if not os.path.exists(new_archive_dir):
                        archive_dir = new_archive_dir
                        os.makedirs(archive_dir)
                        break
                else:
                    raise IOError("Nie można utworzyć unikalnego katalogu archiwum")
            except PermissionError as e:
                raise PermissionError(f"Brak uprawnień do utworzenia katalogu archiwum: {str(e)}")
                
            # Kopiuj pliki logów
            for log_type in ['trades', 'errors', 'ai']:
                src_path = self.get_logs_path(log_type)
                if os.path.exists(src_path):
                    dst_path = os.path.join(archive_dir, os.path.basename(src_path))
                    try:
                        shutil.copy2(src_path, dst_path)
                    except PermissionError as e:
                        raise PermissionError(f"Brak uprawnień do skopiowania pliku {src_path}: {str(e)}")
                        
            # Wyczyść oryginalne pliki
            self.clear_logs()
            
        except (FileNotFoundError, PermissionError) as e:
            raise e
        except Exception as e:
            raise IOError(f"Błąd podczas archiwizacji logów: {str(e)}")
            
    def log_strategy_stats(self, stats: Dict[str, Any]):
        """
        Loguje statystyki strategii.
        
        Args:
            stats: Słownik ze statystykami
        """
        msg = f"📊 Statystyki strategii {self.strategy_name}:\n"
        msg += f"Liczba transakcji: {stats.get('total_trades', 0)}\n"
        msg += f"Zyskowne: {stats.get('winning_trades', 0)}\n"
        msg += f"Stratne: {stats.get('losing_trades', 0)}\n"
        msg += f"Win rate: {stats.get('win_rate', 0):.2%}\n"
        msg += f"Profit factor: {stats.get('profit_factor', 0):.2f}\n"
        msg += f"Max drawdown: {stats.get('max_drawdown', 0):.2%}\n"
        msg += f"Sharpe ratio: {stats.get('sharpe_ratio', 0):.2f}\n"
        msg += f"Całkowity zysk: {stats.get('total_profit', 0):.2f}"
        
        self.logger.info(msg)
        
    def log_market_data(self, market_data: Dict[str, Any]):
        """
        Loguje dane rynkowe.
        
        Args:
            market_data: Słownik z danymi rynkowymi
        """
        msg = f"📈 {market_data.get('symbol', 'N/A')} {market_data.get('timeframe', 'N/A')}\n"
        msg += f"O: {market_data.get('open', 0):.5f} "
        msg += f"H: {market_data.get('high', 0):.5f} "
        msg += f"L: {market_data.get('low', 0):.5f} "
        msg += f"C: {market_data.get('close', 0):.5f} "
        msg += f"V: {market_data.get('volume', 0)}"
        
        self.logger.debug(msg)
        
    def log_signal(self, signal: Dict[str, Any]):
        """
        Loguje sygnał tradingowy.
        
        Args:
            signal: Słownik z informacjami o sygnale
        """
        try:
            # Formatowanie wiadomości
            msg = f"🎯 Sygnał {signal.get('direction', 'N/A')} dla {signal.get('symbol', 'N/A')}\n"
            msg += f"Siła: {int(signal.get('strength', 0)*100)}%\n"
            
            # Dodaj informacje o wskaźnikach
            indicators = signal.get('indicators', {})
            if indicators:
                msg += "Wskaźniki:\n"
                for name, value in indicators.items():
                    msg += f"- {name.upper()}: {value}\n"
            
            self.logger.info(msg)
        except Exception as e:
            self.log_error(f"Błąd podczas logowania sygnału: {str(e)}")
            
    def log_performance(self, metrics: Dict[str, Any]):
        """
        Loguje metryki wydajności.
        
        Args:
            metrics: Słownik z metrykami wydajności
        """
        msg = f"⚡ Metryki wydajności:\n"
        msg += f"Czas wykonania: {metrics.get('execution_time', 0):.3f}s\n"
        msg += f"Użycie pamięci: {metrics.get('memory_usage', 0):.1f}MB\n"
        msg += f"Użycie CPU: {metrics.get('cpu_usage', 0):.1f}%"
        
        self.logger.debug(msg)
        
    def _get_log_file_path(self, filename: str) -> str:
        """
        Zwraca pełną ścieżkę do pliku logów.
        
        Args:
            filename: Nazwa pliku
            
        Returns:
            str: Pełna ścieżka do pliku
        """
        return os.path.join(self.logs_dir, filename)
    
    def _get_json_file_path(self) -> str:
        """
        Zwraca ścieżkę do pliku JSON z logami transakcji.
        
        Returns:
            str: Ścieżka do pliku JSON
        """
        return os.path.join(self.logs_dir, self.json_log_file)
    
    def _get_archive_dir_path(self) -> str:
        """
        Zwraca pełną ścieżkę do katalogu archiwum.
        
        Returns:
            str: Pełna ścieżka do katalogu archiwum
        """
        return os.path.join(self.logs_dir, "archive")
    
    def log_trade_to_json(self, trade_info: Dict[str, Any]) -> None:
        """
        Zapisuje informacje o transakcji do pliku JSON.
        
        Args:
            trade_info: Słownik z informacjami o transakcji
            
        Raises:
            IOError: Gdy wystąpi błąd podczas zapisu do pliku
        """
        json_path = self._get_json_file_path()
        trade_info["timestamp"] = datetime.now().isoformat()
        
        trades = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
            except json.JSONDecodeError as e:
                error_msg = f"Błąd parsowania pliku JSON: {str(e)}"
                self.log_error(error_msg)
                raise IOError(error_msg)
                
        trades.append(trade_info)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(trades, f, indent=4, ensure_ascii=False)
        except PermissionError:
            error_msg = "Błąd podczas zapisu do pliku JSON: Brak uprawnień"
            self.log_error(error_msg)
            raise IOError(error_msg)
        except TypeError:
            error_msg = "Błąd podczas zapisu do pliku JSON: Invalid type"
            self.log_error(error_msg)
            raise IOError(error_msg)
        except IOError as e:
            error_msg = f"Błąd podczas zapisu do pliku JSON: {str(e)}"
            self.log_error(error_msg)
            raise IOError(error_msg)

    def info(self, message: str):
        """Loguje informację."""
        self.logger.info(message)
        
    def error(self, message: str):
        """Loguje błąd."""
        self.logger.error(message)
        
    def warning(self, message: str):
        """Loguje ostrzeżenie."""
        self.logger.warning(message)
        
    def debug(self, message: str):
        """Loguje wiadomość debug."""
        self.logger.debug(message)
        
    def critical(self, message: str):
        """Loguje błąd krytyczny."""
        self.logger.critical(message) 