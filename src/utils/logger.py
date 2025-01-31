"""
Moduł zawierający klasę do logowania operacji tradingowych.
"""
import os
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional

class TradingLogger:
    """Klasa do logowania operacji tradingowych."""
    
    def __init__(
        self,
        logs_dir: str = 'logs',
        trades_log: str = 'trades.log',
        errors_log: str = 'errors.log',
        ai_log: str = 'ai_analysis.log'
    ):
        """
        Inicjalizacja loggera.
        
        Args:
            logs_dir: Katalog z logami
            trades_log: Nazwa pliku z logami transakcji
            errors_log: Nazwa pliku z logami błędów
            ai_log: Nazwa pliku z logami analizy AI
        """
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        
        # Logger dla transakcji
        self.trades_logger = logging.getLogger('trades')
        self.trades_logger.setLevel(logging.INFO)
        trades_handler = logging.FileHandler(os.path.join(logs_dir, trades_log), encoding='utf-8')
        trades_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.trades_logger.addHandler(trades_handler)
        
        # Logger dla błędów
        self.error_logger = logging.getLogger('errors')
        self.error_logger.setLevel(logging.WARNING)
        error_handler = logging.FileHandler(os.path.join(logs_dir, errors_log), encoding='utf-8')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.error_logger.addHandler(error_handler)
        
        # Logger dla analizy AI
        self.ai_logger = logging.getLogger('ai')
        self.ai_logger.setLevel(logging.INFO)
        ai_handler = logging.FileHandler(os.path.join(logs_dir, ai_log), encoding='utf-8')
        ai_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
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
            return os.path.join(self.logs_dir, 'trades.log')
        elif log_type == 'errors':
            return os.path.join(self.logs_dir, 'errors.log')
        elif log_type == 'ai':
            return os.path.join(self.logs_dir, 'ai_analysis.log')
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
            open(log_path, 'w', encoding='utf-8').close()
        else:
            # Wyczyść wszystkie pliki
            for lt in ['trades', 'errors', 'ai']:
                log_path = self.get_logs_path(lt)
                open(log_path, 'w', encoding='utf-8').close()
                
    def archive_logs(self):
        """
        Archiwizuje pliki z logami.
        
        Raises:
            IOError: Gdy nie można utworzyć katalogu archiwum lub skopiować plików
        """
        try:
            # Utwórz katalog archiwum
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_dir = os.path.join(self.logs_dir, 'archive', timestamp)
            os.makedirs(archive_dir, exist_ok=True)
            
            # Kopiuj pliki logów
            for log_type in ['trades', 'errors', 'ai']:
                src_path = self.get_logs_path(log_type)
                if os.path.exists(src_path):
                    dst_path = os.path.join(archive_dir, os.path.basename(src_path))
                    shutil.copy2(src_path, dst_path)
                    
            # Wyczyść oryginalne pliki
            self.clear_logs()
        except (IOError, OSError) as e:
            raise IOError(f"Błąd podczas archiwizacji logów: {str(e)}") 