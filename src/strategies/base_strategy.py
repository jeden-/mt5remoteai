"""
Moduł zawierający bazową klasę dla strategii tradingowych.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

from ..interfaces.connectors import IMT5Connector, IOllamaConnector, IAnthropicConnector, IDBHandler


class BaseStrategy(ABC):
    """Bazowa klasa dla wszystkich strategii tradingowych."""

    def __init__(
        self,
        mt5_connector: IMT5Connector,
        ollama_connector: IOllamaConnector,
        anthropic_connector: IAnthropicConnector,
        db_handler: IDBHandler,
        config: Dict[str, Any]
    ):
        """
        Inicjalizacja strategii.

        Args:
            mt5_connector: Konektor do MT5
            ollama_connector: Konektor do Ollama
            anthropic_connector: Konektor do Claude
            db_handler: Handler bazy danych
            config: Konfiguracja strategii
        """
        self.mt5 = mt5_connector
        self.ollama = ollama_connector
        self.claude = anthropic_connector
        self.db = db_handler
        self.config = config
        
        # Podstawowe parametry strategii
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% kapitału
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% kapitału
        self.allowed_symbols = config.get('allowed_symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
        
    @abstractmethod
    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """
        Analizuje rynek dla danego symbolu.
        
        Args:
            symbol: Symbol instrumentu
            
        Returns:
            Dict z wynikami analizy
        """
        pass
        
    @abstractmethod
    async def generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generuje sygnały na podstawie analizy.
        
        Args:
            analysis: Wyniki analizy rynku
            
        Returns:
            Dict z sygnałami
        """
        pass
        
    @abstractmethod
    async def execute_signals(self, signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Wykonuje sygnały tradingowe.
        
        Args:
            signals: Sygnały do wykonania
            
        Returns:
            Dict z informacjami o wykonanej transakcji lub None
        """
        pass
        
    async def run(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Uruchamia pełny cykl strategii.
        
        Args:
            symbol: Symbol instrumentu
            
        Returns:
            Dict z informacjami o wykonanej transakcji lub None
        """
        try:
            logger.info(f"🥷 Rozpoczynanie iteracji dla {symbol}")
            
            analysis = await self.analyze_market(symbol)
            logger.debug(f"🥷 Analiza rynku dla {symbol} zakończona")
            
            signals = await self.generate_signals(analysis)
            logger.debug(f"🥷 Wygenerowano sygnały dla {symbol}: {signals['action']}")
            
            result = await self.execute_signals(signals)
            if result:
                logger.info(f"🥷 Wykonano transakcję dla {symbol}: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Błąd w strategii: {str(e)}")
            return None 