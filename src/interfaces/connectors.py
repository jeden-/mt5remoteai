"""
Moduł zawierający interfejsy dla konektorów.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class IMT5Connector(ABC):
    """Interfejs dla konektora MT5."""
    
    @abstractmethod
    def get_rates(self, symbol: str, timeframe: str, count: int) -> List[Dict[str, Any]]:
        """Pobiera dane historyczne."""
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Pobiera informacje o koncie."""
    
    @abstractmethod
    def place_order(self, **kwargs) -> Dict[str, Any]:
        """Składa zlecenie."""


class IOllamaConnector(ABC):
    """Interfejs dla konektora Ollama."""
    
    @abstractmethod
    async def analyze_market_data(self, market_data: Dict[str, Any], prompt_template: str) -> str:
        """Analizuje dane rynkowe."""


class IAnthropicConnector(ABC):
    """Interfejs dla konektora Anthropic."""
    
    @abstractmethod
    async def analyze_market_conditions(self, market_data: Dict[str, Any], prompt_template: str) -> str:
        """Analizuje warunki rynkowe."""


class IDBHandler(ABC):
    """Interfejs dla handlera bazy danych."""
    
    @abstractmethod
    def save_trade(self, trade_info: Dict[str, Any]) -> None:
        """Zapisuje informacje o transakcji.""" 