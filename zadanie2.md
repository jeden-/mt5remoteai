ZADANIE #2 - Implementacja konektorów AI i podstawowej strategii

1. W pliku src/connectors/ollama_connector.py dodaj:
```python
import requests
from typing import Dict, Any
import json

class OllamaConnector:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    async def analyze_market_data(self, market_data: Dict[str, Any], prompt_template: str) -> str:
        """
        Analizuje dane rynkowe używając lokalnego modelu Ollama.
        
        Args:
            market_data: Słownik zawierający dane rynkowe
            prompt_template: Szablon promptu do analizy
            
        Returns:
            str: Odpowiedź modelu
        """
        prompt = prompt_template.format(**market_data)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": "mistral",  # lub inny model dostępny w Ollama
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Błąd Ollama API: {response.status_code}")

2. W pliku src/connectors/anthropic_connector.py dodaj:
from anthropic import Anthropic
from typing import Dict, Any
import asyncio

class AnthropicConnector:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        
    async def analyze_market_conditions(self, market_data: Dict[str, Any], prompt_template: str) -> str:
        """
        Analizuje warunki rynkowe używając Claude'a.
        
        Args:
            market_data: Słownik zawierający dane rynkowe
            prompt_template: Szablon promptu do analizy
            
        Returns:
            str: Odpowiedź Claude'a
        """
        prompt = prompt_template.format(**market_data)
        
        response = await asyncio.to_thread(
            self.client.messages.create,
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        return response.content[0].text

3. W pliku src/strategies/base_strategy.py dodaj:
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..connectors.mt5_connector import MT5Connector
from ..connectors.ollama_connector import OllamaConnector
from ..connectors.anthropic_connector import AnthropicConnector
from ..database.postgres_handler import PostgresHandler

class BaseStrategy(ABC):
    def __init__(
        self,
        mt5_connector: MT5Connector,
        ollama_connector: OllamaConnector,
        anthropic_connector: AnthropicConnector,
        db_handler: PostgresHandler,
        config: Dict[str, Any]
    ):
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
        Analizuje rynek dla danego instrumentu.
        
        Args:
            symbol: Symbol instrumentu
            
        Returns:
            Dict zawierający wyniki analizy
        """
        pass
        
    @abstractmethod
    async def generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generuje sygnały handlowe na podstawie analizy.
        
        Args:
            analysis: Wyniki analizy rynku
            
        Returns:
            Dict zawierający sygnały handlowe
        """
        pass
        
    @abstractmethod
    async def execute_signals(self, signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Wykonuje sygnały handlowe.
        
        Args:
            signals: Sygnały handlowe do wykonania
            
        Returns:
            Dict zawierający informacje o wykonanych transakcjach
        """
        pass
        
    async def run_iteration(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Wykonuje pojedynczą iterację strategii.
        
        Args:
            symbol: Symbol instrumentu
            
        Returns:
            Dict zawierający wyniki iteracji
        """
        try:
            analysis = await self.analyze_market(symbol)
            signals = await self.generate_signals(analysis)
            result = await self.execute_signals(signals)
            return result
        except Exception as e:
            print(f"Błąd podczas wykonywania strategii: {e}")
            return None

4. Utwórz nowy plik src/strategies/basic_strategy.py:
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class BasicStrategy(BaseStrategy):
    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """
        Implementacja prostej strategii analizy rynku łączącej MT5, Ollama i Claude.
        """
        # Pobierz dane rynkowe
        rates = self.mt5.get_rates(symbol, timeframe='1H', count=100)
        df = pd.DataFrame(rates)
        
        # Podstawowe wskaźniki
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Przygotuj dane do analizy przez AI
        market_data = {
            'symbol': symbol,
            'current_price': df['close'].iloc[-1],
            'sma_20': df['SMA_20'].iloc[-1],
            'sma_50': df['SMA_50'].iloc[-1],
            'price_change_24h': (df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100,
            'volume_24h': df['volume'].tail(24).sum()
        }
        
        # Analiza przez Ollama
        ollama_prompt = """
        Przeanalizuj następujące dane rynkowe dla {symbol}:
        Cena: {current_price}
        SMA20: {sma_20}
        SMA50: {sma_50}
        Zmiana 24h: {price_change_24h}%
        
        Określ krótkoterminowy trend i potencjalne poziomy wsparcia/oporu.
        """
        
        ollama_analysis = await self.ollama.analyze_market_data(market_data, ollama_prompt)
        
        # Analiza przez Claude'a
        claude_prompt = """
        Przeprowadź dogłębną analizę techniczną i fundamentalną dla {symbol}.
        
        Dane techniczne:
        - Aktualna cena: {current_price}
        - SMA20: {sma_20}
        - SMA50: {sma_50}
        - Zmiana 24h: {price_change_24h}%
        - Wolumen 24h: {volume_24h}
        
        Uwzględnij:
        1. Obecny trend
        2. Potencjalne punkty zwrotne
        3. Rekomendację pozycji (long/short/neutral)
        4. Sugerowane poziomy SL/TP
        """
        
        claude_analysis = await self.claude.analyze_market_conditions(market_data, claude_prompt)
        
        return {
            'market_data': market_data,
            'technical_indicators': {
                'sma_20': df['SMA_20'].iloc[-1],
                'sma_50': df['SMA_50'].iloc[-1]
            },
            'ollama_analysis': ollama_analysis,
            'claude_analysis': claude_analysis
        }
        
    async def generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generuje sygnały handlowe na podstawie połączonej analizy.
        """
        market_data = analysis['market_data']
        technical = analysis['technical_indicators']
        
        # Podstawowe warunki techniczne
        trend_up = technical['sma_20'] > technical['sma_50']
        
        # Interpretacja analiz AI
        signals = {
            'symbol': market_data['symbol'],
            'action': 'WAIT',  # domyślnie czekamy
            'entry_price': market_data['current_price'],
            'stop_loss': None,
            'take_profit': None,
            'position_size': None
        }
        
        # Przykładowa logika sygnałów
        if trend_up and 'kupuj' in analysis['ollama_analysis'].lower() and 'long' in analysis['claude_analysis'].lower():
            signals['action'] = 'BUY'
            signals['stop_loss'] = market_data['current_price'] * 0.995  # 0.5% SL
            signals['take_profit'] = market_data['current_price'] * 1.015  # 1.5% TP
            
        elif not trend_up and 'sprzedaj' in analysis['ollama_analysis'].lower() and 'short' in analysis['claude_analysis'].lower():
            signals['action'] = 'SELL'
            signals['stop_loss'] = market_data['current_price'] * 1.005
            signals['take_profit'] = market_data['current_price'] * 0.985
            
        if signals['action'] != 'WAIT':
            account_info = self.mt5.get_account_info()
            position_size = min(
                account_info['balance'] * self.max_position_size,
                account_info['balance'] * self.max_risk_per_trade / abs(market_data['current_price'] - signals['stop_loss'])
            )
            signals['position_size'] = position_size
            
        return signals
        
    async def execute_signals(self, signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Wykonuje wygenerowane sygnały handlowe.
        """
        if signals['action'] == 'WAIT':
            return None
            
        try:
            # Wykonaj zlecenie przez MT5
            order_result = self.mt5.place_order(
                symbol=signals['symbol'],
                order_type=signals['action'],
                volume=signals['position_size'],
                price=signals['entry_price'],
                sl=signals['stop_loss'],
                tp=signals['take_profit']
            )
            
            # Zapisz do bazy danych
            self.db.save_trade(order_result)
            
            return order_result
        except Exception as e:
            print(f"Błąd podczas wykonywania zlecenia: {e}")
            return None

6. Zaktualizuj main.py:
import asyncio
from src.utils.config import Config
from src.database.postgres_handler import PostgresHandler
from src.connectors.mt5_connector import MT5Connector
from src.connectors.ollama_connector import OllamaConnector
from src.connectors.anthropic_connector import AnthropicConnector
from src.strategies.basic_strategy import BasicStrategy

async def main():
    # Wczytaj konfigurację
    config = Config.load_config()
    
    # Inicjalizacja połączeń
    db = PostgresHandler(config)
    mt5_connector = MT5Connector(config)
    ollama_connector = OllamaConnector()
    anthropic_connector = AnthropicConnector(config.ANTHROPIC_API_KEY)
    
    # Połącz z serwisami
    db.connect()
    mt5_connector.connect()
    
    # Konfiguracja strategii
    strategy_config = {
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'allowed_symbols': ['EURUSD', 'GBPUSD', 'USDJPY']
    }
    
    strategy = BasicStrategy(
        mt5_connector=mt5_connector,
        ollama_connector=ollama_connector,
        anthropic_connector=anthropic_connector,
        db_handler=db,
        config=strategy_config
    )
    
    try:
        while True:
            for symbol in strategy_config['allowed_symbols']:
                result = await strategy.run_iteration(symbol)
                if result:
                    print(f"Wykonano transakcję: {result}")
                
            # Czekaj 5 minut przed kolejną iteracją
            await asyncio.sleep(300)
            
    except KeyboardInterrupt:
        print("Zatrzymywanie systemu...")
    finally:
        mt5_connector.disconnect()
        db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
