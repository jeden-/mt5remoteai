ZADANIE #3 - Implementacja monitoringu i testów

1. Utwórz nowy plik src/utils/logger.py:
```python
import logging
from datetime import datetime
import os

class TradingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Logger dla operacji handlowych
        self.trading_logger = self._setup_logger(
            'trading',
            os.path.join(log_dir, f'trading_{datetime.now().strftime("%Y%m%d")}.log')
        )
        
        # Logger dla AI
        self.ai_logger = self._setup_logger(
            'ai',
            os.path.join(log_dir, f'ai_{datetime.now().strftime("%Y%m%d")}.log')
        )
        
    def _setup_logger(self, name: str, log_file: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        return logger
        
    def log_trade(self, trade_info: dict):
        self.trading_logger.info(f"Trade executed: {trade_info}")
        
    def log_ai_analysis(self, model: str, analysis: str):
        self.ai_logger.info(f"AI Analysis ({model}): {analysis}")
        
    def log_error(self, error_msg: str):
        self.trading_logger.error(f"Error: {error_msg}")
        self.ai_logger.error(f"Error: {error_msg}")

2. Utwórz nowy plik src/utils/prompts.py:
class TradingPrompts:
    @staticmethod
    def get_market_analysis_prompt(data: dict) -> str:
        return f"""
        Przeanalizuj następujące dane rynkowe dla {data['symbol']}:
        
        DANE TECHNICZNE:
        - Aktualna cena: {data['current_price']}
        - SMA20: {data['sma_20']}
        - SMA50: {data['sma_50']}
        - Zmiana 24h: {data['price_change_24h']}%
        - Wolumen 24h: {data['volume_24h']}
        
        Wymagam konkretnej odpowiedzi zawierającej:
        1. Kierunek trendu (UP/DOWN/SIDEWAYS)
        2. Siła trendu (1-10)
        3. Rekomendacja (BUY/SELL/WAIT)
        4. Sugerowany SL (w pips)
        5. Sugerowany TP (w pips)
        
        Format odpowiedzi:
        TREND_DIRECTION: [kierunek]
        TREND_STRENGTH: [siła]
        RECOMMENDATION: [rekomendacja]
        SL_PIPS: [liczba]
        TP_PIPS: [liczba]
        REASONING: [krótkie uzasadnienie]
        """
    
    @staticmethod
    def get_risk_analysis_prompt(data: dict) -> str:
        return f"""
        Oceń ryzyko dla planowanej transakcji:
        
        PARAMETRY:
        - Instrument: {data['symbol']}
        - Typ: {data['action']}
        - Wielkość pozycji: {data['position_size']}
        - Aktualne saldo: {data['balance']}
        - Stop Loss: {data['stop_loss']}
        - Take Profit: {data['take_profit']}
        
        Wymagana odpowiedź:
        1. Ocena ryzyka (LOW/MEDIUM/HIGH)
        2. Risk/Reward Ratio
        3. % kapitału na ryzyku
        4. Rekomendacja (PROCEED/ADJUST/ABORT)
        
        Format odpowiedzi:
        RISK_LEVEL: [poziom]
        RR_RATIO: [liczba]
        CAPITAL_AT_RISK: [procent]
        RECOMMENDATION: [rekomendacja]
        REASONING: [krótkie uzasadnienie]
        """
3. Utwórz nowy plik tests/test_strategy.py:
import pytest
import asyncio
from src.strategies.basic_strategy import BasicStrategy
from src.utils.config import Config

@pytest.fixture
async def setup_strategy():
    config = Config.load_config()
    # Mock connectors for testing
    mt5_connector = MockMT5Connector()
    ollama_connector = MockOllamaConnector()
    anthropic_connector = MockAnthropicConnector()
    db_handler = MockDBHandler()
    
    strategy_config = {
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'allowed_symbols': ['EURUSD']
    }
    
    strategy = BasicStrategy(
        mt5_connector=mt5_connector,
        ollama_connector=ollama_connector,
        anthropic_connector=anthropic_connector,
        db_handler=db_handler,
        config=strategy_config
    )
    
    return strategy

@pytest.mark.asyncio
async def test_market_analysis(setup_strategy):
    strategy = await setup_strategy
    analysis = await strategy.analyze_market('EURUSD')
    
    assert 'market_data' in analysis
    assert 'technical_indicators' in analysis
    assert 'ollama_analysis' in analysis
    assert 'claude_analysis' in analysis

@pytest.mark.asyncio
async def test_signal_generation(setup_strategy):
    strategy = await setup_strategy
    analysis = {
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1000,
            'sma_20': 1.0990,
            'sma_50': 1.0980,
            'price_change_24h': 0.5,
            'volume_24h': 10000
        },
        'technical_indicators': {
            'sma_20': 1.0990,
            'sma_50': 1.0980
        },
        'ollama_analysis': 'kupuj',
        'claude_analysis': 'long'
    }
    
    signals = await strategy.generate_signals(analysis)
    
    assert 'action' in signals
    assert 'entry_price' in signals
    assert 'stop_loss' in signals
    assert 'take_profit' in signals

4. Zaktualizuj main.py o nowe komponenty:
import asyncio
from src.utils.config import Config
from src.utils.logger import TradingLogger
from src.database.postgres_handler import PostgresHandler
from src.connectors.mt5_connector import MT5Connector
from src.connectors.ollama_connector import OllamaConnector
from src.connectors.anthropic_connector import AnthropicConnector
from src.strategies.basic_strategy import BasicStrategy
from src.utils.prompts import TradingPrompts

async def main():
    # Inicjalizacja loggera
    logger = TradingLogger()
    
    try:
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
        
        while True:
            for symbol in strategy_config['allowed_symbols']:
                try:
                    # Analiza rynku
                    analysis = await strategy.analyze_market(symbol)
                    logger.log_ai_analysis('Combined', str(analysis))
                    
                    # Generowanie sygnałów
                    signals = await strategy.generate_signals(analysis)
                    
                    if signals['action'] != 'WAIT':
                        # Wykonanie transakcji
                        result = await strategy.execute_signals(signals)
                        if result:
                            logger.log_trade(result)
                    
                except Exception as e:
                    logger.log_error(f"Error processing {symbol}: {str(e)}")
                
            # Czekaj 5 minut przed kolejną iteracją
            await asyncio.sleep(300)
            
    except KeyboardInterrupt:
        logger.log_trade("System stopped by user")
    except Exception as e:
        logger.log_error(f"Critical error: {str(e)}")
    finally:
        mt5_connector.disconnect()
        db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
