ZADANIE #4 - Implementacja modułu testów demo

1. Utwórz nowy plik src/demo_test/demo_runner.py:
```python
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
from ..utils.logger import TradingLogger
from ..strategies.basic_strategy import BasicStrategy
from ..utils.config import Config

class DemoTestRunner:
    def __init__(
        self,
        strategy: BasicStrategy,
        logger: TradingLogger,
        config: Dict[str, Any]
    ):
        self.strategy = strategy
        self.logger = logger
        self.config = config
        self.test_results: List[Dict[str, Any]] = []
        self.start_balance: float = 0
        self.current_balance: float = 0
        
    async def initialize_test(self):
        """Inicjalizacja testu demo"""
        account_info = self.strategy.mt5.get_account_info()
        self.start_balance = account_info['balance']
        self.current_balance = self.start_balance
        self.logger.log_trade({
            'event': 'demo_test_started',
            'start_balance': self.start_balance,
            'timestamp': datetime.now()
        })
        
    async def run_single_symbol_test(self, symbol: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Przeprowadza test na pojedynczym symbolu"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        trades_count = 0
        profitable_trades = 0
        
        while datetime.now() < end_time:
            try:
                # Analiza rynku
                analysis = await self.strategy.analyze_market(symbol)
                self.logger.log_ai_analysis('demo_test', str(analysis))
                
                # Generowanie sygnałów
                signals = await self.strategy.generate_signals(analysis)
                
                if signals['action'] != 'WAIT':
                    # Wykonanie transakcji
                    result = await self.strategy.execute_signals(signals)
                    if result:
                        trades_count += 1
                        if result.get('profit', 0) > 0:
                            profitable_trades += 1
                        
                        self.logger.log_trade({
                            'event': 'demo_trade',
                            'symbol': symbol,
                            'result': result
                        })
                
                # Aktualizacja salda
                account_info = self.strategy.mt5.get_account_info()
                self.current_balance = account_info['balance']
                
                # Czekaj 1 minutę przed kolejną iteracją
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.log_error(f"Error in demo test for {symbol}: {str(e)}")
                
        # Przygotuj raport z testu
        test_results = {
            'symbol': symbol,
            'duration_minutes': duration_minutes,
            'trades_count': trades_count,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / trades_count if trades_count > 0 else 0,
            'start_balance': self.start_balance,
            'end_balance': self.current_balance,
            'profit': self.current_balance - self.start_balance,
            'profit_percentage': ((self.current_balance - self.start_balance) / self.start_balance) * 100
        }
        
        self.test_results.append(test_results)
        return test_results
        
    async def run_full_test(self, symbols: List[str], duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Przeprowadza testy na wszystkich symbolach"""
        await self.initialize_test()
        
        for symbol in symbols:
            self.logger.log_trade({
                'event': 'demo_test_symbol_started',
                'symbol': symbol,
                'timestamp': datetime.now()
            })
            
            results = await self.run_single_symbol_test(symbol, duration_minutes)
            
            self.logger.log_trade({
                'event': 'demo_test_symbol_finished',
                'symbol': symbol,
                'results': results,
                'timestamp': datetime.now()
            })
            
        return self.test_results
        
    def generate_test_report(self) -> str:
        """Generuje raport z testów"""
        report = ["=== RAPORT Z TESTÓW DEMO ===\n"]
        report.append(f"Data testu: {datetime.now()}\n")
        report.append(f"Saldo początkowe: {self.start_balance}\n")
        report.append(f"Saldo końcowe: {self.current_balance}\n")
        report.append(f"Całkowity zysk/strata: {self.current_balance - self.start_balance}\n")
        report.append(f"Procent zwrotu: {((self.current_balance - self.start_balance) / self.start_balance) * 100}%\n")
        
        for result in self.test_results:
            report.append(f"\nSymbol: {result['symbol']}")
            report.append(f"Liczba transakcji: {result['trades_count']}")
            report.append(f"Udane transakcje: {result['profitable_trades']}")
            report.append(f"Win rate: {result['win_rate']*100}%")
            report.append(f"Zysk: {result['profit']}")
            
        return "\n".join(report)

2. Zaktualizuj main.py o tryb demo:
import asyncio
from src.utils.config import Config
from src.utils.logger import TradingLogger
from src.database.postgres_handler import PostgresHandler
from src.connectors.mt5_connector import MT5Connector
from src.connectors.ollama_connector import OllamaConnector
from src.connectors.anthropic_connector import AnthropicConnector
from src.strategies.basic_strategy import BasicStrategy
from src.demo_test.demo_runner import DemoTestRunner

async def run_demo_test():
    logger = TradingLogger()
    config = Config.load_config()
    
    # Inicjalizacja połączeń
    db = PostgresHandler(config)
    mt5_connector = MT5Connector(config)
    ollama_connector = OllamaConnector()
    anthropic_connector = AnthropicConnector(config.ANTHROPIC_API_KEY)
    
    # Połącz z serwisami
    db.connect()
    mt5_connector.connect()
    
    strategy_config = {
        'max_position_size': 0.01,  # 1% kapitału na transakcję w trybie demo
        'max_risk_per_trade': 0.005,  # 0.5% ryzyka na transakcję
        'allowed_symbols': ['EURUSD']  # Zaczynamy od jednej pary walutowej
    }
    
    strategy = BasicStrategy(
        mt5_connector=mt5_connector,
        ollama_connector=ollama_connector,
        anthropic_connector=anthropic_connector,
        db_handler=db,
        config=strategy_config
    )
    
    demo_runner = DemoTestRunner(strategy, logger, strategy_config)
    
    try:
        # Uruchom test demo na 60 minut
        results = await demo_runner.run_full_test(['EURUSD'], duration_minutes=60)
        
        # Wygeneruj i zapisz raport
        report = demo_runner.generate_test_report()
        print("\nRAPORT Z TESTÓW DEMO:")
        print(report)
        
        # Zapisz raport do pliku
        with open('demo_test_report.txt', 'w') as f:
            f.write(report)
            
    finally:
        mt5_connector.disconnect()
        db.disconnect()

if __name__ == "__main__":
    asyncio.run(run_demo_test())

