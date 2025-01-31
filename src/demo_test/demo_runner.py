"""
Moduł zawierający klasę do przeprowadzania testów demo.
"""
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
from ..utils.logger import TradingLogger
from ..strategies.basic_strategy import BasicStrategy


class DemoTestRunner:
    """Klasa odpowiedzialna za przeprowadzanie testów na koncie demo."""
    
    def __init__(
        self,
        strategy: BasicStrategy,
        logger: TradingLogger,
        config: Dict[str, Any]
    ):
        """
        Inicjalizacja runnera testów demo.
        
        Args:
            strategy: Strategia tradingowa do testowania
            logger: Logger do zapisywania wyników
            config: Konfiguracja testów
        """
        self.strategy = strategy
        self.logger = logger
        self.config = config
        self.test_results: List[Dict[str, Any]] = []
        self.start_balance: float = 0
        self.current_balance: float = 0
        
    async def initialize_test(self):
        """Inicjalizacja testu demo."""
        account_info = self.strategy.mt5.get_account_info()
        self.start_balance = account_info.get('balance', 0.0)
        if self.start_balance <= 0:
            self.start_balance = 10000.0  # Domyślne saldo jeśli nie można pobrać
        self.current_balance = self.start_balance
        self.logger.log_trade({
            'event': 'demo_test_started',
            'start_balance': self.start_balance,
            'timestamp': datetime.now()
        })
        
    async def run_single_symbol_test(self, symbol: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Przeprowadza test na pojedynczym symbolu.
        
        Args:
            symbol: Symbol do testowania
            duration_minutes: Czas trwania testu w minutach
            
        Returns:
            Dict[str, Any]: Wyniki testu
        """
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
                self.current_balance = account_info.get('balance', self.current_balance)
                
                # Czekaj 1 minutę przed kolejną iteracją
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.log_error(f"Error in demo test for {symbol}: {str(e)}")
                break
                
        # Przygotuj raport z testu
        profit = self.current_balance - self.start_balance
        profit_percentage = (profit / self.start_balance * 100) if self.start_balance > 0 else 0
        win_rate = (profitable_trades / trades_count * 100) if trades_count > 0 else 0
        
        test_results = {
            'symbol': symbol,
            'duration_minutes': duration_minutes,
            'trades_count': trades_count,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'start_balance': self.start_balance,
            'end_balance': self.current_balance,
            'profit': profit,
            'profit_percentage': profit_percentage
        }
        
        return test_results
        
    async def run_full_test(self, symbols: List[str], duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Przeprowadza testy na wszystkich symbolach.
        
        Args:
            symbols: Lista symboli do testowania
            duration_minutes: Czas trwania testu w minutach
            
        Returns:
            List[Dict[str, Any]]: Lista wyników testów
        """
        await self.initialize_test()
        self.test_results = []  # Reset wyników przed nowym testem
        
        for symbol in symbols:
            self.logger.log_trade({
                'event': 'demo_test_symbol_started',
                'symbol': symbol,
                'timestamp': datetime.now()
            })
            
            try:
                results = await self.run_single_symbol_test(symbol, duration_minutes)
            except Exception as e:
                self.logger.log_error(f"Error running test for {symbol}: {str(e)}")
                results = {
                    'symbol': symbol,
                    'duration_minutes': duration_minutes,
                    'trades_count': 0,
                    'profitable_trades': 0,
                    'win_rate': 0,
                    'start_balance': self.start_balance,
                    'end_balance': self.current_balance,
                    'profit': 0,
                    'profit_percentage': 0
                }
            
            self.test_results.append(results)
            
            self.logger.log_trade({
                'event': 'demo_test_symbol_finished',
                'symbol': symbol,
                'results': results,
                'timestamp': datetime.now()
            })
            
        return self.test_results
        
    def generate_test_report(self) -> str:
        """
        Generuje raport z testów.
        
        Returns:
            str: Raport w formacie tekstowym
        """
        report = ["=== RAPORT Z TESTÓW DEMO ===\n"]
        report.append(f"Data testu: {datetime.now()}\n")
        report.append(f"Saldo początkowe: {self.start_balance}\n")
        report.append(f"Saldo końcowe: {self.current_balance}\n")
        report.append(f"Całkowity zysk/strata: {self.current_balance - self.start_balance}\n")
        
        profit_percentage = ((self.current_balance - self.start_balance) / self.start_balance * 100) if self.start_balance > 0 else 0
        report.append(f"Procent zwrotu: {profit_percentage:.1f}%\n")
        
        for result in self.test_results:
            report.append(f"\nSymbol: {result['symbol']}")
            report.append(f"Liczba transakcji: {result['trades_count']}")
            report.append(f"Udane transakcje: {result['profitable_trades']}")
            report.append(f"Win rate: {result['win_rate']:.1f}%")
            report.append(f"Zysk: {result['profit']}")
            
        return "\n".join(report) 