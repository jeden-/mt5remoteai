"""
Główny moduł aplikacji MT5 Remote AI.
"""
import asyncio
from datetime import datetime, timedelta
from loguru import logger

from src.utils.config import Config
from src.database.postgres_handler import PostgresHandler
from src.connectors.mt5_connector import MT5Connector
from src.connectors.ollama_connector import OllamaConnector
from src.connectors.anthropic_connector import AnthropicConnector
from src.strategies.basic_strategy import BasicStrategy
from src.backtest.backtester import Backtester
from src.backtest.visualizer import BacktestVisualizer


async def main() -> None:
    """Główna funkcja aplikacji."""
    logger.info("🥷 Uruchamianie systemu tradingowego...")

    # Wczytaj konfigurację
    config = Config.load_config()
    
    # Inicjalizacja połączeń
    logger.info("🥷 Inicjalizacja połączeń...")
    
    db = PostgresHandler(config)
    mt5_connector = MT5Connector(config)
    ollama_connector = OllamaConnector()
    anthropic_connector = AnthropicConnector(config.ANTHROPIC_API_KEY)
    
    # Połącz z serwisami
    try:
        db.connect()
        db.create_tables()
        
        if not mt5_connector.connect():
            logger.error("❌ Nie udało się połączyć z MT5")
            db.disconnect()
            return
            
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
        
        logger.info("🥷 System został zainicjalizowany")
        
        try:
            while True:
                for symbol in strategy_config['allowed_symbols']:
                    result = await strategy.run_iteration(symbol)
                    if result:
                        logger.info(f"🥷 Wykonano transakcję: {result}")
                    
                # Czekaj 5 minut przed kolejną iteracją
                logger.info("🥷 Oczekiwanie 5 minut przed kolejną iteracją...")
                await asyncio.sleep(300)
                
        except KeyboardInterrupt:
            logger.info("🥷 Otrzymano sygnał zatrzymania...")
            
    except Exception as e:
        logger.error(f"❌ Wystąpił błąd: {str(e)}")
        
    finally:
        # Zamknięcie połączeń
        logger.info("🥷 Zamykanie połączeń...")
        mt5_connector.disconnect()
        db.disconnect()
        logger.info("🥷 System został zatrzymany")


async def run_backtest():
    """Uruchamia backtest strategii."""
    logger.info("🥷 Rozpoczynam backtest strategii")
    
    # Inicjalizacja loggera i konfiguracji
    trading_logger = TradingLogger()
    config = Config.load_config()
    
    # Inicjalizacja strategii
    strategy = BasicStrategy(
        mt5_connector=None,  # Nie potrzebujemy połączenia w backteście
        ollama_connector=None,
        anthropic_connector=None,
        db_handler=None,
        config={
            'max_position_size': 0.1,
            'max_risk_per_trade': 0.02,
            'allowed_symbols': ['EURUSD']
        }
    )
    
    # Utworzenie backtestera
    backtester = Backtester(
        strategy=strategy,
        symbol='EURUSD',
        timeframe='1H',
        initial_capital=10000,
        start_date=datetime.now() - timedelta(days=30),
        logger=trading_logger
    )
    
    try:
        # Uruchomienie backtestu
        results = await backtester.run_backtest()
        
        # Wizualizacja wyników
        visualizer = BacktestVisualizer(backtester.data, backtester.trades)
        visualizer.save_dashboard('backtest_results.html')
        
        logger.info("\n📊 WYNIKI BACKTESTU:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value}")
            
    except Exception as e:
        logger.error(f"❌ Błąd podczas backtestingu: {str(e)}")
        raise


if __name__ == "__main__":
    # Dodaj opcję uruchomienia backtestu
    import argparse
    parser = argparse.ArgumentParser(description="MT5 Remote AI Trading System")
    parser.add_argument("--mode", choices=["live", "demo", "backtest"], default="demo", help="Tryb działania")
    parser.add_argument("--symbols", type=str, default="EURUSD", help="Symbole do handlu (oddzielone przecinkami)")
    args = parser.parse_args()
    
    if args.mode == "backtest":
        asyncio.run(run_backtest())
    else:
        asyncio.run(main()) 