"""Główny moduł aplikacji MT5 Remote AI."""

import asyncio
from datetime import datetime, timedelta
from loguru import logger

from .config import Config
from .utils.logger import TradingLogger
from .strategies.basic_strategy import BasicStrategy
from .backtest.backtester import Backtester
from .backtest.visualizer import BacktestVisualizer
from .database.postgres_handler import PostgresHandler

async def run_backtest():
    """Uruchamia backtest strategii."""
    logger = TradingLogger()
    config = Config.load_config()
    
    # Inicjalizacja połączenia z bazą
    db_handler = PostgresHandler(config)
    await db_handler.create_pool()
    await db_handler.create_tables()
    
    try:
        # Inicjalizacja strategii
        strategy = BasicStrategy(
            symbol='EURUSD',
            timeframe='1H',
            sma_fast=20,
            sma_slow=50,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
        
        # Utworzenie backtestera z obsługą bazy
        backtester = Backtester(
            strategy=strategy,
            symbol='EURUSD',
            timeframe='1H',
            initial_capital=10000,
            start_date=datetime.now() - timedelta(days=30),
            logger=logger,
            db_handler=db_handler
        )
        
        # Uruchomienie backtestu
        results = await backtester.run_backtest()
        
        # Wizualizacja wyników
        visualizer = BacktestVisualizer(backtester.data, backtester.trades)
        visualizer.save_dashboard('backtest_results.html')
        
        logger.info("\n🎯 WYNIKI BACKTESTU:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value}")
            
    finally:
        await db_handler.close_pool()

if __name__ == "__main__":
    asyncio.run(run_backtest()) 