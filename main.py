"""
Główny moduł aplikacji MT5 Remote AI.
"""
import asyncio
from loguru import logger

from src.utils.config import Config
from src.database.postgres_handler import PostgresHandler
from src.connectors.mt5_connector import MT5Connector
from src.connectors.ollama_connector import OllamaConnector
from src.connectors.anthropic_connector import AnthropicConnector
from src.strategies.basic_strategy import BasicStrategy


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


if __name__ == "__main__":
    asyncio.run(main()) 