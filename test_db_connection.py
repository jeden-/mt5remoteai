import asyncio
import asyncpg
from loguru import logger
import sys

# Konfiguracja loggera
logger.remove()  # Usuwa domyślny handler
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

async def test_connection():
    """Test połączenia z bazą danych."""
    try:
        logger.info("🔄 Próba połączenia z bazą danych mt5remotetest...")
        conn = await asyncpg.connect(
            user='mt5remote',
            password='mt5remote',
            database='mt5remotetest',
            host='localhost',
            port=5432
        )
        logger.success("✅ Połączenie udane!")
        await conn.close()
        logger.info("🔄 Połączenie zamknięte")
        return True
    except Exception as e:
        logger.error(f"❌ Błąd połączenia: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection()) 