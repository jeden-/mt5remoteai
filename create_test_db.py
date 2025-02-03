import asyncio
import asyncpg

async def create_test_db():
    """Tworzy bazę danych testową."""
    try:
        # Połącz się z bazą postgres aby utworzyć nową bazę
        sys_conn = await asyncpg.connect(
            user='postgres',
            password='mt5remote',
            database='postgres',
            host='localhost'
        )
        
        # Sprawdź czy baza już istnieje
        exists = await sys_conn.fetchval("""
            SELECT 1 FROM pg_database WHERE datname = 'mt5remotetest'
        """)
        
        if not exists:
            await sys_conn.execute('CREATE DATABASE mt5remotetest')
            print('✅ Utworzono bazę testową mt5remotetest')
        else:
            print('ℹ️ Baza testowa mt5remotetest już istnieje')
            
        await sys_conn.close()
        
        # Połącz się z nową bazą i utwórz schemat
        conn = await asyncpg.connect(
            user='postgres',
            password='mt5remote',
            database='mt5remotetest',
            host='localhost'
        )
        
        # Utwórz tabele
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DECIMAL NOT NULL,
                high DECIMAL NOT NULL,
                low DECIMAL NOT NULL,
                close DECIMAL NOT NULL,
                volume DECIMAL NOT NULL,
                UNIQUE (symbol, timestamp)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                order_type VARCHAR(10) NOT NULL,
                volume DECIMAL NOT NULL,
                price DECIMAL NOT NULL,
                sl DECIMAL,
                tp DECIMAL,
                open_time TIMESTAMP NOT NULL,
                close_time TIMESTAMP,
                status VARCHAR(10) NOT NULL,
                profit DECIMAL
            );

            CREATE TABLE IF NOT EXISTS historical_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DECIMAL NOT NULL,
                high DECIMAL NOT NULL,
                low DECIMAL NOT NULL,
                close DECIMAL NOT NULL,
                volume DECIMAL NOT NULL,
                UNIQUE (symbol, timeframe, timestamp)
            );
        """)
        
        print('✅ Utworzono tabele w bazie testowej')
        await conn.close()
        
    except Exception as e:
        print(f'❌ Błąd: {str(e)}')

if __name__ == '__main__':
    asyncio.run(create_test_db()) 