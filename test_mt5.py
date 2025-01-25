import MetaTrader5 as mt5
import logging
import os
from dotenv import load_dotenv

# Konfiguracja loggera
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Wyświetlenie informacji o środowisku
logger.info("Sprawdzanie zmiennych środowiskowych...")
login = int(os.getenv('MT5_LOGIN'))
haslo = os.getenv('MT5_PASSWORD')
serwer = os.getenv('MT5_SERVER')
logger.info(f"Login: {login}")
logger.info(f"Serwer: {serwer}")

# Sprawdzenie czy MT5 jest zainstalowany
logger.info("Sprawdzanie instalacji MT5...")
logger.info(f"Wersja pakietu MT5: {mt5.__version__}")

# Sprawdzenie czy terminal jest już zainicjalizowany
terminal_info = mt5.terminal_info()
if terminal_info is not None:
    logger.info("Terminal jest już zainicjalizowany")
    logger.info(f"Terminal połączony: {terminal_info.connected}")
    logger.info(f"Handel dozwolony: {terminal_info.trade_allowed}")
    logger.info(f"Ścieżka: {terminal_info.path}")
else:
    # Inicjalizacja tylko jeśli nie jest już zainicjalizowany
    logger.info("Próba inicjalizacji MT5...")
    init_status = mt5.initialize()
    logger.info(f"Status inicjalizacji: {init_status}")
    if not init_status:
        error = mt5.last_error()
        logger.error(f"Błąd inicjalizacji: {error}")
        exit(1)

logger.info("MT5 gotowy do użycia")
logger.info(f"Wersja MT5: {mt5.version()}")

# Sprawdzenie informacji o koncie
info_konto = mt5.account_info()
if info_konto is not None:
    logger.info(f"Login: {info_konto.login}")
    logger.info(f"Serwer: {info_konto.server}")
    logger.info(f"Saldo: {info_konto.balance} {info_konto.currency}")
else:
    logger.error("Nie można pobrać informacji o koncie")

# Sprawdzenie informacji o terminalu
info = mt5.terminal_info()
if info is not None:
    logger.info(f"Terminal połączony: {info.connected}")
    logger.info(f"Handel dozwolony: {info.trade_allowed}")
    logger.info(f"Ścieżka: {info.path}")
else:
    logger.error("Nie można pobrać informacji o terminalu")

logger.info("Test zakończony") 