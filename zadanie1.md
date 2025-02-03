ZADANIE #1 - Rozbudowa struktury projektu

Lokalizacja projektu: C:\Users\win\Documents\mt5remoteai\

0. Najpierw utwórz i aktywuj środowisko wirtualne:
```bash
# Utworzenie środowiska wirtualnego
python -m venv .venv

# Aktywacja środowiska (Windows)
.venv\Scripts\activate

ZADANIE #1 - Utworzenie podstawowej struktury projektu

1. Utwórz nowy projekt Python z następującą strukturą katalogów:

mt5remoteai/ (już tu jesteśmy, nie musisz go tworzyć)
├── src/
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── mt5_connector.py
│   │   ├── ollama_connector.py
│   │   └── anthropic_connector.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── postgres_handler.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── base_strategy.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── data_models.py
│   └── utils/
│       ├── __init__.py
│       └── config.py
├── tests/
│   └── __init__.py
├── requirements.txt
└── main.py

2. W pliku requirements.txt dodaj podstawowe zależności:
- fastapi
- uvicorn
- MetaTrader5
- psycopg2-binary
- anthropic
- pandas
- numpy
- python-dotenv

3. W pliku src/connectors/mt5_connector.py zaimplementuj podstawową klasę MT5Connector z metodami:
- __init__ (inicjalizacja połączenia)
- connect (nawiązanie połączenia z MT5)
- disconnect (zamknięcie połączenia)
- get_account_info (pobieranie informacji o koncie)
- get_symbols (lista dostępnych instrumentów)

4. W pliku src/database/postgres_handler.py zaimplementuj klasę PostgresHandler z metodami:
- __init__ (parametry połączenia)
- connect (nawiązanie połączenia)
- disconnect (zamknięcie połączenia)
- create_tables (utworzenie podstawowych tabel)
- save_market_data (zapisywanie danych rynkowych)

5. W pliku src/utils/config.py utwórz klasę Config do obsługi zmiennych środowiskowych:
- parametry połączenia do MT5
- parametry połączenia do PostgreSQL
- klucze API dla Anthropic

Używaj typów (type hints) i docstringów. Kod powinien być zgodny z PEP 8.
