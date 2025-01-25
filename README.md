# NikkeiNinja 🥷

System do analizy sentymentu i automatycznego tradingu na indeksie Nikkei 225, wykorzystujący dane z mediów społecznościowych, analizę techniczną i zaawansowane modele AI.

## Funkcjonalności 🚀

- Integracja z MetaTrader 5 (MT5)
  - Pobieranie danych historycznych i aktualnych
  - Rzeczywisty wolumen transakcji (`real_volume`)
  - Synchronizacja z bazą danych i cache
- Analiza techniczna
  - Wskaźniki cenowe (RSI, MACD, SMA, EMA)
  - Wskaźniki wolumenowe (OBV, ADI, CMF, VWAP)
  - Wykrywanie formacji świecowych
  - Generowanie sygnałów handlowych
- System RAG (Retrieval Augmented Generation)
  - Lokalna baza wiedzy ChromaDB
  - Wyszukiwanie semantyczne
  - Filtrowanie po metadanych
- Scraping danych z mediów społecznościowych (Twitter, Reddit)
- Analiza sentymentu z wykorzystaniem Anthropic Claude API
- Automatyczne sugestie tradingowe (KUPUJ/SPRZEDAJ/CZEKAJ)
- Logowanie operacji i monitorowanie wydajności
- Asynchroniczne przetwarzanie danych

## Architektura 🏗️

### Moduły

- `handel/`
  - `operacje_mt5.py` - Integracja z MT5 i analiza techniczna
  - `strategie.py` - Strategie tradingowe
- `baza_danych/`
  - `modele.py` - Modele danych PostgreSQL
  - `synchronizacja.py` - Synchronizacja danych MT5
  - `cache.py` - System cachowania
- `ai/`
  - `llm_local.py` - Integracja z Anthropic API do analizy tekstu
  - `scraper_social.py` - Pobieranie danych z mediów społecznościowych
  - `system_rag.py` - System RAG oparty na ChromaDB

### Technologie

- Python 3.9+
- PostgreSQL 15+
- MetaTrader 5 (MT5)
- ChromaDB dla systemu RAG
- Anthropic Claude API (model claude-3-opus-20240229)
- pytest dla testów
- asyncio dla operacji asynchronicznych

## Wymagania 📋

- Python 3.9+
- PostgreSQL 15+
- MetaTrader 5
- Klucz API Anthropic
- Dostęp do API mediów społecznościowych

## Konfiguracja 🔧

1. Sklonuj repozytorium
2. Utwórz plik `.env` z kluczami API i konfiguracją:
```
# Baza danych
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nikkeininja
DB_USER=ninja
DB_PASSWORD=ninja

# MT5
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# Anthropic
ANTHROPIC_API_KEY=your_key_here
```

## Testy 🧪

Projekt zawiera kompleksowe testy:
- Testy jednostkowe dla analizy technicznej
- Testy integracyjne dla MT5 i bazy danych
- Testy systemu RAG i scrapera
- Mockowanie zewnętrznych API

Uruchomienie testów:
```bash
pytest
```

## Konwencje 📝

- Język: Polski (docstringi, komentarze, nazwy zmiennych)
- Type hints obowiązkowe
- Docstringi dla wszystkich klas i metod
- Emoji w logach:
  - 🥷 - operacje ninja
  - ⚠️ - ostrzeżenia
  - ❌ - błędy
- PEP 8 (max 120 znaków) 