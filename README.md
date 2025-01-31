# MT5 Remote AI 🤖

System automatycznego tradingu łączący MetaTrader 5 z modelami AI (Ollama i Claude).

## 🎯 Funkcjonalności

- Integracja z MetaTrader 5
- Analiza techniczna (SMA, EMA, RSI)
- Analiza AI z wykorzystaniem:
  - Ollama (model lokalny)
  - Claude (API Anthropic)
- Zarządzanie ryzykiem
- Automatyczne wykonywanie zleceń
- Logowanie operacji i błędów
- Baza danych PostgreSQL
- Testy jednostkowe i integracyjne

## 🚀 Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/jeden-/mt5remoteai.git
cd mt5remoteai
```

2. Stwórz i aktywuj środowisko wirtualne:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

4. Skonfiguruj zmienne środowiskowe w pliku `.env`:
```
MT5_LOGIN=twój_login
MT5_PASSWORD=twoje_hasło
MT5_SERVER=nazwa_serwera
ANTHROPIC_API_KEY=klucz_api_claude
OLLAMA_API_URL=http://localhost:11434
DB_CONNECTION_STRING=postgresql://user:pass@localhost:5432/dbname
```

## 💻 Użycie

1. Uruchom testy demo:
```bash
python main.py --mode demo --symbols EURUSD,GBPUSD
```

2. Uruchom w trybie live:
```bash
python main.py --mode live --symbols EURUSD
```

## 📊 Struktura projektu

```
src/
├── connectors/          # Konektory do zewnętrznych serwisów
├── database/           # Obsługa bazy danych
├── demo_test/         # Testy na demo
├── interfaces/        # Interfejsy
├── models/           # Modele danych
├── strategies/       # Strategie tradingowe
└── utils/           # Narzędzia pomocnicze

tests/               # Testy
```

## 🧪 Testy

Uruchom testy:
```bash
pytest tests/ -v --cov=src
```

## 📝 Licencja

MIT

## 👥 Autorzy

- [@jeden-](https://github.com/jeden-) 