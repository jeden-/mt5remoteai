# NikkeiNinja 🥷

System do analizy sentymentu i automatycznego tradingu na indeksie Nikkei 225, wykorzystujący dane z mediów społecznościowych i zaawansowane modele AI.

## Funkcjonalności 🚀

- Scraping danych z mediów społecznościowych (Twitter, Reddit)
- Analiza sentymentu z wykorzystaniem Anthropic Claude API
- Automatyczne sugestie tradingowe (KUPUJ/SPRZEDAJ/CZEKAJ)
- Logowanie operacji i monitorowanie wydajności
- Asynchroniczne przetwarzanie danych

## Architektura 🏗️

### Moduły

- `ai/`
  - `llm_local.py` - Integracja z Anthropic API do analizy tekstu
  - `scraper_social.py` - Pobieranie danych z mediów społecznościowych

### Technologie

- Python 3.9+
- Anthropic Claude API (model claude-3-opus-20240229)
- pytest dla testów
- asyncio dla operacji asynchronicznych

## Wymagania 📋

- Python 3.9+
- Klucz API Anthropic
- Dostęp do API mediów społecznościowych

## Konfiguracja 🔧

1. Sklonuj repozytorium
2. Utwórz plik `.env` z kluczami API:
```
ANTHROPIC_API_KEY=your_key_here
```

## Testy 🧪

Projekt zawiera kompleksowe testy:
- Testy jednostkowe dla analizy tekstu
- Testy integracyjne dla scrapera
- Mockowanie zewnętrznych API

Uruchomienie testów:
```bash
pytest
```

## Konwencje 📝

- Język: Polski (docstringi, komentarze, nazwy zmiennych)
- Type hints obowiązkowe
- Docstringi dla wszystkich klas i metod
- Emoji w logach
- PEP 8 (max 120 znaków) 