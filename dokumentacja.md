NIKKEININJA 🥷 - DOKUMENTACJA PROJEKTOWA
======================================

1. INFORMACJE PODSTAWOWE
-----------------------
Nazwa: NikkeiNinja
Wersja: 0.1.0
Środowisko: Windows 10 Pro
Framework: Python 3.11+
Trading Platform: MetaTrader 5

2. OPIS PROJEKTU
---------------
System automatycznego tradingu na JP225 (Nikkei), wykorzystujący metodologię Wyckoffa,
wspierany przez sztuczną inteligencję i lokalny system RAG.

3. ŚRODOWISKO DEWELOPERSKIE
--------------------------
- System: Windows 10 Pro (stabilniejszy niż Win11)
- IDE: Cursor AI
- Model AI: claude-3-sonnet
- Python: 3.11+
- Git do wersjonowania

4. STRUKTURA PROJEKTU
--------------------
nikkeininja/
├── rdzen/              # Główne komponenty systemu
│   ├── polaczenie_mt5.py
│   ├── analiza_rynku.py
│   └── zarzadzanie_pozycjami.py
├── strategie/
│   ├── wyckoff.py
│   └── rozpoznawanie_wzorcow.py
├── ai/
│   ├── system_rag.py
│   └── uczenie_rynku.py
├── ryzyko/
│   ├── zarzadzanie_ryzykiem.py
│   └── rozmiar_pozycji.py
├── narzedzia/
│   ├── logowanie.py
│   └── konfiguracja.py
├── interfejs/
│   ├── dashboard.py
│   └── kontrolki.py
├── testy/
│   └── test_rdzen.py
└── config/
    ├── ustawienia.yaml
    └── logowanie.yaml

5. ZALEŻNOŚCI (requirements.txt)
------------------------------
MetaTrader5==5.0.45
pandas==2.1.4
numpy==1.26.2
pytz==2023.3
python-dotenv==1.0.0
fastapi==0.109.0
uvicorn==0.27.0
chromadb==0.4.22
pydantic==2.5.3
pytest==7.4.4
pytest-asyncio==0.23.3
python-logging==0.4.9.6
PyYAML==6.0.1

6. GŁÓWNE KOMPONENTY
-------------------
a) Integracja MT5
   - Połączenie z platformą
   - Pobieranie danych
   - Wykonywanie zleceń
   - Zarządzanie kontem

b) Analiza Rynku
   - Implementacja Wyckoffa
   - Rozpoznawanie wzorców
   - Analiza wieloramkowa
   - Analiza wolumenu

c) System AI
   - Lokalny RAG
   - Uczenie się wzorców
   - Adaptacja strategii
   - Analiza kontekstu

d) Zarządzanie Ryzykiem
   - Sizing pozycji
   - Stop-loss
   - Take-profit
   - Trailing stop

7. KONWENCJE KODU
----------------
- Język: Polski w nazwach
- Asyncio dla operacji asynchronicznych
- Type hints
- Docstringi
- Obsługa błędów
- Logowanie

8. PRZYKŁADOWA STRUKTURA KLASY
-----------------------------
```python
class KomponentNinja:
    """Bazowy szablon komponentu systemu NikkeiNinja."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._wczytaj_config()
        
    async def inicjalizuj(self) -> Dict[str, Any]:
        """Inicjalizacja komponentu z walidacją."""
        try:
            return {'sukces': True, 'komunikat': 'Komponent zainicjalizowany'}
        except Exception as e:
            self.logger.error(f"Błąd inicjalizacji: {str(e)}")
            return {'sukces': False, 'błąd': str(e)}

9. KONWENCJE COMMITÓW
nowa: Nowa funkcjonalność
popr: Poprawka błędu
dok: Dokumentacja
styl: Formatowanie
ref: Refaktoryzacja
test: Testy
admin: Administracja

10. KONFIGURACJA ŚRODOWISKA
# Tworzenie środowiska
python -m venv venv

# Aktywacja (Windows)
venv\Scripts\activate

# Instalacja zależności
pip install -r requirements.txt

11. PLIKI KONFIGURACYJNE
.env:

MT5_LOGIN=login
MT5_PASSWORD=haslo
MT5_SERVER=server
DEBUG=True

ustawienia.yaml:

system:
  nazwa: "NikkeiNinja"
  wersja: "0.1.0"
  debug: true

trading:
  symbol: "JP225"
  timeframes: ["M1", "M5", "M15", "H1"]
  max_pozycji: 3
  
ryzyko:
  max_ryzyko_trade: 0.02
  max_ryzyko_dzienne: 0.06

12. WYMAGANIA SPRZĘTOWE
Procesor: Intel Core i5 lub lepszy
RAM: 16GB (minimum)
Dysk: SSD 128GB (minimum)
System: Windows 10 Pro

13. KOLEJNE KROKI
Konfiguracja środowiska
Implementacja połączenia z MT5
Podstawowa analiza danych
System zarządzania ryzykiem
Implementacja Wyckoffa
System RAG
Interface użytkownika

14. ANALIZA SOCIAL MEDIA

### 14.1 Integracja z snscrape

System wykorzystuje bibliotekę snscrape do monitorowania mediów społecznościowych w poszukiwaniu wzmianek o Nikkei225. Główne funkcje:

- Asynchroniczne pobieranie danych z Twitter/X i LinkedIn
- Analiza sentymentu wypowiedzi
- System alertów dla istotnych wzmianek
- Integracja z systemem RAG do adaptacji strategii

### 14.2 Monitorowane źródła

#### Twitter/X
- Hashtagi: #Nikkei225, #NikkeiIndex, #JapanStocks
- Kluczowe słowa: "Nikkei 225", "Japanese market"
- Wpływowi analitycy i traderzy

#### LinkedIn
- Posty ekspertów rynkowych
- Analizy firm inwestycyjnych
- Raporty makroekonomiczne

### 14.3 Konfiguracja scrapera

```yaml
scraper_social:
  twitter:
    limit_wzmianek: 100
    interval_aktualizacji: 300  # sekundy
    min_followers: 1000
  linkedin:
    limit_wzmianek: 50
    interval_aktualizacji: 600
  alerty:
    prog_istotnosci: 0.8
    prog_sentymentu: 0.6
  rag:
    prog_istotnosci: 0.7
```

### 14.4 Wykorzystanie danych

1. Analiza sentymentu rynku
   - Ocena nastrojów inwestorów
   - Identyfikacja punktów zwrotnych
   - Wykrywanie skrajnych emocji

2. Adaptacja strategii
   - Dostosowanie parametrów na podstawie sentymentu
   - Korelacja z analizą techniczną
   - Walidacja sygnałów

3. System alertów
   - Powiadomienia o istotnych wzmiankach
   - Monitoring wpływowych opinii
   - Ostrzeżenia o potencjalnych zagrożeniach

### 14.5 Integracja z RAG

System RAG wykorzystuje dane z mediów społecznościowych do:

1. Wzbogacania bazy wiedzy
   - Nowe wzorce rynkowe
   - Korelacje sentymentu z ruchami cen
   - Identyfikacja kluczowych wydarzeń

2. Adaptacji strategii
   - Dostosowanie parametrów na podstawie sentymentu
   - Walidacja sygnałów technicznych
   - Optymalizacja momentów wejścia/wyjścia

### 14.6 Przykłady użycia

```python
from narzedzia.scraper_social import ScraperSocial

# Inicjalizacja scrapera
scraper = ScraperSocial(config=config)

# Pobieranie wzmianek
wzmianki = await scraper.pobierz_wzmianki('twitter', 'Nikkei225')

# Monitorowanie w czasie rzeczywistym
await scraper.monitoruj_wzmianki('Nikkei225 OR Japanese market')

# Eksport danych
scraper.eksportuj_do_csv('wzmianki.csv')
```

### 14.7 Obsługa błędów i limitów

1. Limity API
   - Automatyczne przestrzeganie limitów
   - Kolejkowanie zapytań
   - Cache wyników

2. Obsługa błędów
   - Retry dla błędów sieciowych
   - Logowanie problemów
   - Graceful degradation

3. Walidacja danych
   - Filtrowanie spamu
   - Weryfikacja źródeł
   - Kontrola jakości

## NARZĘDZIA

### ScraperSocial

Moduł do monitorowania i analizy wzmianek w mediach społecznościowych.

#### Główne funkcje:
- Asynchroniczne pobieranie danych z Twitter/X i LinkedIn
- Analiza sentymentu wypowiedzi
- System alertów dla istotnych wzmianek
- Integracja z systemem RAG

#### Integracja z komponentami:
- SystemRAG: Wzbogacanie bazy wiedzy o dane z social media
- AnalizaRynku: Korelacja sentymentu z ruchami cen
- ZarzadzanieRyzykiem: Dostosowanie parametrów na podstawie nastrojów

#### Przykład konfiguracji:

```yaml
narzedzia:
  scraper_social:
    twitter:
      limit_wzmianek: 100
      interval_aktualizacji: 300
    linkedin:
      limit_wzmianek: 50
      interval_aktualizacji: 600
    alerty:
      prog_istotnosci: 0.8
    rag:
      prog_istotnosci: 0.7
```