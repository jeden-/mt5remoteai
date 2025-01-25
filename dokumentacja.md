# NikkeiNinja - Dokumentacja 🥷

## Status projektu (03.03.2024)

### 1. Komponenty zaimplementowane ✅

- **Baza danych PostgreSQL**
  - Skonfigurowana i działająca
  - Użytkownik `ninja` z odpowiednimi uprawnieniami
  - Baza `nikkeininja` gotowa do użycia
  - Zaimplementowane modele dla:
    - Transakcji i pozycji
    - Historii cen
    - Metryk handlowych
    - Cache'u systemowego

- **Integracja z MT5**
  - Połączenie działa
  - Możliwość pobierania danych rynkowych
  - Dostęp do historii i aktualnych notowań
  - Pobieranie rzeczywistego wolumenu (`real_volume`)
  - Synchronizacja danych z cachowaniem
  - Wykonywanie zleceń
  - Zarządzanie pozycjami
  - Kalendarz ekonomiczny:
    - Pobieranie wydarzeń dla krajów/walut
    - Dostęp do historycznych wartości wskaźników
    - Filtrowanie po ważności wydarzeń
    - Integracja z systemem decyzyjnym

- **System RAG (Retrieval Augmented Generation)**
  - Lokalny system oparty na ChromaDB
  - Funkcjonalności:
    - Dodawanie i aktualizacja dokumentów
    - Wyszukiwanie semantyczne
    - Filtrowanie po metadanych
    - Zarządzanie bazą wiedzy
  - Pełne pokrycie testami

- **Analiza techniczna**
  - Wskaźniki cenowe (RSI, MACD, SMA, EMA)
  - Wskaźniki wolumenowe:
    - OBV (On Balance Volume)
    - ADI (Accumulation/Distribution Index)
    - CMF (Chaikin Money Flow)
    - VWAP (Volume Weighted Average Price)
    - Relative Volume
  - Wskaźniki momentum:
    - Stochastic Oscillator
    - ATR (Average True Range)
  - Analiza Livermore'a:
    - Wykrywanie trendu
    - Punkty zwrotne
    - Poziomy wsparcia/oporu
    - Siła trendu
  - Wykrywanie formacji świecowych
  - Generowanie sygnałów handlowych
  - Pełne pokrycie testami

- **Strategie tradingowe**
  - Strategia Wyckoffa:
    - Wykrywanie faz rynku
    - Identyfikacja formacji spring/upthrust
    - Generowanie sygnałów long/short
  - Strategia techniczna:
    - Wykorzystanie wskaźników
    - Logika sygnałów
    - Parametryzacja

- **Analiza sentymentu**
  - Scraping danych z Twittera
  - Integracja z Claude API
  - Klasyfikacja sentymentu
  - System punktacji

- **System backtestingu**
  - Symulator rynku:
    - Obsługa zleceń market/limit/stop
    - Śledzenie pozycji i kapitału
    - Obliczanie prowizji i kosztów
    - Historia transakcji
  - Metryki wydajności:
    - Win Rate i Profit Factor
    - Sharpe Ratio
    - Maximum Drawdown
    - Analiza rozkładu zwrotów
  - Optymalizacja parametrów:
    - Grid search
    - Walidacja krzyżowa
    - Optymalizacja wielokryterialna
  - Raporty i wizualizacje:
    - Wykresy equity
    - Statystyki transakcji
    - Analiza ryzyka
    - Raporty HTML z wykresami

### 2. W trakcie rozwoju 🚧

1. **Backtesting i optymalizacja**
   - System testowania strategii
   - Optymalizacja parametrów
   - Analiza wyników

2. **Integracja komponentów**
   - Połączenie sygnałów technicznych i sentymentu
   - System wag dla różnych źródeł sygnałów
   - Filtrowanie fałszywych sygnałów

3. **Rozszerzony monitoring**
   - Statystyki na żywo
   - Alerty krytyczne
   - Monitoring zasobów

### 3. Architektura systemu 🏗️

```ascii
+-------------+     +--------------+     +-------------+
|    MT5      |---->| PostgreSQL   |<----| System RAG  |
|  (Dane      |     | (Baza       |     | (Analiza    |
|   rynkowe)  |     |  danych)     |     |  danych)    |
+-------------+     +--------------+     +-------------+
        |                  ^                    ^
        |                  |                    |
        v                  |                    |
+------------------+  +-------------+    +--------------+
|    Strategie     |->|  Analiza   |<---| Sentyment    |
|    tradingowe    |  | techniczna |    | rynku        |
+------------------+  +-------------+    +--------------+
        |                  |                    |
        v                  v                    v
+--------------------------------------------------+
|                   Dashboard                        |
+--------------------------------------------------+
```

### 4. Konwencje i standardy 📝

- Język dokumentacji i kodu: Polski
- Type hints obowiązkowe
- Docstringi dla wszystkich klas i metod
- Emoji w logach:
  - 🥷 - operacje ninja
  - ⚠️ - ostrzeżenia
  - ❌ - błędy
- Testy jednostkowe i integracyjne
- PEP 8 (max 120 znaków)

### 5. Bezpieczeństwo 🔒

- Dane wrażliwe w zmiennych środowiskowych
- Separacja środowisk (dev/prod)
- Regularne backupy bazy danych
- Monitoring operacji tradingowych
- Limity pozycji i zarządzanie ryzykiem

### 6. Monitorowanie i logi 📊

- Szczegółowe logi operacji
- Metryki wydajności
- Alerty dla zdarzeń krytycznych
- Śledzenie decyzji tradingowych
- Statystyki systemu RAG

### 7. Następne kroki 🎯

1. **Integracja kalendarza ekonomicznego**
   - Implementacja pobierania wydarzeń
   - Integracja z bazą danych
   - Aktualizacja w czasie rzeczywistym
   - Wykorzystanie w strategii hybrydowej

2. **Optymalizacja strategii Wyckoffa**
   - Dostrojenie parametrów wykrywania faz
   - Kalibracja progów wolumenu
   - Redukcja fałszywych sygnałów

3. **Rozbudowa systemu backtestingu**
   - Implementacja symulatora rynku
   - Analiza Monte Carlo
   - Generowanie raportów wydajności

4. **Integracja z zewnętrznymi źródłami**
   - Dodatkowe API giełdowe
   - Dane makroekonomiczne
   - Kalendarz wydarzeń rynkowych

5. **Optymalizacja wydajności**
   - Profilowanie kodu
   - Optymalizacja obliczeń wskaźników
   - Redukcja zużycia pamięci
   - Równoległe przetwarzanie danych

6. **Rozszerzenie analizy technicznej**
   - Dodatkowe wskaźniki i oscylatory
   - Zaawansowane formacje cenowe
   - Machine learning w detekcji wzorców
   - Adaptacyjne parametry wskaźników

7. **Ulepszenie backtestingu**
   - Symulacja różnych warunków rynkowych
   - Stress testing strategii
   - Analiza wrażliwości parametrów
   - Optymalizacja genetyczna