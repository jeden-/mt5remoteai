# NikkeiNinja - Dokumentacja 🥷

## Status projektu (24.01.2024)

### 1. Komponenty zaimplementowane ✅
- **Baza danych PostgreSQL**
  - Skonfigurowana i działająca
  - Użytkownik `ninja` z odpowiednimi uprawnieniami
  - Baza `nikkeininja` gotowa do użycia

- **Integracja z MT5**
  - Połączenie działa
  - Możliwość pobierania danych rynkowych
  - Dostęp do historii i aktualnych notowań

- **System RAG (Retrieval Augmented Generation)**
  - Lokalny system oparty na ChromaDB
  - Funkcjonalności:
    - Dodawanie i aktualizacja dokumentów
    - Wyszukiwanie semantyczne
    - Filtrowanie po metadanych
    - Zarządzanie bazą wiedzy
  - Pełne pokrycie testami

### 2. Następne kroki 🎯

1. **Integracja MT5 z bazą danych**
   - Automatyczne pobieranie danych
   - Struktury tabel dla danych rynkowych
   - System cachowania i aktualizacji

2. **Rozszerzenie systemu RAG**
   - Integracja z danymi z MT5
   - Dodanie źródeł danych z mediów społecznościowych
   - Analiza sentymentu rynku

3. **Implementacja strategii tradingowych**
   - Definicja podstawowych strategii
   - System backtestingu
   - Zarządzanie ryzykiem

### 3. Architektura systemu 🏗️

```ascii
+-------------+     +--------------+     +-------------+
|    MT5      |---->| PostgreSQL   |<----| System RAG  |
|  (Dane      |     | (Baza       |     | (Analiza    |
|   rynkowe)  |     |  danych)     |     |  danych)    |
+-------------+     +--------------+     +-------------+
                           ^
                           |
                    +-------------+
                    |  Strategie  |
                    |  tradingowe |
                    +-------------+
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