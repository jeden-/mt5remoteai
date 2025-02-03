# Plan poprawy pokrycia testów 🎯

## 1. Naprawa konfiguracji testów 🔧

### 1.1. Aktualizacja pytest-asyncio
- [ ] Zaktualizować pytest-asyncio do najnowszej wersji
- [ ] Poprawić konfigurację event_loop w conftest.py
- [ ] Dostosować testy do nowego sposobu obsługi asynchronicznych fikstur

### 1.2. Naprawa testów integracyjnych bazy danych
- [ ] Naprawić błędy w test_db_integration.py
- [ ] Poprawić obsługę fikstur asynchronicznych
- [ ] Dodać prawidłową obsługę połączenia z bazą danych w testach

## 2. Poprawa pokrycia dla modułów z niskim pokryciem 📊

### 2.1. Moduły krytyczne (0-15% pokrycia)
- [ ] src/connectors/mt5_connector.py (0%)
  - [ ] Dodać mocki dla MT5
  - [ ] Pokryć podstawowe operacje
  - [ ] Pokryć obsługę błędów
  
- [ ] src/rag/market_memory.py (10%)
  - [ ] Pokryć inicjalizację i zamykanie
  - [ ] Pokryć operacje na danych
  - [ ] Pokryć obsługę błędów
  
- [ ] src/trading/mt5_handler.py (15%)
  - [ ] Dodać mocki dla operacji tradingowych
  - [ ] Pokryć zarządzanie pozycjami
  - [ ] Pokryć obsługę błędów
  
- [ ] src/utils/technical_indicators.py (15%)
  - [ ] Pokryć wszystkie wskaźniki
  - [ ] Pokryć walidację danych
  - [ ] Pokryć obsługę błędów

### 2.2. Moduły z średnim pokryciem (15-30%)
- [ ] src/rag/embeddings_handler.py (18%)
- [ ] src/trading/position_manager.py (21%)
- [ ] src/utils/logger.py (21%)
- [ ] src/database/postgres_handler.py (27%)
- [ ] src/connectors/ollama_connector.py (29%)
- [ ] src/utils/prompts.py (29%)

### 2.3. Moduły z dobrym pokryciem do dopracowania (70-90%)
- [ ] src/models/validators.py (78%)
- [ ] src/models/data_models.py (85%)
- [ ] src/models/enums.py (87%)

## 3. Optymalizacja i czyszczenie 🧹

### 3.1. Usunięcie ostrzeżeń
- [ ] Naprawić ostrzeżenia o przestarzałym użyciu event_loop
- [ ] Naprawić ostrzeżenia o deprecated features
- [ ] Usunąć nieużywane importy i zmienne

### 3.2. Refaktoryzacja testów
- [ ] Ujednolicić styl testów
- [ ] Poprawić nazewnictwo
- [ ] Dodać lepsze opisy testów
- [ ] Zoptymalizować wykorzystanie fikstur

## 4. Dokumentacja 📚

### 4.1. Aktualizacja dokumentacji testów
- [ ] Dodać opis konfiguracji testów
- [ ] Opisać mocki i fixtury
- [ ] Dodać przykłady uruchamiania testów
- [ ] Dodać opis procesu CI/CD

### 4.2. Aktualizacja README
- [ ] Dodać sekcję o testach
- [ ] Zaktualizować wymagania
- [ ] Dodać badges z pokryciem

## Kolejność działań:

1. Najpierw naprawa konfiguracji (1.1, 1.2)
2. Następnie moduły krytyczne (2.1)
3. Potem moduły ze średnim pokryciem (2.2)
4. Na końcu optymalizacja i dokumentacja (3, 4)

## Status:
🟡 W trakcie realizacji
✅ Ukończone zadania: 0/30
⬜ Pozostałe zadania: 30/30
