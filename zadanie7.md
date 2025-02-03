ZADANIE #7 - ETAP 1: Refaktoryzacja warstwy bazodanowej

1. Migracja na asyncpg:
- Zamień psycopg2 na asyncpg
- Ujednolicenie asynchronicznych operacji
- Implementacja connection pool

2. System migracji:
- Dodaj Alembic do zarządzania migracjami
- Utwórz podstawową strukturę migracji
- Skrypt inicjalizujący bazę danych

3. Testy:
- Testy jednostkowe dla PostgresHandler
- Testy integracyjne z bazą danych
- Mocki dla testów strategii

4. Konfiguracja:
- Rozszerzenie pliku .env
- Konfiguracja dla różnych środowisk
- Parametry połączeń i timeoutów

5. Obsługa błędów:
- Podstawowy system retry
- Obsługa typowych błędów bazy
- Logowanie błędów

Proszę o implementację powyższych elementów z zachowaniem istniejących funkcjonalności.
