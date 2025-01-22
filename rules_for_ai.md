RULES FOR AI - NIKKEININJA PROJECT 🥷
===================================

1. JĘZYK I NAZEWNICTWO
---------------------
- Używaj języka polskiego w nazwach zmiennych, funkcji i klas
- Zachowuj spójną konwencję nazewnictwa
- Używaj emotki 🥷 przy głównych komunikatach systemu
- Używaj jasnych i opisowych nazw

2. STRUKTURA KODU
----------------
- Zawsze implementuj obsługę błędów
- Używaj async/await dla operacji MT5
- Dodawaj type hints
- Implementuj wzorzec Logger
- Każda klasa musi mieć docstring
- Każda funkcja musi mieć docstring z opisem parametrów i zwracanych wartości

3. BEZPIECZEŃSTWO
----------------
- Nigdy nie pokazuj wrażliwych danych w logach
- Wszystkie dane dostępowe trzymaj w .env
- Implementuj walidację danych wejściowych
- Dodawaj zabezpieczenia przed błędami MT5
- Sprawdzaj limity pozycji i ryzyka

4. TRADING
---------
- Zawsze implementuj stop-loss
- Sprawdzaj wielkość pozycji przed wykonaniem
- Waliduj dane rynkowe
- Zabezpiecz przed wielokrotnym wykonaniem zleceń
- Monitoruj stan konta

5. FORMAT KODU
-------------
```python
class KomponentNinja:
    """Opis komponentu.
    
    Attributes:
        logger: Logger komponentu
        config: Konfiguracja komponentu
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._wczytaj_config()
    
    async def operacja(self, parametr: str) -> Dict[str, Any]:
        """Opis operacji.
        
        Args:
            parametr: Opis parametru
            
        Returns:
            Dict zawierający wynik operacji
            
        Raises:
            BladNinja: Opis możliwego błędu
        """
        try:
            # implementacja
            pass
        except Exception as e:
            self.logger.error(f"Błąd: {str(e)}")
            raise BladNinja(f"Błąd operacji: {str(e)}")

6. LOGOWANIE
Używaj odpowiednich poziomów logowania (INFO, WARNING, ERROR)
Loguj wszystkie operacje tradingowe
Dodawaj timestamp do logów
Zapisuj logi do pliku

7. KOMENTARZE
Komentuj skomplikowaną logikę biznesową
Wyjaśniaj wzorce Wyckoffa
Opisuj warunki rynkowe
Dokumentuj założenia

8. TESTOWANIE
Dodawaj testy jednostkowe
Implementuj scenariusze testowe
Testuj przypadki brzegowe
Sprawdzaj obsługę błędów

9. OPTYMALIZACJA
Optymalizuj zapytania do MT5
Unikaj niepotrzebnych operacji
Wykorzystuj cachowanie gdzie możliwe
Monitoruj zużycie pamięci

10. KOMUNIKATY
Używaj czytelnych komunikatów błędów
Dodawaj sugestie rozwiązania problemów
Informuj o statusie operacji
Używaj emoji dla lepszej czytelności

11. GIT
Format commitów:

<typ>: <opis>

Gdzie typ to:
- nowa: 🆕 nowa funkcjonalność
- popr: 🛠️ poprawka błędu
- dok: 📚 dokumentacja
- styl: 🎨 formatowanie
- ref: ♻️ refaktoryzacja
- test: 🧪 testy
- admin: ⚙️ administracja
Copy code

12. PRIORYTETYZACJA
Bezpieczeństwo tradingu
Stabilność systemu
Zarządzanie ryzykiem
Funkcjonalność
Optymalizacja

13. ZAKAZY

Nie używaj magic numbers
Nie zostawiaj zakomentowanego kodu
Nie pomijaj obsługi błędów
Nie używaj print() (tylko logger)
Nie hardcoduj wartości konfiguracyjnych