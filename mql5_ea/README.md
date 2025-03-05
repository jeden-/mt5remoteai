# SignalTrader Expert Advisor dla MT5

## Opis

SignalTrader to Expert Advisor dla MetaTrader 5, który umożliwia wykonywanie transakcji handlowych na podstawie sygnałów przekazywanych przez zewnętrzną aplikację. Jest to rozwiązanie obejściowe dla sytuacji, gdy broker blokuje bezpośrednie wywołania API do otwierania/zamykania pozycji.

## Jak to działa

1. Aplikacja Python generuje sygnały handlowe (kupno, sprzedaż, zamknięcie pozycji) i zapisuje je do pliku CSV.
2. Expert Advisor monitoruje ten plik i na bieżąco wykonuje nowe polecenia handlowe.
3. Po wykonaniu polecenia, EA oznacza sygnał jako przetworzony i zapisuje wynik w pliku logów.

## Instalacja

1. Skopiuj plik `SignalTrader.mq5` do katalogu `MQL5/Experts` w folderze danych MetaTrader 5.
   - Domyślna lokalizacja: `C:\Users\<Użytkownik>\AppData\Roaming\MetaQuotes\Terminal\<ID_TERMINALA>\MQL5\Experts\`
   - Możesz też otworzyć MetaEditor (Alt+F4 w MT5), wybrać File -> Open Data Folder i przejść do folderu MQL5/Experts

2. W MetaEditor, otwórz plik `SignalTrader.mq5` i skompiluj go (F7).

3. **WAŻNE: Upewnij się, że folder sygnałów istnieje** - domyślnie `C:\Users\win\Documents\mt5remoteai\signals\`. Jeśli nie istnieje, utwórz go ręcznie przed uruchomieniem EA.

4. Uruchom MetaTrader 5 i upewnij się, że Auto-Trading jest włączone (przycisk z symbolem automatu w prawym górnym rogu).

5. W oknie Nawigator, znajdź SignalTrader w sekcji Expert Advisors, przeciągnij go na wykres i skonfiguruj parametry:
   - `SignalFilePath`: Ścieżka do folderu z sygnałami (domyślnie: `C:\Users\win\Documents\mt5remoteai\signals\`)
   - `SignalFileName`: Nazwa pliku sygnałów (domyślnie: `trading_signals.csv`)
   - `CheckInterval`: Częstotliwość sprawdzania sygnałów w milisekundach (domyślnie: 500)
   - `DefaultVolume`: Domyślny wolumen transakcji (domyślnie: 0.01)
   - `DefaultSLPoints`: Domyślny Stop Loss w punktach (domyślnie: 100)
   - `DefaultTPPoints`: Domyślny Take Profit w punktach (domyślnie: 200)
   - `SlippagePoints`: Maksymalny dozwolony poślizg w punktach (domyślnie: 20)
   - `EnableDetailedLogging`: Włącz/wyłącz szczegółowe logowanie (domyślnie: true)

6. Kliknij OK, aby zatwierdzić konfigurację i uruchomić Expert Advisor.

7. **Sprawdź, czy EA działa poprawnie** - w dzienniku MT5 (Ctrl+T) powinieneś zobaczyć komunikat "SignalTrader EA uruchomiony".

## Struktura pliku sygnałów

Plik sygnałów jest w formacie CSV i zawiera następujące kolumny:

- `id` - Unikalny identyfikator sygnału
- `timestamp` - Data i czas utworzenia sygnału (format: YYYY-MM-DD HH:MM:SS)
- `symbol` - Symbol instrumentu (np. "EURUSD")
- `action` - Akcja do wykonania ("BUY", "SELL", "CLOSE")
- `volume` - Wolumen transakcji (w lotach)
- `price` - Cena (0 dla ceny rynkowej)
- `sl_points` - Stop Loss w punktach
- `tp_points` - Take Profit w punktach
- `ticket` - Numer pozycji (dla akcji "CLOSE")
- `comment` - Komentarz do transakcji
- `processed` - Czy sygnał został przetworzony (0 - nie, 1 - tak)

## Pliki logów

EA tworzy plik logów `signal_trader_log.txt` w folderze sygnałów, zawierający informacje o wykonanych transakcjach i ewentualnych błędach. Dodatkowo wszystkie komunikaty są wysyłane do dziennika MT5 (Ctrl+T).

## Rozwiązywanie problemów

1. **EA nie widzi pliku sygnałów**
   - Sprawdź, czy ścieżka do pliku jest poprawna
   - **WAŻNE:** Upewnij się, że folder `signals` istnieje
   - Jeśli pliku nie ma, EA powinien go utworzyć przy starcie
   - Sprawdź, czy w dzienniku MT5 nie ma błędów związanych z dostępem do pliku

2. **EA nie wykonuje transakcji**
   - Upewnij się, że Auto-Trading jest włączone w MT5 (przycisk w prawym górnym rogu)
   - Sprawdź ustawienia EA i upewnij się, że masz odpowiednie uprawnienia do handlu
   - W dzienniku MT5 (Ctrl+T) sprawdź, czy nie ma błędów podczas wykonywania transakcji
   - Sprawdź plik logów w folderze sygnałów

3. **Błędy wykonania transakcji**
   - Sprawdź, czy symbol jest dostępny do handlu (może być niedostępny poza godzinami handlu)
   - Upewnij się, że masz wystarczające środki na koncie
   - Sprawdź, czy format dat w pliku sygnałów jest poprawny (YYYY-MM-DD HH:MM:SS)
   - Sprawdź, czy wolumen transakcji jest zgodny z minimalnymi wymaganiami brokera

4. **Problemy z formatem pliku**
   - Upewnij się, że plik CSV ma poprawny format i zawiera wszystkie wymagane kolumny
   - Sprawdź, czy wartości liczbowe używają kropki jako separatora dziesiętnego
   - Unikaj używania polskich znaków w komentarzach
   - Sprawdź, czy plik nie jest otwarty w innym programie podczas działania EA

## Ograniczenia

- EA działa tylko na jednym wykresie w jednym czasie
- Wymagane uprawnienia do handlu i włączone Auto-Trading w MT5
- Pewne opóźnienie między wygenerowaniem sygnału a jego wykonaniem 

## Wskazówki

- Aby przetestować działanie EA, możesz ręcznie dodać nowy wiersz do pliku sygnałów
- Monitoruj dziennik MT5 (Ctrl+T) podczas działania EA, aby szybko wykryć potencjalne problemy
- Dla bezpieczeństwa, rozpocznij od małych wolumenów transakcji

## Rozwiązane problemy w wersji 1.02

W wersji 1.02 EA poprawiono następujące problemy, które mogły powodować brak otwierania/zamykania pozycji:

1. **Problem z formatem daty** - EA teraz poprawnie obsługuje różne formaty daty, w tym daty z przyszłości
2. **Problem z dostępem do plików** - Dodano flagi FILE_SHARE_READ i FILE_SHARE_WRITE, aby umożliwić jednoczesny dostęp do plików przez EA i aplikację Python
3. **Problem z uprawnieniami** - EA wykonuje testy odczytu/zapisu podczas inicjalizacji i wyświetla komunikaty diagnostyczne
4. **Problem z obsługą błędów** - Dodano lepszą obsługę błędów i wyczerpujące logowanie
5. **Problem z przetwarzaniem sygnałów** - EA teraz oznacza sygnały jako przetworzone tylko po pomyślnym wykonaniu transakcji
6. **Zbyt wolne sprawdzanie sygnałów** - Zwiększono częstotliwość sprawdzania pliku sygnałów (zmniejszono CheckInterval do 250ms)
7. **Brak informacji diagnostycznych** - Dodano szczegółowe logowanie do pliku i konsoli MT5

Po zainstalowaniu nowej wersji EA, wykonaj następujące kroki:

1. Upewnij się, że folder `signals` istnieje i nie jest chroniony przed zapisem
2. Utwórz pusty plik `signal_trader_log.txt` w folderze `signals`, jeśli go nie ma
3. Zrestartuj MetaTrader 5 i upewnij się, że Auto-Trading jest włączone
4. Sprawdź dziennik MT5 (Ctrl+T), aby monitorować komunikaty z EA
5. W przypadku dalszych problemów, zgłoś szczegółowe komunikaty błędów z pliku logów 