# EmergencyTrader - Awaryjny EA do MT5

## Opis
EmergencyTrader jest awaryjnym Expert Advisorem (EA) dla MetaTrader 5 stworzonym w celu zarządzania pozycjami, gdy standardowy EA (np. SignalTrader) nie działa poprawnie. Ten EA pozwala na awaryjne zamykanie otwartych pozycji oraz testowanie systemu.

## Funkcje
- Zamykanie wszystkich otwartych pozycji
- Zamykanie określonych pozycji o podanych ticketach
- Zamykanie pozycji na podstawie pliku CSV
- Otwieranie testowej pozycji EURUSD BUY

## Instalacja
1. Skopiuj pliki `EmergencyTrader.mq5` i `SignalTraderFix.mqh` do katalogu:
   ```
   C:\Users\<Użytkownik>\AppData\Roaming\MetaQuotes\Terminal\<ID_TERMINALA>\MQL5\Experts\
   ```
2. Uruchom MetaTrader 5 i otwórz oba pliki w MetaEditor (naciśnij F4)
3. Skompiluj pliki (naciśnij F7)
4. Po skompilowaniu EA będzie dostępny w nawigatorze w sekcji "Eksperci"

## Konfiguracja
EA posiada kilka parametrów konfiguracyjnych:

- **CSVPath** - ścieżka do folderu z plikami CSV (domyślnie: C:\\Users\\win\\Documents\\mt5remoteai\\signals\\)
- **LogFile** - nazwa pliku logów (domyślnie: emergency_log.txt)
- **CheckInterval** - interwał sprawdzania w milisekundach (domyślnie: 1000 ms)
- **ActionType** - typ akcji do wykonania:
  - CLOSE_ALL - zamyka wszystkie otwarte pozycje
  - CLOSE_CSV - zamyka pozycje na podstawie pliku emergency_close.csv
  - OPEN_EURUSD_BUY - otwiera testową pozycję BUY na EURUSD
  - CLOSE_TICKETS - zamyka pozycje o określonych ticketach
- **Ticket1-Ticket5** - numery ticketów do zamknięcia (używane tylko gdy ActionType = CLOSE_TICKETS)

## Użycie
1. Otwórz wykres dowolnego instrumentu w MT5
2. Przeciągnij EmergencyTrader z nawigatora na wykres
3. Skonfiguruj parametry według potrzeb
4. Kliknij "OK" aby uruchomić EA

## Plik emergency_close.csv
Jeśli wybrana została opcja CLOSE_CSV, EA będzie szukał pliku emergency_close.csv w podanej ścieżce (CSVPath).
Plik powinien mieć następujący format:

```
id,timestamp,symbol,action,volume,price,sl_points,tp_points,ticket,comment,processed
1,2024-03-05 12:00:00,EURUSD,CLOSE,0,0,0,0,1234567,"Zamknięcie awaryjne",0
```

Najważniejsze pola to:
- action: musi być "CLOSE"
- ticket: numer pozycji do zamknięcia

## Rozwiązywanie problemów
1. **EA nie widzi pliku logów lub CSV**
   - Sprawdź ścieżkę w parametrze CSVPath
   - Upewnij się, że MT5 ma uprawnienia do zapisu w tej lokalizacji
   - Sprawdź logi w zakładce "Eksperci" w MT5

2. **EA nie zamyka pozycji**
   - Sprawdź numery ticketów (widoczne w zakładce "Trade" w MT5)
   - Upewnij się, że wybrany jest właściwy ActionType
   - Sprawdź logi w zakładce "Eksperci" w MT5

3. **Błędy kompilacji**
   - Upewnij się, że pliki SignalTraderFix.mqh i EmergencyTrader.mq5 znajdują się w tym samym katalogu
   - W razie potrzeby popraw ścieżkę #include w pliku EmergencyTrader.mq5

## Rozwiązanie problemów z SignalTrader
EmergencyTrader to narzędzie tymczasowe do rozwiązania problemów z głównym EA SignalTrader. 
Po zamknięciu pozycji, zalecamy:

1. Wygenerowanie nowych, poprawnych sygnałów handlowych
2. Upewnienie się, że daty w pliku CSV są zgodne z systemem
3. Sprawdzenie uprawnień do plików
4. Restart MT5 i ponowne włączenie SignalTrader

## Kontakt
W razie problemów z EA, skontaktuj się z administratorem systemu. 