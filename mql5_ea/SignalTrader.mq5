//+------------------------------------------------------------------+
//|                                                SignalTrader.mq5 |
//|                                       MT5RemoteAI Trading System |
//+------------------------------------------------------------------+
#property copyright "MT5RemoteAI"
#property link      ""
#property version   "1.02"
#property strict

// Parametry wejściowe
input string   SignalFilePath = "C:\\Users\\win\\Documents\\mt5remoteai\\signals\\";  // Ścieżka do pliku sygnałów
input string   SignalFileName = "trading_signals.csv";                               // Nazwa pliku sygnałów
input int      CheckInterval = 250;                                                 // Interwał sprawdzania sygnałów (ms)
input double   DefaultVolume = 0.01;                                               // Domyślny wolumen transakcji
input int      DefaultSLPoints = 100;                                              // Domyślny Stop Loss w punktach
input int      DefaultTPPoints = 200;                                              // Domyślny Take Profit w punktach
input int      SlippagePoints = 20;                                                // Maksymalny poślizg w punktach
input bool     EnableDetailedLogging = true;                                       // Włącz szczegółowe logowanie

// Zmienne globalne
int fileHandle = INVALID_HANDLE;
datetime lastFileCheck = 0;
int lastProcessedLine = 0;
int lastSignalID = 0;
string fullFilePath = "";
datetime currentTime;
bool isFirstRun = true;

//+------------------------------------------------------------------+
//| Struktura sygnału handlowego                                     |
//+------------------------------------------------------------------+
struct TradeSignal
{
   int         id;             // Unikalny identyfikator sygnału
   string      symbol;         // Symbol instrumentu
   string      action;         // Akcja: "BUY", "SELL", "CLOSE"
   double      volume;         // Wolumen (w lotach)
   double      price;          // Cena (0 dla ceny rynkowej)
   int         slPoints;       // Stop Loss w punktach
   int         tpPoints;       // Take Profit w punktach
   int         ticket;         // Numer pozycji (dla CLOSE)
   datetime    time;           // Czas utworzenia sygnału
   bool        processed;      // Czy sygnał został przetworzony
   string      comment;        // Komentarz do transakcji
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   currentTime = TimeCurrent();
   
   // Wyświetl informację startową
   string startMessage = StringFormat("SignalTrader EA uruchomiony. Monitorowanie sygnałów w: %s. Czas: %s", 
      SignalFilePath + SignalFileName, TimeToString(currentTime));
   Print(startMessage);
   
   // Sprawdź, czy katalog istnieje
   if(!FolderCreate(SignalFilePath))
   {
      if(GetLastError() != 4301)  // 4301 to kod błędu "Folder już istnieje"
      {
         Print("Ostrzeżenie: Problem z katalogiem sygnałów: ", GetLastError());
         // Nie przerywaj inicjalizacji, folder może już istnieć
      }
   }
   
   fullFilePath = SignalFilePath + SignalFileName;
   
   // Utwórz pusty plik sygnałów, jeśli nie istnieje
   if(!FileIsExist(fullFilePath))
   {
      fileHandle = FileOpen(fullFilePath, FILE_WRITE|FILE_CSV|FILE_ANSI);
      if(fileHandle != INVALID_HANDLE)
      {
         FileWrite(fileHandle, "id", "timestamp", "symbol", "action", "volume", "price", "sl_points", "tp_points", "ticket", "comment", "processed");
         FileClose(fileHandle);
         Print("Utworzono nowy plik sygnałów");
      }
      else
      {
         Print("Ostrzeżenie: Nie można utworzyć pliku sygnałów. Kod błędu: ", GetLastError());
         // Nie przerywamy inicjalizacji - plik może być tworzony przez aplikację Python
      }
   }
   
   // Inicjalizuj plik logów
   string logMessage = StringFormat("=============== INICJALIZACJA EA ===============\nWersja: 1.02\nCzas: %s\nŚcieżka sygnałów: %s", 
      TimeToString(currentTime), fullFilePath);
   WriteLog(logMessage);
   
   // Zarejestruj timer do regularnego sprawdzania pliku
   if(!EventSetMillisecondTimer(CheckInterval))
   {
      Print("Ostrzeżenie: Nie można ustawić timera. Kod błędu: ", GetLastError());
      WriteLog("BŁĄD: Nie można ustawić timera. Kod błędu: " + IntegerToString(GetLastError()));
   }
   
   // Sprawdź uprawnienia i pisz do dziennika tylko przy pierwszym uruchomieniu
   if(isFirstRun)
   {
      // Testuj pisanie do pliku logów
      if(!TestFileWrite(SignalFilePath + "signal_trader_log.txt"))
      {
         Print("OSTRZEŻENIE: Problem z zapisem do pliku logów! Sprawdź uprawnienia.");
      }
      
      // Testuj odczyt pliku sygnałów
      if(!TestFileRead(fullFilePath))
      {
         Print("OSTRZEŻENIE: Problem z odczytem pliku sygnałów! Sprawdź uprawnienia.");
      }
      
      isFirstRun = false;
   }
   
   Print("Inicjalizacja EA zakończona pomyślnie.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Testuj możliwość zapisu do pliku                                 |
//+------------------------------------------------------------------+
bool TestFileWrite(string filePath)
{
   int testHandle = FileOpen(filePath, FILE_WRITE|FILE_READ|FILE_TXT|FILE_ANSI);
   if(testHandle == INVALID_HANDLE)
   {
      Print("Test zapisu do pliku nie powiódł się: ", GetLastError());
      return false;
   }
   
   // Zapisz testową linię
   FileWriteString(testHandle, "Test zapisu: " + TimeToString(TimeCurrent()) + "\n");
   FileClose(testHandle);
   return true;
}

//+------------------------------------------------------------------+
//| Testuj możliwość odczytu pliku                                   |
//+------------------------------------------------------------------+
bool TestFileRead(string filePath)
{
   int testHandle = FileOpen(filePath, FILE_READ|FILE_CSV|FILE_ANSI);
   if(testHandle == INVALID_HANDLE)
   {
      Print("Test odczytu pliku nie powiódł się: ", GetLastError());
      return false;
   }
   
   FileClose(testHandle);
   return true;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Wyłącz timer
   EventKillTimer();
   
   // Zamknij plik, jeśli jest otwarty
   if(fileHandle != INVALID_HANDLE)
   {
      FileClose(fileHandle);
   }
   
   WriteLog("EA zatrzymany. Przyczyna: " + IntegerToString(reason));
   Print("SignalTrader EA zatrzymany. Przyczyna: ", reason);
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   currentTime = TimeCurrent();
   
   // Sprawdź, czy plik istnieje
   if(!FileIsExist(fullFilePath))
   {
      // Plik nie istnieje, nie podejmuj żadnych działań
      if(EnableDetailedLogging) 
      {
         Print("Plik sygnałów nie istnieje: ", fullFilePath);
      }
      return;
   }
   
   // Przetworz sygnały bez względu na czas modyfikacji
   ProcessSignals();
}

//+------------------------------------------------------------------+
//| Przetwarzanie sygnałów z pliku                                  |
//+------------------------------------------------------------------+
void ProcessSignals()
{
   // Próbuj otworzyć plik kilka razy w przypadku błędów
   int attempts = 0;
   int maxAttempts = 3;
   
   while(attempts < maxAttempts)
   {
      // Otwórz plik sygnałów do odczytu
      fileHandle = FileOpen(fullFilePath, FILE_READ|FILE_CSV|FILE_ANSI|FILE_SHARE_READ|FILE_SHARE_WRITE);
      if(fileHandle != INVALID_HANDLE)
      {
         break;  // Udało się otworzyć plik
      }
      
      attempts++;
      Sleep(100);  // Poczekaj chwilę przed kolejną próbą
   }
   
   if(fileHandle == INVALID_HANDLE)
   {
      if(EnableDetailedLogging) 
      {
         Print("Nie można otworzyć pliku sygnałów po ", maxAttempts, " próbach. Kod błędu: ", GetLastError());
      }
      return;
   }
   
   // Przejdź przez nagłówki
   if(FileIsEnding(fileHandle) == false)
      FileReadString(fileHandle);  // Przeczytaj pierwszą kolumnę (id)
   
   while(!FileIsLineEnding(fileHandle) && !FileIsEnding(fileHandle))
      FileReadString(fileHandle);  // Pomiń resztę nagłówków
   
   // Przejdź do następnej linii po nagłówkach
   if(!FileIsEnding(fileHandle))
      FileReadString(fileHandle);
   
   // Przetwórz wszystkie sygnały
   TradeSignal signal;
   int signalsProcessed = 0;
   bool errorOccurred = false;
   
   while(!FileIsEnding(fileHandle))
   {
      string idStr = FileReadString(fileHandle);
      if(idStr == "" || StringLen(idStr) == 0)
         break;  // Pusta linia, zakończ przetwarzanie
      
      // Odczytaj dane sygnału
      signal.id = (int)StringToInteger(idStr);
      
      string timestampStr = FileReadString(fileHandle);
      // Obsłuż różne formaty daty
      if(StringLen(timestampStr) > 0)
      {
         signal.time = StringToTime(timestampStr);
         if(signal.time == 0) 
         {
            // Spróbuj inny format daty lub użyj aktualnego czasu
            signal.time = currentTime;
            
            if(EnableDetailedLogging) 
            {
               WriteLog("Ostrzeżenie: Błędny format daty w sygnale ID=" + IntegerToString(signal.id) + 
                       ", oryginalna data: " + timestampStr + ", używam aktualnego czasu: " + TimeToString(currentTime));
            }
         }
      }
      else
      {
         signal.time = currentTime;
      }
      
      signal.symbol = FileReadString(fileHandle);
      signal.action = FileReadString(fileHandle);
      
      string volumeStr = FileReadString(fileHandle);
      signal.volume = (StringLen(volumeStr) > 0) ? StringToDouble(volumeStr) : 0.0;
      
      string priceStr = FileReadString(fileHandle);
      signal.price = (StringLen(priceStr) > 0) ? StringToDouble(priceStr) : 0.0;
      
      string slStr = FileReadString(fileHandle);
      signal.slPoints = (StringLen(slStr) > 0) ? (int)StringToInteger(slStr) : 0;
      
      string tpStr = FileReadString(fileHandle);
      signal.tpPoints = (StringLen(tpStr) > 0) ? (int)StringToInteger(tpStr) : 0;
      
      string ticketStr = FileReadString(fileHandle);
      signal.ticket = (StringLen(ticketStr) > 0) ? (int)StringToInteger(ticketStr) : 0;
      
      signal.comment = FileReadString(fileHandle);
      
      string processedStr = FileReadString(fileHandle);
      signal.processed = (processedStr == "1" || processedStr == "true" || processedStr == "TRUE");
      
      // Jeśli sygnał nie został jeszcze przetworzony
      if(!signal.processed)
      {
         // Zapisz do logu informację o przetwarzanym sygnale
         string signalInfo = StringFormat("Przetwarzanie sygnału ID=%d, Action=%s, Symbol=%s, Volume=%.2f, Ticket=%d", 
            signal.id, signal.action, signal.symbol, signal.volume, signal.ticket);
         WriteLog(signalInfo);
         Print(signalInfo);
         
         bool success = false;
         
         // Obsłuż sygnał w zależności od akcji
         if(signal.action == "BUY" || signal.action == "SELL")
         {
            success = ExecuteTradeSignal(signal);
         }
         else if(signal.action == "CLOSE" && signal.ticket > 0)
         {
            success = ClosePosition(signal);
         }
         else
         {
            WriteLog("Nieznana akcja: " + signal.action + " dla ID=" + IntegerToString(signal.id));
            errorOccurred = true;
         }
         
         // Oznacz sygnał jako przetworzony tylko jeśli operacja się powiodła
         if(success)
         {
            signalsProcessed++;
            MarkSignalAsProcessed(signal.id);
         }
      }
      
      // Przejdź do następnej linii, jeśli nie jesteśmy na końcu pliku
      if(!FileIsEnding(fileHandle))
      {
         while(!FileIsLineEnding(fileHandle) && !FileIsEnding(fileHandle))
            FileReadString(fileHandle);
         
         if(!FileIsEnding(fileHandle))
            FileReadString(fileHandle);
      }
   }
   
   // Zamknij plik
   FileClose(fileHandle);
   
   // Zapisz podsumowanie do logu
   if(signalsProcessed > 0 || errorOccurred) 
   {
      string summary = StringFormat("Podsumowanie przetwarzania: Przetworzono %d sygnałów. Czas: %s", 
         signalsProcessed, TimeToString(TimeCurrent()));
      WriteLog(summary);
      Print(summary);
   }
}

//+------------------------------------------------------------------+
//| Wykonanie sygnału handlowego                                     |
//+------------------------------------------------------------------+
bool ExecuteTradeSignal(TradeSignal &signal)
{
   Print("Przetwarzanie sygnału #", signal.id, ": ", signal.action, " ", signal.symbol);
   
   // Ustaw wolumen, SL i TP na wartości domyślne, jeśli nie zostały podane
   double volume = (signal.volume <= 0) ? DefaultVolume : signal.volume;
   int slPoints = (signal.slPoints <= 0) ? DefaultSLPoints : signal.slPoints;
   int tpPoints = (signal.tpPoints <= 0) ? DefaultTPPoints : signal.tpPoints;
   
   // Wybierz symbol
   if(!SymbolSelect(signal.symbol, true))
   {
      string errorMsg = StringFormat("Błąd: Nie można wybrać symbolu %s. Kod błędu: %d", 
         signal.symbol, GetLastError());
      WriteLog(errorMsg);
      Print(errorMsg);
      return false;
   }
   
   // Pobierz informacje o symbolu
   double ask = SymbolInfoDouble(signal.symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(signal.symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(signal.symbol, SYMBOL_POINT);
   
   if(ask == 0 || bid == 0 || point == 0)
   {
      string errorMsg = StringFormat("Błąd: Nieprawidłowe dane rynkowe dla %s. Ask: %.5f, Bid: %.5f, Point: %.5f", 
         signal.symbol, ask, bid, point);
      WriteLog(errorMsg);
      Print(errorMsg);
      return false;
   }
   
   // Wypisz dane rynkowe do logów
   string marketData = StringFormat("Dane rynkowe: Symbol=%s, Ask=%.5f, Bid=%.5f, Point=%.5f", 
      signal.symbol, ask, bid, point);
   WriteLog(marketData);
   
   // Ustal typ zlecenia i cenę
   ENUM_ORDER_TYPE orderType;
   double price;
   
   if(signal.action == "BUY")
   {
      orderType = ORDER_TYPE_BUY;
      price = ask;
   }
   else if(signal.action == "SELL")
   {
      orderType = ORDER_TYPE_SELL;
      price = bid;
   }
   else
   {
      WriteLog("Błąd: Nieznana akcja " + signal.action);
      return false;
   }
   
   // Oblicz poziomy SL i TP
   double sl = 0.0;
   double tp = 0.0;
   
   if(slPoints > 0)
   {
      sl = (orderType == ORDER_TYPE_BUY) ? price - slPoints * point : price + slPoints * point;
   }
   
   if(tpPoints > 0)
   {
      tp = (orderType == ORDER_TYPE_BUY) ? price + tpPoints * point : price - tpPoints * point;
   }
   
   // Przygotuj strukturę żądania
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = signal.symbol;
   request.volume = volume;
   request.type = orderType;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = SlippagePoints;
   request.magic = 123456;  // Unikalny numer Magic
   request.comment = (signal.comment == "") ? "SignalTrader EA" : signal.comment;
   request.type_filling = ORDER_FILLING_FOK;
   request.type_time = ORDER_TIME_GTC;
   
   // Wypisz szczegóły żądania
   string requestDetails = StringFormat("Żądanie otwarcia pozycji: Action=%s, Symbol=%s, Volume=%.2f, Price=%.5f, SL=%.5f, TP=%.5f", 
      EnumToString(request.action), request.symbol, request.volume, request.price, request.sl, request.tp);
   WriteLog(requestDetails);
   
   // Wyślij żądanie
   bool success = OrderSend(request, result);
   
   if(success && result.retcode == TRADE_RETCODE_DONE)
   {
      string successMsg = StringFormat("Zlecenie wykonane: %s %s, Ticket: %d, Cena: %.5f", 
         signal.action, signal.symbol, result.order, price);
      WriteLog(successMsg);
      Print(successMsg);
      return true;
   }
   else
   {
      string errorMsg = StringFormat("Błąd wykonania zlecenia: %d, %s. Request: %s", 
         result.retcode, GetErrorDescription(result.retcode), requestDetails);
      WriteLog(errorMsg);
      Print(errorMsg);
      return false;
   }
}

//+------------------------------------------------------------------+
//| Zamknięcie pozycji                                              |
//+------------------------------------------------------------------+
bool ClosePosition(TradeSignal &signal)
{
   string ticketMsg = StringFormat("Próba zamknięcia pozycji #%d", signal.ticket);
   WriteLog(ticketMsg);
   Print(ticketMsg);
   
   // Sprawdź, czy pozycja istnieje
   if(!PositionSelectByTicket(signal.ticket))
   {
      string errorMsg = StringFormat("Błąd: Pozycja #%d nie istnieje. Kod błędu: %d", 
         signal.ticket, GetLastError());
      WriteLog(errorMsg);
      Print(errorMsg);
      
      // Sprawdź wszystkie otwarte pozycje aby zweryfikować, czy ticket jest poprawny
      int totalPositions = PositionsTotal();
      if(totalPositions > 0)
      {
         string positionsInfo = "Aktualne pozycje: ";
         for(int i = 0; i < totalPositions; i++)
         {
            ulong posTicket = PositionGetTicket(i);
            if(posTicket > 0)
            {
               string symbol = PositionGetString(POSITION_SYMBOL);
               double volume = PositionGetDouble(POSITION_VOLUME);
               positionsInfo += StringFormat("#%d (%s, %.2f) ", posTicket, symbol, volume);
            }
         }
         WriteLog(positionsInfo);
         Print(positionsInfo);
      }
      else
      {
         WriteLog("Brak otwartych pozycji");
      }
      
      return false;
   }
   
   // Pobierz informacje o pozycji
   string symbol = PositionGetString(POSITION_SYMBOL);
   double volume = PositionGetDouble(POSITION_VOLUME);
   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   
   string positionInfo = StringFormat("Znaleziono pozycję: Ticket=%d, Symbol=%s, Type=%s, Volume=%.2f", 
      signal.ticket, symbol, EnumToString(posType), volume);
   WriteLog(positionInfo);
   
   // Określ typ zlecenia przeciwny do aktualnej pozycji
   ENUM_ORDER_TYPE orderType = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   
   // Pobierz cenę
   double price = (orderType == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
   
   // Przygotuj strukturę żądania
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = volume;
   request.type = orderType;
   request.position = signal.ticket;
   request.price = price;
   request.deviation = SlippagePoints;
   request.magic = 123456;
   request.comment = "Close by SignalTrader EA";
   request.type_filling = ORDER_FILLING_FOK;
   request.type_time = ORDER_TIME_GTC;
   
   // Wypisz szczegóły żądania
   string requestDetails = StringFormat("Żądanie zamknięcia pozycji: Action=%s, Symbol=%s, Volume=%.2f, Price=%.5f, Ticket=%d", 
      EnumToString(request.action), request.symbol, request.volume, request.price, signal.ticket);
   WriteLog(requestDetails);
   
   // Wyślij żądanie
   bool success = OrderSend(request, result);
   
   if(success && result.retcode == TRADE_RETCODE_DONE)
   {
      string successMsg = StringFormat("Pozycja zamknięta: Ticket: %d, Symbol: %s, Cena: %.5f", 
         signal.ticket, symbol, price);
      WriteLog(successMsg);
      Print(successMsg);
      return true;
   }
   else
   {
      string errorMsg = StringFormat("Błąd zamykania pozycji: %d, %s. Request: %s", 
         result.retcode, GetErrorDescription(result.retcode), requestDetails);
      WriteLog(errorMsg);
      Print(errorMsg);
      return false;
   }
}

//+------------------------------------------------------------------+
//| Oznaczenie sygnału jako przetworzonego                           |
//+------------------------------------------------------------------+
void MarkSignalAsProcessed(int signalId)
{
   // Wczytaj cały plik
   int tempHandle = FileOpen(fullFilePath, FILE_READ|FILE_CSV|FILE_ANSI|FILE_SHARE_READ|FILE_SHARE_WRITE);
   if(tempHandle == INVALID_HANDLE)
   {
      WriteLog(StringFormat("Błąd podczas otwierania pliku sygnałów dla oznaczenia ID=%d: %d", 
         signalId, GetLastError()));
      return;
   }
   
   string fileContent = "";
   string line;
   int currentId;
   
   // Przeczytaj nagłówki
   line = "";
   int columnsCount = 0;
   while(!FileIsLineEnding(tempHandle) && !FileIsEnding(tempHandle))
   {
      line += FileReadString(tempHandle) + ",";
      columnsCount++;
   }
   line = StringSubstr(line, 0, StringLen(line) - 1); // Usuń ostatni przecinek
   fileContent += line + "\n";
   
   // Pomiń koniec linii
   if(!FileIsEnding(tempHandle))
      FileReadString(tempHandle);
   
   // Przeczytaj pozostałe linie
   while(!FileIsEnding(tempHandle))
   {
      line = "";
      
      // Przeczytaj ID
      string idStr = FileReadString(tempHandle);
      currentId = (int)StringToInteger(idStr);
      line += idStr + ",";
      
      // Przeczytaj pozostałe kolumny oprócz ostatniej (processed)
      for(int i = 1; i < columnsCount - 1; i++)
      {
         line += FileReadString(tempHandle) + ",";
      }
      
      // Przeczytaj ostatnią kolumnę (processed) i zmodyfikuj, jeśli to jest nasz sygnał
      if(currentId == signalId)
      {
         FileReadString(tempHandle); // Pomiń aktualną wartość
         line += "1"; // Oznacz jako przetworzone
         WriteLog(StringFormat("Oznaczono sygnał ID=%d jako przetworzony", signalId));
      }
      else
      {
         line += FileReadString(tempHandle); // Zachowaj aktualną wartość
      }
      
      fileContent += line + "\n";
      
      // Pomiń koniec linii
      if(!FileIsEnding(tempHandle))
         FileReadString(tempHandle);
   }
   
   FileClose(tempHandle);
   
   // Zapisz zmodyfikowaną zawartość
   int writeHandle = FileOpen(fullFilePath, FILE_WRITE|FILE_CSV|FILE_ANSI);
   if(writeHandle == INVALID_HANDLE)
   {
      WriteLog(StringFormat("Błąd podczas otwierania pliku sygnałów do zapisu: %d", GetLastError()));
      return;
   }
   
   FileWriteString(writeHandle, fileContent);
   FileClose(writeHandle);
}

//+------------------------------------------------------------------+
//| Zapisanie komunikatu do pliku log                                |
//+------------------------------------------------------------------+
void WriteLog(string message)
{
   string logFilePath = SignalFilePath + "signal_trader_log.txt";
   int attempts = 0;
   int maxAttempts = 3;
   int logHandle = INVALID_HANDLE;
   
   // Próbuj otworzyć plik kilka razy w przypadku błędów
   while(attempts < maxAttempts && logHandle == INVALID_HANDLE)
   {
      logHandle = FileOpen(logFilePath, FILE_WRITE|FILE_READ|FILE_TXT|FILE_ANSI|FILE_SHARE_WRITE);
      if(logHandle == INVALID_HANDLE)
      {
         attempts++;
         Sleep(100);  // Poczekaj chwilę przed kolejną próbą
      }
   }
   
   if(logHandle != INVALID_HANDLE)
   {
      // Przesuń wskaźnik na koniec pliku
      FileSeek(logHandle, 0, SEEK_END);
      
      // Zapisz komunikat z datą i czasem
      string logEntry = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + " - " + message + "\n";
      FileWriteString(logHandle, logEntry);
      FileClose(logHandle);
   }
   else
   {
      // W przypadku problemu z zapisem do pliku, wypisz komunikat tylko do dziennika MT5
      Print("Błąd podczas zapisywania do logu: ", GetLastError(), " - ", message);
   }
}

//+------------------------------------------------------------------+
//| Pobierz opis błędu na podstawie kodu                             |
//+------------------------------------------------------------------+
string GetErrorDescription(int errorCode)
{
   switch(errorCode)
   {
      case 10004: return "Niepoprawne parametry żądania";
      case 10006: return "Brak połączenia z serwerem handlowym";
      case 10007: return "Operacja niedozwolona";
      case 10008: return "Za dużo żądań";
      case 10009: return "Nieprawidłowa operacja";
      case 10010: return "Nieprawidłowy kontrakt";
      case 10011: return "Nieprawidłowa konfiguracja żądania";
      case 10012: return "Nieprawidłowy typ zlecenia";
      case 10013: return "Nieprawidłowy wolumen";
      case 10014: return "Nieprawidłowa cena";
      case 10015: return "Nieprawidłowe stop loss";
      case 10016: return "Nieprawidłowe take profit";
      case 10017: return "Handel wyłączony";
      case 10018: return "Rynek zamknięty";
      case 10019: return "Niewystarczające środki";
      case 10020: return "Ceny się zmieniły";
      case 10021: return "Brak cen";
      case 10022: return "Nieprawidłowa data wygaśnięcia zlecenia";
      case 10023: return "Stan zlecenia się zmienił";
      case 10024: return "Za dużo żądań otwartych";
      case 10025: return "Serwer handlowy niedostępny";
      case 10026: return "Operacja anulowana przez brokera";
      case 10027: return "Błąd zamykania pozycji";
      case 10028: return "Błąd modyfikacji pozycji";
      case 10029: return "Broker odrzucił żądanie";
      case 10030: return "Nieobsługiwany tryb wypełnienia";
      case 10031: return "Brak połączenia";
      case 10032: return "Pozycja zablokowana";
      case 10033: return "Nieprawidłowa funkcja stop loss";
      case 10034: return "Nieprawidłowa pozycja";
      case 10041: return "Przekroczony maksymalny wolumen dla symbolu";
      case 10045: return "Niepoprawny kontekst handlowy";
      case 10043: return "Zablokowany z powodu naruszenia zasad";
      case 10050: return "Nieprawidłowy ticket";
      case 10052: return "Zlecenie już istnieje";
      default: return "Nieznany błąd: " + IntegerToString(errorCode);
   }
} 