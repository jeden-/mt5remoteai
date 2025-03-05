//+------------------------------------------------------------------+
//|                                              SignalTraderFix.mqh |
//|                                      MT5RemoteAI Trading System  |
//+------------------------------------------------------------------+
#property copyright "MT5RemoteAI"
#property link      ""
#property version   "1.03"

// Flagi dla operacji plikowych
#define FILE_FLAGS_CSV (FILE_CSV|FILE_READ|FILE_WRITE|FILE_ANSI)
#define FILE_FLAGS_TXT (FILE_TXT|FILE_READ|FILE_WRITE|FILE_ANSI)

// Struktura sygnału handlowego
struct TradeSignal
{
   int         id;             // ID sygnału
   string      symbol;         // Symbol instrumentu
   string      action;         // BUY, SELL, CLOSE
   double      volume;         // Wolumen w lotach
   double      price;          // Cena (0 = rynkowa)
   int         slPoints;       // Stop Loss w punktach
   int         tpPoints;       // Take Profit w punktach
   int         ticket;         // Ticket pozycji (dla CLOSE)
   string      comment;        // Komentarz
   datetime    time;           // Czas sygnału
   bool        processed;      // Czy został przetworzony
};

//+------------------------------------------------------------------+
//| Funkcja testująca dostęp do plików                               |
//+------------------------------------------------------------------+
bool TestFileAccess(string path, string logFileName)
{
   string testLogPath = path + "test_access.tmp";
   int testHandle = FileOpen(testLogPath, FILE_FLAGS_TXT);
   
   if(testHandle != INVALID_HANDLE)
   {
      FileWrite(testHandle, "Test dostępu do plików: " + TimeToString(TimeCurrent()));
      FileClose(testHandle);
      FileDelete(testLogPath);
      WriteToLog(logFileName, "Test dostępu do plików: Sukces");
      return true;
   }
   else
   {
      WriteToLog(logFileName, "Test dostępu do plików: Błąd. Kod: " + IntegerToString(GetLastError()));
      return false;
   }
}

//+------------------------------------------------------------------+
//| Zapisuje komunikat do pliku logów                                |
//+------------------------------------------------------------------+
void WriteToLog(string logFileName, string message)
{
   string logPath = "C:\\Users\\win\\Documents\\mt5remoteai\\signals\\" + logFileName;
   int fileHandle = FileOpen(logPath, FILE_FLAGS_TXT|FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_ANSI);
   
   if(fileHandle != INVALID_HANDLE)
   {
      FileSeek(fileHandle, 0, SEEK_END);
      FileWrite(fileHandle, TimeToString(TimeCurrent()) + ": " + message);
      FileClose(fileHandle);
   }
   else
   {
      Print("Błąd podczas zapisywania do logu: " + IntegerToString(GetLastError()));
   }
}

//+------------------------------------------------------------------+
//| Zamyka pozycję po numerze ticketu                                |
//+------------------------------------------------------------------+
bool ClosePositionByTicket(int ticket, string comment = "")
{
   // Sprawdź, czy pozycja istnieje
   if(!PositionSelectByTicket(ticket))
   {
      Print("Błąd: Nie można znaleźć pozycji o numerze " + IntegerToString(ticket));
      return false;
   }
   
   // Przygotuj strukturę żądania
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   // Pobierz dane pozycji
   string symbol = PositionGetString(POSITION_SYMBOL);
   double volume = PositionGetDouble(POSITION_VOLUME);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   
   // Ustawienia żądania
   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = symbol;
   request.volume = volume;
   request.deviation = 10;
   request.magic = 123456;
   
   if(comment != "")
      request.comment = comment;
   
   // Ustaw typ przeciwny do otwartej pozycji
   if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
   {
      request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
      request.type = ORDER_TYPE_SELL;
   }
   else
   {
      request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
      request.type = ORDER_TYPE_BUY;
   }
   
   // Wyślij żądanie
   bool success = OrderSend(request, result);
   
   if(success && result.retcode == TRADE_RETCODE_DONE)
   {
      Print("Pozycja #" + IntegerToString(ticket) + " zamknięta pomyślnie");
      return true;
   }
   else
   {
      Print("Błąd podczas zamykania pozycji #" + IntegerToString(ticket) + 
             ". Kod: " + IntegerToString(result.retcode));
      return false;
   }
}

//+------------------------------------------------------------------+
//| Otwiera nową pozycję                                             |
//+------------------------------------------------------------------+
bool OpenPosition(string symbol, string actionType, double volume, double requestedPrice,
                 int slPoints, int tpPoints, string comment = "")
{
   // Ustaw typ zlecenia i pobierz aktualną cenę
   ENUM_ORDER_TYPE orderType;
   double price;
   
   if(actionType == "BUY")
   {
      orderType = ORDER_TYPE_BUY;
      price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   }
   else if(actionType == "SELL")
   {
      orderType = ORDER_TYPE_SELL;
      price = SymbolInfoDouble(symbol, SYMBOL_BID);
   }
   else
   {
      Print("Nieznana akcja: " + actionType);
      return false;
   }
   
   // Użyj konkretnej ceny, jeśli została podana
   if(requestedPrice > 0)
   {
      price = requestedPrice;
   }
   
   // Ustawienie Stop Loss i Take Profit
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double sl = 0, tp = 0;
   
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
   request.symbol = symbol;
   request.volume = volume;
   request.type = orderType;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = 123456;
   
   if(comment != "")
      request.comment = comment;
   
   // Standardowe ustawienia dla MT5
   request.type_filling = ORDER_FILLING_FOK;
   request.type_time = ORDER_TIME_GTC;
   
   // Wyślij żądanie
   bool success = OrderSend(request, result);
   
   if(success && result.retcode == TRADE_RETCODE_DONE)
   {
      Print(actionType + " pozycja dla " + symbol + 
            " otwarta pomyślnie. Ticket: " + IntegerToString(result.order));
      return true;
   }
   else
   {
      Print("Błąd podczas otwierania pozycji " + symbol + 
            ". Kod: " + IntegerToString(result.retcode));
      return false;
   }
}

//+------------------------------------------------------------------+
//| Zwraca opis błędu                                               |
//+------------------------------------------------------------------+
string GetErrorDescription(int errorCode)
{
   string errorDescription;
   
   switch(errorCode)
   {
      case TRADE_RETCODE_REQUOTE:
         errorDescription = "Rekwotacja";
         break;
      case TRADE_RETCODE_REJECT:
         errorDescription = "Żądanie odrzucone";
         break;
      case TRADE_RETCODE_CANCEL:
         errorDescription = "Żądanie anulowane przez tradera";
         break;
      case TRADE_RETCODE_PLACED:
         errorDescription = "Zlecenie złożone";
         break;
      case TRADE_RETCODE_DONE:
         errorDescription = "Żądanie wykonane";
         break;
      case TRADE_RETCODE_DONE_PARTIAL:
         errorDescription = "Żądanie wykonane częściowo";
         break;
      case TRADE_RETCODE_ERROR:
         errorDescription = "Błąd przetwarzania żądania";
         break;
      case TRADE_RETCODE_TIMEOUT:
         errorDescription = "Upłynął czas oczekiwania na żądanie";
         break;
      case TRADE_RETCODE_INVALID:
         errorDescription = "Nieprawidłowe żądanie";
         break;
      case TRADE_RETCODE_INVALID_VOLUME:
         errorDescription = "Nieprawidłowy wolumen w żądaniu";
         break;
      case TRADE_RETCODE_INVALID_PRICE:
         errorDescription = "Nieprawidłowa cena w żądaniu";
         break;
      case TRADE_RETCODE_INVALID_STOPS:
         errorDescription = "Nieprawidłowe poziomy stop w żądaniu";
         break;
      case TRADE_RETCODE_TRADE_DISABLED:
         errorDescription = "Handel wyłączony";
         break;
      case TRADE_RETCODE_MARKET_CLOSED:
         errorDescription = "Rynek zamknięty";
         break;
      case TRADE_RETCODE_NO_MONEY:
         errorDescription = "Brak wystarczających środków do wykonania żądania";
         break;
      case TRADE_RETCODE_PRICE_CHANGED:
         errorDescription = "Ceny zmieniły się";
         break;
      case TRADE_RETCODE_PRICE_OFF:
         errorDescription = "Brak wycen dla przetworzenia żądania";
         break;
      case TRADE_RETCODE_INVALID_EXPIRATION:
         errorDescription = "Nieprawidłowa data wygaśnięcia zlecenia";
         break;
      case TRADE_RETCODE_ORDER_CHANGED:
         errorDescription = "Stan zlecenia zmienił się";
         break;
      case TRADE_RETCODE_TOO_MANY_REQUESTS:
         errorDescription = "Zbyt wiele żądań";
         break;
      case TRADE_RETCODE_NO_CHANGES:
         errorDescription = "Brak zmian w żądaniu";
         break;
      case TRADE_RETCODE_SERVER_DISABLES_AT:
         errorDescription = "Autotrading wyłączony przez serwer";
         break;
      case TRADE_RETCODE_CLIENT_DISABLES_AT:
         errorDescription = "Autotrading wyłączony przez terminal klienta";
         break;
      case TRADE_RETCODE_LOCKED:
         errorDescription = "Żądanie zablokowane do przetworzenia";
         break;
      case TRADE_RETCODE_FROZEN:
         errorDescription = "Zlecenie lub pozycja zamrożone";
         break;
      case TRADE_RETCODE_INVALID_FILL:
         errorDescription = "Nieprawidłowy typ wykonania zlecenia";
         break;
      case TRADE_RETCODE_CONNECTION:
         errorDescription = "Brak połączenia z serwerem tradingowym";
         break;
      case TRADE_RETCODE_ONLY_REAL:
         errorDescription = "Operacja dozwolona tylko dla rachunków rzeczywistych";
         break;
      case TRADE_RETCODE_LIMIT_ORDERS:
         errorDescription = "Osiągnięto limit liczby zleceń aktywnych";
         break;
      case TRADE_RETCODE_LIMIT_VOLUME:
         errorDescription = "Osiągnięto limit wolumenu dla tego instrumentu";
         break;
      default:
         errorDescription = "Nieznany błąd " + IntegerToString(errorCode);
   }
   
   return errorDescription;
} 