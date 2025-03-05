//+------------------------------------------------------------------+
//|                                           SignalTrader_Fixed.mq5 |
//|                                       MT5RemoteAI Trading System  |
//+------------------------------------------------------------------+
#property copyright "MT5RemoteAI"
#property link      ""
#property version   "1.03"
#property strict

// Dołącz plik nagłówkowy - zastąp poniższą ścieżkę rzeczywistą ścieżką po skopiowaniu
#include "SignalTraderFix.mqh"

// Parametry wejściowe
input string   SignalPath = "C:\\Users\\win\\Documents\\mt5remoteai\\signals\\";  // Ścieżka do plików sygnałów
input string   SignalFile = "trading_signals.csv";                               // Plik z sygnałami
input string   LogFile = "signal_trader_log.txt";                                // Plik logów
input int      CheckInterval = 1000;                                             // Interwał sprawdzania (ms)
input double   DefaultVolume = 0.01;                                             // Domyślny wolumen (gdy 0 w CSV)
input int      DefaultSL = 100;                                                  // Domyślny Stop Loss w punktach
input int      DefaultTP = 200;                                                  // Domyślny Take Profit w punktach
input bool     EnableDiagnostics = true;                                         // Włącz diagnostykę
input bool     EnableVerboseLogging = true;                                      // Szczegółowe logowanie

// Globalne zmienne
int csvHandle = INVALID_HANDLE;
datetime lastModified = 0;
datetime lastFileCheck = 0;
bool isFirstRun = true;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Logowanie startu EA
   string welcome = "SignalTrader v1.03 uruchomiony.";
   Print(welcome);
   WriteToLog(LogFile, welcome);
   
   // Test dostępu do plików
   bool fileAccessOk = TestFileAccess(SignalPath, LogFile);
   if(!fileAccessOk)
   {
      Print("UWAGA: Problemy z dostępem do plików!");
      WriteToLog(LogFile, "UWAGA: Problemy z dostępem do plików! Sprawdź uprawnienia.");
   }
   
   // Test dostępu do pliku sygnałów
   string fullSignalPath = SignalPath + SignalFile;
   WriteToLog(LogFile, "Sprawdzanie pliku sygnałów: " + fullSignalPath);
   
   if(FileIsExist(fullSignalPath))
   {
      WriteToLog(LogFile, "Plik sygnałów istnieje.");
      
      // Sprawdź datę modyfikacji
      datetime fileTime = (datetime)FileGetInteger(fullSignalPath, FILE_MODIFY_DATE);
      WriteToLog(LogFile, "Ostatnia modyfikacja pliku: " + TimeToString(fileTime));
      
      // Sprawdź zawartość pliku
      if(EnableDiagnostics)
      {
         AnalyzeSignalFile(fullSignalPath);
      }
   }
   else
   {
      WriteToLog(LogFile, "UWAGA: Plik sygnałów nie istnieje!");
      
      // Spróbuj utworzyć pusty plik CSV
      int tempHandle = FileOpen(fullSignalPath, FILE_FLAGS_CSV);
      if(tempHandle != INVALID_HANDLE)
      {
         // Zapisz nagłówki
         FileWrite(tempHandle, "id", "timestamp", "symbol", "action", "volume", "price", 
                  "sl_points", "tp_points", "ticket", "comment", "processed");
         FileClose(tempHandle);
         WriteToLog(LogFile, "Utworzono pusty plik sygnałów.");
      }
      else
      {
         WriteToLog(LogFile, "BŁĄD: Nie można utworzyć pliku sygnałów! Kod: " + IntegerToString(GetLastError()));
      }
   }
   
   // Ustaw timer
   if(!EventSetMillisecondTimer(CheckInterval))
   {
      WriteToLog(LogFile, "BŁĄD: Nie można ustawić timera!");
      return INIT_FAILED;
   }
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Zatrzymaj timer
   EventKillTimer();
   
   // Zamknij plik jeśli otwarty
   if(csvHandle != INVALID_HANDLE)
   {
      FileClose(csvHandle);
      csvHandle = INVALID_HANDLE;
   }
   
   // Logowanie zatrzymania
   WriteToLog(LogFile, "SignalTrader zatrzymany. Powód: " + IntegerToString(reason));
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Sprawdź czas i sprawdź plik co CheckInterval milisekund
   datetime currentTime = TimeCurrent();
   if(currentTime - lastFileCheck < CheckInterval/1000 && !isFirstRun)
      return;
      
   lastFileCheck = currentTime;
   
   // Pełna ścieżka do pliku sygnałów
   string fullSignalPath = SignalPath + SignalFile;
   
   // Sprawdź, czy plik istnieje
   if(!FileIsExist(fullSignalPath))
   {
      if(EnableVerboseLogging || isFirstRun)
      {
         WriteToLog(LogFile, "UWAGA: Plik sygnałów nie istnieje: " + fullSignalPath);
      }
      isFirstRun = false;
      return;
   }
   
   // Sprawdź datę modyfikacji
   datetime fileTime = (datetime)FileGetInteger(fullSignalPath, FILE_MODIFY_DATE);
   
   // Jeśli plik nie był modyfikowany od ostatniego sprawdzenia, pomiń
   if(fileTime <= lastModified && !isFirstRun)
   {
      if(EnableVerboseLogging)
      {
         WriteToLog(LogFile, "Plik sygnałów nie był modyfikowany.");
      }
      return;
   }
   
   // Plik został zmodyfikowany, przetwórz sygnały
   lastModified = fileTime;
   
   if(EnableVerboseLogging || isFirstRun)
   {
      WriteToLog(LogFile, "Znaleziono nowe sygnały. Ostatnia modyfikacja: " + TimeToString(fileTime));
   }
   
   // Przetwórz sygnały
   ProcessSignals(fullSignalPath);
   
   isFirstRun = false;
}

//+------------------------------------------------------------------+
//| Funkcja przetwarzająca sygnały z pliku CSV                       |
//+------------------------------------------------------------------+
void ProcessSignals(string filePath)
{
   WriteToLog(LogFile, "Przetwarzanie sygnałów z pliku: " + filePath);
   
   // Otwórz plik
   int fileHandle = FileOpen(filePath, FILE_FLAGS_CSV);
   if(fileHandle == INVALID_HANDLE)
   {
      WriteToLog(LogFile, "BŁĄD: Nie można otworzyć pliku sygnałów. Kod: " + IntegerToString(GetLastError()));
      return;
   }
   
   // Pomiń nagłówki
   if(!FileIsEnding(fileHandle))
   {
      for(int i = 0; i < 11; i++)
      {
         if(!FileIsEnding(fileHandle))
            FileReadString(fileHandle);
      }
      
      // Przejdź do następnej linii
      if(!FileIsLineEnding(fileHandle) && !FileIsEnding(fileHandle))
         FileReadString(fileHandle);
   }
   
   // Zliczaj przetworzone sygnały
   int totalSignals = 0;
   int processedSignals = 0;
   int successSignals = 0;
   
   // Odczytaj sygnały
   while(!FileIsEnding(fileHandle))
   {
      TradeSignal signal;
      bool validSignal = true;
      
      // Odczytaj ID
      if(FileIsEnding(fileHandle)) break;
      string idStr = FileReadString(fileHandle);
      signal.id = (int)StringToInteger(idStr);
      
      // Odczytaj timestamp
      if(FileIsEnding(fileHandle)) break;
      string timestampStr = FileReadString(fileHandle);
      signal.time = StringToTime(timestampStr);
      
      // Jeśli nieprawidłowa data, spróbuj różne podejścia
      if(signal.time == 0)
      {
         // W MT5 nie możemy używać parametru formatu w StringToTime
         // Zamiast tego spróbujmy kilku popularnych formatów
         string altTimestampStr;
         
         // Zamień format YYYY-MM-DD na DD.MM.YYYY (który jest domyślny dla MT5)
         if(StringLen(timestampStr) >= 10)
         {
            if(StringSubstr(timestampStr, 4, 1) == "-" && StringSubstr(timestampStr, 7, 1) == "-")
            {
               // Format YYYY-MM-DD HH:MM:SS
               string year = StringSubstr(timestampStr, 0, 4);
               string month = StringSubstr(timestampStr, 5, 2);
               string day = StringSubstr(timestampStr, 8, 2);
               string timeStr = "";
               
               if(StringLen(timestampStr) > 10)
                  timeStr = StringSubstr(timestampStr, 10);
                  
               altTimestampStr = day + "." + month + "." + year + timeStr;
               signal.time = StringToTime(altTimestampStr);
            }
         }
         
         // Jeśli nadal nie działa, użyj bieżącego czasu
         if(signal.time == 0)
         {
            WriteToLog(LogFile, "OSTRZEŻENIE: Nieprawidłowy format daty: " + timestampStr + 
                        " w sygnale #" + idStr + ". Próbowano: " + altTimestampStr);
            signal.time = TimeCurrent(); // Użyj bieżącego czasu jako alternatywy
         }
      }
      
      // Odczytaj symbol
      if(FileIsEnding(fileHandle)) break;
      signal.symbol = FileReadString(fileHandle);
      
      // Odczytaj akcję
      if(FileIsEnding(fileHandle)) break;
      signal.action = FileReadString(fileHandle);
      
      // Odczytaj wolumen
      if(FileIsEnding(fileHandle)) break;
      string volumeStr = FileReadString(fileHandle);
      signal.volume = StringToDouble(volumeStr);
      
      // Jeśli wolumen jest 0, użyj domyślnego
      if(signal.volume <= 0)
      {
         signal.volume = DefaultVolume;
         if(EnableVerboseLogging)
         {
            WriteToLog(LogFile, "Użyto domyślnego wolumenu dla sygnału #" + idStr);
         }
      }
      
      // Odczytaj cenę
      if(FileIsEnding(fileHandle)) break;
      string priceStr = FileReadString(fileHandle);
      signal.price = StringToDouble(priceStr);
      
      // Odczytaj SL
      if(FileIsEnding(fileHandle)) break;
      string slStr = FileReadString(fileHandle);
      signal.slPoints = (int)StringToInteger(slStr);
      
      // Jeśli SL jest 0, użyj domyślnego
      if(signal.slPoints <= 0)
      {
         signal.slPoints = DefaultSL;
      }
      
      // Odczytaj TP
      if(FileIsEnding(fileHandle)) break;
      string tpStr = FileReadString(fileHandle);
      signal.tpPoints = (int)StringToInteger(tpStr);
      
      // Jeśli TP jest 0, użyj domyślnego
      if(signal.tpPoints <= 0)
      {
         signal.tpPoints = DefaultTP;
      }
      
      // Odczytaj ticket
      if(FileIsEnding(fileHandle)) break;
      string ticketStr = FileReadString(fileHandle);
      signal.ticket = (int)StringToInteger(ticketStr);
      
      // Odczytaj komentarz
      if(FileIsEnding(fileHandle)) break;
      signal.comment = FileReadString(fileHandle);
      
      // Odczytaj status przetworzenia
      if(FileIsEnding(fileHandle)) break;
      string processedStr = FileReadString(fileHandle);
      signal.processed = (bool)StringToInteger(processedStr);
      
      totalSignals++;
      
      // Przetworz sygnał jeśli nie był jeszcze przetworzony
      if(!signal.processed)
      {
         string signalInfo = StringFormat("Sygnał #%d: %s %s, Wolumen: %.2f, Czas: %s", 
                               signal.id, signal.action, signal.symbol, signal.volume,
                               TimeToString(signal.time));
                               
         WriteToLog(LogFile, "Przetwarzanie: " + signalInfo);
         
         bool success = false;
         
         // Wykonaj odpowiednią akcję
         if(signal.action == "BUY" || signal.action == "SELL")
         {
            success = ExecuteTradeSignal(signal);
         }
         else if(signal.action == "CLOSE")
         {
            success = CloseTradeSignal(signal);
         }
         else
         {
            WriteToLog(LogFile, "BŁĄD: Nieznana akcja: " + signal.action + " w sygnale #" + IntegerToString(signal.id));
         }
         
         processedSignals++;
         
         if(success)
         {
            successSignals++;
            
            // Oznacz sygnał jako przetworzony w pliku
            signal.processed = true;
            UpdateSignalStatus(filePath, signal.id, true);
         }
      }
      else
      {
         if(EnableVerboseLogging)
         {
            WriteToLog(LogFile, "Pominięto już przetworzony sygnał #" + IntegerToString(signal.id));
         }
      }
      
      // Przejdź do następnej linii
      if(!FileIsLineEnding(fileHandle) && !FileIsEnding(fileHandle))
         FileReadString(fileHandle);
   }
   
   // Zamknij plik
   FileClose(fileHandle);
   
   if(totalSignals > 0)
   {
      WriteToLog(LogFile, StringFormat("Przetwarzanie zakończone. Łącznie: %d, Przetworzono: %d, Udane: %d", 
                              totalSignals, processedSignals, successSignals));
   }
   else
   {
      if(EnableVerboseLogging || isFirstRun)
      {
         WriteToLog(LogFile, "Brak sygnałów do przetworzenia w pliku.");
      }
   }
}

//+------------------------------------------------------------------+
//| Funkcja wykonująca sygnał handlowy (BUY/SELL)                    |
//+------------------------------------------------------------------+
bool ExecuteTradeSignal(TradeSignal &signal)
{
   WriteToLog(LogFile, StringFormat("Wykonywanie sygnału #%d: %s %s, Wolumen: %.2f", 
                           signal.id, signal.action, signal.symbol, signal.volume));
   
   // Sprawdź, czy symbol jest dostępny
   if(!SymbolSelect(signal.symbol, true))
   {
      WriteToLog(LogFile, "BŁĄD: Nie można wybrać symbolu: " + signal.symbol + 
                  ". Kod: " + IntegerToString(GetLastError()));
      return false;
   }
   
   // Otwórz pozycję
   bool result = OpenPosition(signal.symbol, signal.action, signal.volume, signal.price,
                              signal.slPoints, signal.tpPoints, "Signal #" + IntegerToString(signal.id));
   
   if(result)
   {
      WriteToLog(LogFile, "SUKCES: Wykonano sygnał #" + IntegerToString(signal.id));
   }
   else
   {
      WriteToLog(LogFile, "BŁĄD: Nie można wykonać sygnału #" + IntegerToString(signal.id));
   }
   
   return result;
}

//+------------------------------------------------------------------+
//| Funkcja zamykająca pozycję na podstawie sygnału                  |
//+------------------------------------------------------------------+
bool CloseTradeSignal(TradeSignal &signal)
{
   // Sprawdź, czy mamy numer ticketu
   if(signal.ticket <= 0)
   {
      WriteToLog(LogFile, "BŁĄD: Brak numeru ticketu dla sygnału zamknięcia #" + 
                  IntegerToString(signal.id));
      return false;
   }
   
   WriteToLog(LogFile, "Zamykanie pozycji #" + IntegerToString(signal.ticket) + 
                " dla sygnału #" + IntegerToString(signal.id));
   
   // Zamknij pozycję
   bool result = ClosePositionByTicket(signal.ticket, "Close by Signal #" + IntegerToString(signal.id));
   
   if(result)
   {
      WriteToLog(LogFile, "SUKCES: Zamknięto pozycję #" + IntegerToString(signal.ticket) + 
                  " dla sygnału #" + IntegerToString(signal.id));
   }
   else
   {
      WriteToLog(LogFile, "BŁĄD: Nie można zamknąć pozycji #" + IntegerToString(signal.ticket) + 
                  " dla sygnału #" + IntegerToString(signal.id));
   }
   
   return result;
}

//+------------------------------------------------------------------+
//| Funkcja aktualizująca status przetworzenia sygnału w pliku CSV   |
//+------------------------------------------------------------------+
void UpdateSignalStatus(string filePath, int signalId, bool processed)
{
   // Stwórz tymczasowy plik
   string tempFile = SignalPath + "temp_signals.csv";
   
   // Otwórz plik źródłowy do odczytu
   int sourceHandle = FileOpen(filePath, FILE_FLAGS_CSV);
   if(sourceHandle == INVALID_HANDLE)
   {
      WriteToLog(LogFile, "BŁĄD: Nie można otworzyć pliku źródłowego do aktualizacji. Kod: " + 
                  IntegerToString(GetLastError()));
      return;
   }
   
   // Otwórz plik tymczasowy do zapisu
   int tempHandle = FileOpen(tempFile, FILE_FLAGS_CSV);
   if(tempHandle == INVALID_HANDLE)
   {
      WriteToLog(LogFile, "BŁĄD: Nie można utworzyć pliku tymczasowego. Kod: " + 
                  IntegerToString(GetLastError()));
      FileClose(sourceHandle);
      return;
   }
   
   // Skopiuj nagłówki
   string headers[11];
   for(int i = 0; i < 11; i++)
   {
      if(!FileIsEnding(sourceHandle))
         headers[i] = FileReadString(sourceHandle);
      else
         headers[i] = "";
   }
   
   // Zapisz nagłówki do pliku tymczasowego
   FileWrite(tempHandle, headers[0], headers[1], headers[2], headers[3], headers[4], 
              headers[5], headers[6], headers[7], headers[8], headers[9], headers[10]);
   
   // Przejdź do następnej linii w pliku źródłowym
   if(!FileIsLineEnding(sourceHandle) && !FileIsEnding(sourceHandle))
      FileReadString(sourceHandle);
   
   // Kopiuj linie, aktualizując status przetworzenia dla wybranego ID
   while(!FileIsEnding(sourceHandle))
   {
      string fields[11];
      
      // Odczytaj wszystkie pola
      for(int i = 0; i < 11; i++)
      {
         if(!FileIsEnding(sourceHandle))
            fields[i] = FileReadString(sourceHandle);
         else
            fields[i] = "";
      }
      
      // Sprawdź, czy to szukane ID
      int lineId = (int)StringToInteger(fields[0]);
      if(lineId == signalId)
      {
         // Aktualizuj status przetworzenia
         fields[10] = processed ? "1" : "0";
         
         if(EnableVerboseLogging)
         {
            WriteToLog(LogFile, "Aktualizacja statusu dla sygnału #" + IntegerToString(signalId) + 
                        " na " + (processed ? "przetworzony" : "nieprzetworzony"));
         }
      }
      
      // Zapisz pola do pliku tymczasowego
      FileWrite(tempHandle, fields[0], fields[1], fields[2], fields[3], fields[4], 
                 fields[5], fields[6], fields[7], fields[8], fields[9], fields[10]);
      
      // Przejdź do następnej linii
      if(!FileIsLineEnding(sourceHandle) && !FileIsEnding(sourceHandle))
         FileReadString(sourceHandle);
   }
   
   // Zamknij oba pliki
   FileClose(sourceHandle);
   FileClose(tempHandle);
   
   // Usuń oryginalny plik
   if(!FileDelete(filePath))
   {
      WriteToLog(LogFile, "OSTRZEŻENIE: Nie można usunąć oryginalnego pliku. Kod: " + 
                  IntegerToString(GetLastError()));
      return;
   }
   
   // Zmień nazwę pliku tymczasowego - poprawiona sygnatura FileMove z 4 parametrami
   // FileMove(źródło, flagi_źródła, cel, flagi_celu)
   if(!FileMove(tempFile, 0, filePath, 0))
   {
      WriteToLog(LogFile, "BŁĄD: Nie można zmienić nazwy pliku tymczasowego. Kod: " + 
                  IntegerToString(GetLastError()));
   }
}

//+------------------------------------------------------------------+
//| Funkcja analizująca zawartość pliku sygnałów (diagnostyka)       |
//+------------------------------------------------------------------+
void AnalyzeSignalFile(string filePath)
{
   WriteToLog(LogFile, "Analiza pliku sygnałów: " + filePath);
   
   // Otwórz plik
   int handle = FileOpen(filePath, FILE_FLAGS_CSV);
   if(handle == INVALID_HANDLE)
   {
      WriteToLog(LogFile, "BŁĄD: Nie można otworzyć pliku do analizy. Kod: " + 
                  IntegerToString(GetLastError()));
      return;
   }
   
   // Sprawdź nagłówki
   string expectedHeaders[11] = {
      "id", "timestamp", "symbol", "action", "volume", 
      "price", "sl_points", "tp_points", "ticket", "comment", "processed"
   };
   
   string actualHeaders[11];
   for(int i = 0; i < 11; i++)
   {
      if(!FileIsEnding(handle))
         actualHeaders[i] = FileReadString(handle);
      else
         actualHeaders[i] = "";
   }
   
   bool headersOk = true;
   for(int i = 0; i < 11; i++)
   {
      if(StringCompare(actualHeaders[i], expectedHeaders[i], false) != 0)
      {
         WriteToLog(LogFile, "OSTRZEŻENIE: Nieprawidłowy nagłówek #" + IntegerToString(i) + 
                     ". Oczekiwano: '" + expectedHeaders[i] + "', znaleziono: '" + actualHeaders[i] + "'");
         headersOk = false;
      }
   }
   
   if(headersOk)
   {
      WriteToLog(LogFile, "Nagłówki pliku CSV są poprawne.");
   }
   
   // Przejdź do następnej linii
   if(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);
   
   // Policz wiersze
   int totalRows = 0;
   int validRows = 0;
   int buySignals = 0;
   int sellSignals = 0;
   int closeSignals = 0;
   int processedSignals = 0;
   
   // Przeanalizuj zawartość pliku
   while(!FileIsEnding(handle))
   {
      totalRows++;
      bool rowValid = true;
      
      // Odczytaj wszystkie pola
      string fields[11];
      for(int i = 0; i < 11; i++)
      {
         if(!FileIsEnding(handle))
            fields[i] = FileReadString(handle);
         else
         {
            fields[i] = "";
            rowValid = false;
         }
      }
      
      // Sprawdź poprawność wiersza
      if(rowValid)
      {
         validRows++;
         
         // Sprawdź akcję
         if(fields[3] == "BUY")
            buySignals++;
         else if(fields[3] == "SELL")
            sellSignals++;
         else if(fields[3] == "CLOSE")
            closeSignals++;
         
         // Sprawdź status przetworzenia
         if(StringToInteger(fields[10]) > 0)
            processedSignals++;
         
         // Sprawdź datę
         datetime signalTime = StringToTime(fields[1]);
         if(signalTime == 0)
         {
            WriteToLog(LogFile, "OSTRZEŻENIE: Nieprawidłowy format daty w wierszu #" + 
                        IntegerToString(totalRows) + ": " + fields[1]);
         }
         else
         {
            // Sprawdź datę w przyszłości (rok > 2024)
            MqlDateTime dt;
            TimeToStruct(signalTime, dt);
            if(dt.year > 2024)
            {
               WriteToLog(LogFile, "OSTRZEŻENIE: Data w przyszłości w wierszu #" + 
                           IntegerToString(totalRows) + ": " + fields[1]);
            }
         }
         
         // Sprawdź wolumen
         double volume = StringToDouble(fields[4]);
         if(volume <= 0)
         {
            WriteToLog(LogFile, "OSTRZEŻENIE: Brak wolumenu w wierszu #" + 
                        IntegerToString(totalRows) + ": " + fields[4]);
         }
      }
      
      // Przejdź do następnej linii
      if(!FileIsLineEnding(handle) && !FileIsEnding(handle))
         FileReadString(handle);
   }
   
   // Zamknij plik
   FileClose(handle);
   
   // Wypisz statystyki
   WriteToLog(LogFile, StringFormat("Statystyki pliku: Wszystkie wiersze: %d, Poprawne: %d", 
                           totalRows, validRows));
   WriteToLog(LogFile, StringFormat("Sygnały: BUY: %d, SELL: %d, CLOSE: %d, Przetworzone: %d", 
                           buySignals, sellSignals, closeSignals, processedSignals));
} 