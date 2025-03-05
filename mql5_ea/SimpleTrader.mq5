//+------------------------------------------------------------------+
//|                                                SimpleTrader.mq5 |
//|                                       MT5RemoteAI Trading System |
//+------------------------------------------------------------------+
#property copyright "MT5RemoteAI"
#property link      ""
#property version   "1.00"
#property strict

// Parametry wejściowe
input string   SignalFilePath = "C:\\Users\\win\\Documents\\mt5remoteai\\signals\\";  // Ścieżka do pliku sygnałów
input string   SignalFileName = "simple_test.csv";                                  // Nazwa pliku sygnałów
input int      CheckInterval = 1000;                                               // Interwał sprawdzania sygnałów (ms)

// Zmienne globalne
int fileHandle = INVALID_HANDLE;
string logFileName = "simple_test_log.txt";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Wypisz komunikat startowy
   Print("SimpleTrader EA uruchomiony. Wersja 1.00");
   
   // Inicjalizacja pliku logów
   WriteLog("========================");
   WriteLog("SimpleTrader uruchomiony");
   WriteLog("Data: " + TimeToString(TimeCurrent()));
   WriteLog("Plik sygnałów: " + SignalFilePath + SignalFileName);
   
   // Sprawdź, czy plik sygnałów istnieje
   if(FileIsExist(SignalFilePath + SignalFileName))
   {
      WriteLog("Plik sygnałów znaleziony");
   }
   else
   {
      WriteLog("BŁĄD: Plik sygnałów nie istnieje!");
   }
   
   // Sprawdź uprawnienia do pliku
   if(TestFileRead(SignalFilePath + SignalFileName))
   {
      WriteLog("Test odczytu pliku sygnałów: OK");
   }
   else
   {
      WriteLog("BŁĄD: Nie można odczytać pliku sygnałów!");
   }
   
   // Sprawdź uprawnienia do zapisu logów
   if(TestFileWrite(SignalFilePath + logFileName))
   {
      WriteLog("Test zapisu pliku logów: OK");
   }
   else
   {
      WriteLog("BŁĄD: Nie można zapisać do pliku logów!");
   }
   
   // Ustaw timer
   if(!EventSetMillisecondTimer(CheckInterval))
   {
      WriteLog("BŁĄD: Nie można ustawić timera!");
      return(INIT_FAILED);
   }
   
   WriteLog("Inicjalizacja zakończona pomyślnie.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   static datetime lastCheck = 0;
   datetime currentTime = TimeCurrent();
   
   // Sprawdzaj co 5 sekund
   if(currentTime - lastCheck < 5)
      return;
      
   lastCheck = currentTime;
   
   // Sprawdź, czy plik istnieje
   if(!FileIsExist(SignalFilePath + SignalFileName))
   {
      WriteLog("BŁĄD: Plik sygnałów nie istnieje! " + SignalFilePath + SignalFileName);
      return;
   }
   
   // Sprawdź, czy można otworzyć plik
   int handle = FileOpen(SignalFilePath + SignalFileName, FILE_READ|FILE_CSV|FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      WriteLog("BŁĄD: Nie można otworzyć pliku sygnałów! Kod błędu: " + IntegerToString(GetLastError()));
      return;
   }
   
   // Zamknij plik
   FileClose(handle);
   WriteLog("Test odczytu pliku o " + TimeToString(currentTime) + ": OK");
   
   // Spróbuj odczytać zawartość pliku
   ReadSignalFile();
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Zatrzymaj timer
   EventKillTimer();
   
   // Zapisz informację o zatrzymaniu
   WriteLog("SimpleTrader zatrzymany. Przyczyna: " + IntegerToString(reason));
}

//+------------------------------------------------------------------+
//| Testuj możliwość odczytu pliku                                   |
//+------------------------------------------------------------------+
bool TestFileRead(string filePath)
{
   int handle = FileOpen(filePath, FILE_READ|FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("Nie można otworzyć pliku do odczytu: ", filePath, ". Kod błędu: ", GetLastError());
      return false;
   }
   
   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
//| Testuj możliwość zapisu do pliku                                 |
//+------------------------------------------------------------------+
bool TestFileWrite(string filePath)
{
   int handle = FileOpen(filePath, FILE_WRITE|FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("Nie można otworzyć pliku do zapisu: ", filePath, ". Kod błędu: ", GetLastError());
      return false;
   }
   
   FileWriteString(handle, "Test zapisu: " + TimeToString(TimeCurrent()) + "\n");
   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
//| Zapisz komunikat do pliku logów                                  |
//+------------------------------------------------------------------+
void WriteLog(string message)
{
   // Wypisz na ekran
   Print(message);
   
   // Zapisz do pliku
   int handle = FileOpen(SignalFilePath + logFileName, FILE_WRITE|FILE_READ|FILE_ANSI|FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      Print("Błąd podczas otwierania pliku logów: ", GetLastError());
      return;
   }
   
   // Przeczytaj istniejącą zawartość
   string content = "";
   FileSeek(handle, 0, SEEK_SET);
   while(!FileIsEnding(handle))
   {
      content += FileReadString(handle) + "\n";
   }
   
   // Zapisz z powrotem z nowym komunikatem
   FileSeek(handle, 0, SEEK_SET);
   FileWriteString(handle, content);
   FileWriteString(handle, TimeToString(TimeCurrent()) + " - " + message + "\n");
   
   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Odczytaj plik sygnałów                                          |
//+------------------------------------------------------------------+
void ReadSignalFile()
{
   int handle = FileOpen(SignalFilePath + SignalFileName, FILE_READ|FILE_CSV|FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      WriteLog("Nie można otworzyć pliku sygnałów do odczytu!");
      return;
   }
   
   // Pomijamy nagłówek
   string header = "";
   for(int i=0; i<11; i++)
   {
      if(!FileIsEnding(handle))
         header += FileReadString(handle) + ",";
   }
   
   WriteLog("Nagłówek pliku: " + header);
   
   // Przejdź do następnej linii
   if(!FileIsLineEnding(handle))
      FileReadString(handle);
   
   // Czytaj pierwszy sygnał
   if(!FileIsEnding(handle))
   {
      string id = FileReadString(handle);
      string timestamp = FileReadString(handle);
      string symbol = FileReadString(handle);
      string action = FileReadString(handle);
      string volume = FileReadString(handle);
      
      string signalInfo = StringFormat("Znaleziono sygnał: ID=%s, Time=%s, Symbol=%s, Action=%s, Volume=%s", 
         id, timestamp, symbol, action, volume);
      WriteLog(signalInfo);
   }
   else
   {
      WriteLog("Brak sygnałów w pliku.");
   }
   
   FileClose(handle);
} 