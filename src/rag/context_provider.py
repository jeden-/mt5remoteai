"""Moduł do dostarczania kontekstu rynkowego dla systemu RAG."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from .market_memory import MarketMemory
from .embeddings_handler import EmbeddingsHandler

class ContextProvider:
    """Klasa do dostarczania kontekstu rynkowego dla decyzji tradingowych."""
    
    def __init__(self, market_memory: MarketMemory, embeddings_handler: EmbeddingsHandler):
        """
        Inicjalizuje provider kontekstu.
        
        Args:
            market_memory: Instancja MarketMemory
            embeddings_handler: Instancja EmbeddingsHandler
        """
        self.market_memory = market_memory
        self.embeddings_handler = embeddings_handler
        
    def get_market_context(self,
                          query: str,
                          symbol: str,
                          timeframe: str,
                          n_results: int = 5,
                          time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Pobiera kontekst rynkowy na podstawie zapytania.
        
        Args:
            query: Tekst zapytania
            symbol: Symbol instrumentu
            timeframe: Interwał czasowy
            n_results: Liczba wyników do zwrócenia
            time_window: Okno czasowe do przeszukania
            
        Returns:
            Słownik z kontekstem rynkowym
            
        Raises:
            ValueError: Gdy parametry wejściowe są nieprawidłowe
        """
        # Walidacja parametrów wejściowych
        if not query or not isinstance(query, str):
            raise ValueError("Query musi być niepustym stringiem")
            
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol musi być niepustym stringiem")
            
        if not timeframe or not isinstance(timeframe, str):
            raise ValueError("Timeframe musi być niepustym stringiem")
            
        if not isinstance(n_results, int) or n_results <= 0:
            raise ValueError("n_results musi być dodatnią liczbą całkowitą")
            
        if time_window is not None and not isinstance(time_window, timedelta):
            raise ValueError("time_window musi być instancją timedelta lub None")
            
        # Walidacja timeframe
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
        if timeframe not in valid_timeframes:
            raise ValueError(f"Nieprawidłowy timeframe. Dozwolone wartości: {', '.join(valid_timeframes)}")
            
        # Pobierz dane z pamięci rynku
        try:
            market_data = self.market_memory.query_market_data(
                query_text=query,
                symbol=symbol,
                timeframe=timeframe,
                n_results=n_results
            )
        except Exception as e:
            raise RuntimeError(f"Błąd podczas pobierania danych rynkowych: {str(e)}")
            
        if not market_data:
            raise ValueError("Nie znaleziono danych rynkowych dla podanych parametrów")
        
        # Przygotuj dane do analizy semantycznej
        documents = [self._format_market_data(data) for data in market_data]
        
        # Znajdź najbardziej podobne konteksty
        similar_contexts = self.embeddings_handler.get_most_similar(
            query=query,
            documents=documents,
            k=n_results
        )
        
        # Przygotuj pełny kontekst
        context = {
            'query': query,
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'similar_contexts': similar_contexts,
            'summary': self._generate_context_summary(market_data, similar_contexts)
        }
        
        return context
        
    def _format_market_data(self, data: Dict[str, Any]) -> str:
        """
        Formatuje dane rynkowe do postaci tekstowej.
        
        Args:
            data: Słownik z danymi rynkowymi
            
        Returns:
            Sformatowany tekst
            
        Raises:
            ValueError: Gdy dane wejściowe są nieprawidłowe
        """
        if not isinstance(data, dict) or 'data' not in data or 'metadata' not in data:
            raise ValueError("Nieprawidłowy format danych wejściowych")
            
        market_data = data['data']
        metadata = data['metadata']
        
        required_fields = ['timestamp', 'open', 'high', 'low', 'close']
        for field in required_fields:
            if field not in market_data:
                raise ValueError(f"Brak wymaganego pola: {field}")
        
        # Walidacja timestamp
        timestamp = market_data['timestamp']
        if not isinstance(timestamp, datetime):
            raise ValueError("Nieprawidłowy format daty")
            
        if timestamp > datetime.now():
            raise ValueError("Data nie może być z przyszłości")
        
        text = f"Symbol: {metadata['symbol']}, Timeframe: {metadata['timeframe']}\n"
        text += f"Timestamp: {timestamp}\n"
        text += f"Open: {market_data['open']:.4f}, High: {market_data['high']:.4f}, "
        text += f"Low: {market_data['low']:.4f}, Close: {market_data['close']:.4f}\n"
        
        # Dodaj wskaźniki techniczne (tylko niepuste)
        for key, value in market_data.items():
            if key not in ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and value is not None:
                text += f"{key}: {value:.4f}\n"
                
        return text
        
    def _generate_context_summary(self,
                                market_data: List[Dict[str, Any]],
                                similar_contexts: List[Dict[str, Any]]) -> str:
        """
        Generuje podsumowanie kontekstu rynkowego.
        
        Args:
            market_data: Lista danych rynkowych
            similar_contexts: Lista podobnych kontekstów
            
        Returns:
            Podsumowanie kontekstu
            
        Raises:
            ValueError: Gdy dane wejściowe są nieprawidłowe
        """
        if not market_data or not similar_contexts:
            raise ValueError("Brak danych do wygenerowania podsumowania")
            
        # Przeanalizuj trendy
        try:
            closes = [float(data['data']['close']) for data in market_data]
            trend = 'wzrostowy' if closes[-1] > closes[0] else 'spadkowy'
            
            # Oblicz zmianę procentową
            price_change = ((closes[-1] - closes[0]) / closes[0]) * 100
            
            # Znajdź najważniejsze poziomy
            high = max(float(data['data']['high']) for data in market_data)
            low = min(float(data['data']['low']) for data in market_data)
            
            summary = f"Trend {trend} z zmianą {price_change:.2f}%\n"
            summary += f"Najwyższy poziom: {high:.4f}, Najniższy poziom: {low:.4f}\n"
            summary += f"Znaleziono {len(similar_contexts)} podobnych kontekstów "
            summary += f"z najwyższym podobieństwem {similar_contexts[0]['similarity']:.2f}\n"
            
            return summary
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ValueError(f"Błąd podczas generowania podsumowania: {str(e)}") 