"""
Testy jednostkowe dla modułu context_provider.py
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List, Generator, Callable

from src.rag.context_provider import ContextProvider
from src.rag.market_memory import MarketMemory
from src.rag.embeddings_handler import EmbeddingsHandler


@pytest.fixture
def mock_market_memory() -> Generator[Mock, None, None]:
    """Fixture dostarczający zamockowaną instancję MarketMemory."""
    memory = Mock(spec=MarketMemory)
    memory.query_market_data.return_value = [
        {
            'data': {
                'timestamp': datetime.now() - timedelta(hours=1),
                'open': Decimal('1.1000'),
                'high': Decimal('1.1100'),
                'low': Decimal('1.0900'),
                'close': Decimal('1.1050'),
                'volume': Decimal('1000.0'),
                'rsi': 65.0,
                'sma': 1.1025
            },
            'metadata': {
                'symbol': 'EURUSD',
                'timeframe': 'H1'
            }
        }
    ]
    yield memory


@pytest.fixture
def mock_embeddings_handler() -> Generator[Mock, None, None]:
    """Fixture dostarczający zamockowaną instancję EmbeddingsHandler."""
    handler = Mock(spec=EmbeddingsHandler)
    handler.get_most_similar.return_value = [
        {
            'text': 'Przykładowy kontekst',
            'similarity': 0.85
        }
    ]
    yield handler


@pytest.fixture
def context_provider(mock_market_memory: Mock, mock_embeddings_handler: Mock) -> Generator[ContextProvider, None, None]:
    """Fixture dostarczający instancję ContextProvider z zamockowanymi zależnościami."""
    provider = ContextProvider(
        market_memory=mock_market_memory,
        embeddings_handler=mock_embeddings_handler
    )
    yield provider


class TestContextProvider:
    """Testy dla klasy ContextProvider."""

    def test_initialization(self, context_provider: ContextProvider, 
                          mock_market_memory: Mock, 
                          mock_embeddings_handler: Mock) -> None:
        """Test poprawnej inicjalizacji providera."""
        assert isinstance(context_provider, ContextProvider)
        assert context_provider.market_memory == mock_market_memory
        assert context_provider.embeddings_handler == mock_embeddings_handler

    def test_get_market_context(self, context_provider: ContextProvider) -> None:
        """Test pobierania kontekstu rynkowego."""
        # Przygotowanie parametrów
        query = "Analiza trendu EURUSD"
        symbol = "EURUSD"
        timeframe = "H1"
        n_results = 5

        # Wywołanie metody
        context = context_provider.get_market_context(
            query=query,
            symbol=symbol,
            timeframe=timeframe,
            n_results=n_results
        )

        # Sprawdzenie wyników
        assert isinstance(context, dict)
        assert context['query'] == query
        assert context['symbol'] == symbol
        assert context['timeframe'] == timeframe
        assert 'timestamp' in context
        assert 'market_data' in context
        assert 'similar_contexts' in context
        assert 'summary' in context

        # Sprawdzenie czy metody zależności zostały wywołane
        context_provider.market_memory.query_market_data.assert_called_once_with(
            query_text=query,
            symbol=symbol,
            timeframe=timeframe,
            n_results=n_results
        )

    def test_format_market_data(self, context_provider):
        """Test formatowania danych rynkowych do postaci tekstowej."""
        # Przygotowanie danych testowych
        data = {
            'data': {
                'timestamp': datetime.now(),
                'open': Decimal('1.1000'),
                'high': Decimal('1.1100'),
                'low': Decimal('1.0900'),
                'close': Decimal('1.1050'),
                'volume': Decimal('1000.0'),
                'rsi': 65.0,
                'sma': 1.1025
            },
            'metadata': {
                'symbol': 'EURUSD',
                'timeframe': 'H1'
            }
        }

        # Wywołanie metody
        formatted_text = context_provider._format_market_data(data)

        # Sprawdzenie wyników
        assert isinstance(formatted_text, str)
        assert 'Symbol: EURUSD' in formatted_text
        assert 'Timeframe: H1' in formatted_text
        assert 'Open: 1.1000' in formatted_text
        assert 'High: 1.1100' in formatted_text
        assert 'Low: 1.0900' in formatted_text
        assert 'Close: 1.1050' in formatted_text
        assert 'rsi: 65.0000' in formatted_text
        assert 'sma: 1.1025' in formatted_text

    def test_error_handling(self, context_provider):
        """Test obsługi błędów."""
        # Test dla nieprawidłowego symbolu
        with pytest.raises(ValueError, match="Symbol musi być niepustym stringiem"):
            context_provider.get_market_context(
                query="test",
                symbol="",  # pusty symbol
                timeframe="H1",
                n_results=5
            )

        # Test dla nieprawidłowego timeframe
        with pytest.raises(ValueError, match="Nieprawidłowy timeframe"):
            context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe="INVALID",  # nieprawidłowy timeframe
                n_results=5
            )

        # Test dla nieprawidłowej liczby wyników
        with pytest.raises(ValueError, match="n_results musi być dodatnią liczbą całkowitą"):
            context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe="H1",
                n_results=0  # nieprawidłowa liczba wyników
            )

        # Test dla pustego query
        with pytest.raises(ValueError, match="Query musi być niepustym stringiem"):
            context_provider.get_market_context(
                query="",
                symbol="EURUSD",
                timeframe="H1",
                n_results=5
            )

        # Test dla nieprawidłowego time_window
        with pytest.raises(ValueError, match="time_window musi być instancją timedelta lub None"):
            context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe="H1",
                n_results=5,
                time_window="24h"  # nieprawidłowy format
            )

        # Symulacja błędu w MarketMemory
        context_provider.market_memory.query_market_data.side_effect = Exception("Test error")
        with pytest.raises(RuntimeError, match="Błąd podczas pobierania danych rynkowych"):
            context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe="H1",
                n_results=5
            )

    def test_context_limits(self, context_provider: ContextProvider) -> None:
        """Test limitów kontekstu."""
        # Test maksymalnej liczby wyników
        max_results = 100
        context = context_provider.get_market_context(
            query="test",
            symbol="EURUSD",
            timeframe="H1",
            n_results=max_results
        )
        assert len(context['market_data']) <= max_results
        assert len(context['similar_contexts']) <= max_results

        # Test limitu czasu
        time_window = timedelta(days=7)
        context = context_provider.get_market_context(
            query="test",
            symbol="EURUSD",
            timeframe="H1",
            n_results=5,
            time_window=time_window
        )
        
        # Sprawdź czy wszystkie dane są w zakresie czasowym
        for data in context['market_data']:
            data_time = data['data']['timestamp']
            assert datetime.now() - data_time <= time_window

    def test_generate_context_summary(self, context_provider: ContextProvider) -> None:
        """Test generowania podsumowania kontekstu."""
        # Przygotowanie danych testowych
        market_data = [
            {
                'data': {
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'open': Decimal('1.1000'),
                    'high': Decimal('1.1200'),
                    'low': Decimal('1.0900'),
                    'close': Decimal('1.1100')
                },
                'metadata': {'symbol': 'EURUSD', 'timeframe': 'H1'}
            },
            {
                'data': {
                    'timestamp': datetime.now() - timedelta(hours=1),
                    'open': Decimal('1.1100'),
                    'high': Decimal('1.1300'),
                    'low': Decimal('1.1000'),
                    'close': Decimal('1.1200')
                },
                'metadata': {'symbol': 'EURUSD', 'timeframe': 'H1'}
            }
        ]

        similar_contexts = [
            {'text': 'Kontekst 1', 'similarity': 0.9},
            {'text': 'Kontekst 2', 'similarity': 0.8}
        ]

        # Wywołanie metody
        summary = context_provider._generate_context_summary(market_data, similar_contexts)

        # Sprawdzenie wyników
        assert isinstance(summary, str)
        assert 'Trend wzrostowy' in summary  # cena zamknięcia wzrosła
        assert 'zmianą' in summary
        assert 'Najwyższy poziom: 1.1300' in summary
        assert 'Najniższy poziom: 1.0900' in summary
        assert 'Znaleziono 2 podobnych kontekstów' in summary
        assert '0.90' in summary  # najwyższe podobieństwo

        # Test dla pustych danych
        with pytest.raises(ValueError, match="Brak danych do wygenerowania podsumowania"):
            context_provider._generate_context_summary([], [])

        # Test dla nieprawidłowych danych
        with pytest.raises(ValueError, match="Błąd podczas generowania podsumowania"):
            context_provider._generate_context_summary(
                [{'data': {'invalid': 'data'}}],
                similar_contexts
            )

    def test_input_validation(self, context_provider: ContextProvider) -> None:
        """Test walidacji danych wejściowych."""
        # Test dla nieprawidłowego formatu danych
        with pytest.raises(ValueError, match="Nieprawidłowy format danych wejściowych"):
            context_provider._format_market_data({'invalid': 'data'})

        # Test dla brakujących pól
        with pytest.raises(ValueError, match="Brak wymaganego pola"):
            context_provider._format_market_data({
                'data': {'timestamp': datetime.now()},
                'metadata': {'symbol': 'EURUSD', 'timeframe': 'H1'}
            })

        # Test dla nieprawidłowych typów danych
        with pytest.raises(ValueError, match="Query musi być niepustym stringiem"):
            context_provider.get_market_context(
                query=123,  # nieprawidłowy typ
                symbol="EURUSD",
                timeframe="H1",
                n_results=5
            )

        # Test dla nieprawidłowego formatu timeframe
        with pytest.raises(ValueError, match="Nieprawidłowy timeframe"):
            context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe="invalid",
                n_results=5
            )

    def test_format_market_data_with_many_indicators(self, context_provider: ContextProvider) -> None:
        """Test formatowania danych z wieloma wskaźnikami technicznymi."""
        data = {
            'data': {
                'timestamp': datetime.now(),
                'open': Decimal('1.1000'),
                'high': Decimal('1.1100'),
                'low': Decimal('1.0900'),
                'close': Decimal('1.1050'),
                'volume': Decimal('1000.0'),
                'rsi': 65.0,
                'sma': 1.1025,
                'ema': 1.1030,
                'macd': 0.0025,
                'macd_signal': 0.0020,
                'macd_hist': 0.0005,
                'bb_upper': 1.1200,
                'bb_middle': 1.1100,
                'bb_lower': 1.1000,
                'stoch_k': 80.0,
                'stoch_d': 75.0,
                'adx': 25.0,
                'cci': 100.0,
                'mfi': 60.0
            },
            'metadata': {
                'symbol': 'EURUSD',
                'timeframe': 'H1'
            }
        }

        formatted_text = context_provider._format_market_data(data)
        
        # Sprawdzenie czy wszystkie wskaźniki są w tekście
        assert 'rsi: 65.0000' in formatted_text
        assert 'macd: 0.0025' in formatted_text
        assert 'bb_upper: 1.1200' in formatted_text
        assert 'stoch_k: 80.0000' in formatted_text
        assert 'adx: 25.0000' in formatted_text
        assert 'mfi: 60.0000' in formatted_text

    def test_timeframe_formats(self, context_provider: ContextProvider) -> None:
        """Test różnych formatów timeframe."""
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
        
        for timeframe in valid_timeframes:
            context = context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe=timeframe,
                n_results=1
            )
            assert context['timeframe'] == timeframe

        invalid_timeframes = ['1M', 'H2', 'D2', 'W2', 'M', 'H', 'invalid']
        for timeframe in invalid_timeframes:
            with pytest.raises(ValueError, match="Nieprawidłowy timeframe"):
                context_provider.get_market_context(
                    query="test",
                    symbol="EURUSD",
                    timeframe=timeframe,
                    n_results=1
                )

    def test_empty_indicators(self, context_provider: ContextProvider) -> None:
        """Test obsługi pustych wskaźników technicznych."""
        data = {
            'data': {
                'timestamp': datetime.now(),
                'open': Decimal('1.1000'),
                'high': Decimal('1.1100'),
                'low': Decimal('1.0900'),
                'close': Decimal('1.1050'),
                'volume': Decimal('1000.0'),
                'rsi': None,
                'sma': None
            },
            'metadata': {
                'symbol': 'EURUSD',
                'timeframe': 'H1'
            }
        }

        formatted_text = context_provider._format_market_data(data)
        
        # Sprawdzenie czy wskaźniki z wartością None nie są w tekście
        assert 'rsi: None' not in formatted_text
        assert 'sma: None' not in formatted_text
        
        # Sprawdzenie czy podstawowe pola są obecne
        assert 'Open: 1.1000' in formatted_text
        assert 'High: 1.1100' in formatted_text
        assert 'Low: 1.0900' in formatted_text
        assert 'Close: 1.1050' in formatted_text

    def test_timestamp_validation(self, context_provider: ContextProvider) -> None:
        """Test walidacji formatu daty."""
        # Test dla nieprawidłowego formatu timestamp
        data = {
            'data': {
                'timestamp': "invalid_date",  # nieprawidłowy format
                'open': Decimal('1.1000'),
                'high': Decimal('1.1100'),
                'low': Decimal('1.0900'),
                'close': Decimal('1.1050'),
                'volume': Decimal('1000.0')
            },
            'metadata': {
                'symbol': 'EURUSD',
                'timeframe': 'H1'
            }
        }

        with pytest.raises(ValueError, match="Nieprawidłowy format daty"):
            context_provider._format_market_data(data)

        # Test dla przyszłej daty
        future_date = datetime.now() + timedelta(days=1)
        data['data']['timestamp'] = future_date
        
        with pytest.raises(ValueError, match="Data nie może być z przyszłości"):
            context_provider._format_market_data(data)

    def test_memory_usage(self, context_provider: ContextProvider) -> None:
        """Test zużycia pamięci."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Wykonaj operacje
        for _ in range(1000):
            context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe="H1",
                n_results=5
            )

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB

        # Sprawdź czy wzrost zużycia pamięci jest poniżej 10MB
        assert memory_increase < 10, f"Wzrost zużycia pamięci: {memory_increase:.2f}MB"


@pytest.mark.performance
class TestContextProviderPerformance:
    """Testy wydajnościowe dla ContextProvider."""

    def test_get_market_context_performance(self, context_provider: ContextProvider, benchmark: Callable) -> None:
        """Test wydajności pobierania kontekstu."""
        def run_get_context():
            context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe="H1",
                n_results=5
            )
        
        benchmark(run_get_context)
        assert benchmark.stats['mean'] < 0.1

    def test_format_market_data_performance(self, context_provider, benchmark: Callable) -> None:
        """Test wydajności formatowania danych."""
        data = {
            'data': {
                'timestamp': datetime.now(),
                'open': Decimal('1.1000'),
                'high': Decimal('1.1100'),
                'low': Decimal('1.0900'),
                'close': Decimal('1.1050'),
                'volume': Decimal('1000.0'),
                'rsi': 65.0,
                'sma': 1.1025
            },
            'metadata': {
                'symbol': 'EURUSD',
                'timeframe': 'H1'
            }
        }

        def run_format_data():
            context_provider._format_market_data(data)

        benchmark(run_format_data)
        assert benchmark.stats['mean'] < 0.01

    def test_generate_summary_performance(self, context_provider, benchmark: Callable) -> None:
        """Test wydajności generowania podsumowania."""
        market_data = [
            {
                'data': {
                    'timestamp': datetime.now(),
                    'open': Decimal('1.1000'),
                    'high': Decimal('1.1100'),
                    'low': Decimal('1.0900'),
                    'close': Decimal('1.1050')
                },
                'metadata': {'symbol': 'EURUSD', 'timeframe': 'H1'}
            }
        ]
        similar_contexts = [{'text': 'test', 'similarity': 0.9}]

        def run_generate_summary():
            context_provider._generate_context_summary(market_data, similar_contexts)

        benchmark(run_generate_summary)
        assert benchmark.stats['mean'] < 0.01

    def test_memory_usage(self, context_provider: ContextProvider) -> None:
        """Test zużycia pamięci."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Wykonaj operacje
        for _ in range(1000):
            context_provider.get_market_context(
                query="test",
                symbol="EURUSD",
                timeframe="H1",
                n_results=5
            )

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB

        # Sprawdź czy wzrost zużycia pamięci jest poniżej 10MB
        assert memory_increase < 10, f"Wzrost zużycia pamięci: {memory_increase:.2f}MB" 