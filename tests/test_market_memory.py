"""
Testy jednostkowe dla klasy MarketMemory.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path
import shutil
import tempfile
import concurrent.futures
import threading
import psutil
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json

from src.rag.market_memory import MarketMemory


@pytest.fixture
def market_memory():
    """Fixture tworzący tymczasową instancję MarketMemory do testów."""
    temp_dir = tempfile.mkdtemp()
    memory = None
    try:
        memory = MarketMemory(persist_directory=temp_dir)
        yield memory
    finally:
        if memory:
            try:
                if hasattr(memory, 'collection') and memory.collection:
                    try:
                        memory.close()
                    except Exception as e:
                        print(f"Błąd podczas zamykania połączeń: {e}")
            except Exception as e:
                print(f"Błąd podczas sprawdzania kolekcji: {e}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Błąd podczas usuwania katalogu tymczasowego: {e}")


@pytest.fixture
def sample_data():
    """Fixture tworzący przykładowe dane rynkowe."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'rsi': [50.0, 52.0, 54.0, 56.0, 58.0]
    }, index=dates)
    return data


class TestPodstawoweFunkcje:
    """Testy podstawowych funkcji MarketMemory."""
    
    def test_inicjalizacja_market_memory(self, market_memory):
        """Test sprawdzający poprawną inicjalizację klasy."""
        assert market_memory.collection is not None
        assert market_memory.collection.name == "market_data"

    def test_dodawanie_danych_rynkowych(self, market_memory, sample_data):
        """Test sprawdzający dodawanie danych rynkowych."""
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1",
            metadata={"source": "test"}
        )
        
        results = market_memory.query_market_data(
            query_text="EURUSD D1",
            symbol="EURUSD",
            timeframe="D1"
        )
        
        assert len(results) > 0
        assert results[0]['metadata']['symbol'] == "EURUSD"
        assert results[0]['metadata']['timeframe'] == "D1"
        assert results[0]['metadata']['source'] == "test"


class TestWalidacja:
    """Testy walidacji danych wejściowych i wyszukiwania."""
    
    def test_walidacja_danych_wejsciowych(self, market_memory):
        """Test sprawdzający walidację danych wejściowych."""
        with pytest.raises(ValueError, match="Dane wejściowe nie mogą być puste"):
            market_memory.add_market_data(
                data=pd.DataFrame(),
                symbol="EURUSD",
                timeframe="D1"
            )
        
        with pytest.raises(ValueError, match="Symbol musi być ciągiem znaków"):
            market_memory.add_market_data(
                data=pd.DataFrame({'open': [1.0]}),
                symbol="E",
                timeframe="D1"
            )
        
        with pytest.raises(ValueError, match="Brakujące kolumny"):
            market_memory.add_market_data(
                data=pd.DataFrame({'open': [1.0]}),
                symbol="EURUSD",
                timeframe="D1"
            )

    def test_walidacja_wyszukiwania(self, market_memory):
        """Test sprawdzający walidację parametrów wyszukiwania."""
        with pytest.raises(ValueError, match="Tekst zapytania nie może być pusty"):
            market_memory.query_market_data(query_text="")
        
        with pytest.raises(ValueError, match="Liczba wyników musi być większa od 0"):
            market_memory.query_market_data(query_text="test", n_results=0)
        
        with pytest.raises(ValueError, match="Symbol musi być ciągiem znaków"):
            market_memory.query_market_data(query_text="test", symbol="E")
        
        # Test wyszukiwania w pustej kolekcji
        results = market_memory.query_market_data(query_text="test")
        assert len(results) == 0

    def test_walidacja_usuwania(self, market_memory):
        """Test sprawdzający walidację parametrów usuwania."""
        with pytest.raises(ValueError, match="Symbol musi być ciągiem znaków o długości co najmniej 2"):
            market_memory.delete_market_data(symbol="E")

        with pytest.raises(ValueError, match="Timeframe musi być niepustym ciągiem znaków"):
            market_memory.delete_market_data(timeframe="")

        with pytest.raises(ValueError, match="before_date musi być obiektem datetime"):
            market_memory.delete_market_data(before_date="2024-01-01")


class TestOperacjeDanych:
    """Testy operacji na danych."""

    def test_wyszukiwanie_danych(self, market_memory, sample_data):
        """Test sprawdzający wyszukiwanie danych."""
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test podstawowego wyszukiwania
        results = market_memory.query_market_data(
            query_text="EURUSD D1",
            symbol="EURUSD",
            timeframe="D1",
            n_results=3
        )
        
        assert len(results) <= 3
        assert all(r['metadata']['symbol'] == "EURUSD" for r in results)
        assert all(r['metadata']['timeframe'] == "D1" for r in results)

    def test_usuwanie_danych(self, market_memory, sample_data):
        """Test sprawdzający usuwanie danych."""
        # Test usuwania po symbolu
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Sprawdź czy dane zostały dodane
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD",
            timeframe="D1"
        )
        assert len(results) > 0
        
        # Test usuwania po symbolu
        assert market_memory.delete_market_data(symbol="EURUSD") == True
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0
        
        # Test usuwania po dacie
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Sprawdź czy dane zostały dodane
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD",
            timeframe="D1"
        )
        assert len(results) > 0
        
        cutoff_date = datetime(2024, 1, 3)
        assert market_memory.delete_market_data(before_date=cutoff_date) == True
        
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Sprawdź czy pozostały tylko dane po dacie granicznej
        assert len(results) > 0
        for result in results:
            result_date = datetime.fromisoformat(result['data']['timestamp'])
            assert result_date >= cutoff_date


class TestWydajnosc:
    """Testy wydajności i skalowalności."""

    def test_duze_dane(self, market_memory):
        """Test obsługi dużych zbiorów danych."""
        # Generuj duży zbiór danych
        dates = pd.date_range(start='2024-01-01', periods=10000, freq='5min')
        data = pd.DataFrame({
            'open': np.random.normal(100, 1, size=10000),
            'high': np.random.normal(101, 1, size=10000),
            'low': np.random.normal(99, 1, size=10000),
            'close': np.random.normal(100, 1, size=10000),
            'volume': np.random.randint(1000, 2000, size=10000),
            'rsi': np.random.uniform(0, 100, size=10000)
        }, index=dates)

        # Dodaj dane w mniejszych wsadach
        market_memory.add_market_data(
            data=data,
            symbol="EURUSD",
            timeframe="M5",
            batch_size=1000
        )

        # Sprawdź czy można wyszukiwać w dużym zbiorze danych
        results = market_memory.query_market_data(
            query_text="EURUSD M5 high volume",
            symbol="EURUSD",
            timeframe="M5",
            n_results=10
        )

        assert len(results) == 10
        assert all(r['metadata']['symbol'] == "EURUSD" for r in results)
        assert all(r['metadata']['timeframe'] == "M5" for r in results)

    def test_wiele_symboli_i_timeframes(self, market_memory, sample_data):
        """Test obsługi wielu symboli i timeframes."""
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        timeframes = ["M1", "M5", "H1"]
        
        # Dodaj dane dla różnych kombinacji
        for symbol in symbols:
            for timeframe in timeframes:
                market_memory.add_market_data(
                    data=sample_data,
                    symbol=symbol,
                    timeframe=timeframe
                )
        
        # Sprawdź filtrowanie po symbolu
        for symbol in symbols:
            results = market_memory.query_market_data(
                query_text=f"{symbol} data",
                symbol=symbol
            )
            assert all(r['metadata']['symbol'] == symbol for r in results)
        
        # Sprawdź filtrowanie po timeframe
        for timeframe in timeframes:
            results = market_memory.query_market_data(
                query_text=f"{timeframe} data",
                timeframe=timeframe
            )
            assert all(r['metadata']['timeframe'] == timeframe for r in results)

    def test_operacje_wspolbiezne(self, market_memory, sample_data):
        """Test współbieżnych operacji na pamięci."""
        def add_data(symbol):
            try:
                market_memory.add_market_data(
                    data=sample_data,
                    symbol=symbol,
                    timeframe="D1"
                )
                return True
            except Exception:
                return False
        
        def query_data(symbol):
            try:
                results = market_memory.query_market_data(
                    query_text=f"{symbol} data",
                    symbol=symbol
                )
                return len(results) > 0
            except Exception:
                return False
        
        symbols = [f"SYMBOL{i}" for i in range(10)]
        
        # Testuj współbieżne dodawanie
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(add_data, symbol) for symbol in symbols]
            results = [future.result() for future in as_completed(futures)]
            assert all(results)
        
        # Testuj współbieżne wyszukiwanie
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(query_data, symbol) for symbol in symbols]
            results = [future.result() for future in as_completed(futures)]
            assert all(results)

    def test_zuzycie_pamieci(self, market_memory):
        """Test monitorowania zużycia pamięci podczas operacji na dużych zbiorach danych."""
        # Wymuś garbage collection przed testem
        gc.collect()
        time.sleep(1)
        
        # Pobierz początkowe zużycie pamięci
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generuj duży zbiór danych
        dates = pd.date_range(start='2024-01-01', periods=5000, freq='1min')
        data = pd.DataFrame({
            'open': np.random.normal(100, 1, size=5000),
            'high': np.random.normal(101, 1, size=5000),
            'low': np.random.normal(99, 1, size=5000),
            'close': np.random.normal(100, 1, size=5000),
            'volume': np.random.randint(1000, 2000, size=5000),
            'rsi': np.random.uniform(0, 100, size=5000)
        }, index=dates)

        # Dodaj dane w mniejszych wsadach
        market_memory.add_market_data(
            data=data,
            symbol='EURUSD',
            timeframe='M1',
            batch_size=500
        )

        # Wykonaj kilka operacji wyszukiwania
        for _ in range(5):
            results = market_memory.query_market_data(
                query_text="EURUSD high volume",
                symbol="EURUSD",
                timeframe="M1",
                n_results=50
            )
            assert len(results) > 0

        # Sprawdź końcowe zużycie pamięci
        gc.collect()
        time.sleep(1)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Sprawdź czy wzrost pamięci jest rozsądny (mniej niż 1GB)
        assert memory_increase < 1024, f"Wzrost zużycia pamięci ({memory_increase:.2f} MB) przekroczył 1GB"
        
        # Sprawdź czy pamięć jest zwalniana po wyczyszczeniu danych
        market_memory.delete_market_data(symbol="EURUSD")
        
        # Daj więcej czasu na zwolnienie pamięci i wymuś garbage collection
        time.sleep(2)
        gc.collect()
        time.sleep(1)
        
        cleanup_memory = process.memory_info().rss / 1024 / 1024
        memory_diff = cleanup_memory - final_memory
        assert abs(memory_diff) < 50, f"Różnica w zużyciu pamięci ({memory_diff:.2f} MB) jest zbyt duża"


class TestObslugaBledow:
    """Testy obsługi błędów w MarketMemory."""

    def test_obsluga_bledow_inicjalizacji(self, tmp_path):
        """Test sprawdzający obsługę błędów podczas inicjalizacji."""
        # Test z nieprawidłową ścieżką
        invalid_path = tmp_path / "nieistniejacy" / "katalog"
        with pytest.raises(ValueError):
            MarketMemory(persist_directory=str(invalid_path))

    def test_obsluga_bledow_dodawania(self, market_memory):
        """Test sprawdzający obsługę błędów podczas dodawania danych."""
        # Test z pustymi danymi
        with pytest.raises(ValueError, match="Dane wejściowe nie mogą być puste"):
            market_memory.add_market_data(
                data=pd.DataFrame(),
                symbol="EURUSD",
                timeframe="D1"
            )
            
        # Test z nieprawidłowymi kolumnami
        with pytest.raises(ValueError, match="Brakujące kolumny"):
            market_memory.add_market_data(
                data=pd.DataFrame({'nieprawidlowa': [1.0]}),
                symbol="EURUSD",
                timeframe="D1"
            )
            
        # Test z nieprawidłowym symbolem
        with pytest.raises(ValueError, match="Symbol musi być ciągiem znaków"):
            market_memory.add_market_data(
                data=pd.DataFrame({'open': [1.0]}),
                symbol="E",
                timeframe="D1"
            )

    def test_obsluga_bledow_wyszukiwania(self, market_memory, sample_data):
        """Test sprawdzający obsługę błędów podczas wyszukiwania."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test z pustym zapytaniem
        with pytest.raises(ValueError, match="Tekst zapytania nie może być pusty"):
            market_memory.query_market_data(query_text="")
            
        # Test z nieprawidłową liczbą wyników
        with pytest.raises(ValueError, match="Liczba wyników musi być większa od 0"):
            market_memory.query_market_data(query_text="test", n_results=0)
            
        # Test z nieprawidłowym symbolem
        with pytest.raises(ValueError, match="Symbol musi być ciągiem znaków"):
            market_memory.query_market_data(query_text="test", symbol="E")
            
        # Test z nieprawidłowym timeframe
        with pytest.raises(ValueError, match="Timeframe musi być niepustym ciągiem znaków"):
            market_memory.query_market_data(query_text="test", timeframe="")

    def test_obsluga_bledow_usuwania(self, market_memory, sample_data):
        """Test sprawdzający obsługę błędów podczas usuwania danych."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test z nieprawidłowym symbolem
        with pytest.raises(ValueError, match="Symbol musi być ciągiem znaków"):
            market_memory.delete_market_data(symbol="E")
            
        # Test z nieprawidłowym timeframe
        with pytest.raises(ValueError, match="Timeframe musi być niepustym ciągiem znaków"):
            market_memory.delete_market_data(timeframe="")
            
        # Test z nieprawidłową datą
        with pytest.raises(ValueError, match="before_date musi być obiektem datetime"):
            market_memory.delete_market_data(before_date="2024-01-01")
            
        # Test usuwania nieistniejących danych
        assert market_memory.delete_market_data(symbol="NIEISTNIEJACY") == True

    def test_zamykanie_polaczen(self, market_memory):
        """Test sprawdzający zamykanie połączeń."""
        # Sprawdź czy obiekt ma wymagane atrybuty
        assert hasattr(market_memory, 'collection')
        assert hasattr(market_memory, 'client')
        
        # Wywołaj __del__ ręcznie
        market_memory.__del__()
        
        # Sprawdź czy połączenia zostały zamknięte
        assert market_memory.collection is None
        assert market_memory.client is None

    def test_bledy_dostepu_do_katalogu(self, tmp_path, monkeypatch):
        """Test sprawdzający błędy dostępu do katalogu."""
        # Mockuj Path.write_text aby zasymulować brak uprawnień
        def mock_write_text(*args, **kwargs):
            raise PermissionError("Brak uprawnień do zapisu")
        monkeypatch.setattr(Path, "write_text", mock_write_text)
        
        with pytest.raises(ValueError, match="Brak uprawnień do zapisu"):
            MarketMemory(persist_directory=str(tmp_path))

    def test_bledy_dodawania_do_kolekcji(self, market_memory, sample_data):
        """Test sprawdzający błędy podczas dodawania do kolekcji."""
        # Mockuj collection.add aby zasymulować błąd
        def mock_add(*args, **kwargs):
            raise Exception("Błąd podczas dodawania do kolekcji")
        market_memory.collection.add = mock_add
        
        with pytest.raises(Exception, match="Błąd podczas dodawania do kolekcji"):
            market_memory.add_market_data(
                data=sample_data,
                symbol="EURUSD",
                timeframe="D1"
            )

    def test_bledy_przetwarzania_wynikow(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający błędy podczas przetwarzania wyników wyszukiwania."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Mockuj json.loads aby zasymulować błąd parsowania
        def mock_loads(*args, **kwargs):
            raise json.JSONDecodeError("Błąd parsowania", "", 0)
        monkeypatch.setattr(json, "loads", mock_loads)
        
        # Sprawdź czy funkcja obsługuje błąd i zwraca pustą listę
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

    def test_bledy_usuwania_danych(self, market_memory, sample_data):
        """Test sprawdzający błędy podczas usuwania danych."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Mockuj collection.delete aby zasymulować błąd
        def mock_delete(*args, **kwargs):
            raise Exception("Błąd podczas usuwania danych")
        market_memory.collection.delete = mock_delete
        
        # Sprawdź czy funkcja zwraca False w przypadku błędu
        assert market_memory.delete_market_data(symbol="EURUSD") == False

    def test_bledy_zamykania_polaczen(self, market_memory):
        """Test sprawdzający błędy podczas zamykania połączeń."""
        # Mockuj delete_collection aby zasymulować błąd
        def mock_delete_collection(*args, **kwargs):
            raise Exception("Błąd podczas usuwania kolekcji")
        market_memory.client.delete_collection = mock_delete_collection
        
        # Wywołaj __del__ ręcznie - nie powinno rzucić wyjątku
        market_memory.__del__()
        
        # Sprawdź czy połączenia zostały zamknięte mimo błędu
        assert market_memory.collection is None
        assert market_memory.client is None

    def test_bledy_wyszukiwania(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający błędy podczas wyszukiwania."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Mockuj collection.query aby zasymulować błąd
        def mock_query(*args, **kwargs):
            raise Exception("Błąd podczas wyszukiwania")
        market_memory.collection.query = mock_query
        
        # Sprawdź czy funkcja obsługuje błąd i zwraca pustą listę
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

    def test_bledy_przetwarzania_daty(self, market_memory):
        """Test sprawdzający błędy podczas przetwarzania daty."""
        # Przygotuj dane z nieprawidłową datą
        bad_data = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000],
            'rsi': [50.0]
        }, index=['nieprawidłowa_data'])  # Nieprawidłowy format daty

        # Próba dodania danych z nieprawidłową datą powinna być obsłużona
        with pytest.raises(ValueError, match="Nieprawidłowy format daty w indeksie"):
            market_memory.add_market_data(
                data=bad_data,
                symbol="EURUSD",
                timeframe="D1"
            )

    def test_bledy_usuwania_szczegolowe(self, market_memory):
        """Test sprawdzający szczegółowe błędy podczas usuwania danych."""
        # Test usuwania gdy kolekcja jest None
        market_memory.collection = None
        assert market_memory.delete_market_data(symbol="EURUSD") == False

        # Test usuwania bez kryteriów
        market_memory = MarketMemory()
        assert market_memory.delete_market_data() == False

    def test_bledy_zamykania_polaczen_szczegolowe(self, market_memory):
        """Test sprawdzający szczegółowe błędy podczas zamykania połączeń."""
        # Test zamykania gdy kolekcja jest None
        market_memory.collection = None
        market_memory.close()
        assert market_memory.collection is None

    def test_bledy_zamykania_polaczen_pelne(self, market_memory):
        """Test sprawdzający pełne scenariusze błędów podczas zamykania połączeń."""
        # Test zamykania gdy kolekcja zgłasza wyjątek
        if hasattr(market_memory, 'collection') and market_memory.collection:
            def mock_delete(*args, **kwargs):
                raise Exception("Błąd podczas usuwania kolekcji")
            
            original_delete = market_memory.collection.delete
            try:
                market_memory.collection.delete = mock_delete
                market_memory.close()
                assert market_memory.collection is None
            finally:
                if hasattr(market_memory, 'collection') and market_memory.collection:
                    market_memory.collection.delete = original_delete

    def test_sciezka_walidacji_katalogu(self, tmp_path, monkeypatch):
        """Test sprawdzający różne scenariusze walidacji ścieżki katalogu."""
        # Test z nieistniejącym katalogiem nadrzędnym
        invalid_dir = tmp_path / "nieistniejacy" / "katalog"
        with pytest.raises(ValueError, match="Katalog nadrzędny nie istnieje"):
            MarketMemory(persist_directory=str(invalid_dir))
            
        # Test z błędem podczas tworzenia katalogu
        test_dir = tmp_path / "test_dir"
        
        def mock_mkdir(*args, **kwargs):
            raise PermissionError("Brak uprawnień do zapisu")
            
        monkeypatch.setattr(Path, "mkdir", mock_mkdir)
        
        with pytest.raises(ValueError, match="Brak uprawnień do zapisu"):
            MarketMemory(persist_directory=str(test_dir))

    def test_obsluga_bledow_dodawania_szczegolowa(self, market_memory, sample_data):
        """Test sprawdzający szczegółową obsługę błędów podczas dodawania danych."""
        # Test z niezainicjalizowaną kolekcją
        market_memory.collection = None
        with pytest.raises(ValueError, match="Kolekcja nie jest zainicjalizowana"):
            market_memory.add_market_data(
                data=sample_data,
                symbol="EURUSD",
                timeframe="D1"
            )
            
        # Test z błędem podczas dodawania do kolekcji
        market_memory = MarketMemory()
        def mock_add(*args, **kwargs):
            raise Exception("Błąd podczas dodawania do kolekcji")
        
        original_add = market_memory.collection.add
        market_memory.collection.add = mock_add
        
        with pytest.raises(ValueError, match="Nie udało się dodać danych do kolekcji"):
            market_memory.add_market_data(
                data=sample_data,
                symbol="EURUSD",
                timeframe="D1"
            )
            
        market_memory.collection.add = original_add

    def test_obsluga_bledow_wyszukiwania_szczegolowa(self, market_memory, sample_data):
        """Test sprawdzający szczegółową obsługę błędów podczas wyszukiwania."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test z niezainicjalizowaną kolekcją
        market_memory.collection = None
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0
        
        # Test z błędem podczas parsowania JSON
        market_memory = MarketMemory()
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        def mock_loads(*args, **kwargs):
            raise json.JSONDecodeError("Błąd parsowania", "", 0)
        
        original_loads = json.loads
        json.loads = mock_loads
        
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0
        
        json.loads = original_loads

    def test_obsluga_bledow_usuwania_szczegolowa(self, market_memory, sample_data):
        """Test sprawdzający szczegółową obsługę błędów podczas usuwania danych."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test z niezainicjalizowaną kolekcją
        market_memory.collection = None
        assert market_memory.delete_market_data(symbol="EURUSD") == False
        
        # Test z błędem podczas usuwania
        market_memory = MarketMemory()
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        def mock_delete(*args, **kwargs):
            raise Exception("Błąd podczas usuwania danych")
        
        original_delete = market_memory.collection.delete
        market_memory.collection.delete = mock_delete
        
        assert market_memory.delete_market_data(symbol="EURUSD") == False
        
        market_memory.collection.delete = original_delete

    def test_obsluga_bledow_zamykania_szczegolowa(self, market_memory, monkeypatch):
        """Test sprawdzający szczegółową obsługę błędów podczas zamykania połączeń."""
        # Test z błędem podczas usuwania kolekcji
        def mock_delete(*args, **kwargs):
            raise Exception("Błąd podczas usuwania kolekcji")
        
        if hasattr(market_memory, 'collection') and market_memory.collection:
            monkeypatch.setattr(market_memory.collection, "delete", mock_delete)
            
            market_memory.close()
            assert market_memory.collection is None
            assert market_memory.client is None
        
        # Test z błędem podczas usuwania kolekcji przez klienta
        market_memory = MarketMemory()
        def mock_delete_collection(*args, **kwargs):
            raise Exception("Błąd podczas usuwania kolekcji")
        
        if hasattr(market_memory, 'client') and market_memory.client:
            monkeypatch.setattr(market_memory.client, "delete_collection", mock_delete_collection)
            
            market_memory.close()
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_dodawania_pelna(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający pełną obsługę błędów podczas dodawania danych."""
        # Test z błędem podczas konwersji daty
        bad_data = sample_data.copy()
        bad_data.index = ['nieprawidłowa_data'] * len(bad_data)

        with pytest.raises(ValueError, match="Nieprawidłowy format daty w indeksie"):
            market_memory.add_market_data(
                data=bad_data,
                symbol="EURUSD",
                timeframe="D1"
            )

        # Test z błędem podczas dodawania do kolekcji
        def mock_add(*args, **kwargs):
            raise Exception("Błąd podczas dodawania do kolekcji")

        monkeypatch.setattr(market_memory.collection, "add", mock_add)

        with pytest.raises(ValueError, match="Nie udało się dodać danych do kolekcji"):
            market_memory.add_market_data(
                data=sample_data,
                symbol="EURUSD",
                timeframe="D1"
            )

    def test_obsluga_bledow_wyszukiwania_pelna(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający pełną obsługę błędów podczas wyszukiwania."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test z błędem podczas wyszukiwania
        def mock_query(*args, **kwargs):
            raise Exception("Błąd podczas wyszukiwania")
            
        monkeypatch.setattr(market_memory.collection, "query", mock_query)
        
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0
        
        # Test z błędem podczas parsowania JSON
        def mock_loads(*args, **kwargs):
            raise json.JSONDecodeError("Błąd parsowania", "", 0)
            
        monkeypatch.setattr(json, "loads", mock_loads)
        
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

    def test_obsluga_bledow_usuwania_pelna(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający pełną obsługę błędów podczas usuwania danych."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test z błędem podczas pobierania dokumentów
        def mock_get(*args, **kwargs):
            raise Exception("Błąd podczas pobierania dokumentów")
            
        monkeypatch.setattr(market_memory.collection, "get", mock_get)
        
        assert market_memory.delete_market_data(
            symbol="EURUSD",
            before_date=datetime.now()
        ) == False
        
        # Test z błędem podczas usuwania dokumentów
        def mock_delete(*args, **kwargs):
            raise Exception("Błąd podczas usuwania dokumentów")
            
        monkeypatch.setattr(market_memory.collection, "delete", mock_delete)
        
        assert market_memory.delete_market_data(symbol="EURUSD") == False

    def test_obsluga_bledow_destruktora(self, market_memory, monkeypatch):
        """Test sprawdzający obsługę błędów w destruktorze."""
        # Test z błędem podczas usuwania kolekcji
        def mock_delete_collection(*args, **kwargs):
            raise Exception("Błąd podczas usuwania kolekcji")
            
        monkeypatch.setattr(market_memory.client, "delete_collection", mock_delete_collection)
        
        # Wywołaj destruktor ręcznie
        market_memory.__del__()
        
        assert market_memory.collection is None
        assert market_memory.client is None

    def test_obsluga_bledow_usuwania_wsadow(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający obsługę błędów podczas usuwania wsadów danych."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test z błędem podczas usuwania wsadu
        def mock_delete(*args, **kwargs):
            if 'ids' in kwargs:
                raise Exception("Błąd podczas usuwania wsadu")
            return True
            
        monkeypatch.setattr(market_memory.collection, "delete", mock_delete)
        
        # Usuń dane przed określoną datą
        assert market_memory.delete_market_data(
            symbol="EURUSD",
            before_date=datetime.now()
        ) == False

    def test_obsluga_bledow_przetwarzania_daty_dokumentu(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający obsługę błędów podczas przetwarzania daty dokumentu."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )
        
        # Test z błędem podczas pobierania dokumentów
        def mock_get(*args, **kwargs):
            return {
                'ids': ['test_id'],
                'metadatas': [{'timestamp': 'nieprawidłowa_data'}]
            }
            
        monkeypatch.setattr(market_memory.collection, "get", mock_get)
        
        # Usuń dane przed określoną datą
        assert market_memory.delete_market_data(
            symbol="EURUSD",
            before_date=datetime.now()
        ) == True

    def test_obsluga_bledow_usuwania_kolekcji_szczegolowa(self, market_memory, monkeypatch):
        """Test sprawdzający szczegółową obsługę błędów podczas usuwania kolekcji."""
        # Test z błędem podczas usuwania dokumentów
        def mock_delete(*args, **kwargs):
            if 'where' in kwargs:
                raise Exception("Błąd podczas usuwania dokumentów")
            return True
            
        monkeypatch.setattr(market_memory.collection, "delete", mock_delete)
        
        # Wywołaj close()
        market_memory.close()
        assert market_memory.collection is None
        assert market_memory.client is None

    def test_obsluga_bledow_destruktora_szczegolowa(self, market_memory):
        """Test sprawdzający szczegółową obsługę błędów w destruktorze."""
        # Test z błędem podczas usuwania kolekcji
        def mock_delete_collection(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania kolekcji")

        if hasattr(market_memory, 'client') and market_memory.client:
            # Podmień metodę delete_collection na mock
            market_memory.client.delete_collection = mock_delete_collection

            # Wywołaj destruktor ręcznie
            market_memory.__del__()

            # Sprawdź, czy błąd został obsłużony (powinien być wyświetlony w stdout)
            # Nie próbujemy przywracać oryginalnej metody, bo client jest już None
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_zamykania_polaczen_z_wyjatkiem(self, market_memory, monkeypatch):
        """Test sprawdzający szczegółową obsługę błędów podczas zamykania połączeń z wyjątkiem."""
        def mock_delete(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania dokumentów")
            
        if hasattr(market_memory, 'collection') and market_memory.collection:
            monkeypatch.setattr(market_memory.collection, "delete", mock_delete)
            
            # Wywołaj close()
            market_memory.close()
            
            # Sprawdź czy połączenia zostały zamknięte mimo błędu
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_usuwania_kolekcji_w_close(self, market_memory, monkeypatch):
        """Test sprawdzający obsługę błędów podczas usuwania kolekcji w metodzie close."""
        def mock_delete_collection(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania kolekcji")
            
        if hasattr(market_memory, 'client') and market_memory.client:
            monkeypatch.setattr(market_memory.client, "delete_collection", mock_delete_collection)
            
            # Wywołaj close()
            market_memory.close()
            
            # Sprawdź czy połączenia zostały zamknięte mimo błędu
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_parsowania_daty_indeksu_szczegolowa(self, market_memory, sample_data):
        """Test sprawdzający szczegółową obsługę błędów podczas parsowania daty indeksu."""
        # Przygotuj dane z indeksem, który spowoduje błąd podczas konwersji
        bad_data = sample_data.copy()
        bad_data.index = pd.Index(['nieprawidłowa_data'] * len(bad_data))

        with pytest.raises(ValueError, match="Nieprawidłowy format daty w indeksie"):
            market_memory.add_market_data(
                data=bad_data,
                symbol="EURUSD",
                timeframe="D1"
            )

    def test_obsluga_bledow_wyszukiwania_z_nieprawidlowymi_metadanymi_szczegolowa(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający szczegółową obsługę błędów podczas wyszukiwania z nieprawidłowymi metadanymi."""
        # Mockuj metodę query aby zwracała wyniki z nieprawidłowymi metadanymi
        def mock_query(*args, **kwargs):
            return {
                'documents': [['test']],
                'metadatas': [[{'symbol': 'EURUSD', 'timeframe': 'D1', 'data': 'invalid_json'}]],
                'distances': [[0.5]]
            }
            
        monkeypatch.setattr(market_memory.collection, "query", mock_query)
        
        # Sprawdź czy funkcja obsługuje błąd i kontynuuje przetwarzanie
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

    def test_obsluga_bledow_zamykania_polaczen_szczegolowa_z_wyjatkiem(self, market_memory, monkeypatch):
        """Test sprawdzający szczegółową obsługę błędów podczas zamykania połączeń z wyjątkiem."""
        def mock_delete(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania dokumentów")
            
        if hasattr(market_memory, 'collection') and market_memory.collection:
            monkeypatch.setattr(market_memory.collection, "delete", mock_delete)
            
            # Wywołaj close()
            market_memory.close()
            
            # Sprawdź czy połączenia zostały zamknięte mimo błędu
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_destruktora_szczegolowa_z_wyjatkiem(self, market_memory, monkeypatch):
        """Test sprawdzający szczegółową obsługę błędów w destruktorze z wyjątkiem."""
        def mock_delete_collection(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania kolekcji")
            
        if hasattr(market_memory, 'client') and market_memory.client:
            monkeypatch.setattr(market_memory.client, "delete_collection", mock_delete_collection)
            
            # Wywołaj destruktor ręcznie
            market_memory.__del__()
            
            # Sprawdź czy obiekt został prawidłowo wyczyszczony mimo błędu
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_usuwania_kolekcji_w_close_szczegolowa(self, market_memory, monkeypatch):
        """Test sprawdzający szczegółową obsługę błędów podczas usuwania kolekcji w metodzie close."""
        def mock_delete_collection(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania kolekcji")
            
        if hasattr(market_memory, 'client') and market_memory.client:
            monkeypatch.setattr(market_memory.client, "delete_collection", mock_delete_collection)
            
            # Wywołaj close()
            market_memory.close()
            
            # Sprawdź czy połączenia zostały zamknięte mimo błędu
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_wyszukiwania_z_pustymi_wynikami_szczegolowa(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający szczegółową obsługę błędów podczas wyszukiwania z pustymi wynikami."""
        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )

        # Mockuj metodę query aby zwracała puste wyniki
        def mock_query(*args, **kwargs):
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        monkeypatch.setattr(market_memory.collection, "query", mock_query)
        
        # Sprawdź czy funkcja zwraca pustą listę
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

        # Test z błędem podczas wyszukiwania
        def mock_query_error(*args, **kwargs):
            raise RuntimeError("Błąd podczas wyszukiwania")
        
        monkeypatch.setattr(market_memory.collection, "query", mock_query_error)
        
        # Sprawdź czy funkcja obsługuje błąd i zwraca pustą listę
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

        # Test z nieprawidłowymi metadanymi
        def mock_query_bad_metadata(*args, **kwargs):
            return {
                'documents': [['test']],
                'metadatas': [[{'symbol': 'EURUSD', 'timeframe': 'D1', 'data': None}]],
                'distances': [[0.5]]
            }
        
        monkeypatch.setattr(market_memory.collection, "query", mock_query_bad_metadata)
        
        # Sprawdź czy funkcja obsługuje błąd i kontynuuje przetwarzanie
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

        # Test z błędem podczas zamykania połączeń
        def mock_delete_error(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania dokumentów")
        
        monkeypatch.setattr(market_memory.collection, "delete", mock_delete_error)
        
        # Wywołaj close()
        market_memory.close()
        
        # Sprawdź czy połączenia zostały zamknięte mimo błędu
        assert market_memory.collection is None
        assert market_memory.client is None

        # Test z błędem podczas usuwania kolekcji w destruktorze
        def mock_delete_collection_error(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania kolekcji")
        
        if hasattr(market_memory, 'client') and market_memory.client:
            monkeypatch.setattr(market_memory.client, "delete_collection", mock_delete_collection_error)
            
            # Wywołaj destruktor ręcznie
            market_memory.__del__()
            
            # Sprawdź czy obiekt został prawidłowo wyczyszczony mimo błędu
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_pelna_sciezka(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający pełną ścieżkę obsługi błędów."""
        # Test z błędem podczas konwersji daty
        bad_data = sample_data.copy()
        bad_data.index = pd.Index(['2024-13-45'] * len(bad_data))  # Nieprawidłowa data

        with pytest.raises(ValueError, match="Nieprawidłowy format daty w indeksie"):
            market_memory.add_market_data(
                data=bad_data,
                symbol="EURUSD",
                timeframe="D1"
            )

        # Dodaj przykładowe dane
        market_memory.add_market_data(
            data=sample_data,
            symbol="EURUSD",
            timeframe="D1"
        )

        # Test z błędem podczas wyszukiwania
        def mock_query_error(*args, **kwargs):
            raise RuntimeError("Błąd podczas wyszukiwania")
        
        monkeypatch.setattr(market_memory.collection, "query", mock_query_error)
        
        # Sprawdź czy funkcja obsługuje błąd i zwraca pustą listę
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

        # Test z nieprawidłowymi metadanymi
        def mock_query_bad_metadata(*args, **kwargs):
            return {
                'documents': [['test']],
                'metadatas': [[{'symbol': 'EURUSD', 'timeframe': 'D1', 'data': None}]],
                'distances': [[0.5]]
            }
        
        monkeypatch.setattr(market_memory.collection, "query", mock_query_bad_metadata)
        
        # Sprawdź czy funkcja obsługuje błąd i kontynuuje przetwarzanie
        results = market_memory.query_market_data(
            query_text="EURUSD",
            symbol="EURUSD"
        )
        assert len(results) == 0

        # Test z błędem podczas zamykania połączeń
        def mock_delete_error(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania dokumentów")
        
        monkeypatch.setattr(market_memory.collection, "delete", mock_delete_error)
        
        # Wywołaj close()
        market_memory.close()
        
        # Sprawdź czy połączenia zostały zamknięte mimo błędu
        assert market_memory.collection is None
        assert market_memory.client is None

        # Test z błędem podczas usuwania kolekcji w destruktorze
        def mock_delete_collection_error(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania kolekcji")
        
        if hasattr(market_memory, 'client') and market_memory.client:
            monkeypatch.setattr(market_memory.client, "delete_collection", mock_delete_collection_error)
            
            # Wywołaj destruktor ręcznie
            market_memory.__del__()
            
            # Sprawdź czy obiekt został prawidłowo wyczyszczony mimo błędu
            assert market_memory.collection is None
            assert market_memory.client is None

    def test_obsluga_bledow_pelna_sciezka_z_nieprawidlowym_indeksem(self, market_memory, sample_data, monkeypatch):
        """Test sprawdzający obsługę błędów dla różnych formatów daty i nieprawidłowych indeksów."""
        # Test z indeksem w złym formacie (linia 82)
        bad_data = sample_data.copy()
        bad_data.index = pd.Index(['2024/01/01'] * len(bad_data))  # Nieprawidłowy format daty

        with pytest.raises(ValueError, match="Nieprawidłowy format daty w indeksie"):
            market_memory.add_market_data(
                data=bad_data,
                symbol="EURUSD",
                timeframe="D1"
            )

        # Test z błędem podczas wyszukiwania (linie 275, 286)
        def mock_query(*args, **kwargs):
            raise RuntimeError("Błąd podczas wyszukiwania")
        
        if hasattr(market_memory, 'collection') and market_memory.collection:
            original_query = market_memory.collection.query
            market_memory.collection.query = mock_query
            
            results = market_memory.query_market_data(
                query_text="EURUSD",
                symbol="EURUSD"
            )
            assert len(results) == 0
            
            market_memory.collection.query = original_query

        # Test z błędem podczas zamykania połączeń (linie 316-318)
        def mock_delete(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania dokumentów")
        
        if hasattr(market_memory, 'collection') and market_memory.collection:
            original_delete = market_memory.collection.delete
            market_memory.collection.delete = mock_delete
            
            # Wywołaj close()
            market_memory.close()
            assert market_memory.collection is None
            assert market_memory.client is None
            
            if hasattr(market_memory, 'collection') and market_memory.collection:
                market_memory.collection.delete = original_delete

        # Test z błędem w destruktorze (linie 341-342)
        market_memory = MarketMemory()
        def mock_delete_collection(*args, **kwargs):
            raise RuntimeError("Błąd podczas usuwania kolekcji")
        
        if hasattr(market_memory, 'client') and market_memory.client:
            original_delete_collection = market_memory.client.delete_collection
            market_memory.client.delete_collection = mock_delete_collection
            
            # Wywołaj destruktor ręcznie
            market_memory.__del__()
            assert market_memory.collection is None
            assert market_memory.client is None
            
            if hasattr(market_memory, 'client') and market_memory.client:
                market_memory.client.delete_collection = original_delete_collection

        # Test z błędem podczas usuwania kolekcji w close (linie 355-357)
        market_memory = MarketMemory()
        if hasattr(market_memory, 'client') and market_memory.client:
            market_memory.client.delete_collection = mock_delete_collection
            
            # Wywołaj close()
            market_memory.close()
            assert market_memory.collection is None
            assert market_memory.client is None 