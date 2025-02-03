"""
Testy jednostkowe dla modułu embeddings_handler.py
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any
import time
import psutil
import os
import tempfile
import json

from src.rag.embeddings_handler import EmbeddingsHandler


@pytest.fixture
def mock_sentence_transformer():
    """Fixture dostarczający zamockowany model SentenceTransformer."""
    with patch('src.rag.embeddings_handler.SentenceTransformer') as mock:
        model = Mock()
        # Przygotuj przykładowe embeddingi
        model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock.return_value = model
        yield mock


@pytest.fixture
def embeddings_handler(mock_sentence_transformer):
    """Fixture dostarczający instancję EmbeddingsHandler z zamockowanym modelem."""
    return EmbeddingsHandler(model_name='test-model')


class TestEmbeddingsHandler:
    """Testy dla klasy EmbeddingsHandler."""

    def test_initialization(self, embeddings_handler, mock_sentence_transformer):
        """Test poprawnej inicjalizacji handlera."""
        assert isinstance(embeddings_handler, EmbeddingsHandler)
        mock_sentence_transformer.assert_called_once_with('test-model')

    def test_generate_embeddings(self, embeddings_handler):
        """Test generowania embeddingów."""
        texts = [
            "Przykładowy tekst 1",
            "Przykładowy tekst 2",
            "Przykładowy tekst 3"
        ]

        embeddings = embeddings_handler.generate_embeddings(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 3)  # 3 teksty, 3 wymiary
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64

        # Sprawdź czy model został wywołany poprawnie
        embeddings_handler.model.encode.assert_called_once_with(texts, convert_to_numpy=True)

    def test_calculate_similarity(self, embeddings_handler):
        """Test obliczania podobieństwa między embeddingami."""
        query_embedding = np.array([[1.0, 0.0, 0.0]])  # wektor jednostkowy w kierunku x
        document_embeddings = np.array([
            [1.0, 0.0, 0.0],  # identyczny z query (podobieństwo = 1)
            [0.0, 1.0, 0.0],  # prostopadły do query (podobieństwo = 0)
            [0.5, 0.5, 0.0]   # pod kątem 45 stopni (podobieństwo ≈ 0.707)
        ])

        similarities = embeddings_handler.calculate_similarity(query_embedding, document_embeddings)

        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (3,)
        assert np.isclose(similarities[0], 1.0)  # identyczne wektory
        assert np.isclose(similarities[1], 0.0)  # prostopadłe wektory
        assert np.isclose(similarities[2], 0.707, atol=0.001)  # kąt 45 stopni

    def test_get_most_similar(self, embeddings_handler):
        """Test znajdowania najbardziej podobnych dokumentów."""
        query = "Zapytanie testowe"
        documents = [
            "Dokument pierwszy",
            "Dokument drugi",
            "Dokument trzeci"
        ]

        # Ustaw mock dla generate_embeddings
        query_embedding = np.array([[0.1, 0.2, 0.3]])
        doc_embeddings = np.array([
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.1, 0.2, 0.3]
        ])

        with patch.object(embeddings_handler, 'generate_embeddings') as mock_generate:
            mock_generate.side_effect = [query_embedding, doc_embeddings]
            results = embeddings_handler.get_most_similar(query, documents, k=2)

        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert 'document' in result
            assert 'similarity' in result
            assert isinstance(result['similarity'], float)
            assert 0 <= result['similarity'] <= 1

    def test_error_handling(self, embeddings_handler):
        """Test obsługi błędów."""
        # Test dla pustej listy tekstów
        with pytest.raises(ValueError):
            embeddings_handler.generate_embeddings([])

        # Test dla None
        with pytest.raises(ValueError):
            embeddings_handler.generate_embeddings(None)

        # Test dla nieprawidłowych wymiarów
        with pytest.raises(ValueError):
            query_embedding = np.array([0.1, 0.2, 0.3])  # brak wymiaru batch
            doc_embeddings = np.array([[0.4, 0.5, 0.6]])
            embeddings_handler.calculate_similarity(query_embedding, doc_embeddings)

        # Test dla k > liczby dokumentów
        query = "Test"
        documents = ["Doc1", "Doc2"]
        with pytest.raises(ValueError):
            embeddings_handler.get_most_similar(query, documents, k=3)

    def test_input_validation(self, embeddings_handler):
        """Test walidacji danych wejściowych."""
        # Test nieprawidłowych typów tekstów
        with pytest.raises(TypeError):
            embeddings_handler.generate_embeddings(123)

        with pytest.raises(TypeError):
            embeddings_handler.generate_embeddings(["ok", 123, "not ok"])

        # Test nieprawidłowych typów dla calculate_similarity
        with pytest.raises(TypeError):
            embeddings_handler.calculate_similarity(
                [1, 2, 3],  # nie numpy array
                np.array([[1, 2, 3]])
            )

        # Test nieprawidłowych typów dla get_most_similar
        with pytest.raises(TypeError):
            embeddings_handler.get_most_similar(123, ["doc1", "doc2"])

        with pytest.raises(TypeError):
            embeddings_handler.get_most_similar("query", [1, 2, 3])

        with pytest.raises(TypeError):
            embeddings_handler.get_most_similar("query", ["doc1", "doc2"], k="3")

    def test_save_embeddings(self, embeddings_handler, tmp_path):
        """Test zapisywania embeddingów do pliku."""
        # Przygotuj dane testowe
        texts = [
            "Przykładowy tekst 1",
            "Przykładowy tekst 2",
            "Przykładowy tekst 3"
        ]
        embeddings = embeddings_handler.generate_embeddings(texts)
        
        # Przygotuj ścieżkę do zapisu
        save_path = tmp_path / "embeddings.npz"
        metadata_path = tmp_path / "metadata.json"
        
        # Zapisz embeddingi
        embeddings_handler.save_embeddings(
            embeddings=embeddings,
            texts=texts,
            save_path=str(save_path),
            metadata_path=str(metadata_path)
        )
        
        # Sprawdź czy pliki zostały utworzone
        assert save_path.exists()
        assert metadata_path.exists()
        
        # Sprawdź zawartość plików
        loaded_embeddings = np.load(str(save_path))['embeddings']
        np.testing.assert_array_equal(embeddings, loaded_embeddings)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            assert 'texts' in metadata
            assert metadata['texts'] == texts
            assert 'model_name' in metadata
            assert metadata['model_name'] == 'test-model'
            assert 'created_at' in metadata

    def test_load_embeddings(self, embeddings_handler, tmp_path):
        """Test wczytywania embeddingów z pliku."""
        # Przygotuj dane testowe
        texts = [
            "Przykładowy tekst 1",
            "Przykładowy tekst 2",
            "Przykładowy tekst 3"
        ]
        original_embeddings = embeddings_handler.generate_embeddings(texts)
        
        # Zapisz embeddingi do plików tymczasowych
        save_path = tmp_path / "embeddings.npz"
        metadata_path = tmp_path / "metadata.json"
        
        np.savez(save_path, embeddings=original_embeddings)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': texts,
                'model_name': 'test-model',
                'created_at': '2024-01-01T00:00:00'
            }, f, ensure_ascii=False)
        
        # Wczytaj embeddingi
        loaded_embeddings, loaded_metadata = embeddings_handler.load_embeddings(
            load_path=str(save_path),
            metadata_path=str(metadata_path)
        )
        
        # Sprawdź czy dane są poprawne
        np.testing.assert_array_equal(original_embeddings, loaded_embeddings)
        assert loaded_metadata['texts'] == texts
        assert loaded_metadata['model_name'] == 'test-model'
        
    def test_save_embeddings_errors(self, embeddings_handler):
        """Test obsługi błędów podczas zapisywania embeddingów."""
        # Test nieprawidłowej ścieżki
        with pytest.raises(ValueError):
            embeddings_handler.save_embeddings(
                embeddings=np.array([[1, 2, 3]]),
                texts=["test"],
                save_path="",
                metadata_path=""
            )
        
        # Test niezgodności wymiarów
        with pytest.raises(ValueError):
            embeddings_handler.save_embeddings(
                embeddings=np.array([[1, 2, 3]]),
                texts=["test1", "test2"],  # więcej tekstów niż embeddingów
                save_path="test.npz",
                metadata_path="test.json"
            )
        
        # Test nieprawidłowych typów danych
        with pytest.raises(TypeError):
            embeddings_handler.save_embeddings(
                embeddings=[1, 2, 3],  # nie numpy array
                texts=["test"],
                save_path="test.npz",
                metadata_path="test.json"
            )
            
        # Test nieprawidłowej listy tekstów
        with pytest.raises(TypeError):
            embeddings_handler.save_embeddings(
                embeddings=np.array([[1, 2, 3]]),
                texts=[1, 2, 3],  # nie lista stringów
                save_path="test.npz",
                metadata_path="test.json"
            )

    def test_load_embeddings_errors(self, embeddings_handler, tmp_path):
        """Test obsługi błędów podczas wczytywania embeddingów."""
        # Test nieistniejącego pliku
        with pytest.raises(FileNotFoundError):
            embeddings_handler.load_embeddings(
                load_path="nieistniejacy.npz",
                metadata_path="nieistniejacy.json"
            )

        # Test uszkodzonego pliku metadanych
        embeddings_path = tmp_path / "embeddings.npz"
        metadata_path = tmp_path / "metadata.json"
        
        # Zapisz nieprawidłowe dane
        np.savez(embeddings_path, embeddings=np.array([[1, 2, 3]]))
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({'invalid': 'data'}, f, ensure_ascii=False)
            
        with pytest.raises(ValueError):
            embeddings_handler.load_embeddings(
                load_path=str(embeddings_path),
                metadata_path=str(metadata_path)
            )
            
        # Test niezgodności wymiarów
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': ["test1", "test2"],  # więcej tekstów niż embeddingów
                'model_name': 'test-model'
            }, f, ensure_ascii=False)
            
        with pytest.raises(ValueError):
            embeddings_handler.load_embeddings(
                load_path=str(embeddings_path),
                metadata_path=str(metadata_path)
            )


class TestEmbeddingsHandlerPerformance:
    """Testy wydajnościowe i pamięciowe dla klasy EmbeddingsHandler."""

    @pytest.fixture
    def real_embeddings_handler(self):
        """Fixture dostarczający prawdziwą (nie mockowaną) instancję EmbeddingsHandler."""
        return EmbeddingsHandler(model_name='paraphrase-multilingual-MiniLM-L12-v2')

    def test_generate_embeddings_performance(self, real_embeddings_handler):
        """Test wydajności generowania embeddingów."""
        # Przygotuj dane testowe
        texts = [f"Przykładowy długi tekst numer {i} do testu wydajności generowania embeddingów" for i in range(100)]
        
        # Zmierz czas wykonania
        start_time = time.time()
        embeddings = real_embeddings_handler.generate_embeddings(texts)
        execution_time = time.time() - start_time

        # Asercje wydajnościowe
        assert execution_time < 5.0, f"Generowanie embeddingów trwało zbyt długo: {execution_time:.2f}s"
        assert embeddings.shape[0] == 100, "Nieprawidłowa liczba wygenerowanych embeddingów"

    def test_memory_usage(self, real_embeddings_handler):
        """Test zużycia pamięci podczas pracy z dużą ilością danych."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generuj embeddingi dla dużej ilości tekstów
        texts = [
            f"Długi tekst testowy numer {i} do sprawdzenia zużycia pamięci " * 10
            for i in range(1000)
        ]

        # Wykonaj operacje i zmierz zużycie pamięci
        real_embeddings_handler.generate_embeddings(texts)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Asercje pamięciowe
        assert memory_increase < 1000, f"Zbyt duży wzrost zużycia pamięci: {memory_increase:.2f}MB"

    def test_batch_processing_performance(self, real_embeddings_handler):
        """Test wydajności przetwarzania wsadowego."""
        # Przygotuj duży zestaw danych
        query = "Zapytanie testowe"
        documents = [f"Dokument testowy numer {i}" for i in range(1000)]

        # Zmierz czas wykonania dla różnych rozmiarów wsadu
        batch_sizes = [10, 50, 100]
        times = {}

        for batch_size in batch_sizes:
            start_time = time.time()
            results = real_embeddings_handler.get_most_similar(
                query=query,
                documents=documents[:batch_size],
                k=5
            )
            times[batch_size] = time.time() - start_time

        # Asercje wydajnościowe dla różnych rozmiarów wsadu
        assert times[10] < 1.0, f"Przetwarzanie małego wsadu trwało zbyt długo: {times[10]:.2f}s"
        assert times[50] < 3.0, f"Przetwarzanie średniego wsadu trwało zbyt długo: {times[50]:.2f}s"
        assert times[100] < 5.0, f"Przetwarzanie dużego wsadu trwało zbyt długo: {times[100]:.2f}s"

    def test_similarity_calculation_performance(self, real_embeddings_handler):
        """Test wydajności obliczania podobieństwa dla dużej liczby dokumentów."""
        # Generuj duży zestaw embeddingów
        query_embedding = np.random.rand(1, 384)  # wymiar modelu MiniLM
        doc_embeddings = np.random.rand(10000, 384)  # 10000 dokumentów

        start_time = time.time()
        similarities = real_embeddings_handler.calculate_similarity(query_embedding, doc_embeddings)
        execution_time = time.time() - start_time

        # Asercje wydajnościowe
        assert execution_time < 1.0, f"Obliczanie podobieństwa trwało zbyt długo: {execution_time:.2f}s"
        assert similarities.shape[0] == 10000, "Nieprawidłowa liczba wyników podobieństwa"

    def test_concurrent_requests_performance(self, real_embeddings_handler):
        """Test wydajności przy współbieżnych żądaniach."""
        import concurrent.futures

        def process_query(query_text: str) -> List[Dict[str, Any]]:
            documents = [f"Dokument testowy {i}" for i in range(50)]
            return real_embeddings_handler.get_most_similar(query_text, documents, k=5)

        queries = [f"Zapytanie testowe {i}" for i in range(10)]
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_query, queries))

        execution_time = time.time() - start_time

        # Asercje wydajnościowe
        assert execution_time < 10.0, f"Przetwarzanie współbieżne trwało zbyt długo: {execution_time:.2f}s"
        assert len(results) == 10, "Nieprawidłowa liczba wyników"
        for result in results:
            assert len(result) == 5, "Nieprawidłowa liczba podobnych dokumentów"

    def test_model_loading_performance(self):
        """Test wydajności ładowania modelu."""
        start_time = time.time()
        handler = EmbeddingsHandler(model_name='paraphrase-multilingual-MiniLM-L12-v2')
        loading_time = time.time() - start_time

        # Asercje wydajnościowe
        assert loading_time < 10.0, f"Ładowanie modelu trwało zbyt długo: {loading_time:.2f}s"
        assert handler.model is not None, "Model nie został załadowany" 