"""Moduł do obsługi embeddingów dla systemu RAG."""

from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime

class EmbeddingsHandler:
    """Klasa do generowania i zarządzania embeddingami."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Inicjalizuje handler embeddingów.
        
        Args:
            model_name: Nazwa modelu SentenceTransformers do użycia
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generuje embeddingi dla listy tekstów.
        
        Args:
            texts: Lista tekstów do przetworzenia
            
        Returns:
            Tablica numpy z embeddingami
            
        Raises:
            TypeError: Gdy texts nie jest listą stringów
            ValueError: Gdy lista jest pusta lub None
        """
        if texts is None:
            raise ValueError("Texts nie może być None")
        if not isinstance(texts, list):
            raise TypeError("Texts musi być listą")
        if not texts:
            raise ValueError("Lista tekstów nie może być pusta")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Wszystkie elementy muszą być typu str")
            
        return self.model.encode(texts, convert_to_numpy=True)
        
    def calculate_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """
        Oblicza podobieństwo między zapytaniem a dokumentami.
        
        Args:
            query_embedding: Embedding zapytania
            document_embeddings: Embeddingi dokumentów
            
        Returns:
            Tablica podobieństw kosinusowych
            
        Raises:
            TypeError: Gdy embeddingi nie są tablicami numpy
            ValueError: Gdy wymiary są nieprawidłowe
        """
        if not isinstance(query_embedding, np.ndarray) or not isinstance(document_embeddings, np.ndarray):
            raise TypeError("Embeddingi muszą być tablicami numpy")
            
        if len(query_embedding.shape) != 2 or query_embedding.shape[0] != 1:
            raise ValueError("Query embedding musi mieć kształt (1, dimension)")
            
        return np.dot(document_embeddings, query_embedding.T).flatten() / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
    def get_most_similar(self, query: str, documents: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """
        Znajduje k najbardziej podobnych dokumentów do zapytania.
        
        Args:
            query: Tekst zapytania
            documents: Lista dokumentów do przeszukania
            k: Liczba dokumentów do zwrócenia
            
        Returns:
            Lista słowników z dokumentami i ich podobieństwem
            
        Raises:
            TypeError: Gdy parametry mają nieprawidłowy typ
            ValueError: Gdy k > liczby dokumentów
        """
        if not isinstance(query, str):
            raise TypeError("Query musi być typu str")
        if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
            raise TypeError("Documents musi być listą stringów")
        if not isinstance(k, int):
            raise TypeError("k musi być typu int")
        if k > len(documents):
            raise ValueError("k nie może być większe niż liczba dokumentów")
            
        query_embedding = self.generate_embeddings([query])
        doc_embeddings = self.generate_embeddings(documents)
        
        similarities = self.calculate_similarity(query_embedding, doc_embeddings)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'document': documents[idx],
                'similarity': float(similarities[idx])
            })
            
        return results

    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], save_path: str, metadata_path: str) -> None:
        """
        Zapisuje embeddingi i metadane do plików.
        
        Args:
            embeddings: Tablica numpy z embeddingami
            texts: Lista tekstów odpowiadających embeddingom
            save_path: Ścieżka do zapisu embeddingów
            metadata_path: Ścieżka do zapisu metadanych
            
        Raises:
            TypeError: Gdy parametry mają nieprawidłowy typ
            ValueError: Gdy ścieżki są puste lub liczba tekstów nie zgadza się z liczbą embeddingów
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings musi być tablicą numpy")
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("Texts musi być listą stringów")
        if not save_path or not metadata_path:
            raise ValueError("Ścieżki nie mogą być puste")
        if len(texts) != embeddings.shape[0]:
            raise ValueError("Liczba tekstów musi odpowiadać liczbie embeddingów")
            
        # Zapisz embeddingi
        np.savez(save_path, embeddings=embeddings)
        
        # Zapisz metadane
        metadata = {
            'texts': texts,
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat()
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
    def load_embeddings(self, load_path: str, metadata_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Wczytuje embeddingi i metadane z plików.
        
        Args:
            load_path: Ścieżka do pliku z embeddingami
            metadata_path: Ścieżka do pliku z metadanymi
            
        Returns:
            Krotka (embeddingi, metadane)
            
        Raises:
            FileNotFoundError: Gdy pliki nie istnieją
            ValueError: Gdy dane są niezgodne
            json.JSONDecodeError: Gdy plik metadanych jest uszkodzony
        """
        # Wczytaj embeddingi
        data = np.load(load_path)
        embeddings = data['embeddings']
        
        # Wczytaj metadane
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # Sprawdź czy metadane zawierają wymagane pola
        if 'texts' not in metadata:
            raise ValueError("Brak pola 'texts' w metadanych")
            
        # Sprawdź zgodność danych
        if len(metadata['texts']) != embeddings.shape[0]:
            raise ValueError("Niezgodność między liczbą tekstów a liczbą embeddingów")
            
        return embeddings, metadata 