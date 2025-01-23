"""
Moduł implementujący lokalny system RAG (Retrieval Augmented Generation).
"""

import logging
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
import os

logger = logging.getLogger(__name__)

class SystemRAG:
    """Klasa implementująca lokalny system RAG."""
    
    def __init__(self, persist_directory: str = "./data/chroma"):
        """
        Inicjalizacja systemu RAG.
        
        Args:
            persist_directory: Katalog do przechowywania danych ChromaDB
        """
        self.logger = logger
        
        # Upewniamy się, że katalog istnieje
        os.makedirs(persist_directory, exist_ok=True)
        
        # Inicjalizacja klienta ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Utworzenie lub pobranie kolekcji
        self.collection = self.client.get_or_create_collection(
            name="nikkeininja",
            metadata={"hnsw:space": "cosine"}  # Używamy podobieństwa cosinusowego
        )
        
        self.logger.info("🥷 System RAG zainicjalizowany")
        
    def dodaj_dokument(
        self,
        tekst: str,
        metadata: Dict[str, Any],
        aktualizuj: bool = False
    ) -> bool:
        """
        Dodaje nowy dokument do bazy wiedzy.
        
        Args:
            tekst: Treść dokumentu
            metadata: Metadane dokumentu
            aktualizuj: Czy aktualizować istniejący dokument
            
        Returns:
            bool: True jeśli dodano pomyślnie
        """
        try:
            # Generowanie ID dokumentu
            dokument_id = metadata.get('id', str(hash(tekst)))
            
            # Dodanie ID do metadanych
            metadata = metadata.copy()
            metadata['id'] = dokument_id
            
            # Jeśli aktualizujemy, najpierw usuń stary dokument
            if aktualizuj:
                try:
                    self.collection.delete(
                        ids=[dokument_id]
                    )
                except Exception:
                    pass  # Ignorujemy błąd jeśli dokument nie istniał
            
            # Dodanie dokumentu
            self.collection.add(
                documents=[tekst],
                metadatas=[metadata],
                ids=[dokument_id]
            )
            
            self.logger.info(f"🥷 Dodano dokument: {dokument_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd dodawania dokumentu: {str(e)}")
            return False
        
    def wyszukaj(
        self,
        zapytanie: str,
        n_wynikow: int = 5,
        filtry: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Wyszukuje dokumenty podobne do zapytania.
        
        Args:
            zapytanie: Tekst zapytania
            n_wynikow: Liczba wyników do zwrócenia
            filtry: Opcjonalne filtry metadanych
            
        Returns:
            List[Dict[str, Any]]: Lista znalezionych dokumentów
        """
        try:
            # Przygotowanie filtrów
            where = {}
            if filtry:
                where.update(filtry)
            
            # Wyszukiwanie
            wyniki = self.collection.query(
                query_texts=[zapytanie],
                n_results=n_wynikow,
                where=where if where else None
            )
            
            # Formatowanie wyników
            dokumenty = []
            for i in range(len(wyniki['documents'][0])):
                dokumenty.append({
                    'tekst': wyniki['documents'][0][i],
                    'metadata': wyniki['metadatas'][0][i],
                    'id': wyniki['ids'][0][i],
                    'score': wyniki['distances'][0][i] if 'distances' in wyniki else None
                })
            
            self.logger.info(f"🥷 Znaleziono {len(dokumenty)} dokumentów dla zapytania: {zapytanie}")
            return dokumenty
            
        except Exception as e:
            self.logger.error(f"❌ Błąd wyszukiwania: {str(e)}")
            return []
            
    def usun_dokument(self, dokument_id: str) -> bool:
        """
        Usuwa dokument z bazy wiedzy.
        
        Args:
            dokument_id: ID dokumentu do usunięcia
            
        Returns:
            bool: True jeśli usunięto pomyślnie
        """
        try:
            self.collection.delete(ids=[dokument_id])
            self.logger.info(f"🥷 Usunięto dokument: {dokument_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Błąd usuwania dokumentu: {str(e)}")
            return False
            
    def wyczysc(self) -> bool:
        """
        Usuwa wszystkie dokumenty z bazy wiedzy.
        
        Returns:
            bool: True jeśli wyczyszczono pomyślnie
        """
        try:
            self.collection.delete(ids=self.collection.get()['ids'])
            self.logger.info("🥷 Wyczyszczono bazę wiedzy")
            return True
        except Exception as e:
            self.logger.error(f"❌ Błąd czyszczenia bazy: {str(e)}")
            return False 