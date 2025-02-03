"""Moduł do zarządzania pamięcią rynku w systemie RAG."""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import pandas as pd
from datetime import datetime
import json
import numpy as np
import os
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MarketMemory:
    """Klasa do przechowywania i wyszukiwania informacji o rynku."""
    
    def __init__(self, persist_directory: str = "market_memory", allow_reset: bool = False):
        """
        Inicjalizuje pamięć rynku.
        
        Args:
            persist_directory: Katalog do przechowywania danych
            allow_reset: Czy zezwolić na resetowanie stanu klienta
            
        Raises:
            ValueError: Gdy ścieżka jest nieprawidłowa lub nie ma uprawnień do zapisu
        """
        # Sprawdź czy ścieżka jest prawidłowa
        path = Path(persist_directory)
        if not path.parent.exists():
            raise ValueError(f"Katalog nadrzędny nie istnieje: {path.parent}")
            
        # Sprawdź uprawnienia do zapisu
        try:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".test"
            test_file.write_text("test")
            test_file.unlink()
        except (IOError, OSError) as e:
            raise ValueError(f"Brak uprawnień do zapisu w katalogu: {str(e)}")
            
        settings = Settings(
            allow_reset=allow_reset,
            anonymized_telemetry=False
        )
        self.client = chromadb.PersistentClient(path=str(path), settings=settings)
        self.collection = self.client.get_or_create_collection(
            name="market_data",
            metadata={"description": "Kolekcja danych rynkowych"}
        )
        
    def add_market_data(self, 
                       data: pd.DataFrame,
                       symbol: str,
                       timeframe: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       batch_size: int = 5000) -> None:
        """
        Dodaje dane rynkowe do pamięci.
        
        Args:
            data: DataFrame z danymi rynkowymi
            symbol: Symbol instrumentu
            timeframe: Interwał czasowy
            metadata: Dodatkowe metadane
            batch_size: Rozmiar wsadu przy dodawaniu danych
            
        Raises:
            ValueError: Gdy dane wejściowe są nieprawidłowe
        """
        # Walidacja danych wejściowych
        if data is None or data.empty:
            raise ValueError("Dane wejściowe nie mogą być puste")
            
        if not isinstance(symbol, str) or len(symbol) < 2:
            raise ValueError("Symbol musi być ciągiem znaków o długości co najmniej 2")
            
        if not isinstance(timeframe, str) or len(timeframe) < 1:
            raise ValueError("Timeframe musi być niepustym ciągiem znaków")
            
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Brakujące kolumny w danych: {', '.join(missing_columns)}")
        
        # Sprawdź poprawność indeksu czasowego
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        except Exception as e:
            raise ValueError(f"Nieprawidłowy format daty w indeksie: {str(e)}")
        
        # Podziel dane na mniejsze wsady
        for i in range(0, len(data), batch_size):
            batch_data = data.iloc[i:i+batch_size]
            
            documents = []
            metadatas = []
            embeddings = []
            ids = []
            
            for idx, row in batch_data.iterrows():
                # Przygotuj dokument z danymi rynkowymi
                doc = {
                    'timestamp': str(idx),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row.get('volume', 0))
                }
                
                # Dodaj wskaźniki techniczne jeśli są dostępne
                for col in row.index:
                    if col not in ['open', 'high', 'low', 'close', 'volume']:
                        doc[col] = float(row[col])
                
                # Przygotuj tekst dokumentu do embedowania
                doc_text = f"Symbol: {symbol}, Timeframe: {timeframe}, Data: {str(idx)}, " \
                          f"Open: {doc['open']}, High: {doc['high']}, Low: {doc['low']}, Close: {doc['close']}, " \
                          f"Volume: {doc['volume']}"
                
                documents.append(doc_text)
                embeddings.append(np.random.rand(384).tolist())  # Losowe embeddingi dla testu
                
                # Przygotuj metadane
                meta = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': str(idx),
                    'data': json.dumps(doc)
                }
                if metadata:
                    meta.update(metadata)
                metadatas.append(meta)
                
                # Generuj unikalny ID
                unique_id = f"{symbol}_{timeframe}_{str(idx)}"
                ids.append(unique_id)
            
            # Dodaj wsad do kolekcji
            try:
                if self.collection:
                    self.collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                else:
                    raise ValueError("Kolekcja nie jest zainicjalizowana")
            except Exception as e:
                logger.error(f"Błąd podczas dodawania danych do kolekcji: {str(e)}")
                raise ValueError(f"Nie udało się dodać danych do kolekcji: {str(e)}")
        
    def query_market_data(
        self,
        query_text: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Wyszukuje dane rynkowe na podstawie zapytania tekstowego i opcjonalnych filtrów.

        Args:
            query_text (str): Tekst zapytania do wyszukania.
            symbol (Optional[str]): Symbol instrumentu do filtrowania.
            timeframe (Optional[str]): Timeframe do filtrowania.
            n_results (int): Maksymalna liczba wyników do zwrócenia.

        Returns:
            List[Dict[str, Any]]: Lista znalezionych dokumentów.
            
        Raises:
            ValueError: Gdy parametry są nieprawidłowe
        """
        # Walidacja parametrów
        if not query_text or not isinstance(query_text, str):
            raise ValueError("Tekst zapytania nie może być pusty")
            
        if n_results < 1:
            raise ValueError("Liczba wyników musi być większa od 0")
            
        if symbol is not None:
            if not isinstance(symbol, str) or len(symbol) < 2:
                raise ValueError("Symbol musi być ciągiem znaków o długości co najmniej 2")
            
        if timeframe is not None:
            if not isinstance(timeframe, str) or len(timeframe) < 1:
                raise ValueError("Timeframe musi być niepustym ciągiem znaków")

        try:
            where = None
            if symbol:
                where = {"symbol": symbol}

            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )

            if not results or not results['documents']:
                return []

            processed_results = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                if timeframe and metadata.get('timeframe') != timeframe:
                    continue
                try:
                    data = json.loads(metadata.get('data', '{}'))
                    processed_results.append({
                        'data': data,
                        'metadata': {k: v for k, v in metadata.items() if k != 'data'},
                        'distance': results['distances'][0][len(processed_results)] if 'distances' in results else None
                    })
                except Exception as e:
                    logger.error(f"Błąd podczas przetwarzania wyniku: {str(e)}")
                    continue

            return processed_results

        except Exception as e:
            logger.error(f"Błąd podczas wyszukiwania: {str(e)}")
            return []

    def delete_market_data(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        before_date: Optional[datetime] = None,
        batch_size: int = 1000
    ) -> bool:
        """
        Usuwa dane rynkowe dla określonego symbolu i/lub timeframe'u.

        Args:
            symbol (Optional[str]): Symbol instrumentu do usunięcia.
            timeframe (Optional[str]): Timeframe do usunięcia.
            before_date (Optional[datetime]): Usuń dane przed tą datą.
            batch_size (int): Rozmiar wsadu przy usuwaniu danych.

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym razie.
            
        Raises:
            ValueError: Gdy parametry są nieprawidłowe
        """
        # Walidacja parametrów
        if symbol is not None:
            if not isinstance(symbol, str) or len(symbol) < 2:
                raise ValueError("Symbol musi być ciągiem znaków o długości co najmniej 2")
            
        if timeframe is not None:
            if not isinstance(timeframe, str) or len(timeframe) < 1:
                raise ValueError("Timeframe musi być niepustym ciągiem znaków")
            
        if before_date is not None and not isinstance(before_date, datetime):
            raise ValueError("before_date musi być obiektem datetime")

        try:
            if not self.collection:
                logger.error("Kolekcja nie jest zainicjalizowana")
                return False

            # Pobierz wszystkie dokumenty do filtrowania
            where = {}
            if symbol:
                where["symbol"] = symbol
            if timeframe:
                where["timeframe"] = timeframe

            if not where and not before_date:
                logger.warning("Brak kryteriów usuwania")
                return False

            try:
                if before_date:
                    # Pobierz dokumenty do sprawdzenia daty
                    results = self.collection.get(where=where if where else None)
                    if not results or not results['ids']:
                        return True

                    ids_to_delete = []
                    for i, metadata in enumerate(results['metadatas']):
                        try:
                            doc_date = datetime.fromisoformat(metadata.get('timestamp', ''))
                            if doc_date < before_date:
                                ids_to_delete.append(results['ids'][i])
                        except Exception as e:
                            logger.error(f"Błąd podczas przetwarzania daty dokumentu: {str(e)}")
                            continue

                    # Usuń dane w mniejszych wsadach
                    for i in range(0, len(ids_to_delete), batch_size):
                        batch = ids_to_delete[i:i + batch_size]
                        try:
                            self.collection.delete(ids=batch)
                        except Exception as e:
                            logger.error(f"Błąd podczas usuwania wsadu {i//batch_size + 1}: {str(e)}")
                            return False
                else:
                    # Usuń wszystkie dokumenty spełniające kryteria
                    self.collection.delete(where=where)

                return True

            except Exception as e:
                logger.error(f"Błąd podczas usuwania danych: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Błąd podczas usuwania danych: {str(e)}")
            return False
            
    def close(self) -> None:
        """
        Zamyka połączenia i zwalnia zasoby.
        """
        try:
            if hasattr(self, 'collection') and self.collection:
                try:
                    # Usuń wszystkie dokumenty z kolekcji
                    self.collection.delete(where={})
                except Exception as e:
                    logger.error(f"Błąd podczas usuwania kolekcji: {str(e)}")
                finally:
                    self.collection = None
            
            if hasattr(self, 'client') and self.client:
                try:
                    self.client.delete_collection("market_data")
                except Exception as e:
                    logger.error(f"Błąd podczas usuwania kolekcji: {str(e)}")
                finally:
                    self.client = None
        except Exception as e:
            logger.error(f"Błąd podczas zamykania połączeń: {str(e)}")
            
    def __del__(self):
        """Zamyka połączenia przy usuwaniu obiektu."""
        try:
            if hasattr(self, 'collection') and self.collection:
                try:
                    self.client.delete_collection("market_data")
                except Exception as e:
                    print(f"Błąd podczas usuwania kolekcji: {e}")
                self.collection = None
            if hasattr(self, 'client'):
                self.client = None
        except Exception as e:
            print(f"Błąd podczas zamykania połączeń: {e}")
            pass 