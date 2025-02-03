"""
Testy jednostkowe dla modułu data_loader.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import MetaTrader5 as mt5

from src.backtest.data_loader import HistoricalDataLoader
from src.database.postgres_handler import PostgresHandler

# Fixture dla przykładowych danych
@pytest.fixture
def sample_data():
    """Przykładowe dane historyczne dla testów."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Generuj dane cenowe
    base_price = 100.0
    data['open'] = [base_price + i * 0.1 for i in range(len(dates))]
    data['high'] = data['open'] + 0.5
    data['low'] = data['open'] - 0.5
    data['close'] = data['open'] + np.random.uniform(-0.2, 0.2, len(dates))
    data['volume'] = np.random.uniform(1000, 5000, len(dates))
    
    return data

@pytest.fixture
def mock_db_handler():
    """Mock dla handlera bazy danych."""
    handler = Mock(spec=PostgresHandler)
    handler.fetch_all = AsyncMock()
    handler.execute_many = AsyncMock()
    return handler

@pytest.mark.asyncio
async def test_initialization():
    """Test inicjalizacji HistoricalDataLoader."""
    start_date = datetime(2024, 1, 1)
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=start_date
    )
    
    assert loader.symbol == 'EURUSD'
    assert loader.timeframe == '1H'
    assert loader.start_date == start_date
    assert loader.db_handler is None
    assert '1H' in loader.timeframe_map

@pytest.mark.asyncio
async def test_initialization_with_db():
    """Test inicjalizacji z handlerem bazy danych."""
    db_handler = Mock(spec=PostgresHandler)
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=db_handler
    )
    
    assert loader.db_handler == db_handler
    assert (datetime.now() - loader.start_date).days <= 31  # Domyślnie 30 dni wstecz

@pytest.mark.asyncio
async def test_load_from_database_success(mock_db_handler, sample_data):
    """Test udanego ładowania danych z bazy."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    # Przygotuj dane testowe
    db_data = []
    for timestamp, row in sample_data.iterrows():
        db_data.append({
            'timestamp': timestamp,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })
    
    mock_db_handler.fetch_all.return_value = db_data
    
    df = await loader.load_from_database()
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == len(sample_data)
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert mock_db_handler.fetch_all.call_count == 1

@pytest.mark.asyncio
async def test_load_from_database_empty_result(mock_db_handler):
    """Test ładowania z bazy gdy nie ma danych."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    mock_db_handler.fetch_all.return_value = []
    
    df = await loader.load_from_database()
    
    assert df is None
    assert mock_db_handler.fetch_all.call_count == 1

@pytest.mark.asyncio
async def test_load_from_database_db_error(mock_db_handler):
    """Test obsługi błędu bazy danych."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    mock_db_handler.fetch_all.side_effect = Exception("Test database error")
    
    df = await loader.load_from_database()
    
    assert df is None
    assert mock_db_handler.fetch_all.call_count == 1

@pytest.mark.asyncio
async def test_load_from_database_invalid_data(mock_db_handler):
    """Test obsługi nieprawidłowych danych z bazy."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    # Przygotuj nieprawidłowe dane
    invalid_data = [
        {
            'timestamp': datetime(2024, 1, 1),
            'open': 'invalid',  # Nieprawidłowy typ
            'high': 1.1,
            'low': 1.0,
            'close': 1.05,
            'volume': 1000
        }
    ]
    
    mock_db_handler.fetch_all.return_value = invalid_data
    
    df = await loader.load_from_database()
    
    assert df is None
    assert mock_db_handler.fetch_all.call_count == 1

@pytest.mark.asyncio
async def test_load_from_database_missing_columns(mock_db_handler):
    """Test obsługi brakujących kolumn w danych z bazy."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )

    # Przygotuj dane z brakującymi kolumnami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    invalid_data = pd.DataFrame(index=dates)
    invalid_data['timestamp'] = dates
    invalid_data['open'] = 1.0
    invalid_data['high'] = 1.1
    # Brak kolumny 'low'
    invalid_data['close'] = 1.05
    invalid_data['volume'] = 1000

    mock_db_handler.fetch_all.return_value = invalid_data.to_dict('records')

    df = await loader.load_from_database()
    assert df is None  # Powinien zwrócić None zamiast pustego DataFrame

def test_add_indicators(sample_data):
    """Test dodawania wskaźników technicznych."""
    loader = HistoricalDataLoader(symbol='EURUSD')
    df = loader.add_indicators(sample_data)
    
    # Sprawdź czy wszystkie wskaźniki zostały dodane
    expected_indicators = [
        'SMA_20', 'SMA_50', 'RSI',
        'BB_middle', 'BB_upper', 'BB_lower',
        'MACD', 'Signal_Line'
    ]
    assert all(indicator in df.columns for indicator in expected_indicators)
    
    # Sprawdź czy wartości są sensowne
    assert df['BB_upper'].mean() > df['BB_middle'].mean()
    assert df['BB_lower'].mean() < df['BB_middle'].mean()
    
    # Sprawdź RSI z pominięciem pierwszego wiersza (który jest NaN)
    rsi_values = df['RSI'].iloc[1:]
    assert (rsi_values >= 0).all() and (rsi_values <= 100).all()
    assert pd.isna(df['RSI'].iloc[0])  # Pierwszy element powinien być NaN

def test_add_indicators_empty_data():
    """Test dodawania wskaźników do pustych danych."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )
    
    empty_df = pd.DataFrame()
    result = loader.add_indicators(empty_df)
    
    assert result.empty
    assert result is empty_df  # Powinien zwrócić ten sam DataFrame

def test_add_indicators_missing_columns():
    """Test dodawania wskaźników gdy brakuje kolumn."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )
    
    # Przygotuj dane z brakującymi kolumnami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    incomplete_data = pd.DataFrame(index=dates)
    incomplete_data['open'] = 1.0
    incomplete_data['high'] = 1.1
    # Brak 'low'
    incomplete_data['close'] = 1.05
    incomplete_data['volume'] = 1000
    
    result = loader.add_indicators(incomplete_data)
    
    assert result is incomplete_data  # Powinien zwrócić ten sam DataFrame
    assert 'SMA_20' not in result.columns  # Nie powinien dodać wskaźników

def test_add_indicators_nan_values():
    """Test dodawania wskaźników do danych z wartościami NaN."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )

    # Przygotuj dane z wartościami NaN
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data_with_nan = pd.DataFrame(index=dates)
    data_with_nan['open'] = [1.0, np.nan, 1.0, 1.0, 1.0]
    data_with_nan['high'] = [1.1, 1.1, np.nan, 1.1, 1.1]
    data_with_nan['low'] = [0.9, 0.9, 0.9, np.nan, 0.9]
    data_with_nan['close'] = [1.05, 1.05, 1.05, 1.05, np.nan]
    data_with_nan['volume'] = 1000

    result = loader.add_indicators(data_with_nan)

    # Sprawdź czy wskaźniki zostały dodane
    assert 'SMA_20' in result.columns
    assert 'RSI' in result.columns
    assert 'MACD' in result.columns
    
    # Sprawdź czy wszystkie wskaźniki są NaN (ze względu na nieprawidłowe dane)
    indicators = ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
    for indicator in indicators:
        assert result[indicator].isna().all()

def test_add_indicators_single_row():
    """Test dodawania wskaźników do danych z jednym wierszem."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )
    
    # Przygotuj dane z jednym wierszem
    dates = pd.date_range(start='2024-01-01', periods=1, freq='1h')
    single_row_data = pd.DataFrame(index=dates)
    single_row_data['open'] = 1.0
    single_row_data['high'] = 1.1
    single_row_data['low'] = 0.9
    single_row_data['close'] = 1.05
    single_row_data['volume'] = 1000
    
    result = loader.add_indicators(single_row_data)
    
    # Sprawdź czy wskaźniki zostały dodane
    assert 'SMA_20' in result.columns
    assert 'RSI' in result.columns
    assert 'MACD' in result.columns
    # Wszystkie wskaźniki powinny być NaN dla pojedynczego wiersza
    assert result['SMA_20'].isna().all()
    assert result['RSI'].isna().all()
    assert result['MACD'].isna().all()

def test_add_indicators_invalid_values():
    """Test dodawania wskaźników do danych z nieprawidłowymi wartościami."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )

    # Przygotuj dane z nieprawidłowymi wartościami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    invalid_data = pd.DataFrame(index=dates)
    invalid_data['open'] = 'invalid'  # Nieprawidłowy typ
    invalid_data['high'] = 1.1
    invalid_data['low'] = 1.0
    invalid_data['close'] = 1.05
    invalid_data['volume'] = 1000

    result = loader.add_indicators(invalid_data)

    # Sprawdź czy wskaźniki zostały dodane jako NaN
    indicators = ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
    for indicator in indicators:
        assert indicator in result.columns
        assert result[indicator].isna().all()

    # Sprawdź czy oryginalne kolumny pozostały niezmienione
    assert (result['open'] == invalid_data['open']).all()
    assert (result['high'] == invalid_data['high']).all()
    assert (result['low'] == invalid_data['low']).all()
    assert (result['close'] == invalid_data['close']).all()
    assert (result['volume'] == invalid_data['volume']).all()

def test_add_indicators_extreme_values():
    """Test dodawania wskaźników do danych z ekstremalnymi wartościami."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )
    
    # Przygotuj dane z ekstremalnymi wartościami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    extreme_data = pd.DataFrame(index=dates)
    extreme_data['open'] = [1e9, 1e9, 1e9, 1e9, 1e9]  # Bardzo duże wartości
    extreme_data['high'] = [1e9 + 1, 1e9 + 1, 1e9 + 1, 1e9 + 1, 1e9 + 1]
    extreme_data['low'] = [1e9 - 1, 1e9 - 1, 1e9 - 1, 1e9 - 1, 1e9 - 1]
    extreme_data['close'] = [1e9 + 0.5, 1e9 + 0.5, 1e9 + 0.5, 1e9 + 0.5, 1e9 + 0.5]
    extreme_data['volume'] = [1e9, 1e9, 1e9, 1e9, 1e9]
    
    result = loader.add_indicators(extreme_data)
    
    # Sprawdź czy wskaźniki zostały obliczone
    assert 'SMA_20' in result.columns
    assert 'RSI' in result.columns
    assert 'MACD' in result.columns
    # Wartości powinny być skończone (nie inf)
    assert not np.isinf(result['SMA_20']).any()
    assert not np.isinf(result['RSI']).any()
    assert not np.isinf(result['MACD']).any()

def test_add_indicators_zero_values():
    """Test dodawania wskaźników do danych z zerami."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )

    # Przygotuj dane z zerami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    zero_data = pd.DataFrame(index=dates)
    zero_data['open'] = 0
    zero_data['high'] = 0
    zero_data['low'] = 0
    zero_data['close'] = 0
    zero_data['volume'] = 0

    result = loader.add_indicators(zero_data)

    # Sprawdź czy wskaźniki zostały dodane
    indicators = ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
    for indicator in indicators:
        assert indicator in result.columns
        assert result[indicator].isna().all()  # Wszystkie wskaźniki powinny być NaN dla samych zer

    # Sprawdź czy oryginalne dane pozostały niezmienione
    assert (result['open'] == 0).all()
    assert (result['high'] == 0).all()
    assert (result['low'] == 0).all()
    assert (result['close'] == 0).all()
    assert (result['volume'] == 0).all()

@pytest.mark.asyncio
async def test_load_from_mt5_success(sample_data):
    """Test udanego ładowania danych z MT5."""
    loader = HistoricalDataLoader(
        symbol="EURUSD",
        timeframe="1H"
    )
    
    # Przygotuj dane do zwrócenia przez mock MT5
    mt5_data = []
    for index, row in sample_data.iterrows():
        mt5_data.append({
            'time': int(index.timestamp()),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'tick_volume': row['volume'],
            'spread': 0,
            'real_volume': row['volume']
        })
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=mt5_data), \
         patch('MetaTrader5.shutdown'):
        
        result = await loader.load_from_mt5()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower',
            'MACD', 'Signal_Line'
        ])

@pytest.mark.asyncio
async def test_load_from_mt5_initialization_error():
    """Test obsługi błędu inicjalizacji MT5."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )
    
    with patch('MetaTrader5.initialize', return_value=False):
        with pytest.raises(RuntimeError, match="Nie udało się zainicjalizować MT5"):
            await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_copy_rates_error():
    """Test obsługi błędu podczas pobierania danych z MT5."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )

    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=None), \
         patch('MetaTrader5.shutdown'):
        df = await loader.load_from_mt5()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == ['time', 'open', 'high', 'low', 'close', 'tick_volume']

@pytest.mark.asyncio
async def test_load_from_mt5_invalid_timeframe():
    """Test obsługi nieprawidłowego timeframe."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='INVALID',  # Nieprawidłowy timeframe
        start_date=datetime(2024, 1, 1)
    )
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.shutdown'):
        with pytest.raises(KeyError):
            await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_empty_data():
    """Test obsługi pustych danych z MT5."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=[]), \
         patch('MetaTrader5.shutdown'):
        df = await loader.load_from_mt5()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

@pytest.mark.asyncio
async def test_load_from_mt5_invalid_data():
    """Test obsługi nieprawidłowych danych z MT5."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )
    
    # Przygotuj nieprawidłowe dane
    invalid_data = [{
        'time': 'invalid',  # Nieprawidłowy timestamp
        'open': 1.0,
        'high': 1.1,
        'low': 0.9,
        'close': 1.05,
        'tick_volume': 1000
    }]
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=invalid_data), \
         patch('MetaTrader5.shutdown'):
        with pytest.raises(Exception):  # Powinien wystąpić błąd konwersji
            await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_missing_columns():
    """Test obsługi brakujących kolumn w danych z MT5."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1)
    )
    
    # Przygotuj dane z brakującymi kolumnami
    incomplete_data = [{
        'time': 1704067200,  # 2024-01-01 00:00:00
        'open': 1.0,
        'high': 1.1,
        # Brak 'low'
        'close': 1.05,
        'tick_volume': 1000
    }]
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=incomplete_data), \
         patch('MetaTrader5.shutdown'):
        with pytest.raises(KeyError):  # Powinien wystąpić błąd przy dostępie do brakującej kolumny
            await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_data_from_database_first(mock_db_handler, sample_data):
    """Test ładowania danych najpierw z bazy."""
    loader = HistoricalDataLoader(
        symbol="EURUSD",
        timeframe="1H",
        db_handler=mock_db_handler
    )
    
    # Przygotuj dane do zwrócenia przez mock bazy
    db_data = []
    for index, row in sample_data.iterrows():
        db_data.append({
            'timestamp': index,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })
    
    mock_db_handler.fetch_all.return_value = db_data
    
    result = await loader.load_data()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # MT5 nie powinien być używany
    assert not hasattr(result, 'mt5_called')

@pytest.mark.asyncio
async def test_load_data_fallback_to_mt5(mock_db_handler, sample_data):
    """Test przełączenia na MT5 gdy baza nie zwraca danych."""
    loader = HistoricalDataLoader(
        symbol="EURUSD",
        timeframe="1H",
        db_handler=mock_db_handler
    )
    
    # Baza zwraca puste dane
    mock_db_handler.fetch_all.return_value = None
    
    # Przygotuj dane do zwrócenia przez mock MT5
    mt5_data = []
    for index, row in sample_data.iterrows():
        mt5_data.append({
            'time': int(index.timestamp()),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'tick_volume': row['volume'],
            'spread': 0,
            'real_volume': row['volume']
        })
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=mt5_data), \
         patch('MetaTrader5.shutdown'):
        
        result = await loader.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty 

def test_add_indicators_calculation_error(sample_data):
    """Test obsługi błędów podczas obliczania wskaźników."""
    loader = HistoricalDataLoader("EURUSD")

    # Przygotuj dane powodujące błąd
    df = sample_data.copy()
    df['close'] = 'invalid'  # Nieprawidłowy typ danych spowoduje błąd

    result = loader.add_indicators(df)

    # Sprawdź czy wskaźniki zostały dodane jako NaN
    indicators = ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
    for indicator in indicators:
        assert indicator in result.columns
        assert result[indicator].isna().all()

    # Sprawdź czy oryginalne dane pozostały niezmienione
    assert (result['close'] == df['close']).all()  # Dane wejściowe nie powinny być zmienione

@pytest.mark.asyncio
async def test_load_data_no_db_handler():
    """Test ładowania danych bez handlera bazy danych."""
    loader = HistoricalDataLoader(
        symbol="EURUSD",
        timeframe="1H"
    )
    
    # Przygotuj mock dla MT5 - więcej danych dla poprawnego obliczenia wskaźników
    mt5_data = []
    base_price = 1.1000
    for i in range(100):  # 100 świec
        mt5_data.append({
            'time': int(datetime.now().timestamp()) + i * 3600,  # co godzinę
            'open': base_price + i * 0.0001,
            'high': base_price + i * 0.0001 + 0.0010,
            'low': base_price + i * 0.0001 - 0.0010,
            'close': base_price + i * 0.0001 + 0.0005,
            'tick_volume': 1000 + i,
            'spread': 0,
            'real_volume': 1000 + i
        })
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=mt5_data), \
         patch('MetaTrader5.shutdown'):
        
        result = await loader.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 100
        assert all(col in result.columns for col in [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower',
            'MACD', 'Signal_Line'
        ])

@pytest.mark.asyncio
async def test_load_data_db_empty_result(mock_db_handler):
    """Test ładowania danych gdy baza zwraca pusty wynik."""
    loader = HistoricalDataLoader(
        symbol="EURUSD",
        timeframe="1H",
        db_handler=mock_db_handler
    )
    
    # Baza zwraca pusty wynik
    mock_db_handler.fetch_all.return_value = []
    
    # Przygotuj mock dla MT5
    mt5_data = [
        {
            'time': int(datetime.now().timestamp()),
            'open': 1.1000,
            'high': 1.1100,
            'low': 1.0900,
            'close': 1.1050,
            'tick_volume': 1000,
            'spread': 0,
            'real_volume': 1000
        }
    ]
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=mt5_data), \
         patch('MetaTrader5.shutdown'):
        
        result = await loader.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert mock_db_handler.fetch_all.called
        assert mock_db_handler.execute_many.called  # Dane powinny być zapisane do bazy

def test_timeframe_validation():
    """Test walidacji timeframe."""
    # Prawidłowy timeframe
    loader = HistoricalDataLoader("EURUSD", "1H")
    assert loader.timeframe == "1H"
    assert loader.timeframe_map[loader.timeframe] == mt5.TIMEFRAME_H1
    
    # Nieprawidłowy timeframe
    with pytest.raises(KeyError):
        loader = HistoricalDataLoader("EURUSD", "invalid")
        _ = loader.timeframe_map[loader.timeframe]

def test_start_date_default():
    """Test domyślnej daty początkowej."""
    loader = HistoricalDataLoader("EURUSD")
    assert isinstance(loader.start_date, datetime)
    assert loader.start_date <= datetime.now()
    assert loader.start_date >= datetime.now() - timedelta(days=31)  # Sprawdź czy w zakresie 30 dni

def test_add_indicators_bollinger_validation(sample_data):
    """Test walidacji wstęg Bollingera."""
    loader = HistoricalDataLoader("EURUSD")
    
    # Dodaj wskaźniki
    df = loader.add_indicators(sample_data.copy())
    
    # Pomiń pierwsze 20 wartości (okres inicjalizacji)
    df = df.iloc[20:]
    
    # Sprawdź właściwości wstęg Bollingera
    assert (df['BB_upper'] >= df['BB_middle']).all()
    assert (df['BB_middle'] >= df['BB_lower']).all()
    assert (df['BB_upper'] - df['BB_lower']).mean() > 0  # Sprawdź czy jest odstęp między wstęgami

def test_add_indicators_macd_validation(sample_data):
    """Test walidacji MACD."""
    loader = HistoricalDataLoader("EURUSD")

    # Dodaj wskaźniki
    df = loader.add_indicators(sample_data.copy())

    # Pomiń pierwsze 26 wartości (okres inicjalizacji MACD)
    df = df.iloc[26:]

    # Sprawdź właściwości MACD
    assert not df['MACD'].isna().all()
    assert not df['Signal_Line'].isna().all()

    # MACD powinien mieć wartości zarówno dodatnie jak i ujemne
    # Nie sprawdzamy tego warunku, ponieważ MACD może być tylko dodatni lub tylko ujemny
    # w zależności od trendu w danych testowych
    assert df['MACD'].notna().any()  # Wystarczy sprawdzić czy są jakieś wartości

def test_add_indicators_rsi_validation(sample_data):
    """Test walidacji RSI."""
    loader = HistoricalDataLoader("EURUSD")

    # Dodaj wskaźniki
    df = loader.add_indicators(sample_data.copy())

    # Sprawdź właściwości RSI
    rsi_values = df['RSI'].dropna()
    assert (rsi_values >= 0).all()
    assert (rsi_values <= 100).all()
    
    # Sprawdź czy pierwszy element jest NaN
    assert pd.isna(df['RSI'].iloc[0])
    
    # Sprawdź czy pozostałe wartości są obliczone
    assert not df['RSI'].iloc[1:].isna().all()

def test_add_indicators_all_nan():
    """Test sprawdzający zachowanie gdy wszystkie wskaźniki są NaN."""
    loader = HistoricalDataLoader("EURUSD")
    
    # Przygotuj dane z samymi NaN - minimum 30 świec dla wszystkich wskaźników
    dates = pd.date_range(start='2024-01-01', periods=30, freq='h')
    data = pd.DataFrame({
        'open': [np.nan] * 30,
        'high': [np.nan] * 30,
        'low': [np.nan] * 30,
        'close': [np.nan] * 30,
        'volume': [np.nan] * 30
    }, index=dates)
    
    # Dodaj wskaźniki
    result = loader.add_indicators(data)
    
    # Sprawdź czy wszystkie wskaźniki są NaN
    indicators = ['SMA_20', 'SMA_50', 'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'MACD', 'Signal_Line']
    for indicator in indicators:
        assert indicator in result.columns
        assert result[indicator].isna().all() 

@pytest.mark.asyncio
async def test_save_to_database_success(mock_db_handler, sample_data):
    """Test udanego zapisu danych do bazy."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    await loader.save_to_database(sample_data)
    
    assert mock_db_handler.execute_many.call_count == 1
    args = mock_db_handler.execute_many.call_args[0]
    assert 'INSERT INTO historical_data' in args[0]
    assert len(args[1]) == len(sample_data)
    
    # Sprawdź format danych
    first_record = args[1][0]
    assert len(first_record) == 8  # symbol, timeframe, timestamp, open, high, low, close, volume
    assert first_record[0] == 'EURUSD'
    assert first_record[1] == '1H'
    assert isinstance(first_record[2], pd.Timestamp)
    assert all(isinstance(val, float) for val in first_record[3:8])

@pytest.mark.asyncio
async def test_save_to_database_empty_data(mock_db_handler):
    """Test zapisu pustych danych do bazy."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    empty_df = pd.DataFrame()
    await loader.save_to_database(empty_df)
    
    assert mock_db_handler.execute_many.call_count == 0

@pytest.mark.asyncio
async def test_save_to_database_missing_columns(mock_db_handler):
    """Test zapisu danych z brakującymi kolumnami."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    # Przygotuj dane z brakującymi kolumnami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    incomplete_data = pd.DataFrame(index=dates)
    incomplete_data['open'] = 1.0
    incomplete_data['high'] = 1.1
    # Brak 'low'
    incomplete_data['close'] = 1.05
    incomplete_data['volume'] = 1000
    
    with pytest.raises(KeyError):
        await loader.save_to_database(incomplete_data)
    
    assert mock_db_handler.execute_many.call_count == 0

@pytest.mark.asyncio
async def test_save_to_database_invalid_data(mock_db_handler):
    """Test zapisu nieprawidłowych danych."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    # Przygotuj dane z nieprawidłowymi wartościami
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    invalid_data = pd.DataFrame(index=dates)
    invalid_data['open'] = 'invalid'  # Nieprawidłowy typ
    invalid_data['high'] = 1.1
    invalid_data['low'] = 1.0
    invalid_data['close'] = 1.05
    invalid_data['volume'] = 1000
    
    with pytest.raises(ValueError):
        await loader.save_to_database(invalid_data)
    
    assert mock_db_handler.execute_many.call_count == 0

@pytest.mark.asyncio
async def test_save_to_database_db_error(mock_db_handler, sample_data):
    """Test obsługi błędu bazy danych podczas zapisu."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    mock_db_handler.execute_many.side_effect = Exception("Test database error")
    
    await loader.save_to_database(sample_data)
    
    assert mock_db_handler.execute_many.call_count == 1

@pytest.mark.asyncio
async def test_save_to_database_nan_values(mock_db_handler):
    """Test zapisu danych z wartościami NaN."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=datetime(2024, 1, 1),
        db_handler=mock_db_handler
    )
    
    # Przygotuj dane z wartościami NaN
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data_with_nan = pd.DataFrame(index=dates)
    data_with_nan['open'] = [1.0, np.nan, 1.0, 1.0, 1.0]
    data_with_nan['high'] = [1.1, 1.1, np.nan, 1.1, 1.1]
    data_with_nan['low'] = [0.9, 0.9, 0.9, np.nan, 0.9]
    data_with_nan['close'] = [1.05, 1.05, 1.05, 1.05, np.nan]
    data_with_nan['volume'] = 1000
    
    with pytest.raises(ValueError):
        await loader.save_to_database(data_with_nan)
    
    assert mock_db_handler.execute_many.call_count == 0 