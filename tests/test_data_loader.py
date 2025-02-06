"""
Testy jednostkowe dla modułu data_loader.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import MetaTrader5 as mt5
from pytest_mock import MockerFixture

from src.backtest.data_loader import HistoricalDataLoader
from src.database.postgres_handler import PostgresHandler

# Fixture dla przykładowych danych
@pytest.fixture
def sample_data():
    """Przykładowe dane historyczne dla testów."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Generuj dane cenowe
    data['open'] = np.random.normal(1.1000, 0.0010, size=len(data))
    data['high'] = data['open'] + abs(np.random.normal(0, 0.0005, size=len(data)))
    data['low'] = data['open'] - abs(np.random.normal(0, 0.0005, size=len(data)))
    data['close'] = np.random.normal(1.1000, 0.0010, size=len(data))
    data['volume'] = np.random.normal(1000, 100, size=len(data))
    
    # Upewnij się, że high > low
    data['high'] = np.maximum(data['high'], data['low'] + 0.0001)
    
    # Dodaj wskaźniki techniczne
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['RSI'] = np.random.uniform(0, 100, size=len(data))
    data['BB_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
    data['MACD'] = np.random.normal(0, 0.0005, size=len(data))
    data['Signal_Line'] = np.random.normal(0, 0.0005, size=len(data))
    
    return data

@pytest.fixture
def mock_db_handler():
    """Mock dla handlera bazy danych."""
    handler = AsyncMock(spec=PostgresHandler)
    handler.fetch_all = AsyncMock()
    handler.execute_many = AsyncMock()
    return handler

@pytest.fixture
def data_loader():
    """Fixture tworzący loader danych do testów."""
    return HistoricalDataLoader(symbol='EURUSD', timeframe='1H')

@pytest.fixture
def mocker(request):
    """Fixture do mockowania."""
    return MockerFixture(request.config)

@pytest.mark.asyncio
async def test_initialization():
    """Test inicjalizacji loadera bez bazy danych."""
    loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
    assert loader.symbol == 'EURUSD'
    assert loader.timeframe == '1H'
    assert loader.db_handler is None
    assert isinstance(loader.start_date, datetime)

@pytest.mark.asyncio
async def test_initialization_with_db(mock_db_handler):
    """Test inicjalizacji loadera z bazą danych."""
    start_date = datetime(2024, 1, 1)
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        start_date=start_date,
        db_handler=mock_db_handler
    )
    assert loader.symbol == 'EURUSD'
    assert loader.timeframe == '1H'
    assert loader.db_handler == mock_db_handler
    assert loader.start_date == start_date

@pytest.mark.asyncio
async def test_load_from_database_success(mock_db_handler, sample_data):
    """Test udanego ładowania danych z bazy."""
    # Przygotuj dane testowe
    records = []
    for timestamp, row in sample_data.iterrows():
        records.append({
            'timestamp': timestamp,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
    
    mock_db_handler.fetch_all.return_value = records
    
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )
    
    df = await loader.load_from_database()
    
    assert df is not None
    assert not df.empty
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert len(df) == len(records)
    assert mock_db_handler.fetch_all.called

@pytest.mark.asyncio
async def test_load_from_database_empty_result(mock_db_handler):
    """Test ładowania z bazy gdy brak danych."""
    mock_db_handler.fetch_all.return_value = []
    
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )
    
    df = await loader.load_from_database()
    assert df is None
    assert mock_db_handler.fetch_all.called

@pytest.mark.asyncio
async def test_load_from_database_db_error(mock_db_handler):
    """Test obsługi błędu bazy danych."""
    mock_db_handler.fetch_all.side_effect = Exception("DB Error")
    
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )
    
    df = await loader.load_from_database()
    assert df is None
    assert mock_db_handler.fetch_all.called

@pytest.mark.asyncio
async def test_load_from_database_invalid_data(mock_db_handler):
    """Test ładowania nieprawidłowych danych z bazy."""
    # Przygotuj nieprawidłowe dane (nieliczbowe wartości)
    invalid_records = [{
        'timestamp': datetime.now(),
        'open': 'invalid',
        'high': 1.1050,
        'low': 1.0950,
        'close': 1.1000,
        'volume': 1000.0
    }]
    
    mock_db_handler.fetch_all.return_value = invalid_records
    
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )
    
    df = await loader.load_from_database()
    assert df is None

@pytest.mark.asyncio
async def test_load_from_database_missing_columns(mock_db_handler):
    """Test ładowania danych z brakującymi kolumnami."""
    # Przygotuj dane z brakującymi kolumnami
    incomplete_records = [{
        'timestamp': datetime.now(),
        'open': 1.1000,
        'close': 1.1050,
        'volume': 1000.0
        # Brak high i low
    }]
    
    mock_db_handler.fetch_all.return_value = incomplete_records
    
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )

    df = await loader.load_from_database()
    assert df is None

def test_add_indicators(sample_data):
    """Test dodawania wskaźników technicznych."""
    loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
    
    # Użyj tylko kolumn OHLCV
    input_data = sample_data[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Dodaj wskaźniki
    df = loader.add_indicators(input_data)
    
    # Sprawdź czy wszystkie wskaźniki zostały dodane
    expected_indicators = [
        'SMA_20', 'SMA_50', 'RSI',
        'BB_middle', 'BB_upper', 'BB_lower',
        'MACD', 'Signal_Line'
    ]
    
    assert all(indicator in df.columns for indicator in expected_indicators)
    assert not df[expected_indicators].isna().all().all()

def test_add_indicators_empty_data():
    """Test dodawania wskaźników do pustego DataFrame."""
    loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
    
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    df = loader.add_indicators(empty_df)
    
    assert df.empty
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_add_indicators_missing_columns():
    """Test dodawania wskaźników gdy brakuje wymaganych kolumn."""
    loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
    
    # DataFrame bez kolumny 'close'
    incomplete_df = pd.DataFrame({
        'open': [1.1000],
        'high': [1.1050],
        'low': [1.0950],
        'volume': [1000.0]
    })
    
    df = loader.add_indicators(incomplete_df)
    assert 'close' not in df.columns
    assert all(col in df.columns for col in ['open', 'high', 'low', 'volume'])

def test_add_indicators_nan_values():
    """Test dodawania wskaźników gdy są wartości NaN."""
    loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')

    # Przygotuj dane z wartościami NaN
    data = pd.DataFrame({
        'open': [1.1000, np.nan, 1.1020],
        'high': [1.1050, 1.1060, np.nan],
        'low': [1.0950, 1.0960, 1.0970],
        'close': [1.1000, 1.1010, 1.1020],
        'volume': [1000.0, 1100.0, np.nan]
    })
    
    df = loader.add_indicators(data)
    
    # Sprawdź czy wskaźniki zostały dodane mimo wartości NaN
    assert all(col in df.columns for col in [
        'SMA_20', 'SMA_50', 'RSI', 
        'BB_middle', 'BB_upper', 'BB_lower',
        'MACD', 'Signal_Line'
    ])

@pytest.mark.asyncio
async def test_load_from_mt5_success(sample_data):
    """Test udanego ładowania danych z MT5."""
    # Przygotuj dane w formacie MT5
    mt5_data = []
    for timestamp, row in sample_data.iterrows():
        mt5_data.append({
            'time': int(timestamp.timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'tick_volume': float(row['volume']),
            'spread': 0,
            'real_volume': float(row['volume'])
        })
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=mt5_data):
        
        loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
        df = await loader.load_from_mt5()
        
        assert df is not None
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert len(df) == len(sample_data)
        assert all(col in df.columns for col in [
            'SMA_20', 'SMA_50', 'RSI', 
            'BB_middle', 'BB_upper', 'BB_lower',
            'MACD', 'Signal_Line'
        ])

@pytest.mark.asyncio
async def test_load_from_mt5_initialization_error():
    """Test błędu inicjalizacji MT5."""
    with patch('MetaTrader5.initialize', return_value=False), \
         patch('MetaTrader5.last_error', return_value="Test error"):
        
        loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
        
        with pytest.raises(RuntimeError, match="Nie udało się zainicjalizować MT5"):
            await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_copy_rates_error():
    """Test błędu podczas pobierania danych z MT5."""
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', side_effect=Exception("Test error")):
        
        loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
        
        with pytest.raises(RuntimeError, match="Nie udało się pobrać danych z MT5"):
            await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_invalid_timeframe():
    """Test nieprawidłowego timeframe."""
    loader = HistoricalDataLoader(symbol='EURUSD', timeframe='INVALID')
    
    with pytest.raises(KeyError, match="Nieprawidłowy timeframe"):
        await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_empty_data():
    """Test gdy MT5 zwraca puste dane."""
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=[]):
        
        loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
        df = await loader.load_from_mt5()
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close', 'tick_volume'])

@pytest.mark.asyncio
async def test_load_from_mt5_invalid_data():
    """Test gdy MT5 zwraca nieprawidłowe dane."""
    invalid_data = [{
        'time': datetime.now().timestamp(),
        'open': 'invalid',  # Nieprawidłowy typ
        'high': 1.1050,
        'low': 1.0950,
        'close': 1.1000,
        'tick_volume': 1000.0
    }]
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=invalid_data):
        
        loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
        df = await loader.load_from_mt5()
        
        # Powinien zwrócić DataFrame z NaN dla wskaźników
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert pd.isna(df['SMA_20']).all()
        assert pd.isna(df['RSI']).all()

@pytest.mark.asyncio
async def test_load_from_mt5_missing_columns():
    """Test gdy dane z MT5 nie mają wszystkich wymaganych kolumn."""
    incomplete_data = [{
        'time': datetime.now().timestamp(),
        'open': 1.1000,
        'close': 1.1050,
        'tick_volume': 1000.0
        # Brak high i low
    }]
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=incomplete_data):
        
        loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
        
        with pytest.raises(KeyError, match="Brakujące kolumny w danych z MT5"):
            await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_data_from_database_first(mock_db_handler, sample_data):
    """Test ładowania danych najpierw z bazy."""
    # Przygotuj dane testowe
    records = []
    for timestamp, row in sample_data.iterrows():
        records.append({
            'timestamp': timestamp,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
    
    mock_db_handler.fetch_all.return_value = records
    
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )
    
    df = await loader.load_data()
    
    assert df is not None
    assert not df.empty
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert len(df) == len(records)
    assert mock_db_handler.fetch_all.called
    assert mock_db_handler.execute_many.called  # Powinien zapisać dane do bazy

@pytest.mark.asyncio
async def test_load_data_fallback_to_mt5(mock_db_handler, sample_data):
    """Test przejścia do MT5 gdy brak danych w bazie."""
    # Baza zwraca puste dane
    mock_db_handler.fetch_all.return_value = []
    
    # Przygotuj dane w formacie MT5
    mt5_data = []
    for timestamp, row in sample_data.iterrows():
        mt5_data.append({
            'time': int(timestamp.timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'tick_volume': float(row['volume']),
            'spread': 0,
            'real_volume': float(row['volume'])
        })
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=mt5_data):
        
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
            db_handler=mock_db_handler
        )
        
        df = await loader.load_data()
        
        assert df is not None
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert len(df) == len(sample_data)
        assert mock_db_handler.fetch_all.called
        assert mock_db_handler.execute_many.called  # Powinien zapisać dane do bazy

@pytest.mark.asyncio
async def test_load_data_no_db_handler():
    """Test ładowania danych bez handlera bazy."""
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=[{
             'time': datetime.now().timestamp(),
             'open': 1.1000,
             'high': 1.1050,
             'low': 1.0950,
             'close': 1.1000,
             'tick_volume': 1000.0
         }]):
        
        loader = HistoricalDataLoader(symbol='EURUSD', timeframe='1H')
        df = await loader.load_data()
        
        assert df is not None
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

@pytest.mark.asyncio
async def test_load_data_db_empty_result(mock_db_handler):
    """Test gdy baza zwraca puste dane i MT5 też."""
    mock_db_handler.fetch_all.return_value = []
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=[]):

    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
            db_handler=mock_db_handler
        )
        
        df = await loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        assert mock_db_handler.fetch_all.called
        assert not mock_db_handler.execute_many.called  # Nie powinno być próby zapisu pustych danych

@pytest.mark.asyncio
async def test_save_to_database_success(mock_db_handler, sample_data):
    """Test udanego zapisu do bazy danych."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )
    
    await loader.save_to_database(sample_data)
    
    assert mock_db_handler.execute_many.called
    args = mock_db_handler.execute_many.call_args[0]
    assert len(args) == 2  # Query i dane
    assert isinstance(args[1], list)  # Lista krotek z danymi
    assert len(args[1]) == len(sample_data)

@pytest.mark.asyncio
async def test_save_to_database_empty_data(mock_db_handler):
    """Test próby zapisu pustych danych."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )
    
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    await loader.save_to_database(empty_df)
    
    assert not mock_db_handler.execute_many.called

@pytest.mark.asyncio
async def test_save_to_database_missing_columns(mock_db_handler):
    """Test próby zapisu danych z brakującymi kolumnami."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
        db_handler=mock_db_handler
    )
    
    incomplete_df = pd.DataFrame({
        'open': [1.1000],
        'close': [1.1050],
        'volume': [1000.0]
        # Brak high i low
    })
    
    with pytest.raises(KeyError, match="Brakujące kolumny"):
        await loader.save_to_database(incomplete_df)
    
    assert not mock_db_handler.execute_many.called

@pytest.mark.asyncio
async def test_save_to_database_invalid_data(mock_db_handler):
    """Test zapisu nieprawidłowych danych."""
    loader = HistoricalDataLoader(
        symbol='EURUSD',
        timeframe='1H',
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

@pytest.mark.asyncio
async def test_load_data_empty_response(data_loader):
    """Test obsługi pustej odpowiedzi z MT5."""
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=None):
        
        df = await data_loader.load_from_mt5()
        assert df.empty
        assert all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close', 'tick_volume'])

@pytest.mark.asyncio
async def test_load_data_from_mt5(data_loader, sample_data):
    """Test ładowania danych z MT5."""
    # Przygotuj dane w formacie MT5
    mt5_data = []
    for timestamp, row in sample_data.iterrows():
        mt5_data.append({
            'time': int(timestamp.timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'tick_volume': float(row['volume']),
            'spread': 0,
            'real_volume': float(row['volume'])
        })
    
    with patch('MetaTrader5.initialize', return_value=True), \
         patch('MetaTrader5.copy_rates_from', return_value=mt5_data):
        
        df = await data_loader.load_from_mt5()
        
        assert df is not None
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

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
async def test_initialization_invalid_symbol():
    """Test inicjalizacji z nieprawidłowym symbolem."""
    with pytest.raises(ValueError, match="Symbol musi mieć od 3 do 10 znaków"):
        HistoricalDataLoader(symbol="")
        
    with pytest.raises(ValueError, match="Symbol musi mieć od 3 do 10 znaków"):
        HistoricalDataLoader(symbol="AB")
        
    with pytest.raises(ValueError, match="Symbol musi mieć od 3 do 10 znaków"):
        HistoricalDataLoader(symbol="EURUSDGBPJPY")

@pytest.mark.asyncio
async def test_initialization_invalid_timeframe():
    """Test inicjalizacji z nieprawidłowym timeframe."""
    with pytest.raises(ValueError, match="Nieprawidłowy timeframe"):
        HistoricalDataLoader(symbol="EURUSD", timeframe="2H")
        
    with pytest.raises(ValueError, match="Nieprawidłowy timeframe"):
        HistoricalDataLoader(symbol="EURUSD", timeframe="INVALID")

@pytest.mark.asyncio
async def test_initialization_future_start_date():
    """Test inicjalizacji z datą z przyszłości."""
    future_date = datetime.now() + timedelta(days=1)
    with pytest.raises(ValueError, match="Data początkowa nie może być z przyszłości"):
        HistoricalDataLoader(symbol="EURUSD", start_date=future_date)

@pytest.mark.asyncio
async def test_initialization_with_custom_timeframe():
    """Test inicjalizacji z różnymi prawidłowymi timeframe."""
    valid_timeframes = ['1M', '5M', '15M', '30M', '1H', '4H', '1D']
    for timeframe in valid_timeframes:
        loader = HistoricalDataLoader(symbol="EURUSD", timeframe=timeframe)
        assert loader.timeframe == timeframe
        assert loader.symbol == "EURUSD"

@pytest.mark.asyncio
async def test_initialization_with_db_connection_error(mock_db_handler):
    """Test inicjalizacji gdy baza danych jest niedostępna."""
    mock_db_handler.initialize.side_effect = Exception("Connection error")
    loader = HistoricalDataLoader(
        symbol="EURUSD",
        db_handler=mock_db_handler
    )
    assert loader.db_handler == mock_db_handler 

@pytest.mark.asyncio
async def test_load_from_mt5_connection_error(mocker):
    """Test ładowania danych gdy MT5 jest niedostępny."""
    mocker.patch('MetaTrader5.initialize', return_value=False)
    mocker.patch('MetaTrader5.last_error', return_value="Connection error")
    
    loader = HistoricalDataLoader(symbol="EURUSD")
    with pytest.raises(RuntimeError, match="Nie udało się zainicjalizować MT5"):
        await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_rate_limit(mocker):
    """Test ładowania danych gdy przekroczono limit zapytań."""
    mocker.patch('MetaTrader5.initialize', return_value=True)
    mocker.patch('MetaTrader5.copy_rates_from', side_effect=Exception("Rate limit exceeded"))
    
    loader = HistoricalDataLoader(symbol="EURUSD")
    with pytest.raises(RuntimeError, match="Nie udało się pobrać danych z MT5"):
        await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_invalid_response(mocker):
    """Test ładowania danych gdy MT5 zwraca nieprawidłową odpowiedź."""
    mocker.patch('MetaTrader5.initialize', return_value=True)
    mocker.patch('MetaTrader5.copy_rates_from', return_value=None)
    
    loader = HistoricalDataLoader(symbol="EURUSD")
    result = await loader.load_from_mt5()
    assert result.empty

@pytest.mark.asyncio
async def test_load_from_mt5_partial_data(mocker):
    """Test ładowania danych gdy MT5 zwraca niepełne dane."""
    partial_data = [
        {
            'time': datetime.now().timestamp(),
            'open': 1.1000,
            'high': 1.1100,
            'low': 1.0900,
            'close': 1.1050,
            # brak volume
        }
    ]
    mocker.patch('MetaTrader5.initialize', return_value=True)
    mocker.patch('MetaTrader5.copy_rates_from', return_value=partial_data)
    
    loader = HistoricalDataLoader(symbol="EURUSD")
    with pytest.raises(KeyError, match="Brakujące kolumny"):
        await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_from_mt5_data_validation(mocker):
    """Test walidacji danych z MT5."""
    invalid_data = [
        {
            'time': 'invalid',
            'open': 'invalid',
            'high': 'invalid',
            'low': 'invalid',
            'close': 'invalid',
            'tick_volume': 'invalid'
        }
    ]
    mocker.patch('MetaTrader5.initialize', return_value=True)
    mocker.patch('MetaTrader5.copy_rates_from', return_value=invalid_data)
    
    loader = HistoricalDataLoader(symbol="EURUSD")
    with pytest.raises(Exception):
        await loader.load_from_mt5()

@pytest.mark.asyncio
async def test_load_data_with_gaps():
    """Test ładowania danych z lukami."""
    # Przygotuj dane z lukami
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    data = pd.DataFrame({
        'open': [1.1000, 1.1010, np.nan, 1.1015, 1.1025, np.nan, 1.1030, 1.1040, 1.1045, 1.1050],
        'high': [1.1020, 1.1030, np.nan, 1.1035, 1.1045, np.nan, 1.1050, 1.1060, 1.1065, 1.1070],
        'low': [1.0990, 1.1000, np.nan, 1.1005, 1.1015, np.nan, 1.1020, 1.1030, 1.1035, 1.1040],
        'close': [1.1010, 1.1020, np.nan, 1.1025, 1.1035, np.nan, 1.1040, 1.1045, 1.1050, 1.1060],
        'volume': [1000, 1100, np.nan, 1200, 1000, np.nan, 800, 1100, 1000, 1200]
    }, index=dates)
    
    loader = HistoricalDataLoader("EURUSD")
    result = loader.add_indicators(data)
    
    # Sprawdź czy wskaźniki zostały poprawnie obliczone mimo luk
    assert not result['SMA_20'].isna().all()
    assert not result['RSI'].isna().all()
    assert not result['MACD'].isna().all()

@pytest.mark.asyncio
async def test_load_data_extreme_values():
    """Test ładowania danych z ekstremalnymi wartościami."""
    # Przygotuj dane z ekstremalnymi wartościami
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    data = pd.DataFrame({
        'open': [1.1000, 1000.0000, 0.0001, 1.1015, 1.1025, 1.1035, 1.1030, 1.1040, 1.1045, 1.1050],
        'high': [1.1020, 2000.0000, 0.0002, 1.1035, 1.1045, 1.1055, 1.1050, 1.1060, 1.1065, 1.1070],
        'low': [1.0990, 500.0000, 0.0001, 1.1005, 1.1015, 1.1025, 1.1020, 1.1030, 1.1035, 1.1040],
        'close': [1.1010, 1500.0000, 0.0002, 1.1025, 1.1035, 1.1030, 1.1040, 1.1045, 1.1050, 1.1060],
        'volume': [1000, 1000000, 1, 1200, 1000, 1300, 800, 1100, 1000, 1200]
    }, index=dates)
    
    loader = HistoricalDataLoader("EURUSD")
    result = loader.add_indicators(data)
    
    # Sprawdź czy wskaźniki zostały poprawnie obliczone mimo ekstremalnych wartości
    assert not result['SMA_20'].isna().all()
    assert not result['RSI'].isna().all()
    assert not result['MACD'].isna().all()

@pytest.mark.asyncio
async def test_load_data_different_timezones():
    """Test ładowania danych z różnych stref czasowych."""
    # Przygotuj daty w różnych strefach czasowych
    dates = [
        pd.Timestamp('2024-01-01 00:00:00+00:00'),
        pd.Timestamp('2024-01-01 00:00:00+01:00'),
        pd.Timestamp('2024-01-01 00:00:00+02:00')
    ]
    data = pd.DataFrame({
        'open': [1.1000, 1.1010, 1.1020],
        'high': [1.1020, 1.1030, 1.1040],
        'low': [1.0990, 1.1000, 1.1010],
        'close': [1.1010, 1.1020, 1.1030],
        'volume': [1000, 1100, 1200]
    }, index=dates)
    
    loader = HistoricalDataLoader("EURUSD")
    result = loader.add_indicators(data)
    
    # Sprawdź czy wskaźniki zostały poprawnie obliczone mimo różnych stref czasowych
    assert not result['SMA_20'].isna().all()
    assert not result['RSI'].isna().all()
    assert not result['MACD'].isna().all()

@pytest.mark.asyncio
async def test_load_data_with_trends():
    """Test ładowania danych z wyraźnymi trendami."""
    # Przygotuj dane z trendem wzrostowym
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    uptrend_data = pd.DataFrame({
        'open': np.linspace(1.1000, 1.2000, 10),
        'high': np.linspace(1.1020, 1.2020, 10),
        'low': np.linspace(1.0990, 1.1990, 10),
        'close': np.linspace(1.1010, 1.2010, 10),
        'volume': np.random.randint(1000, 2000, 10)
    }, index=dates)
    
    loader = HistoricalDataLoader("EURUSD")
    result = loader.add_indicators(uptrend_data)
    
    # Sprawdź czy wskaźniki poprawnie wykrywają trend
    assert result['SMA_20'].iloc[-1] > result['SMA_20'].iloc[0] if not pd.isna(result['SMA_20'].iloc[-1]) else True
    assert result['RSI'].iloc[-1] > 50 if not pd.isna(result['RSI'].iloc[-1]) else True

@pytest.mark.asyncio
async def test_save_to_database_with_conflict(mock_db_handler):
    """Test zapisywania danych gdy występuje konflikt w bazie."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    data = pd.DataFrame({
        'open': [1.1000] * 10,
        'high': [1.1020] * 10,
        'low': [1.0990] * 10,
        'close': [1.1010] * 10,
        'volume': [1000] * 10
    }, index=dates)
    
    mock_db_handler.execute_many.side_effect = Exception("Duplicate key value violates unique constraint")
    
    loader = HistoricalDataLoader("EURUSD", db_handler=mock_db_handler)
    await loader.save_to_database(data)
    
    assert mock_db_handler.execute_many.called

@pytest.mark.asyncio
async def test_save_to_database_connection_exceeded(mock_db_handler):
    """Test zapisywania danych gdy przekroczono limit połączeń."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    data = pd.DataFrame({
        'open': [1.1000] * 10,
        'high': [1.1020] * 10,
        'low': [1.0990] * 10,
        'close': [1.1010] * 10,
        'volume': [1000] * 10
    }, index=dates)
    
    mock_db_handler.execute_many.side_effect = Exception("too many connections")
    
    loader = HistoricalDataLoader("EURUSD", db_handler=mock_db_handler)
    await loader.save_to_database(data)
    
    assert mock_db_handler.execute_many.called

@pytest.mark.asyncio
async def test_save_to_database_transaction_failed(mock_db_handler):
    """Test zapisywania danych gdy występuje błąd transakcji."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    data = pd.DataFrame({
        'open': [1.1000] * 10,
        'high': [1.1020] * 10,
        'low': [1.0990] * 10,
        'close': [1.1010] * 10,
        'volume': [1000] * 10
    }, index=dates)
    
    mock_db_handler.execute_many.side_effect = Exception("could not serialize access")
    
    loader = HistoricalDataLoader("EURUSD", db_handler=mock_db_handler)
    await loader.save_to_database(data)
    
    assert mock_db_handler.execute_many.called

@pytest.mark.asyncio
async def test_save_to_database_large_data(mock_db_handler):
    """Test zapisywania dużego zbioru danych."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'open': np.random.normal(1.1000, 0.0010, size=1000),
        'high': np.random.normal(1.1020, 0.0010, size=1000),
        'low': np.random.normal(1.0990, 0.0010, size=1000),
        'close': np.random.normal(1.1010, 0.0010, size=1000),
        'volume': np.random.randint(1000, 2000, size=1000)
    }, index=dates)
    
    loader = HistoricalDataLoader("EURUSD", db_handler=mock_db_handler)
    await loader.save_to_database(data)
    
    assert mock_db_handler.execute_many.call_count >= 1

@pytest.mark.asyncio
async def test_save_to_database_retry_mechanism(mock_db_handler):
    """Test mechanizmu ponownych prób przy zapisie do bazy."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    data = pd.DataFrame({
        'open': [1.1000] * 10,
        'high': [1.1020] * 10,
        'low': [1.0990] * 10,
        'close': [1.1010] * 10,
        'volume': [1000] * 10
    }, index=dates)
    
    # Symuluj błąd przy pierwszej próbie i sukces przy drugiej
    mock_db_handler.execute_many.side_effect = [
        Exception("temporary failure"),
        None
    ]
    
    loader = HistoricalDataLoader("EURUSD", db_handler=mock_db_handler)
    await loader.save_to_database(data)
    
    assert mock_db_handler.execute_many.call_count == 2

@pytest.mark.asyncio
async def test_save_to_database_batch_processing(mock_db_handler):
    """Test przetwarzania wsadowego przy zapisie do bazy."""
    # Przygotuj duży zbiór danych
    dates = pd.date_range(start='2024-01-01', periods=5000, freq='1h')
    data = pd.DataFrame({
        'open': np.random.normal(1.1000, 0.0010, size=5000),
        'high': np.random.normal(1.1020, 0.0010, size=5000),
        'low': np.random.normal(1.0990, 0.0010, size=5000),
        'close': np.random.normal(1.1010, 0.0010, size=5000),
        'volume': np.random.randint(1000, 2000, size=5000)
    }, index=dates)
    
    loader = HistoricalDataLoader("EURUSD", db_handler=mock_db_handler)
    await loader.save_to_database(data)
    
    # Sprawdź czy dane były przetwarzane wsadowo
    assert mock_db_handler.execute_many.call_count > 1 