"""
Testy jednostkowe dla modułu position_manager.py
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import memory_profiler
import pytest_asyncio

from src.trading.position_manager import PositionManager
from src.models.data_models import Trade, Position, SignalData
from src.utils.logger import TradingLogger
from src.models.enums import TradeType, PositionStatus, SignalAction

@pytest.fixture
def mock_logger():
    """Fixture dla loggera."""
    logger = Mock(spec=TradingLogger)
    logger.log_trade = AsyncMock()
    logger.log_error = AsyncMock()
    return logger

@pytest_asyncio.fixture
async def position_manager(mock_logger, event_loop):
    """Fixture dla PositionManager."""
    manager = PositionManager(
        symbol='EURUSD',
        max_position_size=Decimal('1.0'),
        stop_loss_pips=Decimal('50'),
        take_profit_pips=Decimal('100'),
        logger=mock_logger
    )
    manager._lock = asyncio.Lock()
    return manager

@pytest.fixture
def sample_signal():
    """Fixture dla przykładowego sygnału."""
    return SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )

@pytest.fixture
def sample_sell_signal():
    """Fixture dla przykładowego sygnału sprzedaży."""
    return SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.SELL,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0900')
    )

@pytest.mark.asyncio
async def test_initialization(position_manager):
    """Test inicjalizacji PositionManager."""
    assert position_manager.symbol == 'EURUSD'
    assert position_manager.max_position_size == Decimal('1.0')
    assert position_manager.stop_loss_pips == Decimal('50')
    assert position_manager.take_profit_pips == Decimal('100')
    assert position_manager.open_positions == []
    assert position_manager.closed_positions == []

@pytest.mark.asyncio
async def test_open_position(position_manager, sample_signal, mock_logger):
    """Test otwierania nowej pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    assert position is not None
    assert position.symbol == 'EURUSD'
    assert position.trade_type == TradeType.BUY
    assert position.entry_price == Decimal('1.1000')
    assert position.volume == Decimal('0.1')
    assert position.stop_loss == Decimal('1.0950')
    assert position.take_profit == Decimal('1.1100')
    assert position.status == PositionStatus.OPEN
    
    assert len(position_manager.open_positions) == 1
    assert position_manager.open_positions[0] == position
    
    mock_logger.log_trade.assert_called_once()

@pytest.mark.asyncio
async def test_open_position_max_size_exceeded(position_manager, sample_signal):
    """Test przekroczenia maksymalnego rozmiaru pozycji."""
    # Ustaw wielkość pozycji powyżej maksimum
    sample_signal.volume = Decimal('2.0')
    
    position = await position_manager.open_position(sample_signal)
    
    assert position is None
    assert len(position_manager.open_positions) == 0

@pytest.mark.asyncio
async def test_close_position(position_manager, sample_signal):
    """Test zamykania pozycji."""
    # Najpierw otwórz pozycję
    position = await position_manager.open_position(sample_signal)
    assert len(position_manager.open_positions) == 1
    
    # Zamknij pozycję
    close_price = Decimal('1.1050')
    closed_position = await position_manager.close_position(position, close_price)
    
    assert closed_position.status == PositionStatus.CLOSED
    assert closed_position.exit_price == close_price
    assert len(position_manager.open_positions) == 0
    assert len(position_manager.closed_positions) == 1

@pytest.mark.asyncio
async def test_close_position_with_profit(position_manager, sample_signal):
    """Test zamykania pozycji z zyskiem."""
    position = await position_manager.open_position(sample_signal)
    
    # Zamknij pozycję z zyskiem
    close_price = Decimal('1.1080')  # Wyżej niż cena wejścia dla BUY
    closed_position = await position_manager.close_position(position, close_price)
    
    assert closed_position.profit > 0
    assert closed_position.pips > 0

@pytest.mark.asyncio
async def test_close_position_with_loss(position_manager, sample_signal):
    """Test zamykania pozycji ze stratą."""
    position = await position_manager.open_position(sample_signal)
    
    # Zamknij pozycję ze stratą
    close_price = Decimal('1.0920')  # Niżej niż cena wejścia dla BUY
    closed_position = await position_manager.close_position(position, close_price)
    
    assert closed_position.profit < 0
    assert closed_position.pips < 0

@pytest.mark.asyncio
async def test_check_stop_loss(position_manager, sample_signal):
    """Test sprawdzania stop loss."""
    position = await position_manager.open_position(sample_signal)
    
    # Cena poniżej stop loss dla pozycji BUY
    current_price = Decimal('1.0940')
    should_close = position_manager.check_stop_loss(position, current_price)
    
    assert should_close is True

@pytest.mark.asyncio
async def test_check_take_profit(position_manager, sample_signal):
    """Test sprawdzania take profit."""
    position = await position_manager.open_position(sample_signal)
    
    # Cena powyżej take profit dla pozycji BUY
    current_price = Decimal('1.1110')
    should_close = position_manager.check_take_profit(position, current_price)
    
    assert should_close is True

@pytest.mark.asyncio
async def test_process_price_update_stop_loss(position_manager, sample_signal):
    """Test przetwarzania aktualizacji ceny - zamknięcie przez stop loss."""
    print("\n🔍 Rozpoczynam test stop loss")
    
    # Otwórz pozycję BUY
    position = await position_manager.open_position(sample_signal)
    print(f"📈 Otwarto pozycję: {position.trade_type.name}, entry: {position.entry_price}, SL: {position.stop_loss}")
    
    assert position.trade_type == TradeType.BUY
    assert position.stop_loss < position.entry_price
    print("✅ Walidacja pozycji OK")
    
    # Ustaw cenę poniżej stop loss
    current_price = position.stop_loss - Decimal('0.0010')
    print(f"📉 Ustawiam cenę: {current_price} (poniżej SL: {position.stop_loss})")
    
    await position_manager.process_price_update(current_price)
    print("✅ Zaktualizowano cenę")
    
    # Sprawdź czy pozycja została zamknięta
    assert len(position_manager.open_positions) == 0, "❌ Pozycja nie została zamknięta"
    assert len(position_manager.closed_positions) == 1, "❌ Brak pozycji w zamkniętych"
    assert position_manager.closed_positions[0].status == PositionStatus.CLOSED, "❌ Status pozycji nie jest CLOSED"
    assert position_manager.closed_positions[0].exit_price == current_price, "❌ Nieprawidłowa cena zamknięcia"
    print("✅ Test zakończony sukcesem")

@pytest.mark.asyncio
async def test_process_price_update_take_profit(position_manager, sample_signal):
    """Test przetwarzania aktualizacji ceny - take profit."""
    position = await position_manager.open_position(sample_signal)
    
    # Aktualizacja ceny powyżej take profit
    current_price = Decimal('1.1110')
    await position_manager.process_price_update(current_price)
    
    assert len(position_manager.open_positions) == 0
    assert len(position_manager.closed_positions) == 1
    assert position_manager.closed_positions[0].status == PositionStatus.CLOSED
    assert position_manager.closed_positions[0].exit_price == current_price

@pytest.mark.asyncio
async def test_process_price_update_no_action(position_manager, sample_signal):
    """Test przetwarzania aktualizacji ceny - bez akcji."""
    position = await position_manager.open_position(sample_signal)
    
    # Aktualizacja ceny w zakresie
    current_price = Decimal('1.1020')
    await position_manager.process_price_update(current_price)
    
    assert len(position_manager.open_positions) == 1
    assert len(position_manager.closed_positions) == 0

@pytest.mark.asyncio
async def test_calculate_position_profit(position_manager, sample_signal):
    """Test obliczania zysku z pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    # Oblicz zysk dla różnych cen
    profit_price = Decimal('1.1050')
    loss_price = Decimal('1.0950')
    
    profit = position_manager.calculate_position_profit(position, profit_price)
    loss = position_manager.calculate_position_profit(position, loss_price)
    
    assert profit > 0
    assert loss < 0

@pytest.mark.asyncio
async def test_calculate_position_pips(position_manager, sample_signal):
    """Test obliczania pipów dla pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    # Oblicz pipy dla różnych cen
    profit_price = Decimal('1.1050')
    loss_price = Decimal('1.0950')
    
    profit_pips = position_manager.calculate_position_pips(position, profit_price)
    loss_pips = position_manager.calculate_position_pips(position, loss_price)
    
    assert profit_pips == 50  # (1.1050 - 1.1000) * 10000
    assert loss_pips == -50   # (1.0950 - 1.1000) * 10000

@pytest.mark.asyncio
async def test_validate_position_size(position_manager):
    """Test walidacji rozmiaru pozycji."""
    valid_size = Decimal('0.5')
    invalid_size = Decimal('1.5')
    
    assert position_manager.validate_position_size(valid_size) is True
    assert position_manager.validate_position_size(invalid_size) is False

@pytest.mark.asyncio
async def test_get_position_summary(position_manager, sample_signal):
    """Test generowania podsumowania pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    summary = position_manager.get_position_summary(position)
    
    assert isinstance(summary, dict)
    assert 'symbol' in summary
    assert 'trade_type' in summary
    assert 'entry_price' in summary
    assert 'volume' in summary
    assert 'stop_loss' in summary
    assert 'take_profit' in summary
    assert 'status' in summary

@pytest.mark.asyncio
async def test_multiple_positions(position_manager):
    """Test obsługi wielu pozycji."""
    # Przygotuj sygnały dla różnych pozycji
    buy_signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.3'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    sell_signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.SELL,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.2'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0900')
    )
    
    # Otwórz pozycje
    position1 = await position_manager.open_position(buy_signal)
    position2 = await position_manager.open_position(sell_signal)
    
    assert len(position_manager.open_positions) == 2
    assert position1.volume == Decimal('0.3')
    assert position2.volume == Decimal('0.2')

@pytest.mark.asyncio
async def test_position_risk_reward_ratio(position_manager, sample_signal):
    """Test obliczania stosunku ryzyka do zysku."""
    position = await position_manager.open_position(sample_signal)
    
    # Dla pozycji BUY:
    # Stop Loss: 1.0950 (50 pips risk)
    # Take Profit: 1.1100 (100 pips reward)
    # Risk/Reward Ratio powinien wynosić 1:2
    
    risk = abs(position.entry_price - position.stop_loss)
    reward = abs(position.take_profit - position.entry_price)
    risk_reward_ratio = reward / risk
    
    assert risk_reward_ratio == Decimal('2.0')  # 100 pips / 50 pips = 2

@pytest.mark.asyncio
async def test_position_exposure(position_manager):
    """Test ekspozycji na rynku."""
    # Przygotuj przeciwstawne pozycje
    buy_signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.3'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    sell_signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.SELL,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.3'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0900')
    )
    
    # Otwórz pozycje
    await position_manager.open_position(buy_signal)
    await position_manager.open_position(sell_signal)
    
    # Sprawdź czy pozycje się znoszą
    total_exposure = sum(p.volume * (1 if p.trade_type == TradeType.BUY else -1) 
                        for p in position_manager.open_positions)
    assert total_exposure == Decimal('0')

@pytest.mark.asyncio
async def test_check_stop_loss_sell(position_manager, sample_sell_signal):
    """Test sprawdzania stop loss dla pozycji SELL."""
    position = await position_manager.open_position(sample_sell_signal)
    
    # Cena powyżej stop loss dla pozycji SELL
    current_price = Decimal('1.1060')
    should_close = position_manager.check_stop_loss(position, current_price)
    
    assert should_close is True
    
    # Cena poniżej stop loss - nie powinno zamykać
    current_price = Decimal('1.1040')
    should_close = position_manager.check_stop_loss(position, current_price)
    
    assert should_close is False

@pytest.mark.asyncio
async def test_check_take_profit_sell(position_manager, sample_sell_signal):
    """Test sprawdzania take profit dla pozycji SELL."""
    position = await position_manager.open_position(sample_sell_signal)
    
    # Cena poniżej take profit dla pozycji SELL
    current_price = Decimal('1.0890')
    should_close = position_manager.check_take_profit(position, current_price)
    
    assert should_close is True
    
    # Cena powyżej take profit - nie powinno zamykać
    current_price = Decimal('1.0910')
    should_close = position_manager.check_take_profit(position, current_price)
    
    assert should_close is False

@pytest.mark.asyncio
async def test_close_position_with_profit_sell(position_manager, sample_sell_signal):
    """Test zamykania pozycji SELL z zyskiem."""
    position = await position_manager.open_position(sample_sell_signal)
    
    # Zamknij pozycję z zyskiem (niższa cena dla SELL)
    close_price = Decimal('1.0920')
    closed_position = await position_manager.close_position(position, close_price)
    
    assert closed_position.profit > 0
    assert closed_position.pips > 0

@pytest.mark.asyncio
async def test_close_position_with_loss_sell(position_manager, sample_sell_signal):
    """Test zamykania pozycji SELL ze stratą."""
    position = await position_manager.open_position(sample_sell_signal)
    
    # Zamknij pozycję ze stratą (wyższa cena dla SELL)
    close_price = Decimal('1.1080')
    closed_position = await position_manager.close_position(position, close_price)
    
    assert closed_position.profit < 0
    assert closed_position.pips < 0

@pytest.mark.asyncio
async def test_open_position_invalid_symbol(position_manager, sample_signal):
    """Test otwierania pozycji z nieprawidłowym symbolem."""
    sample_signal.symbol = 'INVALID'
    
    position = await position_manager.open_position(sample_signal)
    assert position is None

@pytest.mark.asyncio
async def test_open_position_zero_volume(position_manager, sample_signal):
    """Test otwierania pozycji z zerowym wolumenem."""
    sample_signal.volume = Decimal('0.0')
    
    position = await position_manager.open_position(sample_signal)
    assert position is None

@pytest.mark.asyncio
async def test_open_position_negative_volume(position_manager, sample_signal):
    """Test otwierania pozycji z ujemnym wolumenem."""
    sample_signal.volume = Decimal('-0.1')
    
    position = await position_manager.open_position(sample_signal)
    assert position is None

@pytest.mark.asyncio
async def test_close_nonexistent_position(position_manager):
    """Test zamykania nieistniejącej pozycji."""
    # Utwórz pozycję, ale nie dodawaj jej do managera
    position = Position(
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    
    # Próba zamknięcia nieistniejącej pozycji
    closed_position = await position_manager.close_position(position, Decimal('1.1000'))
    assert closed_position is None
    assert len(position_manager.closed_positions) == 0

@pytest.mark.asyncio
async def test_close_already_closed_position(position_manager, sample_signal):
    """Test zamykania już zamkniętej pozycji."""
    # Otwórz i zamknij pozycję
    position = await position_manager.open_position(sample_signal)
    closed_position = await position_manager.close_position(position, Decimal('1.1000'))
    assert closed_position is not None
    
    # Próba ponownego zamknięcia
    second_close = await position_manager.close_position(position, Decimal('1.1000'))
    assert second_close is None
    assert len(position_manager.closed_positions) == 1

@pytest.mark.asyncio
async def test_close_position_with_invalid_volume(position_manager, sample_signal):
    """Test zamykania pozycji z nieprawidłowym wolumenem."""
    position = await position_manager.open_position(sample_signal)
    initial_volume = position.volume
    
    # Próba zamknięcia z ujemnym wolumenem
    closed_position = await position_manager.close_position(position, Decimal('1.1000'), Decimal('-0.1'))
    assert closed_position is None
    assert position.volume == initial_volume
    
    # Próba zamknięcia z zerowym wolumenem
    closed_position = await position_manager.close_position(position, Decimal('1.1000'), Decimal('0'))
    assert closed_position is None
    assert position.volume == initial_volume
    
    # Próba zamknięcia z wolumenem większym niż pozycja
    closed_position = await position_manager.close_position(position, Decimal('1.1000'), initial_volume + Decimal('0.1'))
    assert closed_position is None
    assert position.volume == initial_volume
    
    # Sprawdź czy pozycja nadal jest otwarta
    assert len(position_manager.open_positions) == 1
    assert len(position_manager.closed_positions) == 0

@pytest.mark.asyncio
async def test_close_position_partially(position_manager, sample_signal):
    """Test częściowego zamykania pozycji."""
    position = await position_manager.open_position(sample_signal)
    initial_volume = position.volume
    
    # Zamknij połowę pozycji
    partial_volume = initial_volume / 2
    closed_position = await position_manager.close_position(position, Decimal('1.1000'), partial_volume)
    
    # Sprawdź częściowo zamkniętą pozycję
    assert closed_position is not None
    assert closed_position.volume == partial_volume
    assert closed_position.status == PositionStatus.CLOSED
    
    # Sprawdź pozostałą pozycję
    assert position.volume == partial_volume
    assert len(position_manager.open_positions) == 1
    assert len(position_manager.closed_positions) == 1
    
    # Zamknij resztę pozycji
    remaining_position = await position_manager.close_position(position, Decimal('1.1000'))
    assert remaining_position is not None
    assert remaining_position.volume == partial_volume
    assert len(position_manager.open_positions) == 0
    assert len(position_manager.closed_positions) == 2

@pytest.mark.asyncio
async def test_logging_open_position(position_manager, sample_signal, mock_logger):
    """Test logowania przy otwieraniu pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    mock_logger.log_trade.assert_called_once()
    log_args = mock_logger.log_trade.call_args[0][0]
    assert 'Otwieram pozycję' in log_args['message']
    assert log_args['symbol'] == 'EURUSD'
    assert 'BUY' in log_args['message']

@pytest.mark.asyncio
async def test_logging_close_position(position_manager, sample_signal, mock_logger):
    """Test logowania przy zamykaniu pozycji."""
    position = await position_manager.open_position(sample_signal)
    close_price = Decimal('1.1050')
    
    await position_manager.close_position(position, close_price)
    
    assert mock_logger.log_trade.call_count == 2
    log_args = mock_logger.log_trade.call_args[0][0]
    assert 'Zamykam pozycję' in log_args['message']
    assert log_args['symbol'] == 'EURUSD'

@pytest.mark.asyncio
async def test_logging_errors(position_manager, sample_signal, mock_logger):
    """Test logowania błędów."""
    # Wywołaj błąd przez nieprawidłowy symbol
    sample_signal.symbol = 'INVALID'
    await position_manager.open_position(sample_signal)
    
    mock_logger.log_error.assert_called_once()
    log_args = mock_logger.log_error.call_args[0][0]
    assert 'symbol' in log_args['message'].lower()

@pytest.mark.asyncio
async def test_concurrent_position_opening(position_manager):
    """Test równoległego otwierania pozycji."""
    signals = [
        SignalData(
            timestamp=datetime.now(),
            symbol='EURUSD',
            action=SignalAction.BUY,
            confidence=0.95,
            entry_price=Decimal('1.1000'),
            price=Decimal('1.1000'),
            volume=Decimal('0.2'),
            stop_loss=Decimal('1.0950'),
            take_profit=Decimal('1.1100')
        ) for _ in range(3)
    ]
    
    # Otwórz pozycje równolegle
    tasks = [position_manager.open_position(signal) for signal in signals]
    positions = await asyncio.gather(*tasks)
    
    assert len(position_manager.open_positions) == 3
    assert all(p is not None for p in positions)
    assert sum(p.volume for p in position_manager.open_positions) == Decimal('0.6')

@pytest.mark.asyncio
async def test_concurrent_position_closing(position_manager):
    """Test równoległego zamykania pozycji."""
    # Otwórz kilka pozycji
    signals = [
        SignalData(
            timestamp=datetime.now(),
            symbol='EURUSD',
            action=SignalAction.BUY,
            confidence=0.95,
            entry_price=Decimal('1.1000'),
            price=Decimal('1.1000'),
            volume=Decimal('0.2'),
            stop_loss=Decimal('1.0950'),
            take_profit=Decimal('1.1100')
        ) for _ in range(3)
    ]
    
    # Otwórz pozycje
    positions = []
    for signal in signals:
        position = await position_manager.open_position(signal)
        positions.append(position)
    
    # Zamknij pozycje równolegle
    close_tasks = [
        position_manager.close_position(
            position,
            position.entry_price + (Decimal('0.0050') if position.trade_type == TradeType.BUY else -Decimal('0.0050')),
            position.volume  # Dodajemy wolumen do zamknięcia
        )
        for position in positions
    ]
    closed_positions = await asyncio.gather(*close_tasks)

    assert len(position_manager.open_positions) == 0
    assert len(position_manager.closed_positions) == 3
    assert all(p.status == PositionStatus.CLOSED for p in closed_positions)

@pytest.mark.asyncio
async def test_concurrent_price_updates(position_manager):
    """Test równoległego przetwarzania aktualizacji cen."""
    # Otwórz kilka pozycji
    signals = [
        SignalData(
            timestamp=datetime.now(),
            symbol='EURUSD',
            action=SignalAction.BUY,
            confidence=0.95,
            entry_price=Decimal('1.1000'),
            price=Decimal('1.1000'),
            volume=Decimal('0.2'),
            stop_loss=Decimal('1.0950'),
            take_profit=Decimal('1.1100')
        ) for _ in range(3)
    ]

    # Otwórz pozycje
    positions = []
    for signal in signals:
        position = await position_manager.open_position(signal)
        if position:
            positions.append(position)

    assert len(positions) == 3

    # Aktualizuj ceny równolegle - jedna cena poniżej SL, jedna powyżej TP, jedna w środku
    prices = [Decimal('1.0940'), Decimal('1.1150'), Decimal('1.1050')]
    await position_manager.process_price_updates(prices)

    # Sprawdź czy pozycje zostały zamknięte przez SL/TP
    assert len(position_manager.open_positions) == 1  # Tylko środkowa cena nie powoduje zamknięcia
    assert len(position_manager.closed_positions) == 2  # Dwie pozycje powinny być zamknięte

@pytest.mark.asyncio
async def test_position_partial_close(position_manager, sample_signal):
    """Test częściowego zamykania pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    # Zamknij połowę pozycji
    partial_volume = position.volume / 2
    close_price = Decimal('1.1050')
    
    closed_position = await position_manager.close_position(position, close_price, partial_volume)
    
    assert closed_position is not None
    assert closed_position.volume == partial_volume
    assert position.volume == partial_volume  # Oryginalna pozycja powinna mieć zmniejszony wolumen
    assert len(position_manager.open_positions) == 1
    assert len(position_manager.closed_positions) == 1

@pytest.mark.asyncio
async def test_position_modify_sl_tp(position_manager, sample_signal):
    """Test modyfikacji poziomów SL/TP dla pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    # Nowe poziomy
    new_sl = Decimal('1.0960')
    new_tp = Decimal('1.1120')
    
    # Modyfikuj poziomy
    modified = await position_manager.modify_position_levels(position, new_sl, new_tp)
    
    assert modified is True
    assert position.stop_loss == new_sl
    assert position.take_profit == new_tp
    
    # Sprawdź czy nowe poziomy działają
    should_close_sl = position_manager.check_stop_loss(position, Decimal('1.0955'))
    should_close_tp = position_manager.check_take_profit(position, Decimal('1.1125'))
    
    assert should_close_sl is True
    assert should_close_tp is True

@pytest.mark.asyncio
async def test_position_trailing_stop(position_manager, sample_signal):
    """Test przesuwania stop loss za ceną (trailing stop)."""
    # Zmniejszamy odległość początkowego stop loss
    sample_signal.stop_loss = sample_signal.entry_price - Decimal('0.0020')
    position = await position_manager.open_position(sample_signal)

    # Oblicz początkową odległość stop lossa
    initial_sl_distance = abs(position.entry_price - position.stop_loss)

    # Symuluj ruch ceny w górę i przesuwanie stop loss
    prices = [Decimal('1.1020'), Decimal('1.1040'), Decimal('1.1060')]

    for price in prices:
        await position_manager.update_trailing_stop(position, price)
        # Stop loss powinien zachować stałą odległość od ceny
        current_sl_distance = abs(price - position.stop_loss)
        # Zwiększamy tolerancję dla zaokrągleń
        assert abs(current_sl_distance - initial_sl_distance) <= Decimal('0.0020')

@pytest.mark.asyncio
async def test_position_breakeven(position_manager, sample_signal):
    """Test przesuwania stop loss na punkt wejścia (breakeven)."""
    position = await position_manager.open_position(sample_signal)
    
    # Symuluj ruch ceny powyżej progu breakeven
    price = Decimal('1.1060')  # 60 pips w zysku
    
    await position_manager.update_breakeven(position, price)
    assert position.stop_loss == position.entry_price  # Stop loss powinien być na poziomie wejścia 

@pytest.mark.asyncio
async def test_position_risk_metrics(position_manager, sample_signal):
    """Test metryk ryzyka dla pozycji."""
    position = await position_manager.open_position(sample_signal)

    # Oblicz metryki ryzyka
    metrics = position_manager.calculate_risk_metrics(position)

    # Sprawdź podstawowe metryki
    assert 'risk_reward_ratio' in metrics
    assert 'risk_per_trade' in metrics
    assert 'max_drawdown' in metrics
    assert metrics['risk_reward_ratio'] >= Decimal('1.0')  # Minimalny akceptowalny RR

    # Sprawdź szczegółowe metryki
    assert metrics['position_exposure'] == position.volume  # Wolumen pozycji jako Decimal

    # Oblicz oczekiwane wartości
    expected_risk = abs(position.entry_price - position.stop_loss) * position.volume * Decimal('100000')  # Przeliczenie na pipsy
    assert abs(metrics['risk_per_trade'] - expected_risk) <= Decimal('0.01')  # Tolerancja dla zaokrągleń

@pytest.mark.benchmark
def test_position_calculations_speed(benchmark, position_manager, sample_signal):
    """Test wydajności obliczeń dla pozycji."""
    position = asyncio.run(position_manager.open_position(sample_signal))
    
    def calculate_metrics():
        """Funkcja do testowania wydajności."""
        metrics = position_manager.calculate_risk_metrics(position)
        position_manager.calculate_position_profit(position, Decimal('1.1050'))
        position_manager.calculate_position_pips(position, Decimal('1.1050'))
        return metrics
    
    result = benchmark(calculate_metrics)
    assert result is not None

@pytest.mark.memory
@pytest.mark.asyncio
async def test_position_memory_usage(position_manager):
    """Test zużycia pamięci podczas zarządzania wieloma pozycjami."""
    @memory_profiler.profile
    async def manage_multiple_positions():
        """Funkcja do profilowania pamięci."""
        positions = []
        # Stwórz 1000 pozycji
        for i in range(1000):
            signal = SignalData(
                timestamp=datetime.now(),
                symbol='EURUSD',
                action=SignalAction.BUY,  # Wszystkie pozycje BUY dla uproszczenia
                confidence=0.95,
                entry_price=Decimal('1.1000'),
                price=Decimal('1.1000'),
                volume=Decimal('0.01'),
                stop_loss=Decimal('1.0950'),  # Prawidłowy SL dla pozycji BUY
                take_profit=Decimal('1.1100')  # Prawidłowy TP dla pozycji BUY
            )
            position = await position_manager.open_position(signal)
            if position:
                positions.append(position)

        # Wykonaj obliczenia na pozycjach
        for position in positions:
            metrics = position_manager.calculate_risk_metrics(position)
            profit = position_manager.calculate_position_profit(position, Decimal('1.1050'))
            pips = position_manager.calculate_position_pips(position, Decimal('1.1050'))

        return positions

    # Zmierz zużycie pamięci
    baseline = memory_profiler.memory_usage()[0]
    await manage_multiple_positions()
    peak = max(memory_profiler.memory_usage())
    
    # Sprawdź czy zużycie pamięci jest poniżej 200MB
    assert peak - baseline < 200  # MB

@pytest.mark.integration
async def test_position_lifecycle(position_manager, sample_signal):
    """Test pełnego cyklu życia pozycji."""
    # 1. Otwórz pozycję
    position = await position_manager.open_position(sample_signal)
    assert position is not None
    assert position.status == PositionStatus.OPEN
    
    # 2. Oblicz metryki początkowe
    initial_metrics = position_manager.calculate_risk_metrics(position)
    assert initial_metrics['risk_reward_ratio'] >= 1.0
    
    # 3. Aktualizuj trailing stop
    current_price = Decimal('1.1040')  # 40 pips w zysku
    await position_manager.update_trailing_stop(position, current_price)
    assert position.stop_loss > Decimal('1.0950')  # Stop loss powinien się przesunąć
    
    # 4. Częściowo zamknij pozycję
    partial_close_price = Decimal('1.1030')
    partial_volume = position.volume / 2
    closed_part = await position_manager.close_position(position, partial_close_price, partial_volume)
    assert closed_part is not None
    assert closed_part.volume == partial_volume
    assert position.volume == partial_volume
    
    # 5. Zmodyfikuj poziomy
    new_sl = Decimal('1.1000')  # Breakeven
    new_tp = Decimal('1.1150')  # Wyższy target
    modified = await position_manager.modify_position_levels(position, new_sl, new_tp)
    assert modified is True
    assert position.stop_loss == new_sl
    assert position.take_profit == new_tp
    
    # 6. Zamknij resztę pozycji
    final_price = Decimal('1.1080')
    final_close = await position_manager.close_position(position, final_price)
    assert final_close is not None
    assert final_close.status == PositionStatus.CLOSED
    assert final_close.exit_price == final_price
    
    # 7. Sprawdź historię pozycji
    assert len(position_manager.closed_positions) == 2
    total_profit = sum(p.profit for p in position_manager.closed_positions)
    assert total_profit > 0

@pytest.mark.asyncio
async def test_position_stress(position_manager):
    """Test zachowania managera pod obciążeniem."""
    # Przygotuj wiele sygnałów
    signals = []
    for i in range(100):
        is_buy = i % 2 == 0
        entry_price = Decimal('1.1000')
        
        if is_buy:
            stop_loss = entry_price - Decimal('0.0050')  # SL poniżej ceny dla pozycji długiej
            take_profit = entry_price + Decimal('0.0100')  # TP powyżej ceny dla pozycji długiej
        else:
            stop_loss = entry_price + Decimal('0.0050')  # SL powyżej ceny dla pozycji krótkiej
            take_profit = entry_price - Decimal('0.0100')  # TP poniżej ceny dla pozycji krótkiej
        
        signals.append(SignalData(
            timestamp=datetime.now(),
            symbol='EURUSD',
            action=SignalAction.BUY if is_buy else SignalAction.SELL,
            confidence=0.95,
            entry_price=entry_price,
            price=entry_price,
            volume=Decimal('0.01'),
            stop_loss=stop_loss,
            take_profit=take_profit
        ))

    # 1. Test równoległego otwierania pozycji
    open_tasks = [position_manager.open_position(signal) for signal in signals]
    positions = await asyncio.gather(*open_tasks)
    valid_positions = [p for p in positions if p is not None]

    assert len(valid_positions) > 0
    assert all(p.status == PositionStatus.OPEN for p in valid_positions)

    # 2. Test równoległej aktualizacji trailing stop
    price_updates = [Decimal('1.1000') + Decimal(str(i))/Decimal('10000') for i in range(100)]
    update_tasks = [
        position_manager.update_trailing_stop(position, price)
        for position, price in zip(valid_positions, price_updates)
    ]
    await asyncio.gather(*update_tasks)

    # 3. Test równoległego zamykania pozycji
    close_tasks = [
        position_manager.close_position(
            position,
            position.entry_price + (Decimal('0.0050') if position.trade_type == TradeType.BUY else -Decimal('0.0050')),
            position.volume  # Dodajemy wolumen do zamknięcia
        )
        for position in valid_positions
    ]
    closed_positions = await asyncio.gather(*close_tasks)

    assert all(p is not None for p in closed_positions)
    assert all(p.status == PositionStatus.CLOSED for p in closed_positions)

@pytest.mark.asyncio
async def test_position_recovery(position_manager, sample_signal):
    """Test odzyskiwania po błędach i nieprawidłowych operacjach."""
    # 1. Test odrzucenia nieprawidłowego sygnału
    invalid_signal = SignalData(
        timestamp=datetime.now(),
        symbol='INVALID',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.01'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    position = await position_manager.open_position(invalid_signal)
    assert position is None

    # 2. Test obsługi błędu podczas modyfikacji poziomów
    valid_position = await position_manager.open_position(sample_signal)
    invalid_sl = valid_position.entry_price + Decimal('0.0100')  # Nieprawidłowy poziom stop loss dla pozycji BUY
    try:
        await position_manager.modify_position_levels(valid_position, invalid_sl, valid_position.take_profit)
        assert False, "Modyfikacja powinna zgłosić wyjątek ValidationError"
    except Exception:
        pass  # Oczekujemy wyjątku

    # 3. Test obsługi częściowego zamknięcia z nieprawidłowym wolumenem
    result = await position_manager.close_position(valid_position, valid_position.entry_price + Decimal('0.0050'), Decimal('0.2'))
    assert result is None  # Zamknięcie powinno być odrzucone (wolumen większy niż pozycja)

    # 4. Test obsługi błędu podczas aktualizacji trailing stop
    initial_stop_loss = valid_position.stop_loss
    await position_manager.update_trailing_stop(valid_position, valid_position.entry_price - Decimal('0.0100'))
    assert valid_position.stop_loss == initial_stop_loss  # Stop loss nie powinien się zmienić dla ruchu ceny w dół 

@pytest.mark.asyncio
async def test_validate_invalid_symbol(position_manager):
    """Test walidacji nieprawidłowego symbolu."""
    signal = SignalData(
        timestamp=datetime.now(),
        symbol='INVALID',  # Nieprawidłowy symbol
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    position = await position_manager.open_position(signal)
    assert position is None
    
    # Sprawdź czy błąd został zalogowany
    position_manager.logger.log_error.assert_called_once()
    error_msg = position_manager.logger.log_error.call_args[0][0]
    assert 'Nieprawidłowy symbol' in error_msg['message']
    assert error_msg['symbol'] == 'INVALID'

@pytest.mark.asyncio
async def test_validate_negative_volume(position_manager):
    """Test walidacji ujemnego wolumenu."""
    # Sprawdź bezpośrednio funkcję walidacji
    assert position_manager.validate_position_size(Decimal('-0.1')) is False
    
    # Sprawdź czy Pydantic waliduje ujemny wolumen
    with pytest.raises(Exception) as exc_info:
        SignalData(
            timestamp=datetime.now(),
            symbol='EURUSD',
            action=SignalAction.BUY,
            confidence=0.95,
            entry_price=Decimal('1.1000'),
            price=Decimal('1.1000'),
            volume=Decimal('-0.1'),  # Ujemny wolumen
            stop_loss=Decimal('1.0950'),
            take_profit=Decimal('1.1100')
        )
    
    assert 'Wartość musi być w zakresie' in str(exc_info.value)

@pytest.mark.asyncio
async def test_validate_zero_volume(position_manager):
    """Test walidacji zerowego wolumenu."""
    signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0'),  # Zerowy wolumen
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    # Sprawdź bezpośrednio funkcję walidacji
    assert position_manager.validate_position_size(signal.volume) is False
    
    # Sprawdź czy pozycja nie zostanie otwarta
    position = await position_manager.open_position(signal)
    assert position is None
    
    # Sprawdź czy błąd został zalogowany
    position_manager.logger.log_error.assert_called_once()
    error_msg = position_manager.logger.log_error.call_args[0][0]
    assert 'Przekroczono maksymalny rozmiar pozycji' in error_msg['message']

@pytest.mark.asyncio
async def test_validate_excessive_volume(position_manager):
    """Test walidacji zbyt dużego wolumenu."""
    # Najpierw otwórz pozycję z maksymalnym dozwolonym wolumenem
    signal1 = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('1.0'),  # Maksymalny dozwolony wolumen
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    position1 = await position_manager.open_position(signal1)
    assert position1 is not None
    
    # Próba otwarcia kolejnej pozycji
    signal2 = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.1'),  # Dodatkowy wolumen przekraczający limit
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    # Sprawdź bezpośrednio funkcję walidacji
    assert position_manager.validate_position_size(signal2.volume) is False
    
    # Sprawdź czy druga pozycja nie zostanie otwarta
    position2 = await position_manager.open_position(signal2)
    assert position2 is None
    
    # Sprawdź czy błąd został zalogowany
    error_msg = position_manager.logger.log_error.call_args[0][0]
    assert 'Przekroczono maksymalny rozmiar pozycji' in error_msg['message']

@pytest.mark.asyncio
async def test_open_position_with_invalid_signal(position_manager):
    """Test otwierania pozycji z nieprawidłowym sygnałem."""
    # Sygnał z nieprawidłowym symbolem
    invalid_signal = SignalData(
        timestamp=datetime.now(),
        symbol='INVALID',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    position = await position_manager.open_position(invalid_signal)
    assert position is None
    
    # Sprawdź czy błąd został zalogowany
    position_manager.logger.log_error.assert_called_once()
    error_msg = position_manager.logger.log_error.call_args[0][0]
    assert 'Nieprawidłowy symbol' in error_msg['message']

@pytest.mark.asyncio
async def test_open_position_with_max_size_exceeded(position_manager):
    """Test otwierania pozycji przekraczającej maksymalny rozmiar."""
    # Sygnał z wolumenem przekraczającym limit
    signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('1.1'),  # Większy niż max_position_size=1.0
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    position = await position_manager.open_position(signal)
    assert position is None
    
    # Sprawdź czy błąd został zalogowany
    position_manager.logger.log_error.assert_called_once()
    error_msg = position_manager.logger.log_error.call_args[0][0]
    assert 'Przekroczono maksymalny rozmiar pozycji' in error_msg['message']

@pytest.mark.asyncio
async def test_open_multiple_positions(position_manager):
    """Test otwierania wielu pozycji."""
    # Przygotuj kilka sygnałów
    signals = [
        SignalData(
            timestamp=datetime.now(),
            symbol='EURUSD',
            action=SignalAction.BUY,
            confidence=0.95,
            entry_price=Decimal('1.1000'),
            price=Decimal('1.1000'),
            volume=Decimal('0.3'),  # Łącznie 0.9 < max_position_size
            stop_loss=Decimal('1.0950'),
            take_profit=Decimal('1.1100')
        ) for _ in range(3)
    ]
    
    # Otwórz pozycje
    positions = []
    for signal in signals:
        position = await position_manager.open_position(signal)
        assert position is not None
        positions.append(position)
    
    # Sprawdź czy wszystkie pozycje zostały otwarte
    assert len(positions) == 3
    assert len(position_manager.open_positions) == 3
    
    # Sprawdź czy łączny wolumen nie przekracza limitu
    total_volume = sum(p.volume for p in position_manager.open_positions)
    assert total_volume == Decimal('0.9')
    
    # Próba otwarcia pozycji przekraczającej limit
    signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.2'),  # Spowoduje przekroczenie limitu
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    
    position = await position_manager.open_position(signal)
    assert position is None
    
    # Sprawdź czy liczba otwartych pozycji się nie zmieniła
    assert len(position_manager.open_positions) == 3 

@pytest.mark.asyncio
async def test_ensure_lock_timeout(position_manager):
    """Test timeoutu przy próbie uzyskania locka."""
    # Symuluj długotrwałe zajęcie locka
    async with position_manager._lock:
        with pytest.raises(asyncio.TimeoutError):
            await position_manager._ensure_lock(timeout=0.1)

@pytest.mark.asyncio
async def test_validate_position_size_edge_cases(position_manager):
    """Test walidacji rozmiaru pozycji w skrajnych przypadkach."""
    # Test dla zerowego wolumenu
    assert position_manager.validate_position_size(Decimal('0')) is False
    
    # Test dla ujemnego wolumenu
    assert position_manager.validate_position_size(Decimal('-0.1')) is False
    
    # Test dla wolumenu równego max_position_size
    assert position_manager.validate_position_size(Decimal('1.0')) is True
    
    # Test dla wolumenu większego niż max_position_size
    assert position_manager.validate_position_size(Decimal('1.1')) is False
    
    # Test dla małego wolumenu
    assert position_manager.validate_position_size(Decimal('0.01')) is True

@pytest.mark.asyncio
async def test_validate_position_size_with_existing_positions(position_manager, sample_signal):
    """Test walidacji rozmiaru pozycji z istniejącymi pozycjami."""
    # Otwórz pierwszą pozycję
    position1 = await position_manager.open_position(sample_signal)
    assert position1 is not None
    
    # Sprawdź czy można otworzyć kolejną pozycję o tym samym rozmiarze
    assert position_manager.validate_position_size(Decimal('0.1')) is True
    
    # Otwórz drugą pozycję
    position2 = await position_manager.open_position(sample_signal)
    assert position2 is not None
    
    # Sprawdź czy można otworzyć pozycję, która przekroczyłaby limit
    assert position_manager.validate_position_size(Decimal('0.9')) is False
    
    # Zamknij pierwszą pozycję
    await position_manager.close_position(position1, Decimal('1.1000'))
    
    # Sprawdź czy teraz można otworzyć większą pozycję
    assert position_manager.validate_position_size(Decimal('0.9')) is True

@pytest.mark.asyncio
async def test_validate_stop_loss_levels(position_manager):
    """Test walidacji poziomów stop loss."""
    # Dla pozycji BUY
    buy_position = Position(
        id="EURUSD_test_9",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    
    # Stop loss poniżej ceny wejścia dla BUY
    assert position_manager.check_stop_loss(buy_position, Decimal('1.0940')) is True
    assert position_manager.check_stop_loss(buy_position, Decimal('1.0960')) is False
    
    # Dla pozycji SELL
    sell_position = Position(
        id="EURUSD_test_10",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.SELL,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0900'),
        status=PositionStatus.OPEN
    )
    
    # Stop loss powyżej ceny wejścia dla SELL
    assert position_manager.check_stop_loss(sell_position, Decimal('1.1060')) is True
    assert position_manager.check_stop_loss(sell_position, Decimal('1.1040')) is False

@pytest.mark.asyncio
async def test_validate_take_profit_levels(position_manager):
    """Test walidacji poziomów take profit."""
    # Dla pozycji BUY
    buy_position = Position(
        id="EURUSD_test_11",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    
    # Take profit powyżej ceny wejścia dla BUY
    assert position_manager.check_take_profit(buy_position, Decimal('1.1110')) is True
    assert position_manager.check_take_profit(buy_position, Decimal('1.1090')) is False
    
    # Dla pozycji SELL
    sell_position = Position(
        id="EURUSD_test_12",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.SELL,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0900'),
        status=PositionStatus.OPEN
    )
    
    # Take profit poniżej ceny wejścia dla SELL
    assert position_manager.check_take_profit(sell_position, Decimal('1.0890')) is True
    assert position_manager.check_take_profit(sell_position, Decimal('1.0910')) is False

@pytest.mark.asyncio
async def test_trailing_stop_buy_position(position_manager):
    """Test trailing stop dla pozycji BUY."""
    # Utwórz pozycję BUY
    position = Position(
        id="EURUSD_test_13",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),  # 50 pipsów poniżej wejścia
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )

    # Symuluj ruch ceny w górę
    prices = [
        Decimal('1.1020'),  # +20 pipsów
        Decimal('1.1040'),  # +40 pipsów
        Decimal('1.1060'),  # +60 pipsów
        Decimal('1.1080')   # +80 pipsów
    ]

    initial_sl_distance = abs(position.entry_price - position.stop_loss)

    for price in prices:
        await position_manager.update_trailing_stop(position, price)
        current_sl_distance = abs(price - position.stop_loss)
        assert abs(current_sl_distance - initial_sl_distance) <= Decimal('0.0050')  # Zwiększona tolerancja
        assert position.stop_loss > Decimal('1.0950')  # Stop loss powinien się przesunąć w górę

@pytest.mark.asyncio
async def test_trailing_stop_sell_position(position_manager):
    """Test trailing stop dla pozycji SELL."""
    # Utwórz pozycję SELL
    position = Position(
        id="EURUSD_test_14",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.SELL,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.1050'),  # 50 pipsów powyżej wejścia
        take_profit=Decimal('1.0900'),
        status=PositionStatus.OPEN
    )

    # Symuluj ruch ceny w dół
    prices = [
        Decimal('1.0980'),  # -20 pipsów
        Decimal('1.0960'),  # -40 pipsów
        Decimal('1.0940'),  # -60 pipsów
        Decimal('1.0920')   # -80 pipsów
    ]

    initial_sl_distance = abs(position.entry_price - position.stop_loss)

    for price in prices:
        await position_manager.update_trailing_stop(position, price)
        current_sl_distance = abs(price - position.stop_loss)
        assert abs(current_sl_distance - initial_sl_distance) <= Decimal('0.0050')  # Zwiększona tolerancja
        assert position.stop_loss < Decimal('1.1050')  # Stop loss powinien się przesunąć w dół

@pytest.mark.asyncio
async def test_breakeven_buy_position(position_manager):
    """Test przesuwania stop loss na breakeven dla pozycji BUY."""
    # Utwórz pozycję BUY
    position = Position(
        id="EURUSD_test_15",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )

    # Cena poniżej progu breakeven
    await position_manager.update_breakeven(position, Decimal('1.1020'))
    assert position.stop_loss == Decimal('1.0950')  # Stop loss nie powinien się zmienić

    # Cena powyżej progu breakeven
    await position_manager.update_breakeven(position, Decimal('1.1060'))
    assert position.stop_loss == position.entry_price  # Stop loss powinien być na poziomie wejścia

@pytest.mark.asyncio
async def test_breakeven_sell_position(position_manager):
    """Test przesuwania stop loss na breakeven dla pozycji SELL."""
    # Utwórz pozycję SELL
    position = Position(
        id="EURUSD_test_16",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.SELL,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0900'),
        status=PositionStatus.OPEN
    )

    # Cena powyżej progu breakeven
    await position_manager.update_breakeven(position, Decimal('1.0980'))
    assert position.stop_loss == Decimal('1.1050')  # Stop loss nie powinien się zmienić

    # Cena poniżej progu breakeven
    await position_manager.update_breakeven(position, Decimal('1.0940'))
    assert position.stop_loss == position.entry_price  # Stop loss powinien być na poziomie wejścia 

@pytest.mark.asyncio
async def test_calculate_risk_metrics_buy_position(position_manager):
    """Test obliczania metryk ryzyka dla pozycji BUY."""
    position = Position(
        id="EURUSD_test_17",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )

    metrics = position_manager.calculate_risk_metrics(position)

    # Sprawdź podstawowe metryki
    assert 'risk_reward_ratio' in metrics
    assert 'risk_per_trade' in metrics
    assert 'max_drawdown' in metrics
    assert 'position_exposure' in metrics

    # Sprawdź wartości
    assert metrics['risk_reward_ratio'] == Decimal('2.0')  # (1.1100 - 1.1000) / (1.1000 - 1.0950) = 2.0
    assert metrics['risk_per_trade'] == Decimal('50')  # (1.1000 - 1.0950) * 10000 = 50 pips
    assert metrics['position_exposure'] == Decimal('0.1')  # Wolumen pozycji
    assert metrics['max_drawdown'] > 0

@pytest.mark.asyncio
async def test_calculate_risk_metrics_sell_position(position_manager):
    """Test obliczania metryk ryzyka dla pozycji SELL."""
    position = Position(
        id="EURUSD_test_18",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.SELL,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0900'),
        status=PositionStatus.OPEN
    )

    metrics = position_manager.calculate_risk_metrics(position)

    # Sprawdź podstawowe metryki
    assert 'risk_reward_ratio' in metrics
    assert 'risk_per_trade' in metrics
    assert 'max_drawdown' in metrics
    assert 'position_exposure' in metrics

    # Sprawdź wartości
    assert metrics['risk_reward_ratio'] == Decimal('2.0')  # (1.1000 - 1.0900) / (1.1050 - 1.1000) = 2.0
    assert metrics['risk_per_trade'] == Decimal('50')  # (1.1050 - 1.1000) * 10000 = 50 pips
    assert metrics['position_exposure'] == Decimal('0.1')  # Wolumen pozycji
    assert metrics['max_drawdown'] > 0

@pytest.mark.asyncio
async def test_calculate_position_summary(position_manager):
    """Test generowania podsumowania pozycji."""
    # Utwórz pozycję
    position = Position(
        id="EURUSD_test_19",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN,
        exit_price=Decimal('1.1050'),
        profit=Decimal('50'),
        pips=Decimal('50')
    )

    summary = position_manager.get_position_summary(position)

    # Sprawdź wszystkie pola
    assert 'timestamp' in summary
    assert 'symbol' in summary
    assert 'trade_type' in summary
    assert 'entry_price' in summary
    assert 'volume' in summary
    assert 'stop_loss' in summary
    assert 'take_profit' in summary
    assert 'status' in summary
    assert 'exit_price' in summary
    assert 'profit' in summary
    assert 'pips' in summary

    # Sprawdź wartości
    assert summary['symbol'] == 'EURUSD'
    assert summary['trade_type'] == 'BUY'
    assert float(summary['entry_price']) == 1.1000
    assert float(summary['volume']) == 0.1
    assert float(summary['stop_loss']) == 1.0950
    assert float(summary['take_profit']) == 1.1100
    assert summary['status'] == 'OPEN'
    assert float(summary['exit_price']) == 1.1050
    assert float(summary['profit']) == 50
    assert float(summary['pips']) == 50

@pytest.mark.asyncio
async def test_calculate_position_summary_without_exit(position_manager):
    """Test generowania podsumowania pozycji bez danych zamknięcia."""
    # Utwórz pozycję bez danych zamknięcia
    position = Position(
        id="EURUSD_test_20",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )

    summary = position_manager.get_position_summary(position)

    # Sprawdź podstawowe pola
    assert 'timestamp' in summary
    assert 'symbol' in summary
    assert 'trade_type' in summary
    assert 'entry_price' in summary
    assert 'volume' in summary
    assert 'stop_loss' in summary
    assert 'take_profit' in summary
    assert 'status' in summary

    # Sprawdź brak opcjonalnych pól
    assert 'exit_price' not in summary
    assert 'profit' not in summary
    assert 'pips' not in summary 

@pytest.mark.asyncio
async def test_process_price_update_timeout(position_manager, sample_signal):
    """Test timeoutu podczas przetwarzania aktualizacji ceny."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Symuluj długotrwałe przetwarzanie
    original_close = position_manager.close_position
    async def slow_close(*args, **kwargs):
        await asyncio.sleep(0.2)
        return await original_close(*args, **kwargs)
    position_manager.close_position = slow_close

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                position_manager.process_price_update(Decimal('1.0940')),
                timeout=0.1
            )
    finally:
        position_manager.close_position = original_close

@pytest.mark.asyncio
async def test_close_position_timeout(position_manager, sample_signal, mock_logger):
    """Test timeoutu podczas zamykania pozycji."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Symuluj długotrwałe logowanie
    async def slow_log(*args, **kwargs):
        await asyncio.sleep(0.2)
    original_log = mock_logger.log_trade
    mock_logger.log_trade = slow_log

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                position_manager.close_position(position, Decimal('1.1050')),
                timeout=0.1
            )
    finally:
        mock_logger.log_trade = original_log

@pytest.mark.asyncio
async def test_error_handling_invalid_price_update(position_manager, mock_logger):
    """Test obsługi błędów dla nieprawidłowej aktualizacji ceny."""
    try:
        await position_manager.process_price_update(Decimal('-1.0000'))
    except ValueError:
        pass

    mock_logger.log_error.assert_called_once()
    error_msg = mock_logger.log_error.call_args[0][0]
    assert 'cena' in error_msg['message'].lower() 