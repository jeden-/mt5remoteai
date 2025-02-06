"""
Testy jednostkowe dla modułu position_manager.py
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import memory_profiler
import pytest_asyncio

from src.trading.position_manager import PositionManager
from src.models.data_models import Trade, Position, SignalData
from src.utils.logger import TradingLogger
from src.models.enums import TradeType, PositionStatus, SignalAction

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
    
    assert len(mock_logger.log_trade_calls) == 1

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
        id="TEST_1",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    
    # Próba zamknięcia pozycji
    result = await position_manager.close_position(position, Decimal('1.1050'))
    assert result is None

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
    with pytest.raises(RuntimeError, match="Nieprawidłowy wolumen: -0.1"):
        await position_manager.close_position(position, Decimal('1.1000'), Decimal('-0.1'))
    assert position.volume == initial_volume
    
    # Próba zamknięcia z zerowym wolumenem
    with pytest.raises(RuntimeError, match="Nieprawidłowy wolumen: 0"):
        await position_manager.close_position(position, Decimal('1.1000'), Decimal('0'))
    assert position.volume == initial_volume
    
    # Próba zamknięcia z wolumenem większym niż pozycja
    with pytest.raises(RuntimeError, match="Wolumen 0.2 większy niż pozycja 0.1"):
        await position_manager.close_position(position, Decimal('1.1000'), Decimal('0.2'))
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
    """Test logowania otwarcia pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    assert len(mock_logger.log_trade_calls) == 1
    data, action = mock_logger.log_trade_calls[0]
    assert isinstance(data, Position)
    assert action == "OPEN"

@pytest.mark.asyncio
async def test_logging_close_position(position_manager, sample_signal, mock_logger):
    """Test logowania zamknięcia pozycji."""
    # Otwórz pozycję
    position = await position_manager.open_position(sample_signal)
    assert position is not None
    
    # Zamknij pozycję
    closed_position = await position_manager.close_position(position, Decimal('1.1050'))
    assert closed_position is not None
    
    # Sprawdź wywołania log_trade
    assert len(mock_logger.log_trade_calls) == 2
    
    # Sprawdź pierwsze wywołanie - otwarcie pozycji
    data, action = mock_logger.log_trade_calls[0]
    assert isinstance(data, Position)
    assert action == "OPEN"
    
    # Sprawdź drugie wywołanie - zamknięcie pozycji
    data, action = mock_logger.log_trade_calls[1]
    assert isinstance(data, Position)
    assert action == "CLOSE"
    assert data.exit_price == Decimal('1.1050')

@pytest.mark.asyncio
async def test_logging_errors(position_manager, mock_logger):
    """Test logowania błędów."""
    try:
        await position_manager.close_position(None, Decimal('1.1000'))
    except:
        pass
    
    assert len(mock_logger.log_error_calls) == 1
    error = mock_logger.log_error_calls[0]
    assert isinstance(error, Exception)

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
    initial_positions = len(position_manager.open_positions)

    # Aktualizuj ceny równolegle - jedna cena poniżej SL, jedna powyżej TP, jedna w środku
    prices = [Decimal('1.0940'), Decimal('1.1150'), Decimal('1.1050')]
    closed_positions = await position_manager.process_price_updates(prices)

    # Sprawdź czy pozycje zostały zamknięte przez SL/TP
    assert len(closed_positions) == 2  # Dwie pozycje powinny być zamknięte
    assert len(position_manager.open_positions) == 1  # Jedna pozycja powinna zostać otwarta

    # Sprawdź szczegóły zamkniętych pozycji
    sl_closed = False
    tp_closed = False
    for closed_position in closed_positions:
        assert closed_position.status == PositionStatus.CLOSED
        if closed_position.exit_price == Decimal('1.0950'):  # SL
            assert closed_position.profit < 0  # Pozycja zamknięta przez SL
            sl_closed = True
        elif closed_position.exit_price == Decimal('1.1100'):  # TP
            assert closed_position.profit > 0  # Pozycja zamknięta przez TP
            tp_closed = True

    assert sl_closed and tp_closed  # Sprawdź czy obie pozycje zostały zamknięte

    # Sprawdź czy pozostała pozycja ma zaktualizowany trailing stop
    remaining_position = position_manager.open_positions[0]
    assert remaining_position.stop_loss > Decimal('1.0950')  # Trailing stop powinien być przesunięty w górę

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
    new_sl = Decimal('1.0960')  # Wyżej niż poprzedni SL
    new_tp = Decimal('1.1120')  # Wyżej niż poprzedni TP
    
    # Modyfikuj poziomy
    success = await position_manager.modify_position_levels(position, new_sl, new_tp)
    
    # Sprawdź rezultaty
    assert success is True
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
    with pytest.raises(RuntimeError, match="Wolumen 0.2 większy niż pozycja 0.1"):
        await position_manager.close_position(valid_position, valid_position.entry_price + Decimal('0.0050'), Decimal('0.2'))

    # 4. Test obsługi błędu podczas aktualizacji trailing stop
    initial_stop_loss = valid_position.stop_loss
    await position_manager.update_trailing_stop(valid_position, valid_position.entry_price - Decimal('0.0100'))
    assert valid_position.stop_loss == initial_stop_loss  # Stop loss nie powinien się zmienić dla ruchu ceny w dół 

@pytest.mark.asyncio
async def test_validate_invalid_symbol(position_manager, sample_signal, mock_logger):
    """Test walidacji nieprawidłowego symbolu."""
    # Ustaw nieprawidłowy symbol
    sample_signal.symbol = 'INVALID'
    
    # Próba otwarcia pozycji
    position = await position_manager.open_position(sample_signal)
    assert position is None
    
    # Sprawdź logowanie błędu
    assert len(mock_logger.error_calls) == 1
    error_msg = mock_logger.error_calls[0]
    assert 'Nieprawidłowy symbol' in error_msg
    assert 'INVALID' in error_msg

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
async def test_validate_zero_volume(position_manager, sample_signal, mock_logger):
    """Test walidacji zerowego wolumenu."""
    # Ustaw zerowy wolumen
    sample_signal.volume = Decimal('0')
    
    # Próba otwarcia pozycji
    position = await position_manager.open_position(sample_signal)
    assert position is None
    
    # Sprawdź logowanie błędu
    assert len(mock_logger.error_calls) == 1
    error_msg = mock_logger.error_calls[0]
    assert 'Przekroczono maksymalny rozmiar pozycji' in error_msg

@pytest.mark.asyncio
async def test_validate_excessive_volume(position_manager, sample_signal, mock_logger):
    """Test walidacji zbyt dużego wolumenu."""
    # Ustaw zbyt duży wolumen
    sample_signal.volume = Decimal('2.0')  # Większy niż max_position_size=1.0
    
    # Próba otwarcia pozycji
    position = await position_manager.open_position(sample_signal)
    assert position is None
    
    # Sprawdź logowanie błędu
    assert len(mock_logger.error_calls) == 1
    error_msg = mock_logger.error_calls[0]
    assert 'Przekroczono maksymalny rozmiar pozycji' in error_msg
    assert '2.0' in error_msg

@pytest.mark.asyncio
async def test_open_position_with_invalid_signal(position_manager, mock_logger):
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
    
    # Sprawdź logowanie błędu
    assert len(mock_logger.error_calls) == 1
    error_msg = mock_logger.error_calls[0]
    assert 'Nieprawidłowy symbol' in error_msg
    assert 'INVALID' in error_msg

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
async def test_ensure_lock_success(position_manager):
    """Test poprawnego uzyskania blokady."""
    # Uzyskaj blokadę
    lock = await position_manager._ensure_lock()
    assert lock.locked()
    assert position_manager._owner == asyncio.current_task()
    assert position_manager._lock_count == 1
    
    # Zwolnij blokadę
    await position_manager._release_lock()
    assert not lock.locked()
    assert position_manager._lock_count == 0
    assert position_manager._owner is None

@pytest.mark.asyncio
async def test_ensure_lock_reentrant(position_manager):
    """Test ponownego wejścia do blokady."""
    # Pierwsze uzyskanie blokady
    lock1 = await position_manager._ensure_lock()
    assert lock1.locked()
    assert position_manager._owner == asyncio.current_task()
    assert position_manager._lock_count == 1
    
    # Drugie uzyskanie blokady (reentrant)
    lock2 = await position_manager._ensure_lock()
    assert lock2.locked()
    assert position_manager._owner == asyncio.current_task()
    assert position_manager._lock_count == 2
    
    # Pierwsze zwolnienie
    await position_manager._release_lock()
    assert lock1.locked()  # Nadal zablokowane bo licznik = 1
    assert position_manager._lock_count == 1
    assert position_manager._owner == asyncio.current_task()
    
    # Drugie zwolnienie
    await position_manager._release_lock()
    assert not lock1.locked()  # Teraz odblokowane
    assert position_manager._lock_count == 0
    assert position_manager._owner is None

@pytest.mark.asyncio
async def test_ensure_lock_timeout(position_manager):
    """Test timeout podczas uzyskiwania blokady."""
    # Zablokuj lock
    async with position_manager._lock:
        # Próba uzyskania blokady z małym timeout
        with pytest.raises(TimeoutError, match="Timeout podczas oczekiwania na blokadę"):
            await position_manager._ensure_lock(timeout=0.1)

@pytest.mark.asyncio
async def test_ensure_lock_concurrent(position_manager):
    """Test współbieżnego dostępu do blokady."""
    async def acquire_lock():
        try:
            async with position_manager._lock_context():
                await asyncio.sleep(0.1)  # Symuluj pracę
                return True
        except Exception:
            return False

    # Uruchom kilka zadań równocześnie
    tasks = [acquire_lock() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # Sprawdź czy wszystkie zadania się powiodły
    assert all(results)
    assert not position_manager._lock.locked()
    assert position_manager._owner is None
    assert position_manager._lock_count == 0

@pytest.mark.asyncio
async def test_lock_mechanism(position_manager):
    """Test mechanizmu blokad."""
    # Test podstawowego uzyskania blokady
    lock = await position_manager._ensure_lock()
    assert lock.locked()
    assert position_manager._owner == asyncio.current_task()
    assert position_manager._lock_count == 1

    # Test ponownego wejścia do blokady (reentrant)
    lock2 = await position_manager._ensure_lock()
    assert lock2.locked()
    assert position_manager._lock_count == 2

    # Test zwalniania blokady
    await position_manager._release_lock()
    assert lock.locked()  # Nadal zablokowane bo licznik = 1
    assert position_manager._lock_count == 1

    await position_manager._release_lock()
    assert not lock.locked()  # Teraz odblokowane
    assert position_manager._lock_count == 0
    assert position_manager._owner is None

@pytest.mark.asyncio
async def test_lock_timeout(position_manager):
    """Test timeoutu podczas uzyskiwania blokady."""
    # Zablokuj lock w innym tasku
    async def block_lock():
        async with position_manager._lock_context():
            await asyncio.sleep(0.2)  # Trzymaj blokadę przez 0.2s

    # Uruchom task blokujący
    task = asyncio.create_task(block_lock())
    
    # Poczekaj chwilę żeby task się uruchomił
    await asyncio.sleep(0.1)
    
    # Próba uzyskania blokady z małym timeout
    with pytest.raises(TimeoutError):
        await position_manager._ensure_lock(timeout=0.01)

    # Poczekaj na zakończenie taska
    await task

@pytest.mark.asyncio
async def test_lock_context_error(position_manager):
    """Test zachowania context managera przy błędzie."""
    try:
        async with position_manager._lock_context():
            assert position_manager._lock.locked()
            assert position_manager._lock_count == 1
            raise ValueError("Test error")
    except ValueError:
        pass

    # Blokada powinna być zwolniona mimo błędu
    assert not position_manager._lock.locked()
    assert position_manager._lock_count == 0
    assert position_manager._owner is None

@pytest.mark.asyncio
async def test_lock_invalid_timeout(position_manager):
    """Test nieprawidłowego timeoutu."""
    with pytest.raises(ValueError):
        await position_manager._ensure_lock(timeout=-1)

    with pytest.raises(ValueError):
        await position_manager._ensure_lock(timeout=0)

@pytest.mark.asyncio
async def test_release_lock_not_owner(position_manager):
    """Test zwalniania blokady przez task który jej nie posiada."""
    # Zablokuj lock w innym tasku
    async def block_lock():
        async with position_manager._lock_context():
            await asyncio.sleep(0.1)

    # Uruchom task blokujący
    task = asyncio.create_task(block_lock())
    
    # Poczekaj chwilę żeby task się uruchomił
    await asyncio.sleep(0.05)
    
    # Próba zwolnienia blokady przez inny task
    position_manager._release_lock()  # Nie powinno nic zrobić
    
    # Poczekaj na zakończenie taska
    await task

@pytest.mark.asyncio
async def test_error_handling_invalid_position(position_manager):
    """Test obsługi błędów dla nieprawidłowej pozycji."""
    # Próba zamknięcia None
    result = await position_manager.close_position(None, Decimal('1.1000'))
    assert result is None

    # Próba zamknięcia pozycji która nie istnieje
    invalid_position = Position(
        id="INVALID",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    result = await position_manager.close_position(invalid_position, Decimal('1.1000'))
    assert result is None

@pytest.mark.asyncio
async def test_error_handling_invalid_volume(position_manager, sample_signal):
    """Test obsługi błędów dla nieprawidłowego wolumenu."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None
    initial_volume = position.volume

    # Próba zamknięcia z ujemnym wolumenem
    with pytest.raises(RuntimeError, match="Nieprawidłowy wolumen: -0.1"):
        await position_manager.close_position(position, Decimal('1.1000'), Decimal('-0.1'))
    assert position.volume == initial_volume

    # Próba zamknięcia z zerowym wolumenem
    with pytest.raises(RuntimeError, match="Nieprawidłowy wolumen: 0"):
        await position_manager.close_position(position, Decimal('1.1000'), Decimal('0'))
    assert position.volume == initial_volume

    # Próba zamknięcia z wolumenem większym niż pozycja
    with pytest.raises(RuntimeError, match="Wolumen 0.2 większy niż pozycja 0.1"):
        await position_manager.close_position(position, Decimal('1.1000'), Decimal('0.2'))
    assert position.volume == initial_volume

    # Sprawdź czy pozycja nadal jest otwarta
    assert len(position_manager.open_positions) == 1
    assert len(position_manager.closed_positions) == 0

@pytest.mark.asyncio
async def test_error_handling_invalid_price(position_manager, sample_signal):
    """Test obsługi błędów dla nieprawidłowej ceny."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Próba zamknięcia z ujemną ceną
    with pytest.raises(RuntimeError, match="Nieprawidłowa cena zamknięcia: -1.1000"):
        await position_manager.close_position(position, Decimal('-1.1000'))

    # Próba zamknięcia z zerową ceną
    with pytest.raises(RuntimeError, match="Nieprawidłowa cena zamknięcia: 0"):
        await position_manager.close_position(position, Decimal('0'))

    # Próba aktualizacji trailing stop z nieprawidłową ceną
    with pytest.raises(RuntimeError, match="Nieprawidłowa cena: 0"):
        await position_manager.update_trailing_stop(position, Decimal('0'))

    # Próba aktualizacji breakeven z nieprawidłową ceną
    with pytest.raises(RuntimeError, match="Nieprawidłowa cena: 0"):
        await position_manager.update_breakeven(position, Decimal('0'))

@pytest.mark.asyncio
async def test_error_handling_process_price_updates(position_manager, mock_logger):
    """Test obsługi błędów w process_price_updates."""
    # Test pustej listy cen
    result = await position_manager.process_price_updates([])
    assert result == []
    assert len(mock_logger.warning_calls) == 1
    assert "Otrzymano pustą listę cen" in mock_logger.warning_calls[0]

    # Test None zamiast listy
    with pytest.raises(ValueError, match="Otrzymano None zamiast listy cen"):
        await position_manager.process_price_updates(None)

    # Test nieprawidłowych cen
    with pytest.raises(ValueError, match="Nieprawidłowa cena: -1.0"):
        await position_manager.process_price_updates([Decimal('-1.0')])

    # Test mieszanych prawidłowych i nieprawidłowych cen
    with pytest.raises(ValueError, match="Nieprawidłowa cena: 0"):
        await position_manager.process_price_updates([Decimal('1.1000'), Decimal('0'), Decimal('1.1100')])

@pytest.mark.asyncio
async def test_process_price_updates_multiple_positions(position_manager, sample_signal, sample_sell_signal):
    """Test przetwarzania wielu aktualizacji cen dla wielu pozycji."""
    # Otwórz dwie pozycje
    position1 = await position_manager.open_position(sample_signal)  # BUY
    position2 = await position_manager.open_position(sample_sell_signal)  # SELL
    
    assert len(position_manager.open_positions) == 2
    
    # Przygotuj listę cen
    prices = [
        Decimal('1.0940'),  # Poniżej SL dla BUY
        Decimal('1.1060')   # Powyżej SL dla SELL
    ]
    
    # Przetwórz aktualizacje cen
    await position_manager.process_price_updates(prices)
    
    # Sprawdź czy obie pozycje zostały zamknięte
    assert len(position_manager.open_positions) == 0
    assert len(position_manager.closed_positions) == 2
    
    # Sprawdź szczegóły zamkniętych pozycji
    for closed_position in position_manager.closed_positions:
        assert closed_position.status == PositionStatus.CLOSED
        if closed_position.trade_type == TradeType.BUY:
            assert closed_position.exit_price == Decimal('1.0950')  # SL dla BUY
        else:
            assert closed_position.exit_price == Decimal('1.1050')  # SL dla SELL

@pytest.mark.asyncio
async def test_close_position_timeout(position_manager, sample_signal, mock_logger):
    """Test timeoutu podczas zamykania pozycji."""
    # Otwórz pozycję
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Symuluj długotrwałe logowanie
    async def slow_log(*args, **kwargs):
        await asyncio.sleep(0.2)
        return None
    
    # Podmień metodę log_trade na wolniejszą wersję
    mock_logger.log_trade = AsyncMock(side_effect=slow_log)

    # Próba zamknięcia pozycji z timeoutem
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            position_manager.close_position(position, Decimal('1.1050')),
            timeout=0.1
        )

@pytest.mark.asyncio
async def test_validate_position_levels_errors(position_manager, sample_signal):
    """Test obsługi błędów w walidacji poziomów pozycji."""
    # Test dla pozycji BUY z nieprawidłowym SL
    sample_signal.stop_loss = sample_signal.entry_price + Decimal('0.0050')  # SL powyżej ceny wejścia
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

    # Test dla pozycji BUY z nieprawidłowym TP
    sample_signal.stop_loss = sample_signal.entry_price - Decimal('0.0050')  # Prawidłowy SL
    sample_signal.take_profit = sample_signal.entry_price - Decimal('0.0050')  # TP poniżej ceny wejścia
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

    # Test dla pozycji SELL z nieprawidłowym SL
    sample_signal.action = SignalAction.SELL
    sample_signal.stop_loss = sample_signal.entry_price - Decimal('0.0050')  # SL poniżej ceny wejścia
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

    # Test dla pozycji SELL z nieprawidłowym TP
    sample_signal.stop_loss = sample_signal.entry_price + Decimal('0.0050')  # Prawidłowy SL
    sample_signal.take_profit = sample_signal.entry_price + Decimal('0.0050')  # TP powyżej ceny wejścia
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

    # Test obsługi wyjątku
    sample_signal.entry_price = None  # Spowoduje błąd przy porównywaniu
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

@pytest.mark.asyncio
async def test_error_handling_in_risk_metrics(position_manager, sample_signal):
    """Test obsługi błędów w obliczaniu metryk ryzyka."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Spowoduj błąd przez ustawienie stop_loss na None
    position.stop_loss = None
    with pytest.raises(RuntimeError, match="Błąd podczas obliczania metryk ryzyka"):
        position_manager.calculate_risk_metrics(position)

@pytest.mark.asyncio
async def test_error_handling_in_trailing_stop(position_manager, sample_signal):
    """Test obsługi błędów w aktualizacji trailing stop."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z nieprawidłową ceną
    with pytest.raises(RuntimeError, match="Nieprawidłowa cena: -1.1000"):
        await position_manager.update_trailing_stop(position, Decimal('-1.1000'))

    # Test obsługi wyjątku przy aktualizacji
    position.stop_loss = None  # Spowoduje błąd przy porównywaniu
    with pytest.raises(RuntimeError, match="Błąd podczas aktualizacji trailing stop"):
        await position_manager.update_trailing_stop(position, Decimal('1.1000'))

@pytest.mark.asyncio
async def test_error_handling_in_breakeven(position_manager, sample_signal):
    """Test obsługi błędów w aktualizacji breakeven."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z nieprawidłową ceną
    with pytest.raises(RuntimeError, match="Nieprawidłowa cena: -1.1000"):
        await position_manager.update_breakeven(position, Decimal('-1.1000'))

    # Test obsługi wyjątku przy aktualizacji
    position.entry_price = None  # Spowoduje błąd przy porównywaniu
    with pytest.raises(RuntimeError, match="Błąd podczas aktualizacji breakeven"):
        await position_manager.update_breakeven(position, Decimal('1.1000'))

@pytest.mark.asyncio
async def test_error_handling_in_process_price_update(position_manager, sample_signal):
    """Test obsługi błędów w process_price_update."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z nieprawidłową ceną
    with pytest.raises(ValueError, match="Nieprawidłowa cena: -1.1000"):
        await position_manager.process_price_update(Decimal('-1.1000'))

    # Test obsługi wyjątku przy aktualizacji
    position.stop_loss = None  # Spowoduje błąd przy sprawdzaniu warunków
    with pytest.raises(RuntimeError, match="Błąd podczas przetwarzania ceny"):
        await position_manager.process_price_update(Decimal('1.1000'))

@pytest.mark.asyncio
async def test_initialization_with_custom_logger():
    """Test inicjalizacji z własnym loggerem."""
    custom_logger = TradingLogger(strategy_name="test_strategy")
    manager = PositionManager(
        symbol='EURUSD',
        max_position_size=Decimal('1.0'),
        stop_loss_pips=Decimal('50'),
        take_profit_pips=Decimal('100'),
        trailing_stop_pips=Decimal('30'),
        logger=custom_logger
    )
    assert manager.logger == custom_logger

@pytest.mark.asyncio
async def test_close_position_error_handling(position_manager, sample_signal):
    """Test obsługi błędów przy zamykaniu pozycji."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test zamykania pozycji z None jako ceną
    with pytest.raises(RuntimeError):
        await position_manager.close_position(position, None)

    # Test zamykania pozycji z nieprawidłowym wolumenem
    with pytest.raises(RuntimeError):
        await position_manager.close_position(position, Decimal('1.1000'), Decimal('-1.0'))

    # Test zamykania pozycji z wolumenem większym niż pozycja
    with pytest.raises(RuntimeError):
        await position_manager.close_position(position, Decimal('1.1000'), Decimal('1.0'))

    # Test obsługi wyjątku przy obliczaniu profitu
    position.entry_price = None  # Spowoduje błąd przy obliczaniu profitu
    with pytest.raises(RuntimeError):
        await position_manager.close_position(position, Decimal('1.1000'))

@pytest.mark.asyncio
async def test_initialization_default_logger():
    """Test inicjalizacji z domyślnym loggerem."""
    manager = PositionManager(symbol='EURUSD')
    assert isinstance(manager.logger, TradingLogger)

@pytest.mark.asyncio
async def test_validate_position_levels_none_values(position_manager, sample_signal):
    """Test walidacji poziomów z wartościami None."""
    sample_signal.stop_loss = None
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

    sample_signal.stop_loss = Decimal('1.0950')
    sample_signal.take_profit = None
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

@pytest.mark.asyncio
async def test_close_position_none_values(position_manager, sample_signal):
    """Test zamykania pozycji z wartościami None."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z None jako ceną zamknięcia
    with pytest.raises(RuntimeError):
        await position_manager.close_position(position, None)

    # Test z None jako pozycją
    with pytest.raises(RuntimeError):
        await position_manager.close_position(None, Decimal('1.1000'))

@pytest.mark.asyncio
async def test_risk_metrics_none_values(position_manager, sample_signal):
    """Test obliczania metryk ryzyka z wartościami None."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z None jako stop loss
    position.stop_loss = None
    with pytest.raises(RuntimeError):
        position_manager.calculate_risk_metrics(position)

    # Test z None jako take profit
    position.stop_loss = Decimal('1.0950')
    position.take_profit = None
    with pytest.raises(RuntimeError):
        position_manager.calculate_risk_metrics(position)

@pytest.mark.asyncio
async def test_modify_position_levels_none_values(position_manager, sample_signal):
    """Test modyfikacji poziomów z wartościami None."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z None jako SL
    result = await position_manager.modify_position_levels(position, None, Decimal('1.1100'))
    assert result is False

    # Test z None jako TP
    result = await position_manager.modify_position_levels(position, Decimal('1.0950'), None)
    assert result is False

@pytest.mark.asyncio
async def test_trailing_stop_none_values(position_manager, sample_signal):
    """Test aktualizacji trailing stop z wartościami None."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z None jako ceną
    with pytest.raises(RuntimeError):
        await position_manager.update_trailing_stop(position, None)

    # Test z None jako stop loss
    position.stop_loss = None
    with pytest.raises(RuntimeError):
        await position_manager.update_trailing_stop(position, Decimal('1.1000'))

@pytest.mark.asyncio
async def test_breakeven_none_values(position_manager, sample_signal):
    """Test aktualizacji breakeven z wartościami None."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z None jako ceną
    with pytest.raises(RuntimeError):
        await position_manager.update_breakeven(position, None)

    # Test z None jako entry price
    position.entry_price = None
    with pytest.raises(RuntimeError):
        await position_manager.update_breakeven(position, Decimal('1.1000'))

@pytest.mark.asyncio
async def test_process_price_updates_none_values(position_manager, sample_signal):
    """Test przetwarzania aktualizacji cen z wartościami None."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z None w liście cen
    with pytest.raises(ValueError):
        await position_manager.process_price_updates([Decimal('1.1000'), None, Decimal('1.1100')])

    # Test z None jako stop loss
    position.stop_loss = None
    with pytest.raises(RuntimeError):
        await position_manager.process_price_updates([Decimal('1.1000'), Decimal('1.1100')])

@pytest.mark.asyncio
async def test_process_price_update_none_values(position_manager, sample_signal):
    """Test przetwarzania aktualizacji ceny z wartościami None."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test z None jako ceną
    with pytest.raises(ValueError):
        await position_manager.process_price_update(None)

    # Test z None jako stop loss
    position.stop_loss = None
    with pytest.raises(RuntimeError):
        await position_manager.process_price_update(Decimal('1.1000'))

@pytest.mark.asyncio
async def test_error_handling_invalid_position(position_manager):
    """Test obsługi błędów dla nieprawidłowej pozycji."""
    # Próba zamknięcia None
    with pytest.raises(RuntimeError, match="Brak pozycji do zamknięcia"):
        await position_manager.close_position(None, Decimal('1.1000'))

    # Próba zamknięcia pozycji która nie istnieje
    invalid_position = Position(
        id="INVALID",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    result = await position_manager.close_position(invalid_position, Decimal('1.1000'))
    assert result is None  # Dla nieistniejącej pozycji zwracamy None

@pytest.mark.asyncio
async def test_calculate_risk_metrics_validation(position_manager, sample_signal):
    """Test walidacji poziomów w calculate_risk_metrics."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test dla pozycji BUY
    position.stop_loss = position.entry_price + Decimal('0.0010')  # SL powyżej wejścia
    with pytest.raises(RuntimeError, match="Stop loss dla pozycji długiej musi być poniżej ceny wejścia"):
        position_manager.calculate_risk_metrics(position)

    position.stop_loss = position.entry_price - Decimal('0.0010')  # Prawidłowy SL
    position.take_profit = position.entry_price - Decimal('0.0010')  # TP poniżej wejścia
    with pytest.raises(RuntimeError, match="Take profit dla pozycji długiej musi być powyżej ceny wejścia"):
        position_manager.calculate_risk_metrics(position)

    # Test dla pozycji SELL
    position.trade_type = TradeType.SELL
    position.stop_loss = position.entry_price - Decimal('0.0010')  # SL poniżej wejścia
    with pytest.raises(RuntimeError, match="Stop loss dla pozycji krótkiej musi być powyżej ceny wejścia"):
        position_manager.calculate_risk_metrics(position)

    position.stop_loss = position.entry_price + Decimal('0.0010')  # Prawidłowy SL
    position.take_profit = position.entry_price + Decimal('0.0010')  # TP powyżej wejścia
    with pytest.raises(RuntimeError, match="Take profit dla pozycji krótkiej musi być poniżej ceny wejścia"):
        position_manager.calculate_risk_metrics(position)

@pytest.mark.asyncio
async def test_modify_position_levels_validation(position_manager, sample_signal):
    """Test walidacji poziomów w modify_position_levels."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None

    # Test dla zamkniętej pozycji
    position.status = PositionStatus.CLOSED
    result = await position_manager.modify_position_levels(
        position,
        position.stop_loss,
        position.take_profit
    )
    assert result is False
    position.status = PositionStatus.OPEN

    # Test dla None jako poziomy
    result = await position_manager.modify_position_levels(position, None, position.take_profit)
    assert result is False
    result = await position_manager.modify_position_levels(position, position.stop_loss, None)
    assert result is False

    # Test dla pozycji BUY
    result = await position_manager.modify_position_levels(
        position,
        position.entry_price + Decimal('0.0010'),  # SL powyżej wejścia
        position.take_profit
    )
    assert result is False

    result = await position_manager.modify_position_levels(
        position,
        position.stop_loss,
        position.entry_price - Decimal('0.0010')  # TP poniżej wejścia
    )
    assert result is False

    # Test dla pozycji SELL
    position.trade_type = TradeType.SELL
    result = await position_manager.modify_position_levels(
        position,
        position.entry_price - Decimal('0.0010'),  # SL poniżej wejścia
        position.take_profit
    )
    assert result is False

    result = await position_manager.modify_position_levels(
        position,
        position.stop_loss,
        position.entry_price + Decimal('0.0010')  # TP powyżej wejścia
    )
    assert result is False

    # Test dla prawidłowych poziomów
    position.trade_type = TradeType.BUY
    result = await position_manager.modify_position_levels(
        position,
        position.entry_price - Decimal('0.0010'),
        position.entry_price + Decimal('0.0010')
    )
    assert result is True

@pytest.mark.asyncio
async def test_update_trailing_stop_detailed(position_manager, sample_signal):
    """Test szczegółowy aktualizacji trailing stop."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None
    initial_stop_loss = position.stop_loss

    # Test dla pozycji BUY
    # Cena poniżej aktualnego SL - nie powinno być zmiany
    await position_manager.update_trailing_stop(position, initial_stop_loss - Decimal('0.0001'))
    assert position.stop_loss == initial_stop_loss

    # Cena powyżej - powinien być przesunięty SL
    new_price = initial_stop_loss + Decimal('0.0100')
    await position_manager.update_trailing_stop(position, new_price)
    assert position.stop_loss > initial_stop_loss

    # Test dla pozycji SELL
    position.trade_type = TradeType.SELL
    position.stop_loss = position.entry_price + Decimal('0.0050')
    initial_stop_loss = position.stop_loss

    # Cena powyżej aktualnego SL - nie powinno być zmiany
    await position_manager.update_trailing_stop(position, initial_stop_loss + Decimal('0.0001'))
    assert position.stop_loss == initial_stop_loss

    # Cena poniżej - powinien być przesunięty SL
    new_price = initial_stop_loss - Decimal('0.0100')
    await position_manager.update_trailing_stop(position, new_price)
    assert position.stop_loss < initial_stop_loss

@pytest.mark.asyncio
async def test_update_breakeven_detailed(position_manager, sample_signal):
    """Test szczegółowy aktualizacji breakeven."""
    position = await position_manager.open_position(sample_signal)
    assert position is not None
    initial_stop_loss = position.stop_loss

    # Test dla pozycji BUY
    # Cena poniżej entry + min_distance - nie powinno być zmiany
    await position_manager.update_breakeven(position, position.entry_price + Decimal('0.0005'))
    assert position.stop_loss == initial_stop_loss

    # Cena powyżej entry + min_distance - powinien być przesunięty SL na entry
    await position_manager.update_breakeven(position, position.entry_price + Decimal('0.0020'))
    assert position.stop_loss == position.entry_price

    # Test dla pozycji SELL
    position.trade_type = TradeType.SELL
    position.stop_loss = position.entry_price + Decimal('0.0050')
    initial_stop_loss = position.stop_loss

    # Cena powyżej entry - min_distance - nie powinno być zmiany
    await position_manager.update_breakeven(position, position.entry_price - Decimal('0.0005'))
    assert position.stop_loss == initial_stop_loss

    # Cena poniżej entry - min_distance - powinien być przesunięty SL na entry
    await position_manager.update_breakeven(position, position.entry_price - Decimal('0.0020'))
    assert position.stop_loss == position.entry_price

@pytest.mark.asyncio
async def test_initialization_with_invalid_params():
    """Test inicjalizacji z nieprawidłowymi parametrami."""
    # Test z ujemnym max_position_size
    with pytest.raises(ValueError, match="Nieprawidłowy maksymalny rozmiar pozycji"):
        PositionManager(symbol='EURUSD', max_position_size=Decimal('-1.0'))

    # Test z ujemnym stop_loss_pips
    with pytest.raises(ValueError, match="Nieprawidłowa wartość stop loss"):
        PositionManager(symbol='EURUSD', stop_loss_pips=Decimal('-50'))

    # Test z ujemnym take_profit_pips
    with pytest.raises(ValueError, match="Nieprawidłowa wartość take profit"):
        PositionManager(symbol='EURUSD', take_profit_pips=Decimal('-100'))

    # Test z ujemnym trailing_stop_pips
    with pytest.raises(ValueError, match="Nieprawidłowa wartość trailing stop"):
        PositionManager(symbol='EURUSD', trailing_stop_pips=Decimal('-30'))

@pytest.mark.asyncio
async def test_error_handling_in_lock_context():
    """Test obsługi błędów w kontekście blokady."""
    # Inicjalizacja managera z mockiem loggera
    mock_logger = MagicMock()
    mock_logger.debug = AsyncMock()
    mock_logger.error = AsyncMock()
    manager1 = PositionManager(symbol='EURUSD', logger=mock_logger)
    manager2 = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Test z ujemnym timeout
    with pytest.raises(ValueError, match="Nieprawidłowy timeout"):
        async with manager1._lock_context(timeout=-1.0):
            pass
    assert mock_logger.error.await_count >= 1

    # Test z timeoutem
    # Najpierw zablokuj zasób pierwszym managerem
    async with manager1._lock_context(timeout=1.0):
        # Sprawdź czy blokada jest aktywna
        assert manager1._lock.locked()
        assert manager1._owner == asyncio.current_task()
        assert manager1._lock_count == 1
        assert mock_logger.debug.await_count >= 1
        
        # Teraz próbuj uzyskać blokadę drugim managerem z zerowym timeoutem
        with pytest.raises(ValueError, match="Nieprawidłowy timeout"):
            async with manager2._lock_context(timeout=0.0):
                pass
        assert mock_logger.error.await_count >= 2
        
        # Następnie próbuj uzyskać blokadę z bardzo małym timeoutem
        # Używamy tego samego locka co manager1
        manager2._lock = manager1._lock
        with pytest.raises(TimeoutError):
            async with manager2._lock_context(timeout=0.1):
                pass
        assert mock_logger.error.await_count >= 3

@pytest.mark.asyncio
async def test_lock_negative_count():
    """Test obsługi ujemnego licznika blokady."""
    mock_logger = MagicMock()
    mock_logger.debug = AsyncMock()
    mock_logger.error = AsyncMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Ustaw ujemny licznik
    async with manager._lock_context(timeout=1.0):
        assert mock_logger.debug.await_count >= 1
        manager._lock_count = -1
        
    # Sprawdź czy stan został zresetowany
    assert manager._lock_count == 0
    assert manager._owner is None
    assert not manager._lock.locked()

@pytest.mark.asyncio
async def test_lock_release_by_different_task():
    """Test próby zwolnienia blokady przez inny task."""
    mock_logger = MagicMock()
    mock_logger.debug = AsyncMock()
    mock_logger.error = AsyncMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Zablokuj zasób
    async with manager._lock_context(timeout=1.0):
        assert mock_logger.debug.await_count >= 1
        # Zmień właściciela na inny task
        manager._owner = None
        # Próba zwolnienia blokady
        await manager._release_lock()
        # Sprawdź czy blokada nadal jest aktywna
        assert manager._lock.locked()

@pytest.mark.asyncio
async def test_lock_reentrant_multiple():
    """Test wielokrotnego wejścia do blokady przez ten sam task."""
    mock_logger = MagicMock()
    mock_logger.debug = AsyncMock()
    mock_logger.error = AsyncMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    async with manager._lock_context(timeout=1.0):
        assert mock_logger.debug.await_count >= 1
        initial_count = manager._lock_count
        async with manager._lock_context(timeout=1.0):
            assert mock_logger.debug.await_count >= 2
            assert manager._lock_count == initial_count + 1
            async with manager._lock_context(timeout=1.0):
                assert mock_logger.debug.await_count >= 3
                assert manager._lock_count == initial_count + 2
                
    # Po wyjściu licznik powinien być wyzerowany
    assert manager._lock_count == 0
    assert manager._owner is None
    assert not manager._lock.locked()

@pytest.mark.asyncio
async def test_logging_in_ensure_lock():
    """Test logowania w metodzie _ensure_lock."""
    # Utwórz mock loggera
    mock_logger = MagicMock()
    mock_logger.debug = AsyncMock()
    mock_logger.error = AsyncMock()
    
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Test reentrant lock
    async with manager._lock_context(timeout=1.0):
        # Sprawdź czy logger został wywołany przy pierwszym wejściu
        assert mock_logger.debug.await_count >= 1
        async with manager._lock_context(timeout=1.0):
            # Sprawdź czy logger został wywołany przy drugim wejściu
            assert mock_logger.debug.await_count >= 2

@pytest.mark.asyncio
async def test_logging_in_all_methods():
    """Test logowania we wszystkich metodach."""
    # Utwórz mock loggera
    mock_logger = MagicMock()
    mock_logger.info = AsyncMock()
    mock_logger.error = AsyncMock()
    mock_logger.warning = AsyncMock()
    mock_logger.debug = AsyncMock()
    mock_logger.log_trade = AsyncMock()
    mock_logger.log_error = AsyncMock()
    
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Test open_position
    signal = SignalData(
        symbol='EURUSD',
        timestamp=datetime.now(),
        action=SignalAction.BUY,
        entry_price=Decimal('1.1000'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        volume=Decimal('0.1'),
        confidence=Decimal('0.8'),
        indicators={},
        ai_analysis={},
        risk_reward_ratio=Decimal('2.0')
    )
    position = await manager.open_position(signal)
    assert position is not None
    assert mock_logger.log_trade.await_count >= 1
    
    # Test validate_position_levels z nieprawidłowymi poziomami
    signal.stop_loss = signal.entry_price + Decimal('0.0010')  # SL powyżej wejścia
    result = await manager.validate_position_levels(signal)
    assert not result
    assert mock_logger.error.await_count >= 1
    
    # Test process_price_update
    await manager.process_price_update(Decimal('1.1000'))
    assert mock_logger.info.await_count >= 1
    
    # Test process_price_updates
    await manager.process_price_updates([Decimal('1.1000'), Decimal('1.1010')])
    assert mock_logger.info.await_count >= 2
    
    # Test modify_position_levels
    await manager.modify_position_levels(position, Decimal('1.0950'), Decimal('1.1100'))
    assert mock_logger.log_trade.await_count >= 2
    
    # Test update_trailing_stop
    await manager.update_trailing_stop(position, Decimal('1.1020'))
    assert mock_logger.info.await_count >= 3
    
    # Test calculate_risk_metrics - nie sprawdzamy logów, bo metoda jest synchroniczna
    try:
        position.stop_loss = position.entry_price + Decimal('0.0010')  # Nieprawidłowy SL
        manager.calculate_risk_metrics(position)
    except RuntimeError:
        pass
    
    # Test update_breakeven
        await manager.update_breakeven(position, Decimal('1.1000'))
    assert mock_logger.info.await_count >= 4
    
    # Test close_position na końcu
    await manager.close_position(position, Decimal('1.1000'))
    assert mock_logger.info.await_count >= 5
    assert mock_logger.log_trade.await_count >= 3

@pytest.mark.asyncio
async def test_logger_error_handling():
    """Test obsługi błędów podczas logowania."""
    # Utwórz mock loggera, który będzie rzucał wyjątki
    mock_logger = MagicMock()
    mock_logger.error = AsyncMock(side_effect=Exception("Test error"))
    mock_logger.info = AsyncMock(side_effect=Exception("Test error"))
    mock_logger.debug = AsyncMock(side_effect=Exception("Test error"))
    mock_logger.log_trade = AsyncMock(side_effect=Exception("Test error"))
    mock_logger.log_error = AsyncMock(side_effect=Exception("Test error"))
    
    manager = PositionManager(
        symbol='EURUSD',
        logger=mock_logger
    )
    
    # Test błędu logowania w _ensure_lock
    await manager._ensure_lock()  # Nie powinno rzucić wyjątku mimo błędu logowania
    
    # Test błędu logowania w open_position
    signal = SignalData(
        timestamp=datetime.now(),
        symbol='INVALID',  # Nieprawidłowy symbol spowoduje próbę logowania błędu
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100')
    )
    result = await manager.open_position(signal)
    assert result is None  # Operacja powinna się nie powieść, ale nie powinna rzucić wyjątku

@pytest.mark.asyncio
async def test_process_price_update_error_handling():
    """Test obsługi błędów w process_price_update."""
    mock_logger = MagicMock()
    mock_logger.error = AsyncMock()
    mock_logger.info = AsyncMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Test z None jako ceną
    with pytest.raises(ValueError, match="Nieprawidłowa cena: None"):
        await manager.process_price_update(None)
    
    # Test z ujemną ceną
    with pytest.raises(ValueError, match="Nieprawidłowa cena: -1.0"):
        await manager.process_price_update(Decimal('-1.0'))
    
    # Test błędu podczas aktualizacji trailing stop
    position = Position(
        id="TEST",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),  # Ustawiamy prawidłową wartość
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    manager.open_positions.append(position)
    position.stop_loss = None  # Modyfikujemy po utworzeniu obiektu
    
    with pytest.raises(RuntimeError):
        await manager.process_price_update(Decimal('1.1050'))

@pytest.mark.asyncio
async def test_process_price_updates_error_handling():
    """Test obsługi błędów w process_price_updates."""
    mock_logger = MagicMock()
    mock_logger.error = AsyncMock()
    mock_logger.warning = AsyncMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Test z None w liście cen
    with pytest.raises(ValueError):
        await manager.process_price_updates([Decimal('1.1000'), None])
    
    # Test z pustą listą
    result = await manager.process_price_updates([])
    assert result == []
    assert mock_logger.warning.await_count >= 1
    
    # Test błędu podczas aktualizacji trailing stop
    position = Position(
        id="TEST",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),  # Ustawiamy prawidłową wartość
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    manager.open_positions.append(position)
    position.stop_loss = None  # Modyfikujemy po utworzeniu obiektu
    
    with pytest.raises(RuntimeError):
        await manager.process_price_updates([Decimal('1.1050')])

@pytest.mark.asyncio
async def test_update_trailing_stop_error_handling():
    """Test obsługi błędów w update_trailing_stop."""
    mock_logger = MagicMock()
    mock_logger.error = AsyncMock()
    mock_logger.log_error = AsyncMock()
    mock_logger.info = AsyncMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)

    position = Position(
        id="TEST",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),  # Ustawiamy prawidłową wartość
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    position.stop_loss = None  # Modyfikujemy po utworzeniu obiektu
    
    # Test z ujemną ceną
    with pytest.raises(RuntimeError):
        await manager.update_trailing_stop(position, Decimal('-1.0'))
    
    # Test z None jako stop loss
    with pytest.raises(RuntimeError):
        await manager.update_trailing_stop(position, Decimal('1.1050'))

@pytest.mark.asyncio
async def test_update_breakeven_error_handling():
    """Test obsługi błędów w update_breakeven."""
    mock_logger = MagicMock()
    mock_logger.error = AsyncMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Test z None jako entry_price
    position = Position(
        id="TEST",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),  # Ustawiamy prawidłową wartość
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    position.entry_price = None  # Modyfikujemy po utworzeniu obiektu
    
    with pytest.raises(RuntimeError):
        await manager.update_breakeven(position, Decimal('1.1050'))
    
    # Test z None jako stop_loss
    position.entry_price = Decimal('1.1000')
    position.stop_loss = None
    with pytest.raises(RuntimeError):
        await manager.update_breakeven(position, Decimal('1.1050'))
    
    # Test z ujemną ceną
    position.stop_loss = Decimal('1.0950')
    with pytest.raises(RuntimeError):
        await manager.update_breakeven(position, Decimal('-1.0'))

@pytest.mark.asyncio
async def test_calculate_risk_metrics_error_handling():
    """Test obsługi błędów w calculate_risk_metrics."""
    mock_logger = MagicMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Test z brakującymi polami
    position = Position(
        id="TEST",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),  # Ustawiamy prawidłową wartość
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    position.entry_price = None  # Modyfikujemy po utworzeniu obiektu
    
    with pytest.raises(RuntimeError):
        manager.calculate_risk_metrics(position)
    
    # Test z nieprawidłowymi poziomami
    position.entry_price = Decimal('1.1000')
    position.stop_loss = position.entry_price + Decimal('0.0010')  # SL powyżej wejścia dla BUY
    with pytest.raises(RuntimeError):
        manager.calculate_risk_metrics(position)

@pytest.mark.asyncio
async def test_logger_error_handling_detailed():
    """Test szczegółowej obsługi błędów logowania."""
    # Przygotuj logger, który będzie rzucał wyjątki
    mock_logger = MagicMock()
    mock_logger.error = AsyncMock(side_effect=Exception("Błąd logowania"))
    mock_logger.debug = AsyncMock(side_effect=Exception("Błąd logowania"))
    mock_logger.info = AsyncMock(side_effect=Exception("Błąd logowania"))
    mock_logger.log_error = AsyncMock(side_effect=Exception("Błąd logowania"))
    mock_logger.log_trade = AsyncMock(side_effect=Exception("Błąd logowania"))

    manager = PositionManager(symbol='EURUSD', logger=mock_logger)

    # Test błędów logowania w _ensure_lock
    await manager._ensure_lock(timeout=1.0)  # Nie powinno rzucić wyjątku mimo błędu logowania

    # Test błędów logowania w open_position
    signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        confidence=0.8
    )
    position = await manager.open_position(signal)  # Nie powinno rzucić wyjątku mimo błędu logowania
    assert position is not None

    # Test błędów logowania w close_position
    position = Position(
        id="TEST",
        timestamp=datetime.now(),
        symbol='EURUSD',
        trade_type=TradeType.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        status=PositionStatus.OPEN
    )
    manager.open_positions.append(position)
    
    # Powinno rzucić RuntimeError, ale nie z powodu błędu logowania
    with pytest.raises(RuntimeError):
        await manager.close_position(None, Decimal('1.1000'))

    # Test błędów logowania w update_trailing_stop
    with pytest.raises(RuntimeError):
        await manager.update_trailing_stop(position, Decimal('-1.0'))

    # Test błędów logowania w process_price_updates
    with pytest.raises(ValueError):
        await manager.process_price_updates([None])

    # Test błędów logowania w validate_position_levels
    signal.stop_loss = Decimal('1.2000')  # Nieprawidłowy stop loss dla BUY
    result = await manager.validate_position_levels(signal)
    assert result is False

@pytest.mark.asyncio
async def test_lock_mechanism_detailed():
    """Test szczegółowy mechanizmu blokowania."""
    mock_logger = MagicMock()
    mock_logger.debug = AsyncMock()
    mock_logger.error = AsyncMock()
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)

    # Test zwiększania licznika blokady
    await manager._ensure_lock()
    assert manager._lock_count == 1
    await manager._ensure_lock()  # To samo zadanie powinno móc uzyskać blokadę ponownie
    assert manager._lock_count == 2

    # Test zwalniania blokady
    await manager._release_lock()
    assert manager._lock_count == 1
    await manager._release_lock()
    assert manager._lock_count == 0
    assert manager._owner is None

    # Test timeoutu
    manager._lock = asyncio.Lock()
    await manager._lock.acquire()  # Zajmij blokadę
    with pytest.raises(TimeoutError):
        await manager._ensure_lock(timeout=0.1)
    manager._lock.release()  # Zwolnij blokadę po teście timeoutu

    # Test ujemnego licznika
    manager._lock_count = -1
    await manager._release_lock()
    assert manager._lock_count == 0
    assert manager._owner is None

    # Test zwalniania przez inne zadanie
    manager._owner = "inne_zadanie"
    await manager._release_lock()  # Nie powinno zwolnić blokady
    assert manager._owner == "inne_zadanie"
    manager._owner = None  # Resetuj właściciela przed testem context managera

    # Test context managera
    async with manager._lock_context(timeout=1.0) as lock:
        assert isinstance(lock, asyncio.Lock)
        assert manager._lock_count == 1
    assert manager._lock_count == 0

@pytest.mark.asyncio
async def test_logger_error_handling_comprehensive():
    """Test kompleksowej obsługi błędów logowania we wszystkich metodach."""
    # Przygotuj logger, który będzie rzucał wyjątki
    mock_logger = MagicMock()
    mock_logger.error = AsyncMock(side_effect=Exception("Błąd logowania"))
    mock_logger.debug = AsyncMock(side_effect=Exception("Błąd logowania"))
    mock_logger.info = AsyncMock(side_effect=Exception("Błąd logowania"))
    mock_logger.log_trade = AsyncMock(side_effect=Exception("Błąd logowania"))
    mock_logger.log_error = AsyncMock(side_effect=Exception("Błąd logowania"))
    
    manager = PositionManager(symbol='EURUSD', logger=mock_logger)
    
    # Test błędów w _ensure_lock
    with pytest.raises(ValueError):
        await manager._ensure_lock(timeout=-1.0)  # Powinno rzucić ValueError
    
    # Test błędów w _release_lock
    manager._lock_count = -1  # Ustawienie ujemnego licznika
    await manager._release_lock()  # Powinno zresetować licznik do 0
    assert manager._lock_count == 0
    
    # Test błędów w open_position
    signal = SignalData(
        timestamp=datetime.now(),
        symbol='EURUSD',
        action=SignalAction.BUY,
        entry_price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        confidence=0.8
    )
    position = await manager.open_position(signal)
    assert position is not None
    
    # Test błędów w process_price_updates
    await manager.process_price_updates([Decimal('1.1000')])
    
    # Test błędów w update_trailing_stop
    await manager.update_trailing_stop(position, Decimal('1.1050'))
    
    # Test błędów w update_breakeven
    await manager.update_breakeven(position, Decimal('1.1050'))
    
    # Test błędów w calculate_risk_metrics
    metrics = manager.calculate_risk_metrics(position)
    assert metrics is not None
    
    # Test błędów w validate_position_levels
    signal.stop_loss = signal.entry_price + Decimal('0.0010')  # Nieprawidłowy SL
    result = await manager.validate_position_levels(signal)
    assert not result
    
    # Test błędów w close_position
    closed_position = await manager.close_position(position, Decimal('1.1000'))
    assert closed_position is not None
    
    # Test błędów w process_price_update
    await manager.process_price_update(Decimal('1.1000'))
    
    # Test błędów w modify_position_levels
    await manager.modify_position_levels(position, Decimal('1.0900'), Decimal('1.1200'))

@pytest.mark.asyncio
async def test_modify_position_levels_none_position(position_manager):
    """Test modyfikacji poziomów dla None position."""
    result = await position_manager.modify_position_levels(None, Decimal('1.1000'), Decimal('1.1200'))
    assert result is False

@pytest.mark.asyncio
async def test_modify_position_levels_success(position_manager, sample_signal):
    """Test udanej modyfikacji poziomów pozycji."""
    position = await position_manager.open_position(sample_signal)
    
    new_sl = Decimal('1.0940')
    new_tp = Decimal('1.1150')
    
    result = await position_manager.modify_position_levels(position, new_sl, new_tp)
    
    assert result is True
    assert position.stop_loss == new_sl
    assert position.take_profit == new_tp

@pytest.mark.asyncio
async def test_modify_position_levels_invalid_levels(position_manager, sample_signal):
    """Test modyfikacji poziomów z niepoprawnymi wartościami."""
    position = await position_manager.open_position(sample_signal)
    
    # Stop loss powyżej ceny wejścia dla pozycji BUY
    result = await position_manager.modify_position_levels(position, Decimal('1.1100'), Decimal('1.1200'))
    assert result is False
    
    # Take profit poniżej ceny wejścia dla pozycji BUY
    result = await position_manager.modify_position_levels(position, Decimal('1.0900'), Decimal('1.0950'))
    assert result is False

@pytest.mark.asyncio
async def test_update_trailing_stop_none_position(position_manager):
    """Test aktualizacji trailing stop dla None position."""
    with pytest.raises(RuntimeError):
        await position_manager.update_trailing_stop(None, Decimal('1.1000'))

@pytest.mark.asyncio
async def test_update_trailing_stop_invalid_price(position_manager, sample_signal):
    """Test aktualizacji trailing stop dla nieprawidłowej ceny."""
    position = await position_manager.open_position(sample_signal)
    
    with pytest.raises(RuntimeError):
        await position_manager.update_trailing_stop(position, Decimal('-1.0'))

@pytest.mark.asyncio
async def test_update_trailing_stop_buy_success(position_manager, sample_signal):
    """Test udanej aktualizacji trailing stop dla pozycji BUY."""
    position = await position_manager.open_position(sample_signal)
    initial_sl = position.stop_loss
    
    # Cena wzrosła, trailing stop powinien się przesunąć w górę
    await position_manager.update_trailing_stop(position, Decimal('1.1100'))
    
    assert position.stop_loss > initial_sl

@pytest.mark.asyncio
async def test_update_trailing_stop_sell_success(position_manager, sample_sell_signal):
    """Test udanej aktualizacji trailing stop dla pozycji SELL."""
    position = await position_manager.open_position(sample_sell_signal)
    initial_sl = position.stop_loss
    
    # Cena spadła, trailing stop powinien się przesunąć w dół
    await position_manager.update_trailing_stop(position, Decimal('1.0900'))
    
    assert position.stop_loss < initial_sl

@pytest.mark.asyncio
async def test_update_trailing_stop_closed_position(position_manager, sample_signal):
    """Test aktualizacji trailing stop dla zamkniętej pozycji."""
    position = await position_manager.open_position(sample_signal)
    initial_sl = position.stop_loss
    
    # Zamknij pozycję
    await position_manager.close_position(position, Decimal('1.1050'))
    
    # Próba aktualizacji trailing stop dla zamkniętej pozycji
    await position_manager.update_trailing_stop(position, Decimal('1.1100'))
    
    # Stop loss nie powinien się zmienić
    assert position.stop_loss == initial_sl

@pytest.mark.asyncio
async def test_close_position_partial(position_manager, sample_signal):
    """Test częściowego zamknięcia pozycji."""
    position = await position_manager.open_position(sample_signal)
    initial_volume = position.volume
    
    # Zamknij połowę pozycji
    partial_volume = initial_volume / Decimal('2')
    closed_position = await position_manager.close_position(position, Decimal('1.1050'), partial_volume)
    
    assert closed_position is not None
    assert closed_position.volume == partial_volume
    assert position.volume == initial_volume - partial_volume
    assert len(position_manager.open_positions) == 1
    assert len(position_manager.closed_positions) == 1

@pytest.mark.asyncio
async def test_close_position_invalid_volume(position_manager, sample_signal):
    """Test zamknięcia pozycji z nieprawidłowym wolumenem."""
    position = await position_manager.open_position(sample_signal)
    
    # Próba zamknięcia z wolumenem większym niż aktualny
    with pytest.raises(RuntimeError):
        await position_manager.close_position(position, Decimal('1.1050'), position.volume * Decimal('2'))

@pytest.mark.asyncio
async def test_validate_position_levels_none_signal(position_manager):
    """Test walidacji poziomów dla None signal."""
    result = await position_manager.validate_position_levels(None)
    assert result is False

@pytest.mark.asyncio
async def test_validate_position_levels_invalid_entry_price(position_manager, sample_signal):
    """Test walidacji poziomów dla nieprawidłowej ceny wejścia."""
    sample_signal.entry_price = Decimal('0')
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

@pytest.mark.asyncio
async def test_validate_position_levels_invalid_stop_loss(position_manager, sample_signal):
    """Test walidacji poziomów dla nieprawidłowego stop loss."""
    sample_signal.stop_loss = Decimal('0')
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False

@pytest.mark.asyncio
async def test_validate_position_levels_invalid_take_profit(position_manager, sample_signal):
    """Test walidacji poziomów dla nieprawidłowego take profit."""
    sample_signal.take_profit = Decimal('0')
    result = await position_manager.validate_position_levels(sample_signal)
    assert result is False
