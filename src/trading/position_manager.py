"""
Moduł odpowiedzialny za zarządzanie pozycjami tradingowymi.
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any, Union, Callable, AsyncGenerator
import asyncio
from dataclasses import asdict
from contextlib import asynccontextmanager
import time

from src.models.data_models import Position, SignalData
from src.utils.logger import TradingLogger
from src.models.enums import TradeType, PositionStatus, SignalAction

class PositionManager:
    """Klasa zarządzająca pozycjami tradingowymi."""

    def __init__(
        self,
        symbol: str,
        max_position_size: Decimal = Decimal('1.0'),
        stop_loss_pips: Decimal = Decimal('50'),
        take_profit_pips: Decimal = Decimal('100'),
        trailing_stop_pips: Decimal = Decimal('30'),
        logger: Optional[TradingLogger] = None
    ) -> None:
        """
        Inicjalizacja managera pozycji.

        Args:
            symbol: Symbol instrumentu
            max_position_size: Maksymalny rozmiar pozycji
            stop_loss_pips: Domyślny stop loss w pipsach
            take_profit_pips: Domyślny take profit w pipsach
            trailing_stop_pips: Wartość trailing stopu w pipsach
            logger: Logger do zapisywania operacji

        Raises:
            ValueError: Gdy któryś z parametrów jest nieprawidłowy
        """
        # Walidacja parametrów
        if max_position_size <= Decimal('0'):
            raise ValueError("Nieprawidłowy maksymalny rozmiar pozycji")
        if stop_loss_pips <= Decimal('0'):
            raise ValueError("Nieprawidłowa wartość stop loss")
        if take_profit_pips <= Decimal('0'):
            raise ValueError("Nieprawidłowa wartość take profit")
        if trailing_stop_pips <= Decimal('0'):
            raise ValueError("Nieprawidłowa wartość trailing stop")

        self.symbol = symbol
        self.max_position_size = max_position_size
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.trailing_stop_pips = trailing_stop_pips
        self.logger = logger or TradingLogger(strategy_name="position_manager")
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self._lock = asyncio.Lock()
        self._owner = None
        self._lock_count = 0
        self._positions: Dict[str, Position] = {}
        self.trailing_stop_pips = Decimal('10')  # 10 pipsów
        self.breakeven_pips = Decimal('10')  # 10 pipsów

    async def _ensure_lock(self, timeout: float = 1.0) -> asyncio.Lock:
        """
        Próbuje uzyskać blokadę z timeoutem.

        Args:
            timeout (float): Maksymalny czas oczekiwania na blokadę w sekundach

        Returns:
            asyncio.Lock: Obiekt blokady

        Raises:
            ValueError: Gdy timeout jest nieprawidłowy
            TimeoutError: Gdy nie udało się uzyskać blokady w zadanym czasie
        """
        if timeout <= 0:
            error_msg = f"Nieprawidłowy timeout: {timeout}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise ValueError(error_msg)

        try:
            current_task = asyncio.current_task()
            
            # Jeśli bieżące zadanie już posiada blokadę, zwiększ licznik
            if self._owner == current_task:
                self._lock_count += 1
                try:
                    if self.logger:
                        await self.logger.debug(f"🔒 Zwiększono licznik blokady do {self._lock_count}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return self._lock

            # Sprawdź czy blokada jest już zajęta
            if self._lock.locked():
                # Próba uzyskania blokady z timeoutem
                try:
                    await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
                except asyncio.TimeoutError:
                    error_msg = f"Timeout podczas oczekiwania na blokadę: {timeout}s"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise TimeoutError(error_msg)
            else:
                # Blokada jest wolna, spróbuj ją uzyskać
                await self._lock.acquire()
            
            # Ustaw właściciela i licznik
            self._owner = current_task
            self._lock_count = 1
            try:
                if self.logger:
                    await self.logger.debug("🔒 Uzyskano nową blokadę")
            except Exception:
                pass  # Ignoruj błędy logowania
            
            return self._lock

        except TimeoutError as e:
            raise  # Przekaż dalej wyjątek TimeoutError
        except Exception as e:
            error_msg = f"Błąd podczas uzyskiwania blokady: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    async def _release_lock(self) -> None:
        """
        Zwalnia blokadę.
        """
        try:
            # Najpierw sprawdź czy licznik jest ujemny
            if self._lock_count <= 0:
                try:
                    if self.logger:
                        await self.logger.error("❌ Licznik blokady jest ujemny lub 0 - resetuję do 0")
                except Exception:
                    pass  # Ignoruj błędy logowania
                self._lock_count = 0
                if self._owner == asyncio.current_task():
                    self._owner = None
                    if self._lock.locked():
                        self._lock.release()
                return

            if self._owner != asyncio.current_task():
                try:
                    if self.logger:
                        await self.logger.debug(f"🔓 Próba zwolnienia blokady przez niewłaściciela {asyncio.current_task()}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return
            
            self._lock_count -= 1
            
            if self._lock_count == 0:
                self._owner = None
                if self._lock.locked():
                    self._lock.release()
                    try:
                        if self.logger:
                            await self.logger.debug("🔓 Blokada zwolniona")
                    except Exception:
                        pass  # Ignoruj błędy logowania

        except Exception as e:
            try:
                if self.logger:
                    await self.logger.error(f"❌ Błąd podczas zwalniania blokady: {str(e)}")
                    await self.logger.log_error(e)
            except Exception:
                pass  # Ignoruj błędy logowania
            # Resetuj stan w przypadku błędu
            self._lock_count = 0
            self._owner = None
            if self._lock.locked():
                self._lock.release()

    @asynccontextmanager
    async def _lock_context(self, timeout: float = 1.0) -> AsyncGenerator[asyncio.Lock, None]:
        """
        Context manager do bezpiecznego zarządzania blokadą.

        Args:
            timeout (float): Maksymalny czas oczekiwania na blokadę w sekundach

        Returns:
            AsyncGenerator[asyncio.Lock, None]: Generator zwracający obiekt blokady

        Raises:
            ValueError: Gdy timeout jest nieprawidłowy
            TimeoutError: Gdy nie udało się uzyskać blokady w zadanym czasie
        """
        try:
            lock = await self._ensure_lock(timeout)
            yield lock
        finally:
            await self._release_lock()

    async def open_position(self, signal: SignalData) -> Optional[Position]:
        """
        Otwiera nową pozycję na podstawie sygnału.

        Args:
            signal: Sygnał tradingowy

        Returns:
            Utworzona pozycja lub None w przypadku błędu

        Raises:
            RuntimeError: Gdy wystąpi błąd podczas otwierania pozycji
        """
        try:
            # Walidacja symbolu
            if signal.symbol != self.symbol:
                error_msg = f"Nieprawidłowy symbol: {signal.symbol}, oczekiwano: {self.symbol}"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return None

            # Walidacja wolumenu
            if signal.volume <= Decimal('0'):
                error_msg = f"Nieprawidłowy wolumen: {signal.volume}"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return None

            # Sprawdź czy nie przekraczamy maksymalnego rozmiaru pozycji
            if not self.validate_position_size(signal.volume):
                return None

            # Walidacja poziomów
            if not await self.validate_position_levels(signal):
                return None

            # Generuj unikalny identyfikator pozycji
            position_id = f"{signal.symbol}_{int(time.time() * 1000)}_{len(self._positions) + 1}"

            # Utwórz nową pozycję
            position = Position(
                id=position_id,
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                trade_type=TradeType.BUY if signal.action == SignalAction.BUY else TradeType.SELL,
                volume=signal.volume,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status=PositionStatus.OPEN
            )
            
            # Dodaj pozycję do słownika i listy otwartych pozycji
            self._positions[position.id] = position
            self.open_positions.append(position)
            
            try:
                if self.logger:
                    await self.logger.info(f"🔓 Otwarto pozycję {position.id}")
                await self.logger.log_trade(position, "OPEN")
            except Exception:
                pass  # Ignoruj błędy logowania
            
            return position

        except Exception as e:
            error_msg = f"Błąd podczas otwierania pozycji: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg)

    async def validate_position_levels(self, signal: SignalData) -> bool:
        """
        Sprawdza poprawność poziomów stop loss i take profit dla sygnału.

        Args:
            signal: Sygnał do sprawdzenia

        Returns:
            bool: True jeśli poziomy są prawidłowe, False w przeciwnym razie
        """
        try:
            if signal.stop_loss is None or signal.take_profit is None:
                error_msg = "Brak wymaganych poziomów SL/TP"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False

            if signal.stop_loss <= Decimal('0') or signal.take_profit <= Decimal('0'):
                error_msg = f"Nieprawidłowe poziomy: SL={signal.stop_loss}, TP={signal.take_profit}"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False

            # Sprawdź logikę poziomów dla pozycji BUY
            if signal.action == SignalAction.BUY:
                if signal.stop_loss >= signal.entry_price:
                    error_msg = "Stop loss dla pozycji long musi być poniżej ceny wejścia"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False

                if signal.take_profit <= signal.entry_price:
                    error_msg = "Take profit dla pozycji long musi być powyżej ceny wejścia"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False

            # Sprawdź logikę poziomów dla pozycji SELL
            else:
                if signal.stop_loss <= signal.entry_price:
                    error_msg = "Stop loss dla pozycji short musi być powyżej ceny wejścia"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False

                if signal.take_profit >= signal.entry_price:
                    error_msg = "Take profit dla pozycji short musi być poniżej ceny wejścia"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False

            return True

        except Exception as e:
            error_msg = f"Błąd podczas walidacji poziomów: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            return False

    async def close_position(self, position: Position, exit_price: Optional[Decimal], volume: Optional[Decimal] = None) -> Optional[Position]:
        """
        Zamyka pozycję lub jej część.
        
        Args:
            position: Pozycja do zamknięcia
            exit_price: Cena zamknięcia
            volume: Opcjonalny wolumen do zamknięcia (jeśli None, zamyka całą pozycję)
            
        Returns:
            Zamknięta pozycja lub None w przypadku gdy pozycja nie istnieje w otwartych pozycjach
            
        Raises:
            RuntimeError: Gdy parametry są nieprawidłowe (None position, nieprawidłowy wolumen/cena)
        """
        # Walidacja pozycji przed blokiem try
        if position is None:
            error_msg = "Brak pozycji do zamknięcia"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
                    await self.logger.log_error(RuntimeError(error_msg))
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg)

        try:
            # Walidacja pozycji
            if position.entry_price is None:
                error_msg = f"Pozycja {position.id} ma nieprawidłową cenę wejścia: None"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                        await self.logger.log_error(RuntimeError(error_msg))
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            # Walidacja ceny zamknięcia
            if exit_price is None:
                error_msg = "Nieprawidłowa cena zamknięcia: None"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                        await self.logger.log_error(RuntimeError(error_msg))
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            if exit_price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena zamknięcia: {exit_price}"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                        await self.logger.log_error(RuntimeError(error_msg))
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            async with self._lock_context():
                # Sprawdź czy pozycja jest otwarta
                if position.status != PositionStatus.OPEN:
                    error_msg = f"Pozycja {position.id} nie jest otwarta"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return None

                # Sprawdź czy pozycja istnieje w otwartych pozycjach
                if position not in self.open_positions:
                    error_msg = f"Pozycja {position.id} nie istnieje w otwartych pozycjach"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return None

                # Obsługa częściowego zamknięcia
                if volume is not None:
                    if volume <= Decimal('0'):
                        error_msg = f"Nieprawidłowy wolumen: {volume}"
                        try:
                            if self.logger:
                                await self.logger.error(f"❌ {error_msg}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        raise RuntimeError(error_msg)

                    if volume > position.volume:
                        error_msg = f"Wolumen do zamknięcia ({volume}) jest większy niż wolumen pozycji ({position.volume})"
                        try:
                            if self.logger:
                                await self.logger.error(f"❌ {error_msg}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        raise RuntimeError(error_msg)

                    # Utwórz nową pozycję dla zamykanej części
                    closed_position = Position(
                        id=f"{position.id}_partial_{int(time.time() * 1000)}",
                        timestamp=datetime.now(),
                        symbol=position.symbol,
                        trade_type=position.trade_type,
                        volume=volume,
                        entry_price=position.entry_price,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        status=PositionStatus.CLOSED,
                        exit_price=exit_price,
                        profit=self.calculate_position_profit(position, exit_price),
                        pips=self.calculate_position_pips(position, exit_price)
                    )

                    # Zmniejsz wolumen oryginalnej pozycji
                    position.volume -= volume

                    try:
                        if self.logger:
                            await self.logger.info(f"📊 Częściowo zamknięto pozycję {position.id}, wolumen: {volume}")
                            await self.logger.log_trade(closed_position, "CLOSE_PARTIAL")
                    except Exception:
                        pass  # Ignoruj błędy logowania

                self.closed_positions.append(closed_position)
                return closed_position

            # Zamknij całą pozycję
            position.status = PositionStatus.CLOSED
            position.exit_price = exit_price
            position.profit = self.calculate_position_profit(position, exit_price)
            position.pips = self.calculate_position_pips(position, exit_price)

            # Usuń z otwartych i dodaj do zamkniętych
            self.open_positions.remove(position)
            self.closed_positions.append(position)

            try:
                if self.logger:
                    await self.logger.info(f"📊 Zamknięto pozycję {position.id}, profit: {position.profit} pips")
                    await self.logger.log_trade(position, "CLOSE")
            except Exception:
                pass  # Ignoruj błędy logowania

            return position

        except Exception as e:
            error_msg = f"Błąd podczas zamykania pozycji: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    def validate_position_size(self, volume: Decimal) -> bool:
        """
        Sprawdza czy wolumen pozycji nie przekracza maksymalnego rozmiaru.

        Args:
            volume: Wolumen do sprawdzenia

        Returns:
            bool: True jeśli wolumen jest prawidłowy, False w przeciwnym razie
        """
        try:
            if volume <= Decimal('0'):
                error_msg = "Nieprawidłowy wolumen: wolumen musi być większy od 0"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False
            
            total_volume = Decimal('0')
            for position in self.open_positions:
                if position.status == PositionStatus.OPEN:
                    total_volume += position.volume

            total_volume += volume
            if total_volume > self.max_position_size:
                error_msg = f"Przekroczono maksymalny rozmiar pozycji: {total_volume} > {self.max_position_size}"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False

            return True

        except Exception as e:
            error_msg = f"Błąd podczas walidacji wolumenu: {str(e)}"
            try:
                if self.logger:
                    self.logger.error_sync(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            return False

    def check_stop_loss(self, position: Position, current_price: Decimal) -> bool:
        """
        Sprawdza czy pozycja powinna zostać zamknięta przez stop loss.

        Args:
            position: Pozycja do sprawdzenia
            current_price: Aktualna cena rynkowa

        Returns:
            bool: True jeśli pozycja powinna zostać zamknięta
            
        Raises:
            ValueError: Gdy pozycja lub cena są nieprawidłowe
            RuntimeError: Gdy wystąpi błąd podczas sprawdzania
        """
        try:
            if position is None:
                error_msg = "Nie podano pozycji do sprawdzenia"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
            raise ValueError(error_msg)
            
            if position.stop_loss is None:
                error_msg = f"Pozycja {position.id} nie ma ustawionego stop loss"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            if current_price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena: {current_price}"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            # Sprawdź warunki stop loss
            if position.trade_type == TradeType.BUY:
                should_close = current_price <= position.stop_loss
            else:  # SELL
                should_close = current_price >= position.stop_loss

            if should_close:
                try:
                    if self.logger:
                        self.logger.info_sync(
                            f"🔴 Stop Loss dla pozycji {position.id} na poziomie {position.stop_loss}, "
                            f"aktualna cena: {current_price}"
                        )
                except Exception:
                    pass  # Ignoruj błędy logowania

            return should_close

        except Exception as e:
            error_msg = f"Błąd podczas sprawdzania stop loss: {str(e)}"
            try:
                if self.logger:
                    self.logger.error_sync(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    def check_take_profit(self, position: Position, current_price: Decimal) -> bool:
        """
        Sprawdza czy pozycja powinna zostać zamknięta przez take profit.

        Args:
            position: Pozycja do sprawdzenia
            current_price: Aktualna cena rynkowa

        Returns:
            bool: True jeśli pozycja powinna zostać zamknięta
            
        Raises:
            ValueError: Gdy pozycja lub cena są nieprawidłowe
            RuntimeError: Gdy wystąpi błąd podczas sprawdzania
        """
        try:
            if position is None:
                error_msg = "Nie podano pozycji do sprawdzenia"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
            raise ValueError(error_msg)
            
            if position.take_profit is None:
                error_msg = f"Pozycja {position.id} nie ma ustawionego take profit"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            if current_price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena: {current_price}"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            # Sprawdź warunki take profit
            if position.trade_type == TradeType.BUY:
                should_close = current_price >= position.take_profit
            else:  # SELL
                should_close = current_price <= position.take_profit

            if should_close:
                try:
                    if self.logger:
                        self.logger.info_sync(
                            f"🟢 Take Profit dla pozycji {position.id} na poziomie {position.take_profit}, "
                            f"aktualna cena: {current_price}"
                        )
                except Exception:
                    pass  # Ignoruj błędy logowania

            return should_close

        except Exception as e:
            error_msg = f"Błąd podczas sprawdzania take profit: {str(e)}"
            try:
                if self.logger:
                    self.logger.error_sync(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    async def process_price_update(self, current_price: Optional[Decimal]) -> List[Position]:
        """
        Przetwarza aktualizację ceny dla wszystkich otwartych pozycji.
        
        Args:
            current_price: Aktualna cena rynkowa

        Returns:
            Lista zamkniętych pozycji

        Raises:
            ValueError: Gdy cena jest None lub nieprawidłowa
            RuntimeError: Gdy wystąpi błąd podczas przetwarzania
        """
        if current_price is None:
            error_msg = "Nieprawidłowa cena: None"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise ValueError(error_msg)

        if current_price <= Decimal('0'):
            error_msg = f"Nieprawidłowa cena: {current_price}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise ValueError(error_msg)

        closed_positions = []

        try:
            async with self._lock_context():
                for position in self.open_positions[:]:  # Kopia listy do iteracji
                    if position.status != PositionStatus.OPEN:
                        continue

                    # Sprawdź czy pozycja ma ustawione poziomy
                    if position.stop_loss is None or position.take_profit is None:
                        error_msg = f"Pozycja {position.id} ma nieustawione poziomy: SL={position.stop_loss}, TP={position.take_profit}"
                        try:
                            if self.logger:
                                await self.logger.error(f"❌ {error_msg}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        raise RuntimeError(error_msg)

                    # Sprawdź warunki zamknięcia
                    if self.check_stop_loss(position, current_price):
                        try:
                            if self.logger:
                                await self.logger.info(f"🛑 Stop loss dla pozycji {position.id} na poziomie {position.stop_loss}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        closed_position = await self.close_position(position, position.stop_loss)
                        if closed_position:
                            closed_positions.append(closed_position)
                        continue

                    if self.check_take_profit(position, current_price):
                        try:
                            if self.logger:
                                await self.logger.info(f"🎯 Take profit dla pozycji {position.id} na poziomie {position.take_profit}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        closed_position = await self.close_position(position, position.take_profit)
                        if closed_position:
                            closed_positions.append(closed_position)
                        continue

                    # Aktualizuj trailing stop i breakeven
                    await self.update_trailing_stop(position, current_price)
                    await self.update_breakeven(position, current_price)

            return closed_positions

        except Exception as e:
            error_msg = f"Błąd podczas przetwarzania ceny: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    async def process_price_updates(self, prices: List[Decimal]) -> List[Position]:
        """
        Przetwarza listę aktualizacji cen.
        
        Args:
            prices: Lista cen do przetworzenia
            
        Returns:
            Lista zamkniętych pozycji
            
        Raises:
            ValueError: Gdy lista cen jest None lub zawiera nieprawidłowe wartości
            RuntimeError: Gdy wystąpi błąd podczas przetwarzania
        """
        if prices is None:
            error_msg = "Otrzymano None zamiast listy cen"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise ValueError(error_msg)
            
        if not prices:
            try:
                if self.logger:
                    await self.logger.warning("⚠️ Otrzymano pustą listę cen")
            except Exception:
                pass  # Ignoruj błędy logowania
            return []
            
        # Sprawdź czy wszystkie ceny są prawidłowe
        for price in prices:
            if price is None or price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena: {price}"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)
                
        closed_positions = []
        try:
            for price in prices:
                positions = await self.process_price_update(price)
                closed_positions.extend(positions)

            try:
                if self.logger and closed_positions:
                    await self.logger.info(f"📊 Zaktualizowano ceny, zamknięto {len(closed_positions)} pozycji")
            except Exception:
                pass  # Ignoruj błędy logowania

            return closed_positions

        except Exception as e:
            error_msg = f"Błąd podczas przetwarzania cen: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    def calculate_position_profit(self, position: Position, exit_price: Decimal) -> Decimal:
        """
        Oblicza zysk/stratę dla pozycji w walucie kwotowanej.

        Args:
            position: Pozycja do obliczenia
            exit_price: Cena zamknięcia

        Returns:
            Decimal: Zysk/strata w walucie kwotowanej

        Raises:
            ValueError: Gdy pozycja lub cena są nieprawidłowe
        """
        try:
            if position is None:
                error_msg = "Nie podano pozycji do obliczenia zysku"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            if position.entry_price is None:
                error_msg = "Pozycja nie ma ustawionej ceny wejścia"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            if exit_price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena wyjścia: {exit_price}"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            # Oblicz różnicę cen w zależności od typu pozycji
            if position.trade_type == TradeType.BUY:
                price_diff = exit_price - position.entry_price
            else:  # SELL
                price_diff = position.entry_price - exit_price

            # Oblicz zysk/stratę w walucie kwotowanej
            contract_size = Decimal('100000')  # Standardowy rozmiar kontraktu dla Forex
            profit = (price_diff * contract_size * position.volume).quantize(Decimal('0.01'))

            try:
                if self.logger:
                    self.logger.debug_sync(f"📊 Obliczono profit {profit} dla pozycji {position.id}")
            except Exception:
                pass  # Ignoruj błędy logowania

            return profit

        except Exception as e:
            error_msg = f"Błąd podczas obliczania zysku: {str(e)}"
            try:
                if self.logger:
                    self.logger.error_sync(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    def calculate_position_pips(self, position: Position, exit_price: Decimal) -> Decimal:
        """
        Oblicza zysk/stratę w pipsach dla pozycji.

        Args:
            position: Pozycja do obliczenia
            exit_price: Cena zamknięcia

        Returns:
            Decimal: Zysk/strata w pipsach

        Raises:
            ValueError: Gdy pozycja lub cena są nieprawidłowe
        """
        try:
            if position is None:
                error_msg = "Nie podano pozycji do obliczenia pipsów"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            if position.entry_price is None:
                error_msg = "Pozycja nie ma ustawionej ceny wejścia"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            if exit_price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena wyjścia: {exit_price}"
                try:
                    if self.logger:
                        self.logger.error_sync(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            # Pobierz wartość pipsa dla danego instrumentu
            pip_value = position.point_value * Decimal('10')  # 1 pip = 10 punktów

            if position.trade_type == TradeType.BUY:
                pips = ((exit_price - position.entry_price) / pip_value).quantize(Decimal('0.1'))
            else:  # SELL
                pips = ((position.entry_price - exit_price) / pip_value).quantize(Decimal('0.1'))

            try:
                if self.logger:
                    self.logger.debug_sync(f"📊 Obliczono {pips} pipsów dla pozycji {position.id}")
            except Exception:
                pass  # Ignoruj błędy logowania

            return pips

        except Exception as e:
            error_msg = f"Błąd podczas obliczania pipsów: {str(e)}"
            try:
                if self.logger:
                    self.logger.error_sync(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    def get_position_summary(self, position: Position) -> Dict[str, Any]:
        """
        Generuje podsumowanie pozycji w formie słownika.

        Args:
            position: Pozycja do podsumowania

        Returns:
            Słownik z danymi pozycji
        """
        summary = {
            'timestamp': position.timestamp,
            'symbol': position.symbol,
            'trade_type': position.trade_type.value,
            'entry_price': float(position.entry_price),
            'volume': float(position.volume),
            'stop_loss': float(position.stop_loss),
            'take_profit': float(position.take_profit),
            'status': position.status.value
        }
        
        if position.exit_price is not None:
            summary['exit_price'] = float(position.exit_price)
        if position.profit is not None:
            summary['profit'] = float(position.profit)
        if position.pips is not None:
            summary['pips'] = float(position.pips)
            
        return summary

    def calculate_risk_metrics(self, position: Position) -> Dict[str, Decimal]:
        """
        Oblicza metryki ryzyka dla pozycji.

        Args:
            position (Position): Pozycja do analizy

        Returns:
            Dict[str, Decimal]: Słownik z metrykami ryzyka

        Raises:
            ValueError: Gdy pozycja jest nieprawidłowa
            RuntimeError: Gdy wystąpi błąd podczas obliczania metryk
        """
        try:
            # Sprawdź czy pozycja ma wszystkie wymagane pola
            if not all([position.entry_price, position.stop_loss, position.take_profit, position.volume]):
                error_msg = f"❌ Brak wymaganych pól w pozycji {position.id}"
                raise ValueError(error_msg)

            # Sprawdź poprawność poziomów SL/TP
            if position.trade_type == TradeType.BUY:
                if position.stop_loss > position.entry_price:  # Zmieniono z >= na >
                    error_msg = "Stop loss dla pozycji długiej musi być poniżej ceny wejścia"
                    raise ValueError(error_msg)
                if position.take_profit <= position.entry_price:
                    error_msg = "Take profit dla pozycji długiej musi być powyżej ceny wejścia"
                    raise ValueError(error_msg)
            else:  # SELL
                if position.stop_loss <= position.entry_price:
                    error_msg = "Stop loss dla pozycji krótkiej musi być powyżej ceny wejścia"
                    raise ValueError(error_msg)
                if position.take_profit >= position.entry_price:
                    error_msg = "Take profit dla pozycji krótkiej musi być poniżej ceny wejścia"
                    raise ValueError(error_msg)

            # Oblicz podstawowe wartości
            pip_value = Decimal('0.0001')
            risk_pips = abs(position.entry_price - position.stop_loss) / pip_value
            reward_pips = abs(position.take_profit - position.entry_price) / pip_value

            # Oblicz wartości w walucie
            risk_amount = risk_pips * pip_value * position.volume * Decimal('100000')
            reward_amount = reward_pips * pip_value * position.volume * Decimal('100000')

            # Oblicz metryki
            risk_reward_ratio = reward_pips / risk_pips if risk_pips > 0 else Decimal('0')
            risk_per_trade = risk_amount
            position_exposure = position.volume
            max_drawdown = risk_amount

            return {
                'risk_reward_ratio': risk_reward_ratio,
                'risk_per_trade': risk_per_trade,
                'position_exposure': position_exposure,
                'max_drawdown': max_drawdown,
                'risk_pips': risk_pips,
                'reward_pips': reward_pips,
                'risk_amount': risk_amount,
                'reward_amount': reward_amount
            }

        except Exception as e:
            error_msg = f"❌ Błąd podczas obliczania metryk ryzyka: {str(e)}"
            raise RuntimeError(error_msg) from e

    async def modify_position_levels(
        self, 
        position_id: Union[str, Position],
        new_stop_loss: Optional[Decimal] = None,
        new_take_profit: Optional[Decimal] = None,
        allow_breakeven: bool = False
    ) -> bool:
        """
        Modyfikuje poziomy stop loss i take profit dla pozycji.

        Args:
            position_id: ID pozycji do modyfikacji
            new_stop_loss: Nowy poziom stop loss (None jeśli bez zmian)
            new_take_profit: Nowy poziom take profit (None jeśli bez zmian)
            allow_breakeven: Czy pozwolić na ustawienie stop loss na poziomie ceny wejścia

        Returns:
            bool: True jeśli modyfikacja się powiodła, False w przeciwnym razie

        Raises:
            ValueError: Gdy podano nieprawidłowe poziomy
        """
        try:
            async with self._lock_context():
                position = await self._get_position(position_id)
                if not position:
                    error_msg = f"Nie znaleziono pozycji o ID: {position_id}"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False

            if position.status != PositionStatus.OPEN:
                error_msg = f"Pozycja {position_id} nie jest otwarta"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False

            # Sprawdź czy podano jakieś poziomy do modyfikacji
            if new_stop_loss is None and new_take_profit is None:
                error_msg = "Nie podano poziomów do modyfikacji"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False

            # Walidacja poziomów
            if new_stop_loss is not None:
                if new_stop_loss <= Decimal('0'):
                    error_msg = f"Nieprawidłowy poziom stop loss: {new_stop_loss}"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)

            if position.trade_type == TradeType.BUY:
                if not allow_breakeven and new_stop_loss >= position.entry_price:
                    error_msg = "Stop loss dla pozycji long musi być poniżej ceny wejścia"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)
                elif allow_breakeven and new_stop_loss > position.entry_price:
                    error_msg = "Stop loss dla pozycji long nie może być powyżej ceny wejścia przy break even"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)

            if position.trade_type == TradeType.SELL:
                if not allow_breakeven and new_stop_loss <= position.entry_price:
                    error_msg = "Stop loss dla pozycji short musi być powyżej ceny wejścia"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)
                elif allow_breakeven and new_stop_loss < position.entry_price:
                    error_msg = "Stop loss dla pozycji short nie może być poniżej ceny wejścia przy break even"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)

            if new_take_profit is not None:
                if new_take_profit <= Decimal('0'):
                    error_msg = f"Nieprawidłowy poziom take profit: {new_take_profit}"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)

                if position.trade_type == TradeType.BUY and new_take_profit <= position.entry_price:
                    error_msg = "Take profit dla pozycji long musi być powyżej ceny wejścia"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)

                if position.trade_type == TradeType.SELL and new_take_profit >= position.entry_price:
                    error_msg = "Take profit dla pozycji short musi być poniżej ceny wejścia"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)

            # Aktualizacja poziomów
            if new_stop_loss is not None:
                position.stop_loss = new_stop_loss
            if new_take_profit is not None:
                position.take_profit = new_take_profit

            try:
                if self.logger:
                    await self.logger.info(
                        f"✅ Zmodyfikowano poziomy dla pozycji {position_id}: "
                        f"SL={position.stop_loss}, TP={position.take_profit}"
                    )
                    await self.logger.log_trade(position, "MODIFY")
            except Exception:
                pass  # Ignoruj błędy logowania
        
            return True

        except Exception as e:
            error_msg = f"Błąd podczas modyfikacji poziomów: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    async def update_trailing_stop(self, position: Position, current_price: Decimal) -> None:
        """
        Aktualizuje trailing stop dla pozycji.

        Args:
            position: Pozycja do aktualizacji
            current_price: Aktualna cena rynkowa

        Raises:
            ValueError: Gdy pozycja jest None lub cena jest nieprawidłowa
            RuntimeError: Gdy pozycja ma nieustawiony stop loss lub wystąpi inny błąd
        """
        try:
            if position is None:
                error_msg = "Nie podano pozycji do aktualizacji trailing stop"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            if current_price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena: {current_price}"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise ValueError(error_msg)

            if position.status != PositionStatus.OPEN:
                try:
                    if self.logger:
                        await self.logger.debug(f"⚠️ Pominięto aktualizację trailing stop dla zamkniętej pozycji {position.id}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return

            if position.stop_loss is None:
                error_msg = f"Pozycja {position.id} ma nieustawiony stop loss"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            if position.entry_price is None:
                error_msg = f"Pozycja {position.id} ma nieustawioną cenę wejścia"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            # Aktualizuj trailing stop tylko jeśli cena poszła w dobrym kierunku
            if position.trade_type == TradeType.BUY:
                # Dla pozycji long, przesuwamy SL w górę gdy cena rośnie
                if current_price > position.entry_price:
                    # Oblicz dystans między obecnym SL a ceną wejścia
                    current_distance = position.entry_price - position.stop_loss
                    # Oblicz nowy potencjalny poziom SL
                    new_stop_loss = current_price - current_distance
                    # Aktualizuj tylko jeśli nowy SL jest wyżej niż obecny
                    if new_stop_loss > position.stop_loss:
                        old_stop_loss = position.stop_loss
                        success = await self.modify_position_levels(position, new_stop_loss=new_stop_loss)
                        if success:
                            try:
                                if self.logger:
                                    await self.logger.info(
                                        f"🔄 Zaktualizowano trailing stop dla pozycji {position.id} "
                                        f"z {old_stop_loss} na {new_stop_loss}"
                                    )
                            except Exception:
                                pass  # Ignoruj błędy logowania

            else:  # SELL
                # Dla pozycji short, przesuwamy SL w dół gdy cena spada
                if current_price < position.entry_price:
                    # Oblicz dystans między obecnym SL a ceną wejścia
                    current_distance = position.stop_loss - position.entry_price
                    # Oblicz nowy potencjalny poziom SL
                    new_stop_loss = current_price + current_distance
                    # Aktualizuj tylko jeśli nowy SL jest niżej niż obecny
                    if new_stop_loss < position.stop_loss:
                        old_stop_loss = position.stop_loss
                        success = await self.modify_position_levels(position, new_stop_loss=new_stop_loss)
                        if success:
                            try:
                                if self.logger:
                                    await self.logger.info(
                                        f"🔄 Zaktualizowano trailing stop dla pozycji {position.id} "
                                        f"z {old_stop_loss} na {new_stop_loss}"
                                    )
                            except Exception:
                                pass  # Ignoruj błędy logowania

        except Exception as e:
            error_msg = f"Błąd podczas aktualizacji trailing stop: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    async def update_breakeven(self, position_id: Union[str, Position], current_price: Decimal) -> bool:
        """
        Aktualizuje stop loss do poziomu break even dla pozycji, która osiągnęła wymagany zysk.

        Args:
            position_id: ID pozycji lub obiekt Position
            current_price: Aktualna cena rynkowa

        Returns:
            bool: True jeśli poziom został zaktualizowany, False w przeciwnym razie
        """
        try:
            position = await self._get_position(position_id)
            if not position:
                error_msg = "Nie znaleziono pozycji do aktualizacji break even"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False

            if position.status != PositionStatus.OPEN:
                error_msg = "Pozycja jest już zamknięta"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False

            if position.stop_loss is None or position.entry_price is None:
                error_msg = "Pozycja nie ma ustawionego stop loss lub ceny wejścia"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                return False

            # Oblicz minimalny ruch ceny wymagany do break even
            min_move = self.breakeven_pips * position.point_value

            if position.trade_type == TradeType.BUY:
                required_price = position.entry_price + min_move
                if current_price >= required_price:
                    success = await self.modify_position_levels(position, new_stop_loss=position.entry_price)
                    if success:
                        try:
                            if self.logger:
                                await self.logger.info(f"✅ Zaktualizowano SL do break even dla pozycji {position.id}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                    return success
            else:  # SELL
                required_price = position.entry_price - min_move
                if current_price <= required_price:
                    success = await self.modify_position_levels(position, new_stop_loss=position.entry_price)
                    if success:
                        try:
                            if self.logger:
                                await self.logger.info(f"✅ Zaktualizowano SL do break even dla pozycji {position.id}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                    return success

            return False

        except Exception as e:
            error_msg = f"Błąd podczas aktualizacji break even: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg) from e

    async def _get_position(self, position_id: Union[str, Position]) -> Optional[Position]:
        """
        Pobiera pozycję na podstawie ID lub obiektu pozycji.

        Args:
            position_id: ID pozycji lub obiekt pozycji

        Returns:
            Position lub None jeśli nie znaleziono
        """
        try:
            if isinstance(position_id, Position):
                return position_id

            for position in self.open_positions:
                if position.id == position_id:
                    return position

            return None

        except Exception as e:
            error_msg = f"Błąd podczas pobierania pozycji: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            return None