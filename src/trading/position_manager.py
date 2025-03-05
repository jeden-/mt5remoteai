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
            if self.logger is not None:
                try:
                    await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignorujemy błędy logowania
            raise ValueError(error_msg)

        try:
            current_task = asyncio.current_task()
            
            # Jeśli bieżące zadanie już posiada blokadę, zwiększ licznik
            if self._owner == current_task:
                self._lock_count += 1
                if self.logger is not None:
                    try:
                        await self.logger.debug(f"🔒 Zwiększono licznik blokady do {self._lock_count}")
                    except Exception:
                        pass  # Ignorujemy błędy logowania
                return self._lock

            # Sprawdź czy blokada jest już zajęta
            if self._lock.locked():
                # Próba uzyskania blokady z timeoutem
                try:
                    await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
                except asyncio.TimeoutError:
                    error_msg = f"Timeout podczas oczekiwania na blokadę: {timeout}s"
                    if self.logger is not None:
                        try:
                            await self.logger.error(f"❌ {error_msg}")
                        except Exception:
                            pass  # Ignorujemy błędy logowania
                    raise TimeoutError(error_msg)
            else:
                # Blokada jest wolna, spróbuj ją uzyskać
                await self._lock.acquire()
            
            # Ustaw właściciela i licznik
            self._owner = current_task
            self._lock_count = 1
            if self.logger is not None:
                try:
                    await self.logger.debug("🔒 Uzyskano nową blokadę")
                except Exception:
                    pass  # Ignorujemy błędy logowania
            
            return self._lock

        except asyncio.TimeoutError:
            error_msg = f"Timeout podczas oczekiwania na blokadę: {timeout}s"
            if self.logger is not None:
                try:
                    await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignorujemy błędy logowania
            raise TimeoutError(error_msg)

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
                error_msg = f"Przekroczono maksymalny rozmiar pozycji: {signal.volume}"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
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
        Sprawdza poprawność poziomów dla nowej pozycji.

        Args:
            signal: Sygnał tradingowy

        Returns:
            True jeśli poziomy są prawidłowe
        """
        try:
            if signal.action == SignalAction.BUY:
                if signal.stop_loss >= signal.entry_price:
                    try:
                        if self.logger:
                    await self.logger.error('❌ Stop loss dla pozycji BUY musi być poniżej ceny wejścia')
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False
                if signal.take_profit <= signal.entry_price:
                    try:
                        if self.logger:
                    await self.logger.error('❌ Take profit dla pozycji BUY musi być powyżej ceny wejścia')
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False
            else:  # SELL
                if signal.stop_loss <= signal.entry_price:
                    try:
                        if self.logger:
                    await self.logger.error('❌ Stop loss dla pozycji SELL musi być powyżej ceny wejścia')
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False
                if signal.take_profit >= signal.entry_price:
                    try:
                        if self.logger:
                    await self.logger.error('❌ Take profit dla pozycji SELL musi być poniżej ceny wejścia')
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return False

            return True

        except Exception as e:
            try:
                if self.logger:
            await self.logger.error(f'❌ Błąd podczas walidacji poziomów: {str(e)}')
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
                try:
                    if self.logger:
                await self.logger.info(f"🔄 Zamykam pozycję {position.id} ({position.trade_type.name})")
                except Exception:
                    pass  # Ignoruj błędy logowania

                # Sprawdź czy pozycja jest otwarta
                if position.id not in [p.id for p in self.open_positions]:
                    error_msg = f"Pozycja {position.id} nie jest otwarta"
                    try:
                        if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
                            await self.logger.log_error(RuntimeError(error_msg))
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    return None

                # Walidacja wolumenu
                if volume is not None:
                    if volume <= Decimal('0'):
                        error_msg = f"Nieprawidłowy wolumen: {volume}"
                        try:
                            if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                        await self.logger.log_error(RuntimeError(error_msg))
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        raise RuntimeError(error_msg)

                    if volume > position.volume:
                        error_msg = f"Wolumen {volume} większy niż pozycja {position.volume}"
                        try:
                            if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                        await self.logger.log_error(RuntimeError(error_msg))
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        raise RuntimeError(error_msg)

                    close_volume = volume
                    position.volume -= volume
                else:
                    close_volume = position.volume

                try:
                    profit = self.calculate_position_profit(position, exit_price)
                    pips = self.calculate_position_pips(position, exit_price)
                except Exception as e:
                    error_msg = f"Błąd podczas obliczania profitu: {str(e)}"
                    try:
                        if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
                    await self.logger.log_error(RuntimeError(error_msg))
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise RuntimeError(error_msg)

                # Utwórz zamkniętą pozycję
                closed_position = Position(
                    id=position.id,
                    timestamp=position.timestamp,
                    symbol=position.symbol,
                    trade_type=position.trade_type,
                    entry_price=position.entry_price,
                    volume=close_volume,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    status=PositionStatus.CLOSED,
                    exit_price=exit_price,
                    profit=profit,
                    pips=pips
                )
                
                # Dodaj do zamkniętych pozycji
                self.closed_positions.append(closed_position)

                # Usuń pozycję z otwartych pozycji tylko jeśli zamykamy cały wolumen
                if volume is None or position.volume <= Decimal('0'):
                    try:
                        # Usuń z listy otwartych pozycji
                        self.open_positions = [p for p in self.open_positions if p.id != position.id]
                        # Usuń ze słownika _positions
                        if position.id in self._positions:
                            del self._positions[position.id]
                    except Exception as e:
                        error_msg = f"Błąd podczas usuwania pozycji {position.id}: {str(e)}"
                        try:
                            if self.logger:
                                await self.logger.error(f"❌ {error_msg}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        return None

                # Zaloguj zamknięcie
                try:
                    if self.logger:
                await self.logger.log_trade(closed_position, "CLOSE")
                await self.logger.info(f"✅ Zamknięto pozycję {position.id}: wolumen={close_volume}, profit={profit:.2f}, pips={pips:.1f}")
                except Exception:
                    pass  # Ignoruj błędy logowania

                return closed_position

        except Exception as e:
            try:
                if position and self.logger:
                await self.logger.error(f"❌ Błąd podczas zamykania pozycji {position.id}: {str(e)}")
                await self.logger.log_error(e)
            except Exception:
                pass  # Ignoruj błędy logowania
            if isinstance(e, RuntimeError):
                raise
            return None

    def validate_position_size(self, volume: Decimal) -> bool:
        """
        Sprawdza czy rozmiar pozycji nie przekracza maksimum.

        Args:
            volume: Wielkość pozycji do sprawdzenia

        Returns:
            True jeśli rozmiar jest prawidłowy
        """
        if volume <= Decimal('0'):
            return False
            
        current_volume = sum(p.volume for p in self.open_positions)
        return (current_volume + volume) <= self.max_position_size

    def check_stop_loss(self, position: Position, current_price: Decimal) -> bool:
        """
        Sprawdza czy pozycja powinna zostać zamknięta przez stop loss.

        Args:
            position (Position): Pozycja do sprawdzenia
            current_price (Decimal): Aktualna cena

        Returns:
            bool: True jeśli pozycja powinna zostać zamknięta
            
        Raises:
            RuntimeError: Gdy cena jest nieprawidłowa
        """
        if current_price <= Decimal('0'):
            error_msg = f"❌ Nieprawidłowa cena: {current_price}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            if position.trade_type == TradeType.BUY:
                return current_price <= position.stop_loss
            else:  # SELL
                return current_price >= position.stop_loss
        except Exception as e:
            error_msg = f"❌ Błąd podczas sprawdzania stop loss dla {position.id}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def check_take_profit(self, position: Position, current_price: Decimal) -> bool:
        """
        Sprawdza czy pozycja powinna zostać zamknięta przez take profit.

        Args:
            position (Position): Pozycja do sprawdzenia
            current_price (Decimal): Aktualna cena

        Returns:
            bool: True jeśli pozycja powinna zostać zamknięta
            
        Raises:
            RuntimeError: Gdy cena jest nieprawidłowa
        """
        if current_price <= Decimal('0'):
            error_msg = f"❌ Nieprawidłowa cena: {current_price}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            if position.trade_type == TradeType.BUY:
                return current_price >= position.take_profit
            else:  # SELL
                return current_price <= position.take_profit
        except Exception as e:
            error_msg = f"❌ Błąd podczas sprawdzania take profit dla {position.id}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def process_price_update(self, current_price: Optional[Decimal]) -> None:
        """
        Przetwarza aktualizację ceny dla wszystkich otwartych pozycji.
        
        Args:
            current_price: Aktualna cena

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

        try:
            async with self._lock_context():
                # Kopiujemy listę pozycji, żeby uniknąć modyfikacji podczas iteracji
                positions = self.open_positions.copy()
                
                for position in positions:
                    try:
                        # Sprawdź czy pozycja ma wszystkie wymagane pola
                        if position.stop_loss is None or position.take_profit is None:
                            error_msg = f"Pozycja {position.id} ma nieprawidłowe poziomy: SL={position.stop_loss}, TP={position.take_profit}"
                            try:
                                if self.logger:
                                    await self.logger.error(f"❌ {error_msg}")
                            except Exception:
                                pass  # Ignoruj błędy logowania
                            raise RuntimeError(error_msg)

                        # Sprawdź stop loss
                        if self.check_stop_loss(position, current_price):
                            try:
                                if self.logger:
                                    await self.logger.info(f"🛑 Stop Loss dla {position.id} na poziomie {position.stop_loss}")
                            except Exception:
                                pass  # Ignoruj błędy logowania
                            await self.close_position(position, position.stop_loss)  # Zamykamy po cenie SL
                            continue

                        # Sprawdź take profit
                        if self.check_take_profit(position, current_price):
                            try:
                                if self.logger:
                                    await self.logger.info(f"🎯 Take Profit dla {position.id} na poziomie {position.take_profit}")
                            except Exception:
                                pass  # Ignoruj błędy logowania
                            await self.close_position(position, position.take_profit)  # Zamykamy po cenie TP
                            continue

                        # Aktualizuj trailing stop
                        try:
                    await self.update_trailing_stop(position, current_price)
                        except Exception as e:
                            try:
                                if self.logger:
                                    await self.logger.error(f"❌ Błąd podczas aktualizacji trailing stop dla {position.id}: {str(e)}")
                            except Exception:
                                pass  # Ignoruj błędy logowania

                        # Aktualizuj breakeven
                        try:
                            await self.update_breakeven(position, current_price)
                        except Exception as e:
                            try:
                                if self.logger:
                                    await self.logger.error(f"❌ Błąd podczas aktualizacji breakeven dla {position.id}: {str(e)}")
                            except Exception:
                                pass  # Ignoruj błędy logowania

                    except Exception as e:
                        error_msg = f"Błąd podczas przetwarzania pozycji {position.id}: {str(e)}"
                        try:
                            if self.logger:
                                await self.logger.error(f"❌ {error_msg}")
                        except Exception:
                            pass  # Ignoruj błędy logowania
                        raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f"Błąd podczas przetwarzania ceny: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg)

    async def process_price_updates(self, prices: List[Optional[Decimal]]) -> List[Position]:
        """
        Przetwarza listę aktualizacji cen.
        
        Args:
            prices: Lista cen do przetworzenia
            
        Returns:
            Lista zamkniętych pozycji
            
        Raises:
            ValueError: Gdy lista jest pusta lub zawiera nieprawidłowe wartości
            RuntimeError: Gdy wystąpi błąd podczas przetwarzania
        """
        if not prices:
            try:
                if self.logger:
                    await self.logger.warning("⚠️ Otrzymano pustą listę cen")
            except Exception:
                pass  # Ignoruj błędy logowania
            return []
            
        closed_positions = []
        try:
        for price in prices:
            if price is None:
                    error_msg = "Nieprawidłowa cena: None"
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignoruj błędy logowania
                    raise ValueError(error_msg)

                # Zapisz stan pozycji przed aktualizacją
                positions_before = set(p.id for p in self.open_positions)
                
                await self.process_price_update(price)
                
                # Sprawdź które pozycje zostały zamknięte
                positions_after = set(p.id for p in self.open_positions)
                closed_position_ids = positions_before - positions_after
                
                # Dodaj zamknięte pozycje do listy
                closed_positions.extend([p for p in self.closed_positions if p.id in closed_position_ids])

                return closed_positions

        except Exception as e:
            error_msg = f"Błąd podczas przetwarzania cen: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise ValueError(error_msg)  # Zmieniamy na ValueError dla spójności z testami

    def calculate_position_profit(self, position: Position, exit_price: Decimal) -> Decimal:
        """Oblicza zysk/stratę dla pozycji."""
        pip_value = Decimal('0.0001')  # Wartość 1 pipsa dla par EURUSD
        multiplier = Decimal('100000')  # Mnożnik dla par walutowych

        if position.trade_type == TradeType.BUY:
            price_diff = exit_price - position.entry_price
        else:  # SELL
            price_diff = position.entry_price - exit_price

        # Oblicz zysk/stratę w walucie bazowej
        profit = (price_diff * multiplier * position.volume).quantize(Decimal('0.00001'))

        return profit

    def calculate_position_pips(self, position: Position, exit_price: Decimal) -> Decimal:
        """Oblicza ilość pipsów zysku/straty dla pozycji."""
        pip_value = Decimal('0.0001')  # Wartość 1 pipsa dla par EURUSD

        if position.trade_type == TradeType.BUY:
            pips = ((exit_price - position.entry_price) / pip_value).quantize(Decimal('0.1'))
        else:  # SELL
            pips = ((position.entry_price - exit_price) / pip_value).quantize(Decimal('0.1'))

        return pips

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

    async def modify_position_levels(self, position: Position, new_stop_loss: Optional[Decimal], new_take_profit: Optional[Decimal]) -> bool:
        """
        Modyfikuje poziomy stop loss i take profit dla pozycji.

        Args:
            position: Pozycja do modyfikacji
            new_stop_loss: Nowy poziom stop loss (None oznacza brak zmiany)
            new_take_profit: Nowy poziom take profit (None oznacza brak zmiany)

        Returns:
            bool: True jeśli modyfikacja się powiodła, False w przeciwnym razie
        """
        # Sprawdź czy pozycja nie jest None
        if position is None:
            try:
                if self.logger:
                    await self.logger.error("❌ Pozycja nie może być None")
            except Exception:
                pass
            return False

        try:
            # Sprawdź czy pozycja istnieje
            if position.id not in self._positions:
                try:
                    if self.logger:
                        await self.logger.error(f"❌ Pozycja {position.id} nie istnieje")
                except Exception:
                    pass
                return False

            # Sprawdź czy pozycja jest otwarta
        if position.status != PositionStatus.OPEN:
                try:
                    if self.logger:
                        await self.logger.error(f"❌ Pozycja {position.id} nie jest otwarta")
                except Exception:
                    pass
            return False

            # Obsługa stop loss
            if new_stop_loss is None:
                if position.stop_loss is None:
                    try:
                        if self.logger:
                            await self.logger.error("❌ Brak aktualnego stop loss")
                    except Exception:
                        pass
                    return False
                new_stop_loss = position.stop_loss
            elif new_stop_loss <= Decimal('0'):
                try:
                    if self.logger:
                        await self.logger.error(f"❌ Nieprawidłowy stop loss: {new_stop_loss}")
                except Exception:
                    pass
                return False

            # Obsługa take profit
            if new_take_profit is None:
                if position.take_profit is None:
                    try:
                        if self.logger:
                            await self.logger.error("❌ Brak aktualnego take profit")
                    except Exception:
                        pass
                    return False
                new_take_profit = position.take_profit
            elif new_take_profit <= Decimal('0'):
                try:
                    if self.logger:
                        await self.logger.error(f"❌ Nieprawidłowy take profit: {new_take_profit}")
                except Exception:
                    pass
            return False

        # Walidacja poziomów dla pozycji BUY
        if position.trade_type == TradeType.BUY:
                if new_stop_loss >= position.entry_price or new_take_profit <= position.entry_price:
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ Nieprawidłowe poziomy dla pozycji BUY: SL={new_stop_loss}, TP={new_take_profit}")
                    except Exception:
                        pass
                return False
        # Walidacja poziomów dla pozycji SELL
        else:
                if new_stop_loss <= position.entry_price or new_take_profit >= position.entry_price:
                    try:
                        if self.logger:
                            await self.logger.error(f"❌ Nieprawidłowe poziomy dla pozycji SELL: SL={new_stop_loss}, TP={new_take_profit}")
                    except Exception:
                        pass
                return False

            # Aktualizacja poziomów
            position.stop_loss = new_stop_loss
            position.take_profit = new_take_profit

            try:
                if self.logger:
                    await self.logger.info(f"✅ Zmodyfikowano poziomy dla {position.id}: SL={new_stop_loss}, TP={new_take_profit}")
                    await self.logger.log_trade(position, "MODIFY")
            except Exception:
                pass
        
        return True

        except Exception as e:
            try:
                if self.logger:
                    await self.logger.error(f"❌ Błąd podczas modyfikacji poziomów: {str(e)}")
            except Exception:
                pass
            return False

    async def update_trailing_stop(self, position: Position, current_price: Decimal) -> None:
        """
        Aktualizuje trailing stop dla pozycji.

        Args:
            position: Pozycja do aktualizacji
            current_price: Aktualna cena
            
        Raises:
            RuntimeError: Gdy parametry są nieprawidłowe (None position, nieprawidłowa cena)
        """
        try:
            # Sprawdź wymagane pola
            if position.entry_price is None:
                error_msg = "Brak ceny wejścia w pozycji"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            if position.stop_loss is None:
                error_msg = "Brak stop loss w pozycji"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            if current_price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena: {current_price}"
                try:
                    if self.logger:
                        await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            # Oblicz minimalny dystans dla trailing stop
            min_distance = Decimal('0.0010')  # 10 pipsów

            if position.trade_type == TradeType.BUY:
                # Dla pozycji długiej, przesuń SL w górę jeśli cena wzrosła
                if current_price > position.stop_loss + min_distance:
                    new_stop_loss = current_price - min_distance
                if new_stop_loss > position.stop_loss:
                    position.stop_loss = new_stop_loss
                        try:
                            if self.logger:
                    await self.logger.info(f"🔄 Przesunięto trailing stop dla {position.id} na {new_stop_loss}")
                        except Exception:
                            pass  # Ignoruj błędy logowania

            elif position.trade_type == TradeType.SELL:
                # Dla pozycji krótkiej, przesuń SL w dół jeśli cena spadła
                if current_price < position.stop_loss - min_distance:
                    new_stop_loss = current_price + min_distance
                if new_stop_loss < position.stop_loss:
                    position.stop_loss = new_stop_loss
                        try:
                            if self.logger:
                    await self.logger.info(f"🔄 Przesunięto trailing stop dla {position.id} na {new_stop_loss}")
                        except Exception:
                            pass  # Ignoruj błędy logowania

        except Exception as e:
            error_msg = f"Błąd podczas aktualizacji trailing stop: {str(e)}"
            try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg)

    async def update_breakeven(self, position: Position, current_price: Decimal) -> None:
        """
        Przesuwa stop loss na poziom wejścia (breakeven).

        Args:
            position: Pozycja do aktualizacji
            current_price: Aktualna cena

        Raises:
            RuntimeError: Gdy wystąpi błąd podczas aktualizacji breakeven
        """
        try:
            # Sprawdź wymagane pola
            if position.entry_price is None:
                error_msg = "Brak ceny wejścia w pozycji"
                try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            if position.stop_loss is None:
                error_msg = "Brak stop loss w pozycji"
                try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            if current_price <= Decimal('0'):
                error_msg = f"Nieprawidłowa cena: {current_price}"
                try:
                if self.logger:
                    await self.logger.error(f"❌ {error_msg}")
                except Exception:
                    pass  # Ignoruj błędy logowania
                raise RuntimeError(error_msg)

            # Oblicz minimalny dystans dla breakeven
            min_distance = Decimal('0.0010')  # 10 pipsów

            if position.trade_type == TradeType.BUY:
                # Dla pozycji długiej, sprawdź czy cena jest wystarczająco wysoko
                if current_price > position.entry_price + min_distance:
                    position.stop_loss = position.entry_price
                    try:
                    if self.logger:
                        await self.logger.info(f"🎯 Przesunięto SL na breakeven dla {position.id}")
                    except Exception:
                        pass  # Ignoruj błędy logowania

            elif position.trade_type == TradeType.SELL:
                # Dla pozycji krótkiej, sprawdź czy cena jest wystarczająco nisko
                if current_price < position.entry_price - min_distance:
                    position.stop_loss = position.entry_price
                    try:
                    if self.logger:
                        await self.logger.info(f"🎯 Przesunięto SL na breakeven dla {position.id}")
                    except Exception:
                        pass  # Ignoruj błędy logowania

        except Exception as e:
            error_msg = f"Błąd podczas aktualizacji breakeven: {str(e)}"
            try:
            if self.logger:
                await self.logger.error(f"❌ {error_msg}")
            except Exception:
                pass  # Ignoruj błędy logowania
            raise RuntimeError(error_msg)