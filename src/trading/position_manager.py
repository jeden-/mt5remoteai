"""
Moduł odpowiedzialny za zarządzanie pozycjami tradingowymi.
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any, Union
import asyncio
from dataclasses import asdict

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
        logger: Optional[TradingLogger] = None
    ) -> None:
        """
        Inicjalizacja managera pozycji.

        Args:
            symbol: Symbol instrumentu
            max_position_size: Maksymalny rozmiar pozycji
            stop_loss_pips: Domyślny stop loss w pipsach
            take_profit_pips: Domyślny take profit w pipsach
            logger: Logger do zapisywania operacji
        """
        self.symbol = symbol
        self.max_position_size = max_position_size
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.logger = logger or TradingLogger()
        
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self._lock = asyncio.Lock()

    async def open_position(self, signal: SignalData) -> Optional[Position]:
        """
        Otwiera nową pozycję na podstawie sygnału.

        Args:
            signal: Sygnał tradingowy

        Returns:
            Utworzona pozycja lub None w przypadku błędu
        """
        if signal.symbol != self.symbol:
            await self.logger.log_error({
                'type': 'ERROR',
                'symbol': signal.symbol,
                'message': f'Nieprawidłowy symbol: {signal.symbol}, oczekiwano: {self.symbol}'
            })
            return None

        if not self.validate_position_size(signal.volume):
            await self.logger.log_error({
                'type': 'ERROR',
                'symbol': signal.symbol,
                'message': f'Przekroczono maksymalny rozmiar pozycji: {signal.volume}'
            })
            return None

        try:
            async with self._lock:
                position = Position(
                    id=f"{signal.symbol}_{int(signal.timestamp.timestamp())}_{len(self.open_positions) + 1}",
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    trade_type=TradeType.BUY if signal.action == SignalAction.BUY else TradeType.SELL,
                    entry_price=signal.entry_price,
                    volume=signal.volume,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    status=PositionStatus.OPEN
                )
                
                self.open_positions.append(position)
                
                await self.logger.log_trade({
                    'type': 'INFO',
                    'symbol': position.symbol,
                    'message': f'Otwieram pozycję {position.trade_type.name}'
                })
                
                return position
        except Exception as e:
            await self.logger.log_error({
                'type': 'ERROR',
                'message': f'Błąd podczas otwierania pozycji: {str(e)}'
            })
            return None

    async def close_position(
        self, 
        position: Position, 
        exit_price: Decimal,
        volume: Optional[Decimal] = None
    ) -> Optional[Position]:
        """
        Zamyka pozycję po określonej cenie.
        
        Args:
            position: Pozycja do zamknięcia
            exit_price: Cena zamknięcia
            volume: Opcjonalny wolumen do zamknięcia (częściowe zamknięcie)
            
        Returns:
            Zamknięta pozycja lub None w przypadku błędu
        """
        if position.status == PositionStatus.CLOSED:
            raise ValueError("Pozycja jest już zamknięta")

        # Sprawdź czy pozycja istnieje w open_positions
        if position not in self.open_positions:
            return None

        # Sprawdź wolumen do zamknięcia
        if volume is not None:
            if volume <= Decimal('0'):
                return None
            if volume > position.volume:
                return None
            close_volume = volume
            # Aktualizuj wolumen oryginalnej pozycji
            position.volume -= volume
            # Jeśli pozostał wolumen, zaktualizuj pozycję w open_positions
            if position.volume > Decimal('0'):
                # Znajdź indeks pozycji w liście
                position_index = self.open_positions.index(position)
                # Zaktualizuj pozycję w liście
                self.open_positions[position_index] = position
            else:
                # Jeśli cały wolumen został zamknięty, usuń pozycję z listy
                self.open_positions = [p for p in self.open_positions if p.id != position.id]
        else:
            close_volume = position.volume
            # Usuń pozycję z otwartych pozycji
            self.open_positions = [p for p in self.open_positions if p.id != position.id]

        # Oblicz zysk/stratę i pipsy
        profit = self.calculate_position_profit(position, exit_price)
        pips = self.calculate_position_pips(position, exit_price)

        # Utwórz zamkniętą pozycję
        closed_position = Position(
            id=position.id,
            timestamp=position.timestamp,
            symbol=position.symbol,
            trade_type=position.trade_type,
            entry_price=position.entry_price,
            volume=close_volume,
            stop_loss=position.entry_price - Decimal('0.0001') if position.trade_type == TradeType.BUY else position.entry_price + Decimal('0.0001'),
            take_profit=position.entry_price + Decimal('0.0001') if position.trade_type == TradeType.BUY else position.entry_price - Decimal('0.0001'),
            status=PositionStatus.CLOSED,
            exit_price=exit_price,
            profit=profit,
            pips=pips
        )
        
        # Dodaj pozycję do zamkniętych pozycji
        self.closed_positions.append(closed_position)

        # Zaloguj zamknięcie pozycji
        await self.logger.log_trade({
            'type': 'INFO',
            'symbol': position.symbol,
            'message': f'Zamykam pozycję {position.trade_type.name} po cenie {exit_price}, profit: {profit}, pipsy: {pips}'
        })

        return closed_position

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
        Sprawdza czy pozycja powinna być zamknięta przez stop loss.

        Args:
            position: Pozycja do sprawdzenia
            current_price: Aktualna cena

        Returns:
            True jeśli stop loss został osiągnięty
        """
        if position.trade_type == TradeType.BUY:
            return current_price <= position.stop_loss
        return current_price >= position.stop_loss

    def check_take_profit(self, position: Position, current_price: Decimal) -> bool:
        """
        Sprawdza czy pozycja powinna być zamknięta przez take profit.

        Args:
            position: Pozycja do sprawdzenia
            current_price: Aktualna cena

        Returns:
            True jeśli take profit został osiągnięty
        """
        if position.trade_type == TradeType.BUY:
            return current_price >= position.take_profit
        return current_price <= position.take_profit

    async def process_price_updates(self, prices: List[Decimal]) -> None:
        """
        Przetwarza wiele aktualizacji cen równolegle.
        
        Args:
            prices: Lista cen do przetworzenia
        """
        try:
            async with self._lock:
                positions_to_close = []
                positions_to_update = []

                # Sprawdź wszystkie otwarte pozycje
                for i, position in enumerate(self.open_positions):
                    # Przypisz cenę do pozycji (cyklicznie)
                    price = prices[i % len(prices)]
                    
                    # Sprawdź warunki zamknięcia
                    if position.trade_type == TradeType.BUY:
                        if price <= position.stop_loss or price >= position.take_profit:
                            positions_to_close.append((position, price))
                        else:
                            positions_to_update.append((position, price))
                    else:  # SELL
                        if price >= position.stop_loss or price <= position.take_profit:
                            positions_to_close.append((position, price))
                        else:
                            positions_to_update.append((position, price))

                # Zamknij pozycje, które spełniają warunki
                for position, price in positions_to_close:
                    try:
                        await asyncio.wait_for(
                            self.close_position(position, price),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        await self.logger.error(f"⚠️ Timeout podczas zamykania pozycji {position.id}")
                    except Exception as e:
                        await self.logger.error(f"❌ Błąd podczas zamykania pozycji {position.id}: {str(e)}")

                # Aktualizuj trailing stop dla pozostałych pozycji
                for position, price in positions_to_update:
                    try:
                        await asyncio.wait_for(
                            self.update_trailing_stop(position, price),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        await self.logger.error(f"⚠️ Timeout podczas aktualizacji trailing stop dla pozycji {position.id}")
                    except Exception as e:
                        await self.logger.error(f"❌ Błąd podczas aktualizacji trailing stop dla pozycji {position.id}: {str(e)}")

        except Exception as e:
            await self.logger.error(f"❌ Błąd podczas przetwarzania aktualizacji cen: {str(e)}")
            raise

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
            position: Pozycja do analizy

        Returns:
            Dict[str, Decimal]: Słownik z metrykami ryzyka
        """
        risk_per_trade = abs(position.entry_price - position.stop_loss) * Decimal('10000')  # W pipsach
        reward_per_trade = abs(position.entry_price - position.take_profit) * Decimal('10000')  # W pipsach
        risk_reward_ratio = reward_per_trade / risk_per_trade if risk_per_trade > 0 else Decimal('0')
        max_drawdown = risk_per_trade * position.volume * Decimal('100000')  # W walucie bazowej
        position_exposure = position.volume

        return {
            'risk_reward_ratio': risk_reward_ratio,
            'risk_per_trade': risk_per_trade,
            'max_drawdown': max_drawdown,
            'position_exposure': Decimal(str(position_exposure))
        }

    async def modify_position_levels(
        self, 
        position: Position, 
        new_sl: Decimal, 
        new_tp: Decimal
    ) -> bool:
        """
        Modyfikuje poziomy SL/TP dla pozycji.

        Args:
            position: Pozycja do modyfikacji
            new_sl: Nowy poziom stop loss
            new_tp: Nowy poziom take profit

        Returns:
            True jeśli modyfikacja się powiodła
        """
        if position.status != PositionStatus.OPEN:
            return False

        position.stop_loss = new_sl
        position.take_profit = new_tp
        
        await self.logger.log_trade({
            'type': 'INFO',
            'symbol': position.symbol,
            'message': f'Zmodyfikowano poziomy SL/TP dla pozycji {position.trade_type.name}'
        })
        
        return True

    async def update_trailing_stop(self, position: Position, current_price: Decimal) -> None:
        """
        Aktualizuje trailing stop dla pozycji.

        Args:
            position: Pozycja do aktualizacji
            current_price: Aktualna cena
        """
        initial_sl_distance = abs(position.entry_price - position.stop_loss)

        if position.trade_type == TradeType.BUY:
            # Dla pozycji BUY, przesuwamy SL w górę gdy cena rośnie
            if current_price > position.entry_price:
                new_sl = current_price - initial_sl_distance
                if new_sl > position.stop_loss:  # Przesuwamy SL tylko w górę
                    await self.modify_position_levels(position, new_sl, position.take_profit)
                    await self.logger.log_trade({
                        'action': 'update_sl',
                        'symbol': position.symbol,
                        'message': f'Aktualizuję trailing stop dla {position.trade_type.name} na {new_sl}'
                    })
        else:
            # Dla pozycji SELL, przesuwamy SL w dół gdy cena spada
            if current_price < position.entry_price:
                new_sl = current_price + initial_sl_distance
                if new_sl < position.stop_loss:  # Przesuwamy SL tylko w dół
                    await self.modify_position_levels(position, new_sl, position.take_profit)
                    await self.logger.log_trade({
                        'action': 'update_sl',
                        'symbol': position.symbol,
                        'message': f'Aktualizuję trailing stop dla {position.trade_type.name} na {new_sl}'
                    })

    async def update_breakeven(self, position: Position, current_price: Decimal) -> None:
        """
        Przesuwa stop loss na poziom wejścia (breakeven).

        Args:
            position: Pozycja do aktualizacji
            current_price: Aktualna cena
        """
        min_profit_pips = Decimal('50')  # Minimalny zysk w pipsach przed przesunięciem na BE
        
        if position.trade_type == TradeType.BUY:
            if current_price >= position.entry_price + min_profit_pips * Decimal('0.0001'):
                position.stop_loss = position.entry_price
        else:  # SELL
            if current_price <= position.entry_price - min_profit_pips * Decimal('0.0001'):
                position.stop_loss = position.entry_price

    async def _ensure_lock(self, timeout: float = 5.0) -> asyncio.Lock:
        """
        Zapewnia dostęp do blokady z timeoutem.

        Args:
            timeout: Maksymalny czas oczekiwania na blokadę w sekundach

        Returns:
            Obiekt blokady

        Raises:
            asyncio.TimeoutError: Gdy nie udało się uzyskać blokady w zadanym czasie
        """
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            return self._lock
        except asyncio.TimeoutError:
            await self.logger.log_error({
                'type': 'ERROR',
                'message': f'Timeout podczas oczekiwania na blokadę: {timeout}s'
            })
            raise

    async def process_price_update(self, current_price: Decimal) -> None:
        """
        Przetwarza aktualizację ceny dla wszystkich otwartych pozycji.

        Args:
            current_price: Aktualna cena
        """
        if current_price <= Decimal('0'):
            await self.logger.log_error({
                'type': 'ERROR',
                'message': f'Nieprawidłowa cena: {current_price}'
            })
            raise ValueError(f'Nieprawidłowa cena: {current_price}')

        await self.process_price_updates([current_price])