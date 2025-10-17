"""
Fill Simulator for Backtesting
=============================

Probabilistic fill model based on quote distance from best bid/ask and historical trade frequency.
Accounts for maker/taker fees and configurable latency for realistic backtesting.
"""

import time
import random
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

from ..strategy import Order, OrderStatus, OrderSide
from ..utils.config import config


class FillReason(Enum):
    """Reason for order fill"""
    MARKET_CROSSED = "market_crossed"  # Market moved through our price
    LIQUIDITY_TAKEN = "liquidity_taken"  # Someone hit our quote
    PARTIAL_FILL = "partial_fill"  # Partial execution
    AGGRESSIVE_FILL = "aggressive_fill"  # Aggressive order filled


@dataclass
class FillEvent:
    """Order fill event"""
    timestamp: float
    order_id: str
    side: OrderSide
    fill_price: float
    fill_quantity: float
    remaining_quantity: float
    fee: float
    fill_reason: FillReason
    latency_ms: float
    is_maker: bool = True


@dataclass
class MarketState:
    """Current market state for fill simulation"""
    timestamp: float
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float
    midprice: float
    spread: float
    recent_trade_rate: float  # trades per second
    volatility: float


class FillSimulator:
    """
    Professional fill simulator for backtesting with realistic market microstructure:
    
    Features:
    - Probabilistic fill model based on market conditions
    - Distance-based fill probability (closer to market = higher probability)
    - Volume-weighted fill timing
    - Latency simulation with configurable distribution
    - Maker/taker fee calculation
    - Partial fill simulation
    - Market impact modeling
    """
    
    def __init__(self,
                 maker_fee: float = 0.001,  # 0.1%
                 taker_fee: float = 0.001,  # 0.1%
                 base_fill_probability: float = 0.8,
                 latency_mean_ms: float = 50.0,
                 latency_std_ms: float = 20.0):
        
        # Fee structure
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # Fill probability parameters
        self.base_fill_probability = base_fill_probability
        self.distance_sensitivity = 10.0  # How sensitive fills are to distance from market
        self.volume_sensitivity = 2.0     # How order size affects fill probability
        
        # Latency simulation
        self.latency_mean_ms = latency_mean_ms
        self.latency_std_ms = latency_std_ms
        
        # Market state tracking
        self.current_market: Optional[MarketState] = None
        self.pending_orders: Dict[str, Order] = {}
        self.fill_history: List[FillEvent] = []
        
        # Fill callbacks
        self.fill_callbacks: List[Callable[[FillEvent], None]] = []
        
        # Statistics
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'total_fill_volume': 0.0,
            'total_fees_paid': 0.0,
            'avg_fill_latency_ms': 0.0,
            'fill_rate': 0.0,
            'maker_fills': 0,
            'taker_fills': 0
        }
        
        logger.info(f"FillSimulator initialized: maker_fee={maker_fee:.3f}, taker_fee={taker_fee:.3f}")
    
    def update_market_state(self,
                           timestamp: float,
                           best_bid: float,
                           best_ask: float,
                           bid_volume: float = 1.0,
                           ask_volume: float = 1.0,
                           trade_rate: float = 0.1,
                           volatility: float = 0.001) -> None:
        """Update current market state for fill simulation"""
        
        midprice = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        
        self.current_market = MarketState(
            timestamp=timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            midprice=midprice,
            spread=spread,
            recent_trade_rate=trade_rate,
            volatility=volatility
        )
        
        # Check for fills on existing orders
        self._check_pending_fills()
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit order to fill simulator.
        Returns True if order accepted, False if rejected.
        """
        try:
            if order.order_id in self.pending_orders:
                logger.warning(f"Order {order.order_id} already exists")
                return False
            
            if not self.current_market:
                logger.warning("No market state available for order submission")
                return False
            
            # Basic validation
            if order.size <= 0 or order.price <= 0:
                logger.warning(f"Invalid order parameters: price={order.price}, size={order.size}")
                return False
            
            # Check if order would immediately cross (aggressive order)
            if self._is_aggressive_order(order):
                # Handle aggressive order immediately
                return self._handle_aggressive_order(order)
            
            # Add to pending orders
            order.status = OrderStatus.ACTIVE
            self.pending_orders[order.order_id] = order
            self.stats['orders_submitted'] += 1
            
            logger.debug(f"Order submitted: {order.side.value} {order.size:.4f} @ {order.price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            order = self.pending_orders.pop(order_id, None)
            if order:
                order.status = OrderStatus.CANCELLED
                self.stats['orders_cancelled'] += 1
                logger.debug(f"Order cancelled: {order_id}")
                return True
            
            logger.warning(f"Cannot cancel order: {order_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def _is_aggressive_order(self, order: Order) -> bool:
        """Check if order is aggressive (crosses the market)"""
        if not self.current_market:
            return False
        
        if order.side == OrderSide.BID:
            return order.price >= self.current_market.best_ask
        else:  # ASK
            return order.price <= self.current_market.best_bid
    
    def _handle_aggressive_order(self, order: Order) -> bool:
        """Handle aggressive order (immediate fill)"""
        try:
            if not self.current_market:
                return False
            
            # Determine fill price (assume we get the best available price)
            if order.side == OrderSide.BID:
                fill_price = self.current_market.best_ask
                available_volume = self.current_market.ask_volume
            else:
                fill_price = self.current_market.best_bid
                available_volume = self.current_market.bid_volume
            
            # Calculate fill quantity (limited by available volume)
            fill_quantity = min(order.size, available_volume)
            
            if fill_quantity <= 0:
                logger.warning("No liquidity available for aggressive order")
                return False
            
            # Simulate latency for aggressive orders (usually faster)
            latency_ms = max(1.0, np.random.normal(self.latency_mean_ms * 0.5, self.latency_std_ms * 0.5))
            
            # Calculate fees (taker fees for aggressive orders)
            fee = fill_quantity * fill_price * self.taker_fee
            
            # Create fill event
            fill_event = FillEvent(
                timestamp=self.current_market.timestamp + (latency_ms / 1000.0),
                order_id=order.order_id,
                side=order.side,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                remaining_quantity=order.size - fill_quantity,
                fee=fee,
                fill_reason=FillReason.AGGRESSIVE_FILL,
                latency_ms=latency_ms,
                is_maker=False
            )
            
            # Process the fill
            self._process_fill(fill_event)
            
            # If partially filled, add remainder to pending orders
            if fill_event.remaining_quantity > 0:
                remaining_order = Order(
                    order_id=f"{order.order_id}_remaining",
                    side=order.side,
                    price=order.price,
                    size=fill_event.remaining_quantity,
                    timestamp=order.timestamp,
                    status=OrderStatus.ACTIVE
                )
                self.pending_orders[remaining_order.order_id] = remaining_order
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling aggressive order: {e}")
            return False
    
    def _check_pending_fills(self) -> None:
        """Check all pending orders for potential fills"""
        if not self.current_market:
            return
        
        orders_to_remove = []
        
        for order_id, order in self.pending_orders.items():
            try:
                # Check if order should be filled
                fill_info = self._calculate_fill_probability(order)
                
                if fill_info and random.random() < fill_info['probability']:
                    # Generate fill event
                    fill_event = self._generate_fill_event(order, fill_info)
                    if fill_event:
                        self._process_fill(fill_event)
                        
                        # Mark order for removal if fully filled
                        if fill_event.remaining_quantity <= 0:
                            orders_to_remove.append(order_id)
                        else:
                            # Update order with remaining quantity
                            order.size = fill_event.remaining_quantity
                            order.filled_size += fill_event.fill_quantity
                
            except Exception as e:
                logger.error(f"Error checking fill for order {order_id}: {e}")
        
        # Remove filled orders
        for order_id in orders_to_remove:
            self.pending_orders.pop(order_id, None)
    
    def _calculate_fill_probability(self, order: Order) -> Optional[Dict]:
        """
        Calculate fill probability based on market conditions and order parameters.
        Returns dictionary with probability and fill details, or None if no fill.
        """
        if not self.current_market:
            return None
        
        try:
            # Check if market has moved through our price
            market_crossed = False
            if order.side == OrderSide.BID:
                market_crossed = self.current_market.best_ask <= order.price
                distance = order.price - self.current_market.best_bid
                available_volume = self.current_market.bid_volume
            else:  # ASK
                market_crossed = self.current_market.best_bid >= order.price
                distance = self.current_market.best_ask - order.price
                available_volume = self.current_market.ask_volume
            
            # If market crossed, high probability of fill
            if market_crossed:
                return {
                    'probability': 0.95,
                    'fill_price': order.price,
                    'fill_quantity': min(order.size, available_volume),
                    'reason': FillReason.MARKET_CROSSED,
                    'is_maker': True
                }
            
            # Calculate distance-based probability
            if distance < 0:
                return None  # Order is away from market
            
            # Normalize distance by spread
            normalized_distance = distance / max(self.current_market.spread, 0.01)
            
            # Base probability decays exponentially with distance
            distance_prob = self.base_fill_probability * np.exp(-self.distance_sensitivity * normalized_distance)
            
            # Adjust for market activity (higher trade rate = higher fill probability)
            activity_multiplier = min(2.0, 1.0 + self.current_market.recent_trade_rate)
            
            # Adjust for order size (larger orders less likely to fill completely)
            size_penalty = np.exp(-self.volume_sensitivity * (order.size / available_volume))
            
            # Adjust for volatility (higher vol = higher fill probability)
            vol_multiplier = min(2.0, 1.0 + self.current_market.volatility * 100)
            
            # Combined probability
            final_probability = distance_prob * activity_multiplier * size_penalty * vol_multiplier
            final_probability = min(0.95, max(0.0, final_probability))
            
            if final_probability < 0.01:
                return None
            
            # Determine fill quantity (could be partial)
            max_fill_quantity = min(order.size, available_volume * 0.8)  # Don't consume all liquidity
            
            # Partial fill probability increases with order size
            partial_fill_prob = min(0.3, order.size / available_volume)
            
            if random.random() < partial_fill_prob:
                fill_quantity = max_fill_quantity * random.uniform(0.3, 0.8)
            else:
                fill_quantity = max_fill_quantity
            
            return {
                'probability': final_probability,
                'fill_price': order.price,
                'fill_quantity': fill_quantity,
                'reason': FillReason.LIQUIDITY_TAKEN,
                'is_maker': True
            }
            
        except Exception as e:
            logger.error(f"Error calculating fill probability: {e}")
            return None
    
    def _generate_fill_event(self, order: Order, fill_info: Dict) -> Optional[FillEvent]:
        """Generate fill event from order and fill info"""
        try:
            # Simulate latency
            latency_ms = max(1.0, np.random.normal(self.latency_mean_ms, self.latency_std_ms))
            
            # Calculate fees
            fee_rate = self.maker_fee if fill_info['is_maker'] else self.taker_fee
            fee = fill_info['fill_quantity'] * fill_info['fill_price'] * fee_rate
            
            return FillEvent(
                timestamp=self.current_market.timestamp + (latency_ms / 1000.0),
                order_id=order.order_id,
                side=order.side,
                fill_price=fill_info['fill_price'],
                fill_quantity=fill_info['fill_quantity'],
                remaining_quantity=order.size - fill_info['fill_quantity'],
                fee=fee,
                fill_reason=fill_info['reason'],
                latency_ms=latency_ms,
                is_maker=fill_info['is_maker']
            )
            
        except Exception as e:
            logger.error(f"Error generating fill event: {e}")
            return None
    
    def _process_fill(self, fill_event: FillEvent) -> None:
        """Process fill event and update statistics"""
        try:
            # Add to fill history
            self.fill_history.append(fill_event)
            
            # Update statistics
            self.stats['orders_filled'] += 1
            self.stats['total_fill_volume'] += fill_event.fill_quantity
            self.stats['total_fees_paid'] += fill_event.fee
            
            if fill_event.is_maker:
                self.stats['maker_fills'] += 1
            else:
                self.stats['taker_fills'] += 1
            
            # Update average latency
            total_latency = (self.stats['avg_fill_latency_ms'] * (self.stats['orders_filled'] - 1) + 
                           fill_event.latency_ms)
            self.stats['avg_fill_latency_ms'] = total_latency / self.stats['orders_filled']
            
            # Update fill rate
            self.stats['fill_rate'] = self.stats['orders_filled'] / max(self.stats['orders_submitted'], 1)
            
            # Notify callbacks
            for callback in self.fill_callbacks:
                try:
                    callback(fill_event)
                except Exception as e:
                    logger.error(f"Error in fill callback: {e}")
            
            logger.debug(f"Fill processed: {fill_event.side.value} {fill_event.fill_quantity:.4f} "
                        f"@ {fill_event.fill_price:.2f} (fee: {fill_event.fee:.4f})")
            
        except Exception as e:
            logger.error(f"Error processing fill: {e}")
    
    def add_fill_callback(self, callback: Callable[[FillEvent], None]) -> None:
        """Add callback to be notified of fills"""
        self.fill_callbacks.append(callback)
    
    def get_pending_orders(self) -> List[Order]:
        """Get list of currently pending orders"""
        return list(self.pending_orders.values())
    
    def get_fill_history(self, limit: int = None) -> List[FillEvent]:
        """Get fill history (optionally limited to recent fills)"""
        if limit:
            return self.fill_history[-limit:]
        return self.fill_history.copy()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive fill simulator statistics"""
        return {
            **self.stats,
            'pending_orders': len(self.pending_orders),
            'fill_history_count': len(self.fill_history),
            'current_market': {
                'timestamp': self.current_market.timestamp if self.current_market else None,
                'midprice': self.current_market.midprice if self.current_market else None,
                'spread': self.current_market.spread if self.current_market else None
            } if self.current_market else None
        }
    
    def reset(self) -> None:
        """Reset fill simulator state"""
        self.current_market = None
        self.pending_orders.clear()
        self.fill_history.clear()
        self.fill_callbacks.clear()
        
        # Reset statistics
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'total_fill_volume': 0.0,
            'total_fees_paid': 0.0,
            'avg_fill_latency_ms': 0.0,
            'fill_rate': 0.0,
            'maker_fills': 0,
            'taker_fills': 0
        }
        
        logger.info("FillSimulator reset")


# Example usage for testing
if __name__ == "__main__":
    # Initialize fill simulator
    fill_simulator = FillSimulator(
        maker_fee=0.001,
        taker_fee=0.001,
        base_fill_probability=0.5
    )
    
    # Add fill callback for monitoring
    def fill_callback(fill_event: FillEvent):
        print(f"Fill: {fill_event.side.value} {fill_event.fill_quantity:.4f} "
              f"@ {fill_event.fill_price:.2f} ({fill_event.fill_reason.value})")
    
    fill_simulator.add_fill_callback(fill_callback)
    
    # Update market state
    fill_simulator.update_market_state(
        timestamp=time.time(),
        best_bid=49995.0,
        best_ask=50005.0,
        bid_volume=2.0,
        ask_volume=2.0,
        trade_rate=0.2,
        volatility=0.001
    )
    
    # Submit some test orders
    bid_order = Order(
        order_id="test_bid_1",
        side=OrderSide.BID,
        price=49990.0,
        size=0.5,
        timestamp=time.time()
    )
    
    ask_order = Order(
        order_id="test_ask_1", 
        side=OrderSide.ASK,
        price=50010.0,
        size=0.5,
        timestamp=time.time()
    )
    
    fill_simulator.submit_order(bid_order)
    fill_simulator.submit_order(ask_order)
    
    print(f"Pending orders: {len(fill_simulator.get_pending_orders())}")
    
    # Simulate market movement that crosses orders
    fill_simulator.update_market_state(
        timestamp=time.time() + 1,
        best_bid=49992.0,
        best_ask=50008.0,
        bid_volume=1.5,
        ask_volume=1.5,
        trade_rate=0.3,
        volatility=0.002
    )
    
    print(f"Statistics: {fill_simulator.get_statistics()}")
    print(f"Fill history: {len(fill_simulator.get_fill_history())} fills")