"""
Quote Management System for HFT Market Making
==========================================

Manages quote lifecycle, order placement, and position tracking.
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from loguru import logger

from .avellaneda_stoikov import AvellanedaStoikovPricer, MarketQuote, QuoteParameters
from .risk_manager import RiskManager
from ..utils.config import config


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side enumeration"""
    BID = "bid"
    ASK = "ask"


@dataclass
class Order:
    """Individual order representation"""
    order_id: str
    side: OrderSide
    price: float
    size: float
    timestamp: float = field(default_factory=time.time)
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    exchange_id: Optional[str] = None
    
    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size
    
    @property
    def is_active(self) -> bool:
        return self.status == OrderStatus.ACTIVE
    
    @property 
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED or self.filled_size >= self.size


@dataclass
class QuoteUpdate:
    """Quote update event"""
    timestamp: float
    quote: MarketQuote
    bid_order: Optional[Order] = None
    ask_order: Optional[Order] = None
    reason: str = ""


class QuoteManager:
    """
    Professional quote management system for market making:
    
    Features:
    - Real-time quote generation and placement
    - Order lifecycle management
    - Fill tracking and position updates
    - Risk integration and controls
    - Performance monitoring
    - Quote cancellation and replacement logic
    """
    
    def __init__(self, 
                 symbol: str,
                 pricer: AvellanedaStoikovPricer,
                 risk_manager: RiskManager,
                 order_callback: Optional[Callable] = None):
        
        self.symbol = symbol
        self.pricer = pricer
        self.risk_manager = risk_manager
        self.order_callback = order_callback  # For actual order placement
        
        # Quote state
        self.current_quote: Optional[MarketQuote] = None
        self.current_bid_order: Optional[Order] = None
        self.current_ask_order: Optional[Order] = None
        self.quote_sequence = 0
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.fill_history: List[Dict] = []
        
        # Quote parameters
        self.default_params = QuoteParameters(
            gamma=config.trading.gamma,
            T=config.trading.time_horizon,
            min_spread=config.trading.min_spread
        )
        
        # Performance tracking
        self.stats = {
            'quotes_generated': 0,
            'quotes_sent': 0,
            'quotes_cancelled': 0,
            'orders_filled': 0,
            'total_fill_volume': 0.0,
            'avg_quote_lifetime_sec': 0.0,
            'quote_hit_rate': 0.0
        }
        
        # Threading and control
        self._lock = threading.RLock()
        self._is_active = True
        self._last_quote_time = 0.0
        
        # Quote update tracking
        self.quote_updates: List[QuoteUpdate] = []
        
        logger.info(f"QuoteManager initialized for {symbol}")
    
    def update_parameters(self, params: QuoteParameters) -> None:
        """Update quote generation parameters"""
        with self._lock:
            self.default_params = params
            logger.info(f"Quote parameters updated: gamma={params.gamma}, T={params.T}")
    
    def generate_quote(self, 
                      midprice: float,
                      params: Optional[QuoteParameters] = None,
                      timestamp: float = None) -> Optional[MarketQuote]:
        """
        Generate new market quote using Avellaneda-Stoikov pricer.
        Returns None if quote generation fails or is blocked by risk.
        """
        try:
            with self._lock:
                if timestamp is None:
                    timestamp = time.time()
                
                # Use provided parameters or defaults
                quote_params = params or self.default_params
                
                # Generate quote using pricer
                quote = self.pricer.compute_quotes(quote_params, midprice, timestamp)
                
                # Validate quote - reject if either side has zero size
                if quote.bid_size <= 0 or quote.ask_size <= 0:
                    if self.stats['quotes_generated'] % 100 == 0:
                        logger.warning(f"⚠️ Quote rejected: zero size. "
                                     f"Bid={quote.bid_size:.4f}, Ask={quote.ask_size:.4f}, "
                                     f"Position={self.risk_manager.current_position:.4f}")
                    return None
                
                # Check risk controls
                if not self.risk_manager.check_quote_risk(quote, midprice):
                    # Reduced logging frequency to prevent terminal spam
                    if self.stats['quotes_generated'] % 500 == 0:
                        logger.warning(f"⚠️ Quote blocked by risk manager. Position: {self.risk_manager.current_position:.4f}, "
                                     f"Notional: ${abs(self.risk_manager.current_position) * midprice:.0f}")
                    return None
                
                self.current_quote = quote
                self.stats['quotes_generated'] += 1
                
                # Log successful quote generation occasionally
                if self.stats['quotes_generated'] % 200 == 0:
                    logger.info(f"✅ Quote generated #{self.stats['quotes_generated']}: "
                              f"Bid={quote.bid_price:.2f}, Ask={quote.ask_price:.2f}, "
                              f"Spread={quote.spread:.4f}")
                
                return quote
                
        except Exception as e:
            logger.error(f"Error generating quote: {e}")
            return None
    
    def place_quote(self, quote: MarketQuote) -> Tuple[bool, str]:
        """
        Place or update market quote by managing bid/ask orders.
        Returns (success, message).
        """
        try:
            with self._lock:
                if not self._is_active:
                    return False, "Quote manager not active"
                
                # Cancel existing orders if they need updating
                cancel_needed = self._should_cancel_existing_orders(quote)
                if cancel_needed:
                    self._cancel_existing_orders("Quote update")
                
                # Generate new orders
                bid_order = self._create_order(
                    side=OrderSide.BID,
                    price=quote.bid_price,
                    size=quote.bid_size,
                    timestamp=quote.timestamp
                )
                
                ask_order = self._create_order(
                    side=OrderSide.ASK,
                    price=quote.ask_price,
                    size=quote.ask_size,
                    timestamp=quote.timestamp
                )
                
                # Place orders via callback
                success = True
                message = "Orders placed successfully"
                
                if self.order_callback:
                    try:
                        # Place bid order
                        bid_result = self.order_callback(bid_order)
                        if bid_result.get('success', True):
                            bid_order.exchange_id = bid_result.get('order_id')
                            bid_order.status = OrderStatus.ACTIVE
                            self.active_orders[bid_order.order_id] = bid_order
                            self.current_bid_order = bid_order
                        else:
                            success = False
                            message += f" Bid failed: {bid_result.get('error', 'Unknown')}"
                        
                        # Place ask order  
                        ask_result = self.order_callback(ask_order)
                        if ask_result.get('success', True):
                            ask_order.exchange_id = ask_result.get('order_id')
                            ask_order.status = OrderStatus.ACTIVE
                            self.active_orders[ask_order.order_id] = ask_order
                            self.current_ask_order = ask_order
                        else:
                            success = False
                            message += f" Ask failed: {ask_result.get('error', 'Unknown')}"
                            
                    except Exception as e:
                        success = False
                        message = f"Order placement error: {e}"
                        logger.error(f"Order callback failed: {e}")
                else:
                    # Simulation mode - just mark as active
                    bid_order.status = OrderStatus.ACTIVE
                    ask_order.status = OrderStatus.ACTIVE
                    self.active_orders[bid_order.order_id] = bid_order
                    self.active_orders[ask_order.order_id] = ask_order
                    self.current_bid_order = bid_order
                    self.current_ask_order = ask_order
                
                if success:
                    self.stats['quotes_sent'] += 1
                    self._last_quote_time = quote.timestamp
                    
                    # Record quote update
                    update = QuoteUpdate(
                        timestamp=quote.timestamp,
                        quote=quote,
                        bid_order=bid_order,
                        ask_order=ask_order,
                        reason="New quote"
                    )
                    self.quote_updates.append(update)
                
                return success, message
                
        except Exception as e:
            logger.error(f"Error placing quote: {e}")
            return False, f"Placement error: {e}"
    
    def _should_cancel_existing_orders(self, new_quote: MarketQuote) -> bool:
        """Determine if existing orders should be cancelled for new quote"""
        # ✅ ALWAYS cancel existing orders on every quote update
        # This prevents fill simulator from checking the same quote multiple times
        # Real HFT market makers cancel-replace on every market update
        # Without this, quotes accumulate ~9 probability checks → 95% fill rate
        # With this, quotes get exactly 1 probability check → realistic 20-40% fill rate
        return True
        
        # OLD LOGIC (kept for reference, but disabled):
        # This caused quotes to stay active for multiple market updates
        # Each update gave another 30% fill probability
        # After ~9 updates: 1 - (0.7^9) = 95.8% cumulative fill probability
        
        # # Cancel if no current orders
        # if not self.current_bid_order or not self.current_ask_order:
        #     return True
        # 
        # # Cancel if orders are not active
        # if (not self.current_bid_order.is_active or 
        #     not self.current_ask_order.is_active):
        #     return True
        # 
        # # Cancel if prices changed significantly
        # price_tolerance = self.pricer.tick_size
        # 
        # if (abs(self.current_bid_order.price - new_quote.bid_price) > price_tolerance or
        #     abs(self.current_ask_order.price - new_quote.ask_price) > price_tolerance):
        #     return True
        # 
        # # Cancel if sizes changed significantly
        # size_tolerance = self.pricer.lot_size
        # 
        # if (abs(self.current_bid_order.size - new_quote.bid_size) > size_tolerance or
        #     abs(self.current_ask_order.size - new_quote.ask_size) > size_tolerance):
        #     return True
        # 
        # return False
    
    def _create_order(self, 
                     side: OrderSide, 
                     price: float, 
                     size: float,
                     timestamp: float) -> Order:
        """Create new order object"""
        self.quote_sequence += 1
        order_id = f"{self.symbol}_{side.value}_{self.quote_sequence}_{int(timestamp*1000)}"
        
        return Order(
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            timestamp=timestamp,
            status=OrderStatus.PENDING
        )
    
    def _cancel_existing_orders(self, reason: str) -> None:
        """Cancel current active orders"""
        orders_to_cancel = []
        
        if self.current_bid_order and self.current_bid_order.is_active:
            orders_to_cancel.append(self.current_bid_order)
        
        if self.current_ask_order and self.current_ask_order.is_active:
            orders_to_cancel.append(self.current_ask_order)
        
        for order in orders_to_cancel:
            self._cancel_order(order, reason)
    
    def _cancel_order(self, order: Order, reason: str) -> bool:
        """Cancel individual order"""
        try:
            if self.order_callback:
                # Real cancellation via callback
                cancel_result = self.order_callback({
                    'action': 'cancel',
                    'order_id': order.exchange_id,
                    'reason': reason
                })
                
                if cancel_result.get('success', True):
                    order.status = OrderStatus.CANCELLED
                    logger.debug(f"Order {order.order_id} cancelled: {reason}")
                else:
                    logger.warning(f"Failed to cancel order {order.order_id}: {cancel_result.get('error')}")
                    return False
            else:
                # Simulation mode
                order.status = OrderStatus.CANCELLED
            
            # Remove from active orders
            self.active_orders.pop(order.order_id, None)
            self.stats['quotes_cancelled'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order.order_id}: {e}")
            return False
    
    def handle_fill(self, 
                   order_id: str, 
                   fill_price: float, 
                   fill_size: float,
                   timestamp: float = None) -> bool:
        """
        Handle order fill notification.
        Updates order status, position tracking, and statistics.
        """
        try:
            with self._lock:
                if timestamp is None:
                    timestamp = time.time()
                
                order = self.active_orders.get(order_id)
                if not order:
                    logger.warning(f"Received fill for unknown order: {order_id}")
                    return False
                
                # Update fill information
                order.filled_size += fill_size
                
                # Calculate average fill price
                total_filled_value = order.avg_fill_price * (order.filled_size - fill_size)
                total_filled_value += fill_price * fill_size
                order.avg_fill_price = total_filled_value / order.filled_size
                
                # Check if fully filled
                if order.filled_size >= order.size:
                    order.status = OrderStatus.FILLED
                    self.active_orders.pop(order_id, None)
                
                # Update position tracking
                position_change = fill_size if order.side == OrderSide.BID else -fill_size
                new_position = self.risk_manager.current_position + position_change
                
                # Calculate new average price
                if new_position != 0:
                    current_value = self.risk_manager.current_position * self.risk_manager.avg_entry_price
                    fill_value = position_change * fill_price
                    new_avg_price = (current_value + fill_value) / new_position
                else:
                    new_avg_price = 0.0
                
                # Update risk manager
                self.risk_manager.update_position(new_position, new_avg_price, timestamp)
                
                # Record fill
                fill_record = {
                    'timestamp': timestamp,
                    'order_id': order_id,
                    'side': order.side.value,
                    'price': fill_price,
                    'size': fill_size,
                    'total_filled': order.filled_size,
                    'remaining': order.remaining_size
                }
                
                self.fill_history.append(fill_record)
                
                # Update statistics
                self.stats['orders_filled'] += 1
                self.stats['total_fill_volume'] += fill_size
                
                # Calculate quote hit rate
                if hasattr(self, '_quotes_with_fills'):
                    self._quotes_with_fills += 1
                    self.stats['quote_hit_rate'] = self._quotes_with_fills / max(self.stats['quotes_sent'], 1)
                
                logger.info(f"Fill processed: {order.side.value} {fill_size:.4f} @ {fill_price:.2f} "
                           f"(order {order_id})")
                
                # Cancel the other side if this was a full fill and we want to rebalance
                if order.is_filled and self._should_cancel_on_fill():
                    self._cancel_opposite_order(order.side)
                
                return True
                
        except Exception as e:
            logger.error(f"Error handling fill: {e}")
            return False
    
    def _should_cancel_on_fill(self) -> bool:
        """Determine if opposite side should be cancelled on fill"""
        # Cancel if position is getting too large
        position_util = abs(self.risk_manager.current_position) / self.risk_manager.limits.max_position
        return position_util > 0.8
    
    def _cancel_opposite_order(self, filled_side: OrderSide) -> None:
        """Cancel order on opposite side"""
        if filled_side == OrderSide.BID and self.current_ask_order:
            if self.current_ask_order.is_active:
                self._cancel_order(self.current_ask_order, "Rebalance after fill")
        elif filled_side == OrderSide.ASK and self.current_bid_order:
            if self.current_bid_order.is_active:
                self._cancel_order(self.current_bid_order, "Rebalance after fill")
    
    def cancel_all_orders(self, reason: str = "Manual cancel") -> int:
        """Cancel all active orders. Returns number of orders cancelled."""
        with self._lock:
            cancelled_count = 0
            
            for order in list(self.active_orders.values()):
                if self._cancel_order(order, reason):
                    cancelled_count += 1
            
            self.current_bid_order = None
            self.current_ask_order = None
            
            logger.info(f"Cancelled {cancelled_count} orders: {reason}")
            return cancelled_count
    
    def update_market_quote(self, midprice: float, timestamp: float = None) -> bool:
        """
        Update quotes based on new market conditions.
        This is the main entry point for quote updates.
        """
        try:
            if not self._is_active:
                return False
            
            if timestamp is None:
                timestamp = time.time()
            
            # Update pricer with new market data
            self.pricer.update_market(midprice, timestamp)
            
            # Generate new quote
            quote = self.generate_quote(midprice, timestamp=timestamp)
            if not quote:
                return False
            
            # Place the quote
            success, message = self.place_quote(quote)
            
            if not success:
                logger.warning(f"Failed to place quote: {message}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating market quote: {e}")
            return False
    
    def get_current_state(self) -> Dict:
        """Get current quote manager state"""
        with self._lock:
            return {
                'is_active': self._is_active,
                'current_quote': {
                    'bid_price': self.current_quote.bid_price if self.current_quote else None,
                    'ask_price': self.current_quote.ask_price if self.current_quote else None,
                    'spread': self.current_quote.spread if self.current_quote else None,
                    'timestamp': self.current_quote.timestamp if self.current_quote else None
                } if self.current_quote else None,
                'active_orders': {
                    'bid': {
                        'order_id': self.current_bid_order.order_id,
                        'price': self.current_bid_order.price,
                        'size': self.current_bid_order.size,
                        'filled': self.current_bid_order.filled_size
                    } if self.current_bid_order and self.current_bid_order.is_active else None,
                    'ask': {
                        'order_id': self.current_ask_order.order_id,
                        'price': self.current_ask_order.price,
                        'size': self.current_ask_order.size,
                        'filled': self.current_ask_order.filled_size
                    } if self.current_ask_order and self.current_ask_order.is_active else None
                },
                'position': self.risk_manager.current_position,
                'last_quote_time': self._last_quote_time
            }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive quote manager statistics"""
        with self._lock:
            # Calculate average quote lifetime
            if self.quote_updates:
                lifetimes = []
                for i in range(1, len(self.quote_updates)):
                    lifetime = self.quote_updates[i].timestamp - self.quote_updates[i-1].timestamp
                    lifetimes.append(lifetime)
                
                avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0.0
                self.stats['avg_quote_lifetime_sec'] = avg_lifetime
            
            return {
                **self.stats,
                'active_order_count': len(self.active_orders),
                'order_history_count': len(self.order_history),
                'fill_history_count': len(self.fill_history),
                'quote_update_count': len(self.quote_updates),
                'current_position': self.risk_manager.current_position
            }
    
    def start(self) -> None:
        """Start quote manager operations"""
        with self._lock:
            self._is_active = True
            logger.info("QuoteManager started")
    
    def stop(self) -> None:
        """Stop quote manager and cancel all orders"""
        with self._lock:
            self._is_active = False
            cancelled = self.cancel_all_orders("QuoteManager stopped")
            logger.info(f"QuoteManager stopped, cancelled {cancelled} orders")
    
    def reset(self) -> None:
        """Reset quote manager state (useful for backtesting)"""
        with self._lock:
            self.cancel_all_orders("Reset")
            self.current_quote = None
            self.current_bid_order = None
            self.current_ask_order = None
            self.quote_sequence = 0
            self.active_orders.clear()
            self.order_history.clear()
            self.fill_history.clear()
            self.quote_updates.clear()
            
            # Reset statistics
            self.stats = {
                'quotes_generated': 0,
                'quotes_sent': 0,
                'quotes_cancelled': 0,
                'orders_filled': 0,
                'total_fill_volume': 0.0,
                'avg_quote_lifetime_sec': 0.0,
                'quote_hit_rate': 0.0
            }
            
            logger.info("QuoteManager reset")


# Example usage for testing
if __name__ == "__main__":
    from .avellaneda_stoikov import AvellanedaStoikovPricer
    from .risk_manager import RiskManager, RiskLimits
    
    # Initialize components
    pricer = AvellanedaStoikovPricer(tick_size=0.01, lot_size=0.001)
    risk_manager = RiskManager(RiskLimits())
    
    def mock_order_callback(order_data):
        """Mock order placement callback"""
        if isinstance(order_data, Order):
            return {'success': True, 'order_id': f"MOCK_{order_data.order_id}"}
        else:
            return {'success': True}
    
    # Initialize quote manager
    quote_manager = QuoteManager("BTCUSDT", pricer, risk_manager, mock_order_callback)
    quote_manager.start()
    
    # Simulate market update
    success = quote_manager.update_market_quote(50000.0)
    print(f"Quote update success: {success}")
    
    # Get current state
    state = quote_manager.get_current_state()
    print(f"Current state: {state}")
    
    # Simulate a fill
    if quote_manager.current_bid_order:
        quote_manager.handle_fill(
            quote_manager.current_bid_order.order_id,
            49999.0,
            0.5
        )
    
    # Get statistics
    stats = quote_manager.get_statistics()
    print(f"Statistics: {stats}")
    
    quote_manager.stop()