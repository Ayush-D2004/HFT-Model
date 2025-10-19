"""
FIFO Fill Simulator for Backtesting
===================================

Event-driven fill simulator with realistic FIFO queue priority matching.
Models proper market microstructure with order book depth and queue position.
"""

import heapq
import time
import uuid
import random
import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Callable
from enum import Enum

# Use standard logging instead of loguru
logger = logging.getLogger(__name__)

from src.strategy import Order, OrderStatus, OrderSide


class FillReason(Enum):
    """Reason for order fill"""
    MARKET_CROSSED = "market_crossed"
    LIQUIDITY_TAKEN = "liquidity_taken"
    FIFO_MATCHED = "fifo_matched"
    AGGRESSIVE_FILL = "aggressive_fill"


@dataclass
class LimitOrder:
    """Internal limit order representation with FIFO queue tracking"""
    order_id: str
    owner: str
    side: str  # "BUY" or "SELL"
    price: float
    qty: float
    placed_ts: float  # request timestamp
    active_ts: float  # when order becomes active on book (placed_ts + latency)
    filled_qty: float = 0.0
    cancelled: bool = False

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled_qty)


@dataclass
class FillEvent:
    """Order fill event compatible with existing code"""
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
class TradeEvent:
    """Market trade event from historical data"""
    ts: float
    price: float
    qty: float
    aggressor: str  # "BUY" if buyer was aggressor (hit ask), "SELL" if seller hit bid


@dataclass
class LOBSnapshot:
    """Limit order book snapshot"""
    ts: float
    bids: List[Tuple[float, float]]  # list of (price, size) descending
    asks: List[Tuple[float, float]]  # list of (price, size) ascending


class FIFOFillSimulator:
    """
    Event-driven fill simulator with FIFO queueing for resting limit orders.
    
    Key Features:
    - FIFO queue priority (realistic order book matching)
    - Activation latency (orders become active after network delay)
    - Immediate crossing detection (marketable orders at activation)
    - Maker/taker fee differentiation
    - Partial fills supported
    - Compatible with existing backtest engine interface
    """

    def __init__(self, maker_fee: float = -0.0002, taker_fee: float = 0.0005,
                 latency_mean_ms: float = 50.0, latency_std_ms: float = 20.0):
        # Order book queues: price -> deque[LimitOrder] (FIFO)
        self.bids: Dict[float, Deque[LimitOrder]] = defaultdict(deque)
        self.asks: Dict[float, Deque[LimitOrder]] = defaultdict(deque)
        
        # Price-level sorted lists (maintain for best price lookup)
        self.bid_prices = []  # max-heap (store negative)
        self.ask_prices = []  # min-heap
        
        # Active orders tracking
        self.active_orders: Dict[str, LimitOrder] = {}
        
        # Pending activation queue: heap of (active_ts, order_id)
        self.pending_activation: List[Tuple[float, str]] = []
        
        # Simulation parameters
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.latency_mean_ms = latency_mean_ms
        self.latency_std_ms = latency_std_ms
        
        # Metrics
        self.pnl = 0.0
        self.realized_pnl = 0.0
        self.inventory = 0.0
        self.fill_history = []
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
        
        # Market state
        self.last_mid = None
        self.current_time = 0.0
        
        logger.info(f"FIFO FillSimulator initialized: maker_fee={maker_fee:.4f}, taker_fee={taker_fee:.4f}")

    # --------------------------
    # Price book helpers
    # --------------------------
    def _push_bid_price(self, price: float):
        heapq.heappush(self.bid_prices, -price)

    def _push_ask_price(self, price: float):
        heapq.heappush(self.ask_prices, price)

    def _best_bid(self) -> Optional[float]:
        while self.bid_prices:
            p = -self.bid_prices[0]
            if p in self.bids and self._price_level_size(self.bids[p]) > 0:
                return p
            heapq.heappop(self.bid_prices)
        return None

    def _best_ask(self) -> Optional[float]:
        while self.ask_prices:
            p = self.ask_prices[0]
            if p in self.asks and self._price_level_size(self.asks[p]) > 0:
                return p
            heapq.heappop(self.ask_prices)
        return None

    def _price_level_size(self, dq: Deque[LimitOrder]) -> float:
        s = 0.0
        for o in dq:
            if not o.cancelled:
                s += o.remaining
        return s

    # --------------------------
    # Market data feed (compatibility with existing interface)
    # --------------------------
    def update_market_state(self, timestamp: float, best_bid: float, best_ask: float,
                           bid_volume: float = 1.0, ask_volume: float = 1.0,
                           trade_rate: float = 0.1, volatility: float = 0.001) -> None:
        """Update market state and activate pending orders"""
        self.current_time = timestamp
        self.last_mid = (best_bid + best_ask) / 2.0
        
        # Create synthetic LOB snapshot if book is empty (first update)
        if not self.bids and not self.asks:
            snapshot = LOBSnapshot(
                ts=timestamp,
                bids=[(best_bid, bid_volume), (best_bid - 0.01, bid_volume * 0.8)],
                asks=[(best_ask, ask_volume), (best_ask + 0.01, ask_volume * 0.8)]
            )
            self.feed_lob_snapshot(snapshot)
        
        # Activate any pending orders up to this timestamp
        self._activate_pending(timestamp)
        
        # Simulate market activity: create synthetic trades that hit resting orders
        # This maintains compatibility with the probabilistic simulator behavior
        self._simulate_market_trades(timestamp, best_bid, best_ask, trade_rate)

    def update_position(self, position: float) -> None:
        """Update current position (for compatibility with backtest engine)"""
        # In FIFO simulator, inventory is tracked internally from fills
        # This method is for external position updates if needed
        pass

    def feed_lob_snapshot(self, snapshot: LOBSnapshot):
        """Populate order book with synthetic market liquidity"""
        # Clear existing synthetic orders (keep strategy orders)
        # For simplicity, reset entirely
        self.bids.clear()
        self.asks.clear()
        self.bid_prices.clear()
        self.ask_prices.clear()
        
        # Create synthetic resting orders representing market depth
        for p, sz in snapshot.bids:
            oid = f"mkt_bid_{p:.8f}_{uuid.uuid4().hex[:8]}"
            o = LimitOrder(order_id=oid, owner="market", side="BUY", price=p, qty=sz,
                          placed_ts=snapshot.ts, active_ts=snapshot.ts)
            self.bids[p].append(o)
            self._push_bid_price(p)
            self.active_orders[oid] = o
            
        for p, sz in snapshot.asks:
            oid = f"mkt_ask_{p:.8f}_{uuid.uuid4().hex[:8]}"
            o = LimitOrder(order_id=oid, owner="market", side="SELL", price=p, qty=sz,
                          placed_ts=snapshot.ts, active_ts=snapshot.ts)
            self.asks[p].append(o)
            self._push_ask_price(p)
            self.active_orders[oid] = o
            
        self.last_mid = (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2.0 if snapshot.bids and snapshot.asks else None

    # --------------------------
    # Order submission (compatibility with existing Order class)
    # --------------------------
    def submit_order(self, order: Order) -> bool:
        """Submit order from strategy (converts to internal LimitOrder)"""
        try:
            placed_ts = order.timestamp
            active_ts = placed_ts + self.latency_mean_ms / 1000.0
            
            # Convert to internal format
            side_str = "BUY" if order.side == OrderSide.BID else "SELL"
            
            internal_order = LimitOrder(
                order_id=order.order_id,
                owner="me",
                side=side_str,
                price=order.price,
                qty=order.size,
                placed_ts=placed_ts,
                active_ts=active_ts
            )
            
            self.active_orders[order.order_id] = internal_order
            heapq.heappush(self.pending_activation, (active_ts, order.order_id))
            
            self.stats['orders_submitted'] += 1
            order.status = OrderStatus.ACTIVE
            
            logger.debug(f"Order submitted: {order.order_id} {side_str} {order.size:.4f} @ {order.price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        o = self.active_orders.get(order_id)
        if o is None:
            return False
        o.cancelled = True
        self.stats['orders_cancelled'] += 1
        return True

    # --------------------------
    # Order activation
    # --------------------------
    def _simulate_market_trades(self, timestamp: float, best_bid: float, best_ask: float, 
                                trade_rate: float):
        """
        Simulate market trades hitting resting orders (probabilistic fills).
        This maintains compatibility with the probabilistic simulator while using FIFO logic.
        """
        # Check if we have any strategy orders resting in the book
        strategy_orders = [o for o in self.active_orders.values() 
                          if o.owner == "me" and not o.cancelled and o.remaining > 0]
        
        if not strategy_orders:
            return
        
        # Probabilistic fill based on distance from market (similar to old simulator)
        for order in strategy_orders:
            # Skip if not yet activated
            if order.active_ts > timestamp:
                continue
                
            # Calculate distance from best bid/ask (market maker perspective)
            # Positive distance = away from market (lower fill probability)
            # Zero/negative distance = at or inside market (high fill probability)
            if order.side == "BUY":
                # BUY order: compare to best_bid
                # If price >= best_bid: at/inside market (high fill chance)
                # If price < best_bid: below market (lower fill chance)
                distance_bps = ((best_bid - order.price) / best_bid) * 10000
                
                # Skip if order is too far from market (>50 bps away)
                if distance_bps > 50:
                    continue
                    
            else:  # SELL
                # SELL order: compare to best_ask
                # If price <= best_ask: at/inside market (high fill chance)
                # If price > best_ask: above market (lower fill chance)
                distance_bps = ((order.price - best_ask) / best_ask) * 10000
                
                # Skip if order is too far from market (>50 bps away)
                if distance_bps > 50:
                    continue
            
            # Market maker fill probability curve based on distance from best bid/ask
            # Negative distance = inside market (crossing) → instant fill (handled elsewhere)
            # Zero distance = at best bid/ask → very high probability
            # Positive distance = away from market → decreasing probability
            
            if distance_bps <= 0:
                # At or inside best bid/ask (maker order at market)
                base_prob = 0.95  # Very high chance of fill
            elif distance_bps < 0.5:
                # Within 0.5 bps of best (excellent competitive quote)
                base_prob = 0.90
            elif distance_bps < 2.0:
                # 0.5-2 bps away (good competitive quote)
                base_prob = 0.75
            elif distance_bps < 5.0:
                # 2-5 bps away (moderate competitive quote)
                base_prob = 0.55
            elif distance_bps < 10.0:
                # 5-10 bps away (less competitive)
                base_prob = 0.35
            elif distance_bps < 20.0:
                # 10-20 bps away (poor quote placement)
                base_prob = 0.15
            else:
                # >20 bps away (very unlikely to fill)
                base_prob = 0.05
            
            # Adjust for trade rate
            final_prob = base_prob * min(trade_rate * 2, 1.0)
            
            # Random fill check
            if random.random() < final_prob:
                # Simulate trade hitting this order via FIFO
                trade = TradeEvent(
                    ts=timestamp,
                    price=order.price,
                    qty=order.remaining * random.uniform(0.3, 1.0),  # Partial or full
                    aggressor="BUY" if order.side == "SELL" else "SELL"
                )
                self.process_trade(trade)

    def _activate_pending(self, up_to_ts: float):
        """Activate all pending orders whose active_ts <= up_to_ts"""
        while self.pending_activation and self.pending_activation[0][0] <= up_to_ts:
            active_ts, oid = heapq.heappop(self.pending_activation)
            o = self.active_orders.get(oid)
            if o is None or o.cancelled:
                continue
                
            # Insert into book at TAIL of price level (FIFO)
            if o.side == "BUY":
                self.bids[o.price].append(o)
                self._push_bid_price(o.price)
            else:
                self.asks[o.price].append(o)
                self._push_ask_price(o.price)
                
            # Check if order crosses market at activation
            self._handle_activation_crossing(o)

    def _handle_activation_crossing(self, order: LimitOrder):
        """Handle immediate crossing when order activates"""
        best_ask = self._best_ask()
        best_bid = self._best_bid()
        
        if order.side == "BUY" and best_ask is not None and order.price >= best_ask:
            # Aggressive buy: consume asks
            self._consume_opposite_as_taker(order, "SELL")
        elif order.side == "SELL" and best_bid is not None and order.price <= best_bid:
            # Aggressive sell: consume bids
            self._consume_opposite_as_taker(order, "BUY")

    def _consume_opposite_as_taker(self, taker_order: LimitOrder, opposite_side: str):
        """Consume opposite side as taker (aggressive order)"""
        remaining = taker_order.remaining
        trade_ts = taker_order.active_ts
        
        if opposite_side == "SELL":
            # Consume asks
            while remaining > 0:
                best_ask = self._best_ask()
                if best_ask is None or best_ask > taker_order.price:
                    break
                    
                dq = self.asks[best_ask]
                while dq and remaining > 0:
                    resting = dq[0]
                    if resting.cancelled:
                        dq.popleft()
                        continue
                        
                    take = min(resting.remaining, remaining)
                    resting.filled_qty += take
                    taker_order.filled_qty += take
                    remaining -= take
                    
                    # Record fill for taker (our strategy order)
                    if taker_order.owner == "me":
                        self._record_fill(taker_order, take, best_ask, trade_ts, is_maker=False)
                    
                    if resting.remaining <= 1e-12:
                        dq.popleft()
                        
                if self._price_level_size(dq) <= 1e-12:
                    if best_ask in self.asks:
                        del self.asks[best_ask]
        else:
            # Consume bids
            while remaining > 0:
                best_bid = self._best_bid()
                if best_bid is None or best_bid < taker_order.price:
                    break
                    
                dq = self.bids[best_bid]
                while dq and remaining > 0:
                    resting = dq[0]
                    if resting.cancelled:
                        dq.popleft()
                        continue
                        
                    take = min(resting.remaining, remaining)
                    resting.filled_qty += take
                    taker_order.filled_qty += take
                    remaining -= take
                    
                    if taker_order.owner == "me":
                        self._record_fill(taker_order, take, best_bid, trade_ts, is_maker=False)
                    
                    if resting.remaining <= 1e-12:
                        dq.popleft()
                        
                if self._price_level_size(dq) <= 1e-12:
                    if best_bid in self.bids:
                        del self.bids[best_bid]

    # --------------------------
    # Trade processing (from historical data)
    # --------------------------
    def process_trade(self, trade: TradeEvent):
        """Process incoming historical trade (aggressor hits resting orders)"""
        self._activate_pending(trade.ts)
        self.last_mid = trade.price
        
        if trade.aggressor == "BUY":
            # Buy trade hits asks
            self._consume_quantity_from_book("SELL", trade.price, trade.qty, trade.ts)
        else:
            # Sell trade hits bids
            self._consume_quantity_from_book("BUY", trade.price, trade.qty, trade.ts)

    def _consume_quantity_from_book(self, side: str, price_limit: float, qty: float, trade_ts: float):
        """Consume qty from book side (FIFO) within price limit"""
        remaining = qty
        
        if side == "SELL":
            # Consume asks with price <= price_limit
            while remaining > 0:
                best_ask = self._best_ask()
                if best_ask is None or best_ask > price_limit:
                    break
                    
                dq = self.asks[best_ask]
                while dq and remaining > 0:
                    resting = dq[0]
                    if resting.cancelled:
                        dq.popleft()
                        continue
                        
                    take = min(resting.remaining, remaining)
                    resting.filled_qty += take
                    remaining -= take
                    
                    # If this is our strategy order, record fill as maker
                    if resting.owner == "me":
                        self._record_fill(resting, take, resting.price, trade_ts, is_maker=True)
                    
                    if resting.remaining <= 1e-12:
                        dq.popleft()
                        
                if self._price_level_size(dq) <= 1e-12:
                    if best_ask in self.asks:
                        del self.asks[best_ask]
        else:
            # Consume bids with price >= price_limit
            while remaining > 0:
                best_bid = self._best_bid()
                if best_bid is None or best_bid < price_limit:
                    break
                    
                dq = self.bids[best_bid]
                while dq and remaining > 0:
                    resting = dq[0]
                    if resting.cancelled:
                        dq.popleft()
                        continue
                        
                    take = min(resting.remaining, remaining)
                    resting.filled_qty += take
                    remaining -= take
                    
                    if resting.owner == "me":
                        self._record_fill(resting, take, resting.price, trade_ts, is_maker=True)
                    
                    if resting.remaining <= 1e-12:
                        dq.popleft()
                        
                if self._price_level_size(dq) <= 1e-12:
                    if best_bid in self.bids:
                        del self.bids[best_bid]

    # --------------------------
    # Fill recording
    # --------------------------
    def _record_fill(self, order: LimitOrder, fill_qty: float, fill_price: float,
                    timestamp: float, is_maker: bool):
        """Record fill and update metrics"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = fill_qty * fill_price * abs(fee_rate)
        
        # Convert back to OrderSide enum
        side = OrderSide.BID if order.side == "BUY" else OrderSide.ASK
        
        # Create fill event
        fill_event = FillEvent(
            timestamp=timestamp,
            order_id=order.order_id,
            side=side,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            remaining_quantity=order.remaining,
            fee=fee,
            fill_reason=FillReason.FIFO_MATCHED if is_maker else FillReason.AGGRESSIVE_FILL,
            latency_ms=0.0,  # Already accounted for in activation
            is_maker=is_maker
        )
        
        # Update inventory (PnL calculation handled by BacktestMetrics)
        if order.side == "BUY":
            self.inventory += fill_qty
        else:
            self.inventory -= fill_qty
        
        # Update statistics
        self.stats['orders_filled'] += 1
        self.stats['total_fill_volume'] += fill_qty
        self.stats['total_fees_paid'] += fee
        if is_maker:
            self.stats['maker_fills'] += 1
        else:
            self.stats['taker_fills'] += 1
        self.stats['fill_rate'] = self.stats['orders_filled'] / max(self.stats['orders_submitted'], 1)
        
        # Store in history
        self.fill_history.append(fill_event)
        
        # Notify callbacks
        for callback in self.fill_callbacks:
            try:
                callback(fill_event)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
        
        logger.debug(f"Fill recorded: {order.side} {fill_qty:.4f} @ {fill_price:.2f} "
                    f"(maker={is_maker}, fee={fee:.4f})")

    # --------------------------
    # Compatibility methods
    # --------------------------
    def add_fill_callback(self, callback: Callable[[FillEvent], None]) -> None:
        """Add callback for fill notifications"""
        self.fill_callbacks.append(callback)

    def get_statistics(self) -> Dict:
        """Get fill simulator statistics"""
        return {
            **self.stats,
            'pending_orders': sum(1 for o in self.active_orders.values() 
                                if not o.cancelled and o.remaining > 0),
            'inventory': self.inventory,
            'realized_pnl': self.realized_pnl,
        }

    def get_fill_history(self, limit: int = None) -> List[FillEvent]:
        """Get fill history"""
        if limit:
            return self.fill_history[-limit:]
        return self.fill_history.copy()

    def reset(self) -> None:
        """Reset simulator state"""
        self.bids.clear()
        self.asks.clear()
        self.bid_prices.clear()
        self.ask_prices.clear()
        self.active_orders.clear()
        self.pending_activation.clear()
        self.fill_history.clear()
        self.inventory = 0.0
        self.realized_pnl = 0.0
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
        logger.info("FIFO FillSimulator reset")


# Alias for backward compatibility
FillSimulator = FIFOFillSimulator
