"""
Professional Order Book Implementation for HFT Trading
=====================================================

Maintains a rolling order book with top N levels, correct sequencing using U/u logic,
timestamp handling, and low message-loss rate (<0.01%).
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from decimal import Decimal
import threading
from loguru import logger


@dataclass
class OrderBookLevel:
    """Individual price level in order book"""
    price: float
    quantity: float
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.price = float(self.price)
        self.quantity = float(self.quantity)


@dataclass  
class OrderBookSnapshot:
    """Complete order book snapshot at a point in time"""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float
    last_update_id: int
    
    def midprice(self) -> Optional[float]:
        """Calculate midprice from best bid/ask"""
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0
    
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price
    
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points"""
        mid = self.midprice()
        spread = self.spread()
        if mid is None or spread is None or mid == 0:
            return None
        return (spread / mid) * 10000


class OrderBook:
    """
    High-performance order book implementation with professional features:
    - Maintains sorted price levels with automatic cleanup
    - Handles Binance U/u sequencing logic correctly
    - Thread-safe operations for concurrent access
    - Comprehensive logging and error handling
    - Low latency updates with minimal memory allocation
    """
    
    def __init__(self, symbol: str, max_levels: int = 20):
        self.symbol = symbol
        self.max_levels = max_levels
        
        # Order book state
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}  # price -> quantity
        
        # Sequencing and synchronization
        self.last_update_id: int = 0
        self.first_update_id: Optional[int] = None
        self.is_synchronized: bool = False
        
        # Threading and performance
        self._lock = threading.RLock()
        self._update_count = 0
        self._last_snapshot_time = 0.0
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'dropped_updates': 0,
            'out_of_sequence': 0,
            'snapshot_count': 0
        }
        
        logger.info(f"OrderBook initialized for {symbol} with {max_levels} levels")
    
    def handle_snapshot(self, snapshot_data: dict) -> bool:
        """
        Handle order book snapshot from Binance
        Format: {'lastUpdateId': int, 'bids': [[price, qty]], 'asks': [[price, qty]]}
        """
        try:
            with self._lock:
                # Clear existing order book
                self.bids.clear()
                self.asks.clear()
                
                # Set synchronization state
                self.last_update_id = snapshot_data['lastUpdateId']
                self.first_update_id = self.last_update_id
                self.is_synchronized = True
                
                # Process bids
                for bid in snapshot_data.get('bids', []):
                    price, qty = float(bid[0]), float(bid[1])
                    if qty > 0:
                        self.bids[price] = qty
                
                # Process asks
                for ask in snapshot_data.get('asks', []):
                    price, qty = float(ask[0]), float(ask[1])
                    if qty > 0:
                        self.asks[price] = qty
                
                self._trim_levels()
                self.stats['snapshot_count'] += 1
                
                logger.debug(f"Snapshot processed: {len(self.bids)} bids, {len(self.asks)} asks, "
                           f"update_id={self.last_update_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error processing snapshot: {e}")
            return False
    
    def handle_update(self, update_data: dict) -> bool:
        """
        Handle incremental order book update with U/u sequencing
        Format: {'u': final_update_id, 'U': first_update_id, 'b': [[price, qty]], 'a': [[price, qty]]}
        
        Sequencing rules:
        1. First update: U <= lastUpdateId+1 AND u >= lastUpdateId+1
        2. Subsequent: U == lastUpdateId+1
        """
        try:
            with self._lock:
                first_id = update_data['U']
                final_id = update_data['u']
                
                # Check if we're synchronized
                if not self.is_synchronized:
                    logger.warning(f"Received update before synchronization: U={first_id}, u={final_id}")
                    self.stats['dropped_updates'] += 1
                    return False
                
                # Validate sequencing
                if not self._validate_sequence(first_id, final_id):
                    return False
                
                # Process bid updates
                for bid_update in update_data.get('b', []):
                    price, qty = float(bid_update[0]), float(bid_update[1])
                    if qty == 0:
                        self.bids.pop(price, None)  # Remove level
                    else:
                        self.bids[price] = qty  # Add/update level
                
                # Process ask updates  
                for ask_update in update_data.get('a', []):
                    price, qty = float(ask_update[0]), float(ask_update[1])
                    if qty == 0:
                        self.asks.pop(price, None)  # Remove level
                    else:
                        self.asks[price] = qty  # Add/update level
                
                # Update sequence tracking
                self.last_update_id = final_id
                self._update_count += 1
                self.stats['total_updates'] += 1
                
                # Trim to max levels periodically for performance
                if self._update_count % 100 == 0:
                    self._trim_levels()
                
                return True
                
        except Exception as e:
            logger.error(f"Error processing update: {e}")
            self.stats['dropped_updates'] += 1
            return False
    
    def _validate_sequence(self, first_id: int, final_id: int) -> bool:
        """Validate update sequence according to Binance rules"""
        expected_next = self.last_update_id + 1
        
        # Check if this is the expected next update
        if first_id == expected_next:
            return True
        
        # Check for gap (missing updates)
        if first_id > expected_next:
            logger.warning(f"Sequence gap detected: expected {expected_next}, got {first_id}")
            self.stats['out_of_sequence'] += 1
            # We could request a new snapshot here
            return False
        
        # Check for duplicate/old update
        if final_id <= self.last_update_id:
            logger.debug(f"Ignoring old update: u={final_id}, last={self.last_update_id}")
            return False
        
        return True
    
    def _trim_levels(self):
        """Keep only top N levels on each side for performance"""
        try:
            # Sort and trim bids (highest prices first)
            if len(self.bids) > self.max_levels:
                sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
                self.bids = dict(sorted_bids[:self.max_levels])
            
            # Sort and trim asks (lowest prices first)  
            if len(self.asks) > self.max_levels:
                sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
                self.asks = dict(sorted_asks[:self.max_levels])
                
        except Exception as e:
            logger.error(f"Error trimming levels: {e}")
    
    def get_snapshot(self) -> OrderBookSnapshot:
        """Get current order book snapshot"""
        with self._lock:
            # Sort bids (highest first) and asks (lowest first)
            sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
            
            # Create level objects
            bid_levels = [OrderBookLevel(price, qty) for price, qty in sorted_bids]
            ask_levels = [OrderBookLevel(price, qty) for price, qty in sorted_asks]
            
            return OrderBookSnapshot(
                symbol=self.symbol,
                bids=bid_levels,
                asks=ask_levels,
                timestamp=time.time(),
                last_update_id=self.last_update_id
            )
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices"""
        with self._lock:
            best_bid = max(self.bids.keys()) if self.bids else None
            best_ask = min(self.asks.keys()) if self.asks else None
            return best_bid, best_ask
    
    def get_midprice(self) -> Optional[float]:
        """Get current midprice"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get current spread"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def get_depth(self, levels: int = 5) -> Dict:
        """Get order book depth for specified number of levels"""
        with self._lock:
            sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:levels]
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:levels]
            
            return {
                'bids': [[price, qty] for price, qty in sorted_bids],
                'asks': [[price, qty] for price, qty in sorted_asks],
                'timestamp': time.time(),
                'symbol': self.symbol
            }
    
    def is_healthy(self) -> bool:
        """Check if order book is in healthy state"""
        with self._lock:
            # Check basic requirements
            if not self.is_synchronized:
                return False
            
            if not self.bids or not self.asks:
                return False
            
            # Check for crossed book
            best_bid = max(self.bids.keys())
            best_ask = min(self.asks.keys())
            
            if best_bid >= best_ask:
                logger.warning(f"Crossed book detected: bid={best_bid}, ask={best_ask}")
                return False
            
            return True
    
    def get_statistics(self) -> Dict:
        """Get order book statistics"""
        with self._lock:
            return {
                **self.stats,
                'current_levels': {'bids': len(self.bids), 'asks': len(self.asks)},
                'is_synchronized': self.is_synchronized,
                'last_update_id': self.last_update_id,
                'is_healthy': self.is_healthy(),
                'midprice': self.get_midprice(),
                'spread': self.get_spread()
            }
    
    def reset(self):
        """Reset order book state"""
        with self._lock:
            self.bids.clear()
            self.asks.clear()
            self.last_update_id = 0
            self.first_update_id = None
            self.is_synchronized = False
            self._update_count = 0
            
            # Reset stats
            self.stats = {
                'total_updates': 0,
                'dropped_updates': 0,
                'out_of_sequence': 0,
                'snapshot_count': 0
            }
            
            logger.info(f"OrderBook reset for {self.symbol}")


# Example usage for testing
if __name__ == "__main__":
    # Test order book functionality
    ob = OrderBook("BTCUSDT", max_levels=10)
    
    # Simulate snapshot
    snapshot = {
        'lastUpdateId': 1000,
        'bids': [['50000.0', '1.5'], ['49999.0', '2.0']],
        'asks': [['50001.0', '1.0'], ['50002.0', '1.5']]
    }
    
    ob.handle_snapshot(snapshot)
    print("Snapshot processed")
    print(f"Statistics: {ob.get_statistics()}")
    
    # Simulate update
    update = {
        'U': 1001,
        'u': 1001, 
        'b': [['50000.5', '0.5']],  # New bid
        'a': []
    }
    
    ob.handle_update(update)
    print("Update processed")
    print(f"Best bid/ask: {ob.get_best_bid_ask()}")
    print(f"Midprice: {ob.get_midprice()}")
    print(f"Spread: {ob.get_spread()}")