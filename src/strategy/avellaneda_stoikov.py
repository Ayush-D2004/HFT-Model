"""
Avellaneda-Stoikov Market Making Strategy Implementation
======================================================

Professional implementation of the Avellaneda-Stoikov optimal market making framework
with enhanced features for high-frequency trading.
"""

import math
import time
from collections import deque
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import numpy as np
from loguru import logger

from ..utils.config import config


@dataclass
class QuoteParameters:
    """Parameters for quote generation"""
    gamma: float  # Risk aversion parameter
    T: float      # Time horizon in seconds
    k: Optional[float] = None  # Arrival rate (events/sec)
    min_spread: float = 0.0   # Minimum spread constraint


@dataclass
class MarketQuote:
    """Generated market quote"""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    reservation_price: float
    half_spread: float
    timestamp: float
    confidence: float = 1.0  # Confidence in the quote (0-1)
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property 
    def midprice(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0


class AvellanedaStoikovPricer:
    """
    Enhanced Avellaneda-Stoikov market making pricer with professional features:
    
    Core Components:
    - EWMA volatility estimator on midprice returns
    - Trade arrival rate estimator based on recent activity
    - Inventory-aware reservation price calculation
    - Optimal bid/ask quote generation with risk controls
    - Dynamic sizing based on market conditions
    
    Features:
    - Real-time parameter adaptation
    - Risk management integration
    - Performance monitoring
    - Market microstructure adjustments
    """
    
    def __init__(self,
                 tick_size: float = 0.01,
                 lot_size: float = 0.001,
                 ewma_alpha: float = 0.2,
                 vol_lookback_sec: int = 60,
                 k_lookback_sec: int = 60,
                 max_inventory: float = 10.0):
        
        # Market structure parameters
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_inventory = max_inventory
        
        # Market state tracking
        self.mid = None
        self.timestamp = None
        self.inventory = 0.0
        
        # Volatility estimation (EWMA)
        self.ewma_alpha = ewma_alpha
        self.ewma_var = None
        self.vol_lookback_sec = vol_lookback_sec
        self.price_history = deque(maxlen=1000)  # For fallback vol estimation
        
        # Trade arrival rate estimation
        self.trade_timestamps = deque()
        self.k_lookback_sec = k_lookback_sec
        self.last_k_estimate = 1e-6
        
        # Quote sizing parameters
        self.base_size = config.trading.lot_size
        self.size_adjustment_factor = 1.0
        
        # Performance tracking
        self.stats = {
            'quotes_generated': 0,
            'vol_updates': 0,
            'trade_events': 0,
            'avg_spread_bps': 0.0,
            'inventory_utilization': 0.0
        }
        
        logger.info(f"AvellanedaStoikov pricer initialized: "
                   f"tick_size={tick_size}, lot_size={lot_size}")
    
    def update_market(self, midprice: float, timestamp: float = None) -> None:
        """
        Update market state with new midprice observation.
        Triggers volatility recalculation using EWMA.
        """
        if timestamp is None:
            timestamp = time.time()
        
        midprice = float(midprice)
        timestamp = float(timestamp)
        
        # Store price history
        self.price_history.append((midprice, timestamp))
        
        # Calculate return if we have previous midprice
        if self.mid is not None:
            # Use log returns for better statistical properties
            log_return = math.log(midprice) - math.log(self.mid)
            squared_return = log_return * log_return
            
            # Update EWMA variance
            if self.ewma_var is None:
                self.ewma_var = squared_return
            else:
                alpha = self.ewma_alpha
                self.ewma_var = alpha * squared_return + (1 - alpha) * self.ewma_var
            
            self.stats['vol_updates'] += 1
        
        # Update current state
        self.mid = midprice
        self.timestamp = timestamp
    
    def register_trade_event(self, timestamp: float = None) -> None:
        """
        Register observed trade for arrival rate estimation.
        Call this for every aggressive order/trade event.
        """
        if timestamp is None:
            timestamp = time.time()
        
        timestamp = float(timestamp)
        self.trade_timestamps.append(timestamp)
        
        # Clean old timestamps
        cutoff = timestamp - self.k_lookback_sec
        while self.trade_timestamps and self.trade_timestamps[0] < cutoff:
            self.trade_timestamps.popleft()
        
        self.stats['trade_events'] += 1
    
    def estimate_arrival_rate(self) -> float:
        """
        Estimate trade arrival rate k (events per second).
        Uses recent trade timestamps with smoothing.
        """
        n_events = len(self.trade_timestamps)
        
        if n_events < 2:
            return self.last_k_estimate
        
        time_span = self.trade_timestamps[-1] - self.trade_timestamps[0]
        
        if time_span <= 0:
            # All events at same time, use event density
            return n_events / max(self.k_lookback_sec, 1.0)
        
        raw_k = n_events / time_span
        
        # Apply smoothing to avoid jumpy estimates
        smoothing_factor = 0.1
        self.last_k_estimate = (
            smoothing_factor * raw_k + 
            (1 - smoothing_factor) * self.last_k_estimate
        )
        
        return max(self.last_k_estimate, 1e-6)  # Floor to avoid division by zero
    
    def update_inventory(self, inventory: float) -> None:
        """Update current inventory position"""
        self.inventory = float(inventory)
        self.stats['inventory_utilization'] = abs(inventory) / self.max_inventory
    
    def get_instantaneous_volatility(self) -> float:
        """
        Calculate instantaneous volatility (sigma per second).
        Uses EWMA with fallback to historical estimation.
        """
        if self.ewma_var is not None:
            return math.sqrt(max(self.ewma_var, 1e-12))
        
        # Fallback: calculate from recent price history
        if len(self.price_history) < 2:
            return 1e-6
        
        returns = []
        for i in range(1, len(self.price_history)):
            prev_price = self.price_history[i-1][0]
            curr_price = self.price_history[i][0]
            if prev_price > 0:
                returns.append(math.log(curr_price) - math.log(prev_price))
        
        if not returns:
            return 1e-6
        
        return float(np.std(returns)) if len(returns) > 1 else 1e-6
    
    def compute_reservation_price(self, 
                                midprice: float, 
                                gamma: float, 
                                sigma: float, 
                                T: float) -> float:
        """
        Calculate reservation (indifference) price.
        
        r = s - inventory * gamma * sigma^2 * T
        
        This represents the theoretical fair value adjusted for inventory risk.
        """
        inventory_adjustment = self.inventory * gamma * (sigma ** 2) * T
        return midprice - inventory_adjustment
    
    def compute_optimal_half_spread(self, 
                                  gamma: float, 
                                  sigma: float, 
                                  T: float, 
                                  k: float) -> float:
        """
        Calculate optimal half-spread using Avellaneda-Stoikov formula.
        
        δ* = (γσ²T)/2 + (1/γ) * ln(1 + γ/k)
        
        First term: risk premium
        Second term: adverse selection protection
        """
        # Ensure positive parameters
        k = max(k, 1e-9)
        gamma = max(gamma, 1e-6)
        
        # Risk premium component
        risk_premium = 0.5 * gamma * (sigma ** 2) * T
        
        # Adverse selection component  
        adverse_selection = (1.0 / gamma) * math.log(1.0 + gamma / k)
        
        return risk_premium + adverse_selection
    
    def calculate_quote_size(self, 
                           midprice: float, 
                           volatility: float, 
                           side: str) -> float:
        """
        Calculate optimal quote size based on market conditions and inventory.
        
        Adjusts base size based on:
        - Inventory position (reduce size when position is large)
        - Market volatility (reduce size in high vol)
        - Liquidity conditions
        """
        base_size = self.base_size
        
        # Inventory adjustment - reduce size as inventory grows
        inventory_ratio = abs(self.inventory) / self.max_inventory
        inventory_factor = max(0.1, 1.0 - inventory_ratio * 0.8)
        
        # Volatility adjustment - reduce size in high volatility
        vol_factor = max(0.2, 1.0 / (1.0 + volatility * 100))
        
        # Side-specific adjustment based on inventory
        if side == 'bid' and self.inventory > 0:
            # Long inventory, less aggressive on bids
            side_factor = max(0.5, 1.0 - inventory_ratio * 0.5)
        elif side == 'ask' and self.inventory < 0:
            # Short inventory, less aggressive on asks  
            side_factor = max(0.5, 1.0 - inventory_ratio * 0.5)
        else:
            side_factor = 1.0
        
        # Combine all factors
        adjusted_size = base_size * inventory_factor * vol_factor * side_factor * self.size_adjustment_factor
        
        # Round to lot size and ensure minimum
        lots = max(1, round(adjusted_size / self.lot_size))
        return lots * self.lot_size
    
    def apply_risk_controls(self, 
                          bid: float, 
                          ask: float, 
                          midprice: float) -> Tuple[float, float]:
        """
        Apply risk controls and constraints to quotes.
        
        Controls:
        - Minimum tick size
        - Maximum spread limits
        - Inventory position limits
        - Market structure constraints
        """
        # Ensure minimum tick increments
        bid = self.round_to_tick(bid)
        ask = self.round_to_tick(ask)
        
        # Prevent crossed quotes
        if bid >= ask:
            logger.warning(f"Crossed quotes detected: bid={bid}, ask={ask}")
            mid = (bid + ask) / 2
            bid = mid - self.tick_size / 2
            ask = mid + self.tick_size / 2
            bid = self.round_to_tick(bid)
            ask = self.round_to_tick(ask)
        
        # Ensure minimum spread
        min_spread = max(self.tick_size, config.trading.min_spread)
        if ask - bid < min_spread:
            spread_adjustment = (min_spread - (ask - bid)) / 2
            bid -= spread_adjustment
            ask += spread_adjustment
            bid = self.round_to_tick(bid)
            ask = self.round_to_tick(ask)
        
        # Inventory position limits
        if abs(self.inventory) >= self.max_inventory * 0.9:
            # Widen quotes when near position limits
            if self.inventory > 0:  # Long, widen bid
                bid -= self.tick_size
            else:  # Short, widen ask
                ask += self.tick_size
            
            bid = self.round_to_tick(bid)
            ask = self.round_to_tick(ask)
        
        return bid, ask
    
    def round_to_tick(self, price: float) -> float:
        """Round price to valid tick increment"""
        ticks = round(price / self.tick_size)
        return ticks * self.tick_size
    
    def compute_quotes(self, 
                     params: QuoteParameters,
                     midprice: float = None,
                     timestamp: float = None) -> MarketQuote:
        """
        Generate optimal bid/ask quotes using Avellaneda-Stoikov framework.
        
        Args:
            params: Quote generation parameters (gamma, T, k, min_spread)
            midprice: Current midprice (optional, uses last observed)
            timestamp: Quote timestamp (optional, uses current time)
        
        Returns:
            MarketQuote object with bid/ask prices and sizes
        """
        # Use provided or last observed values
        if midprice is None:
            if self.mid is None:
                raise ValueError("No midprice available. Call update_market() first.")
            s = self.mid
        else:
            s = float(midprice)
        
        if timestamp is None:
            timestamp = time.time()
        
        # Get market state
        sigma = self.get_instantaneous_volatility()
        k = params.k if params.k is not None else self.estimate_arrival_rate()
        
        # Calculate reservation price and optimal spread
        reservation_price = self.compute_reservation_price(s, params.gamma, sigma, params.T)
        half_spread = self.compute_optimal_half_spread(params.gamma, sigma, params.T, k)
        
        # Apply minimum spread constraint
        half_spread = max(half_spread, params.min_spread / 2.0)
        
        # Generate raw quotes
        raw_bid = reservation_price - half_spread
        raw_ask = reservation_price + half_spread
        
        # Apply risk controls
        bid, ask = self.apply_risk_controls(raw_bid, raw_ask, s)
        
        # Calculate quote sizes
        bid_size = self.calculate_quote_size(s, sigma, 'bid')
        ask_size = self.calculate_quote_size(s, sigma, 'ask') 
        
        # Calculate confidence based on market conditions
        confidence = self._calculate_quote_confidence(sigma, k)
        
        # Update statistics
        self.stats['quotes_generated'] += 1
        spread_bps = ((ask - bid) / s) * 10000 if s > 0 else 0
        self.stats['avg_spread_bps'] = (
            0.9 * self.stats['avg_spread_bps'] + 0.1 * spread_bps
        )
        
        quote = MarketQuote(
            bid_price=bid,
            ask_price=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            reservation_price=reservation_price,
            half_spread=half_spread,
            timestamp=timestamp,
            confidence=confidence
        )
        
        logger.debug(f"Generated quote: bid={bid:.4f} ask={ask:.4f} "
                    f"spread={ask-bid:.4f} ({spread_bps:.1f}bps)")
        
        return quote
    
    def _calculate_quote_confidence(self, sigma: float, k: float) -> float:
        """
        Calculate confidence in the generated quote based on market conditions.
        
        Higher confidence when:
        - Volatility is stable (not too high/low)
        - Trade arrival rate is consistent
        - Recent market activity is normal
        """
        # Volatility confidence (prefer moderate volatility)
        vol_confidence = 1.0 / (1.0 + abs(sigma - 0.001) * 1000)
        
        # Arrival rate confidence (prefer consistent activity)
        k_confidence = min(1.0, k / 0.1) if k < 0.1 else 1.0 / (1.0 + (k - 0.1) * 10)
        
        # Market data freshness
        if self.timestamp:
            data_age = time.time() - self.timestamp
            freshness_confidence = max(0.1, 1.0 - data_age / 10.0)
        else:
            freshness_confidence = 0.5
        
        # Combine factors
        overall_confidence = (vol_confidence * k_confidence * freshness_confidence) ** (1/3)
        
        return max(0.1, min(1.0, overall_confidence))
    
    def get_market_state(self) -> Dict:
        """Get current market state and parameters"""
        return {
            'midprice': self.mid,
            'timestamp': self.timestamp,
            'inventory': self.inventory,
            'volatility': self.get_instantaneous_volatility(),
            'arrival_rate': self.estimate_arrival_rate(),
            'ewma_var': self.ewma_var,
            'price_history_length': len(self.price_history),
            'trade_events_length': len(self.trade_timestamps)
        }
    
    def get_statistics(self) -> Dict:
        """Get strategy performance statistics"""
        return {
            **self.stats,
            'market_state': self.get_market_state(),
            'parameters': {
                'tick_size': self.tick_size,
                'lot_size': self.lot_size,
                'ewma_alpha': self.ewma_alpha,
                'max_inventory': self.max_inventory
            }
        }
    
    def reset(self):
        """Reset strategy state (useful for backtesting)"""
        self.mid = None
        self.timestamp = None
        self.inventory = 0.0
        self.ewma_var = None
        self.price_history.clear()
        self.trade_timestamps.clear()
        self.last_k_estimate = 1e-6
        
        # Reset statistics
        self.stats = {
            'quotes_generated': 0,
            'vol_updates': 0,
            'trade_events': 0,
            'avg_spread_bps': 0.0,
            'inventory_utilization': 0.0
        }
        
        logger.info("AvellanedaStoikov strategy reset")


# Example usage and testing
if __name__ == "__main__":
    # Initialize pricer
    pricer = AvellanedaStoikovPricer(
        tick_size=0.01,
        lot_size=0.001,
        ewma_alpha=0.2,
        max_inventory=5.0
    )
    
    # Simulate market data
    import random
    base_price = 50000.0
    
    for i in range(100):
        # Random walk price
        base_price += random.gauss(0, 10)
        timestamp = time.time() + i
        
        # Update market state
        pricer.update_market(base_price, timestamp)
        
        # Simulate some trades
        if random.random() < 0.3:
            pricer.register_trade_event(timestamp)
        
        # Update inventory (simulate fills)
        if random.random() < 0.1:
            fill = random.gauss(0, 0.5)
            pricer.update_inventory(pricer.inventory + fill)
    
    # Generate quotes
    quote_params = QuoteParameters(
        gamma=0.1,
        T=30.0,
        min_spread=0.02
    )
    
    quote = pricer.compute_quotes(quote_params)
    
    print(f"Market State: {pricer.get_market_state()}")
    print(f"Generated Quote:")
    print(f"  Bid: {quote.bid_price:.4f} @ {quote.bid_size:.3f}")
    print(f"  Ask: {quote.ask_price:.4f} @ {quote.ask_size:.3f}")
    print(f"  Spread: {quote.spread:.4f} ({quote.spread/quote.midprice*10000:.1f}bps)")
    print(f"  Reservation Price: {quote.reservation_price:.4f}")
    print(f"  Confidence: {quote.confidence:.2f}")
    print(f"Statistics: {pricer.get_statistics()}")