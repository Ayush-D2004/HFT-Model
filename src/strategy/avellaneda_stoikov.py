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
    gamma: float
    T: float
    k: Optional[float] = None
    min_spread: float = 0.0


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
    confidence: float = 1.0
    
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
        
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_inventory = max_inventory
        
        self.mid = None
        self.timestamp = None
        self.inventory = 0.0
        
        self.ewma_alpha = ewma_alpha
        self.ewma_var = None
        self.vol_lookback_sec = vol_lookback_sec
        self.price_history = deque(maxlen=1000)
        
        self.trade_timestamps = deque()
        self.k_lookback_sec = k_lookback_sec
        self.last_k_estimate = 0.1
        
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
        Triggers volatility recalculation using EWMA with proper time-scaling.
        
        FIX #3: Normalize returns by time delta for consistent per-second volatility.
        """
        if timestamp is None:
            timestamp = time.time()
        
        midprice = float(midprice)
        timestamp = float(timestamp)
        
        self.price_history.append((midprice, timestamp))
        
        if self.mid is not None and self.timestamp is not None:
            # Calculate log return
            log_return = math.log(midprice) - math.log(self.mid)
            
            # FIX #3: Normalize by time delta to get per-second volatility
            # This ensures σ is consistent regardless of sampling rate
            dt = timestamp - self.timestamp
            
            # ✅ FIX #4: Add minimum dt threshold (0.1 seconds)
            # Prevents artificial volatility inflation from rapid updates
            dt = max(dt, 0.1)  # Minimum 100ms between updates (was 1e-6)
            
            # Normalize return by sqrt(dt) to get per-second vol
            # Var(r/√dt) = Var(r)/dt, so σ_per_sec = σ_observed/√dt
            normalized_return = log_return / math.sqrt(dt)
            squared_return = normalized_return * normalized_return
            
            if self.ewma_var is None:
                self.ewma_var = squared_return
            else:
                alpha = self.ewma_alpha
                self.ewma_var = alpha * squared_return + (1 - alpha) * self.ewma_var
            
            self.stats['vol_updates'] += 1
        
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
            return max(self.last_k_estimate, 0.1)
        
        time_span = self.trade_timestamps[-1] - self.trade_timestamps[0]
        
        if time_span <= 0:
            return max(n_events / max(self.k_lookback_sec, 1.0), 0.1)
        
        raw_k = n_events / time_span
        
        smoothing_factor = 0.4
        self.last_k_estimate = (
            smoothing_factor * raw_k + 
            (1 - smoothing_factor) * self.last_k_estimate
        )
        
        return max(self.last_k_estimate, 0.1)
    
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
                                T: float,
                                typical_order_size: float = 0.1) -> float:
        """
        Calculate reservation (indifference) price with professional inventory penalty.
        
        FIX #1: Keep adjustments in basis points (not dollar terms) to avoid over-scaling.
        FIX #5: Dynamic cap based on position size to prevent blow-ups.
        The A-S indifference price should shift by tens of bps, not thousands.
        
        r = s - adjustment_in_bps * s / 10000
        
        This represents the theoretical fair value adjusted for inventory risk.
        """
        if abs(self.inventory) < 1e-6:
            # No inventory, no adjustment
            return midprice
        
        # Standard A-S formula in fractional units
        # inventory * gamma * sigma^2 * T gives adjustment as fraction of price
        base_adjustment_fraction = self.inventory * gamma * (sigma ** 2) * T
        
        # Convert to basis points (1 bp = 0.0001 = 0.01%)
        # Multiply by 10000 to get bps
        adj_bp = base_adjustment_fraction * 1e4
        
        # ✅ FIX #5: DYNAMIC CAP based on position size (prevents blow-ups)
        # Small positions: tighter cap (10 bps)
        # Large positions: wider cap (30 bps) to force mean reversion
        position_ratio = abs(self.inventory) / self.max_inventory
        
        if position_ratio < 0.3:
            # Small position (<30% of max): standard 10 bps cap
            max_adj_bp = 10.0
        elif position_ratio < 0.5:
            # Moderate position (30-50%): 15 bps cap
            max_adj_bp = 15.0
        elif position_ratio < 0.7:
            # Large position (50-70%): 20 bps cap
            max_adj_bp = 20.0
        else:
            # Very large position (>70%): 30 bps cap to force unwind
            max_adj_bp = 30.0
        
        # Apply dynamic cap
        adj_bp = np.clip(adj_bp, -max_adj_bp, max_adj_bp)
        
        # Convert back to price adjustment
        scaled_adjustment = adj_bp * midprice / 1e4
        
        # Log significant adjustments (>2 bps) to monitor effectiveness
        if abs(adj_bp) > 2.0:
            logger.debug(f"Inventory penalty: position={self.inventory:.4f} "
                        f"({position_ratio*100:.1f}% of max), "
                        f"adjustment={adj_bp:.2f} bps (cap={max_adj_bp:.0f} bps), "
                        f"price=${scaled_adjustment:.2f}")
        
        return midprice - scaled_adjustment
    
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
        
        FIX #2: Adaptive cap based on volatility (not fixed 10 bps).
        FIX #5: Dynamic spread widening in volatile periods.
        """
        # Ensure positive parameters with reasonable floors
        k = max(k, 0.1)  # Minimum 0.1 events/sec (1 per 10 seconds)
        gamma = max(gamma, 1e-6)
        sigma = max(sigma, 1e-6)
        
        # Risk premium component
        risk_premium = 0.5 * gamma * (sigma ** 2) * T
        
        # Adverse selection component with practical limits
        gamma_over_k = gamma / k
        gamma_over_k = min(gamma_over_k, 2.0)  # Cap for competitive spreads
        
        adverse_selection = (1.0 / gamma) * math.log(1.0 + gamma_over_k)
        adverse_selection = min(adverse_selection, 0.01)  # Max 1% from adverse selection
        
        # Calculate raw half-spread
        total_half_spread = risk_premium + adverse_selection
        
        # FIX #5: Dynamic spread widening by volatility
        # Widen spread in volatile periods: ~1 bp per 0.0001 sigma
        # For sigma=0.001 (10 bps vol): multiply by ~2x
        # For sigma=0.0001 (1 bp vol): multiply by ~1.1x
        volatility_multiplier = 1.0 + min(2.0, sigma * 1e4)
        total_half_spread *= volatility_multiplier
        
        # FIX #2: Adaptive cap based on volatility (not fixed)
        # In low liquidity or high vol, allow wider spreads (up to 50 bps = 0.5%)
        # In normal conditions, keep tight (10 bps = 0.1%)
        # Formula: min(0.005, 2*sigma) ensures cap scales with market conditions
        # Examples:
        #   - sigma=0.0001 (calm): cap = 0.0002 (2 bps) - very tight
        #   - sigma=0.0005 (normal): cap = 0.001 (10 bps) - competitive
        #   - sigma=0.005 (volatile): cap = 0.005 (50 bps) - wider for risk
        max_half_spread_adaptive = min(0.005, 2.0 * sigma)
        
        if total_half_spread > max_half_spread_adaptive:
            logger.debug(f"Half-spread {total_half_spread:.6f} capped at {max_half_spread_adaptive:.6f}. "
                         f"Params: gamma={gamma:.4f}, sigma={sigma:.6f}, T={T:.1f}, k={k:.4f}, "
                         f"vol_mult={volatility_multiplier:.2f}")
            total_half_spread = max_half_spread_adaptive
        
        return total_half_spread
    
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
        
        # ✅ AGGRESSIVE inventory reduction to prevent position build-up
        # At 50% of max: reduce size to 50%
        # At 75% of max: reduce size to 25%
        # At 90%+ of max: reduce size to 10% (emergency only)
        if inventory_ratio >= 0.9:
            inventory_factor = 0.1  # Emergency: almost at limit
        elif inventory_ratio >= 0.75:
            inventory_factor = 0.25  # Critical: 75-90% of max
        elif inventory_ratio >= 0.5:
            inventory_factor = 0.5  # Warning: 50-75% of max
        elif inventory_ratio >= 0.3:
            inventory_factor = 0.7  # Moderate: 30-50% of max
        else:
            inventory_factor = 1.0  # Normal: <30% of max
        
        # Volatility adjustment - reduce size in high volatility
        vol_factor = 1.0 / (1.0 + 10 * volatility / 0.001) 
        
        # Side-specific adjustment based on inventory
        # ✅ AGGRESSIVE side skewing to force position unwind
        if side == 'bid' and self.inventory > 0:
            # Long inventory, MUCH less aggressive on bids (don't want to buy more)
            if inventory_ratio >= 0.7:
                side_factor = 0.1  # Almost stop buying
            elif inventory_ratio >= 0.5:
                side_factor = 0.3  # Reduce buying significantly
            else:
                side_factor = max(0.5, 1.0 - inventory_ratio * 0.8)
        elif side == 'ask' and self.inventory < 0:
            # Short inventory, MUCH less aggressive on asks (don't want to sell more)
            if inventory_ratio >= 0.7:
                side_factor = 0.1  # Almost stop selling
            elif inventory_ratio >= 0.5:
                side_factor = 0.3  # Reduce selling significantly
            else:
                side_factor = max(0.5, 1.0 - inventory_ratio * 0.8)
        elif side == 'bid' and self.inventory < 0:
            # Short inventory, MORE aggressive on bids (want to buy to cover short)
            side_factor = 1.5  # Increase bid size to unwind faster
        elif side == 'ask' and self.inventory > 0:
            # Long inventory, MORE aggressive on asks (want to sell to unwind)
            side_factor = 1.5  # Increase ask size to unwind faster
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
        
        🔧 FIX: Enforce MINIMUM 1-tick separation to prevent crossed quotes
        """
        # Ensure minimum tick increments
        bid = self.round_to_tick(bid)
        ask = self.round_to_tick(ask)
        
        # 🔧 CRITICAL FIX: Enforce minimum 3-tick spread AFTER rounding
        # This prevents bid==ask which causes crossed quote warnings
        # At $0.01 tick_size: 3 ticks = $0.03 minimum spread
        # This is ~0.08 bps at $3800 ETH - extremely tight but prevents crossing
        min_ticks_apart = 3  # At least 3 ticks difference (was 1, but rounding made them equal)
        min_tick_spread = min_ticks_apart * self.tick_size
        config_min_spread = config.trading.min_spread
        
        # Use the LARGER of: 3 ticks OR configured min_spread
        min_required_spread = max(min_tick_spread, config_min_spread)
        
        actual_spread = ask - bid
        
        if actual_spread < min_required_spread:
            logger.debug(f"Adjusting spread from {actual_spread:.2f} to {min_required_spread:.2f}")
            
            # Calculate how much to adjust each side
            spread_deficit = min_required_spread - actual_spread
            half_adjustment = spread_deficit / 2.0
            
            # Move quotes apart symmetrically around midpoint
            mid = (bid + ask) / 2.0
            bid = self.round_to_tick(mid - min_required_spread / 2.0)
            ask = self.round_to_tick(mid + min_required_spread / 2.0)
            
            # FINAL SAFETY: If still equal after rounding, force 1 tick apart
            if bid >= ask:
                logger.warning(f"Quotes still crossed after adjustment: bid={bid}, ask={ask}")
                # Use midprice as anchor
                bid = self.round_to_tick(midprice - min_required_spread / 2.0)
                ask = bid + min_required_spread
                ask = self.round_to_tick(ask)
                
                # LAST RESORT: If ask rounds down to bid, force it up
                if ask <= bid:
                    ask = bid + self.tick_size
        
        # Inventory position limits - STRENGTHENED FOR BETTER BALANCE
        inventory_pct = abs(self.inventory) / self.max_inventory
        
        if inventory_pct >= 0.7:
            # Approaching limits - aggressive quote adjustment
            adjustment_ticks = int((inventory_pct - 0.7) / 0.1) + 1
            
            if self.inventory > 0:  # Long, make selling more attractive
                bid -= adjustment_ticks * self.tick_size  # Widen bid (less attractive to buy)
                ask -= adjustment_ticks * self.tick_size  # Tighten ask (more attractive to sell)
            else:  # Short, make buying more attractive
                ask += adjustment_ticks * self.tick_size  # Widen ask (less attractive to sell)
                bid += adjustment_ticks * self.tick_size  # Tighten bid (more attractive to buy)
            
            bid = self.round_to_tick(bid)
            ask = self.round_to_tick(ask)
            
        elif inventory_pct >= 0.5:
            # Moderately high inventory - gentle adjustment
            if self.inventory > 0:
                bid -= self.tick_size
            else:
                ask += self.tick_size
            
            bid = self.round_to_tick(bid)
            ask = self.round_to_tick(ask)
        
        # 🔧 FINAL SAFETY CHECK: Ensure quotes are NEVER crossed after all adjustments
        if ask <= bid:
            logger.warning(f"⚠️ Quotes crossed after inventory adjustment: bid={bid}, ask={ask}")
            # Force minimum separation - DON'T round ask after this or it might round back to bid!
            min_sep = max(self.tick_size, config.trading.min_spread)
            ask = bid + min_sep
            # Only round UP to next tick, never down
            remainder = ask % self.tick_size
            if remainder > 1e-10:  # Has fractional ticks
                ask = ask - remainder + self.tick_size  # Round up to next tick
        
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
        
        FIX #6: Asymmetric spread adjustment for inventory mean-reversion.
        FIX #7: Adaptive gamma and T based on volatility regime.
        
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
        
        # FIX #7: Adaptive gamma - increase risk aversion in volatile markets
        # Base gamma scales up with realized volatility
        # Normal vol (0.001): gamma_eff ≈ gamma
        # High vol (0.005): gamma_eff ≈ 6*gamma (much more risk averse)
        base_gamma = params.gamma
        gamma_eff = base_gamma * (1.0 + sigma / 0.001)
        
        # FIX #7: Adaptive T - reduce exposure time in fast markets
        # When volatility spikes, use shorter horizon for risk management
        # Normal vol (0.001): T_eff = T
        # High vol (0.005): T_eff ≈ T/6 (rebalance more frequently)
        base_T = params.T
        T_eff = max(5.0, base_T / (1.0 + sigma / 0.001))
        
        # Log adaptive parameters occasionally
        if self.stats['quotes_generated'] % 100 == 0:
            logger.debug(f"Adaptive params: sigma={sigma:.6f} → "
                        f"gamma {base_gamma:.4f}→{gamma_eff:.4f}, "
                        f"T {base_T:.1f}s→{T_eff:.1f}s")
        
        # Calculate typical order size for inventory penalty scaling
        typical_quote_size = self.calculate_quote_size(s, sigma, 'bid')
        
        # Calculate reservation price with adaptive inventory penalty
        reservation_price = self.compute_reservation_price(
            s, gamma_eff, sigma, T_eff, 
            typical_order_size=max(typical_quote_size, 0.01)
        )
        half_spread = self.compute_optimal_half_spread(gamma_eff, sigma, T_eff, k)
        
        # Apply minimum spread constraint
        half_spread = max(half_spread, params.min_spread / 2.0)
        
        # Generate raw quotes (symmetric around reservation price)
        raw_bid = reservation_price - half_spread
        raw_ask = reservation_price + half_spread
        
        # FIX #6: Asymmetric spread adjustment for inventory mean-reversion
        # Instead of just shifting reservation price, also adjust individual sides
        # This makes one side more aggressive (tighter) to encourage mean-reversion
        # While keeping the other side wider for protection
        if abs(self.inventory) > 1e-6:
            inventory_ratio = self.inventory / self.max_inventory
            # Adjustment factor: 20% of half-spread (prevents over-tightening)
            asymmetric_adjustment = half_spread * 0.2 * abs(inventory_ratio)
            
            if self.inventory > 0:  # Long position - want to sell
                raw_ask -= asymmetric_adjustment  # Make sells more aggressive (tighter)
                raw_bid -= asymmetric_adjustment  # Make buys less aggressive (wider)
                if self.stats['quotes_generated'] % 500 == 0:
                    logger.debug(f"Long inventory {self.inventory:.4f}: "
                                f"tightening ask by ${asymmetric_adjustment:.2f}")
            else:  # Short position - want to buy
                raw_bid += asymmetric_adjustment  # Make buys more aggressive (tighter)
                raw_ask += asymmetric_adjustment  # Make sells less aggressive (wider)
                if self.stats['quotes_generated'] % 500 == 0:
                    logger.debug(f"Short inventory {self.inventory:.4f}: "
                                f"tightening bid by ${asymmetric_adjustment:.2f}")
        
        # Apply risk controls
        bid, ask = self.apply_risk_controls(raw_bid, raw_ask, s)
        
        # Calculate quote sizes
        bid_size = self.calculate_quote_size(s, sigma, 'bid')
        ask_size = self.calculate_quote_size(s, sigma, 'ask')
        
        # ✅ CRITICAL: At extreme positions (>80% of max), ONLY quote on unwinding side
        # This prevents position from growing further
        inventory_ratio = abs(self.inventory) / self.max_inventory
        if inventory_ratio > 0.80:
            if self.inventory > 0:
                # Long position: ONLY allow selling (ask), no buying (bid)
                bid_size = 0.0  # Cancel bid side completely
                logger.warning(f"⚠️ EXTREME LONG POSITION ({inventory_ratio*100:.0f}%): "
                             f"Only quoting ASK side to unwind")
            else:
                # Short position: ONLY allow buying (bid), no selling (ask)
                ask_size = 0.0  # Cancel ask side completely
                logger.warning(f"⚠️ EXTREME SHORT POSITION ({inventory_ratio*100:.0f}%): "
                             f"Only quoting BID side to unwind")
        elif inventory_ratio > 0.60:
            # At 60-80%, heavily favor unwinding side but don't completely stop other side
            if self.inventory > 0:
                bid_size *= 0.2  # Drastically reduce bid size
                ask_size *= 1.5  # Increase ask size
            else:
                ask_size *= 0.2  # Drastically reduce ask size
                bid_size *= 1.5  # Increase bid size
        
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
        
        return quote
    
    def _calculate_quote_confidence(self, sigma: float, k: float) -> float:
        """
        Calculate confidence in the generated quote based on market conditions.
        
        FIX #4: Allow confidence to drop to 0.2 in truly bad conditions.
        This enables skipping quotes when market data is stale or extreme.
        
        Higher confidence when:
        - Volatility is stable (not too high/low)
        - Trade arrival rate is consistent
        - Recent market activity is normal
        """
        # Volatility confidence - be more realistic about extreme conditions
        if sigma < 1e-6:
            vol_confidence = 0.2  # Extremely low vol - possibly stale data
        elif sigma < 0.001:
            vol_confidence = 0.9  # Normal vol range (1-10 bps)
        elif sigma < 0.01:
            vol_confidence = 0.8  # Higher vol (10-100 bps) - still good
        elif sigma < 0.05:
            vol_confidence = 0.5  # Very high vol (100-500 bps) - be cautious
        else:
            vol_confidence = 0.2  # Extreme vol (>500 bps) - likely anomaly
        
        # Arrival rate confidence - penalize very low activity
        if k < 0.001:
            k_confidence = 0.2  # Almost no trades - market may be halted
        elif k < 0.01:
            k_confidence = 0.6  # Low activity (< 1 trade per 100 sec)
        elif k < 0.1:
            k_confidence = 0.8  # Moderate activity
        else:
            k_confidence = 0.9  # Good activity (> 1 trade per 10 sec)
        
        # Market data freshness - be stricter about stale data
        if self.timestamp:
            data_age = time.time() - self.timestamp
            if data_age > 60.0:
                freshness_confidence = 0.2  # Data older than 1 minute - don't quote
            elif data_age > 30.0:
                freshness_confidence = 0.5  # Data 30-60s old - be cautious
            elif data_age > 10.0:
                freshness_confidence = 0.7  # Data 10-30s old - acceptable
            else:
                freshness_confidence = 1.0  # Fresh data (< 10s)
        else:
            freshness_confidence = 0.3  # No timestamp - very suspicious
        
        # Combine factors with weighted average
        overall_confidence = (
            vol_confidence * 0.4 + 
            k_confidence * 0.3 + 
            freshness_confidence * 0.3
        )
        
        # FIX #4: Allow confidence as low as 0.2 (was 0.5 minimum)
        # Caller can use confidence < 0.5 threshold to skip quoting
        return max(0.2, min(1.0, overall_confidence))
    
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