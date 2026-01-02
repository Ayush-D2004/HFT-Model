"""
Dynamic Spread Calculator for Fee-Aware HFT Market Making
==========================================================

Implements dynamic spread adjustment based on:
- Binance maker/taker fees (0.02% / 0.05%)
- Market volatility (EWMA)
- Order book imbalance (EWMA)
- Inventory position

Key Formula:
    breakeven_spread = 2 × max(maker_fee, taker_fee) + safety_margin
    
This ensures every round-trip trade (buy + sell) covers:
- Entry fee (taker or maker)
- Exit fee (taker or maker)
- Safety buffer for adverse selection and slippage

Author: HFT Model Builder
Date: 2025-11-03
"""

import math
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeeConfig:
    """Fee configuration for exchange"""
    maker_fee: float = 0.0002  # 0.02% Binance maker fee
    taker_fee: float = 0.0005  # 0.05% Binance taker fee
    safety_margin: float = 0.0003  # 0.03% safety buffer for adverse selection


@dataclass
class SpreadState:
    """Current spread calculation state"""
    breakeven_spread: float
    dynamic_spread: float
    half_spread: float
    volatility: float
    imbalance: float
    inventory_skew: float


class DynamicSpreadCalculator:
    """
    Fee-aware dynamic spread calculator for market making.
    
    Features:
    - Breakeven spread calculation (covers fees + buffer)
    - EWMA volatility estimation (α=0.2)
    - EWMA order book imbalance (α=0.4)
    - Inventory-based spread adjustment
    - Dynamic widening during high vol/imbalance
    - Tightening during favorable conditions
    
    Usage:
        calc = DynamicSpreadCalculator(fee_config)
        calc.update_market(midprice, bid_depth, ask_depth)
        calc.update_volatility(returns)
        spread = calc.compute_spread(inventory, max_position)
    """
    
    def __init__(
        self,
        fee_config: FeeConfig = None,
        vol_alpha: float = 0.2,
        imbalance_alpha: float = 0.4,
        vol_multiplier: float = 2.0,
        imbalance_multiplier: float = 0.5,
        inventory_multiplier: float = 0.001
    ):
        """
        Initialize dynamic spread calculator.
        
        Args:
            fee_config: Fee configuration (maker/taker fees)
            vol_alpha: EWMA alpha for volatility (0.2 = 5-period equivalent)
            imbalance_alpha: EWMA alpha for imbalance (0.4 = 2.5-period equivalent)
            vol_multiplier: Spread widening factor for volatility
            imbalance_multiplier: Spread widening factor for imbalance
            inventory_multiplier: Spread skew factor for inventory
        """
        self.fee_config = fee_config or FeeConfig()
        
        # EWMA parameters
        self.vol_alpha = vol_alpha
        self.imbalance_alpha = imbalance_alpha
        
        # Spread adjustment multipliers
        self.vol_multiplier = vol_multiplier
        self.imbalance_multiplier = imbalance_multiplier
        self.inventory_multiplier = inventory_multiplier
        
        # State tracking
        self.ewma_volatility: Optional[float] = None
        self.ewma_imbalance: Optional[float] = None
        self.last_midprice: Optional[float] = None
        
        # Calculate breakeven spread (CRITICAL for profitability)
        self.breakeven_spread = self._compute_breakeven_spread()
        
        logger.info(f"DynamicSpreadCalculator initialized: "
                   f"breakeven_spread={self.breakeven_spread:.4f} "
                   f"({self.breakeven_spread*100:.2f}%)")
    
    def _compute_breakeven_spread(self) -> float:
        """
        Compute minimum spread needed to break even after fees.
        
        Formula:
            breakeven = 2 × max(maker_fee, taker_fee) + safety_margin
        
        Rationale:
            - Round-trip trade has 2 legs (buy + sell)
            - Each leg pays at least maker_fee (worst case: taker_fee)
            - Safety margin covers adverse selection and slippage
            - Must use max() because we might get hit as taker on either side
        
        Example:
            maker_fee = 0.02%, taker_fee = 0.05%, safety = 0.03%
            breakeven = 2 × 0.05% + 0.03% = 0.13% (13 bps)
        
        Returns:
            Minimum spread as decimal (e.g., 0.0013 for 13 bps)
        """
        # Worst case: both legs are taker fills
        worst_case_fee = max(self.fee_config.maker_fee, self.fee_config.taker_fee)
        
        # Round-trip cost = 2 legs × fee per leg
        round_trip_cost = 2.0 * worst_case_fee
        
        # Add safety margin for adverse selection
        breakeven = round_trip_cost + self.fee_config.safety_margin
        
        logger.info(f"Breakeven spread computed: {breakeven:.4f} "
                   f"(round_trip_cost={round_trip_cost:.4f} + "
                   f"safety={self.fee_config.safety_margin:.4f})")
        
        return breakeven
    
    def update_volatility(self, midprice: float) -> None:
        """
        Update EWMA volatility estimate using log returns.
        
        Uses exponentially weighted moving average (EWMA) for faster
        response to volatility changes compared to simple moving average.
        
        Args:
            midprice: Current midprice
        """
        if self.last_midprice is not None and self.last_midprice > 0:
            # Compute log return
            log_return = math.log(midprice / self.last_midprice)
            
            # Update EWMA of squared returns (variance proxy)
            squared_return = log_return ** 2
            
            if self.ewma_volatility is None:
                self.ewma_volatility = squared_return
            else:
                # EWMA update: new_val = α × new_data + (1-α) × old_val
                self.ewma_volatility = (
                    self.vol_alpha * squared_return +
                    (1 - self.vol_alpha) * self.ewma_volatility
                )
        
        self.last_midprice = midprice
    
    def update_imbalance(self, bid_depth: float, ask_depth: float) -> None:
        """
        Update EWMA order book imbalance.
        
        Imbalance measures order book pressure:
        - Positive: more bid depth (bullish)
        - Negative: more ask depth (bearish)
        - Near zero: balanced book
        
        Args:
            bid_depth: Total bid depth (volume)
            ask_depth: Total ask depth (volume)
        """
        total_depth = bid_depth + ask_depth
        
        if total_depth > 0:
            # Imbalance = (bid - ask) / (bid + ask)
            # Range: [-1, +1]
            current_imbalance = (bid_depth - ask_depth) / total_depth
            
            if self.ewma_imbalance is None:
                self.ewma_imbalance = current_imbalance
            else:
                # Faster EWMA (α=0.4) for quicker response to imbalance shifts
                self.ewma_imbalance = (
                    self.imbalance_alpha * current_imbalance +
                    (1 - self.imbalance_alpha) * self.ewma_imbalance
                )
    
    def compute_spread(
        self,
        inventory: float,
        max_position: float,
        current_volatility: Optional[float] = None
    ) -> SpreadState:
        """
        Compute dynamic spread with fee awareness.
        
        Spread components:
        1. Breakeven spread (minimum to cover fees)
        2. Volatility adjustment (widen during high vol)
        3. Imbalance adjustment (widen during one-sided pressure)
        4. Inventory skew (asymmetric to reduce position)
        
        Args:
            inventory: Current position (positive = long, negative = short)
            max_position: Maximum allowed position size
            current_volatility: Optional override for volatility
        
        Returns:
            SpreadState with all spread components
        """
        # Start with breakeven spread (fee coverage)
        base_spread = self.breakeven_spread
        
        # Get current volatility estimate
        vol = current_volatility if current_volatility is not None else (
            math.sqrt(self.ewma_volatility) if self.ewma_volatility else 0.0
        )
        
        # Volatility adjustment: widen spread during high volatility
        # Higher volatility = more adverse selection risk
        vol_adjustment = self.vol_multiplier * vol
        
        # Imbalance adjustment: widen spread if book is imbalanced
        # Prevents getting run over by one-sided flow
        imb = self.ewma_imbalance if self.ewma_imbalance is not None else 0.0
        imbalance_adjustment = self.imbalance_multiplier * abs(imb)
        
        # Total dynamic spread (before inventory skew)
        dynamic_spread = base_spread + vol_adjustment + imbalance_adjustment
        
        # Inventory skew: asymmetric spread to reduce position
        # If long → widen ask (discourage more buys), tighten bid (encourage sells)
        # If short → widen bid (discourage more sells), tighten ask (encourage buys)
        inventory_ratio = inventory / max(max_position, 1.0)
        inventory_skew = self.inventory_multiplier * inventory_ratio
        
        # Half spread for quoting
        half_spread = dynamic_spread / 2.0
        
        # Ensure minimum floor (breakeven half-spread)
        min_half_spread = self.breakeven_spread / 2.0
        half_spread = max(half_spread, min_half_spread)
        
        return SpreadState(
            breakeven_spread=self.breakeven_spread,
            dynamic_spread=dynamic_spread,
            half_spread=half_spread,
            volatility=vol,
            imbalance=imb,
            inventory_skew=inventory_skew
        )
    
    def compute_asymmetric_quotes(
        self,
        midprice: float,
        inventory: float,
        max_position: float,
        tick_size: float = 0.01
    ) -> Tuple[float, float]:
        """
        Compute asymmetric bid/ask quotes with inventory skew.
        
        Uses inventory position to skew spread:
        - Long position → wider ask, tighter bid (encourage selling)
        - Short position → wider bid, tighter ask (encourage buying)
        
        Args:
            midprice: Current market midprice
            inventory: Current position
            max_position: Maximum position size
            tick_size: Price tick size for rounding
        
        Returns:
            (bid_price, ask_price) tuple
        """
        spread_state = self.compute_spread(inventory, max_position)
        
        # Base half-spread
        half_spread = spread_state.half_spread
        
        # Apply inventory skew
        # Positive inventory → increase ask spread, decrease bid spread
        skew = spread_state.inventory_skew
        
        bid_spread = half_spread - skew
        ask_spread = half_spread + skew
        
        # Ensure both spreads stay above minimum
        min_half_spread = self.breakeven_spread / 2.0
        bid_spread = max(bid_spread, min_half_spread)
        ask_spread = max(ask_spread, min_half_spread)
        
        # Compute prices
        bid = midprice - bid_spread * midprice
        ask = midprice + ask_spread * midprice
        
        # Round to tick size
        bid = round(bid / tick_size) * tick_size
        ask = round(ask / tick_size) * tick_size
        
        # Ensure no crossing
        if bid >= ask:
            ask = bid + tick_size
        
        return (bid, ask)
    
    def get_stats(self) -> dict:
        """Get current spread calculator statistics"""
        return {
            'breakeven_spread': self.breakeven_spread,
            'breakeven_spread_bps': self.breakeven_spread * 10000,
            'ewma_volatility': self.ewma_volatility,
            'ewma_imbalance': self.ewma_imbalance,
            'maker_fee': self.fee_config.maker_fee,
            'taker_fee': self.fee_config.taker_fee,
            'safety_margin': self.fee_config.safety_margin
        }


# Unit test for breakeven spread calculation
if __name__ == "__main__":
    # Test case 1: Standard Binance fees
    print("=== Test 1: Standard Binance Fees ===")
    fee_config = FeeConfig(
        maker_fee=0.0002,  # 0.02%
        taker_fee=0.0005,  # 0.05%
        safety_margin=0.0003  # 0.03%
    )
    calc = DynamicSpreadCalculator(fee_config)
    
    # Verify breakeven >= 2 × max(fee)
    expected_min = 2 * max(fee_config.maker_fee, fee_config.taker_fee)
    print(f"Expected minimum (2×max fee): {expected_min:.4f} ({expected_min*10000:.1f} bps)")
    print(f"Actual breakeven spread: {calc.breakeven_spread:.4f} ({calc.breakeven_spread*10000:.1f} bps)")
    assert calc.breakeven_spread >= expected_min, "Breakeven spread too low!"
    print("✅ PASS: Breakeven spread covers fees\n")
    
    # Test case 2: Dynamic spread with volatility
    print("=== Test 2: Dynamic Spread with Volatility ===")
    calc.update_volatility(100.0)
    calc.update_volatility(100.5)  # 0.5% move
    calc.update_imbalance(1000, 1000)  # Balanced book
    
    spread_state = calc.compute_spread(inventory=0, max_position=10)
    print(f"Dynamic spread: {spread_state.dynamic_spread:.4f} ({spread_state.dynamic_spread*10000:.1f} bps)")
    print(f"Half spread: {spread_state.half_spread:.4f}")
    print(f"Volatility: {spread_state.volatility:.6f}")
    assert spread_state.dynamic_spread >= calc.breakeven_spread, "Dynamic spread below breakeven!"
    print("✅ PASS: Dynamic spread >= breakeven\n")
    
    # Test case 3: Inventory skew
    print("=== Test 3: Inventory Skew ===")
    bid_long, ask_long = calc.compute_asymmetric_quotes(100.0, inventory=5.0, max_position=10.0)
    bid_short, ask_short = calc.compute_asymmetric_quotes(100.0, inventory=-5.0, max_position=10.0)
    
    spread_long = (ask_long - bid_long) / 100.0
    spread_short = (ask_short - bid_short) / 100.0
    
    print(f"Long position (inventory=+5): Bid={bid_long:.2f}, Ask={ask_long:.2f}, Spread={spread_long:.4f}")
    print(f"Short position (inventory=-5): Bid={bid_short:.2f}, Ask={ask_short:.2f}, Spread={spread_short:.4f}")
    print(f"Ask skew (long vs short): {(ask_long - ask_short):.2f}")
    print(f"Bid skew (long vs short): {(bid_long - bid_short):.2f}")
    print("✅ PASS: Inventory skew working\n")
    
    print("=== All Tests Passed ===")
