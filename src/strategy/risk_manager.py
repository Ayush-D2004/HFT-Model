"""
Risk Management System for HFT Trading
=====================================

Comprehensive risk management with real-time monitoring and controls.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import numpy as np
from loguru import logger

from ..utils.config import config
from .avellaneda_stoikov import MarketQuote


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: float
    position: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    max_drawdown: float
    var_95: float  # 95% Value at Risk
    leverage: float
    position_utilization: float  # Position / Max Position
    risk_level: RiskLevel
    
    
@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position: float = 10.0
    max_daily_loss: float = 1000.0
    max_drawdown: float = 0.50  # 50% for backtesting (use 0.30 for live trading)
    max_leverage: float = 3.0
    max_var_95: float = 500.0
    latency_threshold_ms: float = 100.0
    min_quote_confidence: float = 0.5
    

class RiskManager:
    """
    Professional risk management system with real-time monitoring:
    
    Features:
    - Position and leverage limits
    - PnL tracking and drawdown control
    - Latency monitoring
    - Quote quality assessment
    - Emergency stop mechanisms
    - Risk reporting and alerts
    """
    
    def __init__(self, limits: RiskLimits = None, initial_capital: float = 10000.0):
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital
        
        # Position and PnL tracking
        self.current_position = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Performance tracking
        self.pnl_history: List[Tuple[float, float]] = []  # (timestamp, pnl)
        self.trade_history: List[Dict] = []
        self.fill_history: List[Tuple[float, float]] = []  # (timestamp, fill_size) for fill rate tracking
        self.latency_samples: List[float] = []
        
        # Risk state
        self.is_trading_enabled = True
        self.current_risk_level = RiskLevel.LOW
        self.risk_alerts: List[str] = []
        
        # Threading for real-time monitoring
        self._lock = threading.RLock()
        self._monitoring_active = True
        
        # Statistics
        self.stats = {
            'risk_checks': 0,
            'risk_violations': 0,
            'emergency_stops': 0,
            'quotes_blocked': 0
        }
        
        # Counter for logging frequency control
        self.risk_check_count = 0
        
        logger.info(f"RiskManager initialized with limits: {self.limits}")
    
    def update_position(self, 
                       position: float, 
                       avg_price: float, 
                       timestamp: float = None) -> None:
        """Update current position and average entry price"""
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            
            # Calculate realized PnL if position changed
            if position != self.current_position:
                if self.current_position != 0:
                    # Partial or full close
                    position_change = position - self.current_position
                    if (self.current_position > 0 and position_change < 0) or \
                       (self.current_position < 0 and position_change > 0):
                        # Closing position
                        closed_quantity = min(abs(position_change), abs(self.current_position))
                        if self.current_position > 0:
                            pnl_per_unit = avg_price - self.avg_entry_price
                        else:
                            pnl_per_unit = self.avg_entry_price - avg_price
                        
                        trade_pnl = closed_quantity * pnl_per_unit
                        self.realized_pnl += trade_pnl
                        self.daily_pnl += trade_pnl
                        
                        # Log trade
                        self.trade_history.append({
                            'timestamp': timestamp,
                            'side': 'sell' if position_change < 0 else 'buy',
                            'quantity': closed_quantity,
                            'price': avg_price,
                            'pnl': trade_pnl
                        })
            
            self.current_position = position
            self.avg_entry_price = avg_price
            
            logger.debug(f"Position updated: {position:.4f} @ {avg_price:.2f}")
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate current unrealized PnL"""
        if self.current_position == 0:
            return 0.0
        
        if self.current_position > 0:
            return self.current_position * (current_price - self.avg_entry_price)
        else:
            return abs(self.current_position) * (self.avg_entry_price - current_price)
    
    def update_pnl(self, current_price: float, timestamp: float = None) -> None:
        """Update PnL calculations and drawdown tracking"""
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            
            unrealized_pnl = self.calculate_unrealized_pnl(current_price)
            total_pnl = self.realized_pnl + unrealized_pnl
            
            # Update peak and drawdown
            if total_pnl > self.peak_equity:
                self.peak_equity = total_pnl
                self.current_drawdown = 0.0
            else:
                # Calculate drawdown as percentage of (initial_capital + peak_equity)
                # This prevents artificially high drawdowns when starting from 0 P&L
                equity_base = self.initial_capital + max(self.peak_equity, 0.0)
                self.current_drawdown = (self.peak_equity - total_pnl) / equity_base
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Store PnL history
            self.pnl_history.append((timestamp, total_pnl))
            
            # Keep only recent history (last 24 hours)
            cutoff_time = timestamp - 86400  # 24 hours
            self.pnl_history = [(t, pnl) for t, pnl in self.pnl_history if t >= cutoff_time]
    
    def record_latency(self, latency_ms: float) -> None:
        """Record latency measurement"""
        with self._lock:
            self.latency_samples.append(latency_ms)
            
            # Keep only recent samples (last 1000)
            if len(self.latency_samples) > 1000:
                self.latency_samples = self.latency_samples[-1000:]
    
    def record_fill(self, timestamp: float, fill_size: float) -> None:
        """
        Record a fill event for tracking fill frequency.
        Used for dynamic position limit calculation.
        """
        with self._lock:
            self.fill_history.append((timestamp, fill_size))
            
            # Keep only recent fills (last 1000 fills or 24 hours)
            cutoff_time = timestamp - 86400  # 24 hours
            self.fill_history = [(ts, size) for ts, size in self.fill_history 
                                if ts >= cutoff_time][-1000:]
    
    def check_quote_risk(self, quote: MarketQuote, current_price: float) -> bool:
        """
        Check if a quote is acceptable from risk perspective.
        Returns True if quote should be allowed, False to block.
        
        DYNAMIC POSITION LIMITS:
        - Calculates adaptive position limits based on recent trade frequency
        - Higher trade frequency → tighter position limits (true market making)
        - Lower trade frequency → wider limits (allow some directional exposure)
        - Works generically for any asset, not BTC-specific
        """
        with self._lock:
            self.stats['risk_checks'] += 1
            self.risk_check_count += 1  # Increment counter for logging frequency
            
            # Check if trading is enabled
            if not self.is_trading_enabled:
                self.stats['quotes_blocked'] += 1
                if self.stats['risk_checks'] % 1000 == 1:  # Log only once per 1000 checks
                    logger.warning("⚠️ Trading is DISABLED - quotes blocked")
                return False
            
            # Check quote confidence
            if quote.confidence < self.limits.min_quote_confidence:
                if self.stats['risk_checks'] % 20 == 0:
                    logger.warning(f"⚠️ Quote blocked: Low confidence {quote.confidence:.2f} < {self.limits.min_quote_confidence:.2f}")
                self.stats['quotes_blocked'] += 1
                return False
            
            # DYNAMIC POSITION LIMIT CALCULATION
            # Base limit from config, but adjust based on market making health
            base_limit = self.limits.max_position
            limit_multiplier = 1.0  # Default: use full base limit
            
            # Calculate recent trade frequency (trades per hour) if we have enough data
            if len(self.fill_history) >= 10:
                recent_fills = self.fill_history[-50:] if len(self.fill_history) >= 50 else self.fill_history
                
                if len(recent_fills) >= 2:  # Need at least 2 fills to calculate span
                    time_span = recent_fills[-1][0] - recent_fills[0][0]  # seconds
                    
                    if time_span > 0:
                        fills_per_hour = (len(recent_fills) / time_span) * 3600
                        
                        # TRUE HFT: 100+ trades/hour → tighten position limit to 20% of max
                        # SLOW MM: <10 trades/hour → allow up to 100% of max
                        # Formula: limit_multiplier = 0.2 + 0.8 * exp(-fills_per_hour / 50)
                        # This creates smooth decay:
                        #   - 10 fills/hour → 82% of max limit
                        #   - 50 fills/hour → 49% of max limit
                        #   - 100 fills/hour → 29% of max limit (tight HFT control)
                        limit_multiplier = 0.2 + 0.8 * np.exp(-fills_per_hour / 50.0)
                        
                        # Log position limit adjustment (only occasionally to avoid spam)
                        if self.stats['risk_checks'] % 100 == 0:
                            logger.debug(f"Dynamic position limit: {base_limit * limit_multiplier:.4f} "
                                       f"(base: {base_limit:.4f}, multiplier: {limit_multiplier:.2f}, "
                                       f"fills/hr: {fills_per_hour:.1f})")
            
            dynamic_limit = base_limit * limit_multiplier
            
            # Check position limits with dynamic adjustment
            # ✅ CRITICAL: Check if accepting THIS QUOTE'S FILL would violate position limits
            # This is DIFFERENT from blocking quotes on the heavy side (which comes later)
            # Here we ensure the quote sizes themselves don't violate limits
            
            # Calculate what position would be AFTER potential fills
            potential_long_position = self.current_position + quote.bid_size  # If bid fills
            potential_short_position = self.current_position - quote.ask_size  # If ask fills
            
            # Check if EITHER potential position would exceed limits
            max_allowed_position = dynamic_limit
            
            if abs(potential_long_position) > max_allowed_position:
                # Bid quote too large - would push position over limit if filled
                if self.stats['risk_checks'] % 100 == 0:
                    logger.warning(f"⚠️ Bid quote {quote.bid_size:.4f} would create position "
                                 f"{potential_long_position:.4f} > limit {max_allowed_position:.4f}. "
                                 f"Current position: {self.current_position:.4f}")
                # Adjust bid size to stay within limit
                quote.bid_size = max(0.0, max_allowed_position - self.current_position)
                if quote.bid_size < self.pricer.lot_size:
                    # Can't place minimum size order without violating limit
                    return False
            
            if abs(potential_short_position) > max_allowed_position:
                # Ask quote too large - would push position over limit if filled  
                if self.stats['risk_checks'] % 100 == 0:
                    logger.warning(f"⚠️ Ask quote {quote.ask_size:.4f} would create position "
                                 f"{potential_short_position:.4f} > limit {max_allowed_position:.4f}. "
                                 f"Current position: {self.current_position:.4f}")
                # Adjust ask size to stay within limit
                quote.ask_size = max(0.0, max_allowed_position + self.current_position)
                if quote.ask_size < self.pricer.lot_size:
                    # Can't place minimum size order without violating limit
                    return False
            
            # Original checks for quote sizes vs dynamic limits (keep for backward compatibility)
            max_bid_size = dynamic_limit - self.current_position
            max_ask_size = dynamic_limit + self.current_position
            
            if quote.bid_size > max_bid_size and max_bid_size > 0:
                if self.stats['risk_checks'] % 20 == 0:
                    logger.warning(f"⚠️ Bid size {quote.bid_size:.4f} > dynamic limit {max_bid_size:.4f} "
                                 f"(position={self.current_position:.4f}, dynamic_limit={dynamic_limit:.4f})")
                return False
            
            if quote.ask_size > max_ask_size and max_ask_size > 0:
                if self.stats['risk_checks'] % 20 == 0:
                    logger.warning(f"⚠️ Ask size {quote.ask_size:.4f} > dynamic limit {max_ask_size:.4f} "
                                 f"(position={self.current_position:.4f}, dynamic_limit={dynamic_limit:.4f})")
                return False
            
            # ✅ FIX #12: CONSOLIDATED SINGLE ADAPTIVE POSITION LIMIT
            # Combines percentage, notional, and dynamic trade frequency into one check
            
            # Calculate position metrics
            position_ratio = abs(self.current_position) / self.limits.max_position
            notional_value = abs(self.current_position) * current_price
            
            # ADAPTIVE NOTIONAL LIMIT based on trade frequency
            # Fast trading (>50 fills/hr): Tighter limit (true HFT)
            # Slow trading (<10 fills/hr): Looser limit (allow some directional)
            base_notional_limit = 500000  # $500k baseline - increased from $5k for realistic HFT volume
            
            if len(self.fill_history) >= 10:
                recent_fills = self.fill_history[-50:] if len(self.fill_history) >= 50 else self.fill_history
                
                if len(recent_fills) >= 2:
                    time_span = recent_fills[-1][0] - recent_fills[0][0]
                    
                    if time_span > 0:
                        fills_per_hour = (len(recent_fills) / time_span) * 3600
                        
                        # Adjust notional limit based on trade frequency
                        # High frequency → tighter limit (true MM)
                        # Low frequency → looser limit (allow some directional exposure)
                        if fills_per_hour > 50:
                            # Very fast trading: use $300k
                            notional_limit = 300000
                        elif fills_per_hour > 20:
                            # Fast trading: use $400k
                            notional_limit = 400000
                        elif fills_per_hour > 10:
                            # Moderate trading: use $500k baseline
                            notional_limit = base_notional_limit
                        else:
                            # Slow trading: allow up to $600k
                            notional_limit = 600000
                    else:
                        notional_limit = base_notional_limit
                else:
                    notional_limit = base_notional_limit
            else:
                notional_limit = base_notional_limit
            
            # SINGLE CONSOLIDATED CHECK: Block if EITHER threshold exceeded
            # 🚀 PROFESSIONAL HFT: Allow quotes up to 70% of max position (was 30%)
            # Market makers NEED inventory to provide liquidity
            # Only block quotes that would INCREASE position beyond limit
            # ALWAYS allow quotes on the reducing side (flatten inventory)
            
            if position_ratio > 0.70:  # 🔧 INCREASED FROM 30% TO 70% for real HFT
                # Reduced logging frequency to prevent terminal spam
                if self.risk_check_count % 500 == 0:
                    logger.warning(f"⚠️ POSITION LIMIT (70%) EXCEEDED: {position_ratio*100:.1f}% of max position. "
                                 f"Notional: ${notional_value:.0f}. Blocking heavy side only.")
                
                # 🔧 CRITICAL: Block quotes that would INCREASE position
                # If long (position > 0), block BID (buy) quotes - they increase long position
                # If short (position < 0), block ASK (sell) quotes - they increase short position
                # ALWAYS allow the opposite side (reduces inventory)
                if self.current_position > 0:  # Long position
                    # Block entire quote if position too high
                    # TODO: In future, could allow ASK side only (to reduce position)
                    self.stats['quotes_blocked'] += 1
                    self.stats['position_limit_blocks'] = self.stats.get('position_limit_blocks', 0) + 1
                    logger.debug(f"Position limit: blocking quote (long position {self.current_position:.4f})")
                    return False  # BLOCK THE QUOTE
                elif self.current_position < 0:  # Short position
                    # Block entire quote if position too high
                    self.stats['quotes_blocked'] += 1
                    self.stats['position_limit_blocks'] = self.stats.get('position_limit_blocks', 0) + 1
                    logger.debug(f"Position limit: blocking quote (short position {self.current_position:.4f})")
                    return False  # BLOCK THE QUOTE
            
            elif notional_value > notional_limit:
                # 🔧 PROFESSIONAL HFT: Log but don't hard-block on notional
                # Notional is adaptive based on trading frequency
                # Allow quoting, just log for monitoring
                if self.risk_check_count % 500 == 0:
                    logger.warning(f"⚠️ NOTIONAL LIMIT EXCEEDED: ${notional_value:.0f} > ${notional_limit:.0f} "
                               f"(adaptive limit based on trading frequency). Allowing quotes with monitoring.")
                
                # Log but DON'T block - just track for statistics
                self.stats['notional_warnings'] = self.stats.get('notional_warnings', 0) + 1
                # Return True to allow quote (don't block HFT)
            
            # Warning zone (80% of limits) - log very rarely
            elif position_ratio > 0.56 or notional_value > notional_limit * 0.8:  # 🔧 INCREASED from 0.24 to 0.56 (80% of new 70% limit)
                if self.risk_check_count % 1000 == 0:
                    logger.warning(f"⚠️ POSITION WARNING: {position_ratio*100:.1f}% of max, "
                             f"${notional_value:.0f} notional (limit: ${notional_limit:.0f}). "
                             f"Approaching limits.")
                self.stats['position_warnings'] = self.stats.get('position_warnings', 0) + 1
            
            # Check drawdown limits
            if self.current_drawdown > self.limits.max_drawdown:
                logger.error(f"Drawdown limit exceeded: {self.current_drawdown:.3f} > {self.limits.max_drawdown:.3f}")
                self._trigger_emergency_stop("Drawdown limit exceeded")
                return False
            
            # Check daily loss limits
            if self.daily_pnl < -self.limits.max_daily_loss:
                logger.error(f"Daily loss limit exceeded: {self.daily_pnl:.2f} < -{self.limits.max_daily_loss:.2f}")
                self._trigger_emergency_stop("Daily loss limit exceeded")
                return False
            
            return True
    
    def assess_risk_level(self, current_price: float) -> RiskLevel:
        """Assess current risk level based on multiple factors"""
        with self._lock:
            risk_score = 0.0
            
            # Position utilization risk
            position_util = abs(self.current_position) / self.limits.max_position
            risk_score += position_util * 30
            
            # Drawdown risk
            risk_score += self.current_drawdown / self.limits.max_drawdown * 40
            
            # Daily PnL risk
            if self.daily_pnl < 0:
                pnl_risk = abs(self.daily_pnl) / self.limits.max_daily_loss
                risk_score += pnl_risk * 30
            
            # Latency risk
            if self.latency_samples:
                avg_latency = sum(self.latency_samples[-10:]) / min(len(self.latency_samples), 10)
                if avg_latency > self.limits.latency_threshold_ms:
                    risk_score += (avg_latency / self.limits.latency_threshold_ms - 1) * 20
            
            # Determine risk level
            if risk_score < 25:
                return RiskLevel.LOW
            elif risk_score < 50:
                return RiskLevel.MEDIUM
            elif risk_score < 75:
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL
    
    def _trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop mechanism"""
        with self._lock:
            self.is_trading_enabled = False
            self.current_risk_level = RiskLevel.CRITICAL
            self.risk_alerts.append(f"{time.time()}: EMERGENCY STOP - {reason}")
            self.stats['emergency_stops'] += 1
            
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
    
    def enable_trading(self, force: bool = False) -> bool:
        """
        Enable trading after risk checks.
        Returns True if trading enabled successfully.
        """
        with self._lock:
            if not force:
                # Check if conditions allow re-enabling
                if self.current_drawdown > self.limits.max_drawdown * 0.8:
                    logger.warning("Cannot enable trading: drawdown too high")
                    return False
                
                if self.daily_pnl < -self.limits.max_daily_loss * 0.8:
                    logger.warning("Cannot enable trading: daily loss too high")
                    return False
            
            self.is_trading_enabled = True
            self.risk_alerts.append(f"{time.time()}: Trading enabled")
            logger.info("Trading enabled")
            return True
    
    def disable_trading(self, reason: str = "Manual disable") -> None:
        """Disable trading manually"""
        with self._lock:
            self.is_trading_enabled = False
            self.risk_alerts.append(f"{time.time()}: Trading disabled - {reason}")
            logger.warning(f"Trading disabled: {reason}")
    
    def get_risk_metrics(self, current_price: float) -> RiskMetrics:
        """Get comprehensive risk metrics snapshot"""
        with self._lock:
            unrealized_pnl = self.calculate_unrealized_pnl(current_price)
            
            # Calculate leverage (if applicable)
            leverage = 1.0  # Spot trading
            if hasattr(self, 'margin_used') and self.margin_used > 0:
                leverage = abs(self.current_position * current_price) / self.margin_used
            
            # Calculate VaR (simplified using standard deviation)
            var_95 = 0.0
            if len(self.pnl_history) > 10:
                recent_pnl = [pnl for _, pnl in self.pnl_history[-100:]]
                pnl_std = np.std(recent_pnl) if len(recent_pnl) > 1 else 0
                var_95 = 1.645 * pnl_std  # 95% confidence level
            
            position_util = abs(self.current_position) / self.limits.max_position
            risk_level = self.assess_risk_level(current_price)
            
            return RiskMetrics(
                timestamp=time.time(),
                position=self.current_position,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self.realized_pnl,
                daily_pnl=self.daily_pnl,
                max_drawdown=self.max_drawdown,
                var_95=var_95,
                leverage=leverage,
                position_utilization=position_util,
                risk_level=risk_level
            )
    
    def get_statistics(self) -> Dict:
        """Get risk management statistics"""
        with self._lock:
            # Calculate latency statistics
            latency_stats = {}
            if self.latency_samples:
                latency_stats = {
                    'avg_latency_ms': sum(self.latency_samples) / len(self.latency_samples),
                    'p95_latency_ms': np.percentile(self.latency_samples, 95),
                    'p99_latency_ms': np.percentile(self.latency_samples, 99),
                    'max_latency_ms': max(self.latency_samples)
                }
            
            return {
                **self.stats,
                'is_trading_enabled': self.is_trading_enabled,
                'current_risk_level': self.current_risk_level.value,
                'current_position': self.current_position,
                'realized_pnl': self.realized_pnl,
                'daily_pnl': self.daily_pnl,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'trade_count': len(self.trade_history),
                'alert_count': len(self.risk_alerts),
                **latency_stats
            }
    
    def reset_daily(self) -> None:
        """Reset daily statistics (call at start of each trading day)"""
        with self._lock:
            self.daily_pnl = 0.0
            self.trade_history.clear()
            
            # Keep only recent risk alerts
            current_time = time.time()
            self.risk_alerts = [
                alert for alert in self.risk_alerts 
                if float(alert.split(':')[0]) > current_time - 86400
            ]
            
            logger.info("Daily risk statistics reset")
    
    def export_report(self) -> Dict:
        """Export comprehensive risk report"""
        with self._lock:
            return {
                'timestamp': time.time(),
                'limits': {
                    'max_position': self.limits.max_position,
                    'max_daily_loss': self.limits.max_daily_loss,
                    'max_drawdown': self.limits.max_drawdown,
                    'max_leverage': self.limits.max_leverage
                },
                'current_state': {
                    'position': self.current_position,
                    'avg_entry_price': self.avg_entry_price,
                    'realized_pnl': self.realized_pnl,
                    'daily_pnl': self.daily_pnl,
                    'max_drawdown': self.max_drawdown,
                    'is_trading_enabled': self.is_trading_enabled
                },
                'statistics': self.get_statistics(),
                'recent_trades': self.trade_history[-10:],
                'recent_alerts': self.risk_alerts[-5:]
            }


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Initialize risk manager
    limits = RiskLimits(
        max_position=5.0,
        max_daily_loss=500.0,
        max_drawdown=0.03
    )
    
    risk_manager = RiskManager(limits)
    
    # Simulate some trading activity
    current_price = 50000.0
    
    # Update position
    risk_manager.update_position(2.5, 49950.0)
    risk_manager.update_pnl(current_price)
    
    # Record some latency samples
    for _ in range(10):
        risk_manager.record_latency(np.random.normal(50, 10))
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics(current_price)
    print(f"Risk Level: {metrics.risk_level.value}")
    print(f"Position Utilization: {metrics.position_utilization:.2f}")
    print(f"Unrealized PnL: {metrics.unrealized_pnl:.2f}")
    
    # Test quote risk check
    from .avellaneda_stoikov import MarketQuote
    
    test_quote = MarketQuote(
        bid_price=49995.0,
        ask_price=50005.0,
        bid_size=1.0,
        ask_size=1.0,
        reservation_price=50000.0,
        half_spread=5.0,
        timestamp=time.time(),
        confidence=0.8
    )
    
    allowed = risk_manager.check_quote_risk(test_quote, current_price)
    print(f"Quote allowed: {allowed}")
    
    # Print statistics
    print(f"Statistics: {risk_manager.get_statistics()}")