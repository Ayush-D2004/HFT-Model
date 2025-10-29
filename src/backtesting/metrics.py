"""
Backtesting Metrics and Performance Analysis
==========================================

Comprehensive metrics calculation for strategy evaluation including:
- PnL analysis (realized/unrealized, Sharpe, Sortino, drawdown)
- Trading metrics (fill rate, quote lifetime, adverse selection)
- Risk metrics and performance attribution
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

# Use standard logging instead of loguru
logger = logging.getLogger(__name__)

from .fill_simulator_fifo import FillEvent
from src.strategy import Order, OrderSide


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # PnL Metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    gross_pnl: float
    net_pnl: float
    total_return_pct: float  # Total return as percentage
    
    # Returns and Risk
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: float
    volatility: float
    
    # Trading Metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_trade_pnl: float  # Average PnL per trade
    profit_factor: float
    
    # Market Making Metrics
    fill_rate: float
    quote_hit_rate: float
    avg_spread_captured: float
    inventory_turnover: float
    adverse_selection_rate: float
    
    # Execution Metrics
    avg_fill_latency_ms: float
    total_fees: float
    fee_rate: float
    total_volume: float  # Total traded volume in USDT
    
    # Time-based Metrics
    start_time: float
    end_time: float
    duration_hours: float
    
    # Chart Data - for dashboard visualization
    pnl_history: List[float] = None  # Cumulative P&L over time
    timestamps: List[float] = None   # Timestamps for P&L history
    
    
class BacktestMetrics:
    """
    Professional backtesting metrics calculator with institutional-grade analytics:
    
    Features:
    - Real-time PnL tracking and attribution
    - Risk-adjusted performance metrics
    - Market making specific analytics
    - Trade execution analysis
    - Drawdown and risk monitoring
    - Performance attribution
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        
        # Core tracking
        self.fill_events: List[FillEvent] = []
        self.pnl_series: List[Tuple[float, float]] = []  # (timestamp, pnl)
        self.position_series: List[Tuple[float, float]] = []  # (timestamp, position)
        self.quote_events: List[Dict] = []
        
        # Current state
        self.current_position = 0.0
        self.current_cash = initial_capital
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        
        # Trade tracking
        self.trades: List[Dict] = []  # Completed round-trip trades
        self.open_positions: Dict[str, Dict] = {}  # Partially filled positions
        
        # Quote and execution tracking
        self.quote_updates = 0
        self.quote_hits = 0
        self.fill_latencies: List[float] = []
        self.spread_captures: List[float] = []
        
        # Performance cache
        self._metrics_cache: Optional[PerformanceMetrics] = None
        self._cache_timestamp = 0.0
        
        logger.info(f"BacktestMetrics initialized with capital: {initial_capital:,.2f}")
    
    def record_fill(self, fill_event: FillEvent, current_price: float) -> None:
        """
        🚀 PROFESSIONAL HFT: Record EVERY fill as a trade (industry standard).
        
        Real HFT market makers count every fill as a separate trade, not round-trips.
        This gives accurate trade count, frequency, and P&L attribution.
        
        Args:
            fill_event: The fill event to record
            current_price: Current market price for unrealized PnL calculation
        """
        try:
            self.fill_events.append(fill_event)
            
            # Update position and cash
            if fill_event.side == OrderSide.BID:
                # Buy order filled
                position_change = fill_event.fill_quantity
                cash_change = -(fill_event.fill_quantity * fill_event.fill_price + fill_event.fee)
            else:
                # Sell order filled
                position_change = -fill_event.fill_quantity
                cash_change = fill_event.fill_quantity * fill_event.fill_price - fill_event.fee
            
            # 🚀 PROFESSIONAL HFT: Record EVERY fill as a trade immediately
            # Calculate instantaneous P&L based on current market vs fill price
            if fill_event.side == OrderSide.BID:
                # Bought at fill_price, current market value at current_price
                instant_pnl = fill_event.fill_quantity * (current_price - fill_event.fill_price)
            else:
                # Sold at fill_price, current market value at current_price
                instant_pnl = fill_event.fill_quantity * (fill_event.fill_price - current_price)
            
            # Record this fill as a trade (industry standard for HFT)
            self._record_trade(fill_event, fill_event.fill_quantity, instant_pnl)
            
            # Calculate new average entry price for inventory tracking (separate from trade counting)
            if self.current_position + position_change != 0:
                if self.current_position == 0:
                    # Opening new position
                    new_avg_price = fill_event.fill_price
                elif (self.current_position > 0 and position_change > 0) or \
                     (self.current_position < 0 and position_change < 0):
                    # Adding to existing position - weighted average
                    total_cost = (self.current_position * self.avg_entry_price + 
                                position_change * fill_event.fill_price)
                    new_avg_price = total_cost / (self.current_position + position_change)
                else:
                    # Reducing position - realize some PnL
                    closed_quantity = min(abs(position_change), abs(self.current_position))
                    
                    # Calculate trade P&L (price difference only, fees tracked separately)
                    if self.current_position > 0:
                        # Closing long position
                        trade_pnl = closed_quantity * (fill_event.fill_price - self.avg_entry_price)
                    else:
                        # Closing short position  
                        trade_pnl = closed_quantity * (self.avg_entry_price - fill_event.fill_price)
                    
                    self.realized_pnl += trade_pnl
                    
                    # Update average price (unchanged if reducing position)
                    new_avg_price = self.avg_entry_price
            else:
                # Position going to zero (full close)
                if self.current_position != 0:
                    closed_quantity = abs(self.current_position)
                    
                    # Calculate trade P&L (price difference only)
                    if self.current_position > 0:
                        trade_pnl = closed_quantity * (fill_event.fill_price - self.avg_entry_price)
                    else:
                        trade_pnl = closed_quantity * (self.avg_entry_price - fill_event.fill_price)
                    
                    self.realized_pnl += trade_pnl
                
                new_avg_price = 0.0
            
            # Update state
            self.current_position += position_change
            self.current_cash += cash_change
            self.avg_entry_price = new_avg_price
            self.total_fees += fill_event.fee
            
            # Record execution metrics
            self.fill_latencies.append(fill_event.latency_ms)
            
            # Calculate spread capture if this is a market making fill
            if fill_event.is_maker:
                # Estimate spread captured (simplified - assumes we were at mid when quote was placed)
                spread_capture = abs(fill_event.fill_price - current_price)
                self.spread_captures.append(spread_capture)
            
            # Update PnL series
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            total_pnl = self.realized_pnl + unrealized_pnl
            self.pnl_series.append((fill_event.timestamp, total_pnl))
            self.position_series.append((fill_event.timestamp, self.current_position))
            
            # Invalidate cache
            self._metrics_cache = None
            
            logger.debug(f"Fill recorded: {fill_event.side.value} {fill_event.fill_quantity:.4f} "
                        f"@ {fill_event.fill_price:.2f}, PnL: {total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording fill: {e}")
    
    def record_quote_update(self, timestamp: float, was_hit: bool = False) -> None:
        """Record quote update event"""
        self.quote_updates += 1
        if was_hit:
            self.quote_hits += 1
        
        self.quote_events.append({
            'timestamp': timestamp,
            'was_hit': was_hit,
            'quote_number': self.quote_updates
        })
    
    def _record_trade(self, fill_event: FillEvent, quantity: float, pnl: float) -> None:
        """
        Record a completed trade with ENHANCED CONTEXT for outlier diagnosis.
        
        Captures all 4 diagnostic variables from outlier framework:
        - fill_price vs midprice (ΔMid)
        - timestamp gaps (Δt)
        - inventory changes (notional)
        - spread context (for P&L ratio)
        """
        # Get current market context if available
        midprice = fill_event.fill_price  # Fallback
        spread = 0.001 * fill_event.fill_price  # Default 10 bps
        
        # Try to extract from last PnL series point (has market context)
        if len(self.pnl_series) > 0:
            # Midprice is approximated from recent trades
            recent_prices = [self.trades[i]['exit_price'] for i in range(max(0, len(self.trades)-10), len(self.trades))]
            if recent_prices:
                midprice = np.median(recent_prices)
        
        # Calculate time gap from previous trade
        prev_timestamp = self.trades[-1]['timestamp'] if self.trades else fill_event.timestamp
        time_gap = fill_event.timestamp - prev_timestamp
        
        # Enhanced trade record with diagnostic context
        trade = {
            'timestamp': fill_event.timestamp,
            'side': 'close_long' if fill_event.side == OrderSide.ASK else 'close_short',
            'quantity': quantity,
            'entry_price': self.avg_entry_price,
            'exit_price': fill_event.fill_price,
            'fill_price': fill_event.fill_price,  # Alias for consistency
            'pnl': pnl,
            'fee': fill_event.fee,
            'duration': 0.0,  # Could calculate from entry time if tracked
            
            # OUTLIER DIAGNOSTIC CONTEXT
            'midprice': midprice,  # For ΔMid calculation
            'spread': spread,  # For P&L ratio calculation
            'inventory_before': self.current_position + (quantity if fill_event.side == OrderSide.ASK else -quantity),
            'inventory_after': self.current_position,
            'time_gap': time_gap,  # Δt for stale fill detection
            'notional': quantity * fill_event.fill_price,  # For blow-up detection
            'is_maker': fill_event.is_maker,
            'latency_ms': fill_event.latency_ms
        }
        
        self.trades.append(trade)
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price"""
        if self.current_position == 0 or self.avg_entry_price == 0:
            return 0.0
        
        if self.current_position > 0:
            # Long position
            return self.current_position * (current_price - self.avg_entry_price)
        else:
            # Short position
            return abs(self.current_position) * (self.avg_entry_price - current_price)
    
    def update_pnl(self, timestamp: float, current_price: float) -> None:
        """Update PnL series with current market price"""
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        total_pnl = self.realized_pnl + unrealized_pnl
        
        # Only add if time has progressed
        if not self.pnl_series or timestamp > self.pnl_series[-1][0]:
            self.pnl_series.append((timestamp, total_pnl))
            self.position_series.append((timestamp, self.current_position))
    
    def calculate_performance_metrics(self, current_price: float) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        Uses caching to avoid expensive recalculations.
        """
        current_time = time.time()
        
        # Check cache validity (refresh every 5 seconds)
        if (self._metrics_cache and 
            current_time - self._cache_timestamp < 5.0 and
            len(self.pnl_series) > 0):
            return self._metrics_cache
        
        try:
            # 🔧 CRITICAL FIX: Use per-fill trade P&Ls instead of round-trip realized_pnl
            # Problem: realized_pnl only tracks CLOSED positions (round-trip accounting)
            # Solution: Sum individual trade P&Ls (per-fill accounting)
            # This matches our per-fill trade recording logic
            
            # Calculate total P&L from individual trade records
            trade_pnls_sum = sum(trade['pnl'] for trade in self.trades) if self.trades else 0.0
            
            # Calculate unrealized P&L from open position
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            
            # Gross P&L = sum of all trade P&Ls + unrealized P&L from open position
            gross_pnl = trade_pnls_sum + unrealized_pnl
            
            # Net P&L = gross P&L minus all fees
            net_pnl = gross_pnl - self.total_fees
            total_pnl = net_pnl  # Net P&L is the final P&L
            
            # For dashboard compatibility, also update realized_pnl to match trade sum
            # (This ensures consistency across all P&L displays)
            calculated_realized_pnl = trade_pnls_sum  # Use trade-based calculation
            
            # Time metrics
            if self.pnl_series:
                start_time = self.pnl_series[0][0]
                end_time = self.pnl_series[-1][0]
                duration_hours = (end_time - start_time) / 3600.0
            else:
                start_time = end_time = current_time
                duration_hours = 0.0
            
            # Returns calculation
            if self.pnl_series and len(self.pnl_series) > 1:
                pnl_values = [pnl for _, pnl in self.pnl_series]
                returns = np.diff(pnl_values) / self.initial_capital
                
                # Risk metrics
                volatility = np.std(returns) * np.sqrt(365 * 24) if len(returns) > 1 else 0.0
                
                # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
                sharpe_ratio = (np.mean(returns) * 365 * 24) / volatility if volatility > 0 else 0.0
                
                # Sortino ratio (downside deviation)
                negative_returns = returns[returns < 0]
                downside_vol = np.std(negative_returns) * np.sqrt(365 * 24) if len(negative_returns) > 1 else volatility
                sortino_ratio = (np.mean(returns) * 365 * 24) / downside_vol if downside_vol > 0 else 0.0
                
                # Drawdown calculation
                equity_curve = np.array(pnl_values) + self.initial_capital
                running_max = np.maximum.accumulate(equity_curve)
                drawdown = (running_max - equity_curve) / running_max
                max_drawdown = np.max(drawdown)
                
                # Drawdown duration (simplified)
                max_dd_duration = 0.0
                current_dd_duration = 0.0
                for i in range(1, len(drawdown)):
                    if drawdown[i] > 0:
                        current_dd_duration = self.pnl_series[i][0] - self.pnl_series[0][0]
                        max_dd_duration = max(max_dd_duration, current_dd_duration)
                    else:
                        current_dd_duration = 0.0
                
                # Calmar ratio
                annual_return = (total_pnl / self.initial_capital) * (365 * 24 / max(duration_hours, 1))
                calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
                
            else:
                volatility = sharpe_ratio = sortino_ratio = 0.0
                max_drawdown = max_dd_duration = calmar_ratio = 0.0
            
            # Trading metrics
            total_trades = len(self.trades)
            if total_trades > 0:
                # ✅ ISSUE #10 FIX: Ensure correct P&L calculation
                # Trade P&L should be price difference only (fees tracked separately)
                trade_pnls = [trade['pnl'] for trade in self.trades]
                
                # Validate trade P&Ls are calculated correctly
                # Winning = P&L > 0 (price movement in our favor)
                # Losing = P&L < 0 (price movement against us)
                # Note: Fees are subtracted separately in gross_pnl calculation
                winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
                losing_trades = len([pnl for pnl in trade_pnls if pnl < 0])
                breakeven_trades = len([pnl for pnl in trade_pnls if pnl == 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                
                # Debug: Log trade P&Ls to verify calculation
                if total_trades > 0:
                    logger.info(f"📊 Trade Analysis: Total={total_trades}, "
                              f"Winning={winning_trades}, Losing={losing_trades}, "
                              f"Breakeven={breakeven_trades}, Win Rate={win_rate*100:.1f}%")
                    logger.info(f"   Trade P&Ls sample (first 10): {trade_pnls[:10]}")
                    logger.info(f"   Sum of trade P&Ls: {sum(trade_pnls):.2f}")
                
                winning_pnls = [pnl for pnl in trade_pnls if pnl > 0]
                losing_pnls = [pnl for pnl in trade_pnls if pnl < 0]
                
                avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
                avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
                avg_trade_pnl = np.mean(trade_pnls)  # Average PnL per trade
                
                gross_profit = sum(winning_pnls) if winning_pnls else 0.0
                gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            else:
                winning_trades = losing_trades = 0
                win_rate = avg_win = avg_loss = avg_trade_pnl = profit_factor = 0.0
            
            # Fill and execution metrics
            total_fills = len(self.fill_events)
            fill_rate = self.quote_hits / max(self.quote_updates, 1)
            quote_hit_rate = self.quote_hits / max(self.quote_updates, 1)
            
            avg_fill_latency = np.mean(self.fill_latencies) if self.fill_latencies else 0.0
            avg_spread_captured = np.mean(self.spread_captures) if self.spread_captures else 0.0
            
            # Market making specific metrics
            if duration_hours > 0 and total_fills > 0:
                inventory_turnover = sum(fill.fill_quantity for fill in self.fill_events) / duration_hours
            else:
                inventory_turnover = 0.0
            
            # Adverse selection (simplified - measure of fills against us)
            adverse_fills = 0
            for i, fill in enumerate(self.fill_events):
                if i > 0:
                    prev_fill = self.fill_events[i-1] 
                    time_diff = fill.timestamp - prev_fill.timestamp
                    if time_diff < 1.0:  # Quick succession might indicate adverse selection
                        adverse_fills += 1
            
            adverse_selection_rate = adverse_fills / max(total_fills, 1)
            
            # Volume and fee metrics
            total_volume = sum(fill.fill_quantity * fill.fill_price for fill in self.fill_events)
            fee_rate = self.total_fees / max(total_volume, 1)
            
            # Calculate total return percentage
            total_return_pct = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
            
            # Prepare chart data for dashboard
            pnl_history = [pnl for _, pnl in self.pnl_series] if self.pnl_series else []
            timestamps = [ts for ts, _ in self.pnl_series] if self.pnl_series else []
            
            # 🔧 CRITICAL FIX: Use trade-based realized_pnl for consistency
            # Create metrics object
            metrics = PerformanceMetrics(
                total_pnl=total_pnl,
                realized_pnl=calculated_realized_pnl,  # Use trade sum, not round-trip realized_pnl
                unrealized_pnl=unrealized_pnl,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                total_return_pct=total_return_pct,
                
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                volatility=volatility,
                
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                avg_trade_pnl=avg_trade_pnl,
                profit_factor=profit_factor,
                
                fill_rate=fill_rate,
                quote_hit_rate=quote_hit_rate,
                avg_spread_captured=avg_spread_captured,
                inventory_turnover=inventory_turnover,
                adverse_selection_rate=adverse_selection_rate,
                
                avg_fill_latency_ms=avg_fill_latency,
                total_fees=self.total_fees,
                fee_rate=fee_rate,
                total_volume=total_volume,
                
                start_time=start_time,
                end_time=end_time,
                duration_hours=duration_hours,
                
                # Chart data for dashboard
                pnl_history=pnl_history,
                timestamps=timestamps
            )
            
            # Cache the result
            self._metrics_cache = metrics
            self._cache_timestamp = current_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            # Return default metrics on error
            return PerformanceMetrics(
                total_pnl=0, realized_pnl=0, unrealized_pnl=0, gross_pnl=0, net_pnl=0,
                total_return_pct=0.0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
                max_drawdown_duration=0, volatility=0, total_trades=0, winning_trades=0,
                losing_trades=0, win_rate=0, avg_win=0, avg_loss=0, avg_trade_pnl=0, profit_factor=0,
                fill_rate=0, quote_hit_rate=0, avg_spread_captured=0, inventory_turnover=0,
                adverse_selection_rate=0, avg_fill_latency_ms=0, total_fees=0, fee_rate=0, total_volume=0,
                start_time=current_time, end_time=current_time, duration_hours=0,
                pnl_history=[], timestamps=[]
            )
    
    def get_pnl_series(self) -> List[Tuple[float, float]]:
        """Get PnL time series"""
        return self.pnl_series.copy()
    
    def get_position_series(self) -> List[Tuple[float, float]]:
        """Get position time series"""
        return self.position_series.copy()
    
    def get_trade_history(self) -> List[Dict]:
        """Get completed trade history"""
        return self.trades.copy()
    
    def detect_trade_outliers(self, std_threshold: float = 3.0) -> Dict:
        """
        PROFESSIONAL OUTLIER DETECTION with 7-type taxonomy and root cause diagnosis.
        
        Based on quantitative framework for HFT outlier classification:
        1. Data Spike - corrupted ticks (remove)
        2. Stale Fill - old quote fill (remove)
        3. Position Blow-Up - large inventory close (cap/remove)
        4. Scaling Error - wrong units (remove)
        5. Aggregation Error - merged fills (correct)
        6. Regime Shift - genuine event (keep & tag)
        7. Statistical Tail - legitimate tail (winsorize)
        
        Diagnosis Framework (4 checks per trade):
        - ΔMid = |fill_price - mid| / mid (threshold: 0.002 = 20 bps)
        - Δt between ticks (threshold: 5 seconds)
        - inventory_before × price consistency
        - PnL / (spread × size) ratio (threshold: 10×)
        
        Returns:
            Comprehensive outlier analysis with classification and treatment
        """
        if len(self.trades) < 3:
            return self._empty_outlier_result()
        
        # Extract trade data
        trade_pnls = np.array([trade['pnl'] for trade in self.trades])
        mean_pnl = np.mean(trade_pnls)
        std_pnl = np.std(trade_pnls)
        
        if std_pnl < 1e-9:
            return self._empty_outlier_result()
        
        # Calculate z-scores
        z_scores = np.abs((trade_pnls - mean_pnl) / std_pnl)
        
        # Detect outliers by z-score
        outlier_mask = z_scores > std_threshold
        outlier_indices = np.where(outlier_mask)[0]
        normal_indices = np.where(~outlier_mask)[0]
        
        # CLASSIFY each outlier by root cause
        classified_outliers = []
        for idx in outlier_indices:
            trade = self.trades[idx]
            classification = self._diagnose_outlier(trade, idx)
            
            classified_outliers.append({
                **trade,
                'z_score': z_scores[idx],
                'outlier_type': classification['type'],
                'root_cause': classification['cause'],
                'treatment': classification['treatment'],
                'keep': classification['keep'],
                'diagnostics': classification['diagnostics']
            })
        
        # Separate by treatment
        outliers_to_remove = [o for o in classified_outliers if not o['keep']]
        outliers_to_keep = [o for o in classified_outliers if o['keep']]
        
        # Build clean trade list (removes artefacts, keeps genuine outliers)
        clean_trades = [self.trades[i] for i in normal_indices] + [
            {k: v for k, v in o.items() if k not in ['z_score', 'outlier_type', 'root_cause', 'treatment', 'keep', 'diagnostics']}
            for o in outliers_to_keep
        ]
        
        # Calculate statistics
        clean_pnls = np.array([t['pnl'] for t in clean_trades])
        mean_clean = np.mean(clean_pnls) if len(clean_pnls) > 0 else 0.0
        
        # Winsorized metrics (cap at ±3σ for statistical tails)
        winsorized_pnls = np.clip(clean_pnls, 
                                  mean_clean - 3*np.std(clean_pnls),
                                  mean_clean + 3*np.std(clean_pnls))
        mean_winsorized = np.mean(winsorized_pnls) if len(winsorized_pnls) > 0 else 0.0
        
        total_outlier_impact = np.sum([o['pnl'] for o in classified_outliers])
        outlier_pct = (len(classified_outliers) / len(self.trades)) * 100
        
        # Log comprehensive outlier report
        if len(classified_outliers) > 0:
            logger.info(f"🔍 OUTLIER DETECTION REPORT ({len(classified_outliers)} outliers, {outlier_pct:.1f}% of trades):")
            
            # Group by type
            type_counts = {}
            for o in classified_outliers:
                otype = o['outlier_type']
                type_counts[otype] = type_counts.get(otype, 0) + 1
            
            for otype, count in sorted(type_counts.items()):
                logger.info(f"  {otype}: {count} occurrences")
            
            logger.info(f"  Removed: {len(outliers_to_remove)} artefacts")
            logger.info(f"  Kept: {len(outliers_to_keep)} genuine events")
            logger.info(f"  Total Impact: ${total_outlier_impact:.2f}")
            logger.info(f"  Avg P&L: ${mean_pnl:.2f} (raw) → ${mean_clean:.2f} (clean) → ${mean_winsorized:.2f} (winsorized)")
            
            # Log details of each outlier (first 5)
            logger.info("  Outlier Details:")
            for i, o in enumerate(classified_outliers[:5]):
                logger.info(f"    #{i+1}: P&L=${o['pnl']:.2f}, Type={o['outlier_type']}, "
                          f"z-score={o['z_score']:.2f}, Keep={o['keep']}")
                logger.info(f"        Cause: {o['root_cause']}")
                logger.info(f"        Diagnostics: ΔMid={o['diagnostics'].get('delta_mid_bps', 0):.1f} bps, "
                          f"Δt={o['diagnostics'].get('delta_t', 0):.1f}s, "
                          f"Notional=${o['diagnostics'].get('notional_change', 0):.0f}, "
                          f"P&L Ratio={o['diagnostics'].get('pnl_ratio', 0):.1f}×")
            
            if len(classified_outliers) > 5:
                logger.info(f"    ... and {len(classified_outliers) - 5} more outliers")
        
        return {
            'outlier_trades': classified_outliers,
            'normal_trades': [self.trades[i] for i in normal_indices],
            'clean_trades': clean_trades,
            'outliers_removed': outliers_to_remove,
            'outliers_kept': outliers_to_keep,
            'outlier_count': len(classified_outliers),
            'removed_count': len(outliers_to_remove),
            'kept_count': len(outliers_to_keep),
            'total_trades': len(self.trades),
            'outlier_pct': outlier_pct,
            'mean_raw': mean_pnl,
            'mean_clean': mean_clean,
            'mean_winsorized': mean_winsorized,
            'mean_normal': np.mean([self.trades[i]['pnl'] for i in normal_indices]) if len(normal_indices) > 0 else 0.0,
            'total_outlier_impact': total_outlier_impact,
            'std_pnl': std_pnl,
            'z_threshold': std_threshold,
            'type_counts': type_counts if classified_outliers else {}
        }
    
    def _empty_outlier_result(self) -> Dict:
        """Return empty outlier result for edge cases"""
        mean_pnl = np.mean([t['pnl'] for t in self.trades]) if self.trades else 0.0
        return {
            'outlier_trades': [],
            'normal_trades': self.trades.copy(),
            'clean_trades': self.trades.copy(),
            'outliers_removed': [],
            'outliers_kept': [],
            'outlier_count': 0,
            'removed_count': 0,
            'kept_count': 0,
            'total_trades': len(self.trades),
            'outlier_pct': 0.0,
            'mean_raw': mean_pnl,
            'mean_clean': mean_pnl,
            'mean_winsorized': mean_pnl,
            'mean_normal': mean_pnl,
            'total_outlier_impact': 0.0,
            'type_counts': {}
        }
    
    def _diagnose_outlier(self, trade: Dict, trade_idx: int) -> Dict:
        """
        Diagnose outlier type using 4-variable framework.
        Returns classification with treatment protocol.
        """
        diagnostics = {}
        
        # Extract trade details
        fill_price = trade.get('fill_price', 0)
        pnl = trade['pnl']
        size = trade.get('quantity', 0)
        timestamp = trade.get('timestamp', 0)
        side = trade.get('side', '')
        
        # Get market context (if available)
        midprice = trade.get('midprice', fill_price)  # Fallback to fill price
        spread = trade.get('spread', 0.001 * midprice)  # Default 10 bps
        
        # CHECK 1: ΔMid = |fill_price - mid| / mid
        delta_mid = abs(fill_price - midprice) / midprice if midprice > 0 else 0
        diagnostics['delta_mid'] = delta_mid
        diagnostics['delta_mid_bps'] = delta_mid * 10000
        
        # CHECK 2: Δt between ticks (check against previous trade)
        delta_t = 0
        if trade_idx > 0:
            prev_timestamp = self.trades[trade_idx - 1].get('timestamp', timestamp)
            delta_t = timestamp - prev_timestamp
        diagnostics['delta_t'] = delta_t
        
        # CHECK 3: Inventory × price consistency (detect position blow-ups)
        inventory_before = trade.get('inventory_before', 0)
        inventory_after = trade.get('inventory_after', 0)
        notional_change = abs(inventory_after - inventory_before) * fill_price
        diagnostics['notional_change'] = notional_change
        
        # CHECK 4: PnL / (spread × size) - should be < 5× for normal trades
        expected_pnl = spread * size
        pnl_ratio = abs(pnl) / expected_pnl if expected_pnl > 1e-6 else 0
        diagnostics['pnl_ratio'] = pnl_ratio
        
        # CLASSIFICATION LOGIC (priority order)
        
        # Type 1: Data Spike (extreme slippage)
        if delta_mid > 0.10:  # 10% slippage = data error
            return {
                'type': 'DATA_SPIKE',
                'cause': f'Extreme slippage {delta_mid*100:.1f}% suggests corrupted tick',
                'treatment': 'REMOVE',
                'keep': False,
                'diagnostics': diagnostics
            }
        
        # Type 2: Stale Fill (large time gap + high slippage)
        if delta_t > 5.0 and delta_mid > 0.002:  # 5+ sec gap + 20 bps slip
            return {
                'type': 'STALE_FILL',
                'cause': f'Fill delayed {delta_t:.1f}s with {diagnostics["delta_mid_bps"]:.0f} bps slippage',
                'treatment': 'REMOVE',
                'keep': False,
                'diagnostics': diagnostics
            }
        
        # Type 4: Scaling Error (P&L ratio way off)
        if pnl_ratio > 50:  # P&L > 50× expected spread capture
            return {
                'type': 'SCALING_ERROR',
                'cause': f'P&L {pnl_ratio:.0f}× expected (likely unit error)',
                'treatment': 'REMOVE',
                'keep': False,
                'diagnostics': diagnostics
            }
        
        # Type 3: Position Blow-Up (large notional change)
        if notional_change > 10000:  # $10k+ notional swing (adjust for asset)
            return {
                'type': 'POSITION_BLOWUP',
                'cause': f'Large position close ${notional_change:.0f} (inventory drift)',
                'treatment': 'CAP',
                'keep': False,  # Remove by default, can be kept if validated
                'diagnostics': diagnostics
            }
        
        # Type 6: Regime Shift (moderate slippage, normal timing, high P&L)
        if 0.002 < delta_mid < 0.05 and delta_t < 5.0 and pnl_ratio > 10:
            return {
                'type': 'REGIME_SHIFT',
                'cause': f'Large legitimate move: {diagnostics["delta_mid_bps"]:.0f} bps slippage, {pnl_ratio:.1f}× expected',
                'treatment': 'KEEP_TAG',
                'keep': True,  # Keep but mark as special event
                'diagnostics': diagnostics
            }
        
        # Type 7: Statistical Tail (high z-score, but all checks normal)
        return {
            'type': 'STATISTICAL_TAIL',
            'cause': f'Legitimate tail event (P&L {pnl_ratio:.1f}× expected)',
            'treatment': 'WINSORIZE',
            'keep': True,  # Keep but winsorize in metrics
            'diagnostics': diagnostics
        }
    
    def get_daily_pnl(self) -> Dict[str, float]:
        """Calculate daily PnL breakdown"""
        if not self.pnl_series:
            return {}
        
        daily_pnl = {}
        
        for timestamp, pnl in self.pnl_series:
            date_str = pd.Timestamp(timestamp, unit='s').strftime('%Y-%m-%d')
            daily_pnl[date_str] = pnl
        
        return daily_pnl
    
    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Export metrics data to pandas DataFrames"""
        try:
            # PnL series
            pnl_df = pd.DataFrame(self.pnl_series, columns=['timestamp', 'pnl'])
            pnl_df['datetime'] = pd.to_datetime(pnl_df['timestamp'], unit='s')
            
            # Position series
            position_df = pd.DataFrame(self.position_series, columns=['timestamp', 'position'])
            position_df['datetime'] = pd.to_datetime(position_df['timestamp'], unit='s')
            
            # Trades
            trades_df = pd.DataFrame(self.trades)
            if not trades_df.empty:
                trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='s')
            
            # Fill events
            fills_data = []
            for fill in self.fill_events:
                fills_data.append({
                    'timestamp': fill.timestamp,
                    'order_id': fill.order_id,
                    'side': fill.side.value,
                    'price': fill.fill_price,
                    'quantity': fill.fill_quantity,
                    'fee': fill.fee,
                    'latency_ms': fill.latency_ms,
                    'is_maker': fill.is_maker,
                    'reason': fill.fill_reason.value
                })
            
            fills_df = pd.DataFrame(fills_data)
            if not fills_df.empty:
                fills_df['datetime'] = pd.to_datetime(fills_df['timestamp'], unit='s')
            
            return {
                'pnl': pnl_df,
                'positions': position_df,
                'trades': trades_df,
                'fills': fills_df
            }
            
        except Exception as e:
            logger.error(f"Error exporting to DataFrame: {e}")
            return {}
    
    def reset(self) -> None:
        """Reset all metrics and tracking"""
        self.fill_events.clear()
        self.pnl_series.clear()
        self.position_series.clear()
        self.quote_events.clear()
        self.trades.clear()
        self.open_positions.clear()
        self.fill_latencies.clear()
        self.spread_captures.clear()
        
        self.current_position = 0.0
        self.current_cash = self.initial_capital
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.quote_updates = 0
        self.quote_hits = 0
        
        self._metrics_cache = None
        self._cache_timestamp = 0.0
        
        logger.info("BacktestMetrics reset")


# Example usage for testing
if __name__ == "__main__":
    # Initialize metrics
    metrics = BacktestMetrics(initial_capital=10000.0)
    
    # Simulate some trading activity
    from .fill_simulator_fifo import FillEvent, FillReason
    from src.strategy import OrderSide
    
    current_price = 50000.0
    
    # Record some fills
    fill1 = FillEvent(
        timestamp=time.time(),
        order_id="test_1",
        side=OrderSide.BID,
        fill_price=49995.0,
        fill_quantity=0.1,
        remaining_quantity=0.0,
        fee=5.0,
        fill_reason=FillReason.LIQUIDITY_TAKEN,
        latency_ms=25.0,
        is_maker=True
    )
    
    fill2 = FillEvent(
        timestamp=time.time() + 60,
        order_id="test_2",
        side=OrderSide.ASK,
        fill_price=50005.0,
        fill_quantity=0.1,
        remaining_quantity=0.0,
        fee=5.0,
        fill_reason=FillReason.LIQUIDITY_TAKEN,
        latency_ms=30.0,
        is_maker=True
    )
    
    metrics.record_fill(fill1, current_price)
    metrics.record_fill(fill2, current_price + 10)
    
    # Record some quote updates
    metrics.record_quote_update(time.time(), was_hit=True)
    metrics.record_quote_update(time.time() + 30, was_hit=False)
    
    # Calculate performance
    performance = metrics.calculate_performance_metrics(current_price)
    
    print(f"Performance Metrics:")
    print(f"Total PnL: {performance.total_pnl:.2f}")
    print(f"Realized PnL: {performance.realized_pnl:.2f}")
    print(f"Total Trades: {performance.total_trades}")
    print(f"Win Rate: {performance.win_rate:.2%}")
    print(f"Fill Rate: {performance.fill_rate:.2%}")
    print(f"Avg Latency: {performance.avg_fill_latency_ms:.1f}ms")
    print(f"Total Fees: {performance.total_fees:.2f}")
    
    # Export to DataFrame
    dataframes = metrics.export_to_dataframe()
    if 'fills' in dataframes and not dataframes['fills'].empty:
        print(f"\nFills DataFrame shape: {dataframes['fills'].shape}")
        print(dataframes['fills'].head())