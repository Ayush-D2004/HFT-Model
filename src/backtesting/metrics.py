"""
Backtesting Metrics and Performance Analysis
==========================================

Comprehensive metrics calculation for strategy evaluation including:
- PnL analysis (realized/unrealized, Sharpe, Sortino, drawdown)
- Trading metrics (fill rate, quote lifetime, adverse selection)
- Risk metrics and performance attribution
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from .fill_simulator import FillEvent
from ..strategy import Order, OrderSide


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # PnL Metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    gross_pnl: float
    net_pnl: float
    
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
    
    # Time-based Metrics
    start_time: float
    end_time: float
    duration_hours: float
    
    
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
        Record a fill event and update all relevant metrics.
        
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
            
            # Calculate new average entry price
            if self.current_position + position_change != 0:
                if self.current_position == 0:
                    # Opening new position
                    new_avg_price = fill_event.fill_price
                elif (self.current_position > 0 and position_change > 0) or \
                     (self.current_position < 0 and position_change < 0):
                    # Adding to existing position
                    total_cost = (self.current_position * self.avg_entry_price + 
                                position_change * fill_event.fill_price)
                    new_avg_price = total_cost / (self.current_position + position_change)
                else:
                    # Reducing position - realize some PnL
                    closed_quantity = min(abs(position_change), abs(self.current_position))
                    
                    if self.current_position > 0:
                        # Closing long position
                        trade_pnl = closed_quantity * (fill_event.fill_price - self.avg_entry_price)
                    else:
                        # Closing short position  
                        trade_pnl = closed_quantity * (self.avg_entry_price - fill_event.fill_price)
                    
                    self.realized_pnl += trade_pnl
                    
                    # Record completed trade
                    self._record_trade(fill_event, closed_quantity, trade_pnl)
                    
                    # Update average price (unchanged if reducing position)
                    new_avg_price = self.avg_entry_price
            else:
                # Position going to zero
                if self.current_position != 0:
                    if self.current_position > 0:
                        trade_pnl = self.current_position * (fill_event.fill_price - self.avg_entry_price)
                    else:
                        trade_pnl = abs(self.current_position) * (self.avg_entry_price - fill_event.fill_price)
                    
                    self.realized_pnl += trade_pnl
                    self._record_trade(fill_event, abs(self.current_position), trade_pnl)
                
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
        """Record a completed trade"""
        trade = {
            'timestamp': fill_event.timestamp,
            'side': 'close_long' if fill_event.side == OrderSide.ASK else 'close_short',
            'quantity': quantity,
            'entry_price': self.avg_entry_price,
            'exit_price': fill_event.fill_price,
            'pnl': pnl,
            'fee': fill_event.fee,
            'duration': 0.0  # Could calculate from entry time if tracked
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
            # Calculate current PnL
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            total_pnl = self.realized_pnl + unrealized_pnl
            gross_pnl = total_pnl + self.total_fees  # Before fees
            net_pnl = total_pnl  # After fees
            
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
                trade_pnls = [trade['pnl'] for trade in self.trades]
                winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
                losing_trades = len([pnl for pnl in trade_pnls if pnl < 0])
                win_rate = winning_trades / total_trades
                
                winning_pnls = [pnl for pnl in trade_pnls if pnl > 0]
                losing_pnls = [pnl for pnl in trade_pnls if pnl < 0]
                
                avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
                avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
                
                gross_profit = sum(winning_pnls) if winning_pnls else 0.0
                gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            else:
                winning_trades = losing_trades = 0
                win_rate = avg_win = avg_loss = profit_factor = 0.0
            
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
            
            # Fee metrics
            fee_rate = self.total_fees / max(sum(fill.fill_quantity * fill.fill_price for fill in self.fill_events), 1)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                total_pnl=total_pnl,
                realized_pnl=self.realized_pnl,
                unrealized_pnl=unrealized_pnl,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                
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
                profit_factor=profit_factor,
                
                fill_rate=fill_rate,
                quote_hit_rate=quote_hit_rate,
                avg_spread_captured=avg_spread_captured,
                inventory_turnover=inventory_turnover,
                adverse_selection_rate=adverse_selection_rate,
                
                avg_fill_latency_ms=avg_fill_latency,
                total_fees=self.total_fees,
                fee_rate=fee_rate,
                
                start_time=start_time,
                end_time=end_time,
                duration_hours=duration_hours
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
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
                max_drawdown_duration=0, volatility=0, total_trades=0, winning_trades=0,
                losing_trades=0, win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                fill_rate=0, quote_hit_rate=0, avg_spread_captured=0, inventory_turnover=0,
                adverse_selection_rate=0, avg_fill_latency_ms=0, total_fees=0, fee_rate=0,
                start_time=current_time, end_time=current_time, duration_hours=0
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
    from .fill_simulator import FillEvent, FillReason
    from ..strategy import OrderSide
    
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