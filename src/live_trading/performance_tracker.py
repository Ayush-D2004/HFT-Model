"""
Live Performance Tracker
========================

Real-time performance tracking for live trading with continuous
metrics calculation and monitoring.
"""

import time
import math
from typing import Dict, List, Optional
from collections import deque
import numpy as np
from datetime import datetime, timedelta

from ..utils.logger import get_logger


class LivePerformanceTracker:
    """
    Tracks live trading performance with real-time metrics:
    - P&L tracking
    - Sharpe ratio calculation
    - Drawdown monitoring
    - Trade analytics
    - Risk metrics
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.logger = get_logger('performance_tracker')
        
        # Trade history
        self.trades = deque(maxlen=10000)  # Keep last 10k trades
        self.pnl_history = deque(maxlen=1440)  # Keep last 1440 minutes (24h)
        
        # Performance metrics
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_volume = 0.0
        self.total_fees = 0.0
        
        # Time-based tracking
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Real-time metrics (updated every minute)
        self.minute_pnl = deque(maxlen=1440)  # 24h of minute data
        self.hourly_pnl = deque(maxlen=168)   # 7 days of hourly data
        self.daily_pnl = deque(maxlen=30)     # 30 days of daily data
        
        self.logger.info(f"Performance tracker initialized with balance: ${initial_balance}")
    
    def add_trade(self, trade_data: Dict):
        """Add a new trade and update performance metrics"""
        try:
            timestamp = trade_data.get('timestamp', time.time())
            side = trade_data.get('side')  # 'buy' or 'sell'
            price = float(trade_data.get('price', 0))
            quantity = float(trade_data.get('quantity', 0))
            value = price * quantity
            
            # Calculate trade P&L (simplified for market making)
            # In real implementation, this would track position and calculate actual P&L
            trade_pnl = self._calculate_trade_pnl(trade_data)
            
            # Create trade record
            trade_record = {
                'timestamp': timestamp,
                'side': side,
                'price': price,
                'quantity': quantity,
                'value': value,
                'pnl': trade_pnl,
                'balance_after': self.current_balance + trade_pnl,
                'trade_id': len(self.trades) + 1
            }
            
            # Update metrics
            self.trades.append(trade_record)
            self.total_trades += 1
            self.total_volume += value
            self.total_pnl += trade_pnl
            self.current_balance += trade_pnl
            
            # Update win/loss statistics
            if trade_pnl > 0:
                self.winning_trades += 1
            elif trade_pnl < 0:
                self.losing_trades += 1
            
            # Update peak and drawdown
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            self.logger.debug(f"Trade added: {side} {quantity:.4f} @ {price:.4f}, P&L: {trade_pnl:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error adding trade: {e}")
    
    def _calculate_trade_pnl(self, trade_data: Dict) -> float:
        """Calculate P&L for a trade (simplified market making model)"""
        try:
            # For market making, assume we capture a portion of the spread
            # This is a simplified model - real implementation would track positions
            
            side = trade_data.get('side')
            price = float(trade_data.get('price', 0))
            quantity = float(trade_data.get('quantity', 0))
            
            # Simulate spread capture for market making
            if trade_data.get('type') == 'market_making':
                # Assume we capture 50% of a typical spread (0.05% for crypto)
                spread_capture = price * quantity * 0.0005 * 0.5
                
                # Subtract fees (assume 0.01% maker fee)
                fees = price * quantity * 0.0001
                self.total_fees += fees
                
                return spread_capture - fees
            
            # For other trade types, return 0 (would implement proper P&L calculation)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating trade P&L: {e}")
            return 0.0
    
    def update_periodic_metrics(self):
        """Update time-based performance metrics (call every minute)"""
        try:
            current_time = time.time()
            
            # Add current P&L to minute history
            self.minute_pnl.append({
                'timestamp': current_time,
                'pnl': self.total_pnl,
                'balance': self.current_balance,
                'drawdown': self.max_drawdown
            })
            
            # Update hourly and daily metrics if needed
            self._update_hourly_metrics()
            self._update_daily_metrics()
            
            self.last_update_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating periodic metrics: {e}")
    
    def _update_hourly_metrics(self):
        """Update hourly performance metrics"""
        try:
            if len(self.minute_pnl) < 60:
                return
            
            # Check if we need to add a new hourly data point
            last_hour_time = self.hourly_pnl[-1]['timestamp'] if self.hourly_pnl else 0
            current_time = time.time()
            
            if current_time - last_hour_time >= 3600:  # 1 hour
                hour_start_idx = max(0, len(self.minute_pnl) - 60)
                hour_data = list(self.minute_pnl)[hour_start_idx:]
                
                hourly_pnl = hour_data[-1]['pnl'] - hour_data[0]['pnl']
                
                self.hourly_pnl.append({
                    'timestamp': current_time,
                    'pnl': hourly_pnl,
                    'total_pnl': self.total_pnl,
                    'balance': self.current_balance
                })
                
        except Exception as e:
            self.logger.error(f"Error updating hourly metrics: {e}")
    
    def _update_daily_metrics(self):
        """Update daily performance metrics"""
        try:
            if len(self.minute_pnl) < 1440:  # Need at least 24h of data
                return
            
            # Check if we need to add a new daily data point
            last_day_time = self.daily_pnl[-1]['timestamp'] if self.daily_pnl else 0
            current_time = time.time()
            
            if current_time - last_day_time >= 86400:  # 1 day
                day_start_idx = max(0, len(self.minute_pnl) - 1440)
                day_data = list(self.minute_pnl)[day_start_idx:]
                
                daily_pnl = day_data[-1]['pnl'] - day_data[0]['pnl']
                
                self.daily_pnl.append({
                    'timestamp': current_time,
                    'pnl': daily_pnl,
                    'total_pnl': self.total_pnl,
                    'balance': self.current_balance
                })
                
        except Exception as e:
            self.logger.error(f"Error updating daily metrics: {e}")
    
    def calculate_sharpe_ratio(self, period_minutes: int = 60) -> float:
        """Calculate Sharpe ratio over specified period"""
        try:
            if len(self.minute_pnl) < period_minutes:
                return 0.0
            
            # Get returns for the period
            recent_data = list(self.minute_pnl)[-period_minutes:]
            returns = []
            
            for i in range(1, len(recent_data)):
                ret = (recent_data[i]['pnl'] - recent_data[i-1]['pnl']) / self.initial_balance
                returns.append(ret)
            
            if len(returns) < 2:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualize (assuming 525,600 minutes per year)
            sharpe = (mean_return / std_return) * math.sqrt(525600)
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def get_current_performance(self) -> Dict:
        """Get comprehensive current performance metrics"""
        try:
            current_time = time.time()
            runtime_hours = (current_time - self.start_time) / 3600
            
            # Calculate win rate
            win_rate = self.winning_trades / max(self.total_trades, 1)
            
            # Calculate average trade P&L
            avg_trade_pnl = self.total_pnl / max(self.total_trades, 1)
            
            # Calculate return percentage
            total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
            
            # Calculate Sharpe ratios for different periods
            sharpe_1h = self.calculate_sharpe_ratio(60)
            sharpe_24h = self.calculate_sharpe_ratio(1440)
            
            # Calculate recent performance
            recent_pnl_1h = self._get_recent_pnl(60)
            recent_pnl_24h = self._get_recent_pnl(1440)
            
            # Calculate trade frequency
            trades_per_hour = self.total_trades / max(runtime_hours, 1/60)
            
            return {
                'timestamp': current_time,
                'runtime_hours': runtime_hours,
                
                # Balance and P&L
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'total_pnl': self.total_pnl,
                'total_return_pct': total_return,
                
                # Recent performance
                'pnl_1h': recent_pnl_1h,
                'pnl_24h': recent_pnl_24h,
                
                # Risk metrics
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown * 100,
                'peak_balance': self.peak_balance,
                
                # Sharpe ratios
                'sharpe_1h': sharpe_1h,
                'sharpe_24h': sharpe_24h,
                
                # Trade statistics
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'avg_trade_pnl': avg_trade_pnl,
                'total_volume': self.total_volume,
                'total_fees': self.total_fees,
                'trades_per_hour': trades_per_hour,
                
                # Time series data for charts
                'pnl_history': list(self.minute_pnl)[-60:],  # Last hour
                'balance_history': [{'timestamp': p['timestamp'], 'balance': p['balance']} 
                                  for p in list(self.minute_pnl)[-60:]]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current performance: {e}")
            return {}
    
    def _get_recent_pnl(self, minutes: int) -> float:
        """Get P&L for recent period"""
        try:
            if len(self.minute_pnl) < minutes:
                return 0.0
            
            recent_data = list(self.minute_pnl)[-minutes:]
            if len(recent_data) < 2:
                return 0.0
            
            return recent_data[-1]['pnl'] - recent_data[0]['pnl']
            
        except Exception:
            return 0.0
    
    def reset_performance(self):
        """Reset all performance metrics"""
        self.logger.info("Resetting performance metrics")
        
        self.current_balance = self.initial_balance
        self.trades.clear()
        self.pnl_history.clear()
        self.minute_pnl.clear()
        self.hourly_pnl.clear()
        self.daily_pnl.clear()
        
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_volume = 0.0
        self.total_fees = 0.0
        
        self.start_time = time.time()