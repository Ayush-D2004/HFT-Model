"""
Live Avellaneda-Stoikov Strategy
===============================

Real-time implementation of the Avellaneda-Stoikov market making strategy
for live trading with continuous market data processing.
"""

import time
import math
from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np

from ..strategy.avellaneda_stoikov import AvellanedaStoikovPricer
from ..utils.logger import get_logger


class LiveAvellanedaStoikov:
    """
    Live implementation of Avellaneda-Stoikov strategy with:
    - Real-time market data processing
    - Continuous quote generation
    - Dynamic risk management
    - Performance optimization for live trading
    """
    
    def __init__(self, 
                 symbol: str,
                 tick_size: float,
                 lot_size: float,
                 gamma: float = 0.1,
                 time_horizon: float = 30.0,
                 max_position: float = 5.0,
                 min_spread: float = 0.01):
        
        self.symbol = symbol
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.gamma = gamma
        self.time_horizon = time_horizon
        self.max_position = max_position
        self.min_spread = min_spread
        
        self.logger = get_logger('live_strategy')
        
        # Core strategy engine
        self.pricer = AvellanedaStoikovPricer(
            tick_size=tick_size,
            ewma_alpha=0.2,
            vol_lookback_sec=60,
            k_lookback_sec=60
        )
        
        # Live trading state
        self.current_quotes = None
        self.last_quote_time = None
        self.quote_count = 0
        
        # Market state tracking
        self.current_midprice = None
        self.last_spread = None
        self.current_volatility = None
        
        # Performance tracking
        self.total_quoted_volume = 0.0
        self.quote_update_frequency = deque(maxlen=100)  # Last 100 quote times
        
        # Risk management
        self.position_limit_breached = False
        self.emergency_stop = False
        
        self.logger.info(f"Live strategy initialized for {symbol} with gamma={gamma}")
    
    def process_market_data(self, market_data: Dict) -> Optional[Dict]:
        """
        Process incoming market data and generate quotes if needed.
        
        Args:
            market_data: Real-time market data from WebSocket
            
        Returns:
            Quote dictionary if new quotes generated, None otherwise
        """
        try:
            # Extract market information
            timestamp = market_data.get('timestamp', time.time())
            midprice = market_data.get('midprice')
            spread = market_data.get('spread')
            
            if not midprice:
                return None
            
            # Update strategy with market data
            self.pricer.update_market(midprice, timestamp)
            self.current_midprice = midprice
            self.last_spread = spread
            
            # Update volatility estimate
            self.current_volatility = self.pricer.instantaneous_sigma()
            
            # Check if we should generate new quotes
            if self._should_update_quotes():
                quotes = self._generate_quotes()
                
                if quotes:
                    self.current_quotes = quotes
                    self.last_quote_time = timestamp
                    self.quote_count += 1
                    
                    # Track quote frequency
                    self.quote_update_frequency.append(timestamp)
                    
                    # self.logger.debug(f"Generated quotes: BID={quotes['bid']:.4f}, ASK={quotes['ask']:.4f}")
                    return quotes
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return None
    
    def register_trade_event(self, trade_data: Dict):
        """Register trade event for arrival rate estimation"""
        try:
            timestamp = trade_data.get('timestamp', time.time())
            self.pricer.register_trade_event(timestamp)
            
        except Exception as e:
            self.logger.error(f"Error registering trade event: {e}")
    
    def update_position(self, position: float):
        """Update current inventory position"""
        try:
            self.pricer.update_inventory(position)
            
            # Check position limits
            if abs(position) > self.max_position:
                if not self.position_limit_breached:
                    self.logger.warning(f"Position limit breached: {position:.4f} > {self.max_position}")
                    self.position_limit_breached = True
            else:
                if self.position_limit_breached:
                    self.logger.info("Position back within limits")
                    self.position_limit_breached = False
                    
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
    
    def _should_update_quotes(self) -> bool:
        """Determine if quotes should be updated based on market conditions"""
        try:
            # For HFT - generate quotes continuously on every market update
            return True
            
            # Original logic (commented out for testing)
            # Always update if no quotes exist
            if self.current_quotes is None:
                return True
            
            # Update if significant time has passed (max 5 seconds)
            if self.last_quote_time and time.time() - self.last_quote_time > 5.0:
                return True
            
            # Update if volatility has changed significantly
            if self._volatility_change_significant():
                return True
            
            # Update if position limit is breached (to adjust quotes)
            if self.position_limit_breached:
                return True
            
            # Update if market has moved significantly
            if self._market_movement_significant():
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking quote update condition: {e}")
            return False
    
    def _volatility_change_significant(self) -> bool:
        """Check if volatility has changed significantly"""
        try:
            if not hasattr(self, '_last_volatility'):
                self._last_volatility = self.current_volatility
                return False
            
            if self.current_volatility and self._last_volatility:
                vol_change = abs(self.current_volatility - self._last_volatility) / self._last_volatility
                self._last_volatility = self.current_volatility
                return vol_change > 0.1  # 10% volatility change threshold
            
            return False
            
        except Exception:
            return False
    
    def _market_movement_significant(self) -> bool:
        """Check if market has moved significantly since last quote"""
        try:
            if not hasattr(self, '_last_midprice') or not self.current_midprice:
                self._last_midprice = self.current_midprice
                return False
            
            if self._last_midprice:
                price_change = abs(self.current_midprice - self._last_midprice) / self._last_midprice
                return price_change > 0.001  # 0.1% price movement threshold
            
            return False
            
        except Exception:
            return False
    
    def _generate_quotes(self) -> Optional[Dict]:
        """Generate bid/ask quotes using Avellaneda-Stoikov framework"""
        try:
            if self.emergency_stop:
                self.logger.warning("Emergency stop active, not generating quotes")
                return None
            
            if not self.current_midprice:
                self.logger.warning("No current midprice available")
                return None
            
            # self.logger.debug(f"Generating quotes with midprice={self.current_midprice}")
            
            # Get optimal quotes from strategy
            bid_price, ask_price = self.pricer.compute_quotes(
                gamma=self.gamma,
                T=self.time_horizon,
                midprice=self.current_midprice,
                min_spread=self.min_spread
            )
            
            # Use fixed lot size for simplicity in testing
            quote_size = self.lot_size
            
            quotes = {
                'symbol': self.symbol,
                'timestamp': time.time(),
                'bid': bid_price,
                'ask': ask_price,
                'bid_size': quote_size,
                'ask_size': quote_size,
                'midprice': self.current_midprice,
                'spread': ask_price - bid_price,
                'volatility': self.current_volatility or 0.0
            }
            
            # self.logger.info(f"Generated quotes: BID={bid_price:.4f}({quote_size}) ASK={ask_price:.4f}({quote_size})")
            return quotes
            
        except Exception as e:
            self.logger.error(f"Error generating quotes: {e}")
            return None
    
    def emergency_stop_trading(self):
        """Activate emergency stop"""
        self.emergency_stop = True
        self.logger.warning("EMERGENCY STOP ACTIVATED")
    
    def resume_trading(self):
        """Resume trading after emergency stop"""
        self.emergency_stop = False
        self.logger.info("Trading resumed")
    
    def get_strategy_state(self) -> Dict:
        """Get current strategy state for monitoring"""
        return {
            'symbol': self.symbol,
            'current_midprice': self.current_midprice,
            'current_volatility': self.current_volatility,
            'inventory': self.pricer.inventory if self.pricer else 0.0,
            'quote_count': self.quote_count,
            'total_quoted_volume': self.total_quoted_volume,
            'last_quote_time': self.last_quote_time,
            'position_limit_breached': self.position_limit_breached,
            'emergency_stop': self.emergency_stop,
            'current_quotes': self.current_quotes,
            'parameters': {
                'gamma': self.gamma,
                'time_horizon': self.time_horizon,
                'max_position': self.max_position,
                'min_spread': self.min_spread
            }
        }
    
    def update_parameters(self, **kwargs):
        """Update strategy parameters dynamically"""
        try:
            if 'gamma' in kwargs:
                self.gamma = float(kwargs['gamma'])
                self.logger.info(f"Updated gamma to {self.gamma}")
            
            if 'time_horizon' in kwargs:
                self.time_horizon = float(kwargs['time_horizon'])
                self.logger.info(f"Updated time_horizon to {self.time_horizon}")
            
            if 'max_position' in kwargs:
                self.max_position = float(kwargs['max_position'])
                self.logger.info(f"Updated max_position to {self.max_position}")
            
            if 'min_spread' in kwargs:
                self.min_spread = float(kwargs['min_spread'])
                self.logger.info(f"Updated min_spread to {self.min_spread}")
                
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")
    
    def get_quote_frequency(self) -> float:
        """Calculate current quote update frequency (quotes per second)"""
        try:
            if len(self.quote_update_frequency) < 2:
                return 0.0
            
            time_span = self.quote_update_frequency[-1] - self.quote_update_frequency[0]
            if time_span > 0:
                return (len(self.quote_update_frequency) - 1) / time_span
            
            return 0.0
            
        except Exception:
            return 0.0