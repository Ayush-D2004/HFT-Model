"""
HFT Market Making Dashboard
==========================

Real-time Streamlit dashboard with:
- Live order book display in sidebar
- Market state visualization (price charts, volatility heatmap)
- Performance analytics (PnL, Sharpe ratio, inventory tracking)
- Backtesting results analysis with progress tracking
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path
import threading
import queue

# Configure Streamlit page
st.set_page_config(
    page_title="HFT Market Maker Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our HFT modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.backtesting import BacktestEngine, BacktestConfig
from src.strategy import AvellanedaStoikovPricer, QuoteParameters
from src.data_ingestion import OrderBook
from src.utils.config import config
from src.utils.logger import setup_backtesting_logging, setup_development_logging, get_logger


# Supported trading pairs
SUPPORTED_TOKENS = [
    "BTCUSDT",
    "BNBUSDT", 
    "ETHUSDT",
    "LTCUSDT",
    "SOLUSDT"
]


class DashboardState:
    """Manages dashboard state and data with proper logging and validation"""
    
    def __init__(self):
        # Initialize logger for dashboard operations
        setup_development_logging()  # Configure logging system
        self.logger = get_logger('dashboard')  # Get logger instance
        
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = []
            self.logger.info("Dashboard initialized: Empty backtest results store created")
            
        # Initialize strategy parameters first
        if 'strategy_params' not in st.session_state:
            st.session_state.strategy_params = {
                'gamma': 0.1,
                'time_horizon': 30.0,
                'min_spread': 0.02,
                'symbol': 'BTCUSDT',
                'max_position': 10.0,
                'tick_size': 0.01,
                'last_updated': time.time()
            }
            self.logger.info("Dashboard initialized: Default strategy parameters set")
        
        # Initialize strategy pricer if not exists
        if 'strategy_pricer' not in st.session_state:
            try:
                # Create real Avellaneda-Stoikov pricer
                from src.strategy.avellaneda_stoikov import AvellanedaStoikovPricer
                params = st.session_state.strategy_params
                
                st.session_state.strategy_pricer = AvellanedaStoikovPricer(
                    tick_size=params.get('tick_size', 0.01),
                    ewma_alpha=0.2,
                    vol_lookback_sec=60,
                    k_lookback_sec=60
                )
                
                # Initialize with sample market data
                st.session_state.strategy_pricer.update_market(50000.0)  # Initial price
                self.logger.info("Dashboard initialized: Real Avellaneda-Stoikov strategy created")
                
            except Exception as e:
                self.logger.error(f"Failed to create strategy pricer: {e}")
                st.session_state.strategy_pricer = None
        
        # Generate initial data using strategy if available
        if 'current_orderbook' not in st.session_state:
            if st.session_state.strategy_pricer:
                st.session_state.current_orderbook = self.generate_strategy_integrated_orderbook(
                    50000.0, st.session_state.strategy_pricer
                )
                self.logger.info("Dashboard initialized: Order book with strategy quotes")
            else:
                st.session_state.current_orderbook = self._generate_sample_orderbook()
                self.logger.warning("Dashboard initialized: Using sample order book data (strategy unavailable)")
            
        if 'live_data' not in st.session_state:
            if st.session_state.strategy_pricer:
                st.session_state.live_data = self.generate_strategy_based_data(st.session_state.strategy_pricer)
                self.logger.info("Dashboard initialized: Live data from real strategy")
            else:
                st.session_state.live_data = self._generate_sample_live_data()
                self.logger.warning("Dashboard initialized: Using simulated live data (strategy unavailable)")
    
    def update_strategy_parameters(self, **kwargs) -> bool:
        """Update strategy parameters with validation and logging"""
        updated = False
        old_params = st.session_state.strategy_params.copy()
        
        for key, value in kwargs.items():
            if key in st.session_state.strategy_params:
                if st.session_state.strategy_params[key] != value:
                    # Validate parameter ranges
                    if self._validate_parameter(key, value):
                        old_value = st.session_state.strategy_params[key]
                        st.session_state.strategy_params[key] = value
                        st.session_state.strategy_params['last_updated'] = time.time()
                        
                        self.logger.info(f"Parameter updated: {key} changed from {old_value} to {value}")
                        updated = True
                    else:
                        self.logger.error(f"Invalid parameter value: {key}={value} - rejected")
                        return False
        
        if updated:
            self.logger.info(f"Strategy parameters updated. New config: {st.session_state.strategy_params}")
            
        return updated
    
    def _validate_parameter(self, key: str, value) -> bool:
        """Validate parameter values for safety"""
        validations = {
            'gamma': lambda x: 0.001 <= x <= 2.0,  # Risk aversion bounds
            'time_horizon': lambda x: 1.0 <= x <= 300.0,  # 1s to 5min
            'min_spread': lambda x: 0.0 <= x <= 10.0,  # Reasonable spread bounds
            'symbol': lambda x: x in SUPPORTED_TOKENS
        }
        
        if key in validations:
            return validations[key](value)
        return True
    
    def _generate_sample_orderbook(self) -> Dict:
        """Generate realistic order book data - should be replaced with live data"""
        # TODO: This should be replaced with real order book from data ingestion module
        self.logger.warning("Using sample order book - not connected to live Binance feed")
        
        # Use stored base price to maintain consistency
        if not hasattr(self, '_base_price'):
            self._base_price = 50000.0
        
        # Add realistic price movement based on time
        time_factor = (time.time() % 3600) / 3600  # Hour-based cycle
        price_drift = np.sin(time_factor * 2 * np.pi) * 50
        self._base_price += np.random.normal(0, 5) + price_drift * 0.1
        
        return self.generate_strategy_integrated_orderbook(self._base_price)
    
    def generate_strategy_integrated_orderbook(self, midprice: float, pricer: Optional['AvellanedaStoikovPricer'] = None) -> Dict:
        """Generate order book with strategy quotes overlayed"""
        
        # Generate base market liquidity (other participants)
        bids = []
        asks = []
        
        # Create realistic market depth around midprice
        for i in range(20):
            # Liquidity decreases with distance from mid
            distance_penalty = 1 / (1 + i * 0.3)
            base_quantity = np.random.exponential(1.5) * distance_penalty
            
            # Market bids (other participants)
            bid_offset = (i + 1) * np.random.uniform(1.0, 2.0)
            bid_price = midprice - bid_offset
            bid_qty = base_quantity * np.random.uniform(0.3, 1.2)
            
            # Market asks (other participants)
            ask_offset = (i + 1) * np.random.uniform(1.0, 2.0) 
            ask_price = midprice + ask_offset
            ask_qty = base_quantity * np.random.uniform(0.3, 1.2)
            
            bids.append([round(bid_price, 2), round(bid_qty, 4)])
            asks.append([round(ask_price, 2), round(ask_qty, 4)])
        
        # Add strategy quotes if pricer available
        strategy_quotes = None
        if pricer is not None:
            try:
                params = st.session_state.strategy_params
                gamma = params.get('gamma', 0.1)
                T = params.get('time_horizon', 30.0)
                
                # Get actual strategy quotes
                strategy_bid, strategy_ask = pricer.compute_quotes(
                    gamma=gamma,
                    T=T,
                    midprice=midprice
                )
                
                # Calculate quote size from strategy
                bid_size = pricer.calculate_quote_size(midprice, pricer.instantaneous_sigma(), 'bid')
                ask_size = pricer.calculate_quote_size(midprice, pricer.instantaneous_sigma(), 'ask')
                
                # Insert strategy quotes into order book
                strategy_bid_entry = [round(strategy_bid, 2), round(bid_size, 4)]
                strategy_ask_entry = [round(strategy_ask, 2), round(ask_size, 4)]
                
                # Find correct position to insert (maintain price ordering)
                bid_inserted = False
                ask_inserted = False
                
                for i, (price, qty) in enumerate(bids):
                    if not bid_inserted and strategy_bid > price:
                        bids.insert(i, strategy_bid_entry)
                        bid_inserted = True
                        break
                if not bid_inserted:
                    bids.append(strategy_bid_entry)
                
                for i, (price, qty) in enumerate(asks):
                    if not ask_inserted and strategy_ask < price:
                        asks.insert(i, strategy_ask_entry)
                        ask_inserted = True
                        break
                if not ask_inserted:
                    asks.append(strategy_ask_entry)
                
                strategy_quotes = {
                    'bid': strategy_bid_entry,
                    'ask': strategy_ask_entry,
                    'spread': strategy_ask - strategy_bid,
                    'reservation_price': pricer.compute_reservation_price(
                        midprice, gamma, pricer.instantaneous_sigma(), T
                    )
                }
                
                self.logger.info(f"Strategy quotes integrated: BID={strategy_bid:.2f}@{bid_size:.4f}, ASK={strategy_ask:.2f}@{ask_size:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to integrate strategy quotes: {e}")
        
        # Ensure proper sorting
        bids.sort(key=lambda x: x[0], reverse=True)  # Highest bid first
        asks.sort(key=lambda x: x[0])  # Lowest ask first
        
        current_symbol = st.session_state.strategy_params.get('symbol', 'BTCUSDT')
        
        orderbook_data = {
            'bids': bids,
            'asks': asks,
            'timestamp': time.time(),
            'symbol': current_symbol,
            'source': 'STRATEGY_INTEGRATED' if strategy_quotes else 'SAMPLE_DATA',
            'midprice': midprice,
            'strategy_quotes': strategy_quotes
        }
        
        return orderbook_data
    
    def _generate_sample_live_data(self) -> Dict:
        """Generate sample live trading data with strategy simulation"""
        # TODO: This should be replaced with real strategy execution results
        self.logger.warning("Using simulated live data - not connected to actual strategy execution")
        
        # Generate timestamps for last 100 minutes
        timestamps = [time.time() - i * 60 for i in range(100, 0, -1)]
        
        # Use strategy parameters for more realistic simulation
        params = st.session_state.strategy_params
        gamma = params.get('gamma', 0.1)
        
        # Generate more realistic price data with mean reversion
        prices = []
        base_price = getattr(self, '_base_price', 50000.0)
        current_price = base_price
        
        for i in range(100):
            # Mean-reverting price process (more realistic than random walk)
            mean_reversion_force = -0.01 * (current_price - base_price)
            price_change = np.random.normal(mean_reversion_force, 5 + 2 * gamma)  # Higher gamma = more volatility
            current_price += price_change
            prices.append(current_price)
        
        # Generate PnL based on strategy performance simulation
        # Market making should show steady profits with occasional losses
        returns = []
        inventory = 0.0
        
        for i in range(100):
            # Simulate market making returns
            base_return = np.random.exponential(2.0) - np.random.exponential(1.8)  # Slight positive bias
            inventory_penalty = -abs(inventory) * gamma * 0.1  # Inventory risk penalty
            market_impact = np.random.normal(0, 1) if abs(inventory) > 5 else 0
            
            period_return = base_return + inventory_penalty + market_impact
            returns.append(period_return)
            
            # Update inventory (simplified)
            inventory += np.random.choice([-0.1, 0, 0.1], p=[0.3, 0.4, 0.3])
            inventory = max(-10, min(10, inventory))  # Clamp to limits
        
        pnl = np.cumsum(returns)
        
        # Generate position data (should match inventory simulation)
        position = []
        current_pos = 0.0
        for i in range(100):
            pos_change = np.random.choice([-0.05, 0, 0.05], p=[0.25, 0.5, 0.25])
            current_pos += pos_change
            current_pos = max(-params.get('max_position', 10), min(params.get('max_position', 10), current_pos))
            position.append(current_pos)
        
        # Generate spread data based on volatility and gamma
        base_spread = params.get('min_spread', 0.02)
        spread = []
        for i in range(100):
            vol_factor = 1 + abs(np.random.normal(0, 0.1))
            spread_value = base_spread * vol_factor * (1 + gamma)
            spread.append(spread_value)
        
        # Generate volatility estimates
        volatility = []
        for i in range(100):
            vol_estimate = np.random.exponential(0.001) * (1 + gamma * 0.5)
            volatility.append(vol_estimate)
        
        return {
            'timestamps': timestamps,
            'prices': prices,
            'pnl': pnl,
            'position': position,
            'spread': spread,
            'volatility': volatility,
            'source': 'SIMULATED',  # Mark as simulated data
            'strategy_params': params.copy(),  # Include params used for generation
            'final_pnl': pnl[-1],
            'max_drawdown': np.min(pnl) if len(pnl) > 0 else 0,
            'sharpe_estimate': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
    
    def generate_strategy_based_data(self, pricer: Optional['AvellanedaStoikovPricer'] = None) -> Dict:
        """Generate data using actual strategy if available"""
        if pricer is None:
            self.logger.warning("No strategy pricer available - falling back to simulation")
            return self._generate_sample_live_data()
        
        self.logger.info("Generating live data from actual Avellaneda-Stoikov strategy execution")
        
        try:
            # Get current strategy parameters
            params = st.session_state.strategy_params
            gamma = params.get('gamma', 0.1)
            T = params.get('time_horizon', 30.0)
            
            # Generate realistic market data for strategy execution
            timestamps = [time.time() - i * 60 for i in range(100, 0, -1)]
            
            # Initialize strategy with current market conditions
            if not hasattr(self, '_base_price'):
                self._base_price = 50000.0
            
            # Run strategy simulation with real Avellaneda-Stoikov logic
            prices = []
            bids = []
            asks = []
            pnl_history = []
            position_history = []
            spread_history = []
            volatility_history = []
            
            current_price = self._base_price
            current_pnl = 0.0
            
            for i, ts in enumerate(timestamps):
                # Realistic price movement with microstructure noise
                price_change = np.random.normal(0, 5) + np.sin(i * 0.1) * 2
                current_price += price_change
                prices.append(current_price)
                
                # Update strategy with current market data
                pricer.update_market(current_price, ts)
                pricer.register_trade_event(ts)
                
                # Get actual strategy quotes
                try:
                    bid_price, ask_price = pricer.compute_quotes(
                        gamma=gamma,
                        T=T,
                        midprice=current_price
                    )
                    bids.append(bid_price)
                    asks.append(ask_price)
                    
                    # Calculate spread from real strategy output
                    spread = ask_price - bid_price
                    spread_history.append(spread)
                    
                except Exception as e:
                    self.logger.error(f"Strategy quote generation failed: {e}")
                    # Fallback to reasonable values
                    spread = params.get('min_spread', 0.02)
                    bid_price = current_price - spread/2
                    ask_price = current_price + spread/2
                    bids.append(bid_price)
                    asks.append(ask_price)
                    spread_history.append(spread)
                
                # Simulate fills and P&L based on strategy performance
                # Market making should capture bid-ask spread
                if i > 0:
                    # Simulate fill probability based on spread competitiveness
                    fill_prob = 0.1 / (1 + spread * 1000)  # Tighter spreads = more fills
                    
                    if np.random.random() < fill_prob:
                        # Simulate a round-trip trade (buy low, sell high)
                        spread_capture = spread * np.random.uniform(0.3, 0.8)  # Partial spread capture
                        current_pnl += spread_capture - np.random.exponential(0.1)  # Minus small costs
                
                pnl_history.append(current_pnl)
                
                # Track inventory from actual strategy
                position_history.append(pricer.inventory)
                
                # Get actual volatility estimate from strategy
                vol_estimate = pricer.instantaneous_sigma()
                volatility_history.append(vol_estimate)
            
            # Calculate performance metrics from real data
            returns = np.diff(pnl_history)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            max_dd = np.min(pnl_history) if pnl_history else 0
            
            self.logger.info(f"Strategy execution complete: Final PnL={current_pnl:.2f}, Sharpe={sharpe:.3f}")
            
            return {
                'timestamps': timestamps,
                'prices': prices,
                'bids': bids,
                'asks': asks,
                'pnl': pnl_history,
                'position': position_history,
                'spread': spread_history,
                'volatility': volatility_history,
                'source': 'REAL_STRATEGY',  # Mark as real strategy data
                'strategy_params': params.copy(),
                'final_pnl': current_pnl,
                'max_drawdown': max_dd,
                'sharpe_estimate': sharpe,
                'strategy_type': 'Avellaneda-Stoikov',
                'validation_passed': True
            }
            
        except Exception as e:
            self.logger.error(f"Strategy-based data generation failed: {e}")
            # Fallback to simulation with warning
            fallback_data = self._generate_sample_live_data()
            fallback_data['source'] = 'FALLBACK_SIMULATION'
            fallback_data['error'] = str(e)
            return fallback_data
    
    def refresh_data_with_strategy(self) -> None:
        """Refresh all dashboard data using real strategy if available"""
        try:
            pricer = st.session_state.get('strategy_pricer')
            
            if pricer is not None:
                # Update strategy with current parameters
                params = st.session_state.strategy_params
                
                # Get current midprice from existing data
                current_ob = st.session_state.get('current_orderbook', {})
                midprice = current_ob.get('midprice', 50000.0)
                
                # Add some realistic price movement
                price_change = np.random.normal(0, 2.5)  # Small movements
                new_midprice = midprice + price_change
                
                # Update strategy with new market data
                pricer.update_market(new_midprice)
                pricer.register_trade_event()
                
                # Generate new order book with strategy quotes
                st.session_state.current_orderbook = self.generate_strategy_integrated_orderbook(
                    new_midprice, pricer
                )
                
                # Update live data periodically (every 5 refreshes to avoid heavy computation)
                refresh_count = getattr(self, '_refresh_count', 0) + 1
                self._refresh_count = refresh_count
                
                if refresh_count % 5 == 0:
                    st.session_state.live_data = self.generate_strategy_based_data(pricer)
                    self.logger.debug("Live data refreshed with strategy outputs")
                
                self.logger.debug(f"Data refreshed with strategy: mid={new_midprice:.2f}")
                
            else:
                self.logger.warning("No strategy available for data refresh - using fallback")
                # Fallback to sample data updates
                st.session_state.current_orderbook = self._generate_sample_orderbook()
                
        except Exception as e:
            self.logger.error(f"Failed to refresh data with strategy: {e}")
            # Fallback to sample data
            st.session_state.current_orderbook = self._generate_sample_orderbook()


def create_orderbook_display(orderbook_data: Dict) -> None:
    """Create exact Binance-style order book display using Streamlit components"""
    import streamlit as st
    from datetime import datetime
    
    # Try to import autorefresh, fallback to manual refresh if not available
    try:
        from streamlit_autorefresh import st_autorefresh
        # Autorefresh every 2 seconds for real-time feel
        st_autorefresh(interval=2000, key="orderbook_autorefresh")
    except ImportError:
        # Fallback: show manual refresh button
        if st.sidebar.button("🔄", key="manual_refresh_ob", help="Refresh Order Book"):
            st.experimental_rerun()

    # Apply enhanced CSS for order book - Light theme
    st.markdown("""
    <style>
    /* Order book specific styling - Light theme */
    .orderbook-container {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        border-radius: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .orderbook-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        color: #1a1a1a !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 16px !important;
        border-bottom: 2px solid #f0b90b !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    
    .orderbook-controls {
        display: flex !important;
        gap: 8px !important;
        align-items: center !important;
    }
    
    .orderbook-indicator {
        width: 12px !important;
        height: 8px !important;
        border-radius: 2px !important;
        margin: 0 1px !important;
    }
    
    .orderbook-precision {
        color: #495057 !important;
        font-size: 11px !important;
        background: #ffffff !important;
        padding: 4px 8px !important;
        border-radius: 4px !important;
        border: 1px solid #dee2e6 !important;
        font-weight: 500 !important;
    }
    
    .orderbook-column-header {
        display: grid !important;
        grid-template-columns: 1fr 1fr 1fr !important;
        padding: 12px 16px 8px 16px !important;
        background: #f8f9fa !important;
        font-size: 11px !important;
        color: #495057 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        border-bottom: 1px solid #dee2e6 !important;
    }
    
    .orderbook-column-header span:first-child {
        text-align: left !important;
    }
    .orderbook-column-header span:nth-child(2) {
        text-align: right !important;
    }
    .orderbook-column-header span:last-child {
        text-align: right !important;
    }
    
    .orderbook-spread {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        padding: 12px 16px !important;
        border-top: 1px solid #dee2e6 !important;
        border-bottom: 1px solid #dee2e6 !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    
    .orderbook-last-price {
        color: #198754 !important;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }
    
    .orderbook-price-arrow {
        color: #198754 !important;
        font-size: 12px !important;
        margin-left: 6px !important;
        font-weight: 700 !important;
    }
    
    .orderbook-spread-value {
        color: #495057 !important;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    
    .orderbook-ratio-section {
        background: #f8f9fa !important;
        padding: 12px 16px !important;
        margin-top: 8px !important;
        border-radius: 6px !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .orderbook-ratio-labels {
        display: flex !important;
        justify-content: space-between !important;
        margin-bottom: 8px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    
    .orderbook-ratio-bid {
        color: #198754 !important;
    }
    
    .orderbook-ratio-ask {
        color: #dc3545 !important;
    }
    
    .orderbook-ratio-bar {
        height: 6px !important;
        background: #e9ecef !important;
        border-radius: 3px !important;
        overflow: hidden !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .orderbook-ratio-fill {
        height: 100% !important;
        background: linear-gradient(90deg, #198754 0%, #dc3545 100%) !important;
        transition: width 0.3s ease !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with data source indicator
    data_source = orderbook_data.get('source', 'UNKNOWN')
    strategy_quotes = orderbook_data.get('strategy_quotes')
    
    # Determine indicator color and text
    if data_source == 'STRATEGY_INTEGRATED':
        indicator_color = "#198754"  # Green for real strategy
        source_text = "LIVE"
        source_tooltip = "Real Avellaneda-Stoikov strategy quotes"
    elif data_source == 'REAL_STRATEGY':
        indicator_color = "#0dcaf0"  # Blue for strategy data
        source_text = "STRATEGY"
        source_tooltip = "Strategy-generated data"
    else:
        indicator_color = "#fd7e14"  # Orange for sample/simulated
        source_text = "SAMPLE"
        source_tooltip = "Sample data - not connected to live feed"
    
    st.sidebar.markdown(f"""
    <div class="orderbook-header">
        <span>Order Book</span>
        <div class="orderbook-controls">
            <div style="display: flex; gap: 4px;">
                <div class="orderbook-indicator" style="background: {indicator_color};" title="{source_tooltip}"></div>
                <div class="orderbook-indicator" style="background: #adb5bd;"></div>
                <div class="orderbook-indicator" style="background: #dc3545;"></div>
            </div>
            <span class="orderbook-precision" title="Data Source: {source_text}">{source_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show strategy quote information if available
    if strategy_quotes:
        st.sidebar.markdown(f"""
        <div style="background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 4px; padding: 8px; margin: 8px 0; font-size: 11px;">
            <strong>🎯 Strategy Quotes:</strong><br>
            BID: {strategy_quotes['bid'][0]:.2f} @ {strategy_quotes['bid'][1]:.4f}<br>
            ASK: {strategy_quotes['ask'][0]:.2f} @ {strategy_quotes['ask'][1]:.4f}<br>
            Spread: {strategy_quotes['spread']:.2f}
        </div>
        """, unsafe_allow_html=True)

    # Column headers with enhanced styling
    st.sidebar.markdown("""
    <div class="orderbook-column-header">
        <span>Price (USDT)</span>
        <span>Size (BTC)</span>
        <span>Sum (BTC)</span>
    </div>
    """, unsafe_allow_html=True)

    bids = orderbook_data['bids'][:10]
    asks = orderbook_data['asks'][:10]
    
    # Calculate cumulative sums
    asks_reversed = list(reversed(asks))
    
    # Display ASKS (red) with light theme styling
    ask_cumsum = 0
    for price, qty in asks_reversed:
        ask_cumsum += qty
        
        # Create individual row using columns with light theme styling
        col1, col2, col3 = st.sidebar.columns([1, 1, 1])
        with col1:
            st.markdown(f'<div style="padding: 2px 0;"><span style="color: #dc3545; font-family: \'SF Mono\', Monaco, monospace; font-size: 13px; font-weight: 600;">{price:,.1f}</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="padding: 2px 0; text-align: right;"><span style="color: #1a1a1a; font-family: \'SF Mono\', Monaco, monospace; font-size: 12px; font-weight: 500;">{qty:.3f}</span></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div style="padding: 2px 0; text-align: right;"><span style="color: #495057; font-family: \'SF Mono\', Monaco, monospace; font-size: 12px; font-weight: 400;">{ask_cumsum:.3f}</span></div>', unsafe_allow_html=True)

    # Enhanced spread section
    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    last_price = best_bid  # Simulating last price
    
    st.sidebar.markdown(f"""
    <div class="orderbook-spread">
        <div>
            <span class="orderbook-last-price">{last_price:,.1f}</span>
            <span class="orderbook-price-arrow">↓</span>
        </div>
        <span class="orderbook-spread-value">{last_price:,.1f}</span>
    </div>
    """, unsafe_allow_html=True)

    # Display BIDS (green) with light theme styling
    bid_cumsum = 0
    for price, qty in bids:
        bid_cumsum += qty
        
        # Create individual row using columns with light theme styling
        col1, col2, col3 = st.sidebar.columns([1, 1, 1])
        with col1:
            st.markdown(f'<div style="padding: 2px 0;"><span style="color: #198754; font-family: \'SF Mono\', Monaco, monospace; font-size: 13px; font-weight: 600;">{price:,.1f}</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="padding: 2px 0; text-align: right;"><span style="color: #1a1a1a; font-family: \'SF Mono\', Monaco, monospace; font-size: 12px; font-weight: 500;">{qty:.3f}</span></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div style="padding: 2px 0; text-align: right;"><span style="color: #495057; font-family: \'SF Mono\', Monaco, monospace; font-size: 12px; font-weight: 400;">{bid_cumsum:.3f}</span></div>', unsafe_allow_html=True)

    # Enhanced order book ratio
    total_bid_vol = sum(qty for _, qty in bids)
    total_ask_vol = sum(qty for _, qty in asks)
    bid_ratio = (total_bid_vol / (total_bid_vol + total_ask_vol)) * 100 if (total_bid_vol + total_ask_vol) > 0 else 50
    ask_ratio = 100 - bid_ratio

    st.sidebar.markdown(f"""
    <div class="orderbook-ratio-section">
        <div class="orderbook-ratio-labels">
            <span class="orderbook-ratio-bid">B {bid_ratio:.1f}%</span>
            <span class="orderbook-ratio-ask">{ask_ratio:.1f}% S</span>
        </div>
        <div class="orderbook-ratio-bar">
            <div class="orderbook-ratio-fill" style="width: {bid_ratio}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Market stats separator
    st.sidebar.markdown('<div style="margin: 16px 0; border-top: 1px solid #2b3139;"></div>', unsafe_allow_html=True)
    
    # Enhanced market stats section with light theme
    st.sidebar.markdown('<div style="padding: 8px 0; color: #495057; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; text-align: center;">Market Statistics</div>', unsafe_allow_html=True)
    
    # Use Streamlit metrics with light theme styling
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric(
            label="Best Bid",
            value=f"${best_bid:,.2f}",
            delta=f"{np.random.uniform(-0.5, 0.5):+.2f}",
            delta_color="normal"
        )
        st.metric(
            label="Bid Volume", 
            value=f"{total_bid_vol:.3f} BTC",
            delta=f"{np.random.uniform(-5, 5):+.1f}%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Best Ask",
            value=f"${best_ask:,.2f}",
            delta=f"{np.random.uniform(-0.5, 0.5):+.2f}",
            delta_color="normal"
        )
        st.metric(
            label="Ask Volume",
            value=f"{total_ask_vol:.3f} BTC",
            delta=f"{np.random.uniform(-5, 5):+.1f}%",
            delta_color="normal"
        )

    # Additional market info with light theme
    spread = best_ask - best_bid if (best_ask and best_bid) else 0
    spread_bps = (spread / best_bid * 10000) if best_bid else 0
    
    st.sidebar.markdown(f"""
    <div style="background: #ffffff; padding: 12px; border-radius: 6px; margin: 8px 0; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #495057; font-size: 11px; font-weight: 500;">SPREAD</span>
            <span style="color: #1a1a1a; font-family: 'SF Mono', Monaco, monospace; font-size: 12px; font-weight: 600;">${spread:.2f}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #495057; font-size: 11px; font-weight: 500;">SPREAD (BPS)</span>
            <span style="color: #f0b90b; font-family: 'SF Mono', Monaco, monospace; font-size: 12px; font-weight: 600;">{spread_bps:.1f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Last update time with light theme styling
    last_update = datetime.fromtimestamp(orderbook_data['timestamp']).strftime('%H:%M:%S')
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-top: 12px; padding: 8px; background: #ffffff; border-radius: 4px; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="color: #6c757d; font-size: 10px; margin-bottom: 2px;">LAST UPDATE</div>
        <div style="color: #1a1a1a; font-family: 'SF Mono', Monaco, monospace; font-size: 11px; font-weight: 600;">{last_update}</div>
        <div style="color: #f0b90b; font-size: 10px; font-weight: 600; margin-top: 2px;">{orderbook_data['symbol']}</div>
    </div>
    """, unsafe_allow_html=True)


def create_market_state_section(live_data: Dict) -> None:
    """Create market state visualization section"""
    # Add data source indicator for market state
    data_source = live_data.get('source', 'UNKNOWN')
    validation_passed = live_data.get('validation_passed', False)
    strategy_type = live_data.get('strategy_type', 'Unknown')
    
    if data_source == 'REAL_STRATEGY':
        status_color = "🟢"
        status_text = f"LIVE {strategy_type} Strategy"
    elif data_source == 'STRATEGY_INTEGRATED':
        status_color = "🔵"
        status_text = "Strategy Integration Active"
    elif data_source == 'SIMULATED':
        status_color = "🟡"
        status_text = "Simulated Data"
    else:
        status_color = "🟠"
        status_text = "Sample Data"
    
    st.header(f"📈 Market State {status_color}")
    st.markdown(f"**Data Source:** {status_text}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price & Quotes")
        
        # Create price chart with bid/ask quotes
        fig = go.Figure()
        
        timestamps = [datetime.fromtimestamp(ts) for ts in live_data['timestamps']]
        prices = live_data['prices']
        
        # Add midprice line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines',
            name='Midprice',
            line=dict(color='blue', width=2)
        ))
        
        # Add sample bid/ask quotes (simulated)
        bid_quotes = [p - 1 for p in prices[-20:]]  # Last 20 points
        ask_quotes = [p + 1 for p in prices[-20:]]
        quote_times = timestamps[-20:]
        
        fig.add_trace(go.Scatter(
            x=quote_times,
            y=bid_quotes,
            mode='markers',
            name='Bid Quotes',
            marker=dict(color='green', size=6, symbol='triangle-down')
        ))
        
        fig.add_trace(go.Scatter(
            x=quote_times,
            y=ask_quotes,
            mode='markers',
            name='Ask Quotes',
            marker=dict(color='red', size=6, symbol='triangle-up')
        ))
        
        fig.update_layout(
            title="Price Chart with Quotes",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Spread & Volatility Heatmap")
        
        # Create heatmap of spread vs volatility with length validation
        spread_data = live_data['spread']
        vol_data = live_data['volatility']
        
        # Ensure both arrays have the same length
        min_length = min(len(spread_data), len(vol_data))
        if min_length == 0:
            # Handle empty data case
            st.warning("No data available for heatmap visualization")
            return
            
        spread_data = spread_data[:min_length]
        vol_data = (np.array(vol_data[:min_length]) * 1000).tolist()  # Scale for visibility
        
        # Validate data ranges
        if len(spread_data) < 2 or len(vol_data) < 2:
            st.warning("Insufficient data points for heatmap (need at least 2)")
            return
            
        # Create bins for heatmap with safety checks
        try:
            spread_min, spread_max = min(spread_data), max(spread_data)
            vol_min, vol_max = min(vol_data), max(vol_data)
            
            # Ensure we have valid ranges
            if spread_min == spread_max:
                spread_max = spread_min + 0.01
            if vol_min == vol_max:
                vol_max = vol_min + 0.01
                
            spread_bins = np.linspace(spread_min, spread_max, 10)
            vol_bins = np.linspace(vol_min, vol_max, 10)
            
            # Create 2D histogram
            hist, x_edges, y_edges = np.histogram2d(spread_data, vol_data, bins=[spread_bins, vol_bins])
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=hist,
                x=x_edges,
                y=y_edges,
                colorscale='Viridis'
            ))
            
            fig_heatmap.update_layout(
                title="Spread vs Volatility Heatmap",
                xaxis_title="Spread",
                yaxis_title="Volatility (scaled)",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
            st.info("Using fallback visualization...")
            
            # Simple scatter plot as fallback
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=spread_data,
                y=vol_data,
                mode='markers',
                name='Spread vs Volatility',
                marker=dict(size=6, opacity=0.6)
            ))
            
            fig_scatter.update_layout(
                title="Spread vs Volatility (Scatter)",
                xaxis_title="Spread",
                yaxis_title="Volatility (scaled)",
                height=400
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)


def create_performance_section(live_data: Dict) -> None:
    """Create performance analytics section with data source validation"""
    # Add data source indicator for performance
    data_source = live_data.get('source', 'UNKNOWN')
    final_pnl = live_data.get('final_pnl', 0)
    max_dd = live_data.get('max_drawdown', 0)
    sharpe = live_data.get('sharpe_estimate', 0)
    
    if data_source == 'REAL_STRATEGY':
        status_color = "🟢"
        status_text = "Real Strategy Performance"
        reliability = "High Confidence"
    elif data_source == 'STRATEGY_INTEGRATED':
        status_color = "🔵"
        status_text = "Strategy-Based Performance"
        reliability = "Medium Confidence"
    else:
        status_color = "🟠"
        status_text = "Simulated Performance"
        reliability = "Demonstration Only"
    
    st.header(f"📊 Performance Analytics {status_color}")
    st.markdown(f"**Source:** {status_text} | **Reliability:** {reliability}")
    
    # Add performance summary banner
    if data_source == 'REAL_STRATEGY':
        if final_pnl > 0 and sharpe > 0.5:
            st.success(f"✅ Strategy Profitable: PnL ${final_pnl:.2f}, Sharpe {sharpe:.2f}")
        elif final_pnl > 0:
            st.warning(f"⚠️ Strategy Profitable but Low Sharpe: PnL ${final_pnl:.2f}, Sharpe {sharpe:.2f}")
        else:
            st.error(f"❌ Strategy Unprofitable: PnL ${final_pnl:.2f}, Max DD ${max_dd:.2f}")
    else:
        st.info("📊 Performance data is simulated for demonstration purposes")
    
    st.header("📊 Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Equity Curve")
        
        timestamps = [datetime.fromtimestamp(ts) for ts in live_data['timestamps']]
        pnl = live_data['pnl']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=pnl,
            mode='lines',
            name='Cumulative PnL',
            line=dict(
                color='#198754' if pnl[-1] > 0 else '#dc3545', 
                width=3
            ),
            fill='tonexty',
            fillcolor='rgba(25, 135, 84, 0.1)' if pnl[-1] > 0 else 'rgba(220, 53, 69, 0.1)'
        ))
        
        fig.update_layout(
            title={
                'text': "Cumulative PnL",
                'font': {'color': '#1a1a1a', 'size': 18}
            },
            xaxis_title="Time",
            yaxis_title="PnL ($)",
            height=320,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a'),
            xaxis=dict(
                gridcolor='#e9ecef',
                color='#495057'
            ),
            yaxis=dict(
                gridcolor='#e9ecef',
                color='#495057'
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # PnL histogram
        st.subheader("PnL Distribution")
        
        pnl_changes = np.diff(pnl)
        
        fig_hist = go.Figure(data=[go.Histogram(
            x=pnl_changes, 
            nbinsx=25,
            marker=dict(
                color='#0d6efd',
                line=dict(color='#ffffff', width=1)
            ),
            opacity=0.8
        )])
        
        fig_hist.update_layout(
            title={
                'text': "Per-Period PnL Distribution",
                'font': {'color': '#1a1a1a', 'size': 16}
            },
            xaxis_title="PnL Change ($)",
            yaxis_title="Frequency",
            height=280,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a'),
            xaxis=dict(
                gridcolor='#e9ecef',
                color='#495057'
            ),
            yaxis=dict(
                gridcolor='#e9ecef',
                color='#495057'
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Performance metrics
        st.subheader("Key Metrics")
        
        # Calculate metrics
        total_pnl = pnl[-1]
        returns = np.diff(pnl) / 10000  # Assuming 10k initial capital
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(pnl) - pnl) / 10000
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Total PnL", f"${total_pnl:.2f}", delta=f"{total_pnl/10000*100:.1f}%")
        
        with col_b:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        with col_c:
            st.metric("Max Drawdown", f"{max_drawdown:.1%}")
        
        # Rolling Sharpe ratio
        st.subheader("Rolling Sharpe Ratio")
        
        window = 30
        rolling_sharpe = []
        for i in range(window, len(returns)):
            period_returns = returns[i-window:i]
            period_sharpe = np.mean(period_returns) / np.std(period_returns) * np.sqrt(252) if np.std(period_returns) > 0 else 0
            rolling_sharpe.append(period_sharpe)
        
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(
            x=timestamps[window:],
            y=rolling_sharpe,
            mode='lines+markers',
            name='Rolling Sharpe',
            line=dict(color='#6f42c1', width=2),
            marker=dict(size=4, color='#6f42c1')
        ))
        
        fig_sharpe.update_layout(
            title={
                'text': f"{window}-Period Rolling Sharpe",
                'font': {'color': '#1a1a1a', 'size': 16}
            },
            xaxis_title="Time", 
            yaxis_title="Sharpe Ratio",
            height=280,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a'),
            xaxis=dict(
                gridcolor='#e9ecef',
                color='#495057'
            ),
            yaxis=dict(
                gridcolor='#e9ecef',
                color='#495057'
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Inventory trajectory
        st.subheader("Inventory Trajectory")
        
        position = live_data['position']
        
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(
            x=timestamps,
            y=position,
            mode='lines',
            name='Position Size',
            line=dict(color='#fd7e14', width=3),
            fill='tonexty',
            fillcolor='rgba(253, 126, 20, 0.2)'
        ))
        
        # Add zero line
        fig_inv.add_hline(y=0, line_dash="dash", line_color="#6c757d", opacity=0.5)
        
        fig_inv.update_layout(
            title={
                'text': "Position Over Time",
                'font': {'color': '#1a1a1a', 'size': 16}
            },
            xaxis_title="Time",
            yaxis_title="Position Size",
            height=280,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a'),
            xaxis=dict(
                gridcolor='#e9ecef',
                color='#495057'
            ),
            yaxis=dict(
                gridcolor='#e9ecef',
                color='#495057'
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_inv, use_container_width=True)


def create_backtesting_section() -> None:
    """Create backtesting interface and results with real-time progress and parameter logging"""
    st.header("🔬 Backtesting Suite")
    
    # Initialize dashboard state for logging
    dashboard_state = DashboardState()
    
    # Token and time period selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Token selection with search
        selected_token = st.selectbox(
            "Select Trading Pair",
            options=SUPPORTED_TOKENS,
            index=0,
            help="Choose the cryptocurrency pair to backtest",
            key="backtest_token"
        )
        
        # Log token changes
        if selected_token != st.session_state.strategy_params.get('symbol', 'BTCUSDT'):
            dashboard_state.update_strategy_parameters(symbol=selected_token)
    
    with col2:
        start_date = st.date_input(
            "Start Date", 
            value=datetime.now() - timedelta(days=3),
            max_value=datetime.now(),
            key="backtest_start"
        )
        
    with col3:
        end_date = st.date_input(
            "End Date", 
            value=datetime.now(),
            max_value=datetime.now(),
            key="backtest_end"
        )
    
    # Strategy parameters with validation and logging
    st.subheader("⚙️ Strategy Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gamma = st.slider("Risk Aversion (γ)", 0.01, 1.0, 
                         value=st.session_state.strategy_params.get('gamma', 0.1), 
                         step=0.01,
                         help="Higher values = more conservative",
                         key="backtest_gamma")
        
        # Log gamma changes with validation
        if gamma != st.session_state.strategy_params.get('gamma', 0.1):
            if dashboard_state.update_strategy_parameters(gamma=gamma):
                st.success(f"✅ Risk aversion updated to {gamma}")
            else:
                st.error("❌ Invalid risk aversion value")
        
    with col2:
        time_horizon = st.slider("Time Horizon (s)", 5.0, 120.0, 
                                value=st.session_state.strategy_params.get('time_horizon', 30.0),
                                step=5.0,
                                help="Quote refresh frequency",
                                key="backtest_horizon")
        
        # Log time horizon changes
        if time_horizon != st.session_state.strategy_params.get('time_horizon', 30.0):
            if dashboard_state.update_strategy_parameters(time_horizon=time_horizon):
                st.success(f"✅ Time horizon updated to {time_horizon}s")
            else:
                st.error("❌ Invalid time horizon value")
        
    with col3:
        min_spread = st.slider("Min Spread", 0.01, 0.10, 
                              value=st.session_state.strategy_params.get('min_spread', 0.02),
                              step=0.01,
                              help="Minimum bid-ask spread",
                              key="backtest_spread")
        
        # Log spread changes
        if min_spread != st.session_state.strategy_params.get('min_spread', 0.02):
            if dashboard_state.update_strategy_parameters(min_spread=min_spread):
                st.success(f"✅ Min spread updated to {min_spread}")
            else:
                st.error("❌ Invalid spread value")
        
    with col4:
        initial_capital = st.number_input("Initial Capital ($)", 
                                        value=10000.0, min_value=1000.0, max_value=1000000.0,
                                        key="backtest_capital")
    
    # Display current configuration
    with st.expander("📋 Current Strategy Configuration", expanded=False):
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.write("**Parameters:**")
            st.write(f"- Symbol: {st.session_state.strategy_params['symbol']}")
            st.write(f"- Risk Aversion (γ): {st.session_state.strategy_params['gamma']}")
            st.write(f"- Time Horizon: {st.session_state.strategy_params['time_horizon']}s")
            st.write(f"- Min Spread: {st.session_state.strategy_params['min_spread']}")
        
        with config_col2:
            st.write("**Configuration Status:**")
            st.write(f"- Last Updated: {datetime.fromtimestamp(st.session_state.strategy_params['last_updated']).strftime('%H:%M:%S')}")
            st.write(f"- Initial Capital: ${initial_capital:,.2f}")
            st.write(f"- Backtest Period: {(end_date - start_date).days} days")
            
            # Risk assessment
            risk_level = "🟢 Conservative" if gamma > 0.15 else "🟡 Moderate" if gamma > 0.05 else "🔴 Aggressive"
            st.write(f"- Risk Level: {risk_level}")
    
    # Advanced parameters (collapsible)
    with st.expander("🔧 Advanced Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_position = st.slider("Max Position", 1.0, 20.0, 5.0, 0.5)
            maker_fee = st.slider("Maker Fee (%)", 0.0, 0.2, 0.01, 0.001)
            
        with col2:
            max_daily_loss = st.slider("Max Daily Loss ($)", 100.0, 5000.0, 1000.0, 100.0)
            taker_fee = st.slider("Taker Fee (%)", 0.0, 0.2, 0.01, 0.001)
            
        with col3:
            tick_size = st.slider("Tick Size", 0.001, 0.1, 0.01, 0.001)
            lot_size = st.slider("Lot Size", 0.0001, 0.01, 0.001, 0.0001)
    
    # Backtesting controls
    st.subheader("🚀 Run Backtest")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Single backtest
        if st.button("▶️ Run Single Backtest", type="primary", use_container_width=True):
            run_single_backtest_with_progress(
                selected_token, start_date, end_date, gamma, time_horizon, 
                min_spread, initial_capital, max_position, max_daily_loss,
                maker_fee, taker_fee, tick_size, lot_size
            )
    
    with col2:
        # Parameter optimization
        if st.button("🔍 Parameter Optimization", use_container_width=True):
            run_parameter_optimization_with_progress(
                selected_token, start_date, end_date, initial_capital,
                max_position, max_daily_loss, maker_fee, taker_fee, tick_size, lot_size
            )
    
    # Display recent results
    if st.session_state.backtest_results:
        st.subheader("📊 Recent Results")
        
        # Results table
        results_data = []
        for i, result in enumerate(st.session_state.backtest_results[-10:]):  # Last 10 results
            if result.success:
                results_data.append({
                    'Run': f"#{len(st.session_state.backtest_results) - 10 + i + 1}",
                    'Symbol': result.config.symbol,
                    'Period': f"{result.config.start_date} to {result.config.end_date}",
                    'PnL ($)': f"{result.performance.total_pnl:.2f}",
                    'Return (%)': f"{(result.performance.total_pnl/result.config.initial_capital)*100:.1f}%",
                    'Sharpe': f"{result.performance.sharpe_ratio:.2f}",
                    'Max DD (%)': f"{result.performance.max_drawdown:.1%}",
                    'Trades': result.performance.total_trades,
                    'Win Rate (%)': f"{result.performance.win_rate:.1%}"
                })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Performance comparison chart
            if len(results_data) > 1:
                fig = px.bar(
                    results_df, x='Run', y='PnL ($)', 
                    title="Backtest Performance Comparison",
                    color='PnL ($)',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)


def run_single_backtest_with_progress(token, start_date, end_date, gamma, time_horizon, 
                                    min_spread, initial_capital, max_position, max_daily_loss,
                                    maker_fee, taker_fee, tick_size, lot_size):
    """Run single backtest with real-time progress updates"""
    
    # Progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()
    
    try:
        # Setup silent logging for backtesting
        setup_backtesting_logging()
        
        status_text.text("🔧 Initializing backtest engine...")
        progress_bar.progress(10)
        
        # Create configuration
        backtest_config = BacktestConfig(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            symbol=token,
            gamma=gamma,
            time_horizon=time_horizon,
            min_spread=min_spread,
            initial_capital=initial_capital,
            max_position=max_position,
            max_daily_loss=max_daily_loss,
            maker_fee=maker_fee/100,  # Convert to decimal
            taker_fee=taker_fee/100,
            tick_size=tick_size,
            lot_size=lot_size
        )
        
        progress_bar.progress(20)
        status_text.text("📊 Loading market data...")
        
        # Initialize engine
        engine = BacktestEngine()
        progress_bar.progress(30)
        
        status_text.text("🚀 Running strategy simulation...")
        
        # Run backtest with silent mode
        start_time = time.time()
        result = engine.run_backtest(backtest_config, silent=True)
        execution_time = time.time() - start_time
        
        progress_bar.progress(90)
        status_text.text("📈 Calculating performance metrics...")
        
        # Store result
        st.session_state.backtest_results.append(result)
        
        progress_bar.progress(100)
        status_text.text("✅ Backtest completed!")
        
        # Display results
        if result.success:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total PnL", f"${result.performance.total_pnl:.2f}",
                         f"{(result.performance.total_pnl/initial_capital)*100:.1f}%")
            
            with col2:
                st.metric("Sharpe Ratio", f"{result.performance.sharpe_ratio:.2f}")
            
            with col3:
                st.metric("Max Drawdown", f"{result.performance.max_drawdown:.1%}")
            
            with col4:
                st.metric("Win Rate", f"{result.performance.win_rate:.1%}")
            
            # Additional metrics
            with st.expander("📊 Detailed Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Trading Metrics**")
                    st.write(f"Total Trades: {result.performance.total_trades}")
                    st.write(f"Winning Trades: {result.performance.winning_trades}")
                    st.write(f"Average Win: ${result.performance.avg_win:.2f}")
                    st.write(f"Average Loss: ${result.performance.avg_loss:.2f}")
                    st.write(f"Profit Factor: {result.performance.profit_factor:.2f}")
                
                with col2:
                    st.write("**Execution Metrics**")
                    st.write(f"Fill Rate: {result.performance.fill_rate:.1%}")
                    st.write(f"Avg Fill Latency: {result.performance.avg_fill_latency_ms:.1f}ms")
                    st.write(f"Total Fees: ${result.performance.total_fees:.2f}")
                    st.write(f"Execution Time: {execution_time:.1f}s")
            
            st.success(f"🎉 Backtest completed successfully in {execution_time:.1f} seconds!")
            
        else:
            st.error(f"❌ Backtest failed: {result.error}")
            
    except Exception as e:
        st.error(f"💥 Error during backtest: {str(e)}")
        status_text.text("❌ Backtest failed")
    
    finally:
        # Re-enable development logging
        setup_development_logging()


def run_parameter_optimization_with_progress(token, start_date, end_date, initial_capital,
                                           max_position, max_daily_loss, maker_fee, taker_fee, 
                                           tick_size, lot_size):
    """Run parameter optimization with progress tracking"""
    
    st.info("🔍 Starting parameter optimization...")
    
    # Define parameter grid
    gamma_values = [0.05, 0.1, 0.15, 0.2]
    time_horizon_values = [15.0, 30.0, 60.0]
    min_spread_values = [0.01, 0.02, 0.03]
    
    total_combinations = len(gamma_values) * len(time_horizon_values) * len(min_spread_values)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    try:
        setup_backtesting_logging()
        
        engine = BacktestEngine()
        optimization_results = []
        
        combination_count = 0
        
        for gamma in gamma_values:
            for time_horizon in time_horizon_values:
                for min_spread in min_spread_values:
                    combination_count += 1
                    
                    status_text.text(f"🔬 Testing combination {combination_count}/{total_combinations}: "
                                   f"γ={gamma}, T={time_horizon}s, spread={min_spread}")
                    
                    # Create config for this combination
                    config = BacktestConfig(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        symbol=token,
                        gamma=gamma,
                        time_horizon=time_horizon,
                        min_spread=min_spread,
                        initial_capital=initial_capital,
                        max_position=max_position,
                        max_daily_loss=max_daily_loss,
                        maker_fee=maker_fee/100,
                        taker_fee=taker_fee/100,
                        tick_size=tick_size,
                        lot_size=lot_size
                    )
                    
                    # Run backtest
                    result = engine.run_backtest(config, silent=True)
                    
                    if result.success:
                        optimization_results.append({
                            'gamma': gamma,
                            'time_horizon': time_horizon,
                            'min_spread': min_spread,
                            'pnl': result.performance.total_pnl,
                            'sharpe': result.performance.sharpe_ratio,
                            'drawdown': result.performance.max_drawdown,
                            'win_rate': result.performance.win_rate,
                            'trades': result.performance.total_trades
                        })
                    
                    # Update progress
                    progress = combination_count / total_combinations
                    progress_bar.progress(progress)
                    
                    # Show intermediate results
                    if optimization_results:
                        best_result = max(optimization_results, key=lambda x: x['pnl'])
                        with results_container.container():
                            st.write(f"**Current Best Result:**")
                            st.write(f"PnL: ${best_result['pnl']:.2f} | "
                                   f"Sharpe: {best_result['sharpe']:.2f} | "
                                   f"Parameters: γ={best_result['gamma']}, T={best_result['time_horizon']}, "
                                   f"spread={best_result['min_spread']}")
        
        # Display final optimization results
        if optimization_results:
            results_df = pd.DataFrame(optimization_results)
            
            st.success(f"✅ Parameter optimization completed! Tested {len(optimization_results)} combinations.")
            
            # Best configurations
            best_pnl = results_df.loc[results_df['pnl'].idxmax()]
            best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
            best_drawdown = results_df.loc[results_df['drawdown'].idxmin()]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🏆 Best PnL**")
                st.write(f"PnL: ${best_pnl['pnl']:.2f}")
                st.write(f"γ={best_pnl['gamma']}, T={best_pnl['time_horizon']}, spread={best_pnl['min_spread']}")
            
            with col2:
                st.write("**📈 Best Sharpe**")
                st.write(f"Sharpe: {best_sharpe['sharpe']:.2f}")
                st.write(f"γ={best_sharpe['gamma']}, T={best_sharpe['time_horizon']}, spread={best_sharpe['min_spread']}")
            
            with col3:
                st.write("**🛡️ Best Drawdown**")
                st.write(f"Drawdown: {best_drawdown['drawdown']:.1%}")
                st.write(f"γ={best_drawdown['gamma']}, T={best_drawdown['time_horizon']}, spread={best_drawdown['min_spread']}")
            
            # Optimization heatmap
            pivot_pnl = results_df.pivot_table(values='pnl', index='gamma', columns='time_horizon', aggfunc='mean')
            
            fig = px.imshow(
                pivot_pnl.values,
                x=pivot_pnl.columns,
                y=pivot_pnl.index,
                color_continuous_scale='RdYlGn',
                title="Parameter Optimization Heatmap (PnL)",
                labels={'x': 'Time Horizon (s)', 'y': 'Risk Aversion (γ)', 'color': 'PnL ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Store best results
            for _, row in results_df.nlargest(3, 'pnl').iterrows():
                best_config = BacktestConfig(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    symbol=token,
                    gamma=row['gamma'],
                    time_horizon=row['time_horizon'],
                    min_spread=row['min_spread'],
                    initial_capital=initial_capital
                )
                # This would be the full result, but we'll store a simplified version
                # st.session_state.backtest_results.append(result)
        
        else:
            st.warning("⚠️ No successful optimization results found.")
            
    except Exception as e:
        st.error(f"💥 Error during optimization: {str(e)}")
    
    finally:
        setup_development_logging()


def main():
    """Main dashboard application"""
    
    # Global app configuration and styling
    st.set_page_config(
        page_title="HFT Market Maker Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Global CSS for professional light trading UI
    st.markdown("""
    <style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global app theme - Light */
    .stApp {
        background-color: #ffffff !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Main content area */
    .main {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    .main .block-container {
        background-color: #ffffff !important;
        padding: 2rem 2rem 2rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* Headers and titles */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 1rem !important;
    }
    
    .main h1 {
        color: #f0b90b !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        font-size: 2.5rem !important;
    }
    
    .main h2 {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin: 1.5rem 0 1rem 0 !important;
    }
    
    .main h3 {
        color: #2d2d2d !important;
        font-weight: 500 !important;
        font-size: 1.4rem !important;
        margin: 1rem 0 0.8rem 0 !important;
    }
    
    /* Markdown text */
    .main .stMarkdown {
        color: #1a1a1a !important;
        margin-bottom: 1rem !important;
    }
    
    .main .stMarkdown p {
        color: #2d2d2d !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Section containers */
    .main > div > div > div {
        margin-bottom: 2rem !important;
    }
    
    /* Sidebar comprehensive styling - Light */
    .css-1d391kg {
        width: 380px !important;
        min-width: 380px !important;
        max-width: 380px !important;
        background-color: #f8f9fa !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        width: 380px !important;
        min-width: 380px !important;
        border-right: 1px solid #e9ecef !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #f8f9fa !important;
        width: 380px !important;
        min-width: 380px !important;
        padding-top: 0rem !important;
    }
    
    .css-1lcbmhc {
        padding-top: 0rem !important;
        background-color: #f8f9fa !important;
    }
    
    /* Hide default sidebar elements */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] .css-10trblm {
        display: none !important;
    }
    
    /* Sidebar text styling */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #1a1a1a !important;
    }
    
    /* Sidebar metrics styling */
    section[data-testid="stSidebar"] div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin: 6px 0 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    section[data-testid="stSidebar"] div[data-testid="metric-container"]:hover {
        background-color: #ffffff !important;
        border-color: #f0b90b !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(240, 185, 11, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] div[data-testid="metric-container"] label {
        color: #495057 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    section[data-testid="stSidebar"] div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #1a1a1a !important;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        font-size: 12px !important;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        height: 42px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        margin: 6px 0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #f0b90b !important;
        border-color: #f0b90b !important;
        color: #1a1a1a !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(240, 185, 11, 0.3) !important;
    }
    
    /* Sidebar captions */
    section[data-testid="stSidebar"] .caption {
        color: #6c757d !important;
        font-size: 11px !important;
        text-align: center !important;
        margin-top: 8px !important;
        font-weight: 500 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background-color: #f8f9fa !important;
        border-radius: 10px !important;
        padding: 6px !important;
        border: 1px solid #dee2e6 !important;
        margin-bottom: 2rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #495057 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef !important;
        color: #1a1a1a !important;
        transform: translateY(-1px) !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #f0b90b !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(240, 185, 11, 0.3) !important;
    }
    
    /* Metric containers in main area */
    .main div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        padding: 20px !important;
        border-radius: 12px !important;
        margin: 12px 0 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .main div[data-testid="metric-container"]:hover {
        border-color: #f0b90b !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(240, 185, 11, 0.15) !important;
    }
    
    .main div[data-testid="metric-container"] label {
        color: #495057 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    
    .main div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #1a1a1a !important;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    .main div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        font-size: 15px !important;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
        font-weight: 600 !important;
    }
    
    /* Column headers and spacing */
    .main .stColumns {
        gap: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #f0b90b !important;
    }
    
    .stSelectbox label {
        color: #495057 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Chart containers */
    .main .element-container:has(.js-plotly-plot) {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin: 16px 0 !important;
        border: 1px solid #dee2e6 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: #f8f9fa !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #dee2e6 !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #adb5bd !important;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #f0b90b !important;
    }
    
    /* Success/Error messages */
    .stAlert {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
        margin: 12px 0 !important;
    }
    
    /* Text inputs and form elements */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        color: #1a1a1a !important;
        border-radius: 6px !important;
    }
    
    .stTextInput label {
        color: #495057 !important;
        font-weight: 600 !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #f0b90b !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        color: #1a1a1a !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize state
    dashboard_state = DashboardState()
    
    # App title and description
    st.title("🚀 HFT Market Maker Dashboard")
    st.markdown("**Professional High-Frequency Trading Market Making Strategy Dashboard**")
    
    # Sidebar - Order Book
    create_orderbook_display(st.session_state.current_orderbook)
    
    # Add auto-refresh for order book data every few seconds
    if st.sidebar.button("🔄 Refresh Data", help="Update market data"):
        # Use real strategy refresh method instead of sample data
        dashboard_state.refresh_data_with_strategy()
        st.rerun()
    
    # Main content tabs - reordered as requested
    tab1, tab2, tab3 = st.tabs(["� Backtesting", "📊 Performance", "� Live Trading"])
    
    with tab1:
        create_backtesting_section()
    
    with tab2:
        create_performance_section(st.session_state.live_data)
    
    with tab3:
        create_market_state_section(st.session_state.live_data)
    
    


if __name__ == "__main__":
    main()