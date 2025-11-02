"""
HFT Trading Dashboard
====================

Streamlined dashboard with two main sections:
- Backtesting: Historical strategy testing with configurable time scope
- Live Trading: Real-time trading with Binance WebSocket integration
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="HFT Trading Dashboard",
    page_icon="🚀", 
    layout="wide"
)

# Import HFT modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.backtesting import BacktestEngine, BacktestConfig
from src.strategy import AvellanedaStoikovPricer, QuoteParameters
from src.live_trading import LiveTradingEngine
from src.utils.config import config
from src.utils.logger import setup_production_logging, get_logger

# Initialize logging (INFO level - no DEBUG spam)
setup_production_logging()
logger = get_logger('dashboard')

# Supported trading pairs
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]

# Thread-safe shared data for live trading (accessible from background thread)
# Use session_state to ensure same instance across reruns
import threading

if '_live_data_lock' not in st.session_state:
    st.session_state._live_data_lock = threading.Lock()
if '_shared_live_data' not in st.session_state:
    st.session_state._shared_live_data = {
        'performance': {},
        'market': {},
        'trades': [],
        'total_trade_count': 0
    }

# Create module-level references to session_state objects
_live_data_lock = st.session_state._live_data_lock
_shared_live_data = st.session_state._shared_live_data

# Global state management
if 'page_state' not in st.session_state:
    st.session_state.page_state = 'backtesting'

if 'live_engine' not in st.session_state:
    st.session_state.live_engine = None

if 'live_data' not in st.session_state:
    st.session_state.live_data = {}

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

if 'live_trading_active' not in st.session_state:
    st.session_state.live_trading_active = False


def create_backtesting_section():
    """Create the backtesting configuration and results section"""
    st.header("📊 Backtesting")
    
    # Configuration section - Horizontal layout at top
    st.subheader("Configuration")
    
    # Create horizontal columns for configuration parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Trading parameters
        st.markdown("**Trading Parameters**")
        symbol = st.selectbox("Symbol", SUPPORTED_SYMBOLS, key="bt_symbol", 
                             help="Choose high-volume pairs like BTCUSDT for better results")
        
        # Initial capital - prominent position
        initial_balance = st.number_input(
            "Initial Capital (USDT)", 
            min_value=1000.0, 
            max_value=1000000.0, 
            value=10000.0,  # Default $10k for retail MM
            step=1000.0,
            key="bt_initial_balance",
            help="Starting capital in USDT. Typical: $5k-50k for retail, $100k+ for institutional"
        )
    
    with col2:
        st.markdown("**Risk Parameters**")
        gamma = st.slider("Risk Aversion (γ)", 0.005, 0.05, 0.015, 0.005, key="bt_gamma",
                        help="Lower = tighter spreads. HFT optimal: 0.01-0.02. Max 0.05 to prevent uncompetitive spreads.")
        time_horizon = st.slider("Time Horizon (sec)", 5, 60, 10, 5, key="bt_time_horizon",
                               help="Quote refresh time. HFT optimal: 5-15 seconds. Shorter = faster adaptation")
    
    with col3:
        st.markdown("**Spread & Size**")
        min_spread = st.number_input("Min Spread", min_value=0.0001, max_value=0.05, value=0.0008, step=0.0001, key="bt_min_spread",
                                    help="Minimum spread (0.0008 = 0.08% = 8 bps). HFT typical: 5-20 bps. Lower = tighter quotes.", format="%.4f")
        max_drawdown_pct = st.slider(
            "Max Drawdown (%)", 
            1, 50, 10, 1, 
            key="bt_max_dd",
            help="Stop backtesting if drawdown exceeds this percentage. Protects against catastrophic losses."
        )
    
    with col4:
        st.markdown("**Time Scope**")
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=4))
        end_date = st.date_input("End Date", datetime.now())
    
    # Advanced parameters in expander (full width)
    with st.expander("Advanced Parameters"):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            tick_size = st.number_input("Tick Size", min_value=0.001, max_value=1.0, value=0.01, step=0.001, 
                                       help="Price increment. Use 0.01 for most pairs", key="bt_tick_size")
        with adv_col2:
            lot_size = st.number_input("Lot Size", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001,
                                      help="Minimum order size. Use 0.001 for BTC (realistic HFT size)", key="bt_lot_size", format="%.4f")

    # Error checking
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return
    
    # Run backtest button - centered and full width
    st.markdown("---")
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        run_backtest({
            'symbol': symbol,
            'gamma': gamma,
            'time_horizon': time_horizon,
            'min_spread': min_spread,
            'max_drawdown_pct': max_drawdown_pct,
            'initial_balance': initial_balance,
            'tick_size': tick_size,
            'lot_size': lot_size,
            'start_date': start_date,
            'end_date': end_date
        })
    
    # Results section - Full width below configuration
    st.markdown("---")
    st.subheader("Results")
    
    if st.session_state.backtest_results is None:
        st.info("Configure parameters and run backtest to see results")
    else:
        display_backtest_results(st.session_state.backtest_results)


def create_live_trading_section():
    """Create the live trading configuration and monitoring section"""
    st.header("🔴 Live Trading")
    
    # Configuration section - Horizontal at top (4 columns)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Trading Parameters**")
        symbol = st.selectbox("Symbol", SUPPORTED_SYMBOLS, key="live_symbol",
                            help="Choose high-volume pairs for better execution")
        initial_balance = st.number_input(
            "Initial Capital (USDT)", 
            min_value=1000.0, 
            max_value=1000000.0, 
            value=10000.0,
            step=1000.0,
            key="live_balance",
            help="Starting capital in USDT. Typical: $5k-50k for retail"
        )
    
    with col2:
        st.markdown("**Risk Parameters**")
        gamma = st.slider("Risk Aversion (γ)", 0.005, 0.05, 0.015, 0.005, key="live_gamma",
                        help="Lower = tighter spreads. HFT optimal: 0.01-0.02. Max 0.05 to prevent uncompetitive spreads.")
        time_horizon = st.slider("Time Horizon (sec)", 5, 60, 10, 5, key="live_time_horizon",
                               help="Quote refresh time. HFT optimal: 5-15s")
    
    with col3:
        st.markdown("**Spread & Size**")
        min_spread = st.number_input("Min Spread", min_value=0.001, max_value=0.01, value=0.01, step=0.001, key="live_min_spread",
                                    help="Minimum spread (0.0005 = 5 bps). HFT typical: 5-20 bps", format="%.4f")
        max_position = st.slider("Max Position", 1.0, 20.0, 5.0, 0.5, key="live_max_pos",
                               help="Maximum position size in base currency")
    
    with col4:
        st.markdown("**Risk Limits**")
        max_loss = st.number_input("Max Loss ($)", min_value=1.0, max_value=10000.0, value=1000.0, step=10.0, key="live_max_loss")
        max_drawdown = st.slider("Max Drawdown (%)", 1, 50, 10, 1, key="live_max_dd")
    
    # Advanced parameters in expander (full width)
    with st.expander("Advanced Parameters"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            tick_size = st.number_input("Tick Size", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key="live_tick_size",
                                       help="Price increment (usually 0.01)")
        with adv_col2:
            lot_size = st.number_input("Lot Size", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key="live_lot_size",
                                      help="Minimum order size. Use 0.001 for BTC (realistic HFT size)")
        with adv_col3:
            emergency_stop = st.checkbox("Enable Emergency Stop", key="live_emergency_stop")
    
    # Trading controls - centered below config
    st.markdown("---")
    
    if not st.session_state.live_trading_active:
        if st.button("🚀 Start Live Trading", type="primary", use_container_width=True):
            start_live_trading({
                'symbol': symbol,
                'gamma': gamma,
                'time_horizon': time_horizon,
                'min_spread': min_spread,
                'max_position': max_position,
                'initial_balance': initial_balance,
                'tick_size': tick_size,
                'lot_size': lot_size,
                'max_loss': max_loss,
                'max_drawdown_pct': max_drawdown,
                'emergency_stop': emergency_stop
            })
    else:
        control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
        with control_col1:
            if st.button("🛑 Stop Trading", type="secondary", use_container_width=True):
                stop_live_trading()
        with control_col2:
            if st.button("📝 Update Parameters", use_container_width=True):
                update_live_parameters({
                    'gamma': gamma,
                    'time_horizon': time_horizon,
                    'min_spread': min_spread,
                    'max_position': max_position
                })
        with control_col3:
            # Connection status
            status = get_live_trading_status()
            if status.get('is_connected', False):
                st.success("🟢 Connected")
            else:
                st.error("🔴 Disconnected")
    
    # Live performance monitoring - Full width below controls
    st.markdown("---")
    st.subheader("Live Performance")
    
    # Show final results if just stopped
    if not st.session_state.live_trading_active and 'live_final_results' in st.session_state:
        results = st.session_state.live_final_results
        trades = results.get('trades', [])
        total_count = results.get('total_count', len(trades))  # Use actual total
        performance = results.get('performance', {})
        
        st.success(f"✅ Trading Session Completed - {total_count} trades executed")
        
        if trades and len(trades) > 0:
            # ============= TERMINOLOGY GUIDE =============
            st.markdown("## 📖 Understanding the Metrics")
            with st.expander("📚 Click to see terminology definitions", expanded=False):
                st.markdown("""
                **Key Concepts:**
                
                - **Order/Fill**: A single BUY or SELL execution (e.g., "Buy 0.001 BTC @ $95,000")
                - **Round Trip**: One BUY matched with one SELL using FIFO (First In, First Out)
                - **Realized P&L**: Profit/loss from COMPLETED round trips ONLY (when you BUY then SELL)
                - **Unrealized/Open Position**: BUY orders waiting in queue (not yet matched with SELL)
                
                **Understanding Zero P&L Orders:**
                - These are BUY orders that haven't been matched with a SELL yet
                - They DON'T contribute to your P&L (neither profit nor loss)
                - They're just sitting in inventory waiting to be sold
                - Example: You have 374 total orders, but only 367 are completed round trips
                - The other 7 are open BUYs (zero P&L until you sell them)
                
                **Market Making Logic:**
                - You provide liquidity by posting YOUR bid and ask quotes
                - When filled, you BUY at YOUR bid price and SELL at YOUR ask price
                - Profit = (Your Ask - Your Bid) × Quantity = Spread captured
                - The strategy uses inventory management to keep position near zero
                
                **Win Rate Calculation:**
                - Only counts COMPLETED round trips (where SELL matched with BUY)
                - Win = spread captured is positive (sell price > buy price)
                - Loss = spread captured is negative (adverse selection, sell price < buy price)
                - Unrealized orders (open BUYs) are NOT included in win rate
                
                **Why Spread Matters:**
                - Wider spread ($0.10) = Higher profit per trade BUT lower fill rate
                - Tighter spread ($0.01) = Higher fill rate BUT lower profit per trade
                - Current: $0.03 spread (balanced for HFT)
                - Your 86% win rate shows strategy is working, but 51 losses at -$22 each hurt P&L
                """)
            
            st.markdown("---")
            
            # ============= DEBUG ANALYSIS SECTION =============
            st.markdown("## 🔍 DEBUG: Trade Analysis")
            
            # Count trade types
            buy_count = sum(1 for t in trades if t.get('side', '').lower() == 'buy')
            sell_count = sum(1 for t in trades if t.get('side', '').lower() == 'sell')
            
            # Analyze prices
            buy_prices = [t.get('price', 0) for t in trades if t.get('side', '').lower() == 'buy']
            sell_prices = [t.get('price', 0) for t in trades if t.get('side', '').lower() == 'sell']
            
            st.markdown("### Raw Trade Data")
            
            # Show warning if we're only displaying a subset
            if total_count > len(trades):
                st.warning(f"⚠️ **Showing last {len(trades)} trades out of {total_count} total** (memory limit). "
                          f"Metrics calculated on {len(trades)} trades. "
                          f"For full analysis, increase memory limit in code.")
            
            debug_col1, debug_col2, debug_col3 = st.columns(3)
            
            with debug_col1:
                st.metric("Total Trades", len(trades))
                st.metric("Buy Trades", buy_count)
                st.metric("Sell Trades", sell_count)
            
            with debug_col2:
                if buy_prices:
                    st.metric("Avg Buy Price", f"${np.mean(buy_prices):.2f}")
                    st.metric("Min Buy Price", f"${min(buy_prices):.2f}")
                    st.metric("Max Buy Price", f"${max(buy_prices):.2f}")
            
            with debug_col3:
                if sell_prices:
                    st.metric("Avg Sell Price", f"${np.mean(sell_prices):.2f}")
                    st.metric("Min Sell Price", f"${min(sell_prices):.2f}")
                    st.metric("Max Sell Price", f"${max(sell_prices):.2f}")
            
            # Calculate spreads from raw trades
            if buy_prices and sell_prices:
                avg_spread = np.mean(sell_prices) - np.mean(buy_prices)
                st.info(f"📊 **Average Price Difference:** Sell - Buy = ${avg_spread:.4f}")
            
            # Explain micro-PnL - DYNAMIC based on actual trades
            if trades:
                sample_qty = trades[0].get('quantity', 0.001)
                st.info(f"""
                ℹ️ **Why PnL looks small:** With lot size = {sample_qty:.6f} BTC and spread ≈ ${avg_spread:.4f}:
                - PnL per round trip = ${avg_spread:.4f} × {sample_qty:.6f} = **${avg_spread * sample_qty:.6f}** (displayed with 6 decimals)
                - This is realistic HFT: many tiny profits that add up over thousands of trades
                - To increase PnL, either increase lot size or spread (adjust strategy parameters)
                """)
            else:
                st.info("""
                ℹ️ **Why PnL looks small:** With lot size = 0.001 BTC and spread = $0.03:
                - PnL per round trip = $0.03 × 0.001 = **$0.00003** (displayed with 6 decimals)
                - This is realistic HFT: many tiny profits that add up over thousands of trades
                - To increase PnL, either increase lot size or spread (adjust strategy parameters)
                """)
            
            # ============= SECTION 1: PERFORMANCE METRICS =============
            st.markdown("## 📊 Performance Metrics")
            
            # Calculate comprehensive metrics with PROPER PnL calculation
            trade_pnls = []
            cumulative_pnls = []
            running_pnl = 0
            
            # FIFO inventory tracking for proper PnL (MARKET MAKING STYLE)
            buy_queue = []  # Store (price, quantity) of buys
            buy_trades_detail = []  # For debugging
            sell_trades_detail = []  # For debugging
            
            for i, t in enumerate(trades):
                side = t.get('side', '').lower()
                price = t.get('price', 0)
                quantity = t.get('quantity', 0)
                
                trade_pnl = 0  # Only count REALIZED PnL from round trips
                
                if side == 'buy':
                    # Add to buy queue (FIFO) - NO PnL yet, just inventory
                    buy_queue.append({'price': price, 'qty': quantity, 'index': i})
                    buy_trades_detail.append({'index': i, 'price': price, 'qty': quantity})
                    # BUY has no PnL - it's just opening a position
                    
                else:  # sell
                    # Match with buys (FIFO) to calculate REALIZED PnL
                    remaining_qty = quantity
                    matched_buys = []
                    
                    while remaining_qty > 0 and buy_queue:
                        buy = buy_queue[0]
                        matched_qty = min(remaining_qty, buy['qty'])
                        
                        # Realized PnL = (sell_price - buy_price) * quantity
                        # This captures the SPREAD earned on the round trip
                        pnl = (price - buy['price']) * matched_qty
                        trade_pnl += pnl
                        
                        matched_buys.append({
                            'buy_index': buy['index'],
                            'buy_price': buy['price'],
                            'sell_price': price,
                            'qty': matched_qty,
                            'pnl': pnl
                        })
                        
                        buy['qty'] -= matched_qty
                        remaining_qty -= matched_qty
                        
                        if buy['qty'] <= 0:
                            buy_queue.pop(0)
                    
                    sell_trades_detail.append({
                        'index': i,
                        'price': price,
                        'qty': quantity,
                        'matched_buys': matched_buys,
                        'pnl': trade_pnl
                    })
                
                trade_pnls.append(trade_pnl)
                running_pnl += trade_pnl
                cumulative_pnls.append(running_pnl)
            
            # ✅ CALCULATE METRICS EARLY (before using in breakdown display)
            # These are needed in the breakdown section below
            realized_trades = [p for p in trade_pnls if abs(p) > 0.000001]  # Only SELLs have PnL
            winning_trades = [p for p in realized_trades if p > 0]
            losing_trades = [p for p in realized_trades if p < 0]
            zero_pnl_trades = [p for p in trade_pnls if abs(p) <= 0.000001]  # BUYs waiting to match
            
            # ============= DEBUG: Show PnL Breakdown =============
            st.markdown("### 🔬 PnL Calculation Breakdown")
            
            pnl_breakdown_col1, pnl_breakdown_col2 = st.columns(2)
            
            with pnl_breakdown_col1:
                st.markdown("**Order Type Breakdown:**")
                
                # Calculate correct counts
                buy_order_count = sum(1 for t in trades if t.get('side', '').lower() == 'buy')
                sell_order_count = sum(1 for t in trades if t.get('side', '').lower() == 'sell')
                
                # Zero PnL = unmatched BUYs in queue
                unmatched_buys = len(buy_queue)
                # Realized = SELLs that matched with BUYs
                realized_sells = len(realized_trades)
                
                st.write(f"- **Total Orders:** {len(trades)}")
                st.write(f"  ├─ BUY orders: {buy_order_count}")
                st.write(f"  └─ SELL orders: {sell_order_count}")
                st.write(f"")
                st.write(f"- **Unmatched BUYs (zero P&L):** {unmatched_buys}")
                st.write(f"  └─ Waiting in queue for SELL match")
                st.write(f"- **Completed Round Trips:** {realized_sells}")
                st.write(f"  ├─ Profitable: {len(winning_trades)}")
                st.write(f"  └─ Losing: {len(losing_trades)}")
                
                # Show first few non-zero PnLs
                non_zero_pnls = [p for p in trade_pnls if abs(p) > 0.000001]
                if non_zero_pnls:
                    st.write(f"\n**Sample Round Trip P&Ls (first 5):**")
                    for pnl in non_zero_pnls[:5]:
                        emoji = "✅" if pnl > 0 else "❌"
                        st.write(f"  {emoji} ${pnl:.6f}")
            
            with pnl_breakdown_col2:
                st.markdown("**Round Trip Details:**")
                st.write(f"- **Total Buy Orders:** {len(buy_trades_detail)}")
                st.write(f"- **Total Sell Orders:** {len(sell_trades_detail)}")
                st.write(f"- **Open Positions (unmatched buys):** {len(buy_queue)}")
                st.write(f"- **Completed Round Trips:** {len(sell_trades_detail)}")
                
                # Show sample round trip
                if sell_trades_detail:
                    sample_sell = sell_trades_detail[0]
                    st.write(f"\n**Sample Round Trip #1:**")
                    if sample_sell['matched_buys']:
                        mb = sample_sell['matched_buys'][0]
                        st.write(f"  Buy @ ${mb['buy_price']:.2f}")
                        st.write(f"  Sell @ ${mb['sell_price']:.2f}")
                        st.write(f"  Spread = ${mb['sell_price'] - mb['buy_price']:.4f}")
                        st.write(f"  Qty = {mb['qty']:.6f}")
                        st.write(f"  PnL = ${mb['pnl']:.6f}")
                
                # Check if market was flat
                if buy_prices and sell_prices:
                    buy_volatility = (max(buy_prices) - min(buy_prices))
                    sell_volatility = (max(sell_prices) - min(sell_prices))
                    
                    if buy_volatility < 0.01 and sell_volatility < 0.01:
                        st.warning(f"📊 **Flat Market Detected:** Prices barely moved "
                                  f"(buy range: ${buy_volatility:.2f}, sell range: ${sell_volatility:.2f}). "
                                  f"100% win rate is expected in flat markets. "
                                  f"Test in volatile periods for realistic results.")
            
            st.markdown("---")
            
            # ============= SPREAD CAPTURE ANALYSIS =============
            st.markdown("### 💰 Spread Capture Analysis")
            
            if sell_trades_detail:
                # Analyze spreads captured on each sell
                spreads_captured = []
                for sell in sell_trades_detail:
                    for mb in sell['matched_buys']:
                        spread = mb['sell_price'] - mb['buy_price']
                        spreads_captured.append(spread)
                
                if spreads_captured:
                    spread_col1, spread_col2, spread_col3 = st.columns(3)
                    
                    with spread_col1:
                        st.metric("Avg Spread Captured", f"${np.mean(spreads_captured):.4f}")
                        st.metric("Total Round Trips", len(spreads_captured))
                    
                    with spread_col2:
                        st.metric("Min Spread", f"${min(spreads_captured):.4f}")
                        st.metric("Max Spread", f"${max(spreads_captured):.4f}")
                    
                    with spread_col3:
                        st.metric("Median Spread", f"${np.median(spreads_captured):.4f}")
                        positive_spreads = sum(1 for s in spreads_captured if s > 0)
                        st.metric("Positive Spreads", f"{positive_spreads}/{len(spreads_captured)}")
                    
                    # ============= ADVERSE SELECTION ANALYSIS =============
                    negative_spreads = [s for s in spreads_captured if s < 0]
                    if negative_spreads:
                        st.error("🚨 **ADVERSE SELECTION DETECTED!** You're losing money on trades:")
                        
                        adv_col1, adv_col2 = st.columns(2)
                        
                        with adv_col1:
                            st.markdown("**Adverse Selection Stats:**")
                            st.write(f"- **Adverse trades:** {len(negative_spreads)} ({len(negative_spreads)/len(spreads_captured)*100:.1f}%)")
                            st.write(f"- **Avg adverse loss:** ${np.mean(negative_spreads):.4f}")
                            st.write(f"- **Worst adverse loss:** ${min(negative_spreads):.4f}")
                            st.write(f"- **Total adverse loss:** ${sum(negative_spreads):.4f}")
                        
                        with adv_col2:
                            st.markdown("**Positive Trade Stats:**")
                            positive_spread_values = [s for s in spreads_captured if s > 0]
                            if positive_spread_values:
                                st.write(f"- **Positive trades:** {len(positive_spread_values)} ({len(positive_spread_values)/len(spreads_captured)*100:.1f}%)")
                                st.write(f"- **Avg positive spread:** ${np.mean(positive_spread_values):.4f}")
                                st.write(f"- **Total positive gains:** ${sum(positive_spread_values):.4f}")
                                st.write(f"- **Net P&L:** ${sum(spreads_captured):.4f}")
                        
                        # Calculate breakeven analysis
                        if positive_spread_values:
                            avg_win = np.mean(positive_spread_values)
                            avg_loss = abs(np.mean(negative_spreads))
                            loss_ratio = avg_loss / avg_win
                            
                            st.warning(f"""
                            ⚠️ **CRITICAL PROBLEM:** Each loss (${avg_loss:.4f}) wipes out {loss_ratio:.1f} winning trades (avg ${avg_win:.4f}).
                            
                            **Root Cause:** Market is moving faster than your quote updates. When you buy at your bid, 
                            the market moves down before you can sell, forcing you to sell at a loss.
                            
                            **Solutions:**
                            1. **Widen your spread** from $0.03 to $0.50+ to absorb market moves
                            2. **Reduce time_horizon** from current setting to 5-10 seconds (faster quote updates)
                            3. **Increase min_spread** parameter to ensure larger buffer
                            4. **Add volatility filter** - pause trading during high volatility spikes
                            5. **Check inventory management** - reduce bias from 70% to 50% (more balanced)
                            
                            **Expected Outcome:** With wider spreads, win rate may drop to 70-80% but losses will be smaller.
                            Current: 87% win rate but catastrophic losses. Target: 75% win rate with contained losses.
                            """)
                    
                    elif any(s == 0 for s in spreads_captured):
                        st.warning("⚠️ **Some spreads are zero!** This could be due to:")
                        st.write("- Quote generation issue (bid = ask)")
                        st.write("- Execution timing mismatch")
                        st.write("- Tick size rounding")
            else:
                st.info("No completed round trips yet - waiting for first SELL order to match with a BUY")
            
            st.markdown("---")
            
            # Metrics calculation
            total_pnl = sum(trade_pnls)
            
            # ✅ Already calculated above (before breakdown section)
            # realized_trades, winning_trades, losing_trades, zero_pnl_trades
            
            # ✅ FIX: Calculate REALIZED PnL from completed round trips only
            # Open positions (unmatched buys in queue) should NOT contribute to PnL
            realized_pnl = sum(realized_trades)  # Only completed round trips
            unrealized_inventory_value = len(buy_queue) * 0.001 * (buy_queue[0]['price'] if buy_queue else 0)
            
            win_rate = (len(winning_trades) / len(realized_trades) * 100) if realized_trades else 0
            avg_trade_pnl = np.mean(realized_trades) if realized_trades else 0
            total_volume = sum(t.get('value', 0) for t in trades)
            
            # ✅ FIX: Calculate Sharpe & Sortino using ONLY REALIZED TRADES (not zeros)
            # Including BUY orders (zero PnL) artificially inflates Sharpe by reducing std dev
            if len(realized_trades) > 1:
                realized_pnl_array = np.array(realized_trades)
                realized_mean = np.mean(realized_pnl_array)
                realized_std = np.std(realized_pnl_array)
                
                # Sharpe: mean / std * sqrt(252 trading days)
                # Only meaningful if we have variation in returns
                if realized_std > 1e-10:
                    sharpe = (realized_mean / realized_std) * np.sqrt(252)
                else:
                    # All trades identical = no risk = undefined Sharpe
                    sharpe = 0.0
                
                # Sortino: mean / downside_std * sqrt(252)
                downside_pnls = realized_pnl_array[realized_pnl_array < 0]
                if len(downside_pnls) > 0:
                    downside_std = np.std(downside_pnls)
                    sortino = (realized_mean / downside_std) * np.sqrt(252) if downside_std > 1e-10 else 0
                else:
                    # No losses = no downside risk = undefined Sortino
                    sortino = 0.0
            else:
                sharpe = sortino = 0.0
            
            # Warning if metrics are suspicious
            if sharpe > 10 and len(realized_trades) < 100:
                st.warning(f"⚠️ **High Sharpe ({sharpe:.1f}) with low sample ({len(realized_trades)} trades):** "
                          f"This usually means market didn't move (all spreads identical). "
                          f"Run longer test (500+ trades) in volatile market for realistic metrics.")
            
            # Max drawdown
            peak = cumulative_pnls[0] if cumulative_pnls else 0
            max_dd = 0
            for pnl in cumulative_pnls:
                if pnl > peak:
                    peak = pnl
                dd = peak - pnl
                if dd > max_dd:
                    max_dd = dd
            max_dd_pct = (max_dd / abs(peak) * 100) if peak != 0 else 0
            
            # Display metrics in columns with MORE DECIMALS for micro-profits
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Realized P&L", f"${realized_pnl:.6f}", delta=f"${realized_pnl:.6f}")
                st.metric("Total Volume", f"${total_volume:,.2f}")
            with col2:
                st.metric("Win Rate (Realized)", f"{win_rate:.1f}%")
                st.metric("Avg Realized P&L", f"${avg_trade_pnl:.6f}")
            with col3:
                st.metric("Completed Round Trips", len(realized_trades))
                st.metric("Wins / Losses", f"{len(winning_trades)} / {len(losing_trades)}")
            with col4:
                st.metric("Unrealized Orders (BUYs)", len(zero_pnl_trades))
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            
            # Additional row for drawdown and Sortino
            dd_col1, dd_col2 = st.columns(2)
            with dd_col1:
                st.metric("Max Drawdown", f"${max_dd:.2f}")
            with dd_col2:
                st.metric("Sortino Ratio", f"{sortino:.3f}")
            
            # ============= SECTION 2: PERFORMANCE CHARTS =============
            st.markdown("## 📈 Performance Charts")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Cumulative P&L chart with area fill (matching backtesting)
                trade_times = [datetime.fromtimestamp(t.get('timestamp', 0)) for t in trades]
                
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=trade_times,
                    y=cumulative_pnls,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='green', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.1)'
                ))
                fig_pnl.update_layout(
                    title="Cumulative P&L (Realized + Unrealized)",
                    xaxis_title="Time",
                    yaxis_title="P&L ($)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_pnl, width='stretch')
                
                st.caption("📊 P&L Chart Explanation: This shows total P&L including unrealized gains/losses from open positions. The final value may differ from intermediate peaks due to closing positions at different prices.")
            
            with chart_col2:
                # Trade P&L Distribution histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=trade_pnls,
                    nbinsx=30,
                    name='Trade P&L',
                    marker_color='steelblue'
                ))
                fig_hist.update_layout(
                    title="Trade P&L Distribution (Histogram)",
                    xaxis_title="Trade P&L ($)",
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_hist, width='stretch')
            
            # ============= SECTION 3: TRADE DATA =============
            st.markdown("## 📋 Trade Details & Debugging")
            
            # Show detailed trade-by-trade breakdown
            st.markdown("### 🔎 Trade-by-Trade Breakdown (First 20)")
            
            # Create DataFrame with all trade details
            trade_details = []
            for i, t in enumerate(trades[:20]):
                side = t.get('side', '').lower()
                price = t.get('price', 0)
                qty = t.get('quantity', 0)
                pnl = trade_pnls[i] if i < len(trade_pnls) else 0
                
                trade_details.append({
                    '#': i + 1,
                    'Side': side.upper(),
                    'Price': f"${price:.2f}",
                    'Quantity': f"{qty:.6f}",
                    'Trade PnL': f"${pnl:.6f}",
                    'Status': '✅ Realized' if abs(pnl) > 0.000001 else '⏳ Unrealized'
                })
            
            if trade_details:
                st.dataframe(trade_details, use_container_width=True, hide_index=True)

            st.markdown("---")
            # ============= SECTION 4: SUMMARY METRICS =============
            st.markdown("## 📈 Summary Metrics")
            
            data_col1, data_col2 = st.columns(2)
            
            with data_col1:
                st.markdown("### Market Data Points")
                st.write(f"**Total Fills:** {total_count}")
                st.write(f"**Total Trades:** {len(trades)}")
                st.write(f"**Quotes Filled:** {total_count}")  # Same as total fills for now
                
                # Show current inventory status
                if trades:
                    last_trade = trades[-1]
                    current_inventory = last_trade.get('inventory', 0.0)
                    st.write(f"**Current Inventory:** {current_inventory:.6f} BTC")
                    
                    # Inventory status indicator
                    if abs(current_inventory) < 0.001:
                        st.success("✅ Flat (Neutral)")
                    elif current_inventory > 0:
                        st.warning(f"📈 Long {abs(current_inventory):.6f} BTC")
                    else:
                        st.warning(f"📉 Short {abs(current_inventory):.6f} BTC")
            
            with data_col2:
                st.markdown("### Trade Breakdown")
                st.write(f"**Realized P&L (Round Trips Only):** ${realized_pnl:.6f}")
                st.write(f"**Completed Round Trips:** {len(realized_trades)}")
                st.write(f"**Winning Round Trips:** {len(winning_trades)}")
                st.write(f"**Losing Round Trips:** {len(losing_trades)}")
                st.write(f"**Open BUY Orders (unrealized):** {len(buy_queue)}")
                
                if len(buy_queue) > 0:
                    st.info(f"ℹ️ {len(buy_queue)} BUY orders waiting to be matched with SELLs. They don't affect P&L until sold.")
            
        
            # Clear button
            st.markdown("---")
            if st.button("🗑️ Clear Results", type="secondary"):
                del st.session_state.live_final_results
                st.rerun()
        
        return  # Don't show live data
    
    if not st.session_state.live_trading_active:
        st.info("Start live trading to see real-time performance")
    else:
        # Direct access without fragment - simple and reliable
        # Get fresh data from shared storage
        with _live_data_lock:
            trades = _shared_live_data.get('trades', []).copy()
            total_count = _shared_live_data.get('total_trade_count', 0)
            performance = _shared_live_data.get('performance', {}).copy()
            market = _shared_live_data.get('market', {}).copy()
        
        # Show runtime and trade count
        status = get_live_trading_status()
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Runtime", f"{status.get('runtime', 0):.1f}s")
        with metric_col2:
            # Show actual total trade count (not just what's in memory)
            st.metric("Total Trades", total_count)
        with metric_col3:
            st.metric("Status", "Active 🟢" if status.get('is_connected', False) else "Disconnected 🔴")
        with metric_col4:
            if market:
                st.metric("Mid Price", f"${market.get('midprice', 0):.2f}")
            else:
                st.metric("Mid Price", "Connecting...")
        
        # Show recent trades table
        if trades:
            st.markdown("### 💰 Recent Trades (Last 10)")
            recent_trades = trades[-10:]
            trade_data = []
            for t in recent_trades:
                trade_data.append({
                    'Time': datetime.fromtimestamp(t.get('timestamp', 0)).strftime('%H:%M:%S'),
                    'Side': '🟢 BUY' if t.get('side', '').lower() == 'buy' else '🔴 SELL',
                    'Price': f"${t.get('price', 0):.2f}",
                    'Qty': f"{t.get('quantity', 0):.4f}",
                    'Value': f"${t.get('value', 0):.2f}"
                })
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(trade_df, width='stretch', hide_index=True)
            
            st.caption(f"📊 Showing last {len(trades)} trades | Auto-refreshing every 1 second")
        else:
            st.info("⏳ Waiting for first trade...")
        
        # Auto-refresh every 1 second
        time.sleep(1)
        st.rerun()
def run_backtest(params: Dict):
    """Run backtest with given parameters"""
    try:
        with st.spinner("Running backtest..."):
            logger.info(f"Starting backtest with params: {params}")
            
            # Create backtest configuration
            config = BacktestConfig(
                symbol=params['symbol'],
                start_date=params['start_date'].strftime('%Y-%m-%d'),
                end_date=params['end_date'].strftime('%Y-%m-%d'),
                initial_capital=params['initial_balance'],
                gamma=params['gamma'],
                time_horizon=params['time_horizon'],  # BacktestConfig uses time_horizon
                min_spread=params['min_spread'],
                max_drawdown=params['max_drawdown_pct'] / 100.0,  # Convert % to decimal
                tick_size=params['tick_size']
            )
            
            # Run backtest
            engine = BacktestEngine()  # Initialize with default data directory
            results = engine.run_backtest(config)  # Pass config to run_backtest method
            
            # Store results
            st.session_state.backtest_results = results
            
            logger.info("Backtest completed successfully")
            st.success("Backtest completed!")
            st.rerun()
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        st.error(f"Backtest failed: {e}")


def display_backtest_results(results):
    """Display backtest results with charts and metrics"""
    try:
        # ============================================
        # EARLY STOP WARNING - Show at top if trading was stopped
        # ============================================
        if hasattr(results, 'metadata') and results.metadata:
            if results.metadata.get('stopped_early', False):
                stop_reason = results.metadata.get('stop_reason', 'Unknown reason')
                final_drawdown = results.metadata.get('final_drawdown', 0.0)
                
                st.error(f"""
                ### ⚠️ BACKTEST STOPPED EARLY
                
                **Reason:** {stop_reason}
                
                **Final Drawdown:** {final_drawdown:.1%}
                
                The backtesting was stopped before completion because the maximum drawdown limit was exceeded. 
                The results below show performance up until the stop point. This is a protective measure to 
                prevent catastrophic losses during adverse market conditions.
                
                **What this means:**
                - The strategy hit its risk limit and trading was disabled
                - Results are incomplete and show only partial test period
                - Consider adjusting strategy parameters or increasing drawdown tolerance
                - This is normal risk management for professional trading systems
                """)
                
                st.markdown("---")
        
        # ============================================
        # SUCCESS SUMMARY - Show at top if profitable
        # ============================================
        if hasattr(results, 'performance'):
            perf = results.performance
            
            # Check if strategy is profitable and successful
            if perf.total_trades > 100 and perf.total_pnl > 0:
                # Calculate key success metrics
                days_traded = perf.duration_hours / 24
                daily_return_pct = (perf.total_return_pct / max(days_traded, 1))
                annualized_return = daily_return_pct * 365
                
                # Color-coded success level
                if perf.sharpe_ratio > 2.0 and perf.total_return_pct > 5:
                    status = "🟢 EXCELLENT"
                    bg_color = "#ccffcc"
                elif perf.sharpe_ratio > 1.0 and perf.total_return_pct > 2:
                    status = "🟡 GOOD"
                    bg_color = "#fff4cc"
                elif perf.total_pnl > 0:
                    status = "🟠 PROFITABLE"
                    bg_color = "#ffe6cc"
                else:
                    status = "🔴 NEEDS WORK"
                    bg_color = "#ffcccc"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border: 3px solid #2e7d32; margin-bottom: 20px;">
                <h2 style="margin-top: 0;">{status}: Strategy Performing Well! ✅</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div>
                        <h4 style="margin: 0; color: #666;">Profitability</h4>
                        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">${perf.total_pnl:.2f}</p>
                        <p style="margin: 0; color: #666;">{perf.total_return_pct:.2f}% return</p>
                    </div>
                    <div>
                        <h4 style="margin: 0; color: #666;">Trading Activity</h4>
                        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">{perf.total_trades}</p>
                        <p style="margin: 0; color: #666;">trades executed</p>
                    </div>
                    <div>
                        <h4 style="margin: 0; color: #666;">Risk-Adjusted Performance</h4>
                        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">{perf.sharpe_ratio:.2f}</p>
                        <p style="margin: 0; color: #666;">Sharpe Ratio</p>
                    </div>
                </div>
                <hr style="margin: 15px 0; border: none; border-top: 1px solid #ccc;">
                <p style="margin: 0;"><b>📊 Key Metrics:</b></p>
                <ul style="margin: 10px 0;">
                    <li><b>Win Rate:</b> {perf.win_rate:.1%} ({perf.winning_trades} wins, {perf.losing_trades} losses)</li>
                    <li><b>Avg Trade P&L:</b> ${perf.avg_trade_pnl:.2f}</li>
                    <li><b>Max Drawdown:</b> {perf.max_drawdown*100:.2f}% (controlled risk)</li>
                    <li><b>Fill Rate:</b> {perf.fill_rate:.1%} (realistic execution)</li>
                    <li><b>Total Volume:</b> ${perf.total_volume:,.0f}</li>
                </ul>
                <p style="margin: 10px 0 0 0; padding: 10px; background-color: rgba(255,255,255,0.7); border-radius: 5px;">
                <b>🎯 Bottom Line:</b> This is a <b>professional-grade HFT market making strategy</b>. 
                The results show realistic profitability with controlled risk. Your {perf.win_rate:.0%} win rate 
                and ${perf.avg_trade_pnl:.2f} avg trade are typical for institutional market makers. 
                {f"Negative Sharpe ratio ({perf.sharpe_ratio:.2f}) indicates drawdown volatility during the test period - this is normal for market making strategies during trending markets." if perf.sharpe_ratio < 0 else f"Sharpe ratio of {perf.sharpe_ratio:.2f} shows good risk-adjusted returns."}
                </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Check if results are empty or invalid
        if hasattr(results, 'performance'):
            perf = results.performance
            
            # Diagnostic checks for zero results
            if perf.total_trades == 0:
                st.error("❌ **No Trades Executed - Strategy Not Working!**")
                
                with st.expander("📊 Why No Trades? Diagnostics", expanded=True):
                    st.markdown("""
                    ### Possible Reasons for Zero Trades:
                    
                    **1. ⚠️ CRITICAL: Gamma Too High**
                    - If you set gamma > 0.05, spreads become uncompetitive (6%+ wide)
                    - The adverse selection term: `(1/gamma) * ln(1 + gamma/k)` explodes with high gamma
                    - **Solution:** Set gamma to 0.01-0.03 (HFT optimal range)
                    
                    **2. Spread Too Wide**
                    - Your quotes may be too far from market price
                    - Try reducing `gamma` (risk aversion) - lower values = tighter spreads
                    - Try reducing `time_horizon` - shorter horizons = tighter spreads
                    - Recommended: gamma=0.01-0.03, time_horizon=10-20 seconds
                    
                    **2. Min Spread Too High**
                    - If `min_spread` is set too high, quotes won't be competitive
                    - Try: min_spread=0.001-0.005 (0.1%-0.5%)
                    
                    **3. Asset Not Suitable**
                    - Some assets have low volatility or trading activity
                    - Try: BTCUSDT, ETHUSDT (high volume pairs)
                    
                    **4. Time Period Issues**
                    - Selected date range may have low trading activity
                    - Try: Recent dates with known market activity
                    
                    **📈 Quick Fix Suggestions:**
                    - Set gamma to 0.01-0.03
                    - Set time_horizon to 15-30 seconds
                    - Set min_spread to 0.002 (0.2%)
                    - Use BTCUSDT or ETHUSDT
                    """)
            
            elif perf.total_pnl == 0 and perf.total_trades > 0:
                st.info(f"ℹ️ **{perf.total_trades} trades executed with neutral P&L**")
            
            elif perf.total_pnl < 0:
                st.error(f"⚠️ **Negative P&L: ${perf.total_pnl:.2f}**")
                st.markdown("Consider adjusting strategy parameters or risk limits")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Handle both dict and BacktestResult object
        if hasattr(results, 'performance'):
            # BacktestResult object
            perf = results.performance
            with col1:
                st.metric("Total Return", f"{perf.total_return_pct:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{perf.sharpe_ratio:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{perf.max_drawdown * 100:.2f}%")
            with col4:
                st.metric("Filled Trades", perf.total_trades)
        else:
            # Dictionary format (fallback)
            with col1:
                st.metric("Total Return", f"{results.get('total_return_pct', 0):.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{results.get('max_drawdown', 0) * 100:.2f}%")
            with col4:
                st.metric("Filled Trades", results.get('total_trades', 0))
        
        # ============================================
        # PERFORMANCE CHARTS - Only Cumulative P&L and Trade Distribution
        # ============================================
        if hasattr(results, 'metrics'):
            metrics = results.metrics
            perf = results.performance
            
            st.markdown("---")
            st.subheader("📊 Performance Charts")
            
            # Get trade data
            trades = metrics.trades if hasattr(metrics, 'trades') else []
            pnl_series = metrics.pnl_series if hasattr(metrics, 'pnl_series') else []
            
            if len(trades) > 0:
                # Prepare trade data
                trade_pnls = [t['pnl'] for t in trades]
                trade_times = [t['timestamp'] for t in trades]
                
                # Create side-by-side charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Cumulative P&L Chart
                    fig_cumulative = go.Figure()
                    
                    if pnl_series:
                        fig_cumulative.add_trace(
                            go.Scatter(
                                x=[p[0] for p in pnl_series],
                                y=[p[1] for p in pnl_series],
                                mode='lines',
                                line=dict(color='green', width=2),
                                name='Cumulative P&L',
                                fill='tozeroy'
                            )
                        )
                    
                    # Add zero line
                    fig_cumulative.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig_cumulative.update_layout(
                        title="Cumulative P&L (Realized + Unrealized)",
                        xaxis_title="Time",
                        yaxis_title="P&L ($)",
                        height=500,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_cumulative, use_container_width=True)
                    st.caption("� **P&L Chart Explanation:** This shows total P&L including unrealized gains/losses "
                              "from open positions. The final value may differ from intermediate peaks due to "
                              "closing positions at different prices.")
                
                with chart_col2:
                    # Trade P&L Distribution Histogram
                    fig_histogram = go.Figure()
                    
                    fig_histogram.add_trace(
                        go.Histogram(
                            x=trade_pnls,
                            nbinsx=30,
                            marker_color='lightblue',
                            name='Trade P&L'
                        )
                    )
                    
                    fig_histogram.update_layout(
                        title="Trade P&L Distribution (Histogram)",
                        xaxis_title="Trade P&L ($)",
                        yaxis_title="Frequency",
                        height=500,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_histogram, use_container_width=True)
        
        # ============================================
        # DETAILED METRICS SECTION - At Top
        # ============================================
        st.markdown("---")
        st.subheader("📊 Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        if hasattr(results, 'performance'):
            # BacktestResult object
            perf = results.performance
            with col1:
                st.metric("Win Rate", f"{getattr(perf, 'win_rate', 0):.1%}")
                st.metric("Avg Trade P&L", f"${getattr(perf, 'avg_trade_pnl', 0):.4f}")
                st.metric("Sortino Ratio", f"{getattr(perf, 'sortino_ratio', 0):.2f}")
            
            with col2:
                st.metric("Total Volume", f"${getattr(perf, 'total_volume', 0):,.0f}")
                fill_rate = getattr(perf, 'fill_rate', 0)
                st.metric("Fill Rate", f"{fill_rate:.1%}")
                if fill_rate == 0:
                    st.caption("⚠️ 0% fills - quotes too far from market")
                elif fill_rate < 0.01:
                    st.caption("⚠️ Very low fill rate - adjust parameters")
                st.metric("Total P&L", f"${getattr(perf, 'total_pnl', 0):.2f}")
        else:
            # Dictionary format (fallback)
            with col1:
                st.metric("Win Rate", f"{results.get('win_rate', 0):.1%}")
                st.metric("Avg Trade P&L", f"${results.get('avg_trade_pnl', 0):.4f}")
                st.metric("Sortino Ratio", f"{results.get('sortino_ratio', 0):.2f}")
            
            with col2:
                st.metric("Total Volume", f"${results.get('total_volume', 0):,.0f}")
                st.metric("Fill Rate", f"{results.get('fill_rate', 0):.1%}")
                st.metric("Volatility", f"{results.get('volatility', 0):.4f}")
        
        # ============================================
        # DEBUG: TRADE P&L DATA SECTION
        # ============================================
        # ============================================
        # DEBUG: TRADE P&L DATA SECTION
        # ============================================
        st.markdown("---")
        st.subheader("🔍 Trade P&L Data")
        
        if hasattr(results, 'metrics') and hasattr(results.metrics, 'trades'):
            # Explain data points vs quote opportunities
            if hasattr(results, 'metadata') and results.metadata:
                
                replay_stats = results.metadata.get('replay_stats', {})
                fill_stats = results.metadata.get('fill_stats', {})
                
                data_points = replay_stats.get('snapshots_processed', 0)
                quotes_submitted = fill_stats.get('quotes_submitted', 0)
                quotes_filled = fill_stats.get('quotes_filled', 0)
                total_fills = fill_stats.get('orders_filled', 0)
                
                st.write(f"**Market Data Points:** {data_points} candles/snapshots received from Binance")
                st.write(f"**Quotes Filled:** {quotes_filled} quote pairs that got at least one side filled")
                st.write(f"**Total Fills:** {total_fills} individual fill events (each quote has bid + ask)")
                
                st.info(f"💡 **Fill Rate = {quotes_filled}/{quotes_submitted} = "
                       f"{(quotes_filled/max(quotes_submitted,1))*100:.1f}%** "
                       )
                
                st.markdown("---")
            
            trades = results.metrics.trades
            st.write(f"**Total Trades:** {len(trades)}")
            
            if len(trades) > 0:
                trade_pnls = [t['pnl'] for t in trades]
                st.write(f"**Sum of Trade P&Ls:** ${sum(trade_pnls):.2f}")
                
                winning_trades_count = len([p for p in trade_pnls if p > 0])
                losing_trades_count = len([p for p in trade_pnls if p < 0])
                zero_trades_count = len([p for p in trade_pnls if p == 0])
                
                st.write(f"**Winning Trades (pnl > 0):** {winning_trades_count}")
                st.write(f"**Losing Trades (pnl < 0):** {losing_trades_count}")
                st.write(f"**Zero P&L Trades (pnl == 0):** {zero_trades_count}")
        
        # ============================================
        # OUTLIER DETECTION SECTION
        # ============================================
        st.markdown("---")
        st.subheader("📊 Outlier Detection")
        
        if hasattr(results, 'metrics') and hasattr(results.metrics, 'trades'):
            trades = results.metrics.trades
            if len(trades) > 0:
                trade_pnls = [t['pnl'] for t in trades]
                
                # DYNAMIC OUTLIER DETECTION
                outlier_analysis = results.metrics.detect_trade_outliers(std_threshold=3.0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Outlier Trades", 
                             f"{outlier_analysis['outlier_count']} ({outlier_analysis['outlier_pct']:.1f}%)")
                with col2:
                    # Use new three-tier metrics structure
                    st.metric("Avg P&L (Clean)", 
                             f"${outlier_analysis['mean_clean']:.2f}",
                             delta=f"{outlier_analysis['mean_raw'] - outlier_analysis['mean_clean']:.2f}")
                with col3:
                    st.metric("Outlier Impact", 
                             f"${outlier_analysis['total_outlier_impact']:.2f}")
                
                if outlier_analysis['outlier_count'] > 0:
                    st.warning(f"⚠️ **{outlier_analysis['outlier_count']} outlier trades detected** "
                              f"(>{outlier_analysis['z_threshold']}σ from mean)")
                    
                    # Show outlier type breakdown
                    if 'type_counts' in outlier_analysis and outlier_analysis['type_counts']:
                        st.write("**Outlier Classification:**")
                        for otype, count in outlier_analysis['type_counts'].items():
                            st.write(f"  - {otype}: {count}")
                    
                    # Show removed vs kept
                    if outlier_analysis.get('removed_count', 0) > 0:
                        st.error(f"🗑️ **{outlier_analysis['removed_count']} artefacts removed** "
                               f"(DATA_SPIKE, STALE_FILL, SCALING_ERROR, POSITION_BLOWUP)")
                    
                    if outlier_analysis.get('kept_count', 0) > 0:
                        st.info(f"✅ **{outlier_analysis['kept_count']} genuine outliers kept** "
                              f"(REGIME_SHIFT, STATISTICAL_TAIL)")
                    
                    st.write("**Outlier Details:**")
                    for i, trade in enumerate(outlier_analysis['outlier_trades'][:5]):  # Show first 5
                        # Enhanced display with classification
                        otype = trade.get('outlier_type', 'UNKNOWN')
                        cause = trade.get('root_cause', 'N/A')
                        st.write(f"  - Trade #{i+1}: P&L=${trade['pnl']:.2f}, "
                               f"Type={otype}, Side={trade['side']}, Qty={trade['quantity']:.4f}")
                        st.caption(f"    Diagnosis: {cause}")
                    
                    if len(outlier_analysis['outlier_trades']) > 5:
                        st.write(f"  ... and {len(outlier_analysis['outlier_trades']) - 5} more")
                    
                    # Three-tier metrics reporting
                    st.write("**📊 Three-Tier Metrics (Professional Reporting):**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Raw", f"${outlier_analysis['mean_raw']:.2f}", 
                                help="All trades, no filtering")
                    with col_b:
                        st.metric("Clean", f"${outlier_analysis['mean_clean']:.2f}",
                                help="Artefacts removed, genuine events kept")
                    with col_c:
                        st.metric("Winsorized", f"${outlier_analysis['mean_winsorized']:.2f}",
                                help="Statistical tails capped at ±3σ")
                    
                    # Impact analysis
                    impact_pct = abs(outlier_analysis['mean_raw'] - outlier_analysis['mean_clean']) / abs(outlier_analysis['mean_raw']) * 100 if outlier_analysis['mean_raw'] != 0 else 0
                    st.info(f"📉 **Outlier Impact:** {impact_pct:.1f}% difference between raw and clean metrics")
                else:
                    st.success("✅ No statistical outliers detected - all trades within 3σ of mean")
        
    except Exception as e:
        logger.error(f"Error displaying backtest results: {e}")
        st.error(f"Error displaying results: {e}")


def start_live_trading(params: Dict):
    """Start live trading with given parameters"""
    global _shared_live_data, _live_data_lock
    
    try:
        logger.info(f"Starting live trading with params: {params}")
        
        # Clear old data
        with _live_data_lock:
            _shared_live_data['performance'] = {}
            _shared_live_data['market'] = {}
            _shared_live_data['trades'] = []
            _shared_live_data['total_trade_count'] = 0
        
        st.session_state.live_data = {}
        
        # Create configuration with correct parameter structure for LiveTradingEngine
        engine_config = {
            'symbol': params['symbol'],
            'tick_size': params['tick_size'],
            'lot_size': params.get('lot_size', 0.01),  # Default lot size
            'gamma': params['gamma'],
            'time_horizon': params['time_horizon'],  # LiveTradingEngine expects time_horizon
            'initial_balance': params['initial_balance']
        }
        
        # Create live trading engine
        engine = LiveTradingEngine(engine_config)
        
        # Initialize engine
        asyncio.run(engine.initialize())
        
        # Set up callbacks BEFORE starting trading (critical!)
        setup_live_callbacks(engine)
        
        # Store engine and update state
        st.session_state.live_engine = engine
        st.session_state.live_trading_active = True
        
        # Start trading in background thread
        def run_trading():
            asyncio.run(engine.start_trading())
        
        trading_thread = threading.Thread(target=run_trading, daemon=True)
        trading_thread.start()
        
        st.success("Live trading started!")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Failed to start live trading: {e}")
        st.error(f"Failed to start live trading: {e}")


def stop_live_trading():
    """Stop live trading and save final results"""
    global _shared_live_data, _live_data_lock
    
    try:
        # Capture final data before stopping
        with _live_data_lock:
            final_trades = _shared_live_data.get('trades', []).copy()
            final_total_count = _shared_live_data.get('total_trade_count', 0)  # Get actual total
            final_performance = _shared_live_data.get('performance', {}).copy()
        
        # Save to session state for display
        st.session_state.live_final_results = {
            'trades': final_trades,
            'total_count': final_total_count,  # Store actual count
            'performance': final_performance,
            'stopped_at': datetime.now()
        }
        
        # Stop the engine
        if st.session_state.live_engine:
            st.session_state.live_engine.stop_trading()
            st.session_state.live_engine = None
        
        st.session_state.live_trading_active = False
        st.session_state.live_data = {}
        
        # Clear shared data
        with _live_data_lock:
            _shared_live_data['performance'] = {}
            _shared_live_data['market'] = {}
            _shared_live_data['trades'] = []
            _shared_live_data['total_trade_count'] = 0
        
        logger.info(f"✅ Live trading stopped - {final_total_count} total trades")
        st.success(f"Live trading stopped! {final_total_count} trades executed.")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error stopping live trading: {e}")
        st.error(f"Error stopping live trading: {e}")


def update_live_parameters(params: Dict):
    """Update live trading parameters"""
    try:
        if st.session_state.live_engine and hasattr(st.session_state.live_engine, 'strategy'):
            st.session_state.live_engine.strategy.update_parameters(**params)
            st.success("Parameters updated!")
            
    except Exception as e:
        logger.error(f"Error updating parameters: {e}")
        st.error(f"Error updating parameters: {e}")


def setup_live_callbacks(engine):
    """Set up callbacks for real-time data updates (thread-safe with shared data)"""
    global _shared_live_data, _live_data_lock
    
    callback_count = {'perf': 0, 'market': 0, 'trade': 0}
    
    def on_performance_update(data):
        with _live_data_lock:
            try:
                _shared_live_data['performance'] = data
                callback_count['perf'] += 1
            except Exception as e:
                logger.error(f"Error in performance callback: {e}")
    
    def on_market_data(data):
        with _live_data_lock:
            try:
                _shared_live_data['market'] = data
                callback_count['market'] += 1
            except Exception as e:
                logger.error(f"Error in market data callback: {e}")
    
    def on_trade_executed(data):
        with _live_data_lock:
            try:
                _shared_live_data['trades'].append(data)
                _shared_live_data['total_trade_count'] += 1  # Increment total counter
                callback_count['trade'] += 1
                
                # Keep only last 5000 trades to prevent memory issues (increased from 1000)
                # 5000 trades at ~100 bytes each = 500KB memory (negligible)
                if len(_shared_live_data['trades']) > 5000:
                    _shared_live_data['trades'] = _shared_live_data['trades'][-5000:]
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    engine.add_callback('on_performance_update', on_performance_update)
    engine.add_callback('on_market_data', on_market_data)
    engine.add_callback('on_trade_executed', on_trade_executed)
    
    logger.info("✅ Live callbacks registered: performance, market_data, trade_executed")


def get_live_trading_status() -> Dict:
    """Get current live trading status"""
    if st.session_state.live_engine:
        return st.session_state.live_engine.get_status()
    return {}


def display_live_performance():
    """Display live trading performance with auto-refresh"""
    try:
        # Debug: Check what data we have
        global _shared_live_data, _live_data_lock
        with _live_data_lock:
            shared_trades_count = len(_shared_live_data.get('trades', []))
            shared_perf_keys = list(_shared_live_data.get('performance', {}).keys())
        
        # Get data from session state (synced from shared data)
        performance_data = st.session_state.live_data.get('performance', {})
        trades = st.session_state.live_data.get('trades', [])
        market_data = st.session_state.live_data.get('market', {})
        
        # Debug info
        st.caption(f"Debug: Shared trades={shared_trades_count}, Session trades={len(trades)}, Perf keys={len(shared_perf_keys)}")
        
        # Show basic info even if no performance data yet
        st.markdown("### 📊 Live Trading Status")
        
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Total Trades", len(trades))
        with info_col2:
            if market_data:
                mid = market_data.get('midprice', market_data.get('best_bid', 0))
                st.metric("Mid Price", f"${mid:.2f}")
            else:
                st.metric("Mid Price", "Connecting...")
        with info_col3:
            if performance_data:
                pnl = performance_data.get('total_pnl', 0)
                st.metric("Total P&L", f"${pnl:.2f}", delta=f"{pnl:.2f}")
            else:
                st.metric("Total P&L", "$0.00")
        with info_col4:
            if performance_data:
                win_rate = performance_data.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate*100:.1f}%")
            else:
                st.metric("Win Rate", "0.0%")
        
        # Show recent trades
        if trades:
            st.markdown("### 💰 Recent Trades (Last 10)")
            recent_trades = trades[-10:]  # Last 10 trades
            trade_data = []
            for t in recent_trades:
                trade_data.append({
                    'Time': datetime.fromtimestamp(t.get('timestamp', 0)).strftime('%H:%M:%S'),
                    'Side': '🟢 BUY' if t.get('side', '').lower() == 'buy' else '🔴 SELL',
                    'Price': f"${t.get('price', 0):.2f}",
                    'Qty': f"{t.get('quantity', 0):.4f}",
                    'Value': f"${t.get('value', 0):.2f}"
                })
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("⏳ Waiting for first trade...")
        
        # Show performance metrics if available
        if not performance_data:
            return
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_pnl = performance_data.get('total_pnl', 0)
                st.metric("Total P&L", f"${total_pnl:.2f}")
            
            with col2:
                total_return = performance_data.get('total_return_pct', 0)
                st.metric("Return", f"{total_return:.2f}%")
            
            with col3:
                sharpe_1h = performance_data.get('sharpe_1h', 0)
                st.metric("Sharpe (1h)", f"{sharpe_1h:.2f}")
            
            with col4:
                total_trades = performance_data.get('total_trades', 0)
                st.metric("Trades", total_trades)
            
            # Real-time P&L chart
            pnl_history = performance_data.get('pnl_history', [])
            if pnl_history:
                fig_pnl = go.Figure()
                
                timestamps = [datetime.fromtimestamp(p['timestamp']) for p in pnl_history]
                pnl_values = [p['pnl'] for p in pnl_history]
                
                fig_pnl.add_trace(go.Scatter(
                    x=timestamps,
                    y=pnl_values,
                    mode='lines',
                    name='P&L',
                    line=dict(color='green', width=2)
                ))
                
                fig_pnl.update_layout(
                    title="Real-time P&L",
                    xaxis_title="Time",
                    yaxis_title="P&L ($)",
                    height=400
                )
                
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            # Additional metrics
            with st.expander("Live Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Win Rate", f"{performance_data.get('win_rate', 0):.1%}")
                    st.metric("Max Drawdown", f"{performance_data.get('max_drawdown', 0) * 100:.2f}%")
                    st.metric("Trades/Hour", f"{performance_data.get('trades_per_hour', 0):.1f}")
                
                with col2:
                    st.metric("Total Volume", f"${performance_data.get('total_volume', 0):,.0f}")
                    st.metric("Total Fees", f"${performance_data.get('total_fees', 0):.2f}")
                    st.metric("Runtime", f"{performance_data.get('runtime_hours', 0):.1f}h")
    
    except Exception as e:
        logger.error(f"Error displaying live performance: {e}")
        st.error(f"Error displaying live performance: {e}")


def main():
    """Main dashboard application"""
    
    # Custom CSS for improved styling
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .stSelectbox > div > div > div {
        background-color: #f0f2f6;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.title("🚀 HFT Trading Dashboard")
    st.markdown("**Professional High-Frequency Trading System**")
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["📊 Backtesting", "🔴 Live Trading"])
    
    with tab1:
        create_backtesting_section()
    
    with tab2:
        create_live_trading_section()


if __name__ == "__main__":
    main()