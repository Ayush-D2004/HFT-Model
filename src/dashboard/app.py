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
from src.utils.logger import setup_development_logging, get_logger

# Initialize logging
setup_development_logging()
logger = get_logger('dashboard')

# Supported trading pairs
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]

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
    
    # Configuration columns
    config_col, spacer, results_col = st.columns([1, 0.1, 2])
    
    with config_col:
        st.subheader("Configuration")
        
        # Trading parameters
        with st.expander("Trading Parameters", expanded=True):
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
            
            gamma = st.slider("Risk Aversion (γ)", 0.005, 0.05, 0.015, 0.005, key="bt_gamma",
                            help="Lower = tighter spreads. HFT optimal: 0.01-0.02. Max 0.05 to prevent uncompetitive spreads.")
            time_horizon = st.slider("Time Horizon (sec)", 5, 60, 10, 5, key="bt_time_horizon",
                                   help="Quote refresh time. HFT optimal: 5-15 seconds. Shorter = faster adaptation")
            min_spread = st.number_input("Min Spread", 0.0001, 0.01, 0.0005, 0.0001, key="bt_min_spread",
                                        help="Minimum spread (0.0005 = 0.05% = 5 bps). HFT typical: 5-20 bps", format="%.4f")
        
        # Time scope configuration
        with st.expander("Time Scope", expanded=True):
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
            end_date = st.date_input("End Date", datetime.now())
            
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            max_position = st.slider(
                "Max Position", 
                1.0, 20.0, 5.0, 0.5, 
                key="bt_max_pos",
                help="Maximum position size in base currency. Lower = less risk"
            )
            tick_size = st.number_input("Tick Size", 0.001, 1.0, 0.01, 0.001, 
                                       help="Price increment. Use 0.01 for most pairs")
            lot_size = st.number_input("Lot Size", 0.0001, 1.0, 0.001, 0.0001,
                                      help="Minimum order size. Use 0.001 for most pairs")
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            run_backtest({
                'symbol': symbol,
                'gamma': gamma,
                'time_horizon': time_horizon,
                'min_spread': min_spread,
                'max_position': max_position,
                'initial_balance': initial_balance,
                'tick_size': tick_size,
                'lot_size': lot_size,
                'start_date': start_date,
                'end_date': end_date
            })
    
    with results_col:
        st.subheader("Results")
        
        if st.session_state.backtest_results is None:
            st.info("Configure parameters and run backtest to see results")
        else:
            display_backtest_results(st.session_state.backtest_results)


def create_live_trading_section():
    """Create the live trading configuration and monitoring section"""
    st.header("🔴 Live Trading")
    
    # Configuration and controls
    config_col, spacer, monitor_col = st.columns([1, 0.1, 2])
    
    with config_col:
        st.subheader("Configuration")
        
        # Trading parameters
        with st.expander("Trading Parameters", expanded=True):
            symbol = st.selectbox("Symbol", SUPPORTED_SYMBOLS, key="live_symbol",
                                help="Choose high-volume pairs for better execution")
            
            # Initial capital - prominent position
            initial_balance = st.number_input(
                "Initial Capital (USDT)", 
                min_value=1000.0, 
                max_value=1000000.0, 
                value=10000.0,
                step=1000.0,
                key="live_balance",
                help="Starting capital in USDT. Typical: $5k-50k for retail"
            )
            
            gamma = st.slider("Risk Aversion (γ)", 0.005, 0.05, 0.015, 0.005, key="live_gamma",
                            help="Lower = tighter spreads. HFT optimal: 0.01-0.02. Max 0.05 to prevent uncompetitive spreads.")
            time_horizon = st.slider("Time Horizon (sec)", 5, 60, 10, 5, key="live_time_horizon",
                                   help="Quote refresh time. HFT optimal: 5-15s")
            min_spread = st.number_input("Min Spread", 0.0001, 0.01, 0.0005, 0.0001, key="live_min_spread",
                                        help="Minimum spread (0.0005 = 5 bps). HFT typical: 5-20 bps", format="%.4f")
            max_position = st.slider("Max Position", 1.0, 20.0, 5.0, 0.5, key="live_max_pos",
                                   help="Maximum position size in base currency")
            tick_size = st.number_input("Tick Size", 0.001, 1.0, 0.01, 0.001, key="live_tick_size",
                                       help="Price increment (usually 0.01)")
            lot_size = st.number_input("Lot Size", 0.0001, 1.0, 0.001, 0.0001, key="live_lot_size",
                                      help="Minimum order size (usually 0.001)")
        
        # Risk management
        with st.expander("Risk Management"):
            max_loss = st.number_input("Max Loss ($)", 100, 10000, 1000, 100)
            max_drawdown = st.slider("Max Drawdown (%)", 1, 50, 10, 1)
            emergency_stop = st.checkbox("Enable Emergency Stop")
        
        # Trading controls
        st.subheader("Controls")
        
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
            if st.button("🛑 Stop Live Trading", type="secondary", use_container_width=True):
                stop_live_trading()
            
            # Live parameter updates
            if st.button("📝 Update Parameters", use_container_width=True):
                update_live_parameters({
                    'gamma': gamma,
                    'time_horizon': time_horizon,
                    'min_spread': min_spread,
                    'max_position': max_position
                })
        
        # Connection status
        if st.session_state.live_trading_active:
            status = get_live_trading_status()
            if status.get('is_connected', False):
                st.success("🟢 Connected to Binance")
            else:
                st.error("🔴 Disconnected")
            
            st.metric("Runtime", f"{status.get('runtime', 0):.1f}s")
            st.metric("Trades", status.get('trade_count', 0))
    
    with monitor_col:
        st.subheader("Live Performance")
        
        if not st.session_state.live_trading_active:
            st.info("Start live trading to see real-time performance")
        else:
            display_live_performance()


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
                max_position=params['max_position'],
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
                st.metric("Total Trades", perf.total_trades)
        else:
            # Dictionary format (fallback)
            with col1:
                st.metric("Total Return", f"{results.get('total_return_pct', 0):.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{results.get('max_drawdown', 0) * 100:.2f}%")
            with col4:
                st.metric("Total Trades", results.get('total_trades', 0))
        
        # P&L Chart
        pnl_history = None
        timestamps = None
        
        if hasattr(results, 'performance'):
            # BacktestResult object
            if hasattr(results.performance, 'pnl_history'):
                pnl_history = results.performance.pnl_history
                timestamps = results.performance.timestamps if hasattr(results.performance, 'timestamps') else range(len(pnl_history))
        elif 'pnl_history' in results:
            # Dictionary format
            pnl_history = results['pnl_history']
            timestamps = results.get('timestamps', range(len(pnl_history)))
        
        if pnl_history:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=timestamps,
                y=pnl_history,
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='green', width=2)
            ))
            
            fig_pnl.update_layout(
                title="Cumulative P&L",
                xaxis_title="Time",
                yaxis_title="P&L ($)",
                height=400
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No P&L history data available for charting")
        
        # ============================================
        # COMPREHENSIVE PERFORMANCE ANALYSIS CHARTS
        # ============================================
        if hasattr(results, 'metrics'):
            metrics = results.metrics
            perf = results.performance
            
            st.markdown("---")
            st.subheader("📊 Performance Analysis")
            
            # Get trade data
            trades = metrics.trades if hasattr(metrics, 'trades') else []
            position_series = metrics.position_series if hasattr(metrics, 'position_series') else []
            pnl_series = metrics.pnl_series if hasattr(metrics, 'pnl_series') else []
            
            if len(trades) > 0:
                # Prepare trade data
                trade_pnls = [t['pnl'] for t in trades]
                trade_times = [t['timestamp'] for t in trades]
                winning_trades = [p for p in trade_pnls if p > 0]
                losing_trades = [p for p in trade_pnls if p < 0]
                
                # Create subplot figure with 3x2 grid (5 charts total)
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=(
                        'Trade P&L Distribution (Histogram)',
                        'Position Over Time (Inventory)',
                        'Drawdown Over Time',
                        'Rolling Sharpe Ratio (30-trade window)',
                        'Cumulative Trade P&L vs Total P&L',
                        ''  # Empty 6th subplot
                    ),
                    specs=[
                        [{"type": "xy"}, {"type": "xy"}],
                        [{"type": "xy"}, {"type": "xy"}],
                        [{"type": "xy"}, {"type": "xy"}]
                    ],
                    vertical_spacing=0.12,
                    horizontal_spacing=0.12
                )
                
                # 1. Trade P&L Histogram
                fig.add_trace(
                    go.Histogram(
                        x=trade_pnls,
                        nbinsx=30,
                        marker_color='lightblue',
                        name='Trade P&L',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # 2. Position/Inventory Over Time
                if position_series:
                    pos_times = [p[0] for p in position_series]
                    pos_values = [p[1] for p in position_series]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pos_times,
                            y=pos_values,
                            mode='lines',
                            line=dict(color='purple', width=2),
                            fill='tozeroy',
                            name='Position',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
                
                # 3. Drawdown Over Time
                if pnl_series:
                    pnl_vals = [p[1] for p in pnl_series]
                    pnl_times = [p[0] for p in pnl_series]
                    
                    # Calculate drawdown
                    running_max = np.maximum.accumulate(pnl_vals)
                    drawdown = [(pnl - rmax) for pnl, rmax in zip(pnl_vals, running_max)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pnl_times,
                            y=drawdown,
                            mode='lines',
                            line=dict(color='red', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            name='Drawdown',
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                
                # 4. Rolling Sharpe Ratio
                if len(trade_pnls) > 30:
                    window = 30
                    rolling_sharpe = []
                    rolling_times = []
                    
                    for i in range(window, len(trade_pnls)):
                        window_pnls = trade_pnls[i-window:i]
                        if np.std(window_pnls) > 0:
                            sharpe = np.mean(window_pnls) / np.std(window_pnls) * np.sqrt(252)  # Annualized
                            rolling_sharpe.append(sharpe)
                            rolling_times.append(trade_times[i])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=rolling_times,
                            y=rolling_sharpe,
                            mode='lines',
                            line=dict(color='blue', width=2),
                            name='Rolling Sharpe',
                            showlegend=False
                        ),
                        row=2, col=2
                    )
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=2)
                
                # 5. Cumulative Trade P&L vs Total P&L
                cumulative_trade_pnl = np.cumsum(trade_pnls)
                
                fig.add_trace(
                    go.Scatter(
                        x=trade_times,
                        y=cumulative_trade_pnl,
                        mode='lines',
                        line=dict(color='green', width=2),
                        name='Realized P&L (Closed Trades)',
                    ),
                    row=3, col=1
                )
                
                if pnl_series:
                    fig.add_trace(
                        go.Scatter(
                            x=[p[0] for p in pnl_series],
                            y=[p[1] for p in pnl_series],
                            mode='lines',
                            line=dict(color='orange', width=2, dash='dash'),
                            name='Total P&L (Inc. Unrealized)',
                        ),
                        row=3, col=1
                    )
                
                # Update axes labels
                # Row 1, Col 1: Histogram
                fig.update_xaxes(title_text="Trade P&L ($)", row=1, col=1)
                fig.update_yaxes(title_text="Frequency", row=1, col=1)
                
                # Row 1, Col 2: Position
                fig.update_xaxes(title_text="Time", row=1, col=2)
                fig.update_yaxes(title_text="Position (BTC)", row=1, col=2)
                
                # Row 2, Col 1: Drawdown
                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
                
                # Row 2, Col 2: Rolling Sharpe
                fig.update_xaxes(title_text="Time", row=2, col=2)
                fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
                
                # Row 3, Col 1: Cumulative P&L
                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
                
                # Update layout
                fig.update_layout(
                    height=1200,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.05,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ============================================
                # KEY INSIGHTS SECTION
                # ============================================
                st.markdown("---")
                st.subheader("🔍 Key Insights & Diagnostics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📈 P&L Analysis**")
                    realized_pnl = sum(trade_pnls)
                    total_pnl = getattr(perf, 'total_pnl', 0)
                    unrealized_pnl = total_pnl - realized_pnl
                    
                    st.write(f"Realized P&L: ${realized_pnl:.2f}")
                    st.write(f"Unrealized P&L: ${unrealized_pnl:.2f}")
                    st.write(f"Total P&L: ${total_pnl:.2f}")
                    
                    if abs(unrealized_pnl) > abs(realized_pnl):
                        st.warning(f"⚠️ Large open position! Unrealized P&L (${unrealized_pnl:.2f}) > Realized P&L")
                    
                with col2:
                    st.markdown("**🎯 Trade Quality**")
                    if len(winning_trades) > 0 and len(losing_trades) > 0:
                        avg_win = np.mean(winning_trades)
                        avg_loss = np.mean(losing_trades)
                        win_loss_ratio = abs(avg_win / avg_loss)
                        
                        st.write(f"Avg Win: ${avg_win:.2f}")
                        st.write(f"Avg Loss: ${avg_loss:.2f}")
                        st.write(f"Win/Loss Ratio: {win_loss_ratio:.2f}x")
                        
                        if win_loss_ratio < 1.0:
                            st.warning("⚠️ Average losses > average wins!")
                        elif win_loss_ratio > 2.0:
                            st.success("✅ Good risk/reward ratio")
                
                with col3:
                    st.markdown("**⚖️ Market Making Health**")
                    fill_rate = getattr(perf, 'fill_rate', 0)
                    
                    if position_series:
                        final_position = position_series[-1][1] if position_series else 0
                        st.write(f"Final Position: {final_position:.4f} BTC")
                        
                        if abs(final_position) > 0.1:
                            st.warning(f"⚠️ Not market-neutral! Holding {final_position:.4f} BTC")
                        else:
                            st.success("✅ Near-flat position")
                    
                    st.write(f"Fill Rate: {fill_rate:.1%}")
                    if fill_rate < 0.3:
                        st.error("❌ Fill rate too low (<30%)")
                    elif fill_rate < 0.5:
                        st.warning("⚠️ Fill rate below target (30-50%)")
                    elif fill_rate < 0.7:
                        st.info("📊 Moderate fill rate (50-70%)")
                    else:
                        st.success("✅ Good fill rate (70%+)")
        
        # Debug: Show raw trade data + outlier detection
        if hasattr(results, 'metrics') and hasattr(results.metrics, 'trades'):
            with st.expander("🔍 Debug: Trade P&L Data & Outlier Detection", expanded=False):
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
                    
                    st.write(f"**First 10 Trade P&Ls:** {trade_pnls[:10]}")
                    st.write(f"**Last 10 Trade P&Ls:** {trade_pnls[-10:]}")
                    
                    # DYNAMIC OUTLIER DETECTION
                    st.markdown("---")
                    st.markdown("### 📊 Statistical Outlier Analysis")
                    
                    outlier_analysis = results.metrics.detect_trade_outliers(std_threshold=3.0)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Outlier Trades", 
                                 f"{outlier_analysis['outlier_count']} ({outlier_analysis['outlier_pct']:.1f}%)")
                    with col2:
                        st.metric("Avg P&L (Normal)", 
                                 f"${outlier_analysis['mean_normal']:.2f}",
                                 delta=f"{outlier_analysis['mean_with_outliers'] - outlier_analysis['mean_normal']:.2f}")
                    with col3:
                        st.metric("Outlier Impact", 
                                 f"${outlier_analysis['total_outlier_impact']:.2f}")
                    
                    if outlier_analysis['outlier_count'] > 0:
                        st.warning(f"⚠️ **{outlier_analysis['outlier_count']} outlier trades detected** "
                                  f"(>{outlier_analysis['z_threshold']}σ from mean)")
                        
                        st.write("**Outlier Details:**")
                        for i, trade in enumerate(outlier_analysis['outlier_trades'][:5]):  # Show first 5
                            st.write(f"  - Trade #{i+1}: P&L=${trade['pnl']:.2f}, "
                                   f"Side={trade['side']}, Qty={trade['quantity']:.4f}")
                        
                        if len(outlier_analysis['outlier_trades']) > 5:
                            st.write(f"  ... and {len(outlier_analysis['outlier_trades']) - 5} more")
                        
                        # Adjusted metrics
                        st.info(f"💡 **Adjusted Avg P&L (excluding outliers):** ${outlier_analysis['mean_normal']:.2f} per trade")
                        st.info(f"📉 **This is {abs(outlier_analysis['mean_with_outliers'] - outlier_analysis['mean_normal']) / outlier_analysis['mean_with_outliers'] * 100:.1f}% "
                               f"{'lower' if outlier_analysis['mean_normal'] < outlier_analysis['mean_with_outliers'] else 'higher'} "
                               f"than reported avg**")
                    else:
                        st.success("✅ No statistical outliers detected - all trades within 3σ of mean")
                trades = results.metrics.trades
                if len(trades) > 0:
                    trade_pnls = [t['pnl'] for t in trades]
                    st.write(f"**Total Trades:** {len(trades)}")
                    st.write(f"**Sum of Trade P&Ls:** ${sum(trade_pnls):.2f}")
                    st.write(f"**Winning Trades (pnl > 0):** {len([p for p in trade_pnls if p > 0])}")
                    st.write(f"**Losing Trades (pnl < 0):** {len([p for p in trade_pnls if p < 0])}")
                    st.write(f"**Zero P&L Trades (pnl == 0):** {len([p for p in trade_pnls if p == 0])}")
                    st.write(f"**First 10 Trade P&Ls:** {trade_pnls[:10]}")
                    st.write(f"**Last 10 Trade P&Ls:** {trade_pnls[-10:]}")
                else:
                    st.write("No trades recorded")
        
        # Additional metrics
        with st.expander("Detailed Metrics"):
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
        
    except Exception as e:
        logger.error(f"Error displaying backtest results: {e}")
        st.error(f"Error displaying results: {e}")


def start_live_trading(params: Dict):
    """Start live trading with given parameters"""
    try:
        logger.info(f"Starting live trading with params: {params}")
        
        # Create configuration with correct parameter structure for LiveTradingEngine
        engine_config = {
            'symbol': params['symbol'],
            'tick_size': params['tick_size'],
            'lot_size': params.get('lot_size', 0.001),  # Default lot size
            'gamma': params['gamma'],
            'time_horizon': params['time_horizon'],  # LiveTradingEngine expects time_horizon
            'initial_balance': params['initial_balance']
        }
        
        # Create live trading engine
        engine = LiveTradingEngine(engine_config)
        
        # Initialize engine
        asyncio.run(engine.initialize())
        
        # Start trading in background thread
        def run_trading():
            asyncio.run(engine.start_trading())
        
        trading_thread = threading.Thread(target=run_trading, daemon=True)
        trading_thread.start()
        
        # Store engine and update state
        st.session_state.live_engine = engine
        st.session_state.live_trading_active = True
        
        # Set up callbacks for real-time updates
        setup_live_callbacks(engine)
        
        st.success("Live trading started!")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Failed to start live trading: {e}")
        st.error(f"Failed to start live trading: {e}")


def stop_live_trading():
    """Stop live trading"""
    try:
        if st.session_state.live_engine:
            st.session_state.live_engine.stop_trading()
            st.session_state.live_engine = None
        
        st.session_state.live_trading_active = False
        st.session_state.live_data = {}
        
        st.success("Live trading stopped!")
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
    """Set up callbacks for real-time data updates"""
    def on_performance_update(data):
        st.session_state.live_data['performance'] = data
    
    def on_market_data(data):
        st.session_state.live_data['market'] = data
    
    def on_trade_executed(data):
        if 'trades' not in st.session_state.live_data:
            st.session_state.live_data['trades'] = []
        st.session_state.live_data['trades'].append(data)
    
    engine.add_callback('on_performance_update', on_performance_update)
    engine.add_callback('on_market_data', on_market_data)
    engine.add_callback('on_trade_executed', on_trade_executed)


def get_live_trading_status() -> Dict:
    """Get current live trading status"""
    if st.session_state.live_engine:
        return st.session_state.live_engine.get_status()
    return {}


def display_live_performance():
    """Display live trading performance with auto-refresh"""
    try:
        # Use auto-refresh to update data
        placeholder = st.empty()
        
        with placeholder.container():
            performance_data = st.session_state.live_data.get('performance', {})
            
            if not performance_data:
                st.info("Waiting for performance data...")
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
    
    # Auto-refresh for live data
    if st.session_state.live_trading_active:
        time.sleep(1)  # Refresh every second
        st.rerun()


if __name__ == "__main__":
    main()