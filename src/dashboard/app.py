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
from src.utils.config import config
from src.utils.logger import setup_production_logging, get_logger

# Initialize logging (INFO level - no DEBUG spam)
setup_production_logging()
logger = get_logger('dashboard')

# Supported trading pairs
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]

# Session state for backtesting
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None


def create_backtesting_section():
    """Create the backtesting configuration and results section"""
    st.header("📊 Backtesting")
    
    # Important guidance banner
    st.info("""
    💡 **Key Parameter Guidance:**
    - **Min Spread**: Must be wide enough to cover fees (0.04% round-trip) + adverse selection + profit. 
      **Recommended: 0.25-0.35%** (25-35 bps) for consistent profitability.
    - **Gamma**: Lower = tighter quotes. **Optimal: 0.01-0.02** for HFT market making.
    - **Time Horizon**: Quote refresh rate. **Optimal: 10-20 seconds** for balance between responsiveness and stability.
    """)
    
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
        time_horizon = st.slider("Time Horizon (sec)", 5, 60, 15, 5, key="bt_time_horizon",
                               help="Quote refresh time. HFT optimal: 10-20 seconds for market making")
    
    with col3:
        st.markdown("**Spread & Size**")
        min_spread = st.number_input("Min Spread", min_value=0.0001, max_value=0.05, value=0.0025, step=0.0001, key="bt_min_spread",
                                    help="Minimum spread (0.0025 = 0.25% = 25 bps). Must beat fees + adverse selection. Lower spreads = more fills but less profit per trade.", format="%.4f")
        max_drawdown_pct = st.slider(
            "Max Drawdown (%)", 
            1, 50, 30, 5, 
            key="bt_max_dd",
            help="Stop backtesting if drawdown exceeds this percentage. HFT typical: 20-30% for market making."
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
    if st.button("🚀 Run Backtest", type="primary", width="stretch"):
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
                time_horizon=params['time_horizon'],
                min_spread=params['min_spread'],
                max_drawdown=params['max_drawdown_pct'] / 100.0,
                tick_size=params['tick_size']
            )
            
            # Run backtest
            engine = BacktestEngine()
            results = engine.run_backtest(config)
            
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
                    <li><b>Order Fill Rate:</b> {perf.fill_rate:.1%} (realistic execution)</li>
                    <li><b>Total Volume:</b> ${perf.total_volume:,.0f}</li>
                </ul>
                <p style="margin: 10px 0 0 0; padding: 10px; background-color: rgba(255,255,255,0.7); border-radius: 5px;">
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
            realized_pnl_series = metrics.realized_pnl_series if hasattr(metrics, 'realized_pnl_series') else []
            unrealized_pnl_series = metrics.unrealized_pnl_series if hasattr(metrics, 'unrealized_pnl_series') else []
            
            if len(trades) > 0:
                # Prepare trade data
                trade_pnls = [t['pnl'] for t in trades]
                trade_times = [t['timestamp'] for t in trades]
                
                # Create first row with 2 charts side-by-side
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Cumulative P&L Chart (Total)
                    fig_cumulative = go.Figure()
                    
                    if pnl_series:
                        fig_cumulative.add_trace(
                            go.Scatter(
                                x=[p[0] for p in pnl_series],
                                y=[p[1] for p in pnl_series],
                                mode='lines',
                                line=dict(color='green', width=2),
                                name='Total P&L',
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
                    
                    st.plotly_chart(fig_cumulative, width="stretch")
                    st.caption("📈 **Total P&L:** Shows combined realized + unrealized P&L. "
                              "Peaks occur when holding positions during favorable price moves.")
                
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
                    
                    st.plotly_chart(fig_histogram, width="stretch")
        
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
                st.metric("Order Fill Rate", f"{fill_rate:.1%}")
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
                st.metric("Order Fill Rate", f"{results.get('fill_rate', 0):.1%}")
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
                
                # Calculate CORRECT order-level fill rate (industry standard)
                # Total orders = quotes_submitted × 2 (each quote has bid + ask)
                # Fill rate = percentage of individual orders that got filled
                total_orders_submitted = quotes_submitted * 2
                order_fill_rate = (total_fills / max(total_orders_submitted, 1)) * 100
                quotes_per_candle = quotes_submitted / max(data_points, 1)
                
                st.write(f"**Market Data Points:** {data_points:,} 1-minute candles received from Binance")
                st.write(f"**Quotes Submitted:** {quotes_submitted:,} quote pairs sent to market ({quotes_per_candle:.2f} quotes/candle)")
                st.write(f"**Quotes with Fills:** {quotes_filled:,} quote pairs that got at least one side filled")
                st.write(f"**Total Fill Events:** {total_fills:,} individual order fills (bid + ask sides)")
                
                st.success(f"✅ **Order Fill Rate: {order_fill_rate:.1f}%** = {total_fills:,} filled orders / {total_orders_submitted:,} total orders")
                
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
                    outlier_total_pnl = outlier_analysis['total_outlier_impact']
                    st.metric("Outlier Total P&L", 
                             f"${outlier_total_pnl:.2f}",
                             help="Total P&L from all outlier trades (kept in results)")
                
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
                    st.write("**📊 Three-Tier Metrics Breakdown:**")
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



def main():
    """Main dashboard application - Backtesting only"""
    
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
    st.title("HFT Backtesting Dashboard")
    st.markdown("High-Frequency Trading - Backtesting & Analysis")
    
    # Single section - Backtesting only
    create_backtesting_section()


if __name__ == "__main__":
    main()