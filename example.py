"""
HFT Market Maker Example
=======================

Demonstration of the complete HFT market making system.
This example shows how to:
1. Run a backtest with synthetic data
2. Analyze results
3. Run parameter optimization
"""

import time
from datetime import datetime, timedelta

# Import HFT modules
from src.backtesting import BacktestEngine, BacktestConfig
from src.strategy import QuoteParameters
from src.utils.config import config
from src.utils.logger import setup_backtesting_logging, setup_development_logging

def run_single_backtest():
    """Run a single backtest demonstration"""
    print("🔬 Running Single Backtest...")
    
    # Configure backtest
    backtest_config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-02",
        symbol="BTCUSDT",
        gamma=0.1,              # Risk aversion parameter
        time_horizon=30.0,      # 30-second decision horizon
        min_spread=0.02,        # Minimum 2 cent spread
        tick_size=0.01,         # 1 cent tick size
        lot_size=0.001,         # 0.001 BTC minimum order size
        max_position=5.0,       # Maximum 5 BTC position
        max_daily_loss=1000.0,  # $1000 daily loss limit
        initial_capital=10000.0, # $10k starting capital
        maker_fee=0.001,        # 0.1% maker fee
        taker_fee=0.001         # 0.1% taker fee
    )
    
    # Initialize backtest engine
    engine = BacktestEngine()
    
    # Run the backtest with silent logging
    start_time = time.time()
    result = engine.run_backtest(backtest_config, silent=True)
    execution_time = time.time() - start_time
    
    # Display results
    if result.success:
        perf = result.performance
        
        print(f"\n✅ Backtest Completed in {execution_time:.2f}s")
        print("=" * 50)
        print(f"📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total PnL:           ${perf.total_pnl:,.2f}")
        print(f"Realized PnL:        ${perf.realized_pnl:,.2f}")
        print(f"Unrealized PnL:      ${perf.unrealized_pnl:,.2f}")
        print(f"Total Fees:          ${perf.total_fees:,.2f}")
        print(f"Net Return:          {(perf.total_pnl/backtest_config.initial_capital)*100:+.1f}%")
        
        print(f"\n📈 RISK METRICS")
        print("-" * 30)
        print(f"Sharpe Ratio:        {perf.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:       {perf.sortino_ratio:.2f}")
        print(f"Calmar Ratio:        {perf.calmar_ratio:.2f}")
        print(f"Max Drawdown:        {perf.max_drawdown:.1%}")
        print(f"Volatility:          {perf.volatility:.1%}")
        
        print(f"\n💹 TRADING METRICS")
        print("-" * 30)
        print(f"Total Trades:        {perf.total_trades}")
        print(f"Winning Trades:      {perf.winning_trades}")
        print(f"Win Rate:            {perf.win_rate:.1%}")
        print(f"Avg Win:             ${perf.avg_win:.2f}")
        print(f"Avg Loss:            ${perf.avg_loss:.2f}")
        print(f"Profit Factor:       {perf.profit_factor:.2f}")
        
        print(f"\n⚡ EXECUTION METRICS")
        print("-" * 30)
        print(f"Fill Rate:           {perf.fill_rate:.1%}")
        print(f"Quote Hit Rate:      {perf.quote_hit_rate:.1%}")
        print(f"Avg Fill Latency:    {perf.avg_fill_latency_ms:.1f}ms")
        print(f"Avg Spread Captured: ${perf.avg_spread_captured:.4f}")
        print(f"Inventory Turnover:  {perf.inventory_turnover:.2f}/hr")
        
        print(f"\n⏱️  TIME METRICS")
        print("-" * 30)
        print(f"Duration:            {perf.duration_hours:.1f} hours")
        print(f"Start Time:          {datetime.fromtimestamp(perf.start_time)}")
        print(f"End Time:            {datetime.fromtimestamp(perf.end_time)}")
        
    else:
        print(f"❌ Backtest Failed: {result.error}")
    
    return result

def run_parameter_optimization():
    """Run parameter optimization demonstration"""
    print("\n🔍 Running Parameter Optimization...")
    
    # Base configuration
    base_config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-02",
        symbol="BTCUSDT",
        initial_capital=10000.0
    )
    
    # Define parameter grid for optimization
    parameter_grid = {
        'gamma': [0.05, 0.1, 0.15, 0.2],           # Risk aversion levels
        'time_horizon': [15.0, 30.0, 60.0],        # Decision horizons
        'min_spread': [0.01, 0.02, 0.03]           # Minimum spreads
    }
    
    print(f"Testing {len(parameter_grid['gamma']) * len(parameter_grid['time_horizon']) * len(parameter_grid['min_spread'])} parameter combinations...")
    
    # Run parameter sweep with silent logging
    engine = BacktestEngine()
    start_time = time.time()
    
    results = engine.run_parameter_sweep(
        base_config, 
        parameter_grid, 
        max_workers=2,  # Use 2 parallel workers
        silent=True     # Enable silent logging
    )
    
    execution_time = time.time() - start_time
    
    # Analyze results
    analysis = engine.analyze_results(results)
    
    if 'summary' in analysis:
        summary = analysis['summary']
        print(f"\n📊 OPTIMIZATION RESULTS")
        print("=" * 40)
        print(f"Total Backtests:     {summary['total_backtests']}")
        print(f"Successful:          {summary['successful_backtests']}")
        print(f"Success Rate:        {summary['success_rate']:.1%}")
        print(f"Execution Time:      {execution_time:.1f}s")
        
        if 'pnl_stats' in analysis:
            pnl_stats = analysis['pnl_stats']
            print(f"\n💰 PnL STATISTICS")
            print("-" * 25)
            print(f"Mean PnL:            ${pnl_stats['mean']:,.2f}")
            print(f"Best PnL:            ${pnl_stats['max']:,.2f}")
            print(f"Worst PnL:           ${pnl_stats['min']:,.2f}")
            print(f"PnL Std Dev:         ${pnl_stats['std']:,.2f}")
            print(f"Positive PnL Rate:   {pnl_stats['positive_pnl_rate']:.1%}")
        
        if 'best_result' in analysis:
            best_results = analysis['best_result']
            
            print(f"\n🏆 BEST CONFIGURATIONS")
            print("-" * 35)
            
            # Best by PnL
            best_pnl = best_results['by_pnl']
            print(f"Best PnL: ${best_pnl.performance.total_pnl:.2f}")
            print(f"  γ={best_pnl.config.gamma}, T={best_pnl.config.time_horizon}s, spread={best_pnl.config.min_spread}")
            
            # Best by Sharpe
            best_sharpe = best_results['by_sharpe']
            print(f"Best Sharpe: {best_sharpe.performance.sharpe_ratio:.2f}")
            print(f"  γ={best_sharpe.config.gamma}, T={best_sharpe.config.time_horizon}s, spread={best_sharpe.config.min_spread}")
            
            # Best by Drawdown
            best_dd = best_results['by_drawdown']
            print(f"Lowest Drawdown: {best_dd.performance.max_drawdown:.1%}")
            print(f"  γ={best_dd.config.gamma}, T={best_dd.config.time_horizon}s, spread={best_dd.config.min_spread}")
    
    return results

def demonstrate_strategy_components():
    """Demonstrate individual strategy components"""
    print("\n🧠 Demonstrating Strategy Components...")
    
    from src.strategy import AvellanedaStoikovPricer, RiskManager, RiskLimits
    
    # Initialize Avellaneda-Stoikov pricer
    pricer = AvellanedaStoikovPricer(
        tick_size=0.01,
        lot_size=0.001,
        ewma_alpha=0.2,
        max_inventory=5.0
    )
    
    print("\n📈 Price Update Simulation")
    print("-" * 30)
    
    # Simulate market updates
    import random
    current_price = 50000.0
    
    for i in range(10):
        # Random price movement
        price_change = random.gauss(0, 5)
        current_price += price_change
        
        # Update pricer
        pricer.update_market(current_price)
        
        # Simulate some trades for arrival rate estimation
        if random.random() < 0.3:
            pricer.register_trade_event()
        
        # Generate quote
        quote_params = QuoteParameters(
            gamma=0.1,
            T=30.0,
            min_spread=0.02
        )
        
        quote = pricer.compute_quotes(quote_params)
        
        print(f"Update {i+1}: Price=${current_price:.2f}, "
              f"Bid=${quote.bid_price:.2f}, Ask=${quote.ask_price:.2f}, "
              f"Spread=${quote.spread:.2f}")
    
    # Risk manager demonstration
    print(f"\n⚠️  Risk Manager Simulation")
    print("-" * 30)
    
    risk_limits = RiskLimits(
        max_position=5.0,
        max_daily_loss=500.0,
        max_drawdown=0.03
    )
    
    risk_manager = RiskManager(risk_limits)
    
    # Simulate position updates
    risk_manager.update_position(2.5, 49950.0)
    risk_manager.update_pnl(current_price)
    
    # Check quote risk
    allowed = risk_manager.check_quote_risk(quote, current_price)
    
    metrics = risk_manager.get_risk_metrics(current_price)
    
    print(f"Position: {risk_manager.current_position:.2f}")
    print(f"Unrealized PnL: ${metrics.unrealized_pnl:.2f}")
    print(f"Risk Level: {metrics.risk_level.value}")
    print(f"Quote Allowed: {allowed}")

def main():
    """Main demonstration function"""
    print("🚀 HFT Market Maker System Demonstration")
    print("=" * 60)
    
    # Setup clean logging for examples
    setup_development_logging()
    
    try:
        # 1. Demonstrate strategy components
        demonstrate_strategy_components()
        
        # 2. Run single backtest
        single_result = run_single_backtest()
        
        # 3. Run parameter optimization
        optimization_results = run_parameter_optimization()
        
        print(f"\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Next Steps:")
        print("1. Launch the dashboard: python run_dashboard.py")
        print("2. Modify parameters in src/utils/config.py")
        print("3. Add your Binance testnet credentials to .env")
        print("4. Explore backtesting with real historical data")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()