"""Quick test of optimized backtest parameters"""
from src.backtesting import BacktestEngine, BacktestConfig
from datetime import datetime

# Run backtest
engine = BacktestEngine()
config = BacktestConfig(
    symbol='ETHUSDT',
    start_date=datetime(2025, 10, 22),
    end_date=datetime(2025, 10, 23),
    initial_capital=10000
)

result = engine.run_backtest(config)
perf = result.performance

# Display results with proper None handling
print("\n\n=== OPTIMIZED BACKTEST RESULTS ===")
print(f"Total Trades: {perf.total_trades}")
print(f"Fill Rate: {perf.fill_rate*100:.1f}%")

# Handle potential None values
realized_pnl = getattr(perf, 'realized_pnl', None)
total_fees = getattr(perf, 'total_fees', None)
total_pnl = getattr(perf, 'total_pnl', None)

print(f"Realized P&L (Gross): ${realized_pnl:.2f}" if realized_pnl is not None else "Realized P&L (Gross): N/A")
print(f"Total Fees: ${total_fees:.2f}" if total_fees is not None else "Total Fees: N/A")

if realized_pnl is not None and total_fees is not None:
    net_pnl = realized_pnl - total_fees
    print(f"Realized P&L (Net): ${net_pnl:.2f}")
else:
    print(f"Realized P&L (Net): N/A")

print(f"Total P&L: ${total_pnl:.2f}" if total_pnl is not None else "Total P&L: N/A")
print(f"Total Return: {perf.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {perf.sharpe_ratio:.2f}")
print(f"Max Drawdown: {perf.max_drawdown*100:.2f}%")

# Check which lot_size is being used
print("\n=== CONFIG VERIFICATION ===")
from src.utils.config import StrategyConfig
cfg = StrategyConfig()
print(f"✅ Config file lot_size: {cfg.lot_size}")
print(f"✅ Config file min_spread: {cfg.min_spread}")
print(f"✅ Config file gamma: {cfg.gamma}")
print(f"✅ Config file maker_fee: {cfg.maker_fee}")
print(f"✅ Config file taker_fee: {cfg.taker_fee}")
