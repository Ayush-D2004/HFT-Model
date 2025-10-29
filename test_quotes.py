"""Test quote submission rate"""
from src.backtesting import BacktestEngine, BacktestConfig
from datetime import datetime

print("Testing quote submission rate...")

engine = BacktestEngine()
config = BacktestConfig(
    symbol='BTCUSDT',
    start_date=datetime(2025, 10, 27),
    end_date=datetime(2025, 10, 29),
    initial_capital=10000
)

result = engine.run_backtest(config)
perf = result.performance

# Get data from metrics
if hasattr(result, 'metrics') and result.metrics:
    data_points = len(result.metrics.pnl_series)
    quote_updates = result.metrics.quote_updates
    quote_hits = result.metrics.quote_hits
else:
    data_points = 0
    quote_updates = 0
    quote_hits = 0

print(f"\n\n=== QUOTE SUBMISSION ANALYSIS ===")
print(f"=" * 60)
print(f"Data Points Processed: {data_points}")
print(f"Quote Updates Submitted: {quote_updates}")
print(f"Quote Hits (Filled): {quote_hits}")
print(f"Total Trades/Fills: {perf.total_trades}")
print(f"")
print(f"Quote Submission Rate: {(quote_updates / max(data_points, 1)) * 100:.1f}%")
print(f"Fill Rate: {perf.fill_rate * 100:.1f}%")
print(f"")
print(f"PROBLEM: Should be ~100%% submission rate for HFT!")
print(f"TARGET: {data_points} quotes for {data_points} data points")
print(f"ACTUAL: {quote_updates} quotes ({(quote_updates/max(data_points,1))*100:.1f}% of target)")
print(f"")
print(f"ROOT CAUSE: Risk manager blocking quotes when position > 30% of max")
