"""Test P&L calculation fix"""
from src.backtesting import BacktestEngine, BacktestConfig
from datetime import datetime

print("Testing P&L calculation fix...")
print("=" * 60)

engine = BacktestEngine()
config = BacktestConfig(
    symbol='BTCUSDT',
    start_date=datetime(2025, 10, 27),
    end_date=datetime(2025, 10, 29),
    initial_capital=10000
)

result = engine.run_backtest(config)
perf = result.performance

# Get trade data
if hasattr(result, 'metrics') and result.metrics:
    trade_pnls_sum = sum(trade['pnl'] for trade in result.metrics.trades)
else:
    trade_pnls_sum = 0

print("\n=== P&L CALCULATION VERIFICATION ===")
print(f"Sum of Trade P&Ls: ${trade_pnls_sum:.2f}")
print(f"Realized P&L (should match trade sum): ${perf.realized_pnl:.2f}")
print(f"Total Fees: ${perf.total_fees:.2f}")
print(f"Expected Net P&L: ${trade_pnls_sum - perf.total_fees:.2f}")
print(f"Actual Net P&L: ${perf.total_pnl:.2f}")
print(f"")

# Check if they match
if abs(perf.realized_pnl - trade_pnls_sum) < 0.01:
    print("✅ PASS: Realized P&L matches trade sum")
else:
    print(f"❌ FAIL: Realized P&L ({perf.realized_pnl:.2f}) != Trade sum ({trade_pnls_sum:.2f})")
    print(f"   Discrepancy: ${abs(perf.realized_pnl - trade_pnls_sum):.2f}")

expected_net = trade_pnls_sum - perf.total_fees
if abs(perf.total_pnl - expected_net) < 0.01:
    print("✅ PASS: Net P&L calculation correct")
else:
    print(f"❌ FAIL: Net P&L ({perf.total_pnl:.2f}) != Expected ({expected_net:.2f})")
    print(f"   Discrepancy: ${abs(perf.total_pnl - expected_net):.2f}")

print(f"")
print("=== POSITION CONTROL VERIFICATION ===")
if hasattr(result, 'metrics') and result.metrics:
    max_position = max(abs(pos) for _, pos in result.metrics.position_series) if result.metrics.position_series else 0
    position_limit = 0.05  # From BacktestConfig
    print(f"Max Position Reached: {max_position:.4f} BTC")
    print(f"Position Limit (70%): {position_limit * 0.70:.4f} BTC")
    print(f"Hard Limit: {position_limit:.4f} BTC")
    
    if max_position <= position_limit * 0.70:
        print("✅ PASS: Position stayed within 70% limit")
    elif max_position <= position_limit:
        print("⚠️  WARNING: Position exceeded 70% but within hard limit")
    else:
        print(f"❌ FAIL: Position ({max_position:.4f}) exceeded hard limit ({position_limit:.4f})")

print(f"")
print("=== PROFITABILITY CHECK ===")
if perf.total_pnl > 0:
    print(f"✅ PROFITABLE: ${perf.total_pnl:.2f} net profit")
    print(f"   Return: {perf.total_return_pct:.2f}%")
else:
    print(f"❌ UNPROFITABLE: ${perf.total_pnl:.2f} net loss")
    print(f"   Return: {perf.total_return_pct:.2f}%")
    
print(f"")
print(f"Total Trades: {perf.total_trades}")
print(f"Fill Rate: {perf.fill_rate * 100:.1f}%")
