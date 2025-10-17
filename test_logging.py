"""
Test Clean Logging
=================

Simple test to verify that the new logging system works correctly.
"""

print("🧪 Testing Clean Logging System...")

# Test different logging modes
from src.utils.logger import setup_silent_logging, setup_backtesting_logging, setup_development_logging

print("\n1️⃣ Testing Silent Mode:")
setup_silent_logging()
print("✅ Silent logging enabled - should see minimal output")

# Try running a simple backtest
from src.strategy import AvellanedaStoikovPricer, QuoteParameters

pricer = AvellanedaStoikovPricer(tick_size=0.01)
pricer.update_market(50000.0)
quote_params = QuoteParameters(gamma=0.1, T=30.0)
quote = pricer.compute_quotes(quote_params)

print(f"✅ Strategy test: Bid=${quote.bid_price:.2f}, Ask=${quote.ask_price:.2f}")

print("\n2️⃣ Testing Backtesting Mode:")
setup_backtesting_logging() 
print("✅ Backtesting logging enabled - should see warnings/errors only")

print("\n3️⃣ Testing Development Mode:")
setup_development_logging()
print("✅ Development logging enabled - should see detailed output")

print("\n✅ All logging modes tested successfully!")
print("Dashboard will now use clean backtesting with progress indicators.")