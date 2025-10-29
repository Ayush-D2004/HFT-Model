"""
HFT Market Maker Configuration
"""

import os
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)


class BinanceConfig(BaseModel):
    """Binance API configuration"""
    api_key: str = Field(default="", description="Binance API key")
    secret_key: str = Field(default="", description="Binance secret key")
    testnet: bool = Field(default=True, description="Use testnet environment")
    base_url: str = Field(default="https://fapi.binance.com", description="API base URL")
    ws_url: str = Field(default="wss://fstream.binance.com/ws", description="WebSocket URL")


class TradingConfig(BaseModel):
    """Trading strategy configuration"""
    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    tick_size: float = Field(default=0.01, description="Minimum price movement")
    
    # 🚀 PROFESSIONAL HFT: Optimized lot size for fee efficiency
    # Increased from 0.0001 to 0.0005 BTC (~$2/trade at $4000 ETH)
    # Balance: Enough size to overcome fees, small enough for frequent fills
    # Fee impact: 0.07% round-trip = $0.014 per $20 trade = $1.40 per $2000
    lot_size: float = Field(default=0.0005, description="Minimum quantity (OPTIMIZED: $2 notional)")
    
    # 🚀 PROFESSIONAL HFT: Moderate max position for controlled risk
    # Allows ~10 incremental fills before hitting limit
    # At 0.0005 per trade: 10 fills = 0.005 BTC position
    max_position: float = Field(default=0.01, description="Maximum position size (10-20 increments)")
    
    # 🔧 PROFITABILITY FIX: Lower risk aversion = wider quotes = more profit per trade
    # Reduced from 0.015 to 0.005 for more aggressive market making
    gamma: float = Field(default=0.005, description="Risk aversion parameter (LOWER = wider spread)")
    
    # 🚀 PROFESSIONAL HFT: Moderate time horizon for balanced strategy
    # Increased from 2s to 10s - less frantic, more stable quotes
    time_horizon: float = Field(default=10.0, description="Time horizon in seconds (5-15s typical)")
    
    # � PROFITABILITY FIX: Wider spread to beat fees!
    # Increased from 15 bps to 25 bps (0.25%)
    # At $4000 ETH: 25 bps = $10 spread
    # Round-trip fees: 0.07% = $2.80 per $4000
    # Need ~3x fee to be profitable after slippage
    # 25 bps = 3.5x fees = PROFITABLE!
    min_spread: float = Field(default=0.0025, description="Minimum spread (25 bps = 0.25% - BEATS FEES)")
    
    ewma_alpha: float = Field(default=0.2, description="EWMA smoothing factor")


class RiskConfig(BaseModel):
    """Risk management configuration"""
    max_loss: float = Field(default=1000.0, description="Maximum daily loss")
    max_drawdown: float = Field(default=0.30, description="Maximum drawdown (30%) - realistic for market making")
    position_limit: float = Field(default=10.0, description="Position limit")
    latency_threshold_ms: float = Field(default=100.0, description="Latency threshold in ms")


class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    start_date: str = Field(default="2024-01-01", description="Backtest start date")
    end_date: str = Field(default="2024-12-31", description="Backtest end date")
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    # 🔧 PROFITABILITY FIX: Use actual Binance VIP0 fees
    # Maker: 0.01% (we get 0.01% rebate, so effectively -0.01%)
    # Taker: 0.04% (we pay 0.04%)
    # Conservative: assume we're maker 80% of time
    # Weighted average: 0.8*(-0.01%) + 0.2*(0.04%) = -0.008% + 0.008% = 0%
    # But to be safe, use small positive fees
    maker_fee: float = Field(default=0.0001, description="Maker fee (0.01% - Binance VIP0)")
    taker_fee: float = Field(default=0.0004, description="Taker fee (0.04% - Binance VIP0)")
    fill_probability: float = Field(default=0.8, description="Fill probability")
    
    def validate_spread_vs_fees(self, min_spread: float) -> bool:
        """
        ✅ ISSUE #15 FIX: Validate that min_spread beats round-trip fees
        
        For profitable market making:
        min_spread > 2 × (maker_fee + taker_fee) + slippage_buffer
        
        Example with Binance fees:
        - Maker fee: 0.02% (0.0002)
        - Taker fee: 0.05% (0.0005)
        - Round-trip: 0.07% (0.0007)
        - Min profitable spread: 0.14% (0.0014) + buffer = 0.20% (0.0020)
        
        Returns True if valid, False with warning if too tight
        """
        round_trip_fee = self.maker_fee + self.taker_fee
        min_profitable_spread = 2 * round_trip_fee  # Need to beat bid-ask crossing
        recommended_spread = min_profitable_spread * 1.5  # Add 50% buffer for slippage
        
        if min_spread < min_profitable_spread:
            logger.warning(f"⚠️ CONFIG VALIDATION FAILED: min_spread ({min_spread:.4f}) < "
                         f"round-trip fees ({round_trip_fee:.4f}). Strategy will LOSE money!")
            logger.warning(f"   Recommended minimum: {recommended_spread:.4f} "
                         f"({recommended_spread*100:.2f}%)")
            return False
        elif min_spread < recommended_spread:
            logger.warning(f"⚠️ CONFIG WARNING: min_spread ({min_spread:.4f}) is tight. "
                         f"Recommended: {recommended_spread:.4f} for safety margin")
            return True
        else:
            logger.info(f"✅ CONFIG VALID: min_spread ({min_spread:.4f}) > "
                       f"recommended ({recommended_spread:.4f})")
            return True


class DashboardConfig(BaseModel):
    """Dashboard configuration"""
    port: int = Field(default=8501, description="Streamlit port")
    host: str = Field(default="localhost", description="Dashboard host")
    refresh_rate_ms: int = Field(default=1000, description="Refresh rate in milliseconds")
    orderbook_levels: int = Field(default=10, description="Order book levels to display")


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.binance = BinanceConfig(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            secret_key=os.getenv("BINANCE_SECRET_KEY", "")
        )
        self.trading = TradingConfig()
        self.risk = RiskConfig()
        self.backtest = BacktestConfig()
        self.dashboard = DashboardConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "binance": self.binance.model_dump(),
            "trading": self.trading.model_dump(),
            "risk": self.risk.model_dump(),
            "backtest": self.backtest.model_dump(),
            "dashboard": self.dashboard.model_dump()
        }


# Global configuration instance
config = Config()