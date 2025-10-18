"""
HFT Market Maker Configuration
"""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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
    lot_size: float = Field(default=0.001, description="Minimum quantity")
    max_position: float = Field(default=1.0, description="Maximum position size")
    gamma: float = Field(default=0.1, description="Risk aversion parameter")
    time_horizon: float = Field(default=30.0, description="Time horizon in seconds")
    min_spread: float = Field(default=0.02, description="Minimum spread")
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
    maker_fee: float = Field(default=0.001, description="Maker fee (0.1%)")
    taker_fee: float = Field(default=0.001, description="Taker fee (0.1%)")
    fill_probability: float = Field(default=0.8, description="Fill probability")


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