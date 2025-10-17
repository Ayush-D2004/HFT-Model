"""
Professional Grade HFT Market Maker Strategy
===========================================

A high-frequency trading system implementing market making strategies using the Avellaneda-Stoikov framework.
Designed for professional trading with real-time order book management, risk controls, and backtesting capabilities.

Project Structure:
- src/data_ingestion: Binance WebSocket connectivity and order book management
- src/strategy: Avellaneda-Stoikov market making implementation
- src/backtesting: Historical data replay and performance evaluation
- src/dashboard: Real-time Streamlit dashboard with analytics
- src/utils: Common utilities and configuration management
"""

__version__ = "1.0.0"
__author__ = "HFT Trading Team"
__license__ = "Private"

# Core imports for the HFT system
from src.strategy.avellaneda_stoikov import AvellanedaStoikovPricer
from src.data_ingestion.binance_connector import BinanceConnector
from src.data_ingestion.order_book import OrderBook

__all__ = [
    "AvellanedaStoikovPricer",
    "BinanceConnector", 
    "OrderBook"
]