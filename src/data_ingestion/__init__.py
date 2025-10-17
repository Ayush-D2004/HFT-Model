"""
Data Ingestion Module for HFT Trading System
==========================================

Professional-grade market data ingestion with:
- Binance WebSocket connectivity
- Order book maintenance with U/u sequencing
- Low latency message processing
- Robust error handling and reconnection
"""

from .order_book import OrderBook, OrderBookSnapshot, OrderBookLevel
from .binance_connector import BinanceConnector

__all__ = [
    'OrderBook',
    'OrderBookSnapshot', 
    'OrderBookLevel',
    'BinanceConnector'
]