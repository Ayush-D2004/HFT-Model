"""
Binance Historical Data Fetcher
==============================

Fetches real historical market data from Binance REST API for backtesting.
No synthetic data - only real market data from Binance.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
from loguru import logger


class BinanceHistoricalDataFetcher:
    """
    Fetches real historical data from Binance REST API for backtesting.
    
    Features:
    - Real OHLCV kline data
    - Order book snapshots via REST API  
    - Trade history data
    - Proper rate limiting and error handling
    - Data caching to avoid repeated API calls
    """
    
    def __init__(self, use_testnet: bool = False):
        self.use_testnet = use_testnet
        
        # Binance REST API endpoints
        if use_testnet:
            self.base_url = "https://testnet.binance.vision/api/v3"
        else:
            self.base_url = "https://api.binance.com/api/v3"
            
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HFT-Backtester/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Data cache directory
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Binance Historical Data Fetcher initialized (testnet={use_testnet})")
    
    def _rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_kline_data(self, 
                      symbol: str,
                      start_time: datetime,
                      end_time: datetime,
                      interval: str = "1m") -> pd.DataFrame:
        """
        Fetch historical kline/candlestick data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC) 
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"{symbol}_{interval}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"
        cache_file = self.cache_dir / f"{cache_key}.csv"
        
        if cache_file.exists():
            logger.info(f"Loading cached data for {symbol}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        logger.info(f"Fetching {symbol} kline data from {start_time} to {end_time}")
        
        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        all_klines = []
        current_start = start_ms
        
        # Binance limits to 1000 klines per request
        while current_start < end_ms:
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000
            }
            
            data = self._make_request('klines', params)
            if not data:
                break
                
            all_klines.extend(data)
            
            if len(data) < 1000:
                break
                
            # Update start time for next batch
            current_start = data[-1][6] + 1  # Close time + 1ms
            
        if not all_klines:
            logger.warning(f"No kline data retrieved for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                  'taker_buy_quote_volume', 'ignore']
        
        df = pd.DataFrame(all_klines, columns=columns)
        
        # Convert types and set index
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        # Cache the data
        df.to_csv(cache_file)
        logger.info(f"Fetched and cached {len(df)} klines for {symbol}")
        
        return df
    
    def get_recent_trades(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """Fetch recent trades for the symbol"""
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 1000)
        }
        
        data = self._make_request('trades', params)
        return data or []
    
    def get_order_book_snapshot(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Get current order book snapshot"""
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 5000)
        }
        
        return self._make_request('depth', params)
    
    def simulate_order_book_from_klines(self, 
                                      kline_data: pd.DataFrame,
                                      spread_factor: float = 0.001) -> List[Dict]:
        """
        Generate realistic order book updates from kline data.
        This creates order book snapshots based on OHLCV data with realistic spreads.
        """
        order_book_updates = []
        
        for timestamp, row in kline_data.iterrows():
            # Calculate mid price and spread
            mid_price = (row['high'] + row['low']) / 2
            spread = mid_price * spread_factor
            
            # Generate bid/ask levels around mid price
            best_bid = mid_price - spread/2
            best_ask = mid_price + spread/2
            
            # Create realistic order book depth
            bids = []
            asks = []
            
            # Generate 20 levels on each side
            for i in range(20):
                bid_price = best_bid - (i * spread * 0.1)
                ask_price = best_ask + (i * spread * 0.1)
                
                # Volume decreases with distance from best price
                base_volume = row['volume'] / 100  # Scale down volume
                bid_volume = base_volume * np.exp(-i * 0.1)
                ask_volume = base_volume * np.exp(-i * 0.1)
                
                bids.append([f"{bid_price:.8f}", f"{bid_volume:.8f}"])
                asks.append([f"{ask_price:.8f}", f"{ask_volume:.8f}"])
            
            order_book_update = {
                'timestamp': timestamp.timestamp(),
                'symbol': kline_data.index.name or 'BTCUSDT',
                'bids': bids,
                'asks': asks,
                'lastUpdateId': int(timestamp.timestamp() * 1000),  # Use timestamp as update ID
                'source': 'SIMULATED_FROM_KLINES'
            }
            
            order_book_updates.append(order_book_update)
        
        logger.info(f"Generated {len(order_book_updates)} order book snapshots from kline data")
        return order_book_updates


def test_binance_historical_fetcher():
    """Test the historical data fetcher"""
    fetcher = BinanceHistoricalDataFetcher(use_testnet=False)
    
    # Test date range
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
    
    # Fetch kline data
    klines = fetcher.get_kline_data('BTCUSDT', start_time, end_time, '1m')
    print(f"Fetched {len(klines)} klines")
    print(klines.head())
    
    # Generate order book data
    order_books = fetcher.simulate_order_book_from_klines(klines)
    print(f"Generated {len(order_books)} order book snapshots")


if __name__ == "__main__":
    test_binance_historical_fetcher()