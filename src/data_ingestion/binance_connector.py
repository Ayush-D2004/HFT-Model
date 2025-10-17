"""
Professional Binance WebSocket Connector for HFT Trading
========================================================

High-performance WebSocket client with:
- Auto-reconnection and error handling
- Order book and trade stream management  
- Low latency message processing
- Comprehensive logging and monitoring
"""

import asyncio
import json
import time
from typing import Callable, Dict, List, Optional, Set, Any
from dataclasses import dataclass
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from loguru import logger
import threading
from urllib.parse import urlencode

from .order_book import OrderBook
from ..utils.config import config


@dataclass
class StreamConfig:
    """Configuration for WebSocket streams"""
    symbol: str
    streams: List[str]
    callback: Callable[[dict], None]
    
    
class BinanceConnector:
    """
    Professional Binance WebSocket connector with enterprise-grade features:
    - Multiple stream management with automatic subscription
    - Robust reconnection logic with exponential backoff
    - Order book synchronization with REST API fallback
    - Performance monitoring and statistics
    - Thread-safe operations for concurrent access
    """
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 base_url: str = None,
                 ws_url: str = None,
                 api_key: str = None,
                 secret_key: str = None):
        
        self.symbol = symbol.upper()
        self.base_url = base_url or config.binance.base_url
        self.ws_url = ws_url or config.binance.ws_url
        self.api_key = api_key or config.binance.api_key
        self.secret_key = secret_key or config.binance.secret_key
        
        # WebSocket connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.is_running = False
        self.reconnect_count = 0
        
        # Stream management
        self.active_streams: Set[str] = set()
        self.stream_callbacks: Dict[str, Callable] = {}
        
        # Order book instance
        self.order_book = OrderBook(self.symbol, max_levels=config.trading.tick_size)
        
        # Event loop and threading
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        
        # Performance and monitoring
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'connection_errors': 0,
            'reconnections': 0,
            'last_message_time': 0.0,
            'avg_latency_ms': 0.0
        }
        
        # Reconnection settings
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1.0  # Start with 1 second
        self.max_reconnect_delay = 60.0  # Max 60 seconds
        
        logger.info(f"BinanceConnector initialized for {self.symbol}")
    
    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            # Build WebSocket URL with streams
            streams = self._get_default_streams()
            stream_names = '/'.join(streams)
            ws_url = f"{self.ws_url}/{stream_names}"
            
            logger.info(f"Connecting to Binance WebSocket: {ws_url}")
            
            # Connect with timeout and specific headers
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.reconnect_count = 0
            self.reconnect_delay = 1.0  # Reset delay
            
            logger.success(f"WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.is_connected = False
            return False
    
    def _get_default_streams(self) -> List[str]:
        """Get default streams for the symbol"""
        symbol_lower = self.symbol.lower()
        return [
            f"{symbol_lower}@depth20@100ms",  # Order book depth
            f"{symbol_lower}@trade",          # Individual trades
            f"{symbol_lower}@ticker"          # 24hr ticker statistics
        ]
    
    async def disconnect(self):
        """Gracefully disconnect WebSocket"""
        try:
            self.is_running = False
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
                
            self.is_connected = False
            self.websocket = None
            
            logger.info("WebSocket disconnected")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def _message_handler(self):
        """Main message processing loop"""
        while self.is_running and self.websocket:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=30.0
                )
                
                self.stats['messages_received'] += 1
                self.stats['last_message_time'] = time.time()
                
                # Parse and process message
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout")
                break
                
            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                break
                
            except WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                break
                
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                self.stats['connection_errors'] += 1
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle different stream types
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                payload = data['data']
                
                # Route to appropriate handler
                if '@depth' in stream:
                    await self._handle_depth_update(payload)
                elif '@trade' in stream:
                    await self._handle_trade_update(payload)
                elif '@ticker' in stream:
                    await self._handle_ticker_update(payload)
                else:
                    logger.debug(f"Unknown stream type: {stream}")
            
            # Handle single stream format (no 'stream' field)
            elif 'e' in data:  # Event type field
                event_type = data['e']
                if event_type == 'depthUpdate':
                    await self._handle_depth_update(data)
                elif event_type == 'trade':
                    await self._handle_trade_update(data)
                elif event_type == '24hrTicker':
                    await self._handle_ticker_update(data)
            
            self.stats['messages_processed'] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_depth_update(self, data: dict):
        """Handle order book depth update"""
        try:
            # Check if this is a snapshot or update
            if 'lastUpdateId' in data:
                # This is a snapshot
                success = self.order_book.handle_snapshot(data)
                if success:
                    logger.debug("Order book snapshot processed")
            else:
                # This is an incremental update
                success = self.order_book.handle_update(data)
                if not success:
                    # Request new snapshot if update failed
                    await self._request_order_book_snapshot()
                    
        except Exception as e:
            logger.error(f"Error handling depth update: {e}")
    
    async def _handle_trade_update(self, data: dict):
        """Handle individual trade update"""
        try:
            # Extract trade information
            trade_info = {
                'symbol': data.get('s', self.symbol),
                'price': float(data.get('p', 0)),
                'quantity': float(data.get('q', 0)),
                'timestamp': data.get('T', int(time.time() * 1000)),
                'is_buyer_maker': data.get('m', False),
                'trade_id': data.get('t', 0)
            }
            
            # Register trade with order book for arrival rate estimation
            timestamp = trade_info['timestamp'] / 1000.0  # Convert to seconds
            self.order_book.register_trade_event(timestamp)
            
            logger.debug(f"Trade: {trade_info['price']} @ {trade_info['quantity']}")
            
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
    
    async def _handle_ticker_update(self, data: dict):
        """Handle 24hr ticker statistics"""
        try:
            ticker_info = {
                'symbol': data.get('s', self.symbol),
                'price_change': float(data.get('p', 0)),
                'price_change_percent': float(data.get('P', 0)),
                'weighted_avg_price': float(data.get('w', 0)),
                'prev_close_price': float(data.get('x', 0)),
                'last_price': float(data.get('c', 0)),
                'last_qty': float(data.get('Q', 0)),
                'bid_price': float(data.get('b', 0)),
                'ask_price': float(data.get('a', 0)),
                'open_price': float(data.get('o', 0)),
                'high_price': float(data.get('h', 0)),
                'low_price': float(data.get('l', 0)),
                'volume': float(data.get('v', 0)),
                'quote_volume': float(data.get('q', 0)),
                'count': int(data.get('c', 0))
            }
            
            logger.debug(f"Ticker: {ticker_info['last_price']} "
                        f"({ticker_info['price_change_percent']:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error handling ticker update: {e}")
    
    async def _request_order_book_snapshot(self):
        """Request fresh order book snapshot via REST API"""
        try:
            import aiohttp
            
            url = f"{self.base_url}/api/v3/depth"
            params = {'symbol': self.symbol, 'limit': 1000}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.order_book.handle_snapshot(data)
                        logger.info("Order book snapshot refreshed via REST API")
                    else:
                        logger.error(f"Failed to get snapshot: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error requesting snapshot: {e}")
    
    async def _reconnect_loop(self):
        """Handle reconnection with exponential backoff"""
        while self.is_running and self.reconnect_count < self.max_reconnect_attempts:
            try:
                logger.info(f"Attempting reconnection #{self.reconnect_count + 1}")
                
                # Wait before reconnecting (exponential backoff)
                await asyncio.sleep(self.reconnect_delay)
                
                # Attempt to connect
                if await self.connect():
                    logger.success("Reconnection successful")
                    self.stats['reconnections'] += 1
                    return True
                
                # Increase delay for next attempt
                self.reconnect_count += 1
                self.reconnect_delay = min(
                    self.reconnect_delay * 2, 
                    self.max_reconnect_delay
                )
                
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
                self.reconnect_count += 1
        
        logger.error("Max reconnection attempts reached")
        return False
    
    async def run(self):
        """Main run loop with automatic reconnection"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Connect if not connected
                if not self.is_connected:
                    if not await self.connect():
                        await self._reconnect_loop()
                        continue
                
                # Request initial order book snapshot
                await self._request_order_book_snapshot()
                
                # Start message processing
                await self._message_handler()
                
            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                self.is_connected = False
                
                if self.is_running:  # Only reconnect if still supposed to run
                    await self._reconnect_loop()
    
    def start(self):
        """Start the connector in a separate thread"""
        if self.thread and self.thread.is_alive():
            logger.warning("Connector already running")
            return
        
        def run_in_thread():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            
            try:
                loop.run_until_complete(self.run())
            except Exception as e:
                logger.error(f"Error in connector thread: {e}")
            finally:
                loop.close()
        
        self.thread = threading.Thread(target=run_in_thread, daemon=True)
        self.thread.start()
        
        logger.info("BinanceConnector started in background thread")
    
    def stop(self):
        """Stop the connector"""
        self.is_running = False
        
        if self.loop and not self.loop.is_closed():
            # Schedule disconnect in the event loop
            asyncio.run_coroutine_threadsafe(self.disconnect(), self.loop)
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        logger.info("BinanceConnector stopped")
    
    def get_order_book(self) -> OrderBook:
        """Get the order book instance"""
        return self.order_book
    
    def get_statistics(self) -> Dict:
        """Get connector statistics"""
        return {
            **self.stats,
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'reconnect_count': self.reconnect_count,
            'symbol': self.symbol,
            'order_book_stats': self.order_book.get_statistics()
        }
    
    def is_healthy(self) -> bool:
        """Check if connector is in healthy state"""
        return (
            self.is_connected and 
            self.is_running and 
            self.order_book.is_healthy() and
            (time.time() - self.stats['last_message_time']) < 30.0  # Recent activity
        )


# Example usage for testing
if __name__ == "__main__":
    async def test_connector():
        connector = BinanceConnector("BTCUSDT")
        
        try:
            # Start connector
            connector.start()
            
            # Let it run for a bit
            await asyncio.sleep(30)
            
            # Check statistics
            stats = connector.get_statistics()
            print(f"Statistics: {stats}")
            
            # Get current order book
            snapshot = connector.get_order_book().get_snapshot()
            print(f"Midprice: {snapshot.midprice()}")
            print(f"Spread: {snapshot.spread()}")
            
        finally:
            connector.stop()
    
    # Run test
    asyncio.run(test_connector())