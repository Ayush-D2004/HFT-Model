"""
Binance WebSocket Manager
========================

Real-time WebSocket connection to Binance for market data streaming.
Handles order book updates, trade streams, and connection management.
"""

import asyncio
import websockets
import json
import time
from typing import Dict, Callable, Optional
import ssl

from ..utils.logger import get_logger


class BinanceWebSocketManager:
    """
    Manages WebSocket connections to Binance for real-time market data.
    
    Features:
    - Order book depth streams
    - Trade data streams  
    - Automatic reconnection
    - Connection health monitoring
    """
    
    def __init__(self, 
                 symbol: str,
                 on_orderbook_update: Callable[[Dict], None],
                 on_trade_update: Callable[[Dict], None],
                 on_connection_change: Callable[[bool], None],
                 testnet: bool = False):  # Always use live data
        
        self.symbol = symbol.lower()
        self.testnet = False  # Force live data connection
        self.logger = get_logger('websocket_manager')
        
        # Callbacks
        self.on_orderbook_update = on_orderbook_update
        self.on_trade_update = on_trade_update
        self.on_connection_change = on_connection_change
        
        # Connection state
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        
        # WebSocket URLs - Using the working Binance WebSocket endpoint from existing connector
        # Use spot trading WebSocket (not futures)
        self.ws_base = "wss://stream.binance.com:9443/ws"
        
        # Ensure symbol is lowercase 
        self.symbol = symbol.lower()
        
        # Use order book depth stream (most important for market making)
        # Format: wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms
        self.stream_url = f"{self.ws_base}/{self.symbol}@depth20@100ms"
        
        # Message tracking
        self.last_message_time = time.time()
        self.message_count = 0
        
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.logger.info(f"Connecting to LIVE Binance WebSocket for {self.symbol.upper()}")
            self.logger.info(f"Using URL: {self.stream_url}")
            
            # Create SSL context for secure connection
            ssl_context = ssl.create_default_context()
            
            # Connect to WebSocket with the stream URL
            self.websocket = await websockets.connect(
                self.stream_url,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            self.on_connection_change(True)
            
            self.logger.info(f"WebSocket connected successfully to LIVE Binance data")
            
            # Start message processing
            await self._message_handler()
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False
            self.on_connection_change(False)
            
            # Attempt reconnection
            await self._handle_reconnection()
    
    async def disconnect(self):
        """Close WebSocket connection"""
        try:
            self.logger.info("Disconnecting WebSocket...")
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            
            self.is_connected = False
            self.on_connection_change(False)
            
            self.logger.info("WebSocket disconnected")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                self.last_message_time = time.time()
                self.message_count += 1
                
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse WebSocket message: {e}")
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.is_connected = False
            self.on_connection_change(False)
            await self._handle_reconnection()
            
        except Exception as e:
            self.logger.error(f"Error in message handler: {e}")
            self.is_connected = False
            self.on_connection_change(False)
            await self._handle_reconnection()
    
    async def _process_message(self, data: Dict):
        """Process individual WebSocket messages"""
        try:
            # Check if this is a depth stream message (has bids/asks directly)
            if 'bids' in data and 'asks' in data:
                # Direct order book depth update (no wrapper)
                await self._process_orderbook_update(data)
                
            elif 'stream' in data:
                # Wrapped stream message
                stream = data['stream']
                if 'depth' in stream:
                    # Order book depth update
                    await self._process_orderbook_update(data['data'])
                elif 'trade' in stream:
                    # Trade update
                    await self._process_trade_update(data['data'])
                else:
                    self.logger.debug(f"Unknown stream type: {stream}")
            else:
                self.logger.debug(f"Unknown message format: {list(data.keys())}")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    async def _process_orderbook_update(self, data: Dict):
        """Process order book depth updates"""
        try:
            # Handle both direct and wrapped formats
            if 's' in data:
                # Wrapped format from combined streams
                symbol = data['s']
                bids = data['bids']
                asks = data['asks']
            else:
                # Direct format from single stream
                symbol = self.symbol.upper()
                bids = data['bids']
                asks = data['asks']
            
            # Extract order book data
            orderbook_data = {
                'symbol': symbol,
                'timestamp': time.time(),
                'bids': [[float(price), float(qty)] for price, qty in bids],
                'asks': [[float(price), float(qty)] for price, qty in asks],
                'source': 'BINANCE_WEBSOCKET'
            }
            
            # Calculate best bid/ask and spread
            if orderbook_data['bids'] and orderbook_data['asks']:
                best_bid = orderbook_data['bids'][0][0]
                best_ask = orderbook_data['asks'][0][0]
                orderbook_data['best_bid'] = best_bid
                orderbook_data['best_ask'] = best_ask
                orderbook_data['spread'] = best_ask - best_bid
                orderbook_data['midprice'] = (best_bid + best_ask) / 2
            
            # Call callback
            self.on_orderbook_update(orderbook_data)
            
        except Exception as e:
            self.logger.error(f"Error processing orderbook update: {e}")
    
    async def _process_trade_update(self, data: Dict):
        """Process individual trade updates"""
        try:
            # Extract trade data
            trade_data = {
                'symbol': data['s'],
                'timestamp': float(data['T']) / 1000,  # Convert to seconds
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': data['m'],
                'trade_id': data['t'],
                'source': 'BINANCE_WEBSOCKET'
            }
            
            # Call callback
            self.on_trade_update(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error processing trade update: {e}")
    
    async def _handle_reconnection(self):
        """Handle WebSocket reconnection logic"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return
        
        self.reconnect_attempts += 1
        self.logger.info(f"Attempting reconnection ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
        
        # Wait before reconnecting
        await asyncio.sleep(self.reconnect_delay)
        
        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 1.5, 60)
        
        # Attempt reconnection
        await self.connect()
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            'is_connected': self.is_connected,
            'message_count': self.message_count,
            'last_message_time': self.last_message_time,
            'reconnect_attempts': self.reconnect_attempts,
            'symbol': self.symbol.upper(),
            'stream_url': self.stream_url
        }