"""
Live Trading Engine
==================

Main engine for live trading operations with real-time data processing,
strategy execution, and risk management.
"""

import time
import asyncio
import threading
from typing import Dict, Optional, Callable, Any
from datetime import datetime
import queue
import json
from pathlib import Path

from .websocket_manager import BinanceWebSocketManager
from .live_strategy import LiveAvellanedaStoikov
from .performance_tracker import LivePerformanceTracker
from ..utils.logger import get_logger


class LiveTradingEngine:
    """
    Main live trading engine that coordinates:
    - Real-time data ingestion via WebSocket
    - Strategy execution with Avellaneda-Stoikov
    - Performance tracking and risk management
    - State persistence and recovery
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger('live_trading')
        
        # Trading state
        self.is_running = False
        self.is_connected = False
        self.start_time = None
        self.trade_count = 0
        
        # Components
        self.websocket_manager = None
        self.strategy = None
        self.performance_tracker = None
        
        # Data queues for real-time processing
        self.market_data_queue = queue.Queue(maxsize=1000)
        self.order_queue = queue.Queue(maxsize=100)
        
        # Callbacks for dashboard updates
        self.callbacks = {
            'on_market_data': [],
            'on_trade_executed': [],
            'on_performance_update': [],
            'on_error': [],
            'on_status_change': []
        }
        
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            self.logger.info(f"Added callback for {event_type}")
    
    def remove_callback(self, event_type: str, callback: Callable):
        """Remove callback for specific events"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def _emit_event(self, event_type: str, data: Any):
        """Emit event to all registered callbacks"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error for {event_type}: {e}")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing live trading engine...")
            
            # Initialize strategy
            self.strategy = LiveAvellanedaStoikov(
                symbol=self.config['symbol'],
                tick_size=self.config['tick_size'],
                lot_size=self.config['lot_size'],
                gamma=self.config['gamma'],
                time_horizon=self.config['time_horizon']
            )
            
            # Initialize performance tracker
            self.performance_tracker = LivePerformanceTracker(
                initial_balance=self.config.get('initial_balance', 10000.0)
            )
            
            # Initialize WebSocket manager
            self.websocket_manager = BinanceWebSocketManager(
                symbol=self.config['symbol'],
                on_orderbook_update=self._on_orderbook_update,
                on_trade_update=self._on_trade_update,
                on_connection_change=self._on_connection_change
            )
            
            self.logger.info("Live trading engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize live trading engine: {e}")
            raise
    
    async def start_trading(self):
        """Start live trading - simplified for direct processing"""
        if self.is_running:
            self.logger.warning("Trading already running")
            return
        
        try:
            self.logger.info("🚀 Starting live trading with DIRECT processing (HFT mode)...")
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start WebSocket connection (this handles everything via callbacks)
            await self.websocket_manager.connect()
            
            # Start only performance updater as background task
            self.logger.info("Starting performance updater...")
            performance_task = asyncio.create_task(self._performance_updater())
            
            self.logger.info("✅ Live trading started - processing market data directly via callbacks")
            
            # Keep running and wait for performance task
            try:
                await performance_task
            except Exception as e:
                self.logger.error(f"Error in performance updater: {e}")
            
        except Exception as e:
            self.logger.error(f"Error starting live trading: {e}")
            self.is_running = False
            self._emit_event('on_error', str(e))
            raise
    
    def stop_trading(self):
        """Stop live trading (synchronous wrapper)"""
        if not self.is_running:
            self.logger.warning("Trading not running")
            return
        
        self.logger.info("Stopping live trading...")
        self.is_running = False
        
        # Disconnect WebSocket - handle both sync and async contexts
        if self.websocket_manager:
            try:
                # Try to get running loop
                loop = asyncio.get_running_loop()
                # If we're in an async context, create a task
                asyncio.create_task(self.websocket_manager.disconnect())
            except RuntimeError:
                # No running loop - we're in sync context (Streamlit)
                # Run disconnect in a new event loop
                try:
                    asyncio.run(self.websocket_manager.disconnect())
                except Exception as e:
                    self.logger.warning(f"Error disconnecting WebSocket: {e}")
        
        # Save final state
        self._save_trading_state()
        
        self._emit_event('on_status_change', {
            'status': 'stopped',
            'runtime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'total_trades': self.trade_count
        })
        
        self.logger.info("Live trading stopped")
    
    async def _market_data_processor(self):
        """Process incoming market data and generate trading signals"""
        self.logger.info("🚀 Market data processor started - will process queued data")
        processed_count = 0
        last_log_time = time.time()
        
        while self.is_running:
            try:
                # Log processor status every 5 seconds
                if time.time() - last_log_time > 5:
                    queue_size = self.market_data_queue.qsize()
                    self.logger.info(f"📊 Processor alive: processed={processed_count}, queue_size={queue_size}, running={self.is_running}")
                    last_log_time = time.time()
                
                if not self.market_data_queue.empty():
                    market_data = self.market_data_queue.get_nowait()
                    processed_count += 1
                    
                    # Log every 10th processing for monitoring
                    if processed_count % 10 == 0:
                        self.logger.info(f"✅ Processing market data #{processed_count}: midprice={market_data.get('midprice', 'N/A')}")
                    
                    # Update strategy with new market data
                    quotes = self.strategy.process_market_data(market_data)
                    
                    if quotes:
                        self.logger.info(f"💰 Strategy generated quotes: BID={quotes.get('bid', 'N/A'):.4f} ASK={quotes.get('ask', 'N/A'):.4f}")
                        # Add to order queue for execution
                        self.order_queue.put({
                            'type': 'quotes',
                            'data': quotes,
                            'timestamp': time.time()
                        })
                    
                    # Emit market data event for dashboard
                    self._emit_event('on_market_data', {
                        'timestamp': market_data.get('timestamp'),
                        'symbol': market_data.get('symbol'),
                        'best_bid': market_data.get('best_bid'),
                        'best_ask': market_data.get('best_ask'),
                        'spread': market_data.get('spread'),
                        'strategy_quotes': quotes
                    })
                
                await asyncio.sleep(0.001)  # 1ms processing interval for HFT speed
                
            except Exception as e:
                self.logger.error(f"Error in market data processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _order_processor(self):
        """Process orders and simulate execution"""
        while self.is_running:
            try:
                if not self.order_queue.empty():
                    order_data = self.order_queue.get_nowait()
                    
                    # Simulate order execution (replace with real execution in production)
                    execution_result = self._simulate_order_execution(order_data)
                    
                    if execution_result:
                        # Update performance tracker
                        self.performance_tracker.add_trade(execution_result)
                        self.trade_count += 1
                        
                        # Emit trade execution event
                        self._emit_event('on_trade_executed', execution_result)
                
                await asyncio.sleep(0.0001)  # 0.1ms order processing for HFT speed
                
            except Exception as e:
                self.logger.error(f"Error in order processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _performance_updater(self):
        """Update performance metrics periodically"""
        update_count = 0
        while self.is_running:
            try:
                update_count += 1
                
                # Calculate current performance
                performance_data = self.performance_tracker.get_current_performance()
                
                # Add runtime information
                if self.start_time:
                    performance_data['runtime_seconds'] = (datetime.now() - self.start_time).total_seconds()
                
                performance_data['trade_count'] = self.trade_count
                performance_data['is_connected'] = self.is_connected
                
                # Log every 5 updates
                if update_count % 5 == 0:
                    self.logger.info(f"📊 Performance update #{update_count}: trades={self.trade_count}, callbacks={len(self.callbacks.get('on_performance_update', []))}")
                
                # Emit performance update event
                self._emit_event('on_performance_update', performance_data)
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in performance updater: {e}")
                await asyncio.sleep(5.0)
    
    def _on_orderbook_update(self, orderbook_data: Dict):
        """Handle orderbook updates from WebSocket - DIRECT PROCESSING (HFT style)"""
        try:
            if not self.is_running:
                return
            
            # Process IMMEDIATELY - no queue for HFT
            # Update strategy with new market data
            quotes = self.strategy.process_market_data(orderbook_data)
            
            if quotes:
                # Execute immediately (simulate for now)
                execution_result = self._simulate_order_execution({
                    'type': 'quotes',
                    'data': quotes,
                    'timestamp': time.time()
                })
                
                if execution_result:
                    # Update performance tracker
                    self.performance_tracker.add_trade(execution_result)
                    self.trade_count += 1
                    
                    # Emit trade execution event
                    self._emit_event('on_trade_executed', execution_result)
            
            # Emit market data event for dashboard (every 50 updates)
            if not hasattr(self, '_update_count'):
                self._update_count = 0
            self._update_count += 1
            
            if self._update_count % 50 == 0:
                self._emit_event('on_market_data', {
                    'timestamp': orderbook_data.get('timestamp'),
                    'symbol': orderbook_data.get('symbol'),
                    'best_bid': orderbook_data.get('best_bid'),
                    'best_ask': orderbook_data.get('best_ask'),
                    'spread': orderbook_data.get('spread'),
                    'midprice': orderbook_data.get('midprice'),
                    'strategy_quotes': quotes
                })
                self.logger.debug(f"📊 Processed update #{self._update_count}: mid={orderbook_data.get('midprice', 'N/A'):.2f}, trades={self.trade_count}")
        
        except Exception as e:
            self.logger.error(f"Error processing orderbook update: {e}")
    
    def _on_trade_update(self, trade_data: Dict):
        """Handle trade updates from WebSocket"""
        try:
            # Register trade event for strategy
            if self.strategy:
                self.strategy.register_trade_event(trade_data)
                
        except Exception as e:
            self.logger.error(f"Error handling trade update: {e}")
    
    def _on_connection_change(self, is_connected: bool):
        """Handle WebSocket connection status changes"""
        self.is_connected = is_connected
        self.logger.info(f"WebSocket connection status: {'Connected' if is_connected else 'Disconnected'}")
        
        self._emit_event('on_status_change', {
            'status': 'connected' if is_connected else 'disconnected',
            'is_trading': self.is_running
        })
    
    def _simulate_order_execution(self, order_data: Dict) -> Optional[Dict]:
        """
        Simulate order execution with INTELLIGENT INVENTORY MANAGEMENT.
        
        Real market making: Both bid and ask quotes are posted simultaneously.
        Fills happen based on market aggressor flow and inventory position.
        
        Professional HFT behavior:
        - When flat (inventory=0): Equal probability for buy/sell
        - When long (inventory>0): Favor SELL fills to reduce inventory
        - When short (inventory<0): Favor BUY fills to reduce inventory
        
        This minimizes zero PnL trades by ensuring balanced buy/sell execution.
        """
        try:
            if order_data['type'] == 'quotes':
                quotes = order_data['data']
                
                # Track inventory (net position) for intelligent fill simulation
                if not hasattr(self, '_inventory'):
                    self._inventory = 0.0  # Net position in BTC
                
                # Very aggressive fill simulation for HFT - 80% fill probability  
                import random
                if random.random() < 0.8:  # 80% fill probability for HFT simulation
                    
                    # INTELLIGENT SIDE SELECTION based on inventory
                    # Market makers prefer to reduce inventory (mean reversion)
                    
                    if abs(self._inventory) < 0.001:  # Flat position (< 0.001 BTC)
                        # Equal probability when neutral
                        side = random.choice(['buy', 'sell'])
                    elif self._inventory > 0:  # Long position
                        # Favor SELL to reduce inventory (70% sell, 30% buy)
                        side = 'sell' if random.random() < 0.7 else 'buy'
                    else:  # Short position
                        # Favor BUY to reduce inventory (70% buy, 30% sell)
                        side = 'buy' if random.random() < 0.7 else 'sell'
                    
                    # MARKET MAKING: You provide liquidity at YOUR quotes
                    # Your strategy generates bid/ask prices
                    # When filled:
                    #   - BUY filled at YOUR BID (you buy at your bid price)
                    #   - SELL filled at YOUR ASK (you sell at your ask price)
                    # Profit = YOUR ASK - YOUR BID (the spread you set)
                    
                    if side == 'buy':
                        price = quotes['bid']  # You buy at YOUR bid price
                        quantity = quotes.get('bid_size', self.config.get('lot_size', 0.01))
                        self._inventory += quantity  # Increase inventory
                    else:  # sell
                        price = quotes['ask']  # You sell at YOUR ask price
                        quantity = quotes.get('ask_size', self.config.get('lot_size', 0.01))
                        self._inventory -= quantity  # Decrease inventory
                    
                    execution_result = {
                        'timestamp': time.time(),
                        'side': side,
                        'price': price,
                        'quantity': quantity,
                        'value': price * quantity,
                        'type': 'market_making',
                        'latency_ms': random.uniform(0.1, 2.0),  # Simulate HFT latency
                        'spread': quotes.get('spread', quotes['ask'] - quotes['bid']),  # Your spread
                        'inventory': self._inventory  # Track position
                    }
                    
                    # Only log every 100th execution to reduce spam
                    if not hasattr(self, '_exec_count'):
                        self._exec_count = 0
                    self._exec_count += 1
                    if self._exec_count == 1:
                        spread = quotes.get('spread', quotes['ask'] - quotes['bid'])
                        self.logger.info(f"🎯 FIRST ORDER: {side.upper()} {quantity} @ ${price:.2f} | Spread: ${spread:.2f} | Inventory: {self._inventory:.6f}")
                    elif self._exec_count % 100 == 0:
                        spread = quotes.get('spread', quotes['ask'] - quotes['bid'])
                        self.logger.info(f"📊 ORDER #{self._exec_count}: {side.upper()} @ ${price:.2f} | Spread: ${spread:.2f} | Inventory: {self._inventory:.6f}")
                    
                    return execution_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error simulating order execution: {e}")
            return None
    
    def _save_trading_state(self):
        """Save current trading state to file"""
        try:
            state_file = Path('live_trading_state.json')
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'trade_count': self.trade_count,
                'performance': self.performance_tracker.get_current_performance() if self.performance_tracker else {},
                'config': self.config
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.info(f"Trading state saved to {state_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving trading state: {e}")
    
    def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            'is_running': self.is_running,
            'is_connected': self.is_connected,
            'trade_count': self.trade_count,
            'runtime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'config': self.config
        }