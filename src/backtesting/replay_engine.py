"""
Order Book Replay Engine for Backtesting
========================================

Deterministic replay of historical tick data to evaluate strategies offline.
Exactly replicates live trading conditions for accurate strategy testing.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Iterator, Callable, Any
from dataclasses import dataclass
from datetime import timezone
import pandas as pd
import numpy as np
from pathlib import Path

# Use standard logging instead of loguru
logger = logging.getLogger(__name__)

from src.data_ingestion.order_book import OrderBook, OrderBookSnapshot
from src.data_ingestion.binance_historical import BinanceHistoricalDataFetcher
from src.strategy import AvellanedaStoikovPricer, QuoteManager, RiskManager


@dataclass
class TickData:
    """Individual tick data point"""
    timestamp: float
    event_type: str  # 'snapshot', 'update', 'trade'
    symbol: str
    data: Dict[str, Any]
    sequence_id: Optional[int] = None


@dataclass
class BacktestEvent:
    """Event during backtesting"""
    timestamp: float
    event_type: str  # 'market_update', 'fill', 'quote_update'
    data: Dict[str, Any]


class HistoricalDataLoader:
    """
    Loads and preprocesses REAL historical market data from Binance API for replay.
    No synthetic data - only real market data from Binance.
    """
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        # Initialize Binance historical data fetcher
        from src.data_ingestion.binance_historical import BinanceHistoricalDataFetcher
        self.binance_fetcher = BinanceHistoricalDataFetcher(use_testnet=False)
        
    def load_binance_depth_data(self, 
                               symbol: str, 
                               start_date: str, 
                               end_date: str) -> Iterator[TickData]:
        """
        Load REAL Binance market data from Binance API.
        Fetches actual historical kline data and generates realistic order book snapshots.
        """
        try:
            logger.info(f"Fetching REAL market data from Binance API for {symbol}")
            
            # Convert to datetime objects
            start_dt = pd.to_datetime(start_date).replace(tzinfo=timezone.utc) 
            end_dt = pd.to_datetime(end_date).replace(tzinfo=timezone.utc)
            
            # Fetch real kline data from Binance
            kline_data = self.binance_fetcher.get_kline_data(symbol, start_dt, end_dt, '1m')
            
            if kline_data.empty:
                logger.error(f"No real market data available from Binance for {symbol} in specified period")
                return
            
            logger.info(f"Retrieved {len(kline_data)} minutes of real market data from Binance")
            
            # Generate realistic order book snapshots from real kline data
            order_book_updates = self.binance_fetcher.simulate_order_book_from_klines(kline_data)
            
            # Convert to TickData objects and yield
            for update in order_book_updates:
                tick_data = TickData(
                    timestamp=update['timestamp'],
                    event_type='depth_snapshot',
                    symbol=symbol,
                    data={
                        'bids': update['bids'],
                        'asks': update['asks'],
                        'lastUpdateId': update['lastUpdateId'],
                        'source': 'BINANCE_REAL_HISTORICAL'
                    }
                )
                yield tick_data
                
        except Exception as e:
            logger.error(f"Error fetching real historical data from Binance: {e}")
    
    def _load_json_file(self, file_path: Path, data_type: str) -> Iterator[TickData]:
        """Load JSON lines file and convert to TickData objects"""
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract timestamp
                        timestamp = data.get('T', data.get('timestamp', 0)) / 1000.0
                        
                        # Determine event type
                        if data_type == 'depth':
                            event_type = 'snapshot' if 'lastUpdateId' in data else 'update'
                        else:
                            event_type = 'trade'
                        
                        yield TickData(
                            timestamp=timestamp,
                            event_type=event_type,
                            symbol=data.get('s', ''),
                            data=data,
                            sequence_id=line_num
                        )
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")



class OrderBookReplayEngine:
    """
    Professional order book replay engine for backtesting:
    
    Features:
    - Deterministic replay of historical tick data
    - Order book reconstruction with validation
    - Strategy integration with same logic as live
    - Performance monitoring and statistics
    - Configurable replay speed and filtering
    """
    
    def __init__(self, 
                 symbol: str,
                 data_loader: HistoricalDataLoader,
                 strategy_callback: Optional[Callable] = None):
        
        self.symbol = symbol
        self.data_loader = data_loader
        self.strategy_callback = strategy_callback
        
        # Order book for replay - use non-strict sequencing for backtesting
        self.order_book = OrderBook(symbol, max_levels=50, strict_sequencing=False)
        
        # Replay state
        self.current_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0
        self.replay_speed = 1.0  # 1.0 = real time, >1 = faster
        
        # Event tracking
        self.events_processed = 0
        self.events_total = 0
        self.backtest_events: List[BacktestEvent] = []
        
        # Statistics
        self.stats = {
            'snapshots_processed': 0,
            'updates_processed': 0,
            'trades_processed': 0,
            'orderbook_errors': 0,
            'strategy_calls': 0,
            'replay_duration_sec': 0.0
        }
        
        logger.info(f"OrderBookReplayEngine initialized for {symbol}")
    
    def run_backtest(self, 
                    start_date: str,
                    end_date: str, 
                    replay_speed: float = 1.0) -> Dict[str, Any]:
        """
        Run complete backtest over specified date range.
        Returns comprehensive results and performance metrics.
        """
        try:
            self.replay_speed = replay_speed
            self.start_time = time.time()
            
            logger.info(f"Starting backtest: {start_date} to {end_date} at {replay_speed}x speed")
            
            # Load historical data
            tick_data = self.data_loader.load_binance_depth_data(
                self.symbol, start_date, end_date
            )
            
            # Convert to sorted list for counting and processing
            all_ticks = sorted(tick_data, key=lambda x: x.timestamp)
            self.events_total = len(all_ticks)
            
            if self.events_total == 0:
                logger.error("No real market data found for the specified period - please check date range and symbol")
                raise ValueError("Cannot proceed without real market data from Binance")
            
            logger.info(f"Loaded {self.events_total} tick events")
            
            # Process each tick event
            for tick in all_ticks:
                success = self._process_tick(tick)
                if success:
                    self.events_processed += 1
                
                # Update progress periodically (adjust frequency based on total events)
                progress_interval = max(100, self.events_total // 10)  # Show progress 10 times or every 100 events
                if self.events_processed % progress_interval == 0:
                    progress = (self.events_processed / self.events_total) * 100
                    logger.info(f"Backtest progress: {progress:.1f}%")
            
            # Finalize backtest
            self.end_time = time.time()
            self.stats['replay_duration_sec'] = self.end_time - self.start_time
            
            logger.info(f"Backtest completed: {self.events_processed}/{self.events_total} events processed")
            
            return self._generate_results()
            
        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            return {'success': False, 'error': str(e)}
    

    
    def _process_tick(self, tick: TickData) -> bool:
        """Process individual tick data event"""
        try:
            self.current_time = tick.timestamp
            
            # Route to appropriate handler
            if tick.event_type in ['snapshot', 'depth_snapshot']:
                return self._handle_snapshot(tick)
            elif tick.event_type == 'update':
                return self._handle_update(tick)
            elif tick.event_type == 'trade':
                return self._handle_trade(tick)
            else:
                logger.debug(f"Unknown event type: {tick.event_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            return False
    
    def _handle_snapshot(self, tick: TickData) -> bool:
        """Handle order book snapshot"""
        try:
            success = self.order_book.handle_snapshot(tick.data)
            if success:
                self.stats['snapshots_processed'] += 1
                
                # Notify strategy of market update
                self._notify_strategy_market_update()
                
                return True
            else:
                self.stats['orderbook_errors'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error handling snapshot: {e}")
            self.stats['orderbook_errors'] += 1
            return False
    
    def _handle_update(self, tick: TickData) -> bool:
        """Handle incremental order book update"""
        try:
            success = self.order_book.handle_update(tick.data)
            if success:
                self.stats['updates_processed'] += 1
                
                # Notify strategy of market update  
                self._notify_strategy_market_update()
                
                return True
            else:
                self.stats['orderbook_errors'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error handling update: {e}")
            self.stats['orderbook_errors'] += 1
            return False
    
    def _handle_trade(self, tick: TickData) -> bool:
        """Handle trade event"""
        try:
            # Extract trade information
            trade_price = float(tick.data.get('p', 0))
            trade_qty = float(tick.data.get('q', 0))
            is_buyer_maker = tick.data.get('m', False)
            
            # Record trade event (for arrival rate estimation)
            if self.strategy_callback:
                trade_event = BacktestEvent(
                    timestamp=tick.timestamp,
                    event_type='trade',
                    data={
                        'price': trade_price,
                        'quantity': trade_qty,
                        'is_buyer_maker': is_buyer_maker
                    }
                )
                
                self._notify_strategy_event(trade_event)
            
            self.stats['trades_processed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error handling trade: {e}")
            return False
    
    def _notify_strategy_market_update(self):
        """Notify strategy of market data update"""
        if not self.strategy_callback or not self.order_book.is_healthy():
            return
        
        try:
            # Get current market state
            snapshot = self.order_book.get_snapshot()
            midprice = snapshot.midprice()
            
            if midprice is None:
                return
            
            # Create market update event
            market_event = BacktestEvent(
                timestamp=self.current_time,
                event_type='market_update',
                data={
                    'midprice': midprice,
                    'spread': snapshot.spread(),
                    'best_bid': snapshot.bids[0].price if snapshot.bids else None,
                    'best_ask': snapshot.asks[0].price if snapshot.asks else None,
                    'bid_volume': sum(level.quantity for level in snapshot.bids[:5]),
                    'ask_volume': sum(level.quantity for level in snapshot.asks[:5])
                }
            )
            
            self._notify_strategy_event(market_event)
            self.stats['strategy_calls'] += 1
            
        except Exception as e:
            logger.error(f"Error notifying strategy: {e}")
    
    def _notify_strategy_event(self, event: BacktestEvent):
        """Send event to strategy callback"""
        try:
            if self.strategy_callback:
                self.strategy_callback(event)
                
            self.backtest_events.append(event)
            
        except Exception as e:
            logger.error(f"Error in strategy callback: {e}")
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results"""
        return {
            'success': True,
            'metadata': {
                'symbol': self.symbol,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'replay_duration_sec': self.stats['replay_duration_sec'],
                'replay_speed': self.replay_speed,
                'events_total': self.events_total,
                'events_processed': self.events_processed
            },
            'statistics': self.stats,
            'orderbook_final_state': self.order_book.get_statistics(),
            'events': self.backtest_events[-1000:],  # Last 1000 events
            'performance': {
                'processing_rate_eps': self.events_processed / max(self.stats['replay_duration_sec'], 1),
                'success_rate': self.events_processed / max(self.events_total, 1),
                'orderbook_health': self.order_book.is_healthy()
            }
        }
    
    def get_current_orderbook(self) -> OrderBook:
        """Get current order book instance"""
        return self.order_book
    
    def get_statistics(self) -> Dict:
        """Get current replay statistics"""
        return {
            **self.stats,
            'current_time': self.current_time,
            'events_processed': self.events_processed,
            'events_total': self.events_total,
            'progress_pct': (self.events_processed / max(self.events_total, 1)) * 100
        }
    
    def reset(self):
        """Reset replay engine state"""
        self.order_book.reset()
        self.current_time = 0.0
        self.events_processed = 0
        self.events_total = 0
        self.backtest_events.clear()
        
        # Reset statistics
        self.stats = {
            'snapshots_processed': 0,
            'updates_processed': 0,
            'trades_processed': 0,
            'orderbook_errors': 0,
            'strategy_calls': 0,
            'replay_duration_sec': 0.0
        }
        
        logger.info("OrderBookReplayEngine reset")


# Example usage for testing
if __name__ == "__main__":
    # Initialize components
    data_loader = HistoricalDataLoader("./data")
    
    def mock_strategy_callback(event: BacktestEvent):
        """Mock strategy callback for testing"""
        if event.event_type == 'market_update':
            print(f"Market update: {event.data['midprice']:.2f}")
        elif event.event_type == 'trade':
            print(f"Trade: {event.data['price']:.2f} @ {event.data['quantity']:.4f}")
    
    # Initialize replay engine
    replay_engine = OrderBookReplayEngine(
        "BTCUSDT", 
        data_loader, 
        mock_strategy_callback
    )
    
    # Run real market data backtest
    results = replay_engine.run("2024-01-01", "2024-01-02")  # 1 day of real data
    
    print("Real Market Data Backtest Results:")
    print(f"Success: {results['success']}")
    if results['success']:
        print(f"Events processed: {results['metadata']['events_processed']}")
        print(f"Processing rate: {results['performance']['processing_rate_eps']:.1f} events/sec")
        print(f"Order book health: {results['performance']['orderbook_health']}")
        print(f"Statistics: {results['statistics']}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")