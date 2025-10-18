"""
Main Backtesting Engine
=====================

Comprehensive backtesting framework that integrates:
- Order book replay with historical data
- Strategy execution with same logic as live trading
- Fill simulation with realistic market microstructure
- Performance metrics and analysis
- Parameter optimization and grid search
"""

import time
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import itertools
from loguru import logger

from src.utils.logger import setup_backtesting_logging, setup_development_logging

from .replay_engine import OrderBookReplayEngine, HistoricalDataLoader, BacktestEvent
from .fill_simulator import FillSimulator, FillEvent
from .metrics import BacktestMetrics, PerformanceMetrics
from src.strategy import (
    AvellanedaStoikovPricer, QuoteManager, RiskManager, RiskLimits,
    QuoteParameters, Order, OrderSide
)
from src.data_ingestion import OrderBook
from src.utils.config import config


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Time period
    start_date: str
    end_date: str
    symbol: str = "BTCUSDT"
    
    # Strategy parameters - OPTIMIZED FOR HFT MARKET MAKING
    gamma: float = 0.015  # Risk aversion - typical HFT range 0.01-0.02
    time_horizon: float = 10.0  # Time horizon in seconds - HFT uses 5-15s
    min_spread: float = 0.0005  # Minimum spread 0.05% (5 bps) - competitive HFT level
    tick_size: float = 0.01
    lot_size: float = 0.001
    
    # Risk parameters
    max_position: float = 10.0
    max_daily_loss: float = 1000.0  
    max_drawdown: float = 0.30  # 30% - realistic for market making
    
    # Fill simulation - BINANCE ACTUAL FEES
    maker_fee: float = 0.0002  # 0.02% - Binance maker fee (limit orders)
    taker_fee: float = 0.0005  # 0.05% - Binance taker fee (market orders)
    # Note: base_fill_probability removed - now using realistic distance-based curve for 70-85% fill rate
    latency_mean_ms: float = 50.0
    
    # Execution
    initial_capital: float = 100000.0
    replay_speed: float = 1.0
    data_directory: str = "./data"
    

@dataclass
class BacktestResult:
    """Complete backtesting result"""
    config: BacktestConfig
    performance: PerformanceMetrics
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    

class StrategyBacktester:
    """
    Strategy backtester that integrates the strategy with market replay.
    Handles order placement, fill simulation, and performance tracking.
    """
    
    def __init__(self, 
                 config: BacktestConfig,
                 fill_simulator: FillSimulator,
                 metrics: BacktestMetrics):
        
        self.config = config
        self.fill_simulator = fill_simulator
        self.metrics = metrics
        
        # Initialize strategy components
        self.pricer = AvellanedaStoikovPricer(
            tick_size=config.tick_size,
            lot_size=config.lot_size,
            ewma_alpha=0.2,
            max_inventory=config.max_position
        )
        
        risk_limits = RiskLimits(
            max_position=config.max_position,
            max_daily_loss=config.max_daily_loss,
            max_drawdown=config.max_drawdown
        )
        self.risk_manager = RiskManager(risk_limits, initial_capital=config.initial_capital)
        
        # Quote manager with backtesting order callback
        self.quote_manager = QuoteManager(
            symbol=config.symbol,
            pricer=self.pricer,
            risk_manager=self.risk_manager,
            order_callback=self._order_callback
        )
        
        # Add fill callback to simulator
        self.fill_simulator.add_fill_callback(self._fill_callback)
        
        # State tracking
        self.current_price = 0.0
        self.last_quote_time = 0.0
        self.quote_sequence = 0
        
        logger.info(f"StrategyBacktester initialized for {config.symbol}")
    
    def _order_callback(self, order_data) -> Dict[str, Any]:
        """Handle order placement/cancellation from quote manager"""
        try:
            # Handle cancellation requests (dict with 'action': 'cancel')
            if isinstance(order_data, dict) and order_data.get('action') == 'cancel':
                order_id = order_data['order_id']
                success = self.fill_simulator.cancel_order(order_id)
                
                if success:
                    return {'success': True}
                else:
                    return {'success': False, 'error': 'Order not found'}
            
            # Handle new order submission (Order object)
            elif isinstance(order_data, Order):
                success = self.fill_simulator.submit_order(order_data)
                
                if success:
                    return {
                        'success': True,
                        'order_id': order_data.order_id,
                        'timestamp': order_data.timestamp
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Fill simulator rejected order'
                    }
            else:
                return {'success': False, 'error': f'Invalid order_data type: {type(order_data)}'}
                
        except Exception as e:
            logger.error(f"Error in order callback: {e}")
            return {'success': False, 'error': str(e)}
    
    def _fill_callback(self, fill_event: FillEvent) -> None:
        """Handle fill event from fill simulator"""
        try:
            # Update quote manager
            self.quote_manager.handle_fill(
                fill_event.order_id,
                fill_event.fill_price,
                fill_event.fill_quantity,
                fill_event.timestamp
            )
            
            # Update metrics
            self.metrics.record_fill(fill_event, self.current_price)
            # Record that our quote was hit
            self.metrics.record_quote_update(fill_event.timestamp, was_hit=True)
            
            # Update pricer with inventory
            self.pricer.update_inventory(self.risk_manager.current_position)
            
            logger.debug(f"Fill processed: {fill_event.side.value} {fill_event.fill_quantity:.4f} "
                        f"@ {fill_event.fill_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error handling fill callback: {e}")
    
    def handle_market_event(self, event: BacktestEvent) -> None:
        """Process market event from replay engine"""
        try:
            if event.event_type == 'market_update':
                self._handle_market_update(event)
            elif event.event_type == 'trade':
                self._handle_trade_event(event)
                
        except Exception as e:
            logger.error(f"Error handling market event: {e}")
    
    def _handle_market_update(self, event: BacktestEvent) -> None:
        """Handle market data update"""
        try:
            data = event.data
            midprice = data.get('midprice')
            
            if midprice is None:
                return
            
            self.current_price = midprice
            
            # Update fill simulator market state
            self.fill_simulator.update_market_state(
                timestamp=event.timestamp,
                best_bid=data.get('best_bid', midprice - 1),
                best_ask=data.get('best_ask', midprice + 1),
                bid_volume=data.get('bid_volume', 1.0),
                ask_volume=data.get('ask_volume', 1.0),
                trade_rate=0.1,  # Could be calculated from recent trades
                volatility=0.001  # Could be from pricer
            )
            
            # Update fill simulator with current position for inventory-based fills
            self.fill_simulator.update_position(self.risk_manager.current_position)
            
            # Update strategy with new market data
            self.pricer.update_market(midprice, event.timestamp)
            self.risk_manager.update_pnl(self.current_price, event.timestamp)
            
            # Generate new quotes if needed
            if self._should_update_quotes(event.timestamp):
                self._update_quotes(event.timestamp)
                
            # Update metrics
            self.metrics.update_pnl(event.timestamp, self.current_price)
            
        except Exception as e:
            logger.error(f"Error handling market update: {e}")
    
    def _handle_trade_event(self, event: BacktestEvent) -> None:
        """Handle trade event for arrival rate estimation"""
        try:
            self.pricer.register_trade_event(event.timestamp)
            
        except Exception as e:
            logger.error(f"Error handling trade event: {e}")
    
    def _should_update_quotes(self, timestamp: float) -> bool:
        """Determine if quotes should be updated"""
        # Update quotes every few seconds or on significant market moves
        time_since_last = timestamp - self.last_quote_time
        
        return (
            time_since_last > 5.0 or  # Time-based updates
            self.quote_manager.current_quote is None or  # No current quote
            not self.quote_manager.current_bid_order or  # Missing orders
            not self.quote_manager.current_ask_order
        )
    
    def _update_quotes(self, timestamp: float) -> None:
        """Update market quotes"""
        try:
            # Generate quote parameters
            quote_params = QuoteParameters(
                gamma=self.config.gamma,
                T=self.config.time_horizon,
                min_spread=self.config.min_spread
            )
            
            # Update quotes via quote manager
            success = self.quote_manager.update_market_quote(self.current_price, timestamp)
            
            if success:
                self.last_quote_time = timestamp
                self.quote_sequence += 1
                self.metrics.record_quote_update(timestamp, was_hit=False)
            
        except Exception as e:
            logger.error(f"Error updating quotes: {e}")
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.pricer.reset()
        self.risk_manager.reset_daily()
        self.quote_manager.reset()
        self.current_price = 0.0
        self.last_quote_time = 0.0
        self.quote_sequence = 0


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the entire backtesting process:
    
    Features:
    - Historical data replay with order book reconstruction
    - Strategy execution with realistic fill simulation
    - Performance analysis and reporting
    - Parameter optimization with grid search
    - Multi-threaded backtesting for efficiency
    - Comprehensive result storage and analysis
    """
    
    def __init__(self, data_directory: str = "./data"):
        self.data_directory = Path(data_directory)
        self.data_loader = HistoricalDataLoader(str(self.data_directory))
        
        # Results storage
        self.results: List[BacktestResult] = []
        
        logger.info(f"BacktestEngine initialized with data directory: {self.data_directory}")
    
    def run_backtest(self, config: BacktestConfig, silent: bool = False) -> BacktestResult:
        """
        Run single backtest with given configuration.
        Returns comprehensive results including performance metrics.
        
        Args:
            config: Backtest configuration
            silent: If True, suppress verbose logging
        """
        try:
            # Configure logging based on mode
            if silent:
                setup_backtesting_logging()
            
            print(f"Running backtest: {config.symbol} {config.start_date} to {config.end_date}")
            
            # Initialize components
            fill_simulator = FillSimulator(
                maker_fee=config.maker_fee,
                taker_fee=config.taker_fee,
                latency_mean_ms=config.latency_mean_ms
            )
            
            metrics = BacktestMetrics(config.initial_capital)
            
            strategy_backtester = StrategyBacktester(config, fill_simulator, metrics)
            
            # Initialize replay engine
            replay_engine = OrderBookReplayEngine(
                symbol=config.symbol,
                data_loader=self.data_loader,
                strategy_callback=strategy_backtester.handle_market_event
            )
            
            # Run the backtest with REAL market data
            print("Processing REAL historical data from Binance...")
            replay_results = replay_engine.run_backtest(
                start_date=config.start_date,
                end_date=config.end_date
            )
            
            if not replay_results.get('success', False):
                return BacktestResult(
                    config=config,
                    performance=PerformanceMetrics(
                        total_pnl=0, realized_pnl=0, unrealized_pnl=0, gross_pnl=0, net_pnl=0,
                        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
                        max_drawdown_duration=0, volatility=0, total_trades=0, winning_trades=0,
                        losing_trades=0, win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                        fill_rate=0, quote_hit_rate=0, avg_spread_captured=0, inventory_turnover=0,
                        adverse_selection_rate=0, avg_fill_latency_ms=0, total_fees=0, fee_rate=0,
                        start_time=time.time(), end_time=time.time(), duration_hours=0
                    ),
                    success=False,
                    error=replay_results.get('error', 'Replay failed')
                )
            
            print("Backtest completed, calculating metrics...")
            
            # Calculate final performance
            final_price = strategy_backtester.current_price or 50000.0
            performance = metrics.calculate_performance_metrics(final_price)
            
            # Create result
            result = BacktestResult(
                config=config,
                performance=performance,
                success=True,
                metadata={
                    'replay_stats': replay_results.get('statistics', {}),
                    'fill_stats': fill_simulator.get_statistics(),
                    'strategy_stats': {
                        'pricer': strategy_backtester.pricer.get_statistics(),
                        'risk_manager': strategy_backtester.risk_manager.get_statistics(),
                        'quote_manager': strategy_backtester.quote_manager.get_statistics()
                    }
                }
            )
            
            self.results.append(result)
            
            logger.success(f"Backtest completed: PnL={performance.total_pnl:.2f}, "
                          f"Sharpe={performance.sharpe_ratio:.2f}, Trades={performance.total_trades}")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return BacktestResult(
                config=config,
                performance=PerformanceMetrics(
                    total_pnl=0, realized_pnl=0, unrealized_pnl=0, gross_pnl=0, net_pnl=0,
                    sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
                    max_drawdown_duration=0, volatility=0, total_trades=0, winning_trades=0,
                    losing_trades=0, win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                    fill_rate=0, quote_hit_rate=0, avg_spread_captured=0, inventory_turnover=0,
                    adverse_selection_rate=0, avg_fill_latency_ms=0, total_fees=0, fee_rate=0,
                    start_time=time.time(), end_time=time.time(), duration_hours=0
                ),
                success=False,
                error=str(e)
            )
    

    
    def run_parameter_sweep(self, 
                          base_config: BacktestConfig,
                          parameter_grid: Dict[str, List[Any]],
                          max_workers: int = 4,
                          silent: bool = True) -> List[BacktestResult]:
        """
        Run parameter sweep (grid search) over multiple parameter combinations.
        
        Args:
            base_config: Base configuration to modify
            parameter_grid: Dictionary of parameters to sweep over
            max_workers: Number of parallel workers
            silent: If True, suppress verbose logging
        
        Returns:
            List of BacktestResult objects for each parameter combination
        """
        try:
            # Setup logging for parameter sweep
            if silent:
                setup_backtesting_logging()
            
            # Generate all parameter combinations
            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())
            param_combinations = list(itertools.product(*param_values))
            
            print(f"🔍 Starting parameter sweep: {len(param_combinations)} combinations")
            
            # Create configurations for each combination
            configs = []
            for combination in param_combinations:
                config_dict = asdict(base_config)
                
                # Update parameters
                for param_name, param_value in zip(param_names, combination):
                    config_dict[param_name] = param_value
                
                configs.append(BacktestConfig(**config_dict))
            
            # Run backtests in parallel
            results = []
            
            if max_workers == 1:
                # Single-threaded execution
                for i, config in enumerate(configs):
                    print(f"⚙️ Running backtest {i+1}/{len(configs)}")
                    result = self.run_backtest(config, silent=silent)
                    results.append(result)
            else:
                # Multi-threaded execution
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_config = {
                        executor.submit(self.run_backtest, config, silent): config 
                        for config in configs
                    }
                    
                    for i, future in enumerate(future_to_config):
                        try:
                            result = future.result()
                            results.append(result)
                            if result.success:
                                print(f"Completed backtest {i+1}/{len(configs)}: PnL=${result.performance.total_pnl:.2f}")
                            else:
                                print(f"Backtest {i+1}/{len(configs)} failed")
                        except Exception as e:
                            print(f"💥 Backtest {i+1} failed: {e}")
            
            print(f"🎉 Parameter sweep completed: {len([r for r in results if r.success])}/{len(results)} successful")
            
            return results
            
        except Exception as e:
            print(f"💥 Parameter sweep failed: {e}")
            return []
    
    def analyze_results(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """
        Analyze backtest results and generate comprehensive report.
        """
        try:
            if not results:
                return {'error': 'No results to analyze'}
            
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                return {'error': 'No successful backtests to analyze'}
            
            # Extract performance metrics
            performances = [r.performance for r in successful_results]
            
            # Summary statistics
            total_pnls = [p.total_pnl for p in performances]
            sharpe_ratios = [p.sharpe_ratio for p in performances]
            max_drawdowns = [p.max_drawdown for p in performances]
            
            analysis = {
                'summary': {
                    'total_backtests': len(results),
                    'successful_backtests': len(successful_results),
                    'success_rate': len(successful_results) / len(results)
                },
                'pnl_stats': {
                    'mean': np.mean(total_pnls),
                    'std': np.std(total_pnls),
                    'min': np.min(total_pnls),
                    'max': np.max(total_pnls),
                    'median': np.median(total_pnls),
                    'positive_pnl_rate': len([pnl for pnl in total_pnls if pnl > 0]) / len(total_pnls)
                },
                'risk_stats': {
                    'sharpe_mean': np.mean(sharpe_ratios),
                    'sharpe_std': np.std(sharpe_ratios),
                    'drawdown_mean': np.mean(max_drawdowns),
                    'drawdown_max': np.max(max_drawdowns)
                },
                'best_result': {
                    'by_pnl': max(successful_results, key=lambda x: x.performance.total_pnl),
                    'by_sharpe': max(successful_results, key=lambda x: x.performance.sharpe_ratio),
                    'by_drawdown': min(successful_results, key=lambda x: x.performance.max_drawdown)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return {'error': str(e)}
    
    def export_results(self, 
                      results: List[BacktestResult], 
                      output_path: str) -> None:
        """Export backtest results to JSON file"""
        try:
            output_data = []
            
            for result in results:
                output_data.append({
                    'config': asdict(result.config),
                    'performance': asdict(result.performance),
                    'success': result.success,
                    'error': result.error,
                    'metadata': result.metadata
                })
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Results exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
    
    def get_results(self) -> List[BacktestResult]:
        """Get all backtest results"""
        return self.results.copy()
    
    def clear_results(self) -> None:
        """Clear stored results"""
        self.results.clear()
        logger.info("Backtest results cleared")


# Example usage for testing
if __name__ == "__main__":
    # Initialize backtest engine
    engine = BacktestEngine("./data")
    
    # Define base configuration
    base_config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-02", 
        symbol="BTCUSDT",
        gamma=0.1,
        time_horizon=30.0,
        initial_capital=10000.0
    )
    
    # Run single backtest
    print("Running single backtest...")
    result = engine.run_backtest(base_config)
    
    print(f"Backtest Result:")
    print(f"Success: {result.success}")
    print(f"Total PnL: {result.performance.total_pnl:.2f}")
    print(f"Sharpe Ratio: {result.performance.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.performance.max_drawdown:.3f}")
    print(f"Total Trades: {result.performance.total_trades}")
    
    # Run parameter sweep
    print("\nRunning parameter sweep...")
    parameter_grid = {
        'gamma': [0.05, 0.1, 0.2],
        'time_horizon': [15.0, 30.0, 60.0],
        'min_spread': [0.01, 0.02, 0.03]
    }
    
    sweep_results = engine.run_parameter_sweep(
        base_config, 
        parameter_grid, 
        max_workers=2
    )
    
    print(f"Parameter sweep completed: {len(sweep_results)} results")
    
    # Analyze results
    analysis = engine.analyze_results(sweep_results)
    print(f"Analysis: {analysis.get('summary', {})}")
    
    if 'best_result' in analysis:
        best_by_pnl = analysis['best_result']['by_pnl']
        print(f"Best PnL: {best_by_pnl.performance.total_pnl:.2f} "
              f"(gamma={best_by_pnl.config.gamma})")
    
    # Export results
    engine.export_results(sweep_results, "backtest_results.json")
    print("Results exported to backtest_results.json")