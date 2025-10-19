"""
Backtesting Module for HFT Trading System
========================================

Comprehensive backtesting framework with:
- Historical data replay with order book reconstruction
- Realistic fill simulation with market microstructure
- Performance metrics and risk analysis
- Parameter optimization capabilities
"""

from .replay_engine import (
    OrderBookReplayEngine,
    HistoricalDataLoader, 
    TickData,
    BacktestEvent
)
from .fill_simulator import (
    FillSimulator,
    FillEvent,
    FillReason,
    MarketState
)
from .fill_simulator_fifo import (
    FIFOFillSimulator,
    LimitOrder,
    TradeEvent,
    LOBSnapshot
)
from .metrics import (
    BacktestMetrics,
    PerformanceMetrics
)
from .backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    StrategyBacktester
)

__all__ = [
    'OrderBookReplayEngine',
    'HistoricalDataLoader',
    'TickData',
    'BacktestEvent',
    'FillSimulator',
    'FillEvent', 
    'FillReason',
    'MarketState',
    'FIFOFillSimulator',
    'LimitOrder',
    'TradeEvent',
    'LOBSnapshot',
    'BacktestMetrics',
    'PerformanceMetrics',
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'StrategyBacktester'
]