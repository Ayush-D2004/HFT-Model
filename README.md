# HFT Market Maker Trading System

A professional-grade High-Frequency Trading (HFT) system implementing market making strategies using the Avellaneda-Stoikov framework. This system provides real-time market data ingestion, sophisticated trading algorithms, comprehensive backtesting, and interactive analytics.


## 🚀 Key Features

### 📊 Data Ingestion
- **Binance WebSocket Integration**: Real-time market data with order book depth
- **Order Book Management**: Maintains rolling order book with U/u sequencing logic
- **Low Latency Processing**: <0.01% message loss rate with automatic reconnection
- **Clean Architecture**: Modular design with restart and reconnect logic

### 🧠 Trading Strategy
- **Avellaneda-Stoikov Framework**: Optimal market making with inventory risk management
- **Fair Value Estimation**: Dynamic midprice and micro-price calculations
- **Volatility Estimation**: EWMA-based volatility tracking for adaptive quoting
- **Risk Management**: Position, leverage, loss, and latency thresholds
- **Quote Management**: Intelligent bid/ask placement with order lifecycle management

### 🔬 Backtesting Engine
- **Historical Replay**: Deterministic replay of tick data with exact market reconstruction
- **Fill Simulation**: Probabilistic fill model based on market microstructure
- **Performance Metrics**: Comprehensive analytics including Sharpe, Sortino, drawdown
- **Parameter Optimization**: Grid search capabilities for strategy tuning

### 📈 Dashboard & Analytics
- **Real-time Order Book**: Live depth display with bid/ask visualization
- **Market State Charts**: Price movements, volatility heatmaps, spread analysis
- **Performance Tracking**: PnL curves, trade analytics, inventory management
- **Interactive Backtesting**: Parameter optimization and results analysis

## 📦 Installation

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ayush-D2004/HFT-Model.git
   cd "HFT model"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

## 🏃‍♂️ Quick Start

### 1. Launch Dashboard
```bash
python run_dashboard.py
```
or 
```bash
python quickstart.py
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Financial Metrics
- **PnL Analysis**: Realized/unrealized, gross/net returns
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Maximum drawdown and duration

### Trading Metrics  
- **Fill Analysis**: Fill rates, execution quality, latency
- **Market Making**: Spread capture, inventory turnover, adverse selection
- **Quote Performance**: Hit rates, lifetime, market impact

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes with tests**
4. **Run quality checks**: `pytest && black . && mypy .`
5. **Submit pull request**

## 📄 License

This project is for educational and research purposes. See [LICENSE](LICENSE) for details.


---
