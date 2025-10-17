"""
Quick Start Script for HFT Market Maker System
==============================================

This script helps you get started quickly with the HFT system.
Run this script to verify installation and see basic functionality.
"""

import sys
import os
import subprocess

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking Environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or python_version.minor < 9:
        print("❌ Python 3.9+ is required")
        return False
    
    # Check required packages
    required_packages = [
        'numpy', 'pandas', 'streamlit', 'plotly', 
        'websockets', 'loguru', 'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Check if project structure is correct"""
    print("\n📁 Checking Project Structure...")
    
    required_dirs = [
        'src',
        'src/data_ingestion',
        'src/strategy',
        'src/backtesting',
        'src/dashboard',
        'src/utils'
    ]
    
    required_files = [
        'src/__init__.py',
        'src/data_ingestion/__init__.py',
        'src/strategy/__init__.py',
        'src/backtesting/__init__.py',
        'src/dashboard/__init__.py',
        'src/utils/__init__.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - missing")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing")
            all_good = False
    
    return all_good

def run_basic_test():
    """Run a basic functionality test"""
    print("\n🧪 Running Basic Tests...")
    
    try:
        # Test imports
        from src.strategy import AvellanedaStoikovPricer, QuoteParameters
        from src.data_ingestion import OrderBook
        from src.backtesting import BacktestEngine
        print("✅ All modules imported successfully")
        
        # Test Avellaneda-Stoikov pricer
        pricer = AvellanedaStoikovPricer(tick_size=0.01)
        pricer.update_market(50000.0)
        
        quote_params = QuoteParameters(gamma=0.1, T=30.0)
        quote = pricer.compute_quotes(quote_params)
        
        print(f"✅ Pricer working: Bid=${quote.bid_price:.2f}, Ask=${quote.ask_price:.2f}")
        
        # Test order book
        order_book = OrderBook("BTCUSDT", max_levels=10)
        print("✅ Order book initialized")
        
        # Test backtest engine
        engine = BacktestEngine()
        print("✅ Backtest engine initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_usage():
    """Show usage instructions"""
    print("\n📚 Quick Usage Guide")
    print("=" * 40)
    print("1. Run example:          python example.py")
    print("2. Launch dashboard:     python run_dashboard.py")
    print("3. Run backtest:         python -m src.backtesting.backtest_engine")
    print("4. View documentation:   README.md")
    print("\n🔧 Configuration Files:")
    print("- src/utils/config.py    - Main configuration")
    print("- .env                   - API credentials (create if needed)")
    print("\n📊 Dashboard Features:")
    print("- Real-time order book display")
    print("- Market state visualization")
    print("- Backtesting interface")
    print("- Performance analytics")

def main():
    """Main function"""
    print("🚀 HFT Market Maker - Quick Start")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix issues and try again.")
        return
    
    # Check project structure
    if not check_project_structure():
        print("\n❌ Project structure check failed. Please ensure all files are present.")
        return
    
    # Run basic tests
    if not run_basic_test():
        print("\n❌ Basic tests failed. Please check your installation.")
        return
    
    print("\n✅ All checks passed!")
    print("🎉 System is ready to use!")
    
    # Show usage
    show_usage()
    
    # Ask user what to do next
    print("\n" + "=" * 50)
    print("What would you like to do?")
    print("1. Run example demonstration")
    print("2. Launch dashboard")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🔬 Running example demonstration...")
        try:
            subprocess.run([sys.executable, "example.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to run example")
    elif choice == "2":
        print("\n📊 Launching dashboard...")
        try:
            subprocess.run([sys.executable, "run_dashboard.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to launch dashboard")
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()