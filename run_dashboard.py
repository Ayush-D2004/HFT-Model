"""
Dashboard Launcher Script
========================

Simple script to launch the HFT trading dashboard.
"""

import subprocess
import sys
from pathlib import Path

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    
    # Get the dashboard app path
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    
    try:
        print("🚀 Launching HFT Market Maker Dashboard...")
        print(f"Dashboard will open at: http://localhost:8501")
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.headless=false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_dashboard()