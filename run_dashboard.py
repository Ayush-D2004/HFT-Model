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
    
    # Get the dashboard app path - Use absolute path to avoid path issues
    project_root = Path(__file__).parent
    dashboard_path = project_root / "src" / "dashboard" / "app.py"
    
    # Verify file exists
    if not dashboard_path.exists():
        print(f"❌ Error: Dashboard file not found at {dashboard_path}")
        print(f"Project root: {project_root}")
        return
    
    try:
        print("🚀 Launching HFT Market Maker Dashboard...")
        print(f"Dashboard file: {dashboard_path}")
        print(f"Dashboard will open at: http://localhost:8501")
        
        # Launch Streamlit with absolute path
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path.absolute()),
            "--server.port=8501",
            "--server.headless=false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_dashboard()