#!/usr/bin/env python
# Copyright (c) 2025 Richard Wheeler
# Licensed under the Proprietary Evaluation License
# See LICENSE file for details
# For commercial licensing: richard.wheeler@priosym.com
"""
ML Builder CLI Entry Point

This module provides the command-line interface for launching the ML Builder application.
When users install the package and run `ml-builder`, this script will be executed.
"""
import sys
import os
from pathlib import Path


def main():
    """Main entry point for ML Builder application."""
    # Get the package directory
    package_dir = Path(__file__).parent
    app_path = package_dir / "app.py"
    
    # Verify the app.py file exists
    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)
    
    # Change to package directory for proper relative imports
    original_dir = os.getcwd()
    os.chdir(package_dir)
    
    try:
        # Import streamlit CLI and run the app
        from streamlit.web import cli as stcli
        
        # Set up streamlit arguments
        sys.argv = ["streamlit", "run", str(app_path)]
        
        # Launch the streamlit application
        sys.exit(stcli.main())
        
    except ImportError:
        print("Error: Streamlit is not installed. Please install it with:")
        print("pip install streamlit")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error launching ML Builder: {e}")
        sys.exit(1)
        
    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()