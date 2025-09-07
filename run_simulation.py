#!/usr/bin/env python3
"""
Simple run script for SUMO Traffic Signal Control with Deep RL
"""

import os
import sys
import subprocess

# Add src to path
sys.path.insert(0, 'src')

def main():
    """Run the traffic control simulation with reduced episodes for demo."""
    print("SUMO Traffic Signal Control - Deep Reinforcement Learning Demo")
    print("=" * 60)
    
    # Change to src directory
    os.chdir('src')
    
    try:
        # Run the main simulation with reduced episodes for demonstration
        result = subprocess.run([
            sys.executable, 'main.py', 
            '--train_episodes', '20', 
            '--eval_episodes', '10'
        ], capture_output=False, text=True, timeout=300)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("SIMULATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("Check the generated files for detailed results:")
            print("- traffic_control_performance_report.txt")
            print("- traffic_control_comparison.png") 
            print("- traffic_control_results.csv")
        else:
            print(f"Simulation failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("Simulation timed out after 5 minutes")
    except Exception as e:
        print(f"Error running simulation: {e}")
    
    # Change back to root directory
    os.chdir('..')

if __name__ == "__main__":
    main()