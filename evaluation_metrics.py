"""
Performance Evaluation Metrics for Traffic Signal Control

This module provides comprehensive metrics for evaluating the performance
of different traffic signal control strategies including RL agents and baselines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os


class TrafficMetrics:
    """
    Comprehensive traffic performance metrics calculator.
    
    Calculates and analyzes key performance indicators for traffic
    signal control including delay, throughput, queue length, and fairness.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.episode_data = []
        self.controller_results = {}
        
    def add_episode_data(self, controller_name: str, episode_metrics: Dict):
        """
        Add episode data for analysis.
        
        Args:
            controller_name (str): Name of the controller
            episode_metrics (dict): Episode performance metrics
        """
        episode_data = {
            'controller': controller_name,
            **episode_metrics
        }
        self.episode_data.append(episode_data)
        
        # Store by controller for easy access
        if controller_name not in self.controller_results:
            self.controller_results[controller_name] = []
        self.controller_results[controller_name].append(episode_metrics)
        
    def calculate_summary_statistics(self, controller_name: str) -> Dict:
        """
        Calculate summary statistics for a controller.
        
        Args:
            controller_name (str): Name of the controller
            
        Returns:
            dict: Summary statistics
        """
        if controller_name not in self.controller_results:
            return {}
            
        data = self.controller_results[controller_name]
        
        # Extract metrics
        waiting_times = [ep.get('waiting_time', 0) for ep in data]
        queue_lengths = [ep.get('queue_length', 0) for ep in data]
        throughputs = [ep.get('throughput', 0) for ep in data]
        rewards = [ep.get('total_reward', 0) for ep in data]
        
        # Calculate statistics
        stats = {
            'avg_waiting_time': np.mean(waiting_times),
            'std_waiting_time': np.std(waiting_times),
            'avg_queue_length': np.mean(queue_lengths),
            'std_queue_length': np.std(queue_lengths),
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'total_episodes': len(data)
        }
        
        return stats
        
    def calculate_fairness_index(self, controller_name: str) -> float:
        """
        Calculate Jain's fairness index for traffic flow.
        
        Args:
            controller_name (str): Name of the controller
            
        Returns:
            float: Fairness index (0-1, higher is more fair)
        """
        if controller_name not in self.controller_results:
            return 0.0
            
        # For simplicity, we'll use throughput as the fairness metric
        # In practice, you'd want per-approach or per-direction throughput
        throughputs = [ep.get('throughput', 0) for ep in self.controller_results[controller_name]]
        
        if not throughputs or sum(throughputs) == 0:
            return 1.0
            
        # Jain's fairness index: (sum(x_i))^2 / (n * sum(x_i^2))
        n = len(throughputs)
        sum_x = sum(throughputs)
        sum_x_squared = sum(x**2 for x in throughputs)
        
        fairness = (sum_x**2) / (n * sum_x_squared) if sum_x_squared > 0 else 1.0
        
        return fairness
        
    def compare_controllers(self) -> pd.DataFrame:
        """
        Compare performance across all controllers.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for controller_name in self.controller_results.keys():
            stats = self.calculate_summary_statistics(controller_name)
            fairness = self.calculate_fairness_index(controller_name)
            
            comparison_data.append({
                'Controller': controller_name,
                'Avg Delay (s)': stats.get('avg_waiting_time', 0),
                'Std Delay': stats.get('std_waiting_time', 0),
                'Avg Queue Length': stats.get('avg_queue_length', 0),
                'Std Queue Length': stats.get('std_queue_length', 0),
                'Avg Throughput': stats.get('avg_throughput', 0),
                'Std Throughput': stats.get('std_throughput', 0),
                'Fairness Index': fairness,
                'Avg Reward': stats.get('avg_reward', 0),
                'Episodes': stats.get('total_episodes', 0)
            })
            
        return pd.DataFrame(comparison_data)
        
    def plot_performance_comparison(self, save_path: str = None):
        """
        Create performance comparison plots.
        
        Args:
            save_path (str): Path to save the plots
        """
        if not self.episode_data:
            print("No data available for plotting.")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.episode_data)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Traffic Signal Control Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Average Waiting Time
        if 'waiting_time' in df.columns:
            sns.boxplot(data=df, x='controller', y='waiting_time', ax=axes[0, 0])
            axes[0, 0].set_title('Average Waiting Time by Controller')
            axes[0, 0].set_xlabel('Controller')
            axes[0, 0].set_ylabel('Waiting Time (s)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
        # Plot 2: Average Queue Length
        if 'queue_length' in df.columns:
            sns.boxplot(data=df, x='controller', y='queue_length', ax=axes[0, 1])
            axes[0, 1].set_title('Average Queue Length by Controller')
            axes[0, 1].set_xlabel('Controller')
            axes[0, 1].set_ylabel('Queue Length (vehicles)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
        # Plot 3: Throughput
        if 'throughput' in df.columns:
            sns.boxplot(data=df, x='controller', y='throughput', ax=axes[1, 0])
            axes[1, 0].set_title('Vehicle Throughput by Controller')
            axes[1, 0].set_xlabel('Controller')
            axes[1, 0].set_ylabel('Throughput (vehicles)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
        # Plot 4: Reward (for RL controllers)
        if 'total_reward' in df.columns:
            rl_data = df[df['controller'].str.contains('DQN|PPO', case=False, na=False)]
            if not rl_data.empty:
                sns.lineplot(data=rl_data, x=rl_data.index, y='total_reward', 
                           hue='controller', ax=axes[1, 1])
                axes[1, 1].set_title('Learning Progress (Cumulative Reward)')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Total Reward')
            else:
                axes[1, 1].text(0.5, 0.5, 'No RL data available', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Learning Progress (Cumulative Reward)')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to {save_path}")
        else:
            plt.show()
            
    def generate_performance_report(self, save_path: str = "performance_report.txt"):
        """
        Generate a comprehensive performance report.
        
        Args:
            save_path (str): Path to save the report
        """
        with open(save_path, 'w') as f:
            f.write("TRAFFIC SIGNAL CONTROL PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall comparison table
            comparison_df = self.compare_controllers()
            f.write("PERFORMANCE COMPARISON TABLE\n")
            f.write("-" * 30 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # Detailed statistics for each controller
            f.write("DETAILED CONTROLLER STATISTICS\n")
            f.write("-" * 35 + "\n")
            
            for controller_name in self.controller_results.keys():
                f.write(f"\n{controller_name.upper()}\n")
                f.write("-" * len(controller_name) + "\n")
                
                stats = self.calculate_summary_statistics(controller_name)
                fairness = self.calculate_fairness_index(controller_name)
                
                f.write(f"Episodes Analyzed: {stats.get('total_episodes', 0)}\n")
                f.write(f"Average Waiting Time: {stats.get('avg_waiting_time', 0):.2f} ± {stats.get('std_waiting_time', 0):.2f} s\n")
                f.write(f"Average Queue Length: {stats.get('avg_queue_length', 0):.2f} ± {stats.get('std_queue_length', 0):.2f} vehicles\n")
                f.write(f"Average Throughput: {stats.get('avg_throughput', 0):.2f} ± {stats.get('std_throughput', 0):.2f} vehicles\n")
                f.write(f"Fairness Index: {fairness:.3f}\n")
                
                if 'avg_reward' in stats:
                    f.write(f"Average Reward: {stats.get('avg_reward', 0):.2f} ± {stats.get('std_reward', 0):.2f}\n")
                    
            # Performance ranking
            f.write("\n\nPERFORMANCE RANKING\n")
            f.write("-" * 20 + "\n")
            
            # Rank by waiting time (lower is better)
            waiting_time_ranking = comparison_df.sort_values('Avg Delay (s)')
            f.write("Best Average Waiting Time:\n")
            for i, (_, row) in enumerate(waiting_time_ranking.head(3).iterrows(), 1):
                f.write(f"{i}. {row['Controller']}: {row['Avg Delay (s)']:.2f} s\n")
                
            f.write("\n")
            
            # Rank by throughput (higher is better)
            throughput_ranking = comparison_df.sort_values('Avg Throughput', ascending=False)
            f.write("Best Average Throughput:\n")
            for i, (_, row) in enumerate(throughput_ranking.head(3).iterrows(), 1):
                f.write(f"{i}. {row['Controller']}: {row['Avg Throughput']:.2f} vehicles\n")
                
            f.write("\n")
            
            # Rank by fairness (higher is better)
            fairness_ranking = comparison_df.sort_values('Fairness Index', ascending=False)
            f.write("Best Fairness Index:\n")
            for i, (_, row) in enumerate(fairness_ranking.head(3).iterrows(), 1):
                f.write(f"{i}. {row['Controller']}: {row['Fairness Index']:.3f}\n")
                
        print(f"Performance report saved to {save_path}")
        
    def save_results(self, file_path: str = "evaluation_results.csv"):
        """
        Save all evaluation results to CSV.
        
        Args:
            file_path (str): Path to save the results
        """
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            df.to_csv(file_path, index=False)
            print(f"Results saved to {file_path}")
        else:
            print("No data to save.")
            
    def print_summary(self):
        """Print a summary of the evaluation results."""
        if not self.controller_results:
            print("No evaluation data available.")
            return
            
        print("\nTRAFFIC SIGNAL CONTROL EVALUATION SUMMARY")
        print("=" * 45)
        
        comparison_df = self.compare_controllers()
        print(comparison_df.to_string(index=False))
        
        print("\nKey Insights:")
        print("-" * 15)
        
        # Find best performers
        best_delay = comparison_df.loc[comparison_df['Avg Delay (s)'].idxmin()]
        best_throughput = comparison_df.loc[comparison_df['Avg Throughput'].idxmax()]
        best_fairness = comparison_df.loc[comparison_df['Fairness Index'].idxmax()]
        
        print(f"• Lowest average delay: {best_delay['Controller']} ({best_delay['Avg Delay (s)']:.2f} s)")
        print(f"• Highest throughput: {best_throughput['Controller']} ({best_throughput['Avg Throughput']:.2f} vehicles)")
        print(f"• Best fairness: {best_fairness['Controller']} ({best_fairness['Fairness Index']:.3f})")
        
        # Calculate improvements over baseline (assuming Fixed-Time is baseline)
        if 'Fixed-Time' in comparison_df['Controller'].values:
            baseline_delay = comparison_df[comparison_df['Controller'] == 'Fixed-Time']['Avg Delay (s)'].iloc[0]
            baseline_throughput = comparison_df[comparison_df['Controller'] == 'Fixed-Time']['Avg Throughput'].iloc[0]
            
            print(f"\nImprovements over Fixed-Time baseline:")
            print("-" * 40)
            
            for _, row in comparison_df.iterrows():
                if row['Controller'] != 'Fixed-Time':
                    delay_improvement = ((baseline_delay - row['Avg Delay (s)']) / baseline_delay) * 100
                    throughput_improvement = ((row['Avg Throughput'] - baseline_throughput) / baseline_throughput) * 100
                    
                    print(f"• {row['Controller']}:")
                    print(f"  - Delay reduction: {delay_improvement:.1f}%")
                    print(f"  - Throughput increase: {throughput_improvement:.1f}%")