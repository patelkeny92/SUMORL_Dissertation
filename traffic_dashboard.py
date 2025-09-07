"""
Traffic Signal Control Visualization Dashboard

Interactive dashboard for monitoring and analyzing traffic signal control
performance using Deep Reinforcement Learning agents vs baseline controllers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
import json
from datetime import datetime, timedelta
import threading

# Add src to path for imports
sys.path.insert(0, 'src')

# Import simulation runner
from simulation_runner import runner

# Set page configuration
st.set_page_config(
    page_title="Traffic Signal Control Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .controller-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample performance data for demonstration."""
    controllers = ['Fixed-Time', 'Actuated', 'DQN', 'PPO']
    
    # Generate sample episode data for each controller
    data = []
    base_time = datetime.now() - timedelta(hours=2)
    
    for i, controller in enumerate(controllers):
        # Different performance profiles for each controller
        if controller == 'Fixed-Time':
            delays = np.random.normal(45, 8, 50)
            queues = np.random.normal(8.3, 2, 50)
            throughputs = np.random.normal(850, 100, 50)
        elif controller == 'Actuated':
            delays = np.random.normal(38, 6, 50)
            queues = np.random.normal(6.9, 1.5, 50)
            throughputs = np.random.normal(920, 80, 50)
        elif controller == 'DQN':
            delays = np.random.normal(28.5, 4, 50)
            queues = np.random.normal(4.2, 1, 50)
            throughputs = np.random.normal(1150, 120, 50)
        else:  # PPO
            delays = np.random.normal(26.1, 3.5, 50)
            queues = np.random.normal(3.8, 0.8, 50)
            throughputs = np.random.normal(1200, 100, 50)
            
        for j in range(50):
            data.append({
                'controller': controller,
                'episode': j + 1,
                'timestamp': base_time + timedelta(minutes=j*2 + i*100),
                'avg_delay': max(0, delays[j]),
                'avg_queue_length': max(0, queues[j]),
                'throughput': max(0, throughputs[j]),
                'fairness_index': np.random.uniform(0.7, 0.95),
                'reward': np.random.normal(0 if controller in ['Fixed-Time', 'Actuated'] else 100, 20)
            })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_training_progress():
    """Generate sample training progress data for RL agents."""
    episodes = range(1, 101)
    
    # DQN training progress
    dqn_rewards = []
    dqn_losses = []
    reward = -50
    loss = 2.0
    
    for episode in episodes:
        # Simulate learning curve
        reward += np.random.normal(0.8, 5)
        loss *= 0.995 + np.random.normal(0, 0.02)
        
        dqn_rewards.append(reward)
        dqn_losses.append(max(0.001, loss))
    
    # PPO training progress
    ppo_rewards = []
    ppo_losses = []
    reward = -50
    loss = 1.5
    
    for episode in episodes:
        reward += np.random.normal(1.0, 4)
        loss *= 0.992 + np.random.normal(0, 0.015)
        
        ppo_rewards.append(reward)
        ppo_losses.append(max(0.001, loss))
    
    training_data = []
    for i, episode in enumerate(episodes):
        training_data.extend([
            {'agent': 'DQN', 'episode': episode, 'reward': dqn_rewards[i], 'loss': dqn_losses[i]},
            {'agent': 'PPO', 'episode': episode, 'reward': ppo_rewards[i], 'loss': ppo_losses[i]}
        ])
    
    return pd.DataFrame(training_data)

def create_performance_comparison(df):
    """Create performance comparison charts."""
    # Calculate averages by controller
    avg_metrics = df.groupby('controller').agg({
        'avg_delay': 'mean',
        'avg_queue_length': 'mean',
        'throughput': 'mean',
        'fairness_index': 'mean'
    }).reset_index()
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Delay (seconds)', 'Average Queue Length (vehicles)', 
                       'Vehicle Throughput (vehicles/hour)', 'Fairness Index'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = {'Fixed-Time': '#FF6B6B', 'Actuated': '#FFA726', 'DQN': '#42A5F5', 'PPO': '#66BB6A'}
    
    # Average Delay
    fig.add_trace(
        go.Bar(x=avg_metrics['controller'], y=avg_metrics['avg_delay'],
               name='Avg Delay', marker_color=[colors[c] for c in avg_metrics['controller']]),
        row=1, col=1
    )
    
    # Average Queue Length
    fig.add_trace(
        go.Bar(x=avg_metrics['controller'], y=avg_metrics['avg_queue_length'],
               name='Avg Queue', marker_color=[colors[c] for c in avg_metrics['controller']]),
        row=1, col=2
    )
    
    # Throughput
    fig.add_trace(
        go.Bar(x=avg_metrics['controller'], y=avg_metrics['throughput'],
               name='Throughput', marker_color=[colors[c] for c in avg_metrics['controller']]),
        row=2, col=1
    )
    
    # Fairness Index
    fig.add_trace(
        go.Bar(x=avg_metrics['controller'], y=avg_metrics['fairness_index'],
               name='Fairness', marker_color=[colors[c] for c in avg_metrics['controller']]),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Performance Comparison Dashboard")
    return fig

def create_time_series_chart(df, metric, title):
    """Create time series chart for a specific metric."""
    fig = px.line(df, x='episode', y=metric, color='controller',
                  title=title, 
                  color_discrete_map={'Fixed-Time': '#FF6B6B', 'Actuated': '#FFA726', 
                                    'DQN': '#42A5F5', 'PPO': '#66BB6A'})
    fig.update_layout(height=400)
    return fig

def create_training_progress_chart(training_df):
    """Create training progress visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Rewards Progress', 'Training Loss Progress'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = {'DQN': '#42A5F5', 'PPO': '#66BB6A'}
    
    for agent in ['DQN', 'PPO']:
        agent_data = training_df[training_df['agent'] == agent]
        
        # Rewards
        fig.add_trace(
            go.Scatter(x=agent_data['episode'], y=agent_data['reward'],
                      mode='lines', name=f'{agent} Rewards', 
                      line=dict(color=colors[agent], width=2)),
            row=1, col=1
        )
        
        # Losses
        fig.add_trace(
            go.Scatter(x=agent_data['episode'], y=agent_data['loss'],
                      mode='lines', name=f'{agent} Loss',
                      line=dict(color=colors[agent], width=2, dash='dash')),
            row=1, col=2
        )
    
    fig.update_layout(height=400, title_text="RL Agent Training Progress")
    return fig

def create_real_time_metrics():
    """Create real-time metrics display."""
    # Simulate current traffic state
    current_metrics = {
        'North Approach': {'queue': np.random.randint(0, 15), 'waiting': np.random.randint(5, 45), 'occupancy': np.random.randint(10, 80)},
        'South Approach': {'queue': np.random.randint(0, 12), 'waiting': np.random.randint(3, 40), 'occupancy': np.random.randint(15, 75)},
        'East Approach': {'queue': np.random.randint(0, 18), 'waiting': np.random.randint(8, 50), 'occupancy': np.random.randint(20, 85)},
        'West Approach': {'queue': np.random.randint(0, 10), 'waiting': np.random.randint(2, 35), 'occupancy': np.random.randint(5, 70)}
    }
    return current_metrics

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚦 Traffic Signal Control Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (10s)", value=False)
    
    # Controller selection
    selected_controllers = st.sidebar.multiselect(
        "Select Controllers to Display",
        ['Fixed-Time', 'Actuated', 'DQN', 'PPO'],
        default=['Fixed-Time', 'Actuated', 'DQN', 'PPO']
    )
    
    # Simulation parameters
    st.sidebar.subheader("📊 Simulation Parameters")
    episodes_to_show = st.sidebar.slider("Episodes to Display", 10, 50, 30)
    
    # Interactive simulation controls
    st.sidebar.subheader("🚀 Run New Simulation")
    controller_to_run = st.sidebar.selectbox(
        "Select Controller to Test",
        ['Fixed-Time', 'Actuated', 'DQN', 'PPO']
    )
    
    simulation_episodes = st.sidebar.slider("Number of Episodes", 5, 25, 10)
    
    if st.sidebar.button("🏃 Run Simulation", type="primary"):
        with st.spinner(f"Running {controller_to_run} simulation..."):
            # Run the simulation
            result = runner.run_quick_simulation(controller_to_run, simulation_episodes)
            if 'error' not in result:
                st.sidebar.success(f"✅ {controller_to_run} simulation completed!")
                st.sidebar.metric("Avg Delay", f"{result['avg_delay']:.1f}s")
                st.sidebar.metric("Avg Queue", f"{result['avg_queue']:.1f}")
                st.sidebar.metric("Throughput", f"{result['throughput']:.0f}")
            else:
                st.sidebar.error(f"❌ Simulation failed: {result['error']}")
    
    # Clear results
    if st.sidebar.button("🗑️ Clear Results"):
        if os.path.exists("dashboard_results.json"):
            os.remove("dashboard_results.json")
        st.sidebar.success("Results cleared!")
        st.rerun()
    
    # Load data
    df = load_sample_data()
    training_df = generate_training_progress()
    
    # Filter data based on selection
    if selected_controllers:
        df_filtered = df[df['controller'].isin(selected_controllers)].tail(episodes_to_show * len(selected_controllers))
    else:
        df_filtered = df.tail(episodes_to_show * 4)
    
    # Real-time metrics section
    st.subheader("🔴 Live Traffic Metrics")
    
    current_metrics = create_real_time_metrics()
    
    # Create columns for real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    approaches = ['North Approach', 'South Approach', 'East Approach', 'West Approach']
    cols = [col1, col2, col3, col4]
    
    for i, (approach, col) in enumerate(zip(approaches, cols)):
        with col:
            st.markdown(f"**{approach}**")
            st.metric("Queue Length", f"{current_metrics[approach]['queue']} vehicles")
            st.metric("Avg Waiting", f"{current_metrics[approach]['waiting']} seconds")
            st.metric("Occupancy", f"{current_metrics[approach]['occupancy']}%")
    
    # Performance comparison section
    st.subheader("📊 Performance Comparison")
    
    if not df_filtered.empty:
        comparison_fig = create_performance_comparison(df_filtered)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        summary_stats = df_filtered.groupby('controller').agg({
            'avg_delay': ['mean', 'std'],
            'avg_queue_length': ['mean', 'std'],
            'throughput': ['mean', 'std'],
            'fairness_index': ['mean', 'std']
        }).round(2)
        
        st.dataframe(summary_stats, use_container_width=True)
        
        # Time series charts
        col1, col2 = st.columns(2)
        
        with col1:
            delay_fig = create_time_series_chart(df_filtered, 'avg_delay', 'Average Delay Over Episodes')
            st.plotly_chart(delay_fig, use_container_width=True)
        
        with col2:
            throughput_fig = create_time_series_chart(df_filtered, 'throughput', 'Throughput Over Episodes')
            st.plotly_chart(throughput_fig, use_container_width=True)
    
    else:
        st.warning("No controllers selected. Please select at least one controller from the sidebar.")
    
    # Training progress section (only for RL agents)
    if any(agent in selected_controllers for agent in ['DQN', 'PPO']):
        st.subheader("🧠 RL Agent Training Progress")
        
        training_fig = create_training_progress_chart(training_df)
        st.plotly_chart(training_fig, use_container_width=True)
        
        # Training metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("DQN Final Reward", f"{training_df[training_df['agent'] == 'DQN']['reward'].iloc[-1]:.1f}")
            st.metric("DQN Final Loss", f"{training_df[training_df['agent'] == 'DQN']['loss'].iloc[-1]:.4f}")
        
        with col2:
            st.metric("PPO Final Reward", f"{training_df[training_df['agent'] == 'PPO']['reward'].iloc[-1]:.1f}")
            st.metric("PPO Final Loss", f"{training_df[training_df['agent'] == 'PPO']['loss'].iloc[-1]:.4f}")
    
    # Performance insights
    st.subheader("💡 Performance Insights")
    
    if not df_filtered.empty:
        avg_metrics = df_filtered.groupby('controller')['avg_delay'].mean().sort_values()
        best_controller = avg_metrics.index[0]
        worst_controller = avg_metrics.index[-1]
        improvement = ((avg_metrics[worst_controller] - avg_metrics[best_controller]) / avg_metrics[worst_controller]) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"🏆 Best Controller: **{best_controller}**")
            st.info(f"Average Delay: {avg_metrics[best_controller]:.1f} seconds")
        
        with col2:
            st.warning(f"📊 Baseline: **{worst_controller}**")
            st.info(f"Average Delay: {avg_metrics[worst_controller]:.1f} seconds")
        
        with col3:
            st.success(f"🚀 Improvement: **{improvement:.1f}%**")
            st.info("Delay reduction vs baseline")
    
    # Footer
    st.markdown("---")
    st.markdown("**Traffic Signal Control System** - Deep Reinforcement Learning Dashboard | Built with Streamlit")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()