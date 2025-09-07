"""
Main Training and Evaluation Script for SUMO Traffic Signal Control

This script demonstrates the complete workflow for training and evaluating
Deep Reinforcement Learning agents (DQN and PPO) against baseline controllers
for adaptive traffic signal control using SUMO simulation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import argparse
import time

# Import our modules
from traffic_environment import TrafficEnvironment, FixedTimeController, ActuatedController
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from evaluation_metrics import TrafficMetrics


def train_dqn_agent(env: TrafficEnvironment, episodes: int = 100, 
                   save_model: bool = True) -> DQNAgent:
    """
    Train a DQN agent on the traffic environment.
    
    Args:
        env: Traffic environment
        episodes: Number of training episodes
        save_model: Whether to save the trained model
        
    Returns:
        DQNAgent: Trained DQN agent
    """
    print("Training DQN Agent...")
    print("-" * 20)
    
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        gamma=0.95
    )
    
    episode_rewards = []
    episode_losses = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
                
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
            
        episode_rewards.append(total_reward)
        agent.reward_history.append(total_reward)
        
        # Print progress
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_loss = np.mean(episode_losses[-20:]) if episode_losses else 0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
    
    if save_model:
        agent.save_model("trained_dqn_model.h5")
        print("DQN model saved to trained_dqn_model.h5")
    
    env.close()
    return agent


def train_ppo_agent(env: TrafficEnvironment, episodes: int = 100,
                   save_model: bool = True) -> PPOAgent:
    """
    Train a PPO agent on the traffic environment.
    
    Args:
        env: Traffic environment
        episodes: Number of training episodes
        save_model: Whether to save the trained model
        
    Returns:
        PPOAgent: Trained PPO agent
    """
    print("Training PPO Agent...")
    print("-" * 20)
    
    agent = PPOAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2
    )
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        
        # Collect episode data
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        total_reward = 0
        steps = 0
        
        while True:
            # Get action and value
            action, action_prob, log_prob = agent.act(state)
            value = agent.get_value(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store data
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
                
        # Calculate advantages and returns
        next_values = values[1:] + [0]  # Last value is 0 (terminal state)
        advantages, returns = agent.compute_gae(rewards, values, next_values, dones)
        
        # Train agent
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(log_probs)
        
        metrics = agent.train(states, actions, old_log_probs, advantages, returns)
        
        episode_rewards.append(total_reward)
        agent.reward_history.append(total_reward)
        
        # Print progress
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}")
    
    if save_model:
        agent.save_model("trained_ppo_model")
        print("PPO model saved to trained_ppo_model_*")
    
    env.close()
    return agent


def evaluate_controller(env: TrafficEnvironment, controller, controller_name: str,
                       episodes: int = 50) -> Dict[str, Any]:
    """
    Evaluate a controller on the traffic environment.
    
    Args:
        env: Traffic environment
        controller: Controller to evaluate
        controller_name: Name of the controller
        episodes: Number of evaluation episodes
        
    Returns:
        dict: Evaluation results
    """
    print(f"Evaluating {controller_name}...")
    print("-" * 30)
    
    metrics = TrafficMetrics()
    episode_results = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Get action from controller
            if hasattr(controller, 'get_action'):
                # Baseline controller
                action = controller.get_action(state)
            else:
                # RL agent
                action = controller.act(state, training=False)
                
            # Take action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
                
        # Get episode metrics
        episode_metrics = env.get_episode_metrics()
        episode_metrics['total_reward'] = total_reward
        episode_metrics['steps'] = steps
        
        episode_results.append(episode_metrics)
        metrics.add_episode_data(controller_name, episode_metrics)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{episodes} completed")
    
    env.close()
    
    # Calculate summary statistics
    summary_stats = metrics.calculate_summary_statistics(controller_name)
    fairness_index = metrics.calculate_fairness_index(controller_name)
    
    results = {
        'controller_name': controller_name,
        'episode_results': episode_results,
        'summary_stats': summary_stats,
        'fairness_index': fairness_index,
        'metrics_calculator': metrics
    }
    
    return results


def run_complete_evaluation(train_agents: bool = True, train_episodes: int = 100,
                          eval_episodes: int = 50):
    """
    Run complete training and evaluation pipeline.
    
    Args:
        train_agents: Whether to train RL agents
        train_episodes: Number of training episodes
        eval_episodes: Number of evaluation episodes
    """
    print("SUMO Traffic Signal Control - Deep Reinforcement Learning")
    print("=" * 60)
    print(f"Training Episodes: {train_episodes}")
    print(f"Evaluation Episodes: {eval_episodes}")
    print("=" * 60)
    
    # Initialize environment
    env = TrafficEnvironment(gui=False, step_length=10)
    
    # Store all evaluation results
    all_results = {}
    combined_metrics = TrafficMetrics()
    
    # 1. Train and evaluate DQN agent
    if train_agents:
        print("\n1. TRAINING DQN AGENT")
        print("=" * 25)
        
        dqn_agent = train_dqn_agent(env, episodes=train_episodes)
        
        print("\n2. EVALUATING DQN AGENT")
        print("=" * 26)
        
        dqn_results = evaluate_controller(env, dqn_agent, "DQN", eval_episodes)
        all_results['DQN'] = dqn_results
        
        # Add to combined metrics
        for ep_data in dqn_results['episode_results']:
            combined_metrics.add_episode_data('DQN', ep_data)
    
    # 2. Train and evaluate PPO agent
    if train_agents:
        print("\n3. TRAINING PPO AGENT")
        print("=" * 25)
        
        ppo_agent = train_ppo_agent(env, episodes=train_episodes)
        
        print("\n4. EVALUATING PPO AGENT")
        print("=" * 26)
        
        ppo_results = evaluate_controller(env, ppo_agent, "PPO", eval_episodes)
        all_results['PPO'] = ppo_results
        
        # Add to combined metrics
        for ep_data in ppo_results['episode_results']:
            combined_metrics.add_episode_data('PPO', ep_data)
    
    # 3. Evaluate Fixed-Time baseline
    print("\n5. EVALUATING FIXED-TIME CONTROLLER")
    print("=" * 36)
    
    fixed_time_controller = FixedTimeController(env)
    fixed_results = evaluate_controller(env, fixed_time_controller, "Fixed-Time", eval_episodes)
    all_results['Fixed-Time'] = fixed_results
    
    # Add to combined metrics
    for ep_data in fixed_results['episode_results']:
        combined_metrics.add_episode_data('Fixed-Time', ep_data)
    
    # 4. Evaluate Actuated baseline
    print("\n6. EVALUATING ACTUATED CONTROLLER")
    print("=" * 34)
    
    actuated_controller = ActuatedController(env)
    actuated_results = evaluate_controller(env, actuated_controller, "Actuated", eval_episodes)
    all_results['Actuated'] = actuated_results
    
    # Add to combined metrics
    for ep_data in actuated_results['episode_results']:
        combined_metrics.add_episode_data('Actuated', ep_data)
    
    # 5. Generate comprehensive evaluation report
    print("\n7. GENERATING EVALUATION REPORT")
    print("=" * 32)
    
    # Print summary
    combined_metrics.print_summary()
    
    # Generate detailed report
    combined_metrics.generate_performance_report("traffic_control_performance_report.txt")
    
    # Create performance plots
    combined_metrics.plot_performance_comparison("traffic_control_comparison.png")
    
    # Save all results
    combined_metrics.save_results("traffic_control_results.csv")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("- traffic_control_performance_report.txt: Detailed performance report")
    print("- traffic_control_comparison.png: Performance comparison plots")
    print("- traffic_control_results.csv: Raw evaluation data")
    if train_agents:
        print("- trained_dqn_model.h5: Trained DQN model")
        print("- trained_ppo_model_*: Trained PPO model files")


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='SUMO Traffic Signal Control with Deep RL')
    parser.add_argument('--train', action='store_true', default=True,
                       help='Train RL agents (default: True)')
    parser.add_argument('--train_episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                       help='Number of evaluation episodes (default: 50)')
    parser.add_argument('--gui', action='store_true', default=False,
                       help='Use SUMO GUI (default: False)')
    
    args = parser.parse_args()
    
    # Run the complete evaluation
    run_complete_evaluation(
        train_agents=args.train,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes
    )


if __name__ == "__main__":
    main()