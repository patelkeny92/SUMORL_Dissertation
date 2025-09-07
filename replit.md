# Traffic Signal Control with Deep Reinforcement Learning

## Overview

This project implements adaptive traffic signal control using Deep Reinforcement Learning (DRL) agents integrated with SUMO (Simulation of Urban Mobility) traffic simulation. The system trains and evaluates multiple RL algorithms (DQN and PPO) against traditional baseline controllers to optimize traffic flow by minimizing waiting times, reducing queue lengths, and maximizing throughput.

The core system consists of a SUMO-based traffic environment that serves as a realistic simulation platform, two different RL agents (Deep Q-Network and Proximal Policy Optimization), baseline controllers for comparison, and comprehensive evaluation metrics to assess performance across multiple traffic scenarios.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Project Structure
```
├── config/                    # SUMO configuration files
│   ├── traffic_network.net.xml      # Traffic network topology
│   ├── traffic_routes.rou.xml       # Vehicle routes and flows
│   └── traffic_simulation.sumocfg   # Main SUMO configuration
├── src/                       # Source code modules
│   ├── dqn_agent.py                 # Deep Q-Network implementation
│   ├── ppo_agent.py                 # Proximal Policy Optimization implementation
│   ├── traffic_environment.py       # SUMO-TraCI interface
│   ├── evaluation_metrics.py        # Performance analysis tools
│   └── main.py                      # Main training/evaluation script
├── data/                      # Generated data and results
├── models/                    # Trained model storage
├── results/                   # Evaluation outputs
├── demo_simulation.py         # System demonstration script
└── run_simulation.py          # Full training execution script
```

### Environment Management
The system uses SUMO traffic simulation as the core environment, accessed through the TraCI API for real-time interaction. The `TrafficEnvironment` class abstracts the simulation interface, managing state representation (16-dimensional state space covering 4 traffic approaches with 4 features each), action execution (traffic light phase changes), and reward calculation based on traffic performance metrics.

### Agent Architecture
Two distinct RL algorithms are implemented:
- **DQN Agent**: Uses deep neural networks for value function approximation with experience replay buffer and epsilon-greedy exploration strategy (128-128-64 layer architecture)
- **PPO Agent**: Implements actor-critic architecture with clipped objective function for more stable policy gradient learning (separate actor and critic networks)

Both agents share a common interface for training and evaluation, allowing easy comparison and benchmarking. Models are saved using TensorFlow/Keras format for persistence.

### Baseline Controllers
Traditional traffic control methods are implemented for performance comparison:
- **Fixed-Time Controller**: Uses predetermined signal timing cycles (30s green phases)
- **Actuated Controller**: Adjusts timing based on vehicle detection with configurable min/max green times

### Evaluation Framework
The `TrafficMetrics` class provides comprehensive performance analysis including:
- Statistical summary calculations across episodes (mean, std deviation)
- Comparative analysis between different controllers
- Jain's fairness index calculation for traffic equity
- Visualization capabilities for performance trends with matplotlib/seaborn

### Training Pipeline
The main training loop orchestrates the entire process:
- Environment initialization with SUMO configuration
- Agent training with episode management and experience replay
- Performance data collection and real-time monitoring
- Model persistence for trained agents with parameter saving

## External Dependencies

### Simulation Platform
- **SUMO (Simulation of Urban Mobility)**: Traffic simulation engine providing realistic vehicle behavior and traffic dynamics
- **TraCI API**: Interface for real-time communication between RL agents and SUMO simulation

### Machine Learning Framework
- **TensorFlow/Keras**: Deep learning framework for implementing neural network architectures in both DQN and PPO agents
- **NumPy**: Numerical computing for state representation and mathematical operations

### Data Analysis and Visualization
- **Pandas**: Data manipulation and analysis for performance metrics
- **Matplotlib**: Plotting and visualization for training curves and performance comparisons
- **Seaborn**: Statistical visualization for advanced metric analysis

### System Dependencies
- **Python Standard Library**: Collections, pickle, random, os for basic functionality
- **XML Processing**: Built-in xml.etree.ElementTree for SUMO configuration file handling