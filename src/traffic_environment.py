"""
SUMO Traffic Environment for Reinforcement Learning

This module provides the interface between SUMO traffic simulation
and reinforcement learning agents using TraCI API.
"""

import os
import sys
import numpy as np
import traci
import sumolib
from collections import defaultdict
import xml.etree.ElementTree as ET


class TrafficEnvironment:
    """
    SUMO-based traffic environment for RL agent training and evaluation.
    
    Manages the traffic simulation, state representation, action execution,
    and reward calculation for adaptive traffic signal control.
    """
    
    def __init__(self, config_file="traffic_simulation.sumocfg", 
                 gui=False, step_length=10, max_steps=3600):
        """
        Initialize the traffic environment.
        
        Args:
            config_file (str): SUMO configuration file
            gui (bool): Whether to use SUMO GUI
            step_length (int): Action duration in simulation steps
            max_steps (int): Maximum simulation steps per episode
        """
        self.config_file = config_file
        self.use_gui = gui
        self.step_length = step_length
        self.max_steps = max_steps
        self.current_step = 0
        
        # Traffic light control
        self.junction_id = "junction"
        self.traffic_light_phases = {
            0: "GrGr",  # North-South green, East-West red
            1: "rGrG"   # North-South red, East-West green
        }
        
        # State and action spaces
        self.state_size = 16  # 4 approaches × 4 features each
        self.action_size = len(self.traffic_light_phases)
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(float)
        
        # Initialize SUMO
        self._init_sumo()
        
    def _init_sumo(self):
        """Initialize SUMO simulation."""
        if 'SUMO_HOME' in os.environ:
            sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui' if self.use_gui else 'sumo')
        else:
            sumo_binary = "sumo-gui" if self.use_gui else "sumo"
            
        self.sumo_cmd = [sumo_binary, "-c", self.config_file, "--waiting-time-memory", "300"]
        
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            np.array: Initial state
        """
        # Close existing SUMO connection if any
        try:
            traci.close()
        except:
            pass
            
        # Start new SUMO simulation
        traci.start(self.sumo_cmd)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_metrics = defaultdict(float)
        
        # Initialize traffic light
        traci.trafficlight.setProgram(self.junction_id, "0")
        
        # Run a few steps to populate the simulation
        for _ in range(5):
            traci.simulationStep()
            
        return self._get_state()
        
    def step(self, action):
        """
        Execute action and advance simulation.
        
        Args:
            action (int): Traffic light phase to set
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Set traffic light phase
        phase = self.traffic_light_phases[action]
        traci.trafficlight.setRedYellowGreenState(self.junction_id, phase)
        
        # Store metrics before action
        pre_metrics = self._collect_metrics()
        
        # Advance simulation
        for _ in range(self.step_length):
            if traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
            self.current_step += 1
            
        # Store metrics after action
        post_metrics = self._collect_metrics()
        
        # Calculate reward
        reward = self._calculate_reward(pre_metrics, post_metrics)
        
        # Get next state
        next_state = self._get_state()
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps or 
                traci.simulation.getMinExpectedNumber() <= 0)
        
        # Prepare info dictionary
        info = {
            'metrics': post_metrics,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
        
    def _get_state(self):
        """
        Get current state representation.
        
        Returns:
            np.array: State vector with traffic features
        """
        state = []
        
        # Get state for each approach (North, South, East, West)
        approaches = ["north_approach", "south_approach", "east_approach", "west_approach"]
        
        for approach in approaches:
            # Queue length (number of halting vehicles)
            try:
                queue_length = traci.edge.getLastStepHaltingNumber(approach)
            except:
                queue_length = 0
                
            # Waiting time (average waiting time)
            try:
                waiting_time = traci.edge.getWaitingTime(approach)
            except:
                waiting_time = 0
                
            # Lane occupancy (percentage)
            try:
                occupancy = traci.edge.getLastStepOccupancy(approach) * 100
            except:
                occupancy = 0
                
            # Vehicle count
            try:
                vehicle_count = traci.edge.getLastStepVehicleNumber(approach)
            except:
                vehicle_count = 0
                
            state.extend([queue_length, waiting_time, occupancy, vehicle_count])
            
        return np.array(state, dtype=np.float32)
        
    def _collect_metrics(self):
        """
        Collect performance metrics from simulation.
        
        Returns:
            dict: Performance metrics
        """
        metrics = {}
        
        # Get metrics for all approaches
        approaches = ["north_approach", "south_approach", "east_approach", "west_approach"]
        
        total_waiting_time = 0
        total_queue_length = 0
        total_vehicles = 0
        
        for approach in approaches:
            try:
                waiting_time = traci.edge.getWaitingTime(approach)
                queue_length = traci.edge.getLastStepHaltingNumber(approach)
                vehicle_count = traci.edge.getLastStepVehicleNumber(approach)
                
                total_waiting_time += waiting_time
                total_queue_length += queue_length
                total_vehicles += vehicle_count
                
            except:
                continue
                
        metrics['total_waiting_time'] = total_waiting_time
        metrics['total_queue_length'] = total_queue_length
        metrics['total_vehicles'] = total_vehicles
        metrics['average_waiting_time'] = total_waiting_time / max(total_vehicles, 1)
        
        # Throughput (vehicles that have left the network)
        try:
            metrics['throughput'] = traci.simulation.getArrivedNumber()
        except:
            metrics['throughput'] = 0
            
        return metrics
        
    def _calculate_reward(self, pre_metrics, post_metrics):
        """
        Calculate reward based on traffic performance.
        
        Args:
            pre_metrics (dict): Metrics before action
            post_metrics (dict): Metrics after action
            
        Returns:
            float: Reward value
        """
        # Reward components
        waiting_time_reward = -(post_metrics['total_waiting_time'] - pre_metrics['total_waiting_time'])
        queue_length_reward = -(post_metrics['total_queue_length'] - pre_metrics['total_queue_length'])
        throughput_reward = (post_metrics['throughput'] - pre_metrics['throughput']) * 10
        
        # Combined reward with weights
        reward = (0.4 * waiting_time_reward + 
                 0.4 * queue_length_reward + 
                 0.2 * throughput_reward)
        
        # Normalize reward
        reward = reward / 100.0
        
        # Update episode metrics
        self.episode_metrics['total_reward'] += reward
        self.episode_metrics['waiting_time'] = post_metrics['average_waiting_time']
        self.episode_metrics['queue_length'] = post_metrics['total_queue_length']
        self.episode_metrics['throughput'] = post_metrics['throughput']
        
        return reward
        
    def close(self):
        """Close the SUMO simulation."""
        try:
            traci.close()
        except:
            pass
            
    def get_episode_metrics(self):
        """
        Get metrics for the completed episode.
        
        Returns:
            dict: Episode performance metrics
        """
        return dict(self.episode_metrics)


class BaselineController:
    """Base class for baseline traffic signal controllers."""
    
    def __init__(self, environment):
        self.env = environment
        
    def get_action(self, state):
        """Get action from baseline controller."""
        raise NotImplementedError


class FixedTimeController(BaselineController):
    """Fixed-time traffic signal controller."""
    
    def __init__(self, environment, cycle_time=66):
        """
        Initialize fixed-time controller.
        
        Args:
            environment: Traffic environment
            cycle_time (int): Total cycle time in seconds
        """
        super().__init__(environment)
        self.cycle_time = cycle_time
        self.phase_times = [30, 30]  # Green times for each phase
        self.current_phase = 0
        self.phase_start_time = 0
        
    def get_action(self, state):
        """
        Get action based on fixed timing.
        
        Args:
            state: Current state (unused in fixed-time control)
            
        Returns:
            int: Action (traffic light phase)
        """
        current_time = self.env.current_step
        time_in_phase = current_time - self.phase_start_time
        
        if time_in_phase >= self.phase_times[self.current_phase]:
            # Switch to next phase
            self.current_phase = (self.current_phase + 1) % len(self.phase_times)
            self.phase_start_time = current_time
            
        return self.current_phase


class ActuatedController(BaselineController):
    """Actuated (vehicle-responsive) traffic signal controller."""
    
    def __init__(self, environment, min_green=10, max_green=60, extension=3):
        """
        Initialize actuated controller.
        
        Args:
            environment: Traffic environment
            min_green (int): Minimum green time
            max_green (int): Maximum green time
            extension (int): Green extension per vehicle detection
        """
        super().__init__(environment)
        self.min_green = min_green
        self.max_green = max_green
        self.extension = extension
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_detection_time = 0
        
    def get_action(self, state):
        """
        Get action based on vehicle detection.
        
        Args:
            state (np.array): Current state with vehicle counts
            
        Returns:
            int: Action (traffic light phase)
        """
        current_time = self.env.current_step
        time_in_phase = current_time - self.phase_start_time
        
        # Extract vehicle counts for current phase
        if self.current_phase == 0:  # North-South phase
            vehicle_count = state[3] + state[7]  # North + South vehicle counts
        else:  # East-West phase
            vehicle_count = state[11] + state[15]  # East + West vehicle counts
            
        # Check for vehicle detection (extension logic)
        if vehicle_count > 0:
            self.last_detection_time = current_time
            
        # Determine if phase should change
        time_since_detection = current_time - self.last_detection_time
        should_extend = (time_since_detection <= self.extension and 
                        time_in_phase < self.max_green)
        
        if (time_in_phase >= self.min_green and not should_extend):
            # Switch to next phase
            self.current_phase = (self.current_phase + 1) % 2
            self.phase_start_time = current_time
            self.last_detection_time = current_time
            
        return self.current_phase