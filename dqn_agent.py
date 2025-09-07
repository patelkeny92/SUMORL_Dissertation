"""
Deep Q-Network (DQN) Agent for Adaptive Traffic Signal Control

This module implements a DQN agent that learns to control traffic signals
to minimize waiting time and queue length while maximizing throughput.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from collections import deque
import random
import pickle
import os


class DQNAgent:
    """
    Deep Q-Network agent for traffic signal control.
    
    The agent observes traffic state and learns optimal signal timing
    through interaction with the SUMO traffic simulation environment.
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 gamma=0.95, batch_size=32, memory_size=2000):
        """
        Initialize the DQN Agent.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions (traffic light phases)
            learning_rate (float): Learning rate for neural network
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Exploration decay rate
            gamma (float): Discount factor for future rewards
            batch_size (int): Size of training batch
            memory_size (int): Size of experience replay buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # Build the main and target networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
        # Training statistics
        self.loss_history = []
        self.reward_history = []
        
    def _build_model(self):
        """
        Build the neural network model for Q-value approximation.
        
        Returns:
            keras.Model: Compiled neural network model
        """
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
        
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether episode is finished
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state (np.array): Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])
        
    def replay(self):
        """
        Train the agent on a batch of experiences from replay buffer.
        
        Returns:
            float: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Compute target Q-values using target network
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        targets = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the network
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
        
    def update_target_network(self):
        """Update target network with weights from main network."""
        self.target_network.set_weights(self.q_network.get_weights())
        
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        self.q_network.save(filepath)
        
        # Save additional agent parameters
        agent_params = {
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history
        }
        
        with open(f"{filepath}_params.pkl", 'wb') as f:
            pickle.dump(agent_params, f)
            
    def load_model(self, filepath):
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.q_network = keras.models.load_model(filepath)
        self.update_target_network()
        
        # Load additional agent parameters
        param_file = f"{filepath}_params.pkl"
        if os.path.exists(param_file):
            with open(param_file, 'rb') as f:
                params = pickle.load(f)
                self.epsilon = params.get('epsilon', self.epsilon_min)
                self.loss_history = params.get('loss_history', [])
                self.reward_history = params.get('reward_history', [])