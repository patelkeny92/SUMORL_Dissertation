"""
Proximal Policy Optimization (PPO) Agent for Adaptive Traffic Signal Control

This module implements a PPO agent as an alternative to DQN for traffic signal control.
PPO is often more stable and sample-efficient for continuous control problems.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pickle
import os


class PPOAgent:
    """
    Proximal Policy Optimization agent for traffic signal control.
    
    Uses actor-critic architecture with clipped objective for stable training.
    """
    
    def __init__(self, state_size, action_size, learning_rate=3e-4,
                 gamma=0.99, lam=0.95, clip_ratio=0.2, 
                 c1=1.0, c2=0.01, batch_size=64):
        """
        Initialize the PPO Agent.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate for neural networks
            gamma (float): Discount factor
            lam (float): GAE lambda parameter
            clip_ratio (float): PPO clipping parameter
            c1 (float): Value function loss coefficient
            c2 (float): Entropy bonus coefficient
            batch_size (int): Training batch size
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        
        # Build actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Training statistics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.reward_history = []
        
    def _build_actor(self):
        """
        Build the actor (policy) network.
        
        Returns:
            keras.Model: Actor network
        """
        inputs = keras.layers.Input(shape=(self.state_size,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(self.action_size, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
        
    def _build_critic(self):
        """
        Build the critic (value) network.
        
        Returns:
            keras.Model: Critic network
        """
        inputs = keras.layers.Input(shape=(self.state_size,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
        
    def act(self, state, training=True):
        """
        Choose action using the policy network.
        
        Args:
            state (np.array): Current state
            training (bool): Whether in training mode
            
        Returns:
            tuple: (action, action_probability, log_probability)
        """
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        if training:
            # Sample action from probability distribution
            action = np.random.choice(self.action_size, p=action_probs)
        else:
            # Choose action with highest probability
            action = np.argmax(action_probs)
            
        action_prob = action_probs[action]
        log_prob = np.log(action_prob + 1e-8)
        
        return action, action_prob, log_prob
        
    def get_value(self, state):
        """
        Get state value from critic network.
        
        Args:
            state (np.array): Current state
            
        Returns:
            float: State value
        """
        state = np.expand_dims(state, axis=0)
        return self.critic.predict(state, verbose=0)[0, 0]
        
    def compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards (list): List of rewards
            values (list): List of state values
            next_values (list): List of next state values
            dones (list): List of done flags
            
        Returns:
            tuple: (advantages, returns)
        """
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                gae = delta + self.gamma * self.lam * gae
                
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        return np.array(advantages), np.array(returns)
        
    def train(self, states, actions, old_log_probs, advantages, returns):
        """
        Train the agent using PPO algorithm.
        
        Args:
            states (np.array): Batch of states
            actions (np.array): Batch of actions
            old_log_probs (np.array): Batch of old log probabilities
            advantages (np.array): Batch of advantages
            returns (np.array): Batch of returns
            
        Returns:
            dict: Training metrics
        """
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train actor (policy network)
        with tf.GradientTape() as tape:
            current_probs = self.actor(states)
            current_log_probs = tf.math.log(tf.gather(current_probs, actions, batch_dims=1) + 1e-8)
            
            # Compute probability ratio
            ratio = tf.exp(current_log_probs - old_log_probs)
            
            # Compute clipped surrogate objective
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Entropy bonus
            entropy = -tf.reduce_mean(tf.reduce_sum(current_probs * tf.math.log(current_probs + 1e-8), axis=1))
            
            # Total actor loss
            actor_loss = policy_loss - self.c2 * entropy
            
        # Update actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Train critic (value network)
        with tf.GradientTape() as tape:
            current_values = tf.squeeze(self.critic(states))
            value_loss = tf.reduce_mean(tf.square(returns - current_values))
            
        # Update critic
        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Store training metrics
        self.policy_loss_history.append(float(policy_loss))
        self.value_loss_history.append(float(value_loss))
        self.entropy_history.append(float(entropy))
        
        return {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy)
        }
        
    def save_model(self, filepath):
        """
        Save the trained models.
        
        Args:
            filepath (str): Base path to save the models
        """
        self.actor.save(f"{filepath}_actor.h5")
        self.critic.save(f"{filepath}_critic.h5")
        
        # Save training history
        history = {
            'policy_loss_history': self.policy_loss_history,
            'value_loss_history': self.value_loss_history,
            'entropy_history': self.entropy_history,
            'reward_history': self.reward_history
        }
        
        with open(f"{filepath}_history.pkl", 'wb') as f:
            pickle.dump(history, f)
            
    def load_model(self, filepath):
        """
        Load pre-trained models.
        
        Args:
            filepath (str): Base path to the saved models
        """
        self.actor = keras.models.load_model(f"{filepath}_actor.h5")
        self.critic = keras.models.load_model(f"{filepath}_critic.h5")
        
        # Load training history
        history_file = f"{filepath}_history.pkl"
        if os.path.exists(history_file):
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
                self.policy_loss_history = history.get('policy_loss_history', [])
                self.value_loss_history = history.get('value_loss_history', [])
                self.entropy_history = history.get('entropy_history', [])
                self.reward_history = history.get('reward_history', [])