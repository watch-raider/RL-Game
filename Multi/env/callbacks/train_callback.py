import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback

class TrainLogger(BaseCallback):
    """
    Callback for logging training metrics during reinforcement learning.
    
    Args:
        check_freq: Frequency (in steps) to log data
        log_dir: Directory to save logs
        verbose: Verbosity level
    """
    def __init__(self, check_freq=100, log_dir='logs', verbose=1, rl_algorithm='ppo', game_size=500, self_play=False):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.algorithm = rl_algorithm
        self.game_size = game_size
        self.self_play = self_play
        
        # Initialize metrics storage
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_times = []
        self.success_rate = []
        
        # For tracking current episode
        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.start_time = time.time()
        self.last_ep_time = time.time()

        # Track episode successes
        self.ep_count = 0
        self.success_count = 0
        self.timestep_count = 0
        
        # Create CSV log file with game size in the name
        self.log_file = f"{log_dir}/{self.algorithm}_{self.game_size}_train_log.csv"
        if self_play:
            self.log_file = f"{log_dir}/{self.algorithm}_{self.game_size}_human_train_log.csv"
        self.log_columns = ['timestep', 'episode', 'success', 'ep_reward', 'ep_length', 'ep_time', 'mean_reward', 'mean_length', 
                           'mean_time', 'success_rate', 'fps']
        
        # Initialize log dataframe
        if os.path.exists(self.log_file):
            self.log_df = pd.read_csv(self.log_file)
            # Check if the dataframe has any rows before accessing the last row
            if len(self.log_df) > 0:
                self.ep_count = self.log_df['episode'].iloc[-1]
                self.timestep_count = self.log_df['timestep'].iloc[-1]
                self.success_count = sum(self.log_df['success'].tolist())
                self.current_ep_reward = self.log_df['ep_reward'].iloc[-1]
                self.current_ep_length = self.log_df['ep_length'].iloc[-1]
                self.ep_rewards = self.log_df['ep_reward'].tolist()
                self.ep_lengths = self.log_df['ep_length'].tolist()
                self.ep_times = self.log_df['ep_time'].tolist()
                self.success_rate = self.log_df['success_rate'].tolist()
        else:
            self.log_df = pd.DataFrame(columns=self.log_columns)
            self.log_df.to_csv(self.log_file, index=False)
        

    def _on_step(self) -> bool:
        """Called after each step in the environment"""
        # Update current episode stats
        self.training_env.render()

        # In multi-agent scenario, rewards is a dictionary with agent and human rewards
        self.current_ep_reward += self.locals['rewards'][0]  # Use agent's reward
        
        self.current_ep_length += 1

        is_success = self.locals['infos'][0].get('is_success', False)
        is_truncated = self.locals['infos'][0].get('is_truncated', False)
        
        if is_success or is_truncated:
            # Track if this was a successful episode
            if is_success:
                self.success_count += 1
            
            # Record episode data
            self.ep_count += 1
            self.ep_rewards.append(self.current_ep_reward)
            self.ep_lengths.append(self.current_ep_length)
            
            # Track episode time
            current_time = time.time()
            self.ep_times.append(current_time - self.last_ep_time)
            self.last_ep_time = current_time
            
            # Reset episode tracking
            self.current_ep_reward = 0
            self.current_ep_length = 0
            
            # Calculate success rate over last 100 episodes (or all if < 100)
            self.success_rate.append(self.success_count / self.ep_count)
        
            # Calculate metrics
            mean_reward = np.mean(self.ep_rewards[-100:]) if self.ep_rewards else 0
            mean_length = np.mean(self.ep_lengths[-100:]) if self.ep_lengths else 0
            mean_time = np.mean(self.ep_times[-100:]) if self.ep_times else 0
            success_rate_value = self.success_rate[-1] if self.success_rate else 0
            fps = self.check_freq / (time.time() - self.start_time) if self.ep_times else 0
            
            if self.num_timesteps < self.timestep_count:
                self.num_timesteps += self.timestep_count

            # Log to console if verbose
            if self.verbose > 0:
                print(f"Steps: {self.num_timesteps}, Episodes: {self.ep_count}")
                print(f"Mean reward: {mean_reward:.2f}, Mean episode length: {mean_length:.2f}")
                print(f"Success rate: {success_rate_value:.2%}, FPS: {fps:.2f}")
            
            # Update log file
            log_data = {
                'timestep': self.num_timesteps,
                'episode': self.ep_count,
                'success': 1 if is_success else 0,
                'ep_reward': self.ep_rewards[-1],
                'ep_length': self.ep_lengths[-1],
                'ep_time': self.last_ep_time,
                'mean_reward': mean_reward,
                'mean_length': mean_length,
                'mean_time': mean_time, 
                'success_rate': success_rate_value,
                'fps': fps
            }
            
            # Append to CSV
            pd.DataFrame([log_data]).to_csv(self.log_file, mode='a', header=False, index=False)
            
            # Plot learning curves
            self.plot_learning_curves()
            
            # Reset timer for FPS calculation
            self.start_time = time.time()

            if self.ep_count % 5 == 0:
                path = f"./models/{self.algorithm}_model"
                self.model.save(path)
                print(f"Model saved to path: {path}")
        
        return True
    
    def plot_learning_curves(self):
        """Generate plots of the learning metrics"""
        if len(self.ep_rewards) < 2:
            return  # Not enough data to plot
            
        # Create plots directory
        plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Load data from CSV for plotting
        data = pd.read_csv(self.log_file)
        
        # Plot metrics
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward plot
        axs[0, 0].plot(data['episode'], data['mean_reward'])
        axs[0, 0].set_title('Mean Reward')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True)
        
        # Episode length plot
        axs[0, 1].plot(data['episode'], data['mean_length'])
        axs[0, 1].set_title('Mean Episode Length')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Steps')
        axs[0, 1].grid(True)
        
        # Success rate plot
        axs[1, 0].plot(data['episode'], data['success_rate'])
        axs[1, 0].set_title('Success Rate')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Rate')
        axs[1, 0].grid(True)
        
        # FPS plot
        axs[1, 1].plot(data['episode'], data['fps'])
        axs[1, 1].set_title('Training Speed (FPS)')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Frames per second')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        path = f"{plots_dir}/{self.algorithm}_{self.game_size}_train_curves.png"
        if self.self_play:
            path = f"{plots_dir}/{self.algorithm}_{self.game_size}_human_train_curves.png"
        plt.savefig(path)
        plt.close()