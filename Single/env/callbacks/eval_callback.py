import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback

class EvalLogger(BaseCallback):
    """
    Callback for logging training metrics during reinforcement learning.
    
    Args:
        check_freq: Frequency (in steps) to log data
        log_dir: Directory to save logs
        verbose: Verbosity level
    """
    def __init__(self, check_freq=100, log_dir='logs', verbose=1, rl_algorithm='ppo', game_size=500):
        
        pass

    def _on_step(self) -> bool:
        """Called after each step in the environment"""

        
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
        plt.savefig(f"{plots_dir}/{self.algorithm}_{self.game_size}_eval_curves.png")
        plt.close()