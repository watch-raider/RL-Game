from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import time

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_start = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self):
        # Update episode stats
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        print(f"step-{self.num_timesteps} reward: {self.locals['rewards'][0]}")
        
        # Log episode info when done
        if self.locals['dones'][0]:
            print(f"Episode-{len(self.episode_rewards)} ended")
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log to tensorboard
            self.logger.record("episode_reward", self.current_episode_reward)
            self.logger.record("episode_length", self.current_episode_length)
            
            if self.locals['infos'][0].get('is_success', False):
                self.logger.record("success", 1.0)
            else:
                self.logger.record("success", 0.0)
                
            # Reset current episode stats
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Calculate mean reward over last 100 episodes
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                self.logger.record("mean_reward", mean_reward)
            
            # Log training speed
            elapsed_time = time.time() - self.training_start
            fps = self.num_timesteps / elapsed_time
            self.logger.record("fps", fps)
            
            # Dump logs
            self.logger.dump(self.num_timesteps)
            print(f"logs saved")
            
        return True