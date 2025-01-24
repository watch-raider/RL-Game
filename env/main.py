from copy import copy
import pygame

import q_learning

from pygame_env import PygameEnvironment

pygame.init()

# Training parameters
n_training_episodes = 3  # Total training episodes
learning_rate = 0.7  # Learning rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

if __name__ == "__main__":
    env = PygameEnvironment()
    env.reset()

    Qtable = q_learning.initialize_q_table(env.observation_space, env.action_space)
    Qtable = q_learning.train(env, n_training_episodes, min_epsilon, max_epsilon, decay_rate, Qtable)

    print(Qtable)
    
    pygame.quit()