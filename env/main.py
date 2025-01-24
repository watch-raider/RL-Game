from copy import copy
import pygame

import q_learning
import json
import os
import numpy as np

from pygame_env import PygameEnvironment

pygame.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 300

# Training parameters
n_training_episodes = 1  # Total training episodes
learning_rate = 0.7  # Learning rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

if __name__ == "__main__":
    cwd = os.getcwd()
    json_db_path = f"{cwd}/json_db"

    env = PygameEnvironment(SCREEN_WIDTH, SCREEN_HEIGHT)
    env.reset()
    
    q_table_path = f"{json_db_path}/q_table.json"
    isExist = os.path.exists(q_table_path)
    json_dict = {}

    if isExist:
        with open(q_table_path, 'r') as outfile:
            # Reading from json file
            json_dict = json.load(outfile)
            key_name = f"{SCREEN_WIDTH}_{SCREEN_HEIGHT}"
            if key_name in json_dict:
                print(key_name, type(json_dict))
                q_table_list = json_dict[key_name]
                Qtable = np.array(q_table_list)
            else:
                Qtable = Qtable = q_learning.initialize_q_table(env.observation_space, env.action_space)
    else:
        Qtable = q_learning.initialize_q_table(env.observation_space, env.action_space)
    
    Qtable = q_learning.train(env, n_training_episodes, min_epsilon, max_epsilon, decay_rate, Qtable)

    # Serialize JSON after converting NumPy array to list
    json_dict[f"{SCREEN_WIDTH}_{SCREEN_HEIGHT}"] = Qtable.tolist()
    arr_json = json.dumps(json_dict, indent=4)

    with open(q_table_path, 'w') as outfile:
        outfile.write(arr_json)
    
    pygame.quit()