import pygame
import pygame_menu as pm

import q_learning
import json
import os
import numpy as np

from pygame_env import PygameEnvironment

pygame.init()

SCREEN_SIZE = [("500x500", 500), ("600x600", 600), ("700x700", 700), ("800x800", 800)]
LEARNING_MODEL = [("Q Learning", "q_learning")]
MODE = [("TRAINING", "train"), ("EVALUATION", "eval")]

# Standard RGB colors 
RED = (255, 0, 0) 
GREEN = (0, 255, 0) 
BLUE = (0, 0, 255) 
CYAN = (0, 100, 100) 
BLACK = (0, 0, 0) 
WHITE = (255, 255, 255) 

# Training parameters
n_training_episodes = 5  # Total training episodes
learning_rate = 0.7  # Learning rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

# game parameters
screen_height = 500
screen_width = 500
model = "q_learning"
mode = "train"

def set_screen(value, selected_size):
    global screen_width, screen_height
    screen_width = selected_size
    screen_height = selected_size

def set_model(value, selected_model):
    global model
    model = selected_model

def set_mode(value, selected_mode):
    global mode
    mode = selected_mode

def main_menu():
    screen = pygame.display.set_mode((SCREEN_SIZE[0][1],SCREEN_SIZE[0][1]))
    #screen.fill(BLACK)

    # Creating the settings menu 
    settings = pm.Menu(title="Settings", 
                       width=SCREEN_SIZE[0][1], 
                       height=SCREEN_SIZE[0][1], 
                       theme=pm.themes.THEME_GREEN) 
    
    settings.add.dropselect(title="SCREEN SIZE", items=SCREEN_SIZE, onchange=set_screen,
                            dropselect_id="screen_size", default=0)

    settings.add.dropselect(title="LEARNING MODEL", items=LEARNING_MODEL, onchange=set_model,
                            dropselect_id="learning_model", default=0) 
    
    settings.add.dropselect(title="MODE", items=MODE, onchange=set_mode,
                            dropselect_id="mode", default=0) 

    mainMenu = pm.Menu(title="Main Menu",
                       width=SCREEN_SIZE[0][1], 
                       height=SCREEN_SIZE[0][1], 
                       theme=pm.themes.THEME_GREEN)
    
    mainMenu.add.button(title="PLAY", action=start_game, 
                        font_color=WHITE, background_color=GREEN) 
    
    # Dummy label to add some spacing between the settings button and Play button 
    mainMenu.add.label(title="") 
    
    # Settings button. If clicked, it takes to the settings menu 
    mainMenu.add.button(title="Settings", action=settings, font_color=WHITE, 
                        background_color=BLUE) 
  
    # Dummy label to add some spacing between the settings button and exit button 
    mainMenu.add.label(title="") 
  
    # Exit Button. If clicked, it closes the window 
    mainMenu.add.button(title="Exit", action=pm.events.EXIT, 
                        font_color=WHITE, background_color=RED) 
  
    mainMenu.mainloop(screen)

def start_game():
    cwd = os.getcwd()
    json_db_path = f"{cwd}/json_db"
    q_table_path = f"{json_db_path}/q_table.json"

    env = PygameEnvironment(screen_width, screen_height)
    env.reset()
    
    Qtable, json_dict = q_learning.set_q_table(env, q_table_path, screen_width, screen_height)
    if mode == "train":
        Qtable, episode = q_learning.train(env, min_epsilon, max_epsilon, decay_rate, Qtable)
        q_learning.save_q_table(json_dict, q_table_path, screen_width, screen_height, Qtable, episode)
    elif mode == "eval":
        mean_reward, std_reward, episodes = q_learning.evaluate_agent(env, Qtable)
        q_learning.save_eval(json_dict, q_table_path, screen_width, screen_height, mean_reward, std_reward, episodes)
    
    pygame.quit()


if __name__ == "__main__":
    main_menu()