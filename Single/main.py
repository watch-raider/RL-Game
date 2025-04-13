import pygame
import pygame_menu as pm

import os
import numpy as np

from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from env.pygame_env import PygameEnvironment
from env.callbacks.train_callback import TrainLogger, RandomLogger

SCREEN_SIZE = [("400x400", 400), ("500x500", 500), ("600x600", 600), ("700x700", 700), ("800x800", 800)]
LEARNING_MODEL = [("PPO", "ppo"), ("TRPO", "trpo"), ("A2C", "a2c"), ("RANDOM", "random")]
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
decay_rate = 0.05  # Exponential decay rate for exploration prob

# game parameters
screen_height = SCREEN_SIZE[1][1]
screen_width = SCREEN_SIZE[1][1]
model_name = LEARNING_MODEL[0][1]
mode = MODE[0][1]

def set_screen(value, selected_size):
    global screen_width, screen_height
    screen_width = selected_size
    screen_height = selected_size

def set_model(value, selected_model):
    global model_name
    model_name = selected_model

def set_mode(value, selected_mode):
    global mode
    mode = selected_mode

def main_menu():
    screen = pygame.display.set_mode((screen_height,screen_width))
    #screen.fill(BLACK)

    # Creating the settings menu 
    settings = pm.Menu(title="Settings", 
                       width=screen_width, 
                       height=screen_height, 
                       theme=pm.themes.THEME_GREEN) 
    
    settings.add.dropselect(title="SCREEN SIZE", items=SCREEN_SIZE, onchange=set_screen,
                            dropselect_id="screen_size", default=1)

    settings.add.dropselect(title="LEARNING MODEL", items=LEARNING_MODEL, onchange=set_model,
                            dropselect_id="learning_model", default=0) 
    
    settings.add.dropselect(title="MODE", items=MODE, onchange=set_mode,
                            dropselect_id="mode", default=0) 

    mainMenu = pm.Menu(title="Main Menu",
                       width=screen_width, 
                       height=screen_height, 
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
    global screen_width, screen_height, model_name, mode
    
    env = PygameEnvironment(
        SCREEN_WIDTH=screen_width, 
        SCREEN_HEIGHT=screen_height, 
        render_mode="human",
        step_limit=100)
    
    # Include grid size in model path
    model_path = f"./models/{model_name}_model.zip"
    

    if mode == "train":
        if model_name == "random":
            log_callback = RandomLogger(rl_algorithm=model_name, game_size=screen_width)
            observation, info = env.reset()

            termination, truncation = False, False
            timestep = 0

            while True:
                timestep += 1
                if termination or truncation:
                    observation, info = env.reset()
                # this is where you would insert your policy
                action = env.action_space.sample()

                observation, reward, termination, truncation, info = env.step(action)
                env.render()
                log_callback._on_step(reward=reward, info=info, timestep=timestep)
            env.close()
        else:
            # Setup callbacks
            log_callback = TrainLogger(rl_algorithm=model_name, game_size=screen_width)

            # Check if the model file exists
            if os.path.exists(model_path):
                print(f"Load and train existing {model_name.upper()} model...")
                if model_name == "a2c":
                    model = A2C.load(model_path, env=env, device="cpu")
                if model_name == "ppo":
                    model = PPO.load(model_path, env=env, device="cpu")
                elif model_name == "trpo":
                    model = TRPO.load(model_path, env=env, device="cpu")
            else:
                # Define and Train the agent
                print(f"Training new {model_name.upper()} model...")
                if model_name == "a2c":
                    model = A2C("MlpPolicy", env=env, device="cpu")
                if model_name == "ppo":
                    model = PPO("MlpPolicy", env=env, device="cpu")
                elif model_name == "trpo":
                    model = TRPO("MlpPolicy", env=env, device="cpu")
        
        
            model.learn(total_timesteps=10000, callback=log_callback)

    elif mode == "eval":
        print(f"Evaluate {model_name.upper()} model...")
        
        model_path = f"./models/{model_name}_model.zip"

        # Load the trained model
        if os.path.exists(model_path):
            if model_name == "a2c":
                model = A2C.load(model_path)
            if model_name == "ppo":
                model = PPO.load(model_path)
            elif model_name == "trpo":
                model = TRPO.load(model_path)
                
            # Run evaluation
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
            print(f"{model_name.upper()} evaluation results:")
            print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        else:
            print(f"No trained model found at {model_path}. Please train the model first.")
        
    pygame.quit()

def main():
    # Initialize pygame first
    pygame.init()
    
    # Then create and run the menu
    main_menu()
    
    # Cleanup when done
    pygame.quit()

if __name__ == "__main__":
    main()