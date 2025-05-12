import pygame
import pygame_menu as pm

import os
import numpy as np

from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy

import supersuit as ss

from env.multi_pygame_env import PygameEnvironment
from env.callbacks.train_callback import TrainLogger
from tuning import define_model

import optuna
from pettingzoo.test import parallel_api_test

SCREEN_SIZE = [("400x400", 400), ("500x500", 500), ("600x600", 600), ("700x700", 700), ("800x800", 800)]
LEARNING_MODEL = [("PPO", "ppo"), ("TRPO", "trpo"), ("A2C", "a2c"), ("DQN", "dqn")]
MODE = [("TRAINING", "train"), ("EVALUATION", "eval"), ("TUNE", "tune")]
SELF_PLAY = [("HUMAN", False), ("AGENT", True)]

# Standard RGB colors 
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 100, 100)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# game parameters
screen_height = SCREEN_SIZE[1][1]
screen_width = SCREEN_SIZE[1][1]
model_name = LEARNING_MODEL[0][1]
mode = MODE[0][1]
self_play = SELF_PLAY[0][1]

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

def set_self_play(value, selected_play):
    global self_play
    self_play = selected_play

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
    
    settings.add.dropselect(title="SELF PLAY", items=SELF_PLAY, onchange=set_self_play,
                            dropselect_id="self_play", default=0) 

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
    global screen_width, screen_height, model_name, mode, self_play

    valid_models = ["a2c", "ppo", "trpo", "dqn"]
    if model_name not in valid_models:
        raise ValueError(f"Invalid model_name: {model_name}. Choose from {valid_models}")

    # Create environment with render_mode
    env = PygameEnvironment(
        SCREEN_WIDTH=screen_width, 
        SCREEN_HEIGHT=screen_height, 
        step_limit=100,
        self_play=self_play,
        model_name=model_name
    )

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    # Include grid size in model path
    model_path = f"models/{model_name}_model.zip"

    model_configs = {
        "a2c": {'policy': 'MlpPolicy', 'learning_rate': 0.0004027083606751521, 'n_steps': 20, 'gae_lambda': 0.9888133035583276, 'vf_coef': 0.6628359185991031, 'ent_coef': 0.009503796675472118},
        "ppo": {'policy': 'MlpPolicy', 'learning_rate': 0.0002189729779385502, 'batch_size': 128, 'n_steps': 1536, 'clip_range': 0.17761156384383014, 'n_epochs': 6},
        "trpo": {"policy": "MlpPolicy", 'learning_rate': 0.00020121633649761026, 'target_kl': 0.0069158498039823225, 'n_steps': 1536, 'batch_size': 128, 'gae_lambda': 0.9492984415265627},
        "dqn": {'policy': 'MlpPolicy', 'learning_rate': 0.000230912698109279, 'batch_size': 32, 'exploration_fraction': 0.05882560835087349, 'exploration_final_eps': 0.05917832438718523, 'target_update_interval': 3000}
    }

    model_class = {"a2c": A2C, "ppo": PPO, "trpo": TRPO, "dqn": DQN}[model_name]
    device = "cpu"
    n_trials=50
    
    if mode == "tune":
        print(f"Tuning hyperparameters for {model_name.upper()}...")

        # Setup callbacks
        log_callback = TrainLogger(log_dir="logs_tuning", rl_algorithm=model_name, game_size=screen_width, verbose=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(env, trial, model_name, log_callback),
            n_trials=n_trials
        )
        
        # Print the best hyperparameters
        print("Best hyperparameters:", study.best_params)
        print("Best mean reward:", study.best_value)

    if mode == "train":
        # Setup callbacks
        if self_play == True:
            log_dir = "logs"
        else:
            log_dir = "logs_human"
        log_callback = TrainLogger(log_dir=log_dir, rl_algorithm=model_name, game_size=screen_width)

        # Check if the model file exists
        if os.path.exists(model_path):
            print(f"Load and train existing {model_name.upper()} model...")
            model = model_class.load(model_path, env=env, device=device)
        else:
            # Define and Train the agent
            print(f"Training new {model_name.upper()} model...")
            model = model_class(env=env, device=device, **model_configs[model_name])
            #model = model_class(env=env, device=device, policy="MlpPolicy")
        
        model.learn(total_timesteps=250000, callback=log_callback)
        model.save(model_path)
        print(f"Model saved to path: {model_path}")

    elif mode == "eval":
        print(f"Evaluate {model_name.upper()} model...")
        
        # Load the trained model
        if os.path.exists(model_path):
            model = model_class.load(model_path, env=env, device=device)
                
            # Run evaluation
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
            print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        else:
            print(f"No trained model found at {model_path}. Please train the model first.")
        
    pygame.quit()

def objective(env, trial, model_name, log_callback, total_timesteps=10000):
    """Optuna objective function to optimize hyperparameters."""
    model = define_model(model_name, env, trial)
    
    model.learn(total_timesteps=total_timesteps, callback=log_callback)

    # Default to mean reward
    metric_value, _ = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    
    return metric_value

def main():
    # Initialize pygame first
    pygame.init()
    
    # Then create and run the menu
    main_menu()
    
    # Cleanup when done
    pygame.quit()

if __name__ == "__main__":
    main()