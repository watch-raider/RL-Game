import pygame
import random
from enum import Enum
import time
import numpy as np

learning_rate = 0.7  # Learning rate
gamma = 0.95  # Discounting rate

def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space.n, action_space.n))
    return Qtable
    
def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])

    return action
    
def epsilon_greedy_policy(Qtable, state, epsilon, action_space):
    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # else --> exploration
    else:
        action = action_space.sample()

    return action
    
def train(env, n_training_episodes, min_epsilon, max_epsilon, decay_rate, Qtable):
    action = 1

    for episode in range(n_training_episodes):
        print(episode)
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # Reset the environment
        state = env.reset()

        # repeat
        while env.run:
            current_time = time.time()
            if current_time - env.last_action_time > env.action_delay:
                action = epsilon_greedy_policy(Qtable, state, epsilon, env.action_space)

                env.agent_step(action)
                env.last_action_time = current_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.run = False

                if event.type == pygame.KEYDOWN:
                    new_state, reward = env.human_step(event)

                    # Take action At and observe Rt+1 and St+1
                    # Take the action (a) and observe the outcome state(s') and reward (r)

                    # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                    Qtable[state][action] = Qtable[state][action] + learning_rate * (
                        reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
                    )

                    # Our next state is the new state
                    state = new_state

                    # Choose the action At using epsilon greedy policy
                    action = epsilon_greedy_policy(Qtable, new_state, epsilon, env.action_space)
                    env.agent_step(action)
                    env.last_action_time = env.current_time

            if state == 0:
                break

            env.render()

    return Qtable