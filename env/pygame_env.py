import pygame

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete

import random
from enum import Enum
import time
import numpy as np

# Standard RGB colors 
RED = (255, 0, 0) 
GREEN = (0, 255, 0) 
BLUE = (0, 0, 255) 
CYAN = (0, 100, 100) 
BLACK = (0, 0, 0) 
WHITE = (255, 255, 255)

class DiscreteOrMulti(Enum):
    Discrete = 1
    Multi = 2

class AgentAction(Enum):
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    LIGHT_LEFT_TOGGLE = 5
    LIGHT_RIGHT_TOGGLE = 6
    LIGHT_UP_TOGGLE = 7
    LIGHT_DOWN_TOGGLE = 8

class PygameEnvironment(gym.Env):
    metadata = {
        "name": "custom_environment_v0",
        "render_modes": ["human"]
    }
    LIGHT_OFF = (180, 180, 180)
    LIGHT_ON = (255, 165, 0)
    PLAYER_SIZE = 50

    def __init__(self, SCREEN_WIDTH=500, SCREEN_HEIGHT=500, is_multi=True, render_mode="human"):
        self.render_mode = render_mode

        self.goal = None
        self.human = None
        self.agent = None
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.n_cols = len(range(25, SCREEN_WIDTH, 50))
        self.n_rows = len(range(25, SCREEN_HEIGHT, 50))
        self.max_manhattan_dist = self.n_rows + self.n_cols - 2
        self.timestep = 0
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.current_time = time.time()
        self.screen_width = SCREEN_WIDTH # The size of the PyGame window
        self.screen_height = SCREEN_HEIGHT # The size of the PyGame window
        self.size = len(range(25, self.screen_height, 50)) # The size of the square grid
        self.is_multi = is_multi
        self.human_take_action = False

        self.human_colour = RED  # Red for human
        self.agent_colour = BLUE  # Blue for agent
        self.goal_colour = BLACK  # Green for goal
        self.light_radius = 5
        self.reset_lights()

        self.run = True
        self.clock = pygame.time.Clock()
        self.last_action_time = time.time()
        self.action_delay = 2  # 2 seconds between actions

        self.action_space = Discrete(len(AgentAction))
        if not is_multi:
            self.observation_space = Discrete(self.n_cols + self.n_rows - 2)
        else:
            self.observation_space = MultiDiscrete([100, 100, 100, 2, 2, 2, 2])

    def reset(self,seed=None, options=None):
        positions_x = [x for x in range(0, self.screen_width, 50)]
        positions_y = [y for y in range(0, self.screen_height, 50)]

        # Generate non-overlapping positions for human, agent, and goal
        while True:
            human_pos = (random.choice(positions_x), random.choice(positions_y))
            agent_pos = (random.choice(positions_x), random.choice(positions_y))
            goal_pos = (random.choice(positions_x), random.choice(positions_y))

            if human_pos != agent_pos and human_pos != goal_pos and agent_pos != goal_pos:
                break

        self.human = pygame.Rect(human_pos[0], human_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.agent = pygame.Rect(agent_pos[0], agent_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.goal = pygame.Rect(goal_pos[0], goal_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)

        self.reset_lights()

        self.human_take_action = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Execute actions
        current_time = time.time()
        self.human_take_action = False
        truncated = False  # we do not limit the number of steps here
        old_dist = self.calculate_manhattan_distance(self.human, self.goal)

        if action == AgentAction.MOVE_LEFT.value and self.agent.centerx > 25:
            self.agent.move_ip(-50, 0)
        elif action == AgentAction.MOVE_RIGHT.value and self.agent.centerx < self.screen_width - 25:
            self.agent.move_ip(50, 0)
        elif action == AgentAction.MOVE_UP.value and self.agent.centery > 25:
            self.agent.move_ip(0, -50)
        elif action == AgentAction.MOVE_DOWN.value and self.agent.centery < self.screen_height - 25:
            self.agent.move_ip(0, 50)
        elif action == AgentAction.LIGHT_UP_TOGGLE.value:
            self.light_t = self.LIGHT_ON if self.light_t == self.LIGHT_OFF else self.LIGHT_OFF
        elif action == AgentAction.LIGHT_RIGHT_TOGGLE.value:
            self.light_r = self.LIGHT_ON if self.light_r == self.LIGHT_OFF else self.LIGHT_OFF
        elif action == AgentAction.LIGHT_DOWN_TOGGLE.value:
            self.light_b = self.LIGHT_ON if self.light_b == self.LIGHT_OFF else self.LIGHT_OFF
        elif action == AgentAction.LIGHT_LEFT_TOGGLE.value:
            self.light_l = self.LIGHT_ON if self.light_l == self.LIGHT_OFF else self.LIGHT_OFF

        self.last_action_time = time.time()
        
        while current_time - self.last_action_time < self.action_delay and self.human_take_action is False:
            current_time = time.time()
            self.human_step()

        observation = self._get_obs()
        info = self._get_info()
        reward, terminated = self.calculate_reward(old_dist)
        self.timestep += 1

        return observation, reward, terminated, truncated, info

    def human_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.human.centerx > 25:
                    self.human.move_ip(-50, 0)
                    self.human_take_action = True
                elif event.key == pygame.K_RIGHT and self.human.centerx < self.screen_width - 25:
                    self.human.move_ip(50, 0)
                    self.human_take_action = True
                elif event.key == pygame.K_UP and self.human.centery > 25:
                    self.human.move_ip(0, -50)
                    self.human_take_action = True
                elif event.key == pygame.K_DOWN and self.human.centery < self.screen_height - 25:
                    self.human.move_ip(0, 50)
                    self.human_take_action = True

    def render(self):
        """Renders the environment."""
        self.screen.fill(BLACK)

        pix_square_size = (
            self.screen_width / self.size
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                self.screen,
                (255, 255, 255),
                (0, pix_square_size * x),
                (self.screen_width, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.screen,
                (255, 255, 255),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.screen_width),
                width=3,
            )

        pygame.draw.rect(self.screen, self.goal_colour, self.goal)
        pygame.draw.rect(self.screen, self.human_colour, self.human)
        pygame.draw.rect(self.screen, self.agent_colour, self.agent)

        light_t, light_r, light_b, light_l = self.set_light_xy()

        pygame.draw.circle(surface=self.screen, color=self.light_t, center=light_t, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_r, center=light_r, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_b, center=light_b, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_l, center=light_l, radius=self.light_radius)

        # Draw the score to the screen
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (5, 5))

        pygame.display.update()

    def calculate_reward(self, old_dist): 
        new_dist = self.calculate_manhattan_distance(self.human, self.goal)
        reward = 0
        terminated = False

        if new_dist > old_dist:
            reward -= 1
        elif new_dist < old_dist:
            reward += 1

        if self.human.center == self.agent.center:
            reward -= 10
            self.score -= 10

        elif self.human.center == self.goal.center:
            reward += 10
            self.score += 10
            terminated = True

        return reward, terminated

    def _get_obs(self):
        if not self.is_multi:
            return self.calculate_manhattan_distance(self.human, self.goal)
        else:
            H = self.calculate_location(self.human)
            G = self.calculate_location(self.goal)
            A = self.calculate_location(self.agent)
            TL =  1 if self.light_t == self.LIGHT_ON else 0
            RL =  1 if self.light_r == self.LIGHT_ON else 0
            BL =  1 if self.light_b == self.LIGHT_ON else 0
            LL =  1 if self.light_l == self.LIGHT_ON else 0
        return np.array([H, G, A, TL,  RL,  BL,  LL])

    def _get_info(self):
        return {
            "distance": self.calculate_manhattan_distance(self.human, self.goal)
        }

    def calculate_location(self, target):
        current_row, current_col = self.get_current_row_col(target)
        return current_row*self.n_cols+current_col

    def calculate_manhattan_distance(self, origin, target):
        origin_row, origin_col = self.get_current_row_col(origin)
        target_row, target_col = self.get_current_row_col(target)
        return abs(origin_row - target_row) + abs(origin_col - target_col)

    def get_current_row_col(self, target):
        rows = range(25, self.screen_height, 50)
        cols = range(25, self.screen_width, 50)
        return rows.index(target.centery), cols.index(target.centerx)

    def reset_lights(self):
        self.light_t = self.LIGHT_OFF
        self.light_r = self.LIGHT_OFF
        self.light_b = self.LIGHT_OFF
        self.light_l = self.LIGHT_OFF

    def set_light_xy(self):
        light_1 = (self.agent.centerx, self.agent.top+self.light_radius)
        light_2 = (self.agent.right-self.light_radius, self.agent.centery)
        light_3 = (self.agent.centerx, self.agent.bottom-self.light_radius)
        light_4 = (self.agent.left+self.light_radius, self.agent.centery)

        return light_1, light_2, light_3, light_4