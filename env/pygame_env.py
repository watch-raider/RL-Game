import pygame

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box

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

class AgentAction(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    LIGHT_LEFT_TOGGLE = 4
    LIGHT_RIGHT_TOGGLE = 5
    LIGHT_UP_TOGGLE = 6
    LIGHT_DOWN_TOGGLE = 7

class PygameEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    LIGHT_OFF = (180, 180, 180)
    LIGHT_ON = (255, 165, 0)
    PLAYER_SIZE = 50

    def __init__(self, SCREEN_WIDTH=500, SCREEN_HEIGHT=500, render_mode="human"):
        """Initialize the environment with given screen dimensions and mode.
        
        Args:
            SCREEN_WIDTH (int): Width of game window
            SCREEN_HEIGHT (int): Height of game window 
            render_mode (str): Rendering mode
        """
        super().__init__()
        self.render_mode = render_mode
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        # Initialize game objects
        self.goal = None
        self.human = None 
        self.agent = None
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Calculate grid dimensions
        self.grid_size = 50
        self.n_cols = SCREEN_WIDTH // self.grid_size
        self.n_rows = SCREEN_HEIGHT // self.grid_size
        self.size = self.n_rows
        self.max_manhattan_dist = self.n_rows + self.n_cols - 2

        # Game state variables
        #self.timestep = 0
        self.score = 0
        self.run = True
        self.human_take_action = False
        
        # Display elements
        self.font = pygame.font.Font(None, 36)
        self.human_colour = RED
        self.agent_colour = BLUE
        self.goal_colour = BLACK
        self.light_radius = 5

        # Timing controls
        self.clock = pygame.time.Clock()
        self.current_time = time.time()
        self.last_action_time = time.time()
        self.action_delay = 2  # 2 seconds between actions

        self.reset_lights()

        # Set up action and observation spaces
        self.action_space = Discrete(len(AgentAction))
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),  # Agent pos, Human pos, Goal pos, Light states
            high=np.array([self.size**2, self.size**2, self.size**2, 1, 1, 1, 1]),
            dtype=np.int64
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: Optional seed for reproducibility
            options: Optional configuration
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)  # Important for reproducibility
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
        """Execute one environment step.
        
        Args:
            action: Agent's action
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        current_time = time.time()
        self.human_take_action = False
        truncated = False
        old_dist = self.calculate_manhattan_distance(self.human, self.goal)

        # Handle movement actions
        if action == AgentAction.MOVE_LEFT.value and self.agent.centerx > 25:
            self.agent.move_ip(-50, 0)
        elif action == AgentAction.MOVE_RIGHT.value and self.agent.centerx < self.screen_width - 25:
            self.agent.move_ip(50, 0)
        elif action == AgentAction.MOVE_UP.value and self.agent.centery > 25:
            self.agent.move_ip(0, -50)
        elif action == AgentAction.MOVE_DOWN.value and self.agent.centery < self.screen_height - 25:
            self.agent.move_ip(0, 50)
            
        # Handle light toggle actions
        elif action == AgentAction.LIGHT_UP_TOGGLE.value:
            self.light_t = self.LIGHT_ON if self.light_t == self.LIGHT_OFF else self.LIGHT_OFF
        elif action == AgentAction.LIGHT_RIGHT_TOGGLE.value:
            self.light_r = self.LIGHT_ON if self.light_r == self.LIGHT_OFF else self.LIGHT_OFF
        elif action == AgentAction.LIGHT_DOWN_TOGGLE.value:
            self.light_b = self.LIGHT_ON if self.light_b == self.LIGHT_OFF else self.LIGHT_OFF
        elif action == AgentAction.LIGHT_LEFT_TOGGLE.value:
            self.light_l = self.LIGHT_ON if self.light_l == self.LIGHT_OFF else self.LIGHT_OFF

        self.last_action_time = time.time()
        
        # Wait for human action
        while current_time - self.last_action_time < self.action_delay and self.human_take_action is False:
            current_time = time.time()
            self.human_step()

        observation = self._get_obs()
        info = self._get_info()
        reward, terminated = self.calculate_reward(old_dist)

        info['is_success'] = terminated  # Add this line to include episode end information

        return observation, reward, terminated, truncated, info

    def human_step(self):
        """Handle human player input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

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
        """Render the current game state."""
        self.screen.fill(BLACK)

        # Draw grid
        pix_square_size = self.screen_width / self.size
        for x in range(self.size + 1):
            pygame.draw.line(
                self.screen,
                WHITE,
                (0, pix_square_size * x),
                (self.screen_width, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.screen,
                WHITE,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.screen_width),
                width=3,
            )

        # Draw game objects
        pygame.draw.rect(self.screen, self.goal_colour, self.goal)
        pygame.draw.rect(self.screen, self.human_colour, self.human)
        pygame.draw.rect(self.screen, self.agent_colour, self.agent)

        # Draw lights
        light_t, light_r, light_b, light_l = self.set_light_xy()
        pygame.draw.circle(surface=self.screen, color=self.light_t, center=light_t, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_r, center=light_r, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_b, center=light_b, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_l, center=light_l, radius=self.light_radius)

        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (5, 5))

        pygame.display.update()

    def calculate_reward(self, old_dist):
        """Calculate reward based on game state changes.
        
        Args:
            old_dist: Previous manhattan distance between human and goal
            
        Returns:
            tuple: (reward, terminated)
        """
        new_dist = self.calculate_manhattan_distance(self.human, self.goal)
        reward = 0
        terminated = False

        # Distance-based rewards
        if new_dist > old_dist:
            reward -= 1
        elif new_dist < old_dist:
            reward += 1

        # Special state rewards
        if self.human.center == self.agent.center:
            reward -= 10
            self.score -= 10
        elif self.human.center == self.goal.center:
            reward += 10
            self.score += 10
            terminated = True

        return reward, terminated

    def _get_obs(self):
        """Get current observation of environment state."""
        
        H = self.calculate_location(self.human)
        G = self.calculate_location(self.goal)
        A = self.calculate_location(self.agent)
        TL = 1 if self.light_t == self.LIGHT_ON else 0
        RL = 1 if self.light_r == self.LIGHT_ON else 0
        BL = 1 if self.light_b == self.LIGHT_ON else 0
        LL = 1 if self.light_l == self.LIGHT_ON else 0
        
        return np.array([H, G, A, TL, RL, BL, LL])

    def _get_info(self):
        """Get additional information about environment state."""
        return {
            "distance": self.calculate_manhattan_distance(self.human, self.goal)
        }

    def calculate_location(self, target):
        """Convert target position to grid location index."""
        current_row, current_col = self.get_current_row_col(target)
        return current_row * self.n_cols + current_col

    def calculate_manhattan_distance(self, origin, target):
        """Calculate Manhattan distance between two points."""
        origin_row, origin_col = self.get_current_row_col(origin)
        target_row, target_col = self.get_current_row_col(target)
        return abs(origin_row - target_row) + abs(origin_col - target_col)

    def get_current_row_col(self, target):
        """Get grid row and column for target position."""
        rows = range(25, self.screen_height, 50)
        cols = range(25, self.screen_width, 50)
        return rows.index(target.centery), cols.index(target.centerx)

    def reset_lights(self):
        """Reset all lights to off state."""
        self.light_t = self.LIGHT_OFF
        self.light_r = self.LIGHT_OFF
        self.light_b = self.LIGHT_OFF
        self.light_l = self.LIGHT_OFF

    def set_light_xy(self):
        """Calculate light positions relative to agent."""
        light_1 = (self.agent.centerx, self.agent.top + self.light_radius)
        light_2 = (self.agent.right - self.light_radius, self.agent.centery)
        light_3 = (self.agent.centerx, self.agent.bottom - self.light_radius)
        light_4 = (self.agent.left + self.light_radius, self.agent.centery)

        return light_1, light_2, light_3, light_4