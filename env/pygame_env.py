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

    def __init__(self, SCREEN_WIDTH=500, SCREEN_HEIGHT=500, render_mode="human", step_limit=50):
        """Initialize the environment with given screen dimensions and mode.
        
        Args:
            SCREEN_WIDTH (int): Width of game window
            SCREEN_HEIGHT (int): Height of game window 
            render_mode (str): Rendering mode
        """
        super().__init__()
        self.render_mode = render_mode
        self.screen_width = SCREEN_WIDTH
        
        # Add footer height
        self.footer_height = 60
        self.game_height = SCREEN_HEIGHT
        self.screen_height = SCREEN_HEIGHT + self.footer_height
        
        # Initialize game objects
        self.goal = None
        self.human = None 
        self.agent = None
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, self.screen_height))
        
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
        self.step_limit = step_limit
        
        # Effect variables
        self.success_effect_timer = 0
        self.failure_effect_timer = 0
        self.effect_duration = 1.5 
        self.success_colors = [(0, 255, 0), (255, 255, 0)]  # Green and yellow
        self.failure_colors = [(255, 0, 0), (150, 0, 0)]    # Red and dark red
        
        # Display elements
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        self.human_colour = RED
        self.agent_colour = BLUE
        self.goal_colour = BLACK
        self.light_radius = 5

        # Timing controls
        self.clock = pygame.time.Clock()
        self.current_time = time.time()
        self.last_action_time = time.time()
        self.action_delay = 1.5  # 2 seconds between actions

        self.reset_lights()

        # Set up action and observation spaces
        self.action_space = Discrete(len(AgentAction))
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Agent x,y, Human x,y, Goal x,y, Light states
            high=np.array([self.n_cols-1, self.n_rows-1, self.n_cols-1, self.n_rows-1, self.n_cols-1, self.n_rows-1, 1, 1, 1, 1]),
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
        positions_y = [y for y in range(0, self.game_height, 50)]

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

        self.agent_step = 0

        self.reset_lights()
        self.human_take_action = False

        observation = self._get_obs()
        info = self._get_info(is_success=False, is_truncated=False)

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
        human_old_dist = self.calculate_manhattan_distance(self.human, self.goal)
        agent_old_row, agent_old_col = self.get_current_row_col(self.agent)

        # Handle movement actions
        if action == AgentAction.MOVE_LEFT.value and self.agent.centerx > 25:
            self.agent.move_ip(-50, 0)
        elif action == AgentAction.MOVE_RIGHT.value and self.agent.centerx < self.screen_width - 25:
            self.agent.move_ip(50, 0)
        elif action == AgentAction.MOVE_UP.value and self.agent.centery > 25:
            self.agent.move_ip(0, -50)
        elif action == AgentAction.MOVE_DOWN.value and self.agent.centery < self.game_height - 25:
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

        self.agent_step += 1
        self.last_action_time = time.time()
        
        # Wait for human action
        while current_time - self.last_action_time < self.action_delay and self.human_take_action is False:
            current_time = time.time()
            self.human_step()

        if self.agent_step >= self.step_limit:
            self.failure_effect_timer = self.effect_duration
            truncated = True

        reward, terminated = self.calculate_reward(human_old_dist, action, agent_old_row, agent_old_col)

        observation = self._get_obs()
        info = self._get_info(is_success=terminated, is_truncated=truncated)

        if terminated: 
            self.success_effect_timer = self.effect_duration

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
                elif event.key == pygame.K_DOWN and self.human.centery < self.game_height - 25:
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
                (pix_square_size * x, self.game_height),
                width=3,
            )

        # Draw game objects
        #pygame.draw.rect(self.screen, self.goal_colour, self.goal)
        pygame.draw.rect(self.screen, self.human_colour, self.human)
        pygame.draw.rect(self.screen, self.agent_colour, self.agent)

        # Draw lights
        light_t, light_r, light_b, light_l = self.set_light_xy()
        pygame.draw.circle(surface=self.screen, color=self.light_t, center=light_t, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_r, center=light_r, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_b, center=light_b, radius=self.light_radius)
        pygame.draw.circle(surface=self.screen, color=self.light_l, center=light_l, radius=self.light_radius)

        # Draw footer
        footer_rect = pygame.Rect(0, self.game_height, self.screen_width, self.footer_height)
        pygame.draw.rect(self.screen, (50, 50, 50), footer_rect)  # Dark gray background
        
        # Draw footer border
        pygame.draw.line(
            self.screen,
            WHITE,
            (0, self.game_height),
            (self.screen_width, self.game_height),
            width=3,
        )
        
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (self.screen_width - 150, self.game_height + 15))

        # Draw remaining moves
        remaining_time = self.step_limit - self.agent_step
        move_text = self.font.render(f'Agent battery: {int(remaining_time)}', True, WHITE)
        self.screen.blit(move_text, (10, self.game_height + 15))
        
        # Draw success effect
        if self.success_effect_timer > 0:
            # Create flashing effect
            color_idx = 1 if self.success_effect_timer % 6 < 3 else 0
            overlay = pygame.Surface((self.screen_width, self.game_height), pygame.SRCALPHA)
            overlay.fill((*self.success_colors[color_idx], 100))  # Semi-transparent
            self.screen.blit(overlay, (0, 0))
            
            # Display success message
            success_text = self.big_font.render('SUCCESS!', True, WHITE)
            text_rect = success_text.get_rect(center=(self.screen_width//2, self.game_height//2))
            self.screen.blit(success_text, text_rect)
            
            self.success_effect_timer -= 1
        
        # Draw failure effect
        if self.failure_effect_timer > 0:
            # Create flashing effect
            color_idx = 1 if self.failure_effect_timer % 6 < 3 else 0
            overlay = pygame.Surface((self.screen_width, self.game_height), pygame.SRCALPHA)
            overlay.fill((*self.failure_colors[color_idx], 100))  # Semi-transparent
            self.screen.blit(overlay, (0, 0))
            
            # Display failure message
            failure_text = self.big_font.render('BATTERY DEAD!', True, WHITE)
            text_rect = failure_text.get_rect(center=(self.screen_width//2, self.game_height//2))
            self.screen.blit(failure_text, text_rect)
            
            self.failure_effect_timer -= 1

        pygame.display.update()

    def calculate_reward(self, human_old_dist, agent_action, agent_old_row, agent_old_col):
        """Calculate reward based on game state changes.
        
        Args:
            old_dist: Previous manhattan distance between human and goal
            
        Returns:
            tuple: (reward, terminated)
        """
        reward = 0
        terminated = False

        # Add time-based penalty to encourage faster completion
        time_penalty = -0.05  # Small penalty each step to encourage efficiency
        reward += time_penalty
        
        human_new_dist = self.calculate_manhattan_distance(self.human, self.goal)
        if human_new_dist < human_old_dist:
            # Human moved closer to goal
            reward += 0.5  # Significant reward for successfully guiding the human
        elif human_new_dist > human_old_dist:
            # Human moved away from goal
            reward -= 0.2

        # Light-based rewards
        light_reward = self.calculate_light_reward()
        reward += light_reward

         # Penalise trying to move outside of the grid
        agent_current_row, agent_current_col = self.get_current_row_col(self.agent)
        if agent_current_row == agent_old_row and agent_current_col == agent_old_col:
            if agent_action == AgentAction.MOVE_UP.value or agent_action == AgentAction.MOVE_RIGHT.value or agent_action == AgentAction.MOVE_DOWN.value or agent_action == AgentAction.MOVE_LEFT.value:
                reward -= 0.1

        # Special state rewards
        if self.human.center == self.agent.center:
            reward -= 5
            self.score -= 5
        elif self.human.center == self.goal.center:
            reward += 10
            self.score += 10
            terminated = True

        return reward, terminated
    
    def calculate_light_reward(self):
        """Calculate reward based on light signals guiding human toward goal."""
        light_reward = 0
        
        # Get the relative position of the goal compared to the agent
        agent_row, agent_col = self.get_current_row_col(self.agent)
        goal_row, goal_col = self.get_current_row_col(self.goal)
        
        # Calculate directional differences
        row_diff = goal_row - agent_row  # Positive if goal is below agent
        col_diff = goal_col - agent_col  # Positive if goal is to the right of agent
        
        # Reward for correct light signals
        if row_diff > 0 and self.light_b == self.LIGHT_ON:  # Goal is below
            light_reward += 0.2
        if row_diff < 0 and self.light_t == self.LIGHT_ON:  # Goal is above
            light_reward += 0.2
        if col_diff > 0 and self.light_r == self.LIGHT_ON:  # Goal is to the right
            light_reward += 0.2
        if col_diff < 0 and self.light_l == self.LIGHT_ON:  # Goal is to the left
            light_reward += 0.2

        # Reward for turning on the correct light when human and goal is nearby
        human_row, human_col = self.get_current_row_col(self.human)
        human_agent_dist = abs(human_row - agent_row) + abs(human_col - agent_col)
        goal_agent_dist = abs(goal_row - agent_row) + abs(goal_col - agent_col)
        
        if human_agent_dist <= 3 and goal_agent_dist <= 3:  # Human and goal is nearby
            # Increase reward for correct light signals when human and goal is nearby
            light_reward *= 2
        
        # Penalize incorrect light signals
        if row_diff <= 0 and self.light_b == self.LIGHT_ON:  # Goal is not below
            light_reward -= 0.1
        if row_diff >= 0 and self.light_t == self.LIGHT_ON:  # Goal is not above
            light_reward -= 0.1
        if col_diff <= 0 and self.light_r == self.LIGHT_ON:  # Goal is not to the right
            light_reward -= 0.1
        if col_diff >= 0 and self.light_l == self.LIGHT_ON:  # Goal is not to the left
            light_reward -= 0.1
        
        return light_reward

    def _get_obs(self):
        """Get current observation of environment state."""
        
        human_row, human_col = self.get_current_row_col(self.human)
        goal_row, goal_col = self.get_current_row_col(self.goal)
        agent_row, agent_col = self.get_current_row_col(self.agent)
        
        TL = 1 if self.light_t == self.LIGHT_ON else 0
        RL = 1 if self.light_r == self.LIGHT_ON else 0
        BL = 1 if self.light_b == self.LIGHT_ON else 0
        LL = 1 if self.light_l == self.LIGHT_ON else 0
        
        return np.array([agent_col, agent_row, human_col, human_row, goal_col, goal_row, TL, RL, BL, LL])

    def _get_info(self, is_success, is_truncated):
        """Get additional information about environment state."""
        return {
            "distance": self.calculate_manhattan_distance(self.human, self.goal),
            "is_success": is_success,
            "is_truncated": is_truncated
        }

    def calculate_manhattan_distance(self, origin, target):
        """Calculate Manhattan distance between two points."""
        origin_row, origin_col = self.get_current_row_col(origin)
        target_row, target_col = self.get_current_row_col(target)
        return abs(origin_row - target_row) + abs(origin_col - target_col)

    def get_current_row_col(self, target):
        """Get grid row and column for target position."""
        rows = range(25, self.game_height, 50)
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