from gymnasium.spaces import Discrete, Box

from pettingzoo import ParallelEnv

import pygame

import random
from enum import Enum
import time
import numpy as np
from copy import copy


# Standard RGB colors 
RED = (255, 0, 0) 
GREEN = (0, 255, 0) 
BLUE = (0, 0, 255) 
CYAN = (0, 100, 100) 
BLACK = (0, 0, 0) 
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
DARK_RED = (150, 0, 0)

class AgentAction(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    LIGHT_LEFT_TOGGLE = 4
    LIGHT_RIGHT_TOGGLE = 5
    LIGHT_UP_TOGGLE = 6
    LIGHT_DOWN_TOGGLE = 7

class HumanAction(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3

class PygameEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "multi_agent_pygame_environment_v0",
    }
    LIGHT_OFF = (180, 180, 180)
    LIGHT_ON = (255, 165, 0)
    PLAYER_SIZE = 50

    def __init__(self, SCREEN_WIDTH=500, SCREEN_HEIGHT=500, step_limit=100, render_mode="human", self_play=False):
        """Initialize the environment."""
        # Initialize game objects
        self.goal = None
        self.human = None 
        self.agent = None
        self.timestep = None
        self.render_mode = render_mode
        self.self_play = self_play
        self.is_success = False
        self.is_truncated = False

        self.screen_width = SCREEN_WIDTH
        
        # Add footer height
        self.footer_height = 60
        self.game_height = SCREEN_HEIGHT
        self.screen_height = SCREEN_HEIGHT + self.footer_height
        
        # Don't create the screen here - only create it when rendering
        self.screen = None
        self.font = None
        self.big_font = None
        self.clock = None

        self.possible_agents = ["agent", "human"]

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
        self.effect_duration = 1 
        self.success_colors = [GREEN, YELLOW]  # Green and yellow
        self.failure_colors = [RED, DARK_RED]    # Red and dark red
        
        # Display elements
        self.human_colour = RED
        self.agent_colour = BLUE
        self.goal_colour = BLACK
        self.light_radius = 5

        # Timing controls
        self.current_time = time.time()
        self.last_action_time = time.time()
        if self_play:
            self.action_delay = 1 # 1 seconds between actions
        else:
            self.action_delay = 0 # 0.5 seconds between actions

        self.reset_lights()

        # Set up action and observation spaces
        self.action_spaces = {
                "agent": Discrete(len(AgentAction)),
                "human": Discrete(len(AgentAction))  # Pad human action space to match agent
            }
        
        # Use normalized observation space (0 to 1) regardless of grid size
        self.observation_spaces = {
            "agent": Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Agent x,y, Human x,y, Goal x,y, Light states
                high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                dtype=np.float32
            ),
            "human": Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Agent x,y, Human x,y, Goal x,y, Light states
                high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                dtype=np.float32
            )
        }

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - RL agent x and y coordinates
        - human x and y coordinates
        - goal x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)

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

        observations = self._get_obs()
        
        # Get infos with proper structure
        infos = self._get_info(is_success=self.is_success, is_truncated=self.is_truncated)
        
        # Return observations in the format expected by Supersuit
        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - agent x and y coordinates
        - human x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        self.is_success = False
        self.is_truncated = False

        # Execute actions
        agent_action = actions["agent"]
        human_action = actions["human"]

        current_time = time.time()
        self.human_take_action = False
        truncations = {"agent": False, "human": False}
        self.is_success = False
        self.is_truncated = False
        human_old_dist = self.calculate_manhattan_distance(self.human, self.goal)
        agent_old_row, agent_old_col = self.get_current_row_col(self.agent)
        human_old_row, human_old_col = self.get_current_row_col(self.human)

        # Handle movement actions
        if agent_action == AgentAction.MOVE_LEFT.value and self.agent.centerx > 25:
            self.agent.move_ip(-50, 0)
        elif agent_action == AgentAction.MOVE_RIGHT.value and self.agent.centerx < self.screen_width - 25:
            self.agent.move_ip(50, 0)
        elif agent_action == AgentAction.MOVE_UP.value and self.agent.centery > 25:
            self.agent.move_ip(0, -50)
        elif agent_action == AgentAction.MOVE_DOWN.value and self.agent.centery < self.game_height - 25:
            self.agent.move_ip(0, 50)
            
        # Handle light toggle actions
        elif agent_action == AgentAction.LIGHT_UP_TOGGLE.value:
            self.light_t = self.LIGHT_ON if self.light_t == self.LIGHT_OFF else self.LIGHT_OFF
        elif agent_action == AgentAction.LIGHT_RIGHT_TOGGLE.value:
            self.light_r = self.LIGHT_ON if self.light_r == self.LIGHT_OFF else self.LIGHT_OFF
        elif agent_action == AgentAction.LIGHT_DOWN_TOGGLE.value:
            self.light_b = self.LIGHT_ON if self.light_b == self.LIGHT_OFF else self.LIGHT_OFF
        elif agent_action == AgentAction.LIGHT_LEFT_TOGGLE.value:
            self.light_l = self.LIGHT_ON if self.light_l == self.LIGHT_OFF else self.LIGHT_OFF

        self.agent_step += 1
        self.last_action_time = time.time()
        
        #if self.render_mode == "human":
        # Wait for human action
        while current_time - self.last_action_time < self.action_delay and self.human_take_action is False:
            current_time = time.time()
            self.human_step()

        if self.self_play is False:
        # Handle movement actions - only use first 4 actions (movement) for human
            human_action = human_action % 4  # Convert to 0-3 range for movement actions
            if human_action == HumanAction.MOVE_LEFT.value and self.human.centerx > 25:
                self.human.move_ip(-50, 0)
            elif human_action == HumanAction.MOVE_RIGHT.value and self.human.centerx < self.screen_width - 25:
                self.human.move_ip(50, 0)
            elif human_action == HumanAction.MOVE_UP.value and self.human.centery > 25:
                self.human.move_ip(0, -50)
            elif human_action == HumanAction.MOVE_DOWN.value and self.human.centery < self.game_height - 25:
                self.human.move_ip(0, 50)
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

        if self.agent_step >= self.step_limit:
            self.failure_effect_timer = self.effect_duration
            truncations = {"agent": True, "human": True}

        rewards, terminations = self.calculate_reward(human_old_dist, 
                                                      agent_action, agent_old_row, agent_old_col, 
                                                      human_action, human_old_row, human_old_col)

        # Get observations
        observations = self._get_obs()
        
        # Get dummy infos (not used in this example)
        infos = self._get_info(is_success=terminations["human"], is_truncated=truncations["human"])
        self.is_success = terminations["human"]
        self.is_truncated = truncations["human"]

        if terminations["human"] or truncations["human"]:
            self.agents = []

        if terminations["human"]: 
            self.success_effect_timer = self.effect_duration
        
        return observations, rewards, terminations, truncations, infos
    
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
        """Render the environment."""
        if self.render_mode == "human":
            # Initialize Pygame components if needed
            self._init_pygame_components()
            
            # Draw the game elements
            self.screen.fill(WHITE)
            
            # Draw grid lines
            for x in range(0, self.screen_width, self.grid_size):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.game_height))
            for y in range(0, self.game_height, self.grid_size):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_width, y))
            
            # Draw footer
            pygame.draw.rect(self.screen, (240, 240, 240), 
                             (0, self.game_height, self.screen_width, self.footer_height))
            
            if self.self_play is False:
                # Draw goal
                pygame.draw.rect(self.screen, self.goal_colour, self.goal)
            
            # Draw human
            pygame.draw.rect(self.screen, self.human_colour, self.human)
            
            # Draw agent
            pygame.draw.rect(self.screen, self.agent_colour, self.agent)
            
            # Draw lights
            light_1, light_2, light_3, light_4 = self.set_light_xy()
            pygame.draw.circle(self.screen, self.light_t, light_1, self.light_radius)
            pygame.draw.circle(self.screen, self.light_r, light_2, self.light_radius)
            pygame.draw.circle(self.screen, self.light_b, light_3, self.light_radius)
            pygame.draw.circle(self.screen, self.light_l, light_4, self.light_radius)
            
            # Draw score
            score_text = self.font.render(f"Score: {self.score}", True, BLACK)
            self.screen.blit(score_text, (10, self.game_height + 10))
            
            # Draw step counter
            step_text = self.font.render(f"Step: {self.agent_step}/{self.step_limit}", True, BLACK)
            self.screen.blit(step_text, (self.screen_width - 150, self.game_height + 10))
            
            # Draw success/failure effects
            if self.success_effect_timer > 0:
                color_idx = int((self.success_effect_timer / self.effect_duration) * len(self.success_colors))
                color = self.success_colors[min(color_idx, len(self.success_colors)-1)]
                success_text = self.big_font.render("SUCCESS!", True, color)
                text_rect = success_text.get_rect(center=(self.screen_width//2, self.game_height//2))
                self.screen.blit(success_text, text_rect)
                self.success_effect_timer -= 0.05
            
            if self.failure_effect_timer > 0:
                color_idx = int((self.failure_effect_timer / self.effect_duration) * len(self.failure_colors))
                color = self.failure_colors[min(color_idx, len(self.failure_colors)-1)]
                failure_text = self.big_font.render("FAILURE!", True, color)
                text_rect = failure_text.get_rect(center=(self.screen_width//2, self.game_height//2))
                self.screen.blit(failure_text, text_rect)
                self.failure_effect_timer -= 0.05
            
            pygame.display.flip()
            self.clock.tick(30)
        
        return self.screen

    def calculate_reward(self, human_old_dist, agent_action, agent_old_row, agent_old_col, human_action, human_old_row, human_old_col):
        """Calculate reward based on game state changes.
        
        Args:
            old_dist: Previous manhattan distance between human and goal
            
        Returns:
            tuple: (reward, terminated)
        """
        reward = 0
        human_reward = 0
        terminations = {"agent": False, "human": False}

        # Add time-based penalty to encourage faster completion
        time_penalty = -0.05  # Small penalty each step to encourage efficiency
        reward += time_penalty
        human_reward += time_penalty
        
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
        
        human_current_row, human_current_col = self.get_current_row_col(self.human)
        if human_current_row == human_old_row and human_current_col == human_old_col:
            if human_action == HumanAction.MOVE_UP.value or human_action == HumanAction.MOVE_RIGHT.value or human_action == HumanAction.MOVE_DOWN.value or human_action == HumanAction.MOVE_LEFT.value:
                human_reward -= 0.1

        # Special state rewards
        if self.human.center == self.agent.center:
            reward -= 5
            self.score -= 5
        elif self.human.center == self.goal.center:
            reward += 10
            human_reward += 10
            self.score += 10
            terminations = {"agent": True, "human": True}

        # Return a dictionary of rewards:
        rewards = {"agent": reward, "human": human_reward}
        return rewards, terminations
    
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
        """Get current observation of environment state with normalized coordinates."""
        
        human_row, human_col = self.get_current_row_col(self.human)
        goal_row, goal_col = self.get_current_row_col(self.goal)
        agent_row, agent_col = self.get_current_row_col(self.agent)
        
        # Normalize coordinates to [0,1] range
        norm_agent_col = agent_col / (self.n_cols - 1) if self.n_cols > 1 else 0
        norm_agent_row = agent_row / (self.n_rows - 1) if self.n_rows > 1 else 0
        norm_human_col = human_col / (self.n_cols - 1) if self.n_cols > 1 else 0
        norm_human_row = human_row / (self.n_rows - 1) if self.n_rows > 1 else 0
        norm_goal_col = goal_col / (self.n_cols - 1) if self.n_cols > 1 else 0
        norm_goal_row = goal_row / (self.n_rows - 1) if self.n_rows > 1 else 0
        
        TL = 1 if self.light_t == self.LIGHT_ON else 0
        RL = 1 if self.light_r == self.LIGHT_ON else 0
        BL = 1 if self.light_b == self.LIGHT_ON else 0
        LL = 1 if self.light_l == self.LIGHT_ON else 0
        

        return {
            "agent": np.array([
                norm_agent_col, norm_agent_row, 
                norm_human_col, norm_human_row, 
                norm_goal_col, norm_goal_row, 
                TL, RL, BL, LL
            ], dtype=np.float32),
            "human": np.array([
                norm_agent_col, norm_agent_row, 
                norm_human_col, norm_human_row,
                0, 0,
                TL, RL, BL, LL
            ], dtype=np.float32)
        }

    def _get_info(self, is_success, is_truncated):
        """Get additional information about environment state."""
        # Create a base info dictionary
        base_info = {
            "distance": self.calculate_manhattan_distance(self.human, self.goal),
            "is_success": is_success,
            "is_truncated": is_truncated
        }
        
        # Return a dictionary with an entry for each agent
        return {a: base_info.copy() for a in self.agents}

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

    def _init_pygame_components(self):
        """Initialize Pygame components that can't be pickled."""
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.font = pygame.font.Font(None, 36)
            self.big_font = pygame.font.Font(None, 72)
            self.clock = pygame.time.Clock()