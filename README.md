# Human Feedback-Based Reinforcement Learning in Cooperative Environments

## Project Overview

This repository contains two distinct projects exploring how AI agents can learn to cooperate with either humans or other AI agents:

### Single Agent Project
- The environment is a grid-based game where the human player (red square) must find a goal
- The AI agent (blue square) can move around and toggle directional lights to guide the human
- The agent receives rewards when the human moves closer to the goal and penalties when the human moves away
- The agent learns over time to provide better guidance through reinforcement learning

### Multi-Agent Project
- A cooperative environment where two AI agents work together to achieve a common goal
- Uses PettingZoo for multi-agent reinforcement learning
- Both agents can learn from each other's behavior and improve their cooperative strategies
- Agents receive rewards based on their collective performance in reaching the goal

## Demo

![Game Demo](./pygame_clip.gif) 

## Features

### Single Agent Project
- Grid-based environment with human and AI agent interaction
- Multiple reinforcement learning algorithms (PPO, TRPO, A2C)
- Training and evaluation modes
- Performance tracking and visualization
- Customizable environment settings

### Multi-Agent Project
- PettingZoo-based multi-agent environment
- Cooperative learning between AI agents
- Shared reward system
- Training and evaluation modes
- Performance tracking and visualization

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/watch-raider/RL-Game.git
   cd RL-Game
   ```

2. Create virtual environments for each project:
   ```
   # For Single Agent Project
   cd Single
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt

   # For Multi-Agent Project
   cd ../Multi
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Applications

#### Single Agent Project
1. Navigate to the Single directory:
   ```
   cd Single
   ```

2. Run the main application:
   ```
   python main.py
   ```

3. Use the menu to:
   - Select screen size
   - Choose a learning model (PPO, TRPO, or A2C)
   - Select mode (Training or Evaluation)
   - Start the game

#### Multi-Agent Project
1. Navigate to the Multi directory:
   ```
   cd Multi
   ```

2. Run the main application:
   ```
   python main.py
   ```

3. Use the menu to:
   - Select screen size
   - Choose a learning model
   - Select mode (Training or Evaluation)
   - Start the simulation

### How to Play

#### Single Agent Project
- **As a human player**: Use the arrow keys to move the red square
- **Watch the AI agent**: The blue square will move and toggle lights to guide you
- **Goal**: Follow the agent's guidance to find the hidden goal

#### Multi-Agent Project
- **Watch the AI agents**: Two AI agents will work together to achieve the goal
- **Goal**: Observe how the agents learn to cooperate and improve their strategies

## Project Structure

### Single Agent Project
- `env/pygame_env.py`: The main environment implementation
- `env/main.py`: Application entry point with menu system
- `env/callbacks/`: Contains callback implementations for training and logging
- `models/`: Saved model files
- `logs/`: Training logs and performance metrics

### Multi-Agent Project
- `env/`: Environment implementation using PettingZoo
- `main.py`: Application entry point with menu system
- `models/`: Saved model files
- `logs/`: Training logs and performance metrics

## Technical Details

This project uses:
- PyGame for the environment and visualisation
- Stable-Baselines3 for reinforcement learning algorithms
- Gymnasium for single-agent environments
- PettingZoo for multi-agent environments
- Matplotlib and Pandas for data visualisation and analysis

## Future Work

- Implementing more sophisticated guidance mechanisms
- Adding different environment types and challenges
- Improving the human-agent interaction interface
- Enhancing multi-agent cooperation strategies
- Conducting user studies to evaluate effectiveness

## Challenges & Applications

Challenges addressed include:
- Balancing clear communication with adaptability to human behavior
- Optimizing the agent's strategy to ensure efficient goal achievement
- Developing effective cooperation strategies between AI agents
- Managing shared rewards and learning in multi-agent systems

This research has applications in fields like:
- Human-robot interaction
- Virtual assistance
- Autonomous systems
- Multi-agent systems
- Collaborative robotics

## Acknowledgments

This project is being developed in collaboration with Nagy Balázs who is my supervisor for this project
