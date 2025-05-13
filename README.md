# Human Feedback-Based Reinforcement Learning in Cooperative Environments

## Project Overview

This repository contains a project exploring how AI agents can learn to cooperate with each other in a cooperative environment:

### Multi-Agent Project
- A cooperative environment where two AI agents work together to achieve a common goal
- Uses PettingZoo for multi-agent reinforcement learning
- Both agents can learn from each other's behavior and improve their cooperative strategies
- Agents receive rewards based on their collective performance in reaching the goal

## Demo

![Game Demo](./pygame_clip.gif) 

## Features

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

2. Create virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Application

1. Run the main application:
   ```
   python main.py
   ```

2. Use the menu to:
   - Select screen size
   - Choose a learning model
   - Select mode (Training, Evaluation or Tuning)
   - Start the simulation

### How to Use

- **Watch the AI agents**: Two AI agents will work together to achieve the goal
- **Goal**: Observe how the agents learn to cooperate and improve their strategies

## Project Structure

- `env/`: Environment implementation using PettingZoo
- `main.py`: Application entry point with menu system
- `models/`: Saved model files
- `logs/`: Training logs and performance metrics

## Technical Details

This project uses:
- PyGame for the environment and visualisation
- Stable-Baselines3 for reinforcement learning algorithms
- PettingZoo for multi-agent environments
- Matplotlib and Pandas for data visualisation and analysis

## Future Work

- Adding different environment types and challenges
- Enhancing multi-agent cooperation strategies
- Implementing more sophisticated reward mechanisms
- Conducting studies to evaluate effectiveness

## Challenges & Applications

Challenges addressed include:
- Developing effective cooperation strategies between AI agents
- Managing shared rewards and learning in multi-agent systems
- Optimizing the agents' strategies to ensure efficient goal achievement

This research has applications in fields like:
- Multi-agent systems
- Collaborative robotics
- Autonomous systems
- Distributed AI systems

## Acknowledgments

This project is being developed in collaboration with Nagy Balázs who is my supervisor for this project
