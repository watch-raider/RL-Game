REWARD_CONFIG_HUMAN_V1 = {
    "bump_penalty": -0.1,  # Unchanged
    "collision_penalty": -1,  # Increased to minimize collisions
    "goal_reward_base": 10,  # Increased, scaled: 15.0 * (1 + progress_ratio)
    "goal_human_bonus": 10,  # Reduced to balance

    "human_proximity_reward": 0,
    "human_light_correct": 0,
    "human_light_incorrect": 0
}

REWARD_CONFIG_HUMAN_V2 = {
    "bump_penalty": -0.1,  # Unchanged
    "collision_penalty": -1,  # Increased to minimize collisions
    "goal_reward_base": 10,  # Increased, scaled: 15.0 * (1 + progress_ratio)
    "goal_human_bonus": 0,  # Reduced to balance

    "human_proximity_reward": 0.1,
    "human_light_correct": 0.1,
    "human_light_incorrect": -0.05
}