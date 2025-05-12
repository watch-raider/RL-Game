REWARD_CONFIG_V1 = {
    "bump_penalty": -0.1,  # Unchanged
    "collision_penalty": -1,  # Increased to minimize collisions
    "goal_reward_base": 10,  # Increased, scaled: 15.0 * (1 + progress_ratio)
    "goal_human_bonus": 10,  # Reduced to balance

    "human_progress_base": 0,  # Increased to strongly reward progress
    "human_progress_penalty": 0,  # Reduced for variability
    "light_correct": 0,  # Unchanged
    "light_incorrect": 0,  # Unchanged
    "proximity_reward": 0,  # Unchanged
}

REWARD_CONFIG_V2 = {
    "bump_penalty": -0.1,  # Unchanged
    "collision_penalty": -1,  # Increased to minimize collisions
    "goal_reward_base": 10,  # Increased, scaled: 15.0 * (1 + progress_ratio)
    "goal_human_bonus": 10,  # Reduced to balance

    "human_progress_base": 0.3,   # reward if human moves towards goal
    "human_progress_penalty": -0.1,  # penalty if human moves away

    "light_correct": 0.1,
    "light_incorrect": -0.05,

    "proximity_reward": 0.1,  # Unchanged
}

REWARD_CONFIG_HUMAN_V1 = {
    "bump_penalty": -0.1,  # Unchanged
    "collision_penalty": -1,  # Increased to minimize collisions
    "goal_reward_base": 10,  # Increased, scaled: 15.0 * (1 + progress_ratio)
    "goal_human_bonus": 10,  # Reduced to balance

    "human_proximity_reward": 0,
    "human_light_correct": 0,
    "human_light_incorrect": 0,

    "human_progress_base": 0,   # reward if human moves towards goal
    "human_progress_penalty": 0,  # penalty if human moves away
    "light_correct": 0,
    "light_incorrect": 0,
    "proximity_reward": 0,  # Unchanged
}

REWARD_CONFIG_HUMAN_V2 = {
    "bump_penalty": -0.1,  # Unchanged
    "collision_penalty": -1,  # Increased to minimize collisions
    "goal_reward_base": 10,  # Increased, scaled: 15.0 * (1 + progress_ratio)
    "goal_human_bonus": 0,  # Reduced to balance

    "human_proximity_reward": 0.1,
    "human_light_correct": 0.1,
    "human_light_incorrect": -0.05,

    "human_progress_base": 0,   # reward if human moves towards goal
    "human_progress_penalty": 0,  # penalty if human moves away
    "light_correct": 0,
    "light_incorrect": 0,
    "proximity_reward": 0,  # Unchanged
}