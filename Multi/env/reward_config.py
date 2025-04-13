REWARD_CONFIG = {
    "time_penalty": -0.05,

    "human_progress_base": 0.5,   # will be scaled by (Δ distance)
    "human_progress_penalty": -0.2,  # penalty if human moves away

    "light_correct": 0.2,
    "light_incorrect": -0.1,

    "bump_penalty": -0.1,
    "collision_penalty": -5,

    "goal_reward_base": 10.0,  # scaled by speed
    "goal_human_bonus": 10.0,
}