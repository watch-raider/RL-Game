from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO

def define_model(model_name, env, trial, device="cpu"):
    """Define the RL model with hyperparameters suggested by Optuna."""
    if model_name == "a2c":
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        n_steps = trial.suggest_int("n_steps", 5, 20, step=5)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
        vf_coef = trial.suggest_float("vf_coef", 0.25, 0.75)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
        return A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
            n_steps=n_steps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            normalize_advantage=True,
            device=device
        )
    elif model_name == "ppo":
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
        n_steps = trial.suggest_int("n_steps", 1024, 2048, step=512)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        n_epochs = trial.suggest_int("n_epochs", 5, 15)
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            clip_range=clip_range,
            n_epochs=n_epochs,
            device=device
        )
    elif model_name == "trpo":
        # Define reduced hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        target_kl = trial.suggest_float("target_kl", 0.005, 0.02)
        n_steps = trial.suggest_int("n_steps", 1024, 2048, step=512)
        batch_size = trial.suggest_int("batch_size", 64, 128, step=32)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.98)
        return TRPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            target_kl=target_kl,
            n_steps=n_steps,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
            normalize_advantage=True
        )
    elif model_name == "dqn":
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.2)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
        target_update_interval = trial.suggest_int("target_update_interval", 1000, 5000, step=1000)
        return DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            target_update_interval=target_update_interval,
            device=device
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")