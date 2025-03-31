from stable_baselines3.common.callbacks import BaseCallback
import os


class HumanCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0, rl_algorithm='ppo', game_size=500):
        super().__init__(verbose)
        
        self.algorithm = rl_algorithm
        self.game_size = game_size

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.training_env.render()

        # Check if the episode has ended
        if self.locals['infos'][0].get('is_success', False):
            # Save the model when the human reaches the goal (end of episode)
            cwd = os.getcwd()
            self.model.save(f"{cwd}/models/{self.algorithm}_model")
            print(f"Model saved to path: {cwd}/models/{self.algorithm}_{self.game_size}_model")

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass