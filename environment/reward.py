import typing
from environment.state import State

class RewardEvaluator:
    def __call__(self, current_state: State, next_state: State) -> float:
        raise NotImplementedError

class DefaultRewardEvaluator(RewardEvaluator):
    def __init__(
        self,
        is_terminal: typing.Callable[[State], bool],
        goal_reward: int = 10000,
    ):
        self._is_terminal = is_terminal
        self._goal_reward = goal_reward

    def __call__(self, current_state: State, next_state: State) -> float:
        if self._is_terminal(next_state):
            return self._goal_reward  # Reached the objective

        weight = len(next_state)

        if next_state == current_state:
            return -10 * weight # No change applied
        return -weight  # Small penalty for each step taken
