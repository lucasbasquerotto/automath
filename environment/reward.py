from environment.full_state import FullState

class RewardEvaluator:

    def evaluate(self, current_state: FullState, next_state: FullState) -> float:
        raise NotImplementedError

class DefaultRewardEvaluator(RewardEvaluator):
    def __init__(
        self,
        goal_reward: int = 10000,
    ):
        self._goal_reward = goal_reward

    def evaluate(self, current_state: FullState, next_state: FullState) -> float:
        if next_state.goal_achieved():
            return self._goal_reward  # Reached the objective

        weight = len(next_state)

        if next_state.current_state.apply() == current_state.current_state.apply():
            return -10 * weight # No change applied
        return -weight  # Small penalty for each step taken
