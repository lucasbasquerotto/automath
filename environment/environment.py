from environment.full_state import FullState
from environment.action import IAction
from environment.reward import RewardEvaluator, DefaultRewardEvaluator

class Environment:
    def __init__(
        self,
        initial_state: FullState,
        reward_evaluator: RewardEvaluator = DefaultRewardEvaluator(),
        max_steps: int | None = None,
    ):
        self._initial_state = initial_state
        self._current_state = initial_state
        self._reward_evaluator = reward_evaluator
        self._max_steps = max_steps
        self._current_step = 0

    @property
    def current_state(self) -> FullState:
        return self._current_state

    def reset(self) -> FullState:
        self._current_state = self._initial_state
        self._current_step = 0
        return self._current_state

    def step(self, action: IAction[FullState]) -> tuple[FullState, float, bool, bool]:
        reward_evaluator = self._reward_evaluator
        current_state = self._current_state
        next_state = action.run_action(current_state)
        reward = reward_evaluator.evaluate(
            self._current_state,
            next_state)
        self._current_step += 1
        terminated = next_state.goal_achieved()
        truncated = (
            (self._current_step >= self._max_steps and not terminated)
            if self._max_steps is not None
            else False
        )
        self._current_state = next_state
        return next_state, reward, terminated, truncated
