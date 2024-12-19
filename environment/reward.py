import typing
from .state import State, BaseNode

class RewardEvaluator:
    def __call__(self, current_state: State, next_state: State) -> float:
        raise NotImplementedError()

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

        weight = 1
        for _, expr_info in next_state.definitions:
            weight += 100
            weight += 10*len(expr_info.expr.atoms(BaseNode))
        for expr_info_p in current_state.partial_definitions:
            weight += 10
            if expr_info_p is not None:
                weight += len(expr_info_p.expr.atoms(BaseNode))
        for arg_group in current_state.arg_groups:
            for expr in arg_group.expressions:
                weight += 10
                if expr is not None:
                    weight += len(expr.atoms(BaseNode))

        if next_state == current_state:
            return -10 * weight # No change applied
        return -weight  # Small penalty for each step taken
