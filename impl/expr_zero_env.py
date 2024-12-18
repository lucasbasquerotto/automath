import sympy
from utils.types import BASIC_NODE_TYPES
from environment.state import State, ExprInfo
from environment.action import DEFAULT_ACTIONS
from environment.reward import DefaultRewardEvaluator
from environment.full_state import FullState
from environment.environment import Environment
from environment.meta_env import FullEnvMetaInfo
from impl import (
    node_type_handler,
    node_types,
    reformulation_action_types,
    partial_action_types)

class ExprZeroEnv(Environment):
    def __init__(
        self,
        meta_context_idx: int,
        initial_expression: ExprInfo,
        max_steps: int = 100000,
        max_history_size: int | None = None,
    ):
        initial_state, [definition_key] = FullState.initial_state([initial_expression])

        def is_zero(state: State) -> bool | None:
            expr_info = next(item for key, item in state.definitions if key == definition_key)
            assert expr_info is not None
            if not isinstance(expr_info.expr, sympy.Integer):
                return None
            return expr_info.expr == sympy.Integer(0)

        def is_terminal(state: State) -> bool:
            return is_zero(state) is not None

        meta = FullEnvMetaInfo(
            main_context=meta_context_idx,
            node_types=tuple(list(BASIC_NODE_TYPES) + [
                node_types.Add,
            ]),
            node_type_handler=node_type_handler.DefaultNodeTypeHandler(),
            action_types=tuple(DEFAULT_ACTIONS + [
                reformulation_action_types.SimplifyAddAction,
                partial_action_types.ReplaceNodeAction,
            ]),
            reward_evaluator=DefaultRewardEvaluator(is_terminal),
            initial_history=(initial_state,),
            is_terminal=is_terminal,
        )

        super().__init__(
            meta=meta,
            max_steps=max_steps,
            max_history_size=max_history_size,
        )

        self._is_zero = is_zero

    def correct(self, state: State) -> bool | None:
        return self._is_zero(state)
