import numpy as np
from env import core
from env import state
from env import action as action_module
from env import full_state as full_state_module
from env import reward as reward_module
from env import symbol

class Environment:
    def __init__(
        self,
        initial_state: full_state_module.FullState,
        reward_evaluator: reward_module.IRewardEvaluator | None = None,
        max_steps: int | None = None,
    ):
        self._initial_state = initial_state
        self._full_state = initial_state
        self._reward_evaluator = reward_evaluator or reward_module.DefaultRewardEvaluator.create()
        self._max_steps = max_steps
        self._current_step = 0

    @property
    def full_state(self) -> full_state_module.FullState:
        return self._full_state

    def reset(self) -> full_state_module.FullState:
        self._full_state = self._initial_state
        self._current_step = 0
        core.INode.clear_cache()
        return self._full_state

    def step(
        self,
        action: action_module.IAction[full_state_module.FullState],
    ) -> tuple[full_state_module.FullState, float, bool, bool]:
        reward_evaluator = self._reward_evaluator
        current_state = self._full_state
        next_state = action.run_action(current_state)
        reward = reward_evaluator.evaluate(
            self._full_state,
            next_state)
        self._current_step += 1
        terminated = next_state.goal_achieved()
        truncated = (
            (self._current_step >= self._max_steps and not terminated)
            if self._max_steps is not None
            else False
        )
        self._full_state = next_state
        return next_state, reward, terminated, truncated

    @classmethod
    def data_array(
        cls,
        root_node: core.BaseNode,
        node_types: tuple[type[core.INode], ...],
    ) -> np.ndarray[np.int_, np.dtype]:
        size = len(root_node)
        result = np.zeros((size, 8), dtype=np.int_)
        pending_node_stack: list[tuple[int, int, int, int, core.INode]] = [(0, 0, 0, 0, root_node)]
        node_id = 0

        while pending_node_stack:
            current: tuple[int, int, int, int, core.INode] = pending_node_stack.pop()
            parent_id, arg_id, parent_scope_id, context_parent_node_id, node = current
            node_id += 1
            idx = node_id - 1
            node_type_id = node_types.index(type(node)) + 1
            next_context_node_id = (
                (context_parent_node_id + (node_id - parent_id))
                if context_parent_node_id > 0
                else (1 if isinstance(node, state.IContext) else 0)
            )
            context_node_id = (next_context_node_id - 1) if next_context_node_id >= 1 else 0
            assert node_type_id > 0
            scope_id = parent_scope_id

            if isinstance(node, core.IOpaqueScope):
                scope_id = 1
            elif isinstance(node, core.IScope):
                assert isinstance(node, core.IInnerScope)
                assert parent_scope_id > 0
                scope_id = parent_scope_id + 1

            result[idx][0] = node_id
            result[idx][1] = parent_id
            result[idx][2] = arg_id
            result[idx][3] = scope_id
            result[idx][4] = context_node_id
            result[idx][5] = node_type_id

            if isinstance(node, core.ISpecialValue):
                value_aux = node.node_value

                if isinstance(value_aux, core.IInt):
                    value = value_aux.as_int
                elif isinstance(value_aux, core.TypeNode):
                    value = node_types.index(value_aux.type) + 1
                else:
                    raise ValueError(f'Invalid value type: {type(value_aux)}')

                result[idx][6] = value
            else:
                args = node.as_node.args
                args_amount = len(args)
                result[idx][7] = args_amount
                for i in range(args_amount):
                    inner_arg_id = args_amount - i
                    arg = args[inner_arg_id - 1]
                    assert isinstance(arg, core.INode)
                    pending_node_stack.append((
                        node_id,
                        inner_arg_id,
                        scope_id,
                        next_context_node_id,
                        arg,
                    ))

        return result

    @property
    def as_symbol(self) -> symbol.Symbol:
        return self.symbol(self.full_state)

    @property
    def current_state_symbol(self) -> symbol.Symbol:
        current_state = self.full_state.current_state.apply()
        return self.symbol(current_state)

    def symbol(self, node: core.BaseNode) -> symbol.Symbol:
        node_types = self.full_state.node_types()
        return symbol.Symbol(node, node_types)

    def to_data_array(self) -> np.ndarray[np.int_, np.dtype]:
        node_types = self.full_state.node_types()
        return self.data_array(self.full_state, node_types)

    def to_data_array_current_state(self) -> np.ndarray[np.int_, np.dtype]:
        current_state = self.full_state.current_state.apply()
        node_types = self.full_state.node_types()
        return self.data_array(current_state, node_types)
