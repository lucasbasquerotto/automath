import numpy as np
import sympy
from env.core import INode, ISpecialValue, IInt, BaseNode, TypeNode
from env.full_state import FullState
from env.action import IAction
from env.reward import IRewardEvaluator, DefaultRewardEvaluator

class Environment:
    def __init__(
        self,
        initial_state: FullState,
        reward_evaluator: IRewardEvaluator | None = None,
        max_steps: int | None = None,
    ):
        self._initial_state = initial_state
        self._current_state = initial_state
        self._reward_evaluator = reward_evaluator or DefaultRewardEvaluator()
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

    @classmethod
    def state_array(cls, full_state: FullState) -> np.ndarray[np.int_, np.dtype]:
        size = len(full_state)
        result = np.zeros((size, 6), dtype=np.int_)
        pending_node_stack: list[tuple[int, int, INode]] = [(0, 0, full_state)]
        node_types = full_state.node_types()
        node_id = 0

        while pending_node_stack:
            current: tuple[int, int, INode] = pending_node_stack.pop()
            parent_id, arg_id, node = current
            node_id += 1
            idx = node_id - 1
            node_type_id = node_types.index(type(node)) + 1
            assert node_type_id > 0
            result[idx][0] = node_id
            result[idx][1] = parent_id
            result[idx][2] = arg_id
            result[idx][3] = node_type_id

            if isinstance(node, ISpecialValue):
                value_aux = node.node_value

                if isinstance(value_aux, IInt):
                    value = value_aux.as_int
                elif isinstance(value_aux, TypeNode):
                    value = node_types.index(type(node)) + 1
                else:
                    raise ValueError(f'Invalid value type: {type(value_aux)}')

                result[idx][4] = value
            else:
                args = node.as_node.args
                args_amount = len(args)
                result[idx][5] = args_amount
                for i in range(args_amount):
                    inner_arg_id = args_amount - i
                    arg = args[inner_arg_id - 1]
                    assert isinstance(arg, INode)
                    pending_node_stack.append((node_id, inner_arg_id, arg))

        return result

    @classmethod
    def symbolic(cls, node: BaseNode, node_types: tuple[type[INode], ...]) -> sympy.Basic:
        assert isinstance(node, BaseNode)
        name = node.func.__name__

        if isinstance(node, ISpecialValue):
            value_aux = node.node_value

            if isinstance(value_aux, IInt):
                value = value_aux.as_int
                return sympy.Symbol(f'{name}[{value}]')
            elif isinstance(value_aux, TypeNode):
                value = node_types.index(type(node)) + 1
                return sympy.Symbol(f'type[{value}]')
            else:
                raise ValueError(f'Invalid value type: {type(value_aux)}')
        elif len(node.args) == 0:
            return sympy.Symbol(name)

        raw_args = [arg.as_node for arg in node.args if isinstance(arg, INode)]
        assert len(raw_args) == len(node.args)

        fn = sympy.Function(name)
        args: list[sympy.Basic] = [cls.symbolic(arg, node_types) for arg in raw_args]
        # pylint: disable=not-callable
        return fn(*args)

    def to_data_array(self) -> np.ndarray[np.int_, np.dtype]:
        return self.state_array(self.current_state)

    def to_symbolic(self) -> sympy.Basic:
        return self.symbolic(self.current_state, self.current_state.node_types())
