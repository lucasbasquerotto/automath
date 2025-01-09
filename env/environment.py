import numpy as np
import sympy
from sympy.printing.latex import LatexPrinter
from env import core
from env import state
from env import meta_env
from env import action as action_module
from env import full_state as full_state_module
from env import reward as reward_module

WRAPPERS = (
    core.IGroup,
    state.State,
    meta_env.MetaInfo,
    meta_env.SubtypeOuterGroup,
    action_module.FullActionOutput,
    full_state_module.FullState,
    full_state_module.HistoryNode,
    full_state_module.HistoryGroupNode,
)

class SympyWrapper(sympy.Basic):

    def __init__(self, name: sympy.Symbol, *args: sympy.Basic):
        super().__init__()
        self._args = (name, *args)

    def _latex(self, printer: LatexPrinter) -> str:
        name = self.args[0]
        assert isinstance(name, sympy.Symbol)
        args = self.args[1:]
        amount = len(args)
        name_str = r"\text{" + name.name + r"}"
        amount_str = r"\{" + str(amount) + r"\}"
        if len(args) == 0:
            return f"{name_str}{amount_str}"
        args_latex = r" \\ ".join(
            r"\{" + str(i+1) + r"\}\text{ }" + printer.doprint(arg)
            for i, arg in enumerate(args))
        begin = r"\begin{cases}"
        end = r"\end{cases}"
        return f"{name_str}{amount_str} {begin} {args_latex} {end}"

class SympyFunction(sympy.Basic):

    def __init__(self, name: sympy.Symbol, *args: sympy.Basic):
        super().__init__()
        self._args = (name, *args)

    def _latex(self, printer: LatexPrinter) -> str:
        name = self.args[0]
        args = self.args[1:]
        args_latex = r" \\ ".join(printer.doprint(arg) for arg in args)
        if len(args) <= 1:
            return f"{name}({args_latex})"
        args_latex = args_latex.replace(r" \\ ", r" \\ \quad ")
        newline = r" \\ "
        return f"{name}( {newline} \\quad {args_latex} {newline} )"


class Environment:
    def __init__(
        self,
        initial_state: full_state_module.FullState,
        reward_evaluator: reward_module.IRewardEvaluator | None = None,
        max_steps: int | None = None,
    ):
        self._initial_state = initial_state
        self._current_state = initial_state
        self._reward_evaluator = reward_evaluator or reward_module.DefaultRewardEvaluator()
        self._max_steps = max_steps
        self._current_step = 0

    @property
    def current_state(self) -> full_state_module.FullState:
        return self._current_state

    def reset(self) -> full_state_module.FullState:
        self._current_state = self._initial_state
        self._current_step = 0
        return self._current_state

    def step(
        self,
        action: action_module.IAction[full_state_module.FullState],
    ) -> tuple[full_state_module.FullState, float, bool, bool]:
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
    def state_array(
        cls,
        full_state: full_state_module.FullState,
    ) -> np.ndarray[np.int_, np.dtype]:
        size = len(full_state)
        result = np.zeros((size, 6), dtype=np.int_)
        pending_node_stack: list[tuple[int, int, core.INode]] = [(0, 0, full_state)]
        node_types = full_state.node_types()
        node_id = 0

        while pending_node_stack:
            current: tuple[int, int, core.INode] = pending_node_stack.pop()
            parent_id, arg_id, node = current
            node_id += 1
            idx = node_id - 1
            node_type_id = node_types.index(type(node)) + 1
            assert node_type_id > 0
            result[idx][0] = node_id
            result[idx][1] = parent_id
            result[idx][2] = arg_id
            result[idx][3] = node_type_id

            if isinstance(node, core.ISpecialValue):
                value_aux = node.node_value

                if isinstance(value_aux, core.IInt):
                    value = value_aux.as_int
                elif isinstance(value_aux, core.TypeNode):
                    value = node_types.index(value_aux.type) + 1
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
                    assert isinstance(arg, core.INode)
                    pending_node_stack.append((node_id, inner_arg_id, arg))

        return result

    @classmethod
    def symbolic(
        cls,
        node: core.BaseNode,
        node_types: tuple[type[core.INode], ...],
    ) -> sympy.Basic:
        assert isinstance(node, core.BaseNode)
        name = sympy.Symbol(node.func.__name__)

        if isinstance(node, core.ISpecialValue):
            value_aux = node.node_value

            if isinstance(value_aux, core.IInt):
                value = value_aux.as_int
                return sympy.Symbol(f'{name}[{value}]')
            elif isinstance(value_aux, core.TypeNode):
                value = node_types.index(value_aux.type) + 1
                type_name = r"\text{" + value_aux.type.__name__ + r"}"
                return sympy.Symbol(f'type[{value}][{type_name}]')
            else:
                raise ValueError(f'Invalid value type: {type(value_aux)}')

        raw_args = [arg.as_node for arg in node.args if isinstance(arg, core.INode)]
        assert len(raw_args) == len(node.args)

        args: tuple[sympy.Basic, ...] = tuple([
            cls.symbolic(arg, node_types) for arg in raw_args
        ])

        if any(isinstance(node, w) for w in WRAPPERS):
            return SympyWrapper(name, *args)

        if len(node.args) == 0:
            return name

        return SympyFunction(name, *args)

    def to_data_array(self) -> np.ndarray[np.int_, np.dtype]:
        return self.state_array(self.current_state)

    def to_symbolic(self) -> sympy.Basic:
        return self.symbolic(self.current_state, self.current_state.node_types())

    def to_symbolic_state(self) -> sympy.Basic:
        return self.symbolic(self.current_state.current_state.apply(), self.current_state.node_types())

