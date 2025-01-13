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
    state.IContext,
    meta_env.MetaInfo,
    meta_env.SubtypeOuterGroup,
    action_module.FullActionOutput,
    full_state_module.FullState,
    full_state_module.HistoryNode,
    full_state_module.HistoryGroupNode,
)

class SympyShared(sympy.Basic):

    def __init__(self, node_id: sympy.Integer, name: sympy.Symbol, *args: sympy.Basic):
        super().__init__()
        self._args = (node_id, name, *args)

    def _data(self) -> tuple[str, tuple[sympy.Basic, ...]]:
        node_id = self.args[0]
        assert isinstance(node_id, sympy.Integer)
        name = self.args[1]
        assert isinstance(name, sympy.Symbol)
        args = self.args[2:]
        amount = len(args)
        name_str = r"\text{" + name.name + r"<" + str(node_id) + r">}"
        amount_str = r"\{" + str(amount) + r"\}"
        node_name = f"{name_str}{amount_str}"
        return node_name, args

class SympyWrapper(SympyShared):

    def _latex(self, printer: LatexPrinter) -> str:
        node_name, args = self._data()

        if len(args) == 0:
            return node_name
        args_latex = r" \\ ".join(
            r"\{" + str(i+1) + r"\}\text{ }" + printer.doprint(arg)
            for i, arg in enumerate(args))
        begin = r"\begin{cases}"
        end = r"\end{cases}"

        return f"{node_name} {begin} {args_latex} {end}"

class SympyFunction(SympyShared):

    def _latex(self, printer: LatexPrinter) -> str:
        node_name, args = self._data()

        newline = r" \\ \text{} "
        args_latex = newline.join(printer.doprint(arg) for arg in args)
        if len(args) == 0:
            return node_name
        if len(args) <= 1:
            return f"{node_name}({args_latex})"
        args_latex = args_latex.replace(newline, newline + r" \quad ")

        return f"{node_name}( {newline} \\quad {args_latex} {newline} )"


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
        pending_node_stack: list[tuple[int, int, int, int, core.INode]] = [(0, 0, 1, 0, root_node)]
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
            result[idx][0] = node_id
            result[idx][1] = parent_id
            result[idx][2] = arg_id
            result[idx][3] = parent_scope_id
            result[idx][4] = context_node_id
            result[idx][5] = node_type_id

            scope_id = parent_scope_id

            if isinstance(node, core.Scope):
                scope_id_wrapper = node.id
                scope_id = 0

                if isinstance(scope_id_wrapper, core.ScopeId):
                    scope_id_aux = scope_id_wrapper.as_int
                    if isinstance(node, core.OpaqueScope) or (0 < parent_scope_id < scope_id_aux):
                        scope_id = scope_id_aux

                result[idx][6] = scope_id

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

    @classmethod
    def symbolic(
        cls,
        node: core.BaseNode,
        node_types: tuple[type[core.INode], ...],
    ) -> sympy.Basic:
        assert isinstance(node, core.BaseNode)
        node_id = node_types.index(node.func) + 1
        name = node.func.__name__

        if isinstance(node, core.ISpecialValue):
            value_aux = node.node_value

            if isinstance(value_aux, core.IInt):
                value = value_aux.as_int
                name_str = r"\text{" + name + r"<" + str(node_id) + r">}"
                return sympy.Symbol(f'{name_str}[{value}]')
            elif isinstance(value_aux, core.TypeNode):
                name_str = r"\text{" + name + r"<" + str(node_id) + r">}"
                type_id = node_types.index(value_aux.type) + 1
                type_name = value_aux.type.__name__
                type_name_str = r"\text{" + type_name + r"<" + str(type_id) + r">}"
                return sympy.Symbol(f'{name_str}[{type_name_str}]')
            else:
                raise ValueError(f'Invalid value type: {type(value_aux)}')

        if isinstance(node, core.Placeholder):
            name_str = r"\text{" + name + r"<" + str(node_id) + r">}"
            scope_id = node.parent_scope.apply()
            scope_id_str = r"{" + sympy.latex(cls.symbolic(scope_id, node_types)) + r"}"
            index = node.index.apply()
            index_str = r"{" + sympy.latex(cls.symbolic(index, node_types)) + r"}"
            type_node = node.type_node.apply()
            type_node_str = (
                r"[" + sympy.latex(cls.symbolic(type_node, node_types)) + r"]"
                if not isinstance(type_node, core.UnknownType)
                else ''
            )
            return sympy.Symbol(f'{name_str}_{index_str}^{scope_id_str}{type_node_str}')

        raw_args = [arg.as_node for arg in node.args if isinstance(arg, core.INode)]
        assert len(raw_args) == len(node.args)

        args: tuple[sympy.Basic, ...] = tuple([
            cls.symbolic(arg, node_types) for arg in raw_args
        ])

        outer_args = (sympy.Integer(node_id), sympy.Symbol(name), *args)

        if any(isinstance(node, w) for w in WRAPPERS):
            return SympyWrapper(*outer_args)

        return SympyFunction(*outer_args)

    def to_data_array(self) -> np.ndarray[np.int_, np.dtype]:
        node_types = self.full_state.node_types()
        return self.data_array(self.full_state, node_types)

    def to_data_array_current_state(self) -> np.ndarray[np.int_, np.dtype]:
        current_state = self.full_state.current_state.apply()
        node_types = self.full_state.node_types()
        return self.data_array(current_state, node_types)

    def to_symbolic(self) -> sympy.Basic:
        node_types = self.full_state.node_types()
        return self.symbolic(self.full_state, node_types)

    def to_symbolic_current_state(self) -> sympy.Basic:
        current_state = self.full_state.current_state.apply()
        node_types = self.full_state.node_types()
        return self.symbolic(current_state, node_types)
