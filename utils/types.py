import typing
import sympy

BaseNode = sympy.Basic

Integer = sympy.Integer

class ParamBaseNode(sympy.Basic):
    def __init__(self, *args: BaseNode):
        assert all(isinstance(arg, sympy.Basic) for arg in args)
        super().__init__()
        self._args = args

class InheritableNode(sympy.Basic):
    def __init__(self, *args: BaseNode):
        assert all(isinstance(arg, sympy.Basic) for arg in args)
        super().__init__()
        self._args = args

class EmptyNode(InheritableNode):
    def __init__(self):
        super().__init__()

class ValueNode(InheritableNode):
    @classmethod
    def prefix(cls):
        return 'unk'

    def __init__(self, value: int):
        super().__init__(sympy.Integer(value))

    @property
    def value(self) -> int:
        value = self.args[0]
        assert isinstance(value, sympy.Integer)
        return int(value)

class ScopedNode(InheritableNode):
    @classmethod
    def prefix(cls):
        return 'unk'

    def __init__(self, value: int):
        super().__init__(sympy.Integer(value), sympy.Dummy())

    def _latex(self, printer): # pylint: disable=unused-argument
        return r"%s_{%s}" % (self.prefix(), self.args[0])

    @property
    def value(self) -> int:
        value = self.args[0]
        assert isinstance(value, sympy.Integer)
        return int(value)

class FunctionDefinition(ScopedNode):
    @classmethod
    def prefix(cls):
        return 'f'

class ParamVar(ScopedNode):
    @classmethod
    def prefix(cls):
        return 'p'

class BooleanNode(InheritableNode):
    @property
    def value(self) -> bool | None:
        raise NotImplementedError

class MultiArgBooleanNode(BooleanNode):
    def __init__(self, *args: BaseNode):
        assert len(args) > 0
        assert all(isinstance(arg, sympy.Basic) for arg in args)
        super().__init__(*args)

    @property
    def value(self) -> bool | None:
        raise NotImplementedError

class FunctionParams(InheritableNode):
    def __init__(self, *params: ParamVar):
        assert all(isinstance(param, ParamVar) for param in params)
        super().__init__(*params)

    @property
    def params(self) -> tuple[ParamVar, ...]:
        return typing.cast(tuple[ParamVar, ...], self._args)

class FunctionInfo(InheritableNode):
    def __init__(self, expr: BaseNode, params: FunctionParams):
        assert isinstance(params, FunctionParams)
        assert isinstance(expr, BaseNode)
        super().__init__(params, expr)

    @property
    def expr(self) -> BaseNode:
        return self.args[0]

    @property
    def params_group(self) -> FunctionParams:
        params = self.args[1]
        assert isinstance(params, FunctionParams)
        return params

    @property
    def params(self) -> tuple[ParamVar, ...]:
        group = self.params_group
        return group.params

    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionInfo):
            return False

        my_params = self.params
        other_params = other.params

        return self.expr == other.expr.subs({
            other_params[i]: my_params[i]
            for i in range(min(len(other_params), len(my_params)))
        })

class ParamsGroup(InheritableNode):
    def __init__(self, *args: ParamVar):
        assert all(isinstance(arg, ParamVar) for arg in args)
        super().__init__(*args)

    @property
    def params(self) -> tuple[ParamVar, ...]:
        return typing.cast(tuple[ParamVar, ...], self._args)

class ArgsGroup(InheritableNode):
    def __init__(self, *args: BaseNode):
        assert all(isinstance(arg, sympy.Basic) for arg in args)
        super().__init__(*args)

    @property
    def arg_list(self) -> tuple[BaseNode, ...]:
        return typing.cast(tuple[BaseNode, ...], self._args)

    @classmethod
    def from_args(cls, args: typing.Sequence[BaseNode | None]) -> 'ArgsGroup':
        return ArgsGroup(*[arg if arg is not None else EmptyNode() for arg in args])

class ParamsArgsGroup(InheritableNode):
    def __init__(
        self,
        outer_params: ParamsGroup,
        inner_args: ArgsGroup,
    ):
        super().__init__(outer_params, inner_args)

    @property
    def outer_params_group(self) -> ParamsGroup:
        outer_params = self._args[0]
        assert isinstance(outer_params, ParamsGroup)
        return outer_params

    @property
    def inner_args_group(self) -> ArgsGroup:
        inner_args = self._args[1]
        assert isinstance(inner_args, ArgsGroup)
        return inner_args

    @property
    def outer_params(self) -> tuple[ParamVar, ...]:
        outer_params = self._args[0]
        assert isinstance(outer_params, ParamsGroup)
        return outer_params.params

    @property
    def inner_args(self) -> tuple[BaseNode | None, ...]:
        inner_args_group = self.inner_args_group
        arg_list = inner_args_group.arg_list
        return tuple([
            (a if not isinstance(a, EmptyNode) else None)
            for a in arg_list])

BASIC_NODE_TYPES = (
    FunctionDefinition,
    ParamVar,
    sympy.Integer,
)
