import typing
import sympy

BaseNode = sympy.Basic

Integer = sympy.Integer

class InheritableNode(sympy.Basic):
    def __init__(self, *args: BaseNode):
        assert all(isinstance(arg, sympy.Basic) for arg in args)
        super().__init__()
        self._args = args

class UniqueElem(InheritableNode):
    @classmethod
    def prefix(cls):
        return 'unk'

    def __init__(self, value: int):
        super().__init__()
        self._args = (sympy.Integer(value), sympy.Dummy())

    def _latex(self, printer): # pylint: disable=unused-argument
        return r"%s_{%s}" % (self.prefix(), self.args[0])

    @property
    def value(self) -> int:
        value = self.args[0]
        assert isinstance(value, sympy.Integer)
        return int(value)

class FunctionDefinition(UniqueElem):
    @classmethod
    def prefix(cls):
        return 'f'

class ParamVar(UniqueElem):
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
        super().__init__()
        self._args = args

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

class FunctionNode(InheritableNode):
    def __init__(self, expr: BaseNode, params: FunctionParams):
        assert isinstance(params, FunctionParams)
        assert isinstance(expr, BaseNode)
        super().__init__(params, expr)

    @property
    def expr(self) -> BaseNode:
        return self.args[0]

    @property
    def params(self) -> FunctionParams:
        params = self.args[1]
        assert isinstance(params, FunctionParams)
        return params

class FunctionInfo:
    def __init__(self, expr: BaseNode, params: tuple[ParamVar, ...]):
        self._expr = expr
        self._params = params

    @property
    def expr(self) -> BaseNode:
        return self._expr

    @property
    def params(self) -> tuple[ParamVar, ...]:
        return self._params

    def to_expr(self) -> FunctionNode:
        return FunctionNode(self._expr, FunctionParams(*self._params))

    @classmethod
    def from_expr(cls, expr: FunctionNode) -> 'FunctionInfo':
        return cls(expr.expr, expr.params.params)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionInfo):
            return False

        return self.expr == other.expr.subs({
            other.params[i]: self.params[i]
            for i in range(min(len(other.params), len(self.params)))
        })

class ArgGroup:
    def __init__(
        self,
        outer_params: tuple[ParamVar, ...],
        inner_args: tuple[BaseNode | None, ...],
    ):
        self._outer_params = outer_params
        self._inner_args = inner_args

    @property
    def outer_params(self) -> tuple[ParamVar, ...]:
        return self._outer_params

    @property
    def inner_args(self) -> tuple[BaseNode | None, ...]:
        return self._inner_args

BASIC_NODE_TYPES = (
    FunctionDefinition,
    ParamVar,
    sympy.Integer,
)
