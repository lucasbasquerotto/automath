import sympy

BaseNode = sympy.Basic

Assumption = sympy.Basic

class IndexedElem(sympy.Basic):
    @classmethod
    def prefix(cls):
        return 'unk'

    def __init__(self, index: int):
        super().__init__()
        self._args = (sympy.Integer(index), sympy.Dummy())

    def _latex(self, printer): # pylint: disable=unused-argument
        return r"%s_{%s}" % (self.prefix(), self.args[0])

    @property
    def index(self) -> int:
        index = self.args[0]
        assert isinstance(index, sympy.Integer)
        return int(index)

class FunctionDefinition(IndexedElem):
    @classmethod
    def prefix(cls):
        return 'f'

class ParamVar(IndexedElem):
    @classmethod
    def prefix(cls):
        return 'p'

class ExprInfo:
    def __init__(self, expr: BaseNode, params: tuple[ParamVar, ...]):
        self._expr = expr
        self._params = params

    @property
    def expr(self) -> BaseNode:
        return self._expr

    @property
    def params(self) -> tuple[ParamVar, ...]:
        return self._params

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExprInfo):
            return False

        return self.expr == other.expr.subs({
            other_param: self_param
            for other_param, self_param in zip(other.params, self.params)
        })

