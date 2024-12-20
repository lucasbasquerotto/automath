import sympy
from utils.types import FunctionNode, BooleanNode, MultiArgBooleanNode
from environment.state import State
from environment.meta_env import GoalNode

class HaveDefinition(GoalNode):
    def __init__(self, definition: FunctionNode):
        assert isinstance(definition, FunctionNode)
        super().__init__()
        self._args = (definition,)

    def evaluate(self, state: State) -> bool:
        definition = self.args[0]
        assert isinstance(definition, FunctionNode)
        return definition.expr in [f for _, f in state.definitions]

class TrueNode(BooleanNode):
    def __init__(self):
        super().__init__()
        self._args = ()

    @property
    def value(self) -> bool:
        return True

class FalseNode(BooleanNode):
    def __init__(self):
        super().__init__()
        self._args = ()

    @property
    def value(self) -> bool:
        return False

class AndNode(MultiArgBooleanNode):
    @property
    def value(self) -> bool | None:
        args = self.args
        has_none = False
        for arg in args:
            if not isinstance(arg, BooleanNode):
                has_none = True
            elif arg.value is None:
                has_none = True
            elif arg.value is False:
                return False
        return None if has_none else True

class OrNode(MultiArgBooleanNode):
    @property
    def value(self) -> bool | None:
        args = self.args
        has_none = False
        for arg in args:
            if not isinstance(arg, BooleanNode):
                has_none = True
            elif arg.value is None:
                has_none = True
            elif arg.value is True:
                return True
        return None if has_none else False

class Add(sympy.Expr):
    def __new__(cls, *args):
        super().__new__(cls, *args)

    def doit(self, **hints):
        return sympy.Add(*self.args).doit(**hints)

    def _eval_nseries(self, x, n, logx, cdir):
        return sympy.Add(*self.args).series(x, n, logx, cdir)

