from env.core import (
    INode,
    TmpNestedArg,
    Eq,
    IInstantiable)
from env.state import State, StateScratchIndex
from env.meta_env import Goal

class HaveScratch(Goal[INode, StateScratchIndex], IInstantiable):

    idx_definition_expr = 1

    @classmethod
    def goal_type(cls):
        return INode

    @classmethod
    def eval_param_type(cls):
        return StateScratchIndex

    @property
    def definition_expr(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_definition_expr)

    def evaluate(self, state: State, eval_param: StateScratchIndex):
        goal = self.goal.apply()
        assert isinstance(eval_param, StateScratchIndex)
        scratch = eval_param.find_in_node(state).value_or_raise
        content = scratch.content
        return Eq(content, goal)
