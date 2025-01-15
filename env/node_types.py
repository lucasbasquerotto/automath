from env.core import (
    INode,
    IOptional,
    Eq,
    Void,
    IBoolean,
    IDefault,
    IInstantiable,
)
from env.state import (
    State,
    StateScratchIndex,
    IGoal,
    Goal,
    StateDynamicGoalIndex,
)

class HaveScratch(Goal[INode, StateScratchIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return INode

    @classmethod
    def eval_param_type(cls):
        return StateScratchIndex

    def evaluate(self, state: State, eval_param: StateScratchIndex):
        goal_inner_expr = self.goal_inner_expr.apply()
        assert isinstance(eval_param, StateScratchIndex)
        scratch = eval_param.find_in_node(state).value_or_raise
        content = scratch.child.apply().cast(IOptional).value_or_raise
        return Eq(content, goal_inner_expr)

class HaveDynamicGoal(Goal[IGoal, StateDynamicGoalIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return IGoal

    @classmethod
    def eval_param_type(cls):
        return StateDynamicGoalIndex

    def evaluate(self, state: State, eval_param: StateDynamicGoalIndex):
        goal_inner_expr = self.goal_inner_expr.apply()
        assert isinstance(eval_param, StateDynamicGoalIndex)
        dynamic_goal = eval_param.find_in_node(state).value_or_raise
        content = dynamic_goal.goal_expr.apply()
        return Eq(content, goal_inner_expr)

class HaveDynamicGoalAchieved(Goal[Void, StateDynamicGoalIndex], IDefault, IInstantiable):

    @classmethod
    def goal_type(cls):
        return Void

    @classmethod
    def eval_param_type(cls):
        return StateDynamicGoalIndex

    @classmethod
    def create(cls):
        return cls.with_goal(Void())

    def evaluate(self, state: State, eval_param: StateDynamicGoalIndex):
        assert isinstance(eval_param, StateDynamicGoalIndex)
        dynamic_goal = eval_param.find_in_node(state).value_or_raise
        content = dynamic_goal.goal_achieved.apply().cast(IBoolean)
        return content
