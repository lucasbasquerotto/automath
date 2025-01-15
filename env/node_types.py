from env.core import (
    INode,
    IOptional,
    Eq,
    IInstantiable,
)
from env.state import (
    State,
    StateScratchIndex,
    IGoal,
    Goal,
    StateDynamicGoalIndex,
    IGoalAchieved,
)

class HaveScratch(Goal[INode, StateScratchIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return INode

    @classmethod
    def eval_param_type(cls):
        return StateScratchIndex

    def evaluate(self, state: State, eval_param: StateScratchIndex):
        goal = self.goal.apply()
        assert isinstance(eval_param, StateScratchIndex)
        scratch = eval_param.find_in_node(state).value_or_raise
        content = scratch.child.apply().cast(IOptional).value_or_raise
        return Eq(content, goal)

class HaveDynamicGoal(Goal[IGoal, StateDynamicGoalIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return IGoal

    @classmethod
    def eval_param_type(cls):
        return StateDynamicGoalIndex

    def evaluate(self, state: State, eval_param: StateDynamicGoalIndex):
        goal = self.goal.apply()
        assert isinstance(eval_param, StateDynamicGoalIndex)
        dynamic_goal = eval_param.find_in_node(state).value_or_raise
        content = dynamic_goal.goal.apply()
        return Eq(content, goal)

class HaveDynamicGoalAchieved(Goal[IGoalAchieved, StateDynamicGoalIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return IGoalAchieved

    @classmethod
    def eval_param_type(cls):
        return StateDynamicGoalIndex

    def evaluate(self, state: State, eval_param: StateDynamicGoalIndex):
        goal_achieved = self.goal.apply()
        assert isinstance(eval_param, StateDynamicGoalIndex)
        dynamic_goal = eval_param.find_in_node(state).value_or_raise
        content = dynamic_goal.goal_achieved.apply()
        return Eq(content, goal_achieved)
