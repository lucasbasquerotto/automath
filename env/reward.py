import math
from abc import ABC
from env.core import (
    INode,
    IDefault,
    InheritableNode,
    Integer,
    Protocol,
    CountableTypeGroup,
    TmpInnerArg,
    IInstantiable)
from env.full_state import FullState

class IRewardEvaluator(INode, ABC):

    def evaluate(self, current_state: FullState, next_state: FullState) -> float:
        raise NotImplementedError

class DefaultRewardEvaluator(InheritableNode, IRewardEvaluator, IDefault, IInstantiable):

    idx_goal_reward = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(Integer.as_type()))

    @classmethod
    def create(cls):
        return cls(Integer(10000))

    @property
    def goal_reward(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_goal_reward)

    def evaluate(self, current_state: FullState, next_state: FullState) -> float:
        if next_state.goal_achieved():
            goal_reward = self.goal_reward.apply().cast(Integer).as_int
            return goal_reward  # Reached the objective

        if next_state.is_last_step_error():
            return -100

        cost = next_state.final_cost().as_int
        reward = -math.log(cost + 1)

        return reward
