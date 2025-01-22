from abc import ABC
from env.core import (
    INode,
    IDefault,
    InheritableNode,
    Integer,
    ExtendedTypeGroup,
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
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            Integer,
        ]))

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

        weight = len(next_state)

        if next_state.current_state.apply() == current_state.current_state.apply():
            return -10 * weight # No change applied
        return -weight  # Small penalty for each step taken
