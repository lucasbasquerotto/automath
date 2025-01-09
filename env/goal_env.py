import typing
from env import meta_env
from env import full_state
from env import reward
from env.environment import Environment
from env.env_utils import load_all_subclasses_sorted

T = typing.TypeVar("T")

class GoalEnv(Environment):
    def __init__(
        self,
        goal: meta_env.GoalNode,
        reward_evaluator: reward.IRewardEvaluator | None = None,
        max_steps: int | None = None,
    ):
        all_types = [t.as_type() for t in load_all_subclasses_sorted()]
        meta = meta_env.MetaInfo.with_defaults(
            goal=goal,
            all_types=all_types,
        )

        super().__init__(
            initial_state=full_state.FullState.with_child(meta),
            reward_evaluator=reward_evaluator,
            max_steps=max_steps,
        )
