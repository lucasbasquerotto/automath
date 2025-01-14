import typing
from env import meta_env
from env import full_state
from env import reward
from env import action
from env.environment import Environment
from env.env_utils import load_all_subclasses_sorted

T = typing.TypeVar("T")

class GoalEnv(Environment):
    def __init__(
        self,
        goal: meta_env.IGoal,
        fn_initial_state: typing.Callable[
            [meta_env.MetaInfo],
            full_state.FullState,
        ] | None = None,
        reward_evaluator: reward.IRewardEvaluator | None = None,
        allowed_actions: typing.Sequence[action.IAction] | None = None,
        max_steps: int | None = None,
    ):
        all_types = [t.as_type() for t in load_all_subclasses_sorted()]
        allowed_actions_typed = [
            t.as_type()
            for t in allowed_actions
        ] if allowed_actions is not None else None
        meta = meta_env.MetaInfo.with_defaults(
            goal=goal,
            all_types=all_types,
            allowed_actions=allowed_actions_typed,
        )
        initial_state = (
            fn_initial_state(meta)
            if fn_initial_state
            else full_state.FullState.with_child(meta))

        super().__init__(
            initial_state=initial_state,
            reward_evaluator=reward_evaluator,
            max_steps=max_steps,
        )
