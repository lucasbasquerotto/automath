import importlib
import pkgutil
import typing
from env import core
from env import meta_env
from env import full_state
from env import reward
from env.environment import Environment

T = typing.TypeVar("T")

def import_submodules(package_name: str):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)

def get_all_subclasses(cls: type[T]) -> set[type[T]]:
    subclasses = set([cls])
    for subclass in cls.__subclasses__():
        subclasses.update(get_all_subclasses(subclass))
    return subclasses


def get_all_subclasses_sorted(cls: type[T]) -> list[type[T]]:
    import_submodules('env')
    result_set = get_all_subclasses(cls)
    result = sorted([t for t in result_set], key=lambda t: f'{t.__module__}.{t.__qualname__}')
    return result

class GoalEnv(Environment):
    def __init__(
        self,
        goal: meta_env.GoalNode,
        reward_evaluator: reward.IRewardEvaluator | None = None,
        max_steps: int | None = None,
    ):
        all_types = [t.as_type() for t in get_all_subclasses_sorted(core.INode)]
        meta = meta_env.MetaInfo.with_defaults(
            goal=goal,
            all_types=all_types,
        )

        super().__init__(
            initial_state=full_state.FullState.with_child(meta),
            reward_evaluator=reward_evaluator,
            max_steps=max_steps,
        )

        self.all_types = all_types
