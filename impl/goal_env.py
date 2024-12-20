import typing
from utils.types import BASIC_NODE_TYPES, BaseNode
from environment.action import BASIC_ACTION_TYPES, Action
from environment.reward import RewardEvaluator
from environment.full_state import StateHistoryItem
from environment.environment import Environment
from environment.meta_env import FullEnvMetaInfo, NodeTypeHandler, DefaultNodeTypeHandler, GoalNode

class GoalEnv(Environment):
    def __init__(
        self,
        goal: GoalNode,
        reward_evaluator: RewardEvaluator | None = None,
        initial_history: tuple[StateHistoryItem, ...] = tuple(),
        node_types: tuple[typing.Type[BaseNode], ...] = BASIC_NODE_TYPES,
        node_type_handler: NodeTypeHandler = DefaultNodeTypeHandler(),
        action_types: tuple[typing.Type[Action], ...] = BASIC_ACTION_TYPES,
        max_steps: int | None = None,
        max_history_size: int | None = None,
    ):
        meta = FullEnvMetaInfo(
            goal=goal,
            reward_evaluator=reward_evaluator,
            initial_history=initial_history,
            node_types=node_types,
            node_type_handler=node_type_handler,
            action_types=action_types,
        )

        super().__init__(
            meta=meta,
            max_steps=max_steps,
            max_history_size=max_history_size,
        )
