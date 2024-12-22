import typing
from utils.types import BASIC_NODE_TYPES, ScopedNode, Integer, InheritableNode, ValueNode
from .state import State, BaseNode
from .action import BASIC_ACTION_TYPES, Action, ActionMetaInfo, ActionInput, ActionOutput
from .reward import RewardEvaluator, DefaultRewardEvaluator

class GoalNode(InheritableNode):
    def evaluate(self, state: State) -> bool:
        raise NotImplementedError()

class ActionData:
    def __init__(self, type: int, input: ActionInput, output: ActionOutput | None):
        self._type = type
        self._input = input
        self._output = output

    @property
    def type(self) -> int:
        return self._type

    @property
    def input(self) -> ActionInput:
        return self._input

    @property
    def output(self) -> ActionOutput | None:
        return self._output

StateHistoryItem = State | ActionData

class NodeTypeHandler:
    def get_value(self, node: BaseNode) -> int:
        raise NotImplementedError()

class DefaultNodeTypeHandler(NodeTypeHandler):
    def get_value(self, node: BaseNode) -> int:
        if isinstance(node, ScopedNode):
            return node.value
        if isinstance(node, Integer):
            return int(node)
        return 0

class NodeType(ValueNode):
    pass

class ActionType(ValueNode):
    pass

class NodeTypeGroup(InheritableNode):
    @classmethod
    def from_types(cls, amount: int) -> 'NodeTypeGroup':
        return cls(*[NodeType(i+1) for i in range(amount)])

class ActionTypeGroup(InheritableNode):
    @classmethod
    def from_types(cls, amount: int) -> 'ActionTypeGroup':
        return cls(*[ActionType(i+1) for i in range(amount)])

class MetaInfoNode(InheritableNode):
    def __init__(self, goal: GoalNode, node_types: NodeTypeGroup, action_types: ActionTypeGroup):
        super().__init__(goal, node_types, action_types)

class EnvMetaInfo:
    def __init__(
        self,
        goal: GoalNode,
        node_types: tuple[typing.Type[BaseNode], ...],
        node_type_handler: NodeTypeHandler,
        action_types: tuple[typing.Type[Action], ...],
    ):
        self._goal = goal
        self._node_types = node_types
        self._node_type_handler = node_type_handler
        self._action_types = action_types
        self._action_types_info = tuple([
            ActionMetaInfo(
                type_idx=i+1,
                arg_types=action.metadata().arg_types,
            ) for i, action in enumerate(action_types)
        ])

    @property
    def goal(self) -> GoalNode:
        return self._goal

    @property
    def node_types(self) -> tuple[typing.Type[BaseNode], ...]:
        return self._node_types

    @property
    def node_type_handler(self) -> NodeTypeHandler:
        return self._node_type_handler

    @property
    def action_types(self) -> tuple[typing.Type[Action], ...]:
        return self._action_types

    @property
    def action_types_info(self) -> tuple[ActionMetaInfo, ...]:
        return self._action_types_info

    def to_node(self) -> MetaInfoNode:
        return MetaInfoNode(
            self.goal,
            NodeTypeGroup.from_types(len(self.node_types)),
            ActionTypeGroup.from_types(len(self.action_types)),
        )

class FullEnvMetaInfo(EnvMetaInfo):
    def __init__(
        self,
        goal: GoalNode,
        reward_evaluator: RewardEvaluator | None = None,
        initial_history: tuple[StateHistoryItem, ...] = tuple(),
        node_types: tuple[typing.Type[BaseNode], ...] = BASIC_NODE_TYPES,
        node_type_handler: NodeTypeHandler = DefaultNodeTypeHandler(),
        action_types: tuple[typing.Type[Action], ...] = BASIC_ACTION_TYPES,
    ):
        super().__init__(
            goal=goal,
            node_types=node_types,
            node_type_handler=node_type_handler,
            action_types=action_types)

        reward_evaluator = (
            reward_evaluator
            if reward_evaluator is not None
            else DefaultRewardEvaluator(goal.evaluate))

        self._reward_evaluator = reward_evaluator
        self._initial_history = initial_history

    @property
    def reward_evaluator(self) -> RewardEvaluator:
        return self._reward_evaluator

    @property
    def initial_history(self) -> tuple[StateHistoryItem, ...]:
        return self._initial_history

    def is_terminal(self, state: State) -> bool:
        return self.goal.evaluate(state)
