import typing
from environment.core import BASIC_NODE_TYPES, Integer, InheritableNode
from .state import State, BaseNode
from .action import BASIC_ACTION_TYPES, Action, ActionInput, ActionOutput
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

class NodeType(Integer):
    pass

class ActionType(Integer):
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
        allowed_nodes: tuple[typing.Type[BaseNode], ...],
        allowed_actions: tuple[typing.Type[Action], ...],
    ):
        self._goal = goal
        self._node_types = node_types
        self._allowed_nodes = allowed_nodes
        self._allowed_actions = allowed_actions

    @property
    def goal(self) -> GoalNode:
        return self._goal

    @property
    def node_types(self) -> tuple[typing.Type[BaseNode], ...]:
        return self._node_types

    @property
    def allowed_nodes(self) -> tuple[typing.Type[BaseNode], ...]:
        return self._allowed_nodes

    @property
    def allowed_actions(self) -> tuple[typing.Type[Action], ...]:
        return self._allowed_actions

    def to_node(self) -> MetaInfoNode:
        return MetaInfoNode(
            self.goal,
            NodeTypeGroup.from_types(len(self.node_types)),
            ActionTypeGroup.from_types(len(self.allowed_actions)),
        )

class FullEnvMetaInfo(EnvMetaInfo):
    def __init__(
        self,
        goal: GoalNode,
        reward_evaluator: RewardEvaluator | None = None,
        initial_history: tuple[StateHistoryItem, ...] = tuple(),
        node_types: tuple[typing.Type[BaseNode], ...] = BASIC_NODE_TYPES,
        allowed_nodes: tuple[typing.Type[BaseNode], ...] = tuple(),
        allowed_actions: tuple[typing.Type[Action], ...] = BASIC_ACTION_TYPES,
    ):
        super().__init__(
            goal=goal,
            node_types=node_types,
            allowed_nodes=allowed_nodes,
            allowed_actions=allowed_actions)

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
