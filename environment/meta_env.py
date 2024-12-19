import typing
from utils.types import BASIC_NODE_TYPES, IndexedElem, Integer
from .state import State, BaseNode
from .action import BASIC_ACTIONS, Action, ActionMetaInfo, ActionInput, ActionOutput
from .reward import RewardEvaluator

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
        if isinstance(node, IndexedElem):
            return node.index
        if isinstance(node, Integer):
            return int(node)
        return 0

class EnvMetaInfo:
    def __init__(
        self,
        main_context: int,
        node_types: tuple[typing.Type[BaseNode], ...],
        node_type_handler: NodeTypeHandler,
        action_types: tuple[typing.Type[Action], ...],
    ):
        self._main_context = main_context
        self._node_types = node_types
        self._node_type_handler = node_type_handler
        self._action_types = action_types
        self._action_types_info = tuple([
            ActionMetaInfo(
                type_idx=i,
                arg_types=action.metadata().arg_types,
            ) for i, action in enumerate(action_types)
        ])

    @property
    def main_context(self) -> int:
        return self._main_context

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

class FullEnvMetaInfo(EnvMetaInfo):
    def __init__(
        self,
        main_context: int,
        reward_evaluator: RewardEvaluator,
        initial_history: tuple[StateHistoryItem, ...],
        is_terminal: typing.Callable[[State], bool],
        node_types: tuple[typing.Type[BaseNode], ...] = BASIC_NODE_TYPES,
        node_type_handler: NodeTypeHandler = DefaultNodeTypeHandler(),
        action_types: tuple[typing.Type[Action], ...] = BASIC_ACTIONS,
    ):
        super().__init__(
            main_context=main_context,
            node_types=node_types,
            node_type_handler=node_type_handler,
            action_types=action_types)
        self._reward_evaluator = reward_evaluator
        self._initial_history = initial_history
        self._is_terminal = is_terminal

    @property
    def reward_evaluator(self) -> RewardEvaluator:
        return self._reward_evaluator

    @property
    def initial_history(self) -> tuple[StateHistoryItem, ...]:
        return self._initial_history

    def is_terminal(self, state: State) -> bool:
        return self._is_terminal(state)
