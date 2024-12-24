import typing
import numpy as np
from utils.logger import logger
from environment.core import ScopedIntegerNode, InheritableNode
from .state import State, BaseNode, FunctionInfo, FunctionParams, ParamsArgsGroup
from .action import (
    Action,
    ActionInfo,
    InvalidActionException,
    ActionOutput,
    NewPartialDefinitionActionOutput,
    PartialActionOutput,
    RemovePartialDefinitionActionOutput,
    NewArgGroupActionOutput,
    ArgFromExprActionOutput,
    RemoveArgGroupActionOutput,
    NewDefinitionFromPartialActionOutput,
    ReplaceByDefinitionActionOutput,
    ExpandDefinitionActionOutput,
    ReformulationActionOutput)
from .meta_env import EnvMetaInfo, ActionData, StateHistoryItem, MetaInfoNode

class HistoryNode(InheritableNode):
    def __init__(self, state: State, action: ActionInfo | None):
        assert isinstance(state, State)
        if action is not None:
            assert isinstance(action, ActionInfo)
            super().__init__(state, action)
        else:
            super().__init__(state)

    @property
    def state(self) -> State:
        state = self.args[0]
        assert isinstance(state, State)
        return state

class HistoryGroupNode(InheritableNode):
    def __init__(self, *args: HistoryNode):
        assert all(isinstance(arg, HistoryNode) for arg in args)
        super().__init__(*args)

    @property
    def expand(self) -> tuple[HistoryNode, ...]:
        return typing.cast(tuple[HistoryNode, ...], self.args)

class FullStateNode(InheritableNode):
    def __init__(self, meta: MetaInfoNode, current: HistoryNode, history: HistoryGroupNode):
        assert isinstance(meta, EnvMetaInfo)
        assert isinstance(current, HistoryNode)
        assert isinstance(history, HistoryGroupNode)
        super().__init__(meta, current, history)

    @property
    def meta(self) -> EnvMetaInfo:
        meta = self.args[0]
        assert isinstance(meta, EnvMetaInfo)
        return meta

    @property
    def current(self) -> HistoryNode:
        current = self.args[1]
        assert isinstance(current, HistoryNode)
        return current

    @property
    def history(self) -> HistoryGroupNode:
        history = self.args[2]
        assert isinstance(history, HistoryGroupNode)
        return history

UNDEFINED_OR_EMPTY_FIELD = 0

HISTORY_TYPE_META = 1
HISTORY_TYPE_STATE = 2
HISTORY_TYPE_ACTION = 3

ALL_HISTORY_TYPES = [
    HISTORY_TYPE_META,
    HISTORY_TYPE_STATE,
    HISTORY_TYPE_ACTION,
]

CONTEXT_HISTORY_TYPE = 1
CONTEXT_META_MAIN = 2
CONTEXT_STATE_DEFINITION = 3
CONTEXT_STATE_PARTIAL_DEFINITION = 4
CONTEXT_STATE_ARG_GROUP = 5
CONTEXT_ACTION_TYPE = 6
CONTEXT_ACTION_INPUT = 7
CONTEXT_ACTION_OUTPUT = 8
CONTEXT_ACTION_STATUS = 9
CONTEXT_META_TRANSITION = 10

ALL_CONTEXTS = [
    CONTEXT_HISTORY_TYPE,
    CONTEXT_META_MAIN,
    CONTEXT_STATE_DEFINITION,
    CONTEXT_STATE_PARTIAL_DEFINITION,
    CONTEXT_STATE_ARG_GROUP,
    CONTEXT_ACTION_TYPE,
    CONTEXT_ACTION_INPUT,
    CONTEXT_ACTION_OUTPUT,
    CONTEXT_ACTION_STATUS,
    CONTEXT_META_TRANSITION,
]

SUBCONTEXT_ACTION_INPUT_AMOUNT = 1
SUBCONTEXT_ACTION_INPUT_ARG = 2
SUBCONTEXT_ACTION_OUTPUT_PARTIAL_DEFINITION_IDX = 3
SUBCONTEXT_ACTION_OUTPUT_DEFINITION_IDX = 4
SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_IDX = 5
SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_PARAMS_AMOUNT = 6
SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_ARGS_AMOUNT = 7
SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP = 8
SUBCONTEXT_ACTION_OUTPUT_EXPR_ID = 9
SUBCONTEXT_ACTION_OUTPUT_NODE_IDX = 10
SUBCONTEXT_ACTION_OUTPUT_NODE_EXPR = 11
SUBCONTEXT_META_ACTION_TYPE = 12

ALL_SUBCONTEXTS = [
    SUBCONTEXT_ACTION_INPUT_AMOUNT,
    SUBCONTEXT_ACTION_INPUT_ARG,
    SUBCONTEXT_ACTION_OUTPUT_PARTIAL_DEFINITION_IDX,
    SUBCONTEXT_ACTION_OUTPUT_DEFINITION_IDX,
    SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_IDX,
    SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_PARAMS_AMOUNT,
    SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_ARGS_AMOUNT,
    SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP,
    SUBCONTEXT_ACTION_OUTPUT_EXPR_ID,
    SUBCONTEXT_ACTION_OUTPUT_NODE_IDX,
    SUBCONTEXT_ACTION_OUTPUT_NODE_EXPR,
    SUBCONTEXT_META_ACTION_TYPE,
]

GROUP_CONTEXT_ARG_GROUP = 1
GROUP_CONTEXT_ACTION_TYPE = 2

ALL_GROUP_CONTEXTS = [
    GROUP_CONTEXT_ARG_GROUP,
    GROUP_CONTEXT_ACTION_TYPE,
]

GROUP_SUBCONTEXT_ARG_GROUP_MAIN = 1
GROUP_SUBCONTEXT_ARG_GROUP_PARAMS_AMOUNT = 2
GROUP_SUBCONTEXT_ARG_GROUP_ARGS_AMOUNT = 3
GROUP_SUBCONTEXT_ARG_GROUP_EXPR = 4
GROUP_SUBCONTEXT_ACTION_TYPE_MAIN = 5
GROUP_SUBCONTEXT_ACTION_TYPE_ARG = 6

ALL_GROUP_SUBCONTEXTS = [
    GROUP_SUBCONTEXT_ARG_GROUP_MAIN,
    GROUP_SUBCONTEXT_ARG_GROUP_PARAMS_AMOUNT,
    GROUP_SUBCONTEXT_ARG_GROUP_ARGS_AMOUNT,
    GROUP_SUBCONTEXT_ARG_GROUP_EXPR,
    GROUP_SUBCONTEXT_ACTION_TYPE_MAIN,
    GROUP_SUBCONTEXT_ACTION_TYPE_ARG,
]

ITEM_CONTEXT_SYMBOL_IDX = 1
ITEM_CONTEXT_TYPE_IDX = 2
ITEM_CONTEXT_PARAM_IDX = 3
ITEM_CONTEXT_EXPR_IDX = 4

ALL_ITEM_CONTEXTS = [
    ITEM_CONTEXT_SYMBOL_IDX,
    ITEM_CONTEXT_TYPE_IDX,
    ITEM_CONTEXT_PARAM_IDX,
    ITEM_CONTEXT_EXPR_IDX,
]

ACTION_STATUS_SUCCESS_ID = 1
ACTION_STATUS_FAIL_ID = 2

ALL_ACTION_STATUSES = [
    ACTION_STATUS_SUCCESS_ID,
    ACTION_STATUS_FAIL_ID,
]

def _validate_indexes(idx_list: list[int]):
    for i, arg in enumerate(idx_list):
        assert arg == i + 1
_validate_indexes(ALL_HISTORY_TYPES)
_validate_indexes(ALL_CONTEXTS)
_validate_indexes(ALL_SUBCONTEXTS)
_validate_indexes(ALL_GROUP_CONTEXTS)
_validate_indexes(ALL_GROUP_SUBCONTEXTS)
_validate_indexes(ALL_ITEM_CONTEXTS)
_validate_indexes(ALL_ACTION_STATUSES)

ACTION_OUTPUT_TYPES = [
    NewPartialDefinitionActionOutput,
    PartialActionOutput,
    RemovePartialDefinitionActionOutput,
    NewArgGroupActionOutput,
    ArgFromExprActionOutput,
    RemoveArgGroupActionOutput,
    NewDefinitionFromPartialActionOutput,
    ReplaceByDefinitionActionOutput,
    ExpandDefinitionActionOutput,
    ReformulationActionOutput,
]

class NodeItemData:
    def __init__(
        self,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int,
        group_idx: int,
        group_context: int,
        group_subcontext: int,
        item_idx: int,
        item_context: int,
        parent_node_idx: int,
        node_idx: int,
        composite_node: int,
        node_type: int,
        node_value: int,
        history_expr_id: int | None,
        expr: BaseNode | None,
    ):
        assert history_number >= 0
        assert history_type in ALL_HISTORY_TYPES
        assert context in ALL_CONTEXTS
        assert subcontext in ALL_SUBCONTEXTS + [UNDEFINED_OR_EMPTY_FIELD]
        assert group_idx >= 0
        assert group_context in ALL_GROUP_CONTEXTS + [UNDEFINED_OR_EMPTY_FIELD]
        assert group_subcontext in ALL_GROUP_SUBCONTEXTS + [UNDEFINED_OR_EMPTY_FIELD]
        assert item_idx >= 0
        assert item_context in ALL_ITEM_CONTEXTS + [UNDEFINED_OR_EMPTY_FIELD]

        if history_expr_id is not None:
            assert parent_node_idx > 0
            assert node_idx > 0
            assert composite_node in [0, 1]
            assert node_type > 0
            assert history_expr_id > 0
            assert expr is not None
        else:
            assert parent_node_idx == UNDEFINED_OR_EMPTY_FIELD
            assert node_idx == UNDEFINED_OR_EMPTY_FIELD
            assert composite_node == UNDEFINED_OR_EMPTY_FIELD
            assert node_type == UNDEFINED_OR_EMPTY_FIELD
            assert expr is None

        assert node_value >= 0

        self._history_number = history_number
        self._history_type = history_type
        self._context = context
        self._subcontext = subcontext
        self._group_idx = group_idx
        self._group_context = group_context
        self._group_subcontext = group_subcontext
        self._item_idx = item_idx
        self._item_context = item_context
        self._parent_node_idx = parent_node_idx
        self._node_idx = node_idx
        self._composite_node = composite_node
        self._node_type = node_type
        self._node_value = node_value
        self._history_expr_id = history_expr_id
        self._expr = expr

    @classmethod
    def with_defaults(
        cls,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int = UNDEFINED_OR_EMPTY_FIELD,
        group_idx: int = UNDEFINED_OR_EMPTY_FIELD,
        group_context: int = UNDEFINED_OR_EMPTY_FIELD,
        group_subcontext: int = UNDEFINED_OR_EMPTY_FIELD,
        item_idx: int = UNDEFINED_OR_EMPTY_FIELD,
        item_context: int = UNDEFINED_OR_EMPTY_FIELD,
        parent_node_idx: int = UNDEFINED_OR_EMPTY_FIELD,
        node_idx: int = UNDEFINED_OR_EMPTY_FIELD,
        composite_node: int = UNDEFINED_OR_EMPTY_FIELD,
        node_type: int = UNDEFINED_OR_EMPTY_FIELD,
        node_value: int = UNDEFINED_OR_EMPTY_FIELD,
        history_expr_id: int | None = None,
        expr: BaseNode | None = None,
    ) -> 'NodeItemData':
        return cls(
            history_number=history_number,
            history_type=history_type,
            context=context,
            subcontext=subcontext,
            group_idx=group_idx,
            group_context=group_context,
            group_subcontext=group_subcontext,
            item_idx=item_idx,
            item_context=item_context,
            parent_node_idx=parent_node_idx,
            node_idx=node_idx,
            composite_node=composite_node,
            node_type=node_type,
            node_value=node_value,
            history_expr_id=history_expr_id,
            expr=expr,
        )

    @property
    def history_number(self) -> int:
        return self._history_number

    @property
    def history_type(self) -> int:
        return self._history_type

    @property
    def context(self) -> int:
        return self._context

    @property
    def subcontext(self) -> int:
        return self._subcontext

    @property
    def group_idx(self) -> int:
        return self._group_idx

    @property
    def group_context(self) -> int:
        return self._group_context

    @property
    def group_subcontext(self) -> int:
        return self._group_subcontext

    @property
    def item_idx(self) -> int:
        return self._item_idx

    @property
    def item_context(self) -> int:
        return self._item_context

    @property
    def parent_node_idx(self) -> int:
        return self._parent_node_idx

    @property
    def node_idx(self) -> int:
        return self._node_idx

    @property
    def composite_node(self) -> int:
        return self._composite_node

    @property
    def node_type(self) -> int:
        return self._node_type

    @property
    def node_value(self) -> int:
        return self._node_value

    @property
    def history_expr_id(self) -> int | None:
        return self._history_expr_id

    @property
    def expr(self) -> BaseNode | None:
        return self.expr

    def to_array(self) -> np.ndarray[np.int_, np.dtype]:
        return np.array([
            self._history_number,
            self._history_type,
            self._context,
            self._subcontext,
            self._group_idx,
            self._group_context,
            self._group_subcontext,
            self._item_idx,
            self._item_context,
            self._parent_node_idx,
            self._node_idx,
            self._composite_node,
            self._node_type,
            self._node_value,
            self._history_expr_id or UNDEFINED_OR_EMPTY_FIELD,
        ])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeItemData):
            return False
        return np.array_equal(self.to_array(), other.to_array())

class FullState:
    def __init__(
        self,
        meta: EnvMetaInfo,
        history: tuple[StateHistoryItem, ...],
        last_history_idx: int | None = None,
        max_history_state_size: int | None = None,
    ):
        assert len(meta.node_types) > 0
        assert len(meta.node_types) == len(set(meta.node_types))
        assert meta.node_type_handler is not None
        assert len(meta.action_types) > 0
        assert len(meta.action_types) == len(set(meta.action_types))
        if max_history_state_size is not None:
            assert max_history_state_size >= 0

        if len(history) == 0:
            history = (
                State.from_raw(
                    definitions=tuple(),
                    partial_definitions=tuple(),
                    arg_groups=tuple()
                ),
            )

        assert any(isinstance(item, State) for item in history)

        last_history_idx = last_history_idx if last_history_idx is not None else len(history)

        self._meta = meta
        self._history = history
        self._last_history_idx = last_history_idx
        self._max_history_state_size = max_history_state_size

    @property
    def last_state(self) -> State:
        state: State | None = next(
            (item for item in self._history[::-1] if isinstance(item, State)),
            None
        )
        assert state is not None
        return state

    def apply(self, action: Action) -> 'FullState':
        last_state = self.last_state
        assert isinstance(last_state, State)

        action_types = self._meta.action_types
        action_type = action_types.index(type(action)) + 1
        assert action_type >= 1

        action_input = action.input
        action_output: ActionOutput | None

        try:
            next_state, action_output = action.apply(last_state)
        except InvalidActionException as e:
            logger.debug(f"Invalid action: {e}")
            action_type = 0
            action_output = None
            next_state = last_state

        action_data = ActionData(
            type=action_type,
            input=action_input,
            output=action_output,
        )

        history = list(self._history).copy()
        history.append(action_data)
        history.append(next_state)

        last_history_idx = self._last_history_idx + len(history) - len(self._history)

        if self._max_history_state_size is not None:
            amount = 0
            for i, item in enumerate(history[::-1]):
                if isinstance(item, State):
                    amount += 1
                    if amount > self._max_history_state_size:
                        history = history[len(history)-(i+1):]
                        break

        return FullState(
            meta=self._meta,
            history=tuple(history),
            last_history_idx=last_history_idx,
            max_history_state_size=self._max_history_state_size,
        )

    def raw_data(self) -> np.ndarray[np.int_, np.dtype]:
        nodes = self.node_data_list()
        data = np.array([node.to_array() for node in nodes])
        return data

    def node_data_list(self) -> list[NodeItemData]:
        history_number = 0

        def create_node_history_type(history_type: int) -> NodeItemData:
            return NodeItemData.with_defaults(
                history_number=history_number,
                history_type=history_type,
                context=CONTEXT_HISTORY_TYPE,
                node_value=history_type,
            )

        node_history_type = create_node_history_type(HISTORY_TYPE_META)
        nodes_meta: list[NodeItemData] = [node_history_type]

        nodes_meta += self._node_data_list_meta_main(
            history_number=history_number)
        nodes_states: list[NodeItemData] = []
        nodes_actions: list[NodeItemData] = []
        last_history_idx = self._last_history_idx
        history_amount = len(self._history)

        for i, history_item in enumerate(self._history):
            history_number = last_history_idx - history_amount + i + 1
            if isinstance(history_item, State):
                node_history_type = create_node_history_type(HISTORY_TYPE_STATE)
                nodes_states.append(node_history_type)
                nodes_state = self._node_data_list_state(
                    history_number=history_number,
                    state=history_item)
                nodes_states += nodes_state
            elif isinstance(history_item, ActionData):
                node_history_type = create_node_history_type(HISTORY_TYPE_ACTION)
                nodes_states.append(node_history_type)
                nodes_action = self._node_data_list_action(
                    history_number=history_number,
                    action_data=history_item)
                nodes_actions += nodes_action

        nodes = nodes_meta + nodes_states + nodes_actions

        return nodes

    def _node_data_list_meta_main(self, history_number: int) -> list[NodeItemData]:
        meta = self._meta
        nodes: list[NodeItemData] = []

        for action_info in meta.action_types_info:
            nodes.append(NodeItemData.with_defaults(
                history_number=history_number,
                history_type=HISTORY_TYPE_META,
                context=CONTEXT_META_MAIN,
                subcontext=SUBCONTEXT_META_ACTION_TYPE,
                group_idx=action_info.type_idx,
                group_context=GROUP_CONTEXT_ACTION_TYPE,
                group_subcontext=GROUP_SUBCONTEXT_ACTION_TYPE_MAIN,
                node_value=len(action_info.arg_types),
            ))

            for j, arg_type in enumerate(action_info.arg_types):
                nodes.append(NodeItemData.with_defaults(
                    history_number=history_number,
                    history_type=HISTORY_TYPE_META,
                    context=CONTEXT_META_MAIN,
                    subcontext=SUBCONTEXT_META_ACTION_TYPE,
                    group_idx=action_info.type_idx,
                    group_context=GROUP_CONTEXT_ACTION_TYPE,
                    group_subcontext=GROUP_SUBCONTEXT_ACTION_TYPE_ARG,
                    item_idx=j+1,
                    node_value=arg_type,
                ))

        return nodes

    def _node_data_list_state(self, history_number: int, state: State) -> list[NodeItemData]:
        history_expr_id = 1
        function_info: FunctionInfo | None = None

        definitions_nodes: list[NodeItemData] = []
        for i, (definition, function_info) in enumerate(state.definitions):
            definitions_nodes.append(NodeItemData.with_defaults(
                history_number=history_number,
                history_type=HISTORY_TYPE_STATE,
                context=CONTEXT_STATE_DEFINITION,
                subcontext=UNDEFINED_OR_EMPTY_FIELD,
                item_idx=i+1,
                item_context=ITEM_CONTEXT_SYMBOL_IDX,
                node_value=definition.value,
                history_expr_id=history_expr_id,
            ))
            iter_nodes, history_expr_id = self._expr_tree_data_list(
                history_number=history_number,
                history_type=HISTORY_TYPE_STATE,
                context=CONTEXT_STATE_DEFINITION,
                item_idx=i+1,
                history_expr_id=history_expr_id,
                function_info=function_info,
            )
            definitions_nodes += iter_nodes

        partial_definitions_nodes: list[NodeItemData] = []
        for i, function_info in enumerate(state.partial_definitions):
            iter_nodes, history_expr_id = self._expr_tree_data_list(
                history_number=history_number,
                history_type=HISTORY_TYPE_STATE,
                context=CONTEXT_STATE_PARTIAL_DEFINITION,
                item_idx=i+1,
                history_expr_id=history_expr_id,
                function_info=function_info,
            )
            partial_definitions_nodes += iter_nodes

        arg_nodes, history_expr_id = self._context_arg_groups(
            history_number=history_number,
            history_type=HISTORY_TYPE_STATE,
            context=CONTEXT_STATE_ARG_GROUP,
            subcontext=UNDEFINED_OR_EMPTY_FIELD,
            groups=state.arg_groups,
            history_expr_id=history_expr_id,
        )

        nodes: list[NodeItemData] = (
            definitions_nodes +
            partial_definitions_nodes +
            arg_nodes)

        return nodes

    def _node_data_list_action(
        self,
        history_number: int,
        action_data: ActionData,
    ) -> list[NodeItemData]:
        action_input = action_data.input
        action_output = action_data.output
        history_expr_id = 1

        action_type_node = NodeItemData.with_defaults(
            history_number=history_number,
            history_type=HISTORY_TYPE_ACTION,
            context=CONTEXT_ACTION_TYPE,
            node_value=action_data.type,
        )

        action_input_nodes: list[NodeItemData] = [NodeItemData.with_defaults(
            history_number=history_number,
            history_type=HISTORY_TYPE_ACTION,
            context=CONTEXT_ACTION_INPUT,
            subcontext=SUBCONTEXT_ACTION_INPUT_AMOUNT,
            node_value=len(action_input.args),
        )]

        for i, arg in enumerate(action_input.args):
            arg_node = NodeItemData.with_defaults(
                history_number=history_number,
                history_type=HISTORY_TYPE_ACTION,
                context=CONTEXT_ACTION_INPUT,
                subcontext=SUBCONTEXT_ACTION_INPUT_ARG,
                item_idx=i+1,
                node_type=arg.type,
                node_value=arg.value,
            )
            action_input_nodes.append(arg_node)

        action_output_nodes: list[NodeItemData] = []

        if action_output is not None:
            action_output_type = ACTION_OUTPUT_TYPES.index(type(action_output)) + 1
            assert action_output_type >= 1

            def create_node(subcontext: int, node_value: int):
                return NodeItemData.with_defaults(
                    history_number=history_number,
                    history_type=HISTORY_TYPE_ACTION,
                    context=CONTEXT_ACTION_OUTPUT,
                    subcontext=subcontext,
                    item_context=UNDEFINED_OR_EMPTY_FIELD,
                    node_type=action_output_type,
                    node_value=node_value,
                )

            def create_expr_tree(function_info: FunctionInfo, history_expr_id: int):
                output_expr_nodes, history_expr_id = self._expr_tree_data_list(
                    history_number=history_number,
                    history_type=HISTORY_TYPE_ACTION,
                    context=CONTEXT_ACTION_OUTPUT,
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_NODE_EXPR,
                    history_expr_id=history_expr_id,
                    function_info=function_info,
                )
                return output_expr_nodes, history_expr_id

            if isinstance(action_output, NewPartialDefinitionActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_PARTIAL_DEFINITION_IDX,
                    node_value=action_output.partial_definition_idx,
                ))
            elif isinstance(action_output, PartialActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_PARTIAL_DEFINITION_IDX,
                    node_value=action_output.partial_definition_idx,
                ))

                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_NODE_IDX,
                    node_value=action_output.node_idx,
                ))

                output_expr_nodes, history_expr_id = create_expr_tree(
                    function_info=action_output.new_function_info,
                    history_expr_id=history_expr_id,
                )
                action_output_nodes += output_expr_nodes

                if action_output.new_expr_arg_group is not None:
                    arg_group_nodes, history_expr_id = self._context_arg_group(
                        history_number=history_number,
                        history_type=HISTORY_TYPE_ACTION,
                        context=CONTEXT_ACTION_OUTPUT,
                        subcontext=SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP,
                        group_idx=UNDEFINED_OR_EMPTY_FIELD,
                        group=action_output.new_expr_arg_group,
                        history_expr_id=history_expr_id,
                    )
                    action_output_nodes += arg_group_nodes
            elif isinstance(action_output, RemovePartialDefinitionActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_PARTIAL_DEFINITION_IDX,
                    node_value=action_output.partial_definition_idx,
                ))
            elif isinstance(action_output, NewArgGroupActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_IDX,
                    node_value=action_output.arg_group_idx,
                ))

                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_PARAMS_AMOUNT,
                    node_value=action_output.params_amount,
                ))

                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_ARGS_AMOUNT,
                    node_value=action_output.args_amount,
                ))
            elif isinstance(action_output, ArgFromExprActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_IDX,
                    node_value=action_output.arg_group_idx,
                ))

                output_expr_nodes, history_expr_id = create_expr_tree(
                    function_info=action_output.new_function_info,
                    history_expr_id=history_expr_id,
                )
                action_output_nodes += output_expr_nodes
            elif isinstance(action_output, RemoveArgGroupActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_IDX,
                    node_value=action_output.arg_group_idx,
                ))
            elif isinstance(action_output, NewDefinitionFromPartialActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_DEFINITION_IDX,
                    node_value=action_output.definition_idx,
                ))

                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_PARTIAL_DEFINITION_IDX,
                    node_value=action_output.partial_definition_idx,
                ))
            elif isinstance(action_output, ReplaceByDefinitionActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_DEFINITION_IDX,
                    node_value=action_output.definition_idx,
                ))

                if action_output.expr_id is not None:
                    action_output_nodes.append(create_node(
                        subcontext=SUBCONTEXT_ACTION_OUTPUT_EXPR_ID,
                        node_value=action_output.expr_id,
                    ))
            elif isinstance(action_output, ExpandDefinitionActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_DEFINITION_IDX,
                    node_value=action_output.definition_idx,
                ))

                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_EXPR_ID,
                    node_value=action_output.expr_id,
                ))
            elif isinstance(action_output, ReformulationActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=SUBCONTEXT_ACTION_OUTPUT_EXPR_ID,
                    node_value=action_output.expr_id,
                ))

                output_expr_nodes, history_expr_id = create_expr_tree(
                    function_info=action_output.new_function_info,
                    history_expr_id=history_expr_id,
                )
                action_output_nodes += output_expr_nodes
            else:
                raise NotImplementedError(f"Action output not implemented: {type(action_output)}")


        nodes: list[NodeItemData] = [action_type_node] + action_input_nodes + action_output_nodes

        return nodes

    def _context_arg_groups(
        self,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int,
        groups: typing.Sequence[ParamsArgsGroup],
        history_expr_id: int,
    ) -> tuple[list[NodeItemData], int]:
        nodes: list[NodeItemData] = []
        next_history_expr_id = history_expr_id

        for i, group in enumerate(groups):
            iter_nodes, next_history_expr_id = self._context_arg_group(
                history_number=history_number,
                history_type=history_type,
                context=context,
                subcontext=subcontext,
                group_idx=i+1,
                group=group,
                history_expr_id=next_history_expr_id,
            )
            nodes += iter_nodes

        return nodes, next_history_expr_id


    def _context_arg_group(
        self,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int,
        group_idx: int,
        group: ParamsArgsGroup,
        history_expr_id: int,
    ) -> tuple[list[NodeItemData], int]:
        def create_node(group_subcontext: int, node_value: int):
            return NodeItemData.with_defaults(
                history_number=history_number,
                history_type=history_type,
                context=context,
                subcontext=subcontext,
                group_idx=group_idx,
                group_context=GROUP_CONTEXT_ARG_GROUP,
                group_subcontext=group_subcontext,
                item_idx=group_idx,
                item_context=UNDEFINED_OR_EMPTY_FIELD,
                node_type=UNDEFINED_OR_EMPTY_FIELD,
                node_value=node_value,
            )

        nodes: list[NodeItemData] = [create_node(
            group_subcontext=GROUP_SUBCONTEXT_ARG_GROUP_MAIN,
            node_value=group_idx,
        )]

        nodes += [create_node(
            group_subcontext=GROUP_SUBCONTEXT_ARG_GROUP_PARAMS_AMOUNT,
            node_value=len(group.outer_params),
        )]

        nodes += [create_node(
            group_subcontext=GROUP_SUBCONTEXT_ARG_GROUP_ARGS_AMOUNT,
            node_value=len(group.inner_args),
        )]

        for i, expr in enumerate(group.inner_args):
            iter_nodes, history_expr_id = self._expr_tree_data_list(
                history_number=history_number,
                history_type=history_type,
                context=context,
                subcontext=subcontext,
                group_idx=group_idx,
                group_context=GROUP_CONTEXT_ARG_GROUP,
                group_subcontext=GROUP_SUBCONTEXT_ARG_GROUP_EXPR,
                item_idx=i+1,
                history_expr_id=history_expr_id,
                function_info=(
                    FunctionInfo(expr, FunctionParams(*group.outer_params))
                    if expr is not None
                    else None),
                skip_params=True,
            )

            nodes += iter_nodes

        return nodes, history_expr_id

    def _expr_tree_data_list(
        self,
        history_number: int,
        history_type: int,
        context: int,
        history_expr_id: int,
        function_info: FunctionInfo | None,
        subcontext: int = UNDEFINED_OR_EMPTY_FIELD,
        group_idx: int = UNDEFINED_OR_EMPTY_FIELD,
        group_context: int = UNDEFINED_OR_EMPTY_FIELD,
        group_subcontext: int = UNDEFINED_OR_EMPTY_FIELD,
        item_idx: int = UNDEFINED_OR_EMPTY_FIELD,
        skip_params=False,
    ) -> tuple[list[NodeItemData], int]:
        type_data = NodeItemData.with_defaults(
            history_number=history_number,
            history_type=history_type,
            context=context,
            subcontext=subcontext,
            group_idx=group_idx,
            group_context=group_context,
            group_subcontext=group_subcontext,
            item_idx=item_idx,
            item_context=ITEM_CONTEXT_TYPE_IDX,
            expr=function_info.expr if function_info is not None else None,
            node_value=len(function_info.params) if function_info is not None else 0,
        )
        nodes: list[NodeItemData] = [type_data]

        if function_info is None:
            return nodes, history_expr_id

        if not skip_params:
            for param in function_info.params:
                param_data = NodeItemData.with_defaults(
                    history_number=history_number,
                    history_type=history_type,
                    context=context,
                    subcontext=subcontext,
                    group_idx=group_idx,
                    group_context=group_context,
                    group_subcontext=group_subcontext,
                    item_idx=item_idx,
                    item_context=ITEM_CONTEXT_PARAM_IDX,
                    node_value=param.value,
                )
                nodes.append(param_data)

        expr_nodes, _, history_expr_id = self._expr_subtree_data_list(
            history_number=history_number,
            history_type=history_type,
            context=context,
            subcontext=subcontext,
            group_idx=group_idx,
            group_context=group_context,
            group_subcontext=group_subcontext,
            item_idx=item_idx,
            item_context=ITEM_CONTEXT_EXPR_IDX,
            history_expr_id=history_expr_id,
            expr=function_info.expr,
        )
        nodes += expr_nodes

        return nodes, history_expr_id

    def _expr_subtree_data_list(
        self,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int,
        group_idx: int,
        group_context: int,
        group_subcontext: int,
        item_idx: int,
        item_context: int,
        history_expr_id: int,
        expr: BaseNode,
        parent_node_idx: int = 0,
        node_idx: int = 1,
    ) -> tuple[list[NodeItemData], int, int]:
        assert expr is not None

        node_data = self._expr_single_node_data(
            history_number=history_number,
            history_type=history_type,
            context=context,
            subcontext=subcontext,
            group_idx=group_idx,
            group_context=group_context,
            group_subcontext=group_subcontext,
            item_idx=item_idx,
            item_context=item_context,
            parent_node_idx=parent_node_idx,
            node_idx=node_idx,
            history_expr_id=history_expr_id,
            expr=expr,
        )
        nodes: list[NodeItemData] = [node_data]

        parent_node_idx = node_idx
        next_node_idx = node_idx + 1
        next_history_expr_id = history_expr_id + 1
        args = tuple() if isinstance(expr, ScopedIntegerNode) else expr.args

        for arg in args:
            inner_nodes, next_node_idx, next_history_expr_id = self._expr_subtree_data_list(
                history_number=history_number,
                history_type=history_type,
                context=context,
                subcontext=subcontext,
                group_idx=group_idx,
                group_context=group_context,
                group_subcontext=group_subcontext,
                item_idx=item_idx,
                item_context=item_context,
                parent_node_idx=node_idx,
                node_idx=next_node_idx,
                history_expr_id=next_history_expr_id,
                expr=arg,
            )
            nodes += inner_nodes

        return nodes, next_node_idx, next_history_expr_id

    def _expr_single_node_data(
        self,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int,
        group_idx: int,
        group_context: int,
        group_subcontext: int,
        item_idx: int,
        item_context: int,
        history_expr_id: int | None,
        expr: BaseNode | None,
        parent_node_idx: int = 0,
        node_idx: int = 1,
    ) -> NodeItemData:
        meta = self._meta

        if expr is not None:
            node_type_idxs = [i+1 for i, t in enumerate(meta.node_types) if isinstance(expr, t)]
            assert len(node_type_idxs) == 1
            composite_node = (
                False
                if isinstance(expr, ScopedIntegerNode)
                else int(len(expr.args) > 0))
            node_type_idx = node_type_idxs[0]
            node_value = (
                expr.value
                if isinstance(expr, ScopedIntegerNode)
                else meta.node_type_handler.get_value(expr))
        else:
            composite_node = UNDEFINED_OR_EMPTY_FIELD
            node_type_idx = UNDEFINED_OR_EMPTY_FIELD
            node_value = UNDEFINED_OR_EMPTY_FIELD

        result = NodeItemData(
            history_number=history_number,
            history_type=history_type,
            context=context,
            subcontext=subcontext,
            group_idx=group_idx,
            group_context=group_context,
            group_subcontext=group_subcontext,
            item_idx=item_idx,
            item_context=item_context,
            parent_node_idx=parent_node_idx,
            node_idx=node_idx,
            composite_node=composite_node,
            node_type=node_type_idx,
            node_value=node_value,
            history_expr_id=history_expr_id,
            expr=expr,
        )

        return result
