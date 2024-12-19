import typing
import numpy as np
from utils.logger import logger
from .state import State, BaseNode, FunctionDefinition, ExprInfo, ArgGroup
from .action import (
    Action,
    InvalidActionException,
    ActionOutput,
    NewPartialDefinitionActionOutput,
    NewArgGroupActionOutput,
    ArgFromExprActionOutput,
    NewDefinitionFromPartialActionOutput,
    NewDefinitionFromExprActionOutput,
    ReplaceByDefinitionActionOutput,
    ExpandDefinitionActionOutput,
    ReformulationActionOutput,
    PartialActionOutput)
from .meta_env import EnvMetaInfo, ActionData, StateHistoryItem

HISTORY_TYPE_META = 0
HISTORY_TYPE_STATE = 1
HISTORY_TYPE_ACTION = 2

META_MAIN_CONTEXT = 0
META_ACTION_TYPE_CONTEXT = 1
META_ACTION_ARG_CONTEXT = 2

STATE_DEFINITION_CONTEXT = 1
STATE_PARTIAL_DEFINITION_CONTEXT = 2
STATE_ARG_GROUP_CONTEXT = 3
STATE_ARG_EXPR_CONTEXT = 4
STATE_ASSUMPTION_CONTEXT = 5

ACTION_TYPE_CONTEXT = 1
ACTION_INPUT_CONTEXT = 2
ACTION_OUTPUT_CONTEXT = 3
ACTION_STATUS_CONTEXT = 4

ACTION_OUTPUT_SUBCONTEXT_PARTIAL_DEFINITION_IDX = 1
ACTION_OUTPUT_SUBCONTEXT_DEFINITION_IDX = 2
ACTION_OUTPUT_SUBCONTEXT_ARG_GROUP_IDX = 3
ACTION_OUTPUT_SUBCONTEXT_ARG_AMOUNT = 4
ACTION_OUTPUT_SUBCONTEXT_EXPR_ID = 5
ACTION_OUTPUT_SUBCONTEXT_NODE_EXPR = 6

ACTION_STATUS_SKIP_ID = 0
ACTION_STATUS_SUCCESS_ID = 1
ACTION_STATUS_FAIL_ID = 2

GENERAL_ITEM_CONTEXT_SYMBOL_IDX = 1
GENERAL_ITEM_CONTEXT_TYPE_IDX = 2
GENERAL_ITEM_CONTEXT_PARAM_IDX = 3
GENERAL_ITEM_CONTEXT_EXPR_IDX = 4

UNKNOWN_OR_EMPTY_FIELD = 0

action_output_types = [
    NewPartialDefinitionActionOutput,
    NewArgGroupActionOutput,
    ArgFromExprActionOutput,
    NewDefinitionFromPartialActionOutput,
    NewDefinitionFromExprActionOutput,
    ReplaceByDefinitionActionOutput,
    ExpandDefinitionActionOutput,
    ReformulationActionOutput,
    PartialActionOutput,
]

# context index (e.g: main expression, definition expressions, temporary arguments, assumptions)
# subcontext index (e.g: part of an action output, argument group of a argument item)
# item index (e.g: in a list, the index about which definition, which equality, which assumption)
# parent node index (0 for the root node of an expression)
# atomic node (whether the node is atomic (no args, no operation) or not)
# node type index (e.g: symbol/unknown, definition, integer, function/operator)
# node value (e.g: symbol index, definition index, integer value, function/operator index)

class NodeItemData:
    def __init__(
        self,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int,
        item_idx: int,
        item_context: int,
        atomic_node: int = 1,
        parent_node_idx: int = 0,
        node_idx: int = 1,
        node_type: int = UNKNOWN_OR_EMPTY_FIELD,
        node_value: int = UNKNOWN_OR_EMPTY_FIELD,
        expr: BaseNode | None = None,
        history_expr_id: int | None = None,
    ):
        if history_expr_id is not None:
            assert history_expr_id > 0, f"Invalid history expression id: {history_expr_id}"

        self._history_number = history_number
        self._history_type = history_type
        self._context = context
        self._subcontext = subcontext
        self._item = item_idx
        self._item_context = item_context
        self._parent_node_idx = parent_node_idx
        self._node_idx = node_idx
        self._atomic_node = atomic_node
        self._node_type = node_type
        self._node_value = node_value
        self._history_expr_id = history_expr_id
        self._expr = expr

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
    def item(self) -> int:
        return self._item

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
    def atomic_node(self) -> int:
        return self._atomic_node

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
            self._item,
            self._parent_node_idx,
            self._node_idx,
            self._atomic_node,
            self._node_type,
            self._node_value,
            self._history_expr_id or 0,
        ])

class FullState:
    def __init__(
        self,
        meta: EnvMetaInfo,
        history: tuple[StateHistoryItem, ...],
        max_history_size: int | None = None,
    ):
        assert len(meta.node_types) > 0, "No node types"
        assert len(meta.node_types) == len(set(meta.node_types)), "Duplicate node types"
        assert meta.node_type_handler is not None, "No node type handler specified"
        assert len(meta.action_types) > 0, "No action types"
        assert len(meta.action_types) == len(set(meta.action_types)), "Duplicate action types"
        self._meta = meta
        self._history = history
        self._max_history_size = max_history_size

    @classmethod
    def initial_state(
        cls,
        initial_definitions: typing.Sequence[ExprInfo],
    ) -> tuple[State, list[FunctionDefinition]]:
        definition_keys = [FunctionDefinition(i+1) for i in range(len(initial_definitions))]
        definitions = tuple(zip(definition_keys, initial_definitions))
        state = State(
            definitions=definitions,
            partial_definitions=tuple(),
            arg_groups=tuple(),
            assumptions=tuple())
        return state, definition_keys

    @property
    def last_state(self) -> State:
        for i in range(len(self._history) - 1, -1, -1):
            item = self._history[i]
            if isinstance(item, State):
                return item

        raise ValueError("No state found in history")

    def apply(self, action: Action) -> 'FullState':
        last_state = self._history[-1]
        assert isinstance(last_state, State)

        action_types = self._meta.action_types
        # action type index (0 is for no action)
        action_type = action_types.index(type(action)) + 1
        assert action_type >= 1, f"Action type not found: {type(action)}"

        action_input = action.input
        action_output: ActionOutput | None

        try:
            next_state = action.apply(last_state)
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

        if self._max_history_size is not None:
            history = history[-self._max_history_size:]

        return FullState(
            meta=self._meta,
            history=tuple(history),
            max_history_size=self._max_history_size,
        )

    def raw_data(self) -> np.ndarray[np.int_, np.dtype]:
        nodes = self.node_data_list()
        data = np.array([node.to_array() for node in nodes])
        return data

    def node_data_list(self) -> list[NodeItemData]:
        nodes_meta = self._node_data_list_meta(history_number=0)
        nodes_states: list[NodeItemData] = []
        nodes_actions: list[NodeItemData] = []

        for i, history_item in enumerate(self._history):
            if isinstance(history_item, State):
                nodes_state = self._node_data_list_state(
                    history_number=i+1,
                    state=history_item)
                nodes_states += nodes_state
            elif isinstance(history_item, ActionData):
                nodes_action = self._node_data_list_action(
                    history_number=i+1,
                    action_data=history_item)
                nodes_actions += nodes_action

        nodes = nodes_meta + nodes_states + nodes_actions

        return nodes

    def _node_data_list_meta(self, history_number: int) -> list[NodeItemData]:
        meta = self._meta
        nodes: list[NodeItemData] = []

        for action_info in meta.action_types_info:
            nodes.append(NodeItemData(
                history_number=history_number,
                history_type=HISTORY_TYPE_META,
                context=META_ACTION_TYPE_CONTEXT,
                subcontext=UNKNOWN_OR_EMPTY_FIELD,
                item_idx=action_info.type_idx+1,
                item_context=UNKNOWN_OR_EMPTY_FIELD,
                node_type=UNKNOWN_OR_EMPTY_FIELD,
                node_value=len(action_info.arg_types),
            ))

            for j, arg_type in enumerate(action_info.arg_types):
                nodes.append(NodeItemData(
                    history_number=history_number,
                    history_type=HISTORY_TYPE_META,
                    context=META_ACTION_ARG_CONTEXT,
                    subcontext=action_info.type_idx+1,
                    item_idx=j+1,
                    item_context=UNKNOWN_OR_EMPTY_FIELD,
                    node_type=UNKNOWN_OR_EMPTY_FIELD,
                    node_value=arg_type,
                ))

        return nodes

    def _node_data_list_state(self, history_number: int, state: State) -> list[NodeItemData]:
        history_expr_id = 1
        expr_info: ExprInfo | None = None

        definitions_nodes: list[NodeItemData] = []
        for i, (definition, expr_info) in enumerate(state.definitions):
            definitions_nodes.append(NodeItemData(
                history_number=history_number,
                history_type=HISTORY_TYPE_STATE,
                context=STATE_DEFINITION_CONTEXT,
                subcontext=UNKNOWN_OR_EMPTY_FIELD,
                item_idx=i+1,
                item_context=GENERAL_ITEM_CONTEXT_SYMBOL_IDX,
                node_type=UNKNOWN_OR_EMPTY_FIELD,
                node_value=definition.index,
            ))
            iter_nodes, history_expr_id = self._expr_tree_data_list(
                history_number=history_number,
                history_type=HISTORY_TYPE_STATE,
                context=STATE_DEFINITION_CONTEXT,
                subcontext=UNKNOWN_OR_EMPTY_FIELD,
                item_idx=i+1,
                history_expr_id=history_expr_id,
                expr_info=expr_info,
            )
            definitions_nodes += iter_nodes

        partial_definitions_nodes: list[NodeItemData] = []
        for i, expr_info in enumerate(state.partial_definitions):
            iter_nodes, history_expr_id = self._expr_tree_data_list(
                history_number=history_number,
                history_type=HISTORY_TYPE_STATE,
                context=STATE_PARTIAL_DEFINITION_CONTEXT,
                subcontext=UNKNOWN_OR_EMPTY_FIELD,
                item_idx=i+1,
                history_expr_id=history_expr_id,
                expr_info=expr_info,
            )
            partial_definitions_nodes += iter_nodes

        arg_nodes, history_expr_id = self._context_node_data_groups(
            history_number=history_number,
            history_type=HISTORY_TYPE_STATE,
            group_context=STATE_ARG_GROUP_CONTEXT,
            group_subcontext=UNKNOWN_OR_EMPTY_FIELD,
            expression_context=STATE_ARG_EXPR_CONTEXT,
            groups=list(state.arg_groups or []),
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

        action_type_node = NodeItemData(
            history_number=history_number,
            history_type=HISTORY_TYPE_ACTION,
            context=ACTION_TYPE_CONTEXT,
            subcontext=UNKNOWN_OR_EMPTY_FIELD,
            item_idx=UNKNOWN_OR_EMPTY_FIELD,
            item_context=UNKNOWN_OR_EMPTY_FIELD,
            node_type=action_data.type,
            node_value=UNKNOWN_OR_EMPTY_FIELD,
        )

        action_input_nodes: list[NodeItemData] = []

        for i, arg in enumerate(action_input.args):
            arg_node = NodeItemData(
                history_number=history_number,
                history_type=HISTORY_TYPE_ACTION,
                context=ACTION_INPUT_CONTEXT,
                subcontext=UNKNOWN_OR_EMPTY_FIELD,
                item_idx=i+1,
                item_context=UNKNOWN_OR_EMPTY_FIELD,
                node_type=arg.type,
                node_value=arg.value,
            )
            action_input_nodes.append(arg_node)

        action_output_nodes: list[NodeItemData] = []

        if action_output is not None:
            action_output_type = action_output_types.index(type(action_output)) + 1
            assert action_output_type >= 1, f"Action output type not found: {type(action_output)}"

            def create_node(subcontext: int, node_value: int):
                return NodeItemData(
                    history_number=history_number,
                    history_type=HISTORY_TYPE_ACTION,
                    context=ACTION_OUTPUT_CONTEXT,
                    subcontext=subcontext,
                    item_idx=UNKNOWN_OR_EMPTY_FIELD,
                    item_context=UNKNOWN_OR_EMPTY_FIELD,
                    node_type=action_output_type,
                    node_value=node_value,
                )

            def create_expr_tree(expr_info: ExprInfo, history_expr_id: int):
                output_expr_nodes, history_expr_id = self._expr_tree_data_list(
                    history_number=history_number,
                    history_type=HISTORY_TYPE_ACTION,
                    context=ACTION_OUTPUT_CONTEXT,
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_NODE_EXPR,
                    item_idx=UNKNOWN_OR_EMPTY_FIELD,
                    history_expr_id=history_expr_id,
                    expr_info=expr_info,
                )
                return output_expr_nodes, history_expr_id

            if isinstance(action_output, NewPartialDefinitionActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_PARTIAL_DEFINITION_IDX,
                    node_value=action_output.partial_definition_idx,
                ))
            elif isinstance(action_output, NewArgGroupActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_ARG_GROUP_IDX,
                    node_value=action_output.arg_group_idx,
                ))

                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_ARG_AMOUNT,
                    node_value=action_output.amount,
                ))
            elif isinstance(action_output, ArgFromExprActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_ARG_GROUP_IDX,
                    node_value=action_output.arg_group_idx,
                ))

                output_expr_nodes, history_expr_id = create_expr_tree(
                    expr_info=action_output.new_expr_info,
                    history_expr_id=history_expr_id,
                )
                action_output_nodes += output_expr_nodes
            elif isinstance(action_output, NewDefinitionFromPartialActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_DEFINITION_IDX,
                    node_value=action_output.definition_idx,
                ))

                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_PARTIAL_DEFINITION_IDX,
                    node_value=action_output.partial_definition_idx,
                ))
            elif isinstance(action_output, NewDefinitionFromExprActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_DEFINITION_IDX,
                    node_value=action_output.definition_idx,
                ))

                assert action_output.new_expr_info is not None

                output_expr_nodes, history_expr_id = create_expr_tree(
                    expr_info=action_output.new_expr_info,
                    history_expr_id=history_expr_id,
                )
                action_output_nodes += output_expr_nodes
            elif isinstance(action_output, ReplaceByDefinitionActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_DEFINITION_IDX,
                    node_value=action_output.definition_idx,
                ))

                if action_output.expr_id is not None:
                    action_output_nodes.append(create_node(
                        subcontext=ACTION_OUTPUT_SUBCONTEXT_EXPR_ID,
                        node_value=action_output.expr_id,
                    ))
            elif isinstance(action_output, ReformulationActionOutput):
                action_output_nodes.append(create_node(
                    subcontext=ACTION_OUTPUT_SUBCONTEXT_EXPR_ID,
                    node_value=action_output.expr_id,
                ))

                output_expr_nodes, history_expr_id = create_expr_tree(
                    expr_info=action_output.new_expr_info,
                    history_expr_id=history_expr_id,
                )
                action_output_nodes += output_expr_nodes
            else:
                raise NotImplementedError(f"Action output not implemented: {type(action_output)}")


        nodes: list[NodeItemData] = [action_type_node] + action_input_nodes + action_output_nodes

        return nodes

    def _context_node_data_groups(
        self,
        history_number: int,
        history_type: int,
        group_context: int,
        group_subcontext: int,
        expression_context: int,
        groups: list[ArgGroup],
        history_expr_id: int,
    ) -> tuple[list[NodeItemData], int]:
        nodes: list[NodeItemData] = []
        next_history_expr_id = history_expr_id

        for i, group in enumerate(groups):
            iter_nodes, next_history_expr_id = self._context_arg_group(
                history_number=history_number,
                history_type=history_type,
                group_context=group_context,
                group_subcontext=group_subcontext,
                group_item=i+1,
                expression_context=expression_context,
                group=group,
                history_expr_id=next_history_expr_id,
            )

            nodes += iter_nodes

        return nodes, next_history_expr_id


    def _context_arg_group(
        self,
        history_number: int,
        history_type: int,
        group_context: int,
        group_subcontext: int,
        group_item: int,
        expression_context: int,
        group: ArgGroup,
        history_expr_id: int,
    ) -> tuple[list[NodeItemData], int]:
        nodes: list[NodeItemData] = [NodeItemData(
            history_number=history_number,
            history_type=history_type,
            context=group_context,
            subcontext=group_subcontext,
            item_idx=group_item,
            item_context=UNKNOWN_OR_EMPTY_FIELD,
            node_type=UNKNOWN_OR_EMPTY_FIELD,
            node_value=group.amount,
        )]

        for i, expr in enumerate(group.expressions):
            iter_nodes, history_expr_id = self._expr_tree_data_list(
                history_number=history_number,
                history_type=history_type,
                context=expression_context,
                subcontext=group_item,
                item_idx=i+1,
                history_expr_id=history_expr_id,
                expr_info=ExprInfo(expr=expr, params=group.params) if expr is not None else None,
                skip_params=True,
            )

            nodes += iter_nodes

        return nodes, history_expr_id

    def _expr_tree_data_list(
        self,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int,
        item_idx: int,
        history_expr_id: int,
        expr_info: ExprInfo | None,
        skip_params=False,
    ) -> tuple[list[NodeItemData], int]:
        type_data = NodeItemData(
            history_number=history_number,
            history_type=history_type,
            context=context,
            subcontext=subcontext,
            item_idx=item_idx,
            item_context=GENERAL_ITEM_CONTEXT_TYPE_IDX,
            expr=expr_info.expr if expr_info is not None else None,
            node_value=len(expr_info.params) if expr_info is not None else 0,
        )
        nodes: list[NodeItemData] = [type_data]

        if expr_info is None:
            return nodes, history_expr_id

        if not skip_params:
            for param in expr_info.params:
                param_data = NodeItemData(
                    history_number=history_number,
                    history_type=history_type,
                    context=context,
                    subcontext=subcontext,
                    item_idx=item_idx,
                    item_context=GENERAL_ITEM_CONTEXT_PARAM_IDX,
                    node_value=param.index,
                )
                nodes.append(param_data)

        expr_nodes, _, history_expr_id = self._expr_subtree_data_list(
            history_number=history_number,
            history_type=history_type,
            context=context,
            subcontext=subcontext,
            item_idx=item_idx,
            item_context=GENERAL_ITEM_CONTEXT_EXPR_IDX,
            history_expr_id=history_expr_id,
            expr=expr_info.expr,
        )
        nodes += expr_nodes

        return nodes, history_expr_id

    def _expr_subtree_data_list(
        self,
        history_number: int,
        history_type: int,
        context: int,
        subcontext: int,
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
            item_idx=item_idx,
            item_context=item_context,
            parent_node_idx=parent_node_idx,
            node_idx=node_idx,
            history_expr_id=history_expr_id,
            expr=expr,
        )
        nodes: list[NodeItemData] = [node_data]

        next_node_idx = node_idx + 1
        next_history_expr_id = history_expr_id + 1

        for arg in expr.args:
            inner_nodes, next_node_idx, next_history_expr_id = self._expr_subtree_data_list(
                history_number=history_number,
                history_type=history_type,
                context=context,
                subcontext=subcontext,
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
        item_idx: int,
        item_context: int,
        history_expr_id: int | None,
        expr: BaseNode | None,
        parent_node_idx: int = 0,
        node_idx: int = 1,
    ) -> NodeItemData:
        meta = self._meta

        if expr is not None:
            # node_type_idx = 0 is for special node types (e.g: unknown, empty)
            node_type_idxs = [i+1 for i, t in enumerate(meta.node_types) if isinstance(expr, t)]
            assert len(node_type_idxs) == 1, f"Invalid node type: {type(expr)}"
            atomic_node = int(len(expr.args) == 0)
            node_type_idx = node_type_idxs[0]
            node_value = meta.node_type_handler.get_value(expr)
        else:
            atomic_node = 1
            node_type_idx = UNKNOWN_OR_EMPTY_FIELD
            node_value = UNKNOWN_OR_EMPTY_FIELD

        result = NodeItemData(
            history_number=history_number,
            history_type=history_type,
            context=context,
            subcontext=subcontext,
            item_idx=item_idx,
            item_context=item_context,
            parent_node_idx=parent_node_idx,
            node_idx=node_idx,
            atomic_node=atomic_node,
            node_type=node_type_idx,
            node_value=node_value,
            history_expr_id=history_expr_id,
            expr=expr,
        )

        return result
