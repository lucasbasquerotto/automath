from __future__ import annotations
import numpy as np
from env import core, state, meta_env, full_state

class MainNodeData:

    def __init__(
        self,
        node_type: core.TypeNode,
        hidden_index: int | None,
        args: MainNodeData | tuple[MainNodeData, ...] | None,
    ):
        self.node_type = node_type
        self.hidden_index = hidden_index
        self.args = args

_meta_hidden = state.StateMetaHiddenInfo.idx_meta_hidden
_history_amount_to_show = state.StateMetaHiddenInfo.idx_history_amount_to_show
_history_state_hidden = state.StateMetaHiddenInfo.idx_history_state_hidden
_history_meta_hidden = state.StateMetaHiddenInfo.idx_history_meta_hidden
_history_action_hidden = state.StateMetaHiddenInfo.idx_history_action_hidden
_history_action_output_hidden = state.StateMetaHiddenInfo.idx_history_action_output_hidden
_history_action_exception_hidden = state.StateMetaHiddenInfo.idx_history_action_exception_hidden

_main_root_node = MainNodeData(
    node_type=full_state.FullState.as_type(),
    hidden_index=None,
    args=(
        MainNodeData(
            node_type=meta_env.MetaInfo.as_type(),
            hidden_index=_meta_hidden,
            args=None,
        ),
        MainNodeData(
            node_type=full_state.HistoryNode.as_type(),
            hidden_index=None,
            args=((
                MainNodeData(
                    node_type=state.State.as_type(),
                    hidden_index=None,
                    args=None,
                ),
                MainNodeData(
                    node_type=meta_env.MetaData.as_type(),
                    hidden_index=None,
                    args=None,
                ),
                MainNodeData(
                    node_type=meta_env.Optional.as_type(),
                    hidden_index=None,
                    args=None,
                ),

            )),
        ),
        MainNodeData(
            node_type=full_state.HistoryGroupNode.as_type(),
            hidden_index=_history_amount_to_show,
            args=MainNodeData(
                node_type=full_state.HistoryNode.as_type(),
                hidden_index=None,
                args=((
                    MainNodeData(
                        node_type=state.State.as_type(),
                        hidden_index=_history_state_hidden,
                        args=None,
                    ),
                    MainNodeData(
                        node_type=meta_env.MetaData.as_type(),
                        hidden_index=_history_meta_hidden,
                        args=None,
                    ),
                    MainNodeData(
                        node_type=meta_env.Optional.as_type(),
                        hidden_index=None,
                        args=((
                            MainNodeData(
                                node_type=full_state.BaseActionData.as_type(),
                                hidden_index=None,
                                args=((
                                    MainNodeData(
                                        node_type=meta_env.Optional.as_type(),
                                        hidden_index=_history_action_hidden,
                                        args=((
                                            MainNodeData(
                                                node_type=full_state.IAction.as_type(),
                                                hidden_index=None,
                                                args=None,
                                            ),
                                        )),
                                    ),
                                    MainNodeData(
                                        node_type=meta_env.Optional.as_type(),
                                        hidden_index=_history_action_output_hidden,
                                        args=((
                                            MainNodeData(
                                                node_type=full_state.IActionOutput.as_type(),
                                                hidden_index=None,
                                                args=None,
                                            ),
                                        )),
                                    ),
                                    MainNodeData(
                                        node_type=meta_env.Optional.as_type(),
                                        hidden_index=_history_action_exception_hidden,
                                        args=((
                                            MainNodeData(
                                                node_type=full_state.IExceptionInfo.as_type(),
                                                hidden_index=None,
                                                args=None,
                                            ),
                                        )),
                                    ),
                                )),
                            ),
                        )),
                    ),
                )),
            ),
        ),
    ),
)

class NodeData:

    def __init__(
        self,
        node: core.INode,
        node_types: tuple[type[core.INode], ...],
    ):
        self.node = node
        self.node_types = node_types

    def to_data_array(self) -> np.ndarray[np.int_, np.dtype]:
        return self.to_data_array_with_specs(root_node_id=1)

    def to_data_array_with_specs(
        self,
        root_node_id: int,
        initial_parent_id: int = 0,
        initial_arg_id: int = 0,
        initial_scope_id: int = 0,
    ) -> np.ndarray[np.int_, np.dtype]:
        root_node = self.node
        node_types = self.node_types

        main_node = _main_root_node if isinstance(root_node, full_state.FullState) else None
        current_state = (
            root_node.current_state.apply().real(state.State)
            if isinstance(root_node, full_state.FullState)
            else None)
        state_meta = (
            current_state.meta_info.apply().real(state.StateMetaInfo)
            if current_state is not None
            else None)
        hidden_info = (
            state_meta.hidden_info.apply().real(state.StateMetaHiddenInfo)
            if state_meta is not None
            else None)

        size = len(root_node.as_node)
        result = np.zeros((size, 9), dtype=np.int_)
        pending_stack: list[
            tuple[int, int, int, int, core.INode, MainNodeData | None, bool]
        ] = [(
            initial_parent_id,
            initial_arg_id,
            initial_scope_id,
            0,
            root_node,
            main_node,
            False,
        )]
        offset = root_node_id - 1
        node_id = offset

        while pending_stack:
            current: tuple[
                int, int, int, int, core.INode, MainNodeData | None, bool
            ] = pending_stack.pop()
            (
                parent_id,
                arg_id,
                parent_scope_id,
                context_parent_node_id,
                node,
                main_node,
                force_hidden,
            ) = current

            hidden_index: int | None = None
            args_to_show: int | None = None
            hidden = force_hidden
            if main_node is not None and not hidden:
                assert isinstance(node, main_node.node_type.type)
                hidden_index = main_node.hidden_index
                if (
                    hidden_index is not None
                    and hidden_info is not None
                ):
                    if hidden_index == _history_amount_to_show:
                        hidden_value_opt = hidden_info.inner_arg(
                            hidden_index
                        ).apply().real(core.Optional[core.BaseInt])
                        if not hidden_value_opt.is_empty().as_bool:
                            args_to_show = hidden_value_opt.value_or_raise.as_int
                            if args_to_show == 0:
                                hidden = True
                    else:
                        hidden_value = hidden_info.inner_arg(
                            hidden_index
                        ).apply().real(core.BaseIntBoolean)
                        if hidden_value.as_bool:
                            hidden = True

            node_id += 1
            idx = node_id - 1 - offset
            node_type_id = node_types.index(type(node)) + 1
            next_context_node_id_offset = offset if parent_id == initial_parent_id else 0
            next_context_node_id = (
                (context_parent_node_id + (node_id - parent_id - next_context_node_id_offset))
                if context_parent_node_id > 0
                else (1 if isinstance(node, state.IContext) else 0)
            )
            context_node_id = (next_context_node_id - 1) if next_context_node_id >= 1 else 0
            assert node_type_id > 0
            scope_id = parent_scope_id

            if isinstance(node, core.IOpaqueScope):
                scope_id = 1
            elif isinstance(node, core.IScope):
                assert isinstance(node, core.IInnerScope)
                assert parent_scope_id > 0
                scope_id = parent_scope_id + 1

            result[idx][0] = node_id
            result[idx][1] = parent_id
            result[idx][2] = arg_id
            result[idx][3] = scope_id
            result[idx][4] = context_node_id
            result[idx][5] = node_type_id

            if isinstance(node, core.ISpecialValue):
                value_aux = node.node_value

                if isinstance(value_aux, core.IInt):
                    value = value_aux.as_int
                elif isinstance(value_aux, core.TypeNode):
                    value = node_types.index(value_aux.type) + 1
                else:
                    raise ValueError(f'Invalid value type: {type(value_aux)}')

                result[idx][6] = value
            else:
                args = node.as_node.args
                args_amount = len(args)
                result[idx][7] = args_amount
                for i in range(args_amount):
                    inner_arg_id = args_amount - i
                    arg = args[inner_arg_id - 1]
                    assert isinstance(arg, core.INode)
                    arg_main_node: MainNodeData | None = None
                    if main_node is not None and main_node.args is not None:
                        if isinstance(main_node.args, MainNodeData):
                            arg_main_node = main_node.args
                        elif isinstance(main_node.args, tuple):
                            arg_idx = inner_arg_id-1
                            arg_main_node = main_node.args[arg_idx]
                    arg_force_hidden = hidden or (
                        args_to_show is not None and inner_arg_id <= args_amount - args_to_show
                    )
                    pending_stack.append((
                        node_id,
                        inner_arg_id,
                        scope_id,
                        next_context_node_id,
                        arg,
                        arg_main_node,
                        arg_force_hidden,
                    ))

            result[idx][8] = 1 if force_hidden or hidden else 0

        # remove the hidden nodes
        result = result[result[:, -1] == 0][:, :-1]

        return result
