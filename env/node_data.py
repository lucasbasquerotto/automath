import numpy as np
from env import core, state

class NodeData:

    def __init__(
        self,
        node: core.INode,
        node_types: tuple[type[core.INode], ...],
    ):
        self.node = node
        self.node_types = node_types

    def to_data_array(self) -> np.ndarray[np.int_, np.dtype]:
        root_node = self.node
        node_types = self.node_types

        size = len(root_node.as_node)
        result = np.zeros((size, 8), dtype=np.int_)
        pending_stack: list[tuple[int, int, int, int, core.INode]] = [(0, 0, 0, 0, root_node)]
        node_id = 0

        while pending_stack:
            current: tuple[int, int, int, int, core.INode] = pending_stack.pop()
            parent_id, arg_id, parent_scope_id, context_parent_node_id, node = current
            node_id += 1
            idx = node_id - 1
            node_type_id = node_types.index(type(node)) + 1
            next_context_node_id = (
                (context_parent_node_id + (node_id - parent_id))
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
                    pending_stack.append((
                        node_id,
                        inner_arg_id,
                        scope_id,
                        next_context_node_id,
                        arg,
                    ))

        return result
