import typing
from utils.types import BaseNode, FunctionDefinition, ParamVar, Assumption, ExprInfo, ArgGroup

class ExprWithArgs:
    def __init__(self, expr_info: ExprInfo, args: tuple[BaseNode, ...]):
        assert len(expr_info.params) == len(args), \
            f"Invalid amount of arguments: {len(expr_info.params)} != {len(args)}"

        args_dict: dict[ParamVar, BaseNode] = {
            p: args[i]
            for i, p in enumerate(expr_info.params)
        }

        self._expr_info = expr_info
        self._args = args
        self._args_dict = args_dict

    @property
    def expr(self) -> BaseNode:
        return self._expr_info.expr

    @property
    def args(self) -> tuple[BaseNode, ...]:
        return self._args

    @property
    def apply(self) -> BaseNode:
        return self.expr.subs(self._args_dict)

class State:
    def __init__(
        self,
        definitions: tuple[tuple[FunctionDefinition, ExprInfo], ...],
        partial_definitions: tuple[tuple[FunctionDefinition, ExprInfo | None], ...],
        arg_groups: tuple[ArgGroup, ...],
        assumptions: tuple[Assumption, ...],
    ):
        self._definitions = definitions
        self._partial_definitions = partial_definitions
        self._arg_groups = arg_groups
        self._assumptions = assumptions

    @property
    def definitions(self) -> tuple[tuple[FunctionDefinition, ExprInfo], ...]:
        return self._definitions

    @property
    def partial_definitions(self) -> tuple[tuple[FunctionDefinition, ExprInfo | None], ...]:
        return self._partial_definitions

    @property
    def arg_groups(self) -> tuple[ArgGroup, ...]:
        return self._arg_groups

    @property
    def assumptions(self) -> tuple[Assumption, ...]:
        return self._assumptions

    @classmethod
    def index_to_expr(cls, root: BaseNode, index: int) -> BaseNode | None:
        expr, _, __ = cls._index_to_expr(root, index, parent=False)
        return expr

    @classmethod
    def _index_to_expr(
        cls,
        root: BaseNode,
        index: int,
        parent: bool,
        parent_expr: BaseNode | None = None,
        child_index: int | None = None,
    ) -> tuple[BaseNode | None, int, int | None]:
        assert root is not None
        assert index > 0, f"Invalid index for root node: {index}"
        assert isinstance(index, int), f"Invalid index type for root node: {type(index)} ({index})"
        index -= 1
        expr: BaseNode | None = root

        if index > 0:
            parent_expr = root
            for i, arg in enumerate(root.args):
                # recursive call each node arg to traverse its subtree
                expr, index, child_index = cls._index_to_expr(
                    root=arg,
                    index=index,
                    parent=parent,
                    parent_expr=parent_expr,
                    child_index=i)
                assert index >= 0, f"Invalid index for node: {index}"
                # it will end when index = 0 (it's the actual node, if any)
                # otherwise, it will go to the next arg
                if index == 0:
                    break

        return (parent_expr if parent else expr) if (index == 0) else None, index, child_index

    @classmethod
    def _replace_expr_index(
        cls,
        root_info: ExprInfo | None,
        index: int,
        new_expr_info: ExprInfo,
    ) -> tuple[ExprInfo | None, int]:
        assert index > 0, f"Invalid index for root node: {index}"
        assert isinstance(index, int), f"Invalid index type for root node: {type(index)} ({index})"
        index -= 1

        if index == 0:
            outer_params = root_info.params if root_info is not None else tuple()
            assert len(new_expr_info.params) <= len(outer_params), \
                f"Invalid amount of parameters: {len(new_expr_info.params)} > {len(outer_params)}"
            new_expr = new_expr_info.expr.subs({
                p: outer_params[i]
                for i, p in enumerate(new_expr_info.params)
            })
            new_params: tuple[ParamVar, ...] = (
                root_info.params
                if root_info is not None
                else tuple())
            return ExprInfo(expr=new_expr, params=new_params), index

        assert root_info is not None, f"Invalid root node for index {index}"

        args_list: list[BaseNode] = list(root_info.expr.args)

        for i, arg in enumerate(args_list):
            # recursive call each node arg to traverse its subtree
            new_arg_info, index = cls._replace_expr_index(
                root_info=ExprInfo(expr=arg, params=root_info.params),
                index=index,
                new_expr_info=new_expr_info)
            assert index >= 0, f"Invalid index for node: {index}"
            # it will end when index = 0 (it's the actual node, if any)
            # otherwise, it will go to the next arg
            # it returns the actual arg subtree with the new node
            if index == 0:
                assert new_arg_info is not None, "Invalid new arg node"
                args_list[i] = new_arg_info.expr
                return root_info.expr.func(*args_list), index

        return None, index

    def _get_expr_info(
        self,
        index: int,
        parent: bool = False,
    ) -> tuple[ExprInfo | None, int | None]:
        initial_index = index
        definitions: list[ExprInfo] = [
            expr_info
            for _, expr_info in self.definitions or []]
        partial_definitions: list[ExprInfo] = [
            expr_info
            for _, expr_info in self.partial_definitions or []
            if expr_info is not None]
        arg_exprs: list[ExprInfo] = [
            ExprInfo(expr=expr, params=group.params)
            for group in self.arg_groups or []
            for expr in group.expressions
            if expr is not None]

        for expr_info in definitions + partial_definitions + arg_exprs:
            new_expr, index, child_index = self._index_to_expr(
                root=expr_info.expr,
                index=index,
                parent=parent)
            assert index >= 0, f"Invalid index for node: {initial_index}"
            if index == 0:
                assert new_expr is not None, "Invalid node"
                return ExprInfo(expr=new_expr, params=expr_info.params), child_index

        return None, None

    def get_expr(self, index: int) -> ExprInfo | None:
        new_expr, _ = self._get_expr_info(index=index)
        return new_expr

    def get_expr_full_info(
        self,
        root: BaseNode | None,
        node_idx: int,
    ) -> tuple[BaseNode | None, BaseNode | None, int | None]:
        if node_idx == 1:
            return root, None, None

        assert root is not None, "Invalid root"

        index = node_idx

        parent_node, index, child_index = self._index_to_expr(
            root=root, index=index, parent=True)

        assert index >= 0, f"Invalid index for node: {node_idx}"
        if index == 0:
            assert parent_node is not None, "Invalid parent node"
            assert child_index is not None, "Invalid child index"
            node = parent_node.args[child_index]
            return node, parent_node, child_index

        return None, None, None

    def change_partial_definition(
        self,
        partial_definition_idx: int,
        node_idx: int,
        new_expr_info: ExprInfo,
    ) -> 'State':
        partial_definitions_list = list(self.partial_definitions or [])
        assert partial_definition_idx > 0, \
            f"Invalid partial definition: {partial_definition_idx}"
        assert partial_definition_idx <= len(partial_definitions_list), \
            f"Invalid partial definition: {partial_definition_idx}"
        key, root_info = partial_definitions_list[partial_definition_idx - 1]
        new_root, index = self._replace_expr_index(
            root_info=root_info,
            index=node_idx,
            new_expr_info=new_expr_info)
        assert index == 0, f"Node {node_idx} not found " \
            + f"in partial definition: {partial_definition_idx}"
        assert new_root is not None, "Invalid new root node"
        partial_definitions_list[partial_definition_idx - 1] = (key, new_root)
        return State(
            definitions=self.definitions,
            partial_definitions=tuple(partial_definitions_list),
            arg_groups=self.arg_groups,
            assumptions=self.assumptions)

    def change_arg(
        self,
        arg_group_idx: int,
        arg_idx: int,
        new_expr_info: ExprInfo,
    ) -> 'State':
        arg_groups_list = list(self.arg_groups or [])
        assert arg_group_idx > 0, f"Invalid arg group: {arg_group_idx}"
        assert arg_group_idx <= len(arg_groups_list), f"Invalid arg group: {arg_group_idx}"
        arg_group = arg_groups_list[arg_group_idx - 1]
        expr_info_list = list(arg_group.expressions)
        assert arg_group.amount == len(arg_group.params), \
            f"Invalid amount of params: {arg_group.amount} != {len(arg_group.params)}"
        assert arg_group.amount == len(expr_info_list), \
            f"Invalid amount of expressions: {arg_group.amount} != {len(expr_info_list)}"
        assert arg_idx > 0, f"Invalid arg: {arg_idx}"
        assert arg_idx <= len(arg_group.expressions), f"Invalid arg: {arg_idx}"
        assert isinstance(new_expr_info.expr, BaseNode), "Invalid new node"
        assert len(new_expr_info.params) <= len(arg_group.params), \
            f"Invalid amount of parameters: {len(new_expr_info.params)} > {len(arg_group.params)}"
        new_expr = new_expr_info.expr.subs({
            p: arg_group.params[i]
            for i, p in enumerate(new_expr_info.params)
        })
        expr_info_list[arg_idx - 1] = new_expr
        arg_groups_list[arg_group_idx] = ArgGroup(
            amount=arg_group.amount,
            params=arg_group.params,
            expressions=tuple(expr_info_list))
        return State(
            definitions=self.definitions,
            partial_definitions=self.partial_definitions,
            arg_groups=tuple(arg_groups_list),
            assumptions=self.assumptions)

    def apply_new_expr(self, expr_id: int, new_expr_info: ExprInfo) -> 'State':
        assert expr_id is not None, "Empty expression id"
        assert expr_id > 0, f"Invalid expression id: {expr_id}"

        index = expr_id

        definitions_list = list(self.definitions or [])
        for i, (key, expr_info) in enumerate(definitions_list):
            new_root_info, index = self._replace_expr_index(
                root_info=expr_info,
                index=index,
                new_expr_info=new_expr_info)
            assert index >= 0, f"Invalid index for node: {index}"
            if index == 0:
                assert new_root_info is not None, "Invalid new root node (definition)"
                definitions_list[i] = (key, new_root_info)
                return State(
                    definitions=tuple(definitions_list),
                    partial_definitions=self.partial_definitions,
                    arg_groups=self.arg_groups,
                    assumptions=self.assumptions)

        partial_definitions_list = list(self.partial_definitions or [])
        for i, (key, expr_info_p) in enumerate(partial_definitions_list):
            if expr_info_p is not None:
                expr_info = expr_info_p
                new_root_info, index = self._replace_expr_index(
                    root_info=expr_info,
                    index=index,
                    new_expr_info=new_expr_info)
                assert index >= 0, f"Invalid index for node: {index}"
                if index == 0:
                    assert new_root_info is not None, "Invalid new root node (partial definition)"
                    partial_definitions_list[i] = (key, new_root_info)
                    return State(
                        definitions=self.definitions,
                        partial_definitions=tuple(partial_definitions_list),
                        arg_groups=self.arg_groups,
                        assumptions=self.assumptions)

        arg_groups_list = list(self.arg_groups or [])
        for i, arg_group in enumerate(arg_groups_list):
            expressions = list(arg_group.expressions)
            for j, expr_p in enumerate(expressions):
                if expr_p is not None:
                    expr = expr_p
                    new_root_info, index = self._replace_expr_index(
                        root_info=ExprInfo(expr=expr, params=arg_group.params),
                        index=index,
                        new_expr_info=new_expr_info)
                    assert index >= 0, f"Invalid index for node: {index}"
                    if index == 0:
                        assert new_root_info is not None, "Invalid new root node (arg)"
                        expressions[j] = new_root_info.expr
                        arg_groups_list[i] = ArgGroup(
                            amount=arg_group.amount,
                            params=arg_group.params,
                            expressions=tuple(expressions))
                        return State(
                            definitions=self.definitions,
                            partial_definitions=self.partial_definitions,
                            arg_groups=tuple(arg_groups_list),
                            assumptions=self.assumptions)

        raise ValueError(f"Invalid expr_id: {expr_id}")

    @classmethod
    def same_definitions(
        cls,
        my_definitions: typing.Sequence[ExprInfo | None],
        other_definitions: typing.Sequence[ExprInfo | None],
    ) -> bool:
        if len(my_definitions) != len(other_definitions):
            return False

        return all(
            (expr_info == other_expr_info == None)
            or
            (
                expr_info is not None
                and
                other_expr_info is not None
                and
                expr_info.expr == other_expr_info.expr.subs({
                    other_expr_info.params[i]: expr_info.params[i]
                    for i in range(min(len(expr_info.params), len(other_expr_info.params)))
                })
            )
            for expr_info, other_expr_info in zip(my_definitions, other_definitions)
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, State):
            return False

        same_definitions = self.same_definitions(
            my_definitions=[
                expr_info
                for _, expr_info in list(self.definitions or [])],
            other_definitions=[
                expr_info
                for _, expr_info in list(other.definitions or [])])

        if not same_definitions:
            return False

        same_partial_definitions = self.same_definitions(
            my_definitions=[
                expr_info
                for _, expr_info in list(self.partial_definitions or [])],
            other_definitions=[
                expr_info
                for _, expr_info in list(other.partial_definitions or [])])

        if not same_partial_definitions:
            return False

        if len(self.arg_groups) != len(other.arg_groups):
            return False

        for my_arg_group, other_arg_group in zip(self.arg_groups, other.arg_groups):
            if my_arg_group.amount != other_arg_group.amount:
                return False
            if len(my_arg_group.params) != len(other_arg_group.params):
                return False
            if len(my_arg_group.expressions) != len(other_arg_group.expressions):
                return False
            for my_expr, other_expr in zip(
                my_arg_group.expressions,
                other_arg_group.expressions,
            ):
                if my_expr is None and other_expr is None:
                    continue
                if my_expr is None or other_expr is None:
                    return False

                min_params = min(len(my_arg_group.params), len(other_arg_group.params))

                if my_expr != other_expr.subs(
                    {
                        other_arg_group.params[i]: my_arg_group.params[i]
                        for i in range(min_params)
                    }
                ):
                    return False

        return True
