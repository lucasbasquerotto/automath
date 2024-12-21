import typing
from utils.types import BaseNode, FunctionDefinition, ParamVar, FunctionInfo, ArgGroup, ScopedNode

class ExprStatusInfo(FunctionInfo):
    def __init__(self, expr: BaseNode, params: tuple[ParamVar, ...], readonly: bool):
        super().__init__(expr=expr, params=params)
        self._readonly = readonly

    @property
    def readonly(self) -> bool:
        return self._readonly

class State:
    def __init__(
        self,
        definitions: tuple[tuple[FunctionDefinition, FunctionInfo], ...],
        partial_definitions: tuple[FunctionInfo | None, ...],
        arg_groups: tuple[ArgGroup, ...],
    ):
        for i, (d, _) in enumerate(definitions):
            assert d.value == i + 1
        self._definitions = definitions
        self._partial_definitions = partial_definitions
        self._arg_groups = arg_groups

    @property
    def definitions(self) -> tuple[tuple[FunctionDefinition, FunctionInfo], ...]:
        return self._definitions

    @property
    def partial_definitions(self) -> tuple[FunctionInfo | None, ...]:
        return self._partial_definitions

    @property
    def arg_groups(self) -> tuple[ArgGroup, ...]:
        return self._arg_groups

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
        assert index > 0
        assert isinstance(index, int)
        index -= 1
        expr: BaseNode | None = root

        if index > 0:
            parent_expr = root
            args = tuple() if isinstance(expr, ScopedNode) else root.args
            for i, arg in enumerate(args):
                # recursive call each node arg to traverse its subtree
                expr, index, child_index = cls._index_to_expr(
                    root=arg,
                    index=index,
                    parent=parent,
                    parent_expr=parent_expr,
                    child_index=i)
                assert index >= 0
                # it will end when index = 0 (it's the actual node, if any)
                # otherwise, it will go to the next arg
                if index == 0:
                    break

        return (parent_expr if parent else expr) if (index == 0) else None, index, child_index

    @classmethod
    def _replace_expr_index(
        cls,
        root_info: FunctionInfo | None,
        index: int,
        new_function_info: FunctionInfo,
    ) -> tuple[FunctionInfo | None, int]:
        assert index > 0
        assert isinstance(index, int)
        index -= 1

        if index == 0:
            outer_params = root_info.params if root_info is not None else tuple()
            assert len(new_function_info.params) <= len(outer_params)
            new_expr = new_function_info.expr.subs({
                p: outer_params[i]
                for i, p in enumerate(new_function_info.params)
            })
            new_params: tuple[ParamVar, ...] = (
                root_info.params
                if root_info is not None
                else tuple())
            return FunctionInfo(expr=new_expr, params=new_params), index

        assert root_info is not None

        args = tuple() if isinstance(root_info.expr, ScopedNode) else root_info.expr.args
        args_list: list[BaseNode] = list(args)

        for i, arg in enumerate(args_list):
            # recursive call each node arg to traverse its subtree
            new_arg_info, index = cls._replace_expr_index(
                root_info=FunctionInfo(expr=arg, params=root_info.params),
                index=index,
                new_function_info=new_function_info)
            assert index >= 0
            # it will end when index = 0 (it's the actual node, if any)
            # otherwise, it will go to the next arg
            # it returns the actual arg subtree with the new node
            if index == 0:
                assert new_arg_info is not None
                args_list[i] = new_arg_info.expr
                return root_info.expr.func(*args_list), index

        return None, index

    @classmethod
    def get_partial_definition_node(
        cls,
        root_info: FunctionInfo | None,
        index: int,
    ) -> BaseNode | None:
        if root_info is None:
            return None
        node, _, __ = cls._index_to_expr(root_info.expr, index, parent=False)
        return node

    def get_expr(self, expr_id: int) -> ExprStatusInfo | None:
        index = expr_id
        assert index > 0

        expr_list: list[tuple[FunctionDefinition | None, FunctionInfo]] = []
        expr_list += list(self.definitions)
        expr_list += [
            (None, function_info)
            for function_info in self.partial_definitions
            if function_info is not None]
        expr_list += [
            (None, FunctionInfo(expr=expr, params=group.outer_params))
            for group in self.arg_groups
            for expr in group.inner_args
            if expr is not None]

        for definition_key, function_info in expr_list:
            if definition_key is not None:
                index -= 1
                assert index >= 0
                if index == 0:
                    assert function_info is not None
                    return ExprStatusInfo(
                        expr=definition_key,
                        params=function_info.params,
                        readonly=True)

            new_expr, index, _ = self._index_to_expr(
                root=function_info.expr,
                index=index,
                parent=False)
            assert index >= 0
            if index == 0:
                assert new_expr is not None
                return ExprStatusInfo(
                    expr=new_expr,
                    params=function_info.params,
                    readonly=False)

        return None

    def get_expr_full_info(
        self,
        root: BaseNode | None,
        node_idx: int,
    ) -> tuple[BaseNode | None, BaseNode | None, int | None]:
        if node_idx == 1:
            return root, None, None

        assert root is not None

        index = node_idx

        parent_node, index, child_index = self._index_to_expr(
            root=root, index=index, parent=True)

        assert index >= 0
        if index == 0:
            assert parent_node is not None
            assert child_index is not None
            node = parent_node.args[child_index]
            return node, parent_node, child_index

        return None, None, None

    def change_partial_definition(
        self,
        partial_definition_idx: int,
        node_idx: int,
        new_function_info: FunctionInfo,
    ) -> 'State':
        partial_definitions_list = list(self.partial_definitions or [])
        assert partial_definition_idx > 0
        assert partial_definition_idx <= len(partial_definitions_list)
        root_info = partial_definitions_list[partial_definition_idx - 1]
        new_root, index = self._replace_expr_index(
            root_info=root_info,
            index=node_idx,
            new_function_info=new_function_info)
        assert index == 0
        assert new_root is not None
        partial_definitions_list[partial_definition_idx - 1] = new_root
        return State(
            definitions=self.definitions,
            partial_definitions=tuple(partial_definitions_list),
            arg_groups=self.arg_groups)

    def change_arg(
        self,
        arg_group_idx: int,
        arg_idx: int,
        new_function_info: FunctionInfo,
    ) -> 'State':
        arg_groups_list = list(self.arg_groups or [])
        assert arg_group_idx > 0
        assert arg_group_idx <= len(arg_groups_list)
        arg_group = arg_groups_list[arg_group_idx - 1]
        arg_list = list(arg_group.inner_args)
        assert arg_idx > 0
        assert arg_idx <= len(arg_list)
        assert isinstance(new_function_info.expr, BaseNode)
        assert len(new_function_info.params) <= len(arg_group.outer_params)
        new_expr = new_function_info.expr.subs({
            p: arg_group.outer_params[i]
            for i, p in enumerate(new_function_info.params)
        })
        arg_list[arg_idx - 1] = new_expr
        arg_groups_list[arg_group_idx] = ArgGroup(
            outer_params=arg_group.outer_params,
            inner_args=tuple(arg_list))
        return State(
            definitions=self.definitions,
            partial_definitions=self.partial_definitions,
            arg_groups=tuple(arg_groups_list))

    def apply_new_expr(self, expr_id: int, new_function_info: FunctionInfo) -> 'State':
        assert expr_id is not None
        assert expr_id > 0

        index = expr_id

        definitions_list = list(self.definitions or [])
        for i, (key, function_info) in enumerate(definitions_list):
            index -= 1
            assert index > 0

            new_root_info, index = self._replace_expr_index(
                root_info=function_info,
                index=index,
                new_function_info=new_function_info)
            assert index >= 0
            if index == 0:
                assert new_root_info is not None
                definitions_list[i] = (key, new_root_info)
                return State(
                    definitions=tuple(definitions_list),
                    partial_definitions=self.partial_definitions,
                    arg_groups=self.arg_groups)

        partial_definitions_list = list(self.partial_definitions or [])
        for i, function_info_p in enumerate(partial_definitions_list):
            if function_info_p is not None:
                function_info = function_info_p
                new_root_info, index = self._replace_expr_index(
                    root_info=function_info,
                    index=index,
                    new_function_info=new_function_info)
                assert index >= 0
                if index == 0:
                    assert new_root_info is not None
                    partial_definitions_list[i] = new_root_info
                    return State(
                        definitions=self.definitions,
                        partial_definitions=tuple(partial_definitions_list),
                        arg_groups=self.arg_groups)

        arg_groups_list = list(self.arg_groups or [])
        for i, arg_group in enumerate(arg_groups_list):
            expressions = list(arg_group.inner_args)
            for j, expr_p in enumerate(expressions):
                if expr_p is not None:
                    expr = expr_p
                    new_root_info, index = self._replace_expr_index(
                        root_info=FunctionInfo(expr=expr, params=arg_group.outer_params),
                        index=index,
                        new_function_info=new_function_info)
                    assert index >= 0
                    if index == 0:
                        assert new_root_info is not None
                        expressions[j] = new_root_info.expr
                        arg_groups_list[i] = ArgGroup(
                            outer_params=arg_group.outer_params,
                            inner_args=tuple(expressions))
                        return State(
                            definitions=self.definitions,
                            partial_definitions=self.partial_definitions,
                            arg_groups=tuple(arg_groups_list))

        raise ValueError(f"Invalid expr_id: {expr_id}")

    @classmethod
    def same_definitions(
        cls,
        my_definitions: typing.Sequence[FunctionInfo | None],
        other_definitions: typing.Sequence[FunctionInfo | None],
    ) -> bool:
        if len(my_definitions) != len(other_definitions):
            return False

        return all(
            (function_info == other_function_info == None)
            or
            (
                function_info is not None
                and
                other_function_info is not None
                and
                function_info.expr == other_function_info.expr.subs({
                    other_function_info.params[i]: function_info.params[i]
                    for i in range(min(len(function_info.params), len(other_function_info.params)))
                })
            )
            for function_info, other_function_info in zip(my_definitions, other_definitions)
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, State):
            return False

        same_definitions = self.same_definitions(
            my_definitions=[
                function_info
                for _, function_info in list(self.definitions or [])],
            other_definitions=[
                function_info
                for _, function_info in list(other.definitions or [])])

        if not same_definitions:
            return False

        same_partial_definitions = self.same_definitions(
            my_definitions=[
                function_info
                for function_info in list(self.partial_definitions or [])],
            other_definitions=[
                function_info
                for function_info in list(other.partial_definitions or [])])

        if not same_partial_definitions:
            return False

        if len(self.arg_groups) != len(other.arg_groups):
            return False

        for my_arg_group, other_arg_group in zip(self.arg_groups, other.arg_groups):
            if len(my_arg_group.outer_params) != len(other_arg_group.outer_params):
                return False
            if len(my_arg_group.inner_args) != len(other_arg_group.inner_args):
                return False
            for my_expr, other_expr in zip(
                my_arg_group.inner_args,
                other_arg_group.inner_args,
            ):
                if my_expr is None and other_expr is None:
                    continue
                if my_expr is None or other_expr is None:
                    return False

                min_params = min(
                    len(my_arg_group.outer_params),
                    len(other_arg_group.outer_params))

                if my_expr != other_expr.subs(
                    {
                        other_arg_group.outer_params[i]: my_arg_group.outer_params[i]
                        for i in range(min_params)
                    }
                ):
                    return False

        return True
