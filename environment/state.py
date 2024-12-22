import typing
from utils.types import (
    BaseNode,
    InheritableNode,
    FunctionDefinition,
    ParamVar,
    FunctionInfo,
    FunctionParams,
    ParamsArgsGroup,
    ParamsGroup,
    ArgsGroup,
    ScopedNode,
    EmptyNode)

class ExprStatusInfo:
    def __init__(self, function_info: FunctionInfo, readonly: bool):
        self._function_info = function_info
        self._readonly = readonly

    @property
    def function_info(self) -> FunctionInfo:
        return self._function_info

    @property
    def readonly(self) -> bool:
        return self._readonly

class FunctionDefinitionNode(InheritableNode):
    def __init__(self, definition_key: FunctionDefinition, function_info: FunctionInfo):
        assert isinstance(definition_key, FunctionDefinition)
        assert isinstance(function_info, FunctionInfo)
        super().__init__(definition_key, function_info)

    @property
    def definition_key(self) -> FunctionDefinition:
        key = self.args[0]
        assert isinstance(key, FunctionDefinition)
        return key

    @property
    def function_info(self) -> FunctionInfo:
        f = self.args[1]
        assert isinstance(f, FunctionInfo)
        return f

class FunctionDefinitionGroup(InheritableNode):
    def __init__(self, *args: FunctionDefinitionNode):
        assert all(isinstance(arg, FunctionDefinitionNode) for arg in args)
        super().__init__(*args)

    @property
    def expanded(self) -> tuple[tuple[FunctionDefinition, FunctionInfo], ...]:
        args = typing.cast(tuple[FunctionDefinitionNode, ...], self._args)
        return tuple((arg.definition_key, arg.function_info) for arg in args)

    @classmethod
    def from_definitions(
        cls,
        definitions: tuple[tuple[FunctionDefinition, FunctionInfo], ...],
    ) -> 'FunctionDefinitionGroup':
        return cls(*[
            FunctionDefinitionNode(definition_key, function_info)
            for definition_key, function_info in definitions
        ])

class PartialDefinitionNode(InheritableNode):
    def __init__(self, function_info: FunctionInfo):
        assert isinstance(function_info, FunctionInfo)
        super().__init__(function_info)

    @property
    def function_info(self) -> FunctionInfo:
        f = self.args[0]
        assert isinstance(f, FunctionInfo)
        return f

class PartialDefinitionGroup(InheritableNode):
    def __init__(self, *args: PartialDefinitionNode | EmptyNode):
        assert all(
            (isinstance(arg, PartialDefinitionNode) or isinstance(arg, EmptyNode))
            for arg in args)
        super().__init__(*args)

    @property
    def expanded(self) -> tuple[FunctionInfo | None, ...]:
        return tuple(
            arg.function_info if isinstance(arg, PartialDefinitionNode) else None
            for arg in self.args)

    @classmethod
    def from_definitions(
        cls,
        definitions: tuple[FunctionInfo | None, ...],
    ) -> 'PartialDefinitionGroup':
        nodes: list[PartialDefinitionNode | EmptyNode] = [
            PartialDefinitionNode(definition)
            if definition is not None
            else EmptyNode()
            for definition in definitions
        ]
        return cls(*nodes)

class ParamsArgsOuterGroup(InheritableNode):
    def __init__(self, *args: ParamsArgsGroup):
        assert all(isinstance(arg, ParamsArgsGroup) for arg in args)
        super().__init__(*args)

    @property
    def expanded(self) -> tuple[ParamsArgsGroup, ...]:
        return typing.cast(tuple[ParamsArgsGroup, ...], self._args)

class State(InheritableNode):
    def __init__(
        self,
        definitions: FunctionDefinitionGroup,
        partial_definitions: PartialDefinitionGroup,
        arg_groups: ParamsArgsOuterGroup,
    ):
        for i, (d, _) in enumerate(definitions.expanded):
            assert d.value == i + 1
        super().__init__(definitions, partial_definitions, arg_groups)

    @property
    def definitions(self) -> tuple[tuple[FunctionDefinition, FunctionInfo], ...]:
        definitions = self.args[0]
        assert isinstance(definitions, FunctionDefinitionGroup)
        return definitions.expanded

    @property
    def partial_definitions(self) -> tuple[FunctionInfo | None, ...]:
        partial_definitions = self.args[1]
        assert isinstance(partial_definitions, PartialDefinitionGroup)
        return partial_definitions.expanded

    @property
    def arg_groups(self) -> tuple[ParamsArgsGroup, ...]:
        arg_groups = self.args[2]
        assert isinstance(arg_groups, ParamsArgsOuterGroup)
        return arg_groups.expanded

    @classmethod
    def index_to_expr(cls, root: BaseNode, index: int) -> BaseNode | None:
        expr, _, __ = cls._index_to_expr(root, index, parent=False)
        return expr

    @classmethod
    def from_raw(
        cls,
        definitions: tuple[tuple[FunctionDefinition, FunctionInfo], ...],
        partial_definitions: tuple[FunctionInfo | None, ...],
        arg_groups: tuple[ParamsArgsGroup, ...],
    ) -> 'State':
        return cls(
            FunctionDefinitionGroup.from_definitions(definitions),
            PartialDefinitionGroup.from_definitions(partial_definitions),
            ParamsArgsOuterGroup(*arg_groups))

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
            return FunctionInfo(new_expr, FunctionParams(*new_params)), index

        assert root_info is not None

        args = tuple() if isinstance(root_info.expr, ScopedNode) else root_info.expr.args
        args_list: list[BaseNode] = list(args)

        for i, arg in enumerate(args_list):
            # recursive call each node arg to traverse its subtree
            new_arg_info, index = cls._replace_expr_index(
                root_info=FunctionInfo(arg, FunctionParams(*root_info.params)),
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
            (None, FunctionInfo(expr, FunctionParams(*group.outer_params)))
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
                        function_info=FunctionInfo(
                            definition_key,
                            FunctionParams(*function_info.params)),
                        readonly=True)

            new_expr, index, _ = self._index_to_expr(
                root=function_info.expr,
                index=index,
                parent=False)
            assert index >= 0
            if index == 0:
                assert new_expr is not None
                return ExprStatusInfo(
                    function_info=FunctionInfo(
                        new_expr,
                        FunctionParams(*function_info.params)),
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
        return State.from_raw(
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
        arg_groups_list[arg_group_idx] = ParamsArgsGroup(
            ParamsGroup(*arg_group.outer_params),
            ArgsGroup.from_args(arg_list))
        return State.from_raw(
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
                return State.from_raw(
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
                    return State.from_raw(
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
                        root_info=FunctionInfo(
                            expr,
                            FunctionParams(*arg_group.outer_params)),
                        index=index,
                        new_function_info=new_function_info)
                    assert index >= 0
                    if index == 0:
                        assert new_root_info is not None
                        expressions[j] = new_root_info.expr
                        arg_groups_list[i] = ParamsArgsGroup(
                            ParamsGroup(*arg_group.outer_params),
                            ArgsGroup.from_args(expressions))
                        return State.from_raw(
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
