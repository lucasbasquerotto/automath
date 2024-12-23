from __future__ import annotations
import typing
import sympy

T = typing.TypeVar('T', bound='BaseNode')

class BaseNode:
    def __init__(self, *args: int | BaseNode | typing.Type[BaseNode]):
        assert all(
            (
                isinstance(arg, BaseNode)
                or isinstance(arg, int)
                or (isinstance(arg, type) and issubclass(arg, BaseNode))
            )
            for arg in args
        )

        if any(
            (
                isinstance(arg, int)
                or (isinstance(arg, type) and issubclass(arg, BaseNode))
            )
            for arg in args
        ):
            assert len(args) == 1

        self._args = args
        self._cached_length: int | None = None
        self._cached_hash: int | None = None

    @property
    def args(self) -> tuple[int | BaseNode | typing.Type[BaseNode], ...]:
        return self._args

    @property
    def func(self) -> typing.Type[typing.Self]:
        return type(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BaseNode):
            return False
        if not self.func == other.func:
            return False
        if not len(self.args) == len(other.args):
            return False
        for i, arg in enumerate(self.args):
            if arg != other.args[i]:
                return False
        return True

    def _inner_getitem(self, index: int) -> tuple[BaseNode | None, int]:
        if index == 0:
            return self, index
        index -= 1
        for arg in self.args:
            if isinstance(arg, BaseNode):
                # pylint: disable=protected-access
                node, index = arg._inner_getitem(index)
                if index == 0:
                    return node, index
        return None, index

    def __getitem__(self, index: int) -> BaseNode:
        node, index = self._inner_getitem(index)
        if node is None:
            raise IndexError(f"Index out of range: {index}")
        return node

    def __len__(self) -> int:
        if self._cached_length is not None:
            return self._cached_length
        length = 1 + sum(
            len(arg)
            for arg in self.args
            if isinstance(arg, BaseNode)
        )
        self._cached_length = length
        return length

    def __hash__(self) -> int:
        if self._cached_hash is not None:
            return self._cached_hash
        hash_value = hash((self.func, self.args))
        self._cached_hash = hash_value
        return hash_value

    def subs(self, mapping: dict[BaseNode, BaseNode]) -> BaseNode:
        return self.func(*[
            (
                (
                    mapping[arg]
                    if arg in mapping
                    else arg
                )
                if isinstance(arg, BaseNode)
                else arg
            )
            for arg in self.args
        ])

    def find(self, node_type: type[T]) -> set[T]:
        return {
            node
            for node in self.args
            if isinstance(node, node_type)
        }.union({
            node
            for parent_node in self.args
            if isinstance(parent_node, BaseNode)
            for node in parent_node.find(node_type)
        })

    def eval(self) -> sympy.Basic:
        name = self.func.__name__
        if len(self.args) == 0:
            return sympy.Symbol(name)

        def eval_arg(arg: int | BaseNode | typing.Type[BaseNode]):
            if isinstance(arg, BaseNode):
                return arg.eval()
            if isinstance(arg, int):
                return sympy.Integer(arg)
            if isinstance(arg, type) and issubclass(arg, BaseNode):
                return sympy.Symbol(arg.__name__)
            raise ValueError(f"Invalid argument: {arg}")

        fn = sympy.Function(name)
        args: list[sympy.Basic] = [eval_arg(arg) for arg in self.args]
        # pylint: disable=not-callable
        return fn(*args)

class VoidNode(BaseNode):
    def __init__(self):
        super().__init__()

class Integer(BaseNode):
    def __init__(self, value: int):
        super().__init__(value)

    @property
    def value(self) -> int:
        value = self.args[0]
        assert isinstance(value, int)
        return value

class ScopedIntegerNode(Integer):
    def __eq__(self, other) -> bool:
        return self == other

class Param(ScopedIntegerNode):
    @classmethod
    def prefix(cls):
        return 'p'

class TypeNode(BaseNode):
    def __init__(self, type: typing.Type[BaseNode]):
        super().__init__(type)

    @property
    def type(self) -> typing.Type[BaseNode]:
        t = self.args[0]
        assert isinstance(t, type) and issubclass(t, BaseNode)
        return t

class IntTypeNode(TypeNode):
    def __init__(self, type: typing.Type[Integer]):
        super().__init__(type)

    @property
    def type(self) -> typing.Type[Integer]:
        t = self.args[0]
        assert isinstance(t, type) and issubclass(t, Integer)
        return t

class InheritableNode(BaseNode):
    def __init__(self, *args: BaseNode):
        assert all(isinstance(arg, BaseNode) for arg in args)
        super().__init__(*args)

    @property
    def args(self) -> tuple[BaseNode, ...]:
        return typing.cast(tuple[BaseNode, ...], self._args)

class FunctionDefinition(ScopedIntegerNode):
    @classmethod
    def prefix(cls):
        return 'f'

class BooleanNode(InheritableNode):
    @property
    def value(self) -> bool | None:
        raise NotImplementedError

class MultiArgBooleanNode(BooleanNode):
    def __init__(self, *args: BaseNode):
        assert len(args) > 0
        assert all(isinstance(arg, sympy.Basic) for arg in args)
        super().__init__(*args)

    @property
    def value(self) -> bool | None:
        raise NotImplementedError

class FunctionParams(InheritableNode):
    def __init__(self, *params: Param):
        assert all(isinstance(param, Param) for param in params)
        super().__init__(*params)

    @property
    def params(self) -> tuple[Param, ...]:
        return typing.cast(tuple[Param, ...], self._args)

class FunctionInfo(InheritableNode):
    def __init__(self, expr: BaseNode, params: FunctionParams):
        assert isinstance(params, FunctionParams)
        assert isinstance(expr, BaseNode)
        super().__init__(params, expr)

    @property
    def expr(self) -> BaseNode:
        expr = self.args[0]
        assert isinstance(expr, BaseNode)
        return expr

    @property
    def params_group(self) -> FunctionParams:
        params = self.args[1]
        assert isinstance(params, FunctionParams)
        return params

    @property
    def params(self) -> tuple[Param, ...]:
        group = self.params_group
        return group.params

    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionInfo):
            return False

        my_params = self.params
        other_params = other.params

        return self.expr == other.expr.subs({
            other_params[i]: my_params[i]
            for i in range(min(len(other_params), len(my_params)))
        })

class ParamsGroup(InheritableNode):
    def __init__(self, *args: Param):
        assert all(isinstance(arg, Param) for arg in args)
        super().__init__(*args)

    @property
    def params(self) -> tuple[Param, ...]:
        return typing.cast(tuple[Param, ...], self._args)

class ArgsGroup(InheritableNode):
    def __init__(self, *args: BaseNode):
        assert all(isinstance(arg, sympy.Basic) for arg in args)
        super().__init__(*args)

    @property
    def arg_list(self) -> tuple[BaseNode, ...]:
        return typing.cast(tuple[BaseNode, ...], self._args)

    @classmethod
    def from_args(cls, args: typing.Sequence[BaseNode | None]) -> 'ArgsGroup':
        return ArgsGroup(*[arg if arg is not None else VoidNode() for arg in args])

class ParamsArgsGroup(InheritableNode):
    def __init__(
        self,
        outer_params: ParamsGroup,
        inner_args: ArgsGroup,
    ):
        super().__init__(outer_params, inner_args)

    @property
    def outer_params_group(self) -> ParamsGroup:
        outer_params = self._args[0]
        assert isinstance(outer_params, ParamsGroup)
        return outer_params

    @property
    def inner_args_group(self) -> ArgsGroup:
        inner_args = self._args[1]
        assert isinstance(inner_args, ArgsGroup)
        return inner_args

    @property
    def outer_params(self) -> tuple[Param, ...]:
        outer_params = self._args[0]
        assert isinstance(outer_params, ParamsGroup)
        return outer_params.params

    @property
    def inner_args(self) -> tuple[BaseNode | None, ...]:
        inner_args_group = self.inner_args_group
        arg_list = inner_args_group.arg_list
        return tuple([
            (a if not isinstance(a, VoidNode) else None)
            for a in arg_list])

class TypeGroup(InheritableNode):
    def __init__(self, *args: TypeNode):
        assert all(isinstance(arg, TypeNode) for arg in args)
        super().__init__(*args)

    @property
    def arg_types(self) -> tuple[TypeNode, ...]:
        return typing.cast(tuple[TypeNode, ...], self.args)

class IntTypeGroup(TypeGroup):
    def __init__(self, *args: IntTypeNode):
        assert all(isinstance(arg, IntTypeNode) for arg in args)
        super().__init__(*args)

    @property
    def arg_types(self) -> tuple[IntTypeNode, ...]:
        return typing.cast(tuple[IntTypeNode, ...], self.args)

    @classmethod
    def from_types(cls, *types: type[Integer]) -> 'IntTypeGroup':
        return cls(*[IntTypeNode(t) for t in types])

class ValueGroup(InheritableNode):
    def validate(self, type_group: TypeGroup):
        assert isinstance(type_group, TypeGroup)
        assert len(self.args) == len(type_group.args)
        for i, arg in enumerate(self.args):
            t_arg = type_group.args[i]
            assert isinstance(arg, BaseNode)
            assert isinstance(t_arg, TypeNode)
            assert isinstance(arg, t_arg.type)

class IntValueGroup(ValueGroup):
    def __init__(self, *args: Integer):
        assert all(isinstance(arg, Integer) for arg in args)
        super().__init__(*args)

    @property
    def args(self) -> tuple[Integer, ...]:
        return typing.cast(tuple[Integer, ...], self.args)

class GroupWithArgTypes(InheritableNode):
    def __init__(self, group_type: TypeNode, arg_type_group: TypeGroup):
        assert isinstance(group_type, TypeNode)
        assert isinstance(arg_type_group, TypeGroup)
        super().__init__(group_type, arg_type_group)

    @property
    def group_type(self) -> TypeNode:
        group_type = self.args[0]
        assert isinstance(group_type, TypeNode)
        return group_type

    @property
    def arg_types(self) -> tuple[TypeNode, ...]:
        arg_type_group = self.args[1]
        assert isinstance(arg_type_group, TypeGroup)
        return arg_type_group.arg_types

class GroupWithArgs(InheritableNode):
    def __init__(self, group_type: TypeNode, arg_group: ValueGroup):
        assert isinstance(group_type, TypeNode)
        assert isinstance(arg_group, ValueGroup)
        super().__init__(group_type, arg_group)

    @property
    def group_type(self) -> TypeNode:
        group_type = self.args[0]
        assert isinstance(group_type, TypeNode)
        return group_type

    @property
    def arg_group(self) -> ValueGroup:
        arg_group = self.args[1]
        assert isinstance(arg_group, ValueGroup)
        return arg_group

    @property
    def arg_values(self) -> tuple[BaseNode, ...]:
        return self.arg_group.args

BASIC_NODE_TYPES: tuple[type[BaseNode], ...] = (
    BaseNode,
    VoidNode,
    Integer,
    ScopedIntegerNode,
    Param,
    TypeNode,
    InheritableNode,
    FunctionDefinition,
    BooleanNode,
    MultiArgBooleanNode,
    FunctionParams,
    FunctionInfo,
    ParamsGroup,
    ArgsGroup,
    ParamsArgsGroup,
    TypeGroup,
    ValueGroup,
    GroupWithArgTypes,
    GroupWithArgs,
)
