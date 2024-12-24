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
                if index == 0 and node is not None:
                    return node, index
        return None, index

    def _replace(self, index: int, new_node: BaseNode) -> tuple[BaseNode | None, int]:
        if index == 0:
            return new_node, index
        index -= 1
        for i, arg in enumerate(self.args):
            if isinstance(arg, BaseNode):
                new_arg, index = arg._replace(index, new_node)
                if index == 0 and new_arg is not None:
                    return self.func(*[
                        (new_arg if i == j else old_arg)
                        for j, old_arg in enumerate(self.args)
                    ]), index
        return None, index

    def _ancestors(self, index: int) -> tuple[list[BaseNode], int]:
        if index == 0:
            return [self], index
        index -= 1
        for arg in self.args:
            if isinstance(arg, BaseNode):
                # pylint: disable=protected-access
                ancestors, index = arg._ancestors(index)
                if index == 0:
                    return [self] + ancestors, index
        return [], index

    def ancestors(self, index: int) -> tuple[BaseNode, ...]:
        ancestors, index = self._ancestors(index)
        return tuple(ancestors)

    def __getitem__(self, index: 'BaseNodeIndex') -> BaseNode | None:
        if isinstance(index, BaseNodeMainIndex):
            node, _ = self._inner_getitem(index.value)
            return node
        raise ValueError(f"Invalid index: {index}")

    def replace(self, index: 'BaseNodeIndex', new_node: BaseNode) -> BaseNode | None:
        if isinstance(index, BaseNodeMainIndex):
            new_self, _ = self._replace(index.value, new_node)
            return new_self
        raise ValueError(f"Invalid index: {index}")

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

    def as_function(self) -> 'FunctionInfo':
        params = self.find(Param)
        amount_params = max(param.value for param in params)
        new_params = [Param(i+1) for i in range(amount_params)]
        new_expr = self.subs({
            old_param: new_params[old_param.value - 1]
            for old_param in params
        })
        return FunctionInfo(
            new_expr,
            FunctionParams(*new_params))

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

class BaseNodeIndex(Integer):
    pass

class BaseNodeMainIndex(Integer):
    pass

class ScopedIntegerNode(Integer):
    def __eq__(self, other) -> bool:
        # do not use == because it will cause infinite recursion
        return self is other

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

class StrictGroup(InheritableNode, typing.Generic[T]):
    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

    def __init__(self, *args: T):
        item_type = self.item_type()
        assert all(isinstance(arg, item_type) for arg in args)
        super().__init__(*args)

    @property
    def args(self) -> tuple[T, ...]:
        return typing.cast(tuple[T, ...], self._args)

    @property
    def as_tuple(self) -> tuple[T, ...]:
        return self.args

    @classmethod
    def from_items(cls, items: typing.Sequence[T]) -> 'StrictGroup':
        return cls(*items)

class OptionalGroup(InheritableNode, typing.Generic[T]):
    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

    def __init__(self, *args: T | VoidNode):
        item_type = self.item_type()
        assert all(
            (isinstance(arg, item_type) or isinstance(arg, VoidNode))
            for arg in args
        )
        super().__init__(*args)

    @property
    def args(self) -> tuple[T | VoidNode, ...]:
        return typing.cast(tuple[T | VoidNode, ...], self._args)

    @property
    def as_tuple(self) -> tuple[T | None, ...]:
        args: tuple[T | VoidNode, ...] = typing.cast(tuple[T | VoidNode, ...], self.args)
        return tuple([
            (arg if not isinstance(arg, VoidNode) else None)
            for arg in args
        ])

    @classmethod
    def from_items(cls, items: typing.Sequence[T | None]) -> typing.Self:
        args: list[T | VoidNode] = [arg if arg is not None else VoidNode() for arg in items]
        return cls(*args)

class ParamsGroup(StrictGroup[Param]):
    @classmethod
    def item_type(cls):
        return Param

class ArgsGroup(OptionalGroup[BaseNode]):
    @classmethod
    def item_type(cls):
        return BaseNode

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
        return self.outer_params_group.as_tuple

    @property
    def inner_args(self) -> tuple[BaseNode | None, ...]:
        return self.inner_args_group.as_tuple

class TypeGroup(StrictGroup[TypeNode]):
    @classmethod
    def item_type(cls):
        return TypeNode

class IntTypeGroup(TypeGroup):
    @classmethod
    def item_type(cls):
        return IntTypeNode

    @classmethod
    def from_types(cls, *types: type[Integer]) -> 'IntTypeGroup':
        return cls(*[IntTypeNode(t) for t in types])

class BaseValueGroup(StrictGroup[T]):
    @classmethod
    def item_type(cls):
        raise NotImplementedError

    def validate(self, type_group: TypeGroup):
        assert isinstance(type_group, TypeGroup)
        assert len(self.args) == len(type_group.args)
        for i, arg in enumerate(self.args):
            t_arg = type_group.args[i]
            assert isinstance(arg, BaseNode)
            assert isinstance(t_arg, TypeNode)
            assert isinstance(arg, t_arg.type)

class ValueGroup(BaseValueGroup[BaseNode]):
    @classmethod
    def item_type(cls):
        return BaseNode

class IntValueGroup(BaseValueGroup[Integer]):
    @classmethod
    def item_type(cls):
        return Integer

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
        return arg_type_group.as_tuple

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
