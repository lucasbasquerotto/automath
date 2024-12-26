from __future__ import annotations
import typing
import sympy

T = typing.TypeVar('T', bound='BaseNode')


###########################################################
######################## BASE NODE ########################
###########################################################

class BaseNode:
    @classmethod
    def arg_types(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(RestTypeGroup(UnknownTypeNode()))

    def __init__(self, *args: int | BaseNode | typing.Type[BaseNode] | None):
        assert all(
            (
                arg is None
                or isinstance(arg, BaseNode)
                or isinstance(arg, int)
                or (isinstance(arg, type) and issubclass(arg, BaseNode))
            )
            for arg in args
        )

        if any(
            (
                arg is None
                or isinstance(arg, int)
                or (isinstance(arg, type) and issubclass(arg, BaseNode))
            )
            for arg in args
        ):
            assert len(args) == 1

        self._args = args
        self._cached_length: int | None = None
        self._cached_hash: int | None = None

    @property
    def args(self) -> tuple[int | BaseNode | typing.Type[BaseNode] | None, ...]:
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

    def __getitem__(self, index: 'BaseNodeIndex') -> BaseNode | None:
        return index.from_item(self)

    def replace(self, index: 'BaseNodeIndex', new_node: BaseNode) -> BaseNode | None:
        return index.replace_target(self, new_node)

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

        def eval_arg(arg: int | BaseNode | typing.Type[BaseNode] | None):
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

    @classmethod
    def type_as_function(
        cls,
        *args: int | BaseNode | typing.Type[BaseNode] | None,
    ) -> 'FunctionInfo':
        return FunctionInfo(
            cls.arg_types(),
            cls(*args))

    def as_function(self) -> 'FunctionInfo':
        return FunctionInfo(
            self.arg_types(),
            self)

###########################################################
######################## INT NODE #########################
###########################################################

class Integer(BaseNode):
    @classmethod
    def arg_types(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def __init__(self, value: int):
        assert isinstance(value, int)
        super().__init__(value)

    @property
    def value(self) -> int:
        value = self.args[0]
        assert isinstance(value, int)
        return value

###########################################################
######################## TYPE NODE ########################
###########################################################

class TypeNode(BaseNode, typing.Generic[T]):
    @classmethod
    def arg_types(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def __init__(self, type: type[T]):
        assert isinstance(type, type) and issubclass(type, BaseNode)
        super().__init__(type)

    @property
    def type(self) -> type[T]:
        t = self.args[0]
        assert isinstance(t, type) and issubclass(t, BaseNode)
        return typing.cast(type[T], t)

###########################################################
######################## MAIN NODE ########################
###########################################################

class InheritableNode(BaseNode):
    def __init__(self, *args: BaseNode):
        assert all(isinstance(arg, BaseNode) for arg in args)
        BaseValueGroup(*args).validate(self.arg_types())
        super().__init__(*args)

    @property
    def args(self) -> tuple[BaseNode, ...]:
        return typing.cast(tuple[BaseNode, ...], self._args)

    def __getitem__(self, index: 'BaseNodeIndex') -> BaseNode | None:
        if isinstance(index, BaseNodeArgIndex):
            args = self.args
            if index.value > 0 and index.value <= len(args):
                return args[index.value - 1]
            return None
        return super().__getitem__(index)

    def replace(self, index: 'BaseNodeIndex', new_node: BaseNode) -> BaseNode | None:
        assert isinstance(index, BaseNodeIndex)
        assert isinstance(new_node, BaseNode)
        if isinstance(index, BaseNodeArgIndex):
            args = self.args
            if index.value >= 0 and index.value <= len(args):
                return self.func(*[
                    (new_node if i == index.value - 1 else arg)
                    for i, arg in enumerate(args)
                ])
            return None
        return super().replace(index, new_node)

###########################################################
###################### SPECIAL NODES ######################
###########################################################

class UniqueNode(InheritableNode):
    def __eq__(self, other) -> bool:
        # do not use == because it will cause infinite recursion
        return self is other

class VoidNode(InheritableNode):
    def __init__(self):
        super().__init__()

class UnknownTypeNode(TypeNode[BaseNode]):
    def __init__(self):
        super().__init__(self.__class__)

class UnknownValue(InheritableNode):
    def __init__(self):
        super().__init__()

class IntTypeNode(TypeNode[Integer]):
    pass

class UniqueIntegerNode(Integer):
    def __eq__(self, other) -> bool:
        # do not use == because it will cause infinite recursion
        return self is other

class Param(UniqueIntegerNode):
    pass

###########################################################
######################## NODE IDXS ########################
###########################################################

class BaseNodeIndex(InheritableNode):
    def from_item(self, node: BaseNode) -> BaseNode | None:
        raise NotImplementedError

    def replace_target(self, target_node: BaseNode, new_node: BaseNode) -> BaseNode | None:
        raise NotImplementedError

class BaseNodeIntIndex(BaseNodeIndex):
    def __init__(self, index: Integer):
        assert isinstance(index, Integer)
        super().__init__(index)

    @property
    def index(self) -> Integer:
        index = self.args[0]
        assert isinstance(index, Integer)
        return index

    @property
    def value(self) -> int:
        return self.index.value

    def from_item(self, node: BaseNode) -> BaseNode | None:
        raise NotImplementedError

    def replace_target(self, target_node: BaseNode, new_node: BaseNode) -> BaseNode | None:
        raise NotImplementedError

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(Integer(value))

class BaseNodeMainIndex(BaseNodeIntIndex):
    @classmethod
    def _inner_getitem(cls, node: BaseNode, index: int) -> tuple[BaseNode | None, int]:
        assert index > 0
        index -= 1
        if index == 0:
            return node, index
        for arg in node.args:
            if isinstance(arg, BaseNode):
                arg_item, index = cls._inner_getitem(arg, index)
                if index == 0:
                    return arg_item, index
        return None, index

    @classmethod
    def _replace(
        cls,
        target_node: BaseNode,
        new_node: BaseNode,
        index: int,
    ) -> tuple[BaseNode | None, int]:
        assert index > 0
        index -= 1
        if index == 0:
            return new_node, index
        for i, arg in enumerate(target_node.args):
            if isinstance(arg, BaseNode):
                new_arg, index = cls._replace(arg, new_node, index)
                if index == 0:
                    if new_arg is None:
                        return None, index
                    return target_node.func(*[
                        (new_arg if i == j else old_arg)
                        for j, old_arg in enumerate(target_node.args)
                    ]), index
        return None, index

    @classmethod
    def _ancestors(cls, node: BaseNode, index: int) -> tuple[list[BaseNode], int]:
        assert index > 0
        index -= 1
        if index == 0:
            return [node], index
        for arg in node.args:
            if isinstance(arg, BaseNode):
                ancestors, index = cls._ancestors(node, index)
                if index == 0:
                    if len(ancestors) == 0:
                        return [], index
                    return [node] + ancestors, index
        return [], index

    def from_item(self, node: BaseNode) -> BaseNode | None:
        item, _ = self._inner_getitem(node, self.value)
        return item

    def replace_target(self, target_node: BaseNode, new_node: BaseNode) -> BaseNode | None:
        new_target, _ = self._replace(target_node, new_node, self.value)
        return new_target

    def ancestors(self, node: BaseNode) -> tuple[BaseNode, ...]:
        ancestors, _ = self._ancestors(node, self.value)
        return tuple(ancestors)

class BaseNodeArgIndex(BaseNodeIntIndex):
    def from_item(self, node: BaseNode) -> BaseNode | None:
        index = self.value
        if not isinstance(node, InheritableNode):
            return None
        args = node.args
        if index > 0 and index <= len(args):
            return args[index - 1]
        return None

    def replace_target(self, target_node: T, new_node: BaseNode) -> T | None:
        if not isinstance(target_node, InheritableNode):
            return None
        index = self.value
        args = target_node.args
        if index > 0 and index <= len(args):
            new_target = target_node.func(*[
                (new_node if i == index - 1 else arg)
                for i, arg in enumerate(args)
            ])
            assert isinstance(new_target, type(target_node))
            return typing.cast(T, new_target)
        return None

###########################################################
###################### BOOLEAN NODE #######################
###########################################################

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

###########################################################
####################### GROUP NODE ########################
###########################################################

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
    def from_items(cls, items: typing.Sequence[T]) -> typing.Self:
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

class CountableTypeGroup(StrictGroup[TypeNode[T]], typing.Generic[T]):
    @classmethod
    def item_type(cls) -> type[TypeNode[T]]:
        return TypeNode

class RestTypeGroup(InheritableNode, typing.Generic[T]):
    def __init__(self, type_node: TypeNode[T]):
        assert isinstance(type_node, TypeNode)
        super().__init__(type_node)

    @property
    def type(self) -> TypeNode[T]:
        type_node = self.args[0]
        assert isinstance(type_node, TypeNode)
        return type_node

class ExtendedTypeGroup(InheritableNode, typing.Generic[T]):
    def __init__(self, group: CountableTypeGroup[T] | RestTypeGroup[T]):
        assert isinstance(group, CountableTypeGroup) or isinstance(group, RestTypeGroup)
        super().__init__(group)

    @property
    def group(self) -> CountableTypeGroup[T] | RestTypeGroup[T]:
        group = self.args[1]
        assert isinstance(group, CountableTypeGroup) or isinstance(group, RestTypeGroup)
        return group

class BaseValueGroup(StrictGroup[T]):
    @classmethod
    def item_type(cls):
        raise NotImplementedError

    def validate(self, types: ExtendedTypeGroup):
        group = types.group

        if isinstance(group, CountableTypeGroup):
            assert len(self.args) == len(group.args)
            for i, arg in enumerate(self.args):
                t_arg = group.args[i]
                assert isinstance(arg, BaseNode)
                assert isinstance(t_arg, TypeNode)
                if t_arg.type is not UnknownTypeNode:
                    assert isinstance(arg, t_arg.type)
        elif isinstance(group, RestTypeGroup):
            t_arg = group.type
            if t_arg.type is not UnknownTypeNode:
                for arg in self.args:
                    assert isinstance(arg, BaseNode)
                    assert isinstance(t_arg, TypeNode)
                    assert isinstance(arg, t_arg.type)
        else:
            raise ValueError(f"Invalid group type: {group}")

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

###########################################################
###################### FUNCTION NODE ######################
###########################################################

class ParamInfo(InheritableNode):
    def __init__(self, param: Param, type_node: TypeNode[BaseNode]):
        assert isinstance(param, Param)
        assert isinstance(type_node, TypeNode)
        super().__init__(param, type_node)

    @property
    def param(self) -> Param:
        param = self.args[0]
        assert isinstance(param, Param)
        return param

    @property
    def type(self) -> type[BaseNode]:
        type_node = self.args[1]
        assert isinstance(type_node, TypeNode)
        return type_node.type

    @property
    def value(self) -> int:
        return self.param.value

class FunctionInfo(InheritableNode):
    def __init__(self, param_types: ExtendedTypeGroup, expr: BaseNode):
        assert isinstance(param_types, ExtendedTypeGroup)
        assert isinstance(expr, BaseNode)
        super().__init__(param_types, expr)

    @property
    def param_types(self) -> ExtendedTypeGroup:
        param_types = self.args[1]
        assert isinstance(param_types, ExtendedTypeGroup)
        return param_types

    @property
    def expr(self) -> BaseNode:
        expr = self.args[0]
        assert isinstance(expr, BaseNode)
        return expr

    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionInfo):
            return False
        return self.expr == other.expr

###########################################################
####################### BASIC NODES #######################
###########################################################

BASIC_NODE_TYPES: tuple[type[BaseNode], ...] = (
    BaseNode,
    VoidNode,
    Integer,
    UniqueIntegerNode,
    Param,
    TypeNode,
    InheritableNode,
    BooleanNode,
    MultiArgBooleanNode,
    FunctionInfo,
    ParamsGroup,
    ArgsGroup,
    ParamsArgsGroup,
    CountableTypeGroup,
    ValueGroup,
    GroupWithArgs,
)
