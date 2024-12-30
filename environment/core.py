from __future__ import annotations
import typing
import sympy

T = typing.TypeVar('T', bound='BaseNode')


###########################################################
######################## BASE NODE ########################
###########################################################

class BaseNode:
    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
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

    def replace_at(self, index: 'BaseNodeIndex', new_node: BaseNode) -> BaseNode | None:
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

    def find_until(self, node_type: type[T], until_type: type[BaseNode] | None) -> set[T]:
        return {
            node
            for node in self.args
            if isinstance(node, node_type)
        }.union({
            node
            for parent_node in self.args
            if (
                isinstance(parent_node, BaseNode)
                and
                (until_type is None or not isinstance(parent_node, until_type))
            )
            for node in parent_node.find_until(node_type, until_type)
        })

    def find(self, node_type: type[T]) -> set[T]:
        return self.find_until(node_type, None)

    def replace(self, before: BaseNode, after: BaseNode) -> BaseNode:
        if self == before:
            return after
        return self.func(*[
            (
                arg.replace(before, after)
                if isinstance(arg, BaseNode)
                else arg
            )
            for arg in self.args
        ])

    def subs(self, mapping: dict[BaseNode, BaseNode]) -> BaseNode:
        node = self
        for key, value in mapping.items():
            node = node.replace(key, value)
        return node

    def has(self, node: BaseNode) -> bool:
        return node in self.find(node.__class__)

    def has_until(self, node: BaseNode, until_type: type[BaseNode] | None) -> bool:
        return node in self.find_until(node.__class__, until_type)

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

###########################################################
######################## INT NODE #########################
###########################################################

class Integer(BaseNode):
    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
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
    def arg_type_group(cls) -> ExtendedTypeGroup:
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
        self.arg_type_group().validate(BaseValueGroup(*args))
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

    def replace_at(self, index: 'BaseNodeIndex', new_node: BaseNode) -> BaseNode | None:
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
        return super().replace_at(index, new_node)

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

###########################################################
########################## SCOPE ##########################
###########################################################

class ScopeId(Integer):
    pass

class TemporaryScopeId(ScopeId):
    pass

class Scope(InheritableNode):
    @property
    def id(self) -> ScopeId:
        raise NotImplementedError

    def has_dependency(self) -> bool:
        raise NotImplementedError

    def replace_id(self, new_id: ScopeId) -> typing.Self:
        node = self.replace(self.id, new_id)
        assert isinstance(node, self.__class__)
        return node

class SimpleScope(Scope, typing.Generic[T]):
    def __init__(self, id: ScopeId, child: T):
        assert isinstance(id, Integer)
        assert isinstance(child, BaseNode)
        super().__init__(id, child)

    @property
    def id(self) -> ScopeId:
        id = self.args[0]
        assert isinstance(id, ScopeId)
        return id

    @property
    def child(self) -> T:
        child = self.args[1]
        return typing.cast(T, child)

    def has_dependency(self) -> bool:
        return self.child.has_until(self.id, OpaqueScope)

class OpaqueScope(SimpleScope[T], typing.Generic[T]):

    @classmethod
    def _normalize(cls, node: BaseNode, next_id: int) -> BaseNode:
        if isinstance(node, OpaqueScope):
            return node.normalize()
        new_args = [
            cls._normalize(child, next_id+1)
            if isinstance(child, BaseNode)
            else child
            for child in node.args
        ]
        node = node.func(*new_args)
        if isinstance(node, Scope):
            node = node.replace_id(TemporaryScopeId(next_id))
        return node

    def normalize(self) -> typing.Self:
        next_id = 1
        child_args = [
            self._normalize(child, next_id+1)
            if isinstance(child, BaseNode)
            else child
            for child in self.child.args
        ]
        child = self.child.func(*child_args)
        node = self.func(TemporaryScopeId(next_id), child)
        tmp_ids = node.find(TemporaryScopeId)
        for tmp_id in tmp_ids:
            node_aux = node.replace(tmp_id, ScopeId(tmp_id.value))
            assert isinstance(node_aux, self.__class__)
            node = node_aux
        return node

class Param(InheritableNode, typing.Generic[T]):
    def __init__(self, parent_scope: ScopeId, index: Integer, type_node: TypeNode[T]):
        assert isinstance(parent_scope, ScopeId)
        assert isinstance(index, Integer)
        assert isinstance(type_node, TypeNode)
        super().__init__(parent_scope, index, type_node)

    @property
    def parent_scope_id(self) -> ScopeId:
        parent_scope = self.args[0]
        assert isinstance(parent_scope, ScopeId)
        return parent_scope

    @property
    def index(self) -> Integer:
        index = self.args[1]
        assert isinstance(index, Integer)
        return index

    @property
    def type_node(self) -> TypeNode[T]:
        type_node = self.args[2]
        assert isinstance(type_node, TypeNode)
        return type_node

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

class BaseValueGroup(StrictGroup[T], typing.Generic[T]):
    @classmethod
    def item_type(cls):
        raise NotImplementedError

class ExtendedTypeGroup(InheritableNode, typing.Generic[T]):
    def __init__(self, group: CountableTypeGroup[T] | RestTypeGroup[T]):
        assert isinstance(group, CountableTypeGroup) or isinstance(group, RestTypeGroup)
        super().__init__(group)

    @property
    def group(self) -> CountableTypeGroup[T] | RestTypeGroup[T]:
        group = self.args[1]
        assert isinstance(group, CountableTypeGroup) or isinstance(group, RestTypeGroup)
        return group

    def validate(self, values: BaseValueGroup):
        group = self.group
        args = values.args

        if isinstance(group, CountableTypeGroup):
            assert len(args) == len(group.args)
            for i, arg in enumerate(args):
                t_arg = group.args[i]
                assert isinstance(arg, BaseNode)
                assert isinstance(t_arg, TypeNode)
                if t_arg.type is not UnknownTypeNode:
                    assert isinstance(arg, t_arg.type)
        elif isinstance(group, RestTypeGroup):
            t_arg = group.type
            if t_arg.type is not UnknownTypeNode:
                for arg in args:
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

class Function(InheritableNode, typing.Generic[T]):
    def __init__(self, param_type_group: ExtendedTypeGroup, scope: SimpleScope[T]):
        assert isinstance(param_type_group, ExtendedTypeGroup)
        assert isinstance(scope, SimpleScope)
        super().__init__(param_type_group, scope)

    @property
    def param_type_group(self) -> ExtendedTypeGroup:
        param_type_group = self.args[1]
        assert isinstance(param_type_group, ExtendedTypeGroup)
        return param_type_group

    @property
    def scope(self) -> SimpleScope[T]:
        scope = self.args[0]
        assert isinstance(scope, SimpleScope)
        return scope

    @property
    def scope_id(self) -> ScopeId:
        return self.scope.id

    @property
    def expr(self) -> T:
        return self.scope.child

    def owned_params(self) -> typing.Sequence[Param]:
        params = [
            param
            for param in self.expr.find(Param)
            if param.parent_scope_id == self.scope_id]
        params = sorted(params, key=lambda param: param.index.value)
        return params

    def with_args(self, *args: BaseNode) -> BaseNode:
        type_group = self.param_type_group
        group = ValueGroup(*args)
        type_group.validate(group)
        params = self.owned_params()
        scope = self.scope
        for param in params:
            index = param.index.value
            assert index > 0
            assert index <= len(args)
            arg = args[index-1]
            scope_aux = scope.replace(param, arg)
            assert isinstance(scope_aux, SimpleScope)
            scope = scope_aux
        assert not scope.has_dependency()
        return scope.child

    @classmethod
    def equal(cls, a: BaseNode, b: BaseNode) -> bool:
        if a.func != b.func:
            return False
        if len(a.args) != len(b.args):
            return False
        if isinstance(a, Scope) and isinstance(b, Scope):
            b = b.replace_id(a.id)
        return all(
            cls.equal(a_arg, b_arg)
            if isinstance(a_arg, BaseNode) and isinstance(b_arg, BaseNode)
            else a_arg == b_arg
            for a_arg, b_arg in zip(a.args, b.args)
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Function):
            return False
        return self.equal(self, other)

class FunctionId(Integer):
    pass

class FunctionCall(InheritableNode):
    def __init__(self, function_id: FunctionId, arg_group: ValueGroup):
        assert isinstance(function_id, FunctionId)
        assert isinstance(arg_group, ValueGroup)
        super().__init__(function_id, arg_group)

    @property
    def function_id(self) -> FunctionId:
        function_id = self.args[0]
        assert isinstance(function_id, FunctionId)
        return function_id

    @property
    def arg_group(self) -> ValueGroup:
        arg_group = self.args[1]
        assert isinstance(arg_group, ValueGroup)
        return arg_group

class FunctionDefinition(InheritableNode, typing.Generic[T]):
    def __init__(self, function_id: FunctionId, function: Function[T]):
        assert isinstance(function_id, FunctionId)
        assert isinstance(function, Function)
        super().__init__(function_id, function)

    @property
    def function_id(self) -> FunctionId:
        function_id = self.args[0]
        assert isinstance(function_id, FunctionId)
        return function_id

    @property
    def function(self) -> Function[T]:
        f = self.args[1]
        return typing.cast(Function[T], f)

    def call(self, arg_group: ValueGroup) -> FunctionCall:
        self.function.param_type_group.validate(arg_group)
        return FunctionCall(self.function_id, arg_group)

    def expand(self, call: FunctionCall) -> BaseNode:
        assert call.function_id == self.function_id
        return self.function.with_args(*call.arg_group.as_tuple)

###########################################################
####################### BASIC NODES #######################
###########################################################

BASIC_NODE_TYPES: tuple[type[BaseNode], ...] = (
    BaseNode,
    VoidNode,
    Integer,
    Param,
    TypeNode,
    InheritableNode,
    BooleanNode,
    MultiArgBooleanNode,
    Function,
    ParamsGroup,
    ArgsGroup,
    ParamsArgsGroup,
    CountableTypeGroup,
    ValueGroup,
    GroupWithArgs,
)
