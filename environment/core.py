from __future__ import annotations
from abc import ABC
import typing
import sympy

###########################################################
##################### MAIN INTERFACES #####################
###########################################################

class INode:

    @classmethod
    def new(cls, *args: int | INode | typing.Type[INode]) -> typing.Self:
        raise NotImplementedError()

    @property
    def as_node(self) -> BaseNode:
        raise NotImplementedError()

    @property
    def func(self) -> typing.Type[typing.Self]:
        return type(self)

class IDefault(INode):

    @classmethod
    def create(cls) -> typing.Self:
        raise NotImplementedError()

class IFromInt(INode):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        raise NotImplementedError()

class IInt(IFromInt):

    @property
    def to_int(self) -> int:
        raise NotImplementedError()

class INodeIndex(INode):

    def from_node(self, node: INode) -> INode | None:
        raise NotImplementedError

    def replace_target(
        self,
        target_node: INode,
        new_node: INode,
    ) -> INode | None:
        raise NotImplementedError

T = typing.TypeVar('T', bound=INode)
INT = typing.TypeVar('INT', bound=IInt)

class ISingleChild(INode, typing.Generic[T]):

    @classmethod
    def with_child(cls, child: T) -> typing.Self:
        raise NotImplementedError()

    @property
    def child(self) -> T:
        raise NotImplementedError()

class IGroup(INode, typing.Generic[T]):

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError()

    @property
    def as_tuple(self) -> tuple[T, ...]:
        raise NotImplementedError()

    @classmethod
    def from_items(cls, items: typing.Sequence[T]) -> typing.Self:
        return cls(*items)

class IFunction(INode, typing.Generic[T]):

    @property
    def scope(self) -> SimpleScope[T]:
        raise NotImplementedError

    def with_args(self, *args: INode) -> INode:
        raise NotImplementedError()

class IBoolean(INode):

    @property
    def value(self) -> bool | None:
        raise NotImplementedError

###########################################################
######################## BASE NODE ########################
###########################################################

class BaseNode(INode):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(RestTypeGroup(UnknownType()))

    @classmethod
    def new(cls, *args: int | INode | typing.Type[INode]) -> typing.Self:
        return cls(*args)

    @property
    def as_node(self) -> BaseNode:
        return self

    def __init__(self, *args: int | INode | typing.Type[INode]):
        assert all(
            (
                isinstance(arg, INode)
                or isinstance(arg, int)
                or (isinstance(arg, type) and issubclass(arg, INode))
            )
            for arg in args
        )

        if any(
            (
                isinstance(arg, int)
                or (isinstance(arg, type) and issubclass(arg, INode))
            )
            for arg in args
        ):
            assert len(args) == 1

        self._args = args
        self._cached_length: int | None = None
        self._cached_hash: int | None = None

    @property
    def args(self) -> tuple[int | INode | typing.Type[INode], ...]:
        return self._args

    def __eq__(self, other) -> bool:
        if not isinstance(other, INode):
            return False
        if not self.func == other.as_node.func:
            return False
        if not len(self.args) == len(other.as_node.args):
            return False
        for i, arg in enumerate(self.args):
            if arg != other.as_node.args[i]:
                return False
        return True

    def __getitem__(self, index: INodeIndex) -> INode | None:
        return index.from_node(self)

    def replace_at(self, index: INodeIndex, new_node: INode) -> INode | None:
        return index.replace_target(self, new_node)

    def __len__(self) -> int:
        if self._cached_length is not None:
            return self._cached_length
        length = 1 + sum(
            len(arg.as_node)
            for arg in self.args
            if isinstance(arg, INode)
        )
        self._cached_length = length
        return length

    def __hash__(self) -> int:
        if self._cached_hash is not None:
            return self._cached_hash
        hash_value = hash((self.func, self.args))
        self._cached_hash = hash_value
        return hash_value

    def find_until(self, node_type: type[T], until_type: type[INode] | None) -> set[T]:
        return {
            node
            for node in self.args
            if isinstance(node, node_type)
        }.union({
            node
            for parent_node in self.args
            if (
                isinstance(parent_node, INode)
                and
                (until_type is None or not isinstance(parent_node, until_type))
            )
            for node in parent_node.as_node.find_until(node_type, until_type)
        })

    def find(self, node_type: type[T]) -> set[T]:
        return self.find_until(node_type, None)

    def replace_until(
        self,
        before: INode,
        after: INode,
        until_type: type[INode] | None,
    ) -> INode:
        if self == before:
            return after
        return self.func(*[
            (
                arg.as_node.replace_until(before, after, until_type)
                if (
                    isinstance(arg, INode)
                    and
                    (until_type is None or not isinstance(arg, until_type))
                )
                else arg
            )
            for arg in self.args
        ])

    def replace(self, before: INode, after: INode) -> INode:
        return self.replace_until(before, after, None)

    def subs(self, mapping: dict[INode, INode]) -> INode:
        node: INode = self
        for key, value in mapping.items():
            node = node.as_node.replace(key, value)
        return node

    def has(self, node: INode) -> bool:
        return node in self.as_node.find(node.__class__)

    def has_until(self, node: INode, until_type: type[INode] | None) -> bool:
        return node in self.as_node.find_until(node.__class__, until_type)

    def eval(self) -> sympy.Basic:
        name = self.func.__name__
        if len(self.args) == 0:
            return sympy.Symbol(name)

        def eval_arg(arg: int | INode | typing.Type[INode]):
            if isinstance(arg, INode):
                return arg.as_node.eval()
            if isinstance(arg, int):
                return sympy.Integer(arg)
            if isinstance(arg, type) and issubclass(arg, INode):
                return sympy.Symbol(arg.__name__)
            raise ValueError(f"Invalid argument: {arg}")

        fn = sympy.Function(name)
        args: list[sympy.Basic] = [eval_arg(arg) for arg in self.args]
        # pylint: disable=not-callable
        return fn(*args)

    def validate(self):
        for arg in self.args:
            if isinstance(arg, INode):
                arg.as_node.validate()

###########################################################
######################## TYPE NODE ########################
###########################################################

class TypeNode(BaseNode, IFunction['TypeNode'], typing.Generic[T]):

    def with_args(self, *args: INode) -> INode:
        return self.type.new(*args)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def __init__(self, type: type[T]):
        assert isinstance(type, type) and issubclass(type, INode)
        super().__init__(type)

    @property
    def type(self) -> typing.Type[INode]:
        t = self.args[0]
        return typing.cast(typing.Type[INode], t)

    @property
    def scope(self) -> SimpleScope[typing.Self]:
        return OpaqueScope.with_content(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TypeNode):
            return False
        return self.type == other.type

###########################################################
######################## INT NODE #########################
###########################################################

class Integer(BaseNode, IInt):

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

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(value)

    @property
    def to_int(self) -> int:
        return self.value

###########################################################
######################## MAIN NODE ########################
###########################################################

class InheritableNode(BaseNode):

    def __init__(self, *args: INode):
        assert all(isinstance(arg, INode) for arg in args)
        self.arg_type_group().validate_values(DefaultGroup(*args))
        super().__init__(*args)

    @property
    def args(self) -> tuple[INode, ...]:
        return typing.cast(tuple[INode, ...], self._args)

    def __getitem__(self, index: INodeIndex) -> INode | None:
        if isinstance(index, NodeArgIndex):
            args = self.args
            if index.to_int > 0 and index.to_int <= len(args):
                return args[index.to_int - 1]
            return None
        return super().__getitem__(index)

    def replace_at(self, index: INodeIndex, new_node: INode) -> INode | None:
        assert isinstance(index, INodeIndex)
        assert isinstance(new_node, INode)
        if isinstance(index, NodeArgIndex):
            value = index.to_int
            args = self.args
            if value >= 0 and value <= len(args):
                return self.func(*[
                    (new_node if i == value - 1 else arg)
                    for i, arg in enumerate(args)
                ])
            return None
        return super().replace_at(index, new_node)

###########################################################
###################### SPECIAL NODES ######################
###########################################################

class Void(InheritableNode, IDefault):

    @classmethod
    def create(cls) -> Void:
        return cls()

    def __init__(self):
        super().__init__()

class UnknownType(TypeNode[INode], IDefault):

    @classmethod
    def create(cls) -> UnknownType:
        return cls()

    def __init__(self):
        super().__init__(self.__class__)

class Optional(BaseNode, IDefault, ISingleChild[T], typing.Generic[T]):

    @classmethod
    def create(cls) -> typing.Self:
        return cls()

    @classmethod
    def with_child(cls, child: T) -> typing.Self:
        return cls(child)

    def __init__(self, value: T | None = None):
        if value is not None:
            super().__init__()
        else:
            assert isinstance(value, INode)
            super().__init__(value)

    @property
    def value(self) -> T | None:
        if len(self.args) == 0:
            return None
        value = self.args[0]
        return typing.cast(T, value)

    @property
    def child(self) -> T:
        value = self.value
        assert value is not None
        return value

###########################################################
########################## SCOPE ##########################
###########################################################

class ScopeId(Integer, IDefault):

    @classmethod
    def create(cls) -> ScopeId:
        return cls(1)

class TemporaryScopeId(ScopeId):
    pass

class Scope(InheritableNode):

    @property
    def id(self) -> ScopeId:
        raise NotImplementedError

    def has_dependency(self) -> bool:
        raise NotImplementedError

    def replace_id(self, new_id: ScopeId) -> typing.Self:
        node = self.replace_until(self.id, new_id, OpaqueScope)
        assert isinstance(node, self.__class__)
        return node

class SimpleScope(Scope, typing.Generic[T]):

    def __init__(self, id: ScopeId, child: T):
        assert isinstance(id, ScopeId)
        assert isinstance(child, INode)
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
        return self.child.as_node.has_until(self.id, OpaqueScope)

class OpaqueScope(SimpleScope[T], typing.Generic[T]):

    @classmethod
    def with_content(cls, child: T) -> typing.Self:
        return cls(ScopeId(1), child)

    def __init__(self, id: ScopeId, child: T):
        assert id.value == 1
        super().__init__(id, child)

    @classmethod
    def normalize_from(cls, node: INode, next_id: int) -> INode:
        if isinstance(node, OpaqueScope):
            return node.normalize()
        child_scope_id = (next_id+1) if isinstance(node, Scope) else next_id
        new_args = [
            cls.normalize_from(child, child_scope_id)
            if isinstance(child, INode)
            else child
            for child in node.as_node.args
        ]
        node = node.as_node.func(*new_args)
        if isinstance(node, Scope):
            node = node.replace_id(TemporaryScopeId(next_id))
        return node

    def normalize(self) -> typing.Self:
        next_id = 1
        child_args = [
            self.normalize_from(child, next_id+1)
            if isinstance(child, INode)
            else child
            for child in self.child.as_node.args
        ]
        child = self.child.func(*child_args)
        node = self.func(TemporaryScopeId(next_id), child)
        tmp_ids = node.find_until(TemporaryScopeId, OpaqueScope)
        for tmp_id in tmp_ids:
            node_aux = node.replace_until(tmp_id, ScopeId(tmp_id.value), OpaqueScope)
            assert isinstance(node_aux, self.__class__)
            node = node_aux
        return node

    def validate(self):
        assert self == self.normalize()
        super().validate()

class Placeholder(InheritableNode, IFromInt, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(
            ScopeId.create(),
            Integer(value),
            typing.cast(TypeNode[T], UnknownType()))

    def __init__(self, parent_scope: ScopeId, index: Integer, type_node: TypeNode[T]):
        assert isinstance(parent_scope, ScopeId)
        assert isinstance(index, IInt)
        assert isinstance(type_node, TypeNode)
        super().__init__(parent_scope, index, type_node)

    @property
    def parent_scope_id(self) -> ScopeId:
        parent_scope = self.args[0]
        assert isinstance(parent_scope, ScopeId)
        return parent_scope

    @property
    def index(self) -> IInt:
        index = self.args[1]
        assert isinstance(index, IInt)
        return index

    @property
    def type_node(self) -> TypeNode[T]:
        type_node = self.args[2]
        assert isinstance(type_node, TypeNode)
        return type_node

class Param(Placeholder[T], typing.Generic[T]):
    pass

class Var(Placeholder[T], typing.Generic[T]):
    pass

###########################################################
######################## NODE IDXS ########################
###########################################################

class NodeIntBaseIndex(Integer, ABC, INodeIndex, IInt):
    pass

class NodeMainIndex(NodeIntBaseIndex):

    @classmethod
    def _inner_getitem(cls, node: INode, index: int) -> tuple[INode | None, int]:
        assert index > 0
        index -= 1
        if index == 0:
            return node, index
        for arg in node.as_node.args:
            if isinstance(arg, INode):
                arg_item, index = cls._inner_getitem(arg, index)
                if index == 0:
                    return arg_item, index
        return None, index

    @classmethod
    def _replace(
        cls,
        target_node: INode,
        new_node: INode,
        index: int,
    ) -> tuple[INode | None, int]:
        assert index > 0
        index -= 1
        if index == 0:
            return new_node, index
        for i, arg in enumerate(target_node.as_node.args):
            if isinstance(arg, INode):
                new_arg, index = cls._replace(arg, new_node, index)
                if index == 0:
                    if new_arg is None:
                        return None, index
                    return target_node.func(*[
                        (new_arg if i == j else old_arg)
                        for j, old_arg in enumerate(target_node.as_node.args)
                    ]), index
        return None, index

    @classmethod
    def _ancestors(cls, node: INode, index: int) -> tuple[list[INode], int]:
        assert index > 0
        index -= 1
        if index == 0:
            return [node], index
        for arg in node.as_node.args:
            if isinstance(arg, INode):
                ancestors, index = cls._ancestors(node, index)
                if index == 0:
                    if len(ancestors) == 0:
                        return [], index
                    return [node] + ancestors, index
        return [], index

    def from_node(self, node: INode) -> INode | None:
        item, _ = self._inner_getitem(node, self.to_int)
        return item

    def replace_target(self, target_node: INode, new_node: INode) -> INode | None:
        new_target, _ = self._replace(target_node, new_node, self.to_int)
        return new_target

    def ancestors(self, node: INode) -> tuple[INode, ...]:
        ancestors, _ = self._ancestors(node, self.to_int)
        return tuple(ancestors)

class NodeArgIndex(NodeIntBaseIndex):

    def from_node(self, node: INode) -> INode | None:
        index = self.to_int
        if not isinstance(node, InheritableNode):
            return None
        args = node.args
        if index > 0 and index <= len(args):
            return args[index - 1]
        return None

    def replace_target(self, target_node: T, new_node: INode) -> T | None:
        if not isinstance(target_node, InheritableNode):
            return None
        index = self.to_int
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
####################### ITEMS GROUP #######################
###########################################################

class BaseGroup(InheritableNode, ABC, IGroup[T], typing.Generic[T]):

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

class DefaultGroup(BaseGroup[INode]):

    @classmethod
    def item_type(cls):
        return INode

class OptionalValueGroup(BaseGroup[Optional[T]], IFromInt, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(*[Optional.create() for _ in range(value)])

    @classmethod
    def item_type(cls):
        return Optional[T]

    @classmethod
    def from_optional_items(cls, items: typing.Sequence[T | None]) -> typing.Self:
        return cls(*[Optional(item) for item in items])

    def strict(self) -> DefaultGroup:
        values = [item.value for item in self.args if item.value is not None]
        assert len(values) == len(self.args)
        return DefaultGroup(*values)

    def fill_with_void(self) -> DefaultGroup:
        def get_value(v: T | None) -> INode:
            if v is None:
                return Void()
            return v
        values = [get_value(item.value) for item in self.args]
        return DefaultGroup(*values)

###########################################################
####################### TYPE GROUPS #######################
###########################################################

class CountableTypeGroup(BaseGroup[TypeNode[T]], typing.Generic[T]):

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

    def validate_values(self, values: BaseGroup):
        group = self.group
        args = values.args

        if isinstance(group, CountableTypeGroup):
            assert len(args) == len(group.args)
            for i, arg in enumerate(args):
                t_arg = group.args[i]
                assert isinstance(arg, INode)
                assert isinstance(t_arg, TypeNode)
                if t_arg.type is not UnknownType:
                    assert isinstance(arg, t_arg.type)
        elif isinstance(group, RestTypeGroup):
            t_arg = group.type
            if t_arg.type is not UnknownType:
                for arg in args:
                    assert isinstance(arg, INode)
                    assert isinstance(t_arg, TypeNode)
                    assert isinstance(arg, t_arg.type)
        else:
            raise ValueError(f"Invalid group type: {group}")

    @classmethod
    def default(cls, amount: Optional[INT]) -> ExtendedTypeGroup[UnknownType]:
        value = amount.value.to_int if amount.value is not None else None
        if value is None:
            return ExtendedTypeGroup(RestTypeGroup(TypeNode(UnknownType)))
        return ExtendedTypeGroup(
            CountableTypeGroup(*[TypeNode(UnknownType) for _ in range(value)]))

###########################################################
###################### FUNCTION NODE ######################
###########################################################

class Function(InheritableNode, IFunction[T], typing.Generic[T]):

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
        return typing.cast(SimpleScope[T], scope)

    @property
    def scope_id(self) -> ScopeId:
        return self.scope.id

    @property
    def expr(self) -> T:
        return self.scope.child

    def owned_params(self) -> typing.Sequence[Param]:
        params = [
            param
            for param in self.expr.as_node.find_until(Param, OpaqueScope)
            if param.parent_scope_id == self.scope_id]
        params = sorted(params, key=lambda param: param.index.to_int)
        return params

    def with_args(self, *args: INode) -> INode:
        return self.with_arg_group(DefaultGroup(*args))

    def with_arg_group(self, group: BaseGroup) -> INode:
        self.validate()
        type_group = self.param_type_group
        args = group.as_tuple
        type_group.validate_values(group)
        params = self.owned_params()
        scope = self.scope
        for param in params:
            index = param.index.to_int
            assert index > 0
            assert index <= len(args)
            arg = args[index-1]
            param_type = param.type_node.type
            if param_type is not UnknownType:
                assert isinstance(arg, param_type)
            scope_aux = scope.replace_until(param, arg, OpaqueScope)
            assert isinstance(scope_aux, SimpleScope)
            scope = scope_aux
        assert not scope.has_dependency()
        return scope.child

    def validate(self):
        params = self.owned_params()
        group = self.param_type_group.group
        scope = self.scope
        for param in params:
            assert param.index.value > 0
            if isinstance(group, CountableTypeGroup):
                assert param.index.value <= len(group.args)
                assert param.type_node.type == group.args[param.index.value-1].type
            else:
                assert param.type_node.type == group.type.type
        if isinstance(scope, OpaqueScope):
            all_inner_functions = scope.find_until(IFunction, OpaqueScope)
            all_functions = all_inner_functions.union(self)
            all_functions_scope_ids = {f.scope.id for f in all_functions}
            all_inner_params = scope.find_until(Param, OpaqueScope)
            all_inner_params_scope_ids = {p.parent_scope_id for p in all_inner_params}
            assert all_inner_params_scope_ids.issubset(all_functions_scope_ids)
        super().validate()

class FunctionId(Integer):
    pass

class FunctionCall(InheritableNode, typing.Generic[T]):

    def __init__(self, function_id: FunctionId, arg_group: BaseGroup[T]):
        assert isinstance(function_id, FunctionId)
        assert isinstance(arg_group, BaseGroup)
        super().__init__(function_id, arg_group)

    @property
    def function_id(self) -> FunctionId:
        function_id = self.args[0]
        assert isinstance(function_id, FunctionId)
        return function_id

    @property
    def arg_group(self) -> BaseGroup[T]:
        arg_group = self.args[1]
        return typing.cast(BaseGroup[T], arg_group)

class FunctionDefinition(InheritableNode, IFunction[T], typing.Generic[T]):

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

    @property
    def scope(self) -> SimpleScope[T]:
        return self.function.scope

    def with_args(self, *args: INode) -> INode:
        arg_group = DefaultGroup(*args)
        self.function.param_type_group.validate_values(arg_group)
        return FunctionCall(self.function_id, arg_group)

    def expand(self, call: FunctionCall) -> INode:
        assert call.function_id == self.function_id
        return self.function.with_args(*call.arg_group.as_tuple)

###########################################################
####################### BASIC NODES #######################
###########################################################

BASIC_NODE_TYPES: tuple[type[INode], ...] = (
    INode,
    BaseNode,
    Void,
    Integer,
    Param,
    Var,
    TypeNode,
    InheritableNode,
    IBoolean,
    Function,
    CountableTypeGroup,
    DefaultGroup,
    OptionalValueGroup,
)
