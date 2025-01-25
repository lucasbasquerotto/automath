# pylint: disable=too-many-lines
from __future__ import annotations
from abc import ABC
import typing
import sympy

###########################################################
##################### MAIN INTERFACES #####################
###########################################################

class INode(ABC):

    @classmethod
    def as_type(cls) -> TypeNode[typing.Self]:
        return TypeNode(cls)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        raise NotImplementedError(cls)

    @property
    def as_node(self) -> BaseNode:
        raise NotImplementedError(self.__class__)

    @property
    def as_instance(self) -> IInstantiable:
        raise NotImplementedError(self.__class__)

    @property
    def func(self) -> typing.Type[typing.Self]:
        return type(self)

class IInheritableNode(INode, ABC):

    @classmethod
    def new(cls, *args: INode) -> typing.Self:
        raise NotImplementedError(cls)

class ISpecialValue(INode, ABC):

    @property
    def node_value(self) -> INode:
        raise NotImplementedError(self.__class__)

class IDefault(INode, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls()

    @property
    def as_node(self) -> BaseNode:
        raise NotImplementedError(self.__class__)

class IAdditive(INode, ABC):

    def add(self, another: INode, run_info: RunInfo) -> RunInfoResult:
        raise NotImplementedError(self.__class__)

class IFromInt(INode, ABC):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        raise NotImplementedError(cls)

class IInt(IFromInt, IAdditive, ABC):

    @property
    def as_int(self) -> int:
        raise NotImplementedError(self.__class__)

    def add(self, another: INode, run_info: RunInfo):
        info = run_info

        info, node_aux = self.as_node.run(info).as_tuple
        node_1 = node_aux.as_node.cast(IInt)

        info, another_aux = another.as_node.run(info).as_tuple
        node_2 = another_aux.as_node.cast(IInt)

        new_value = node_1.as_int + node_2.as_int
        return info.to_result(self.__class__.from_int(new_value))

class INodeIndex(INode, ABC):

    def find_in_node(self, node: INode) -> IOptional[INode]:
        raise NotImplementedError(self.__class__)

    def replace_in_target(
        self,
        target_node: INode,
        new_node: INode,
    ) -> IOptional[INode]:
        raise NotImplementedError

    def remove_in_target(self, target_node: INode) -> IOptional[INode]:
        raise NotImplementedError(self.__class__)

T = typing.TypeVar('T', bound=INode)
O = typing.TypeVar('O', bound=INode)
K = typing.TypeVar('K', bound=INode)
INT = typing.TypeVar('INT', bound=IInt)

class IFromSingleNode(IInheritableNode, typing.Generic[T], ABC):

    @classmethod
    def with_node(cls, node: T) -> typing.Self:
        return cls.new(node)

class ISingleChild(IFromSingleNode, typing.Generic[T], ABC):

    @property
    def child(self) -> T:
        raise NotImplementedError(self.__class__)

class IOptional(IDefault, IFromSingleNode[T], typing.Generic[T], ABC):

    @property
    def value(self) -> T | None:
        raise NotImplementedError()

    def value_or_else(self, default_value: T) -> T:
        value = self.value
        if value is None:
            value = default_value
        return value

    def is_empty(self) -> IsEmpty:
        return IsEmpty(self)

    @property
    def value_or_raise(self) -> T:
        value = IsEmpty(self).value_or_raise
        return typing.cast(T, value)

    def raise_if_empty(self):
        self.is_empty().raise_if_empty()

class ISingleOptionalChild(ISingleChild[IOptional[T]], typing.Generic[T], ABC):

    @classmethod
    def with_optional(cls, child: T | None) -> typing.Self:
        return cls.with_node(Optional(child) if child is not None else Optional())

    @property
    def child(self) -> IOptional[T]:
        raise NotImplementedError(self.__class__)

class IWrapper(INode, ABC):
    pass

class IGroup(IWrapper, IInheritableNode, IAdditive, typing.Generic[T], ABC):

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError(cls)

    @property
    def as_tuple(self) -> tuple[T, ...]:
        raise NotImplementedError(self.__class__)

    @classmethod
    def from_items(cls, items: typing.Sequence[T]) -> typing.Self:
        return cls(*items)

    def add(self, another: INode, run_info: RunInfo):
        info = run_info

        info, node_aux = self.as_node.run(info).as_tuple
        node_1 = node_aux.as_node.cast(IGroup)

        info, another_aux = another.as_node.run(info).as_tuple
        node_2 = another_aux.as_node.cast(IGroup)

        new_args = list(node_1.as_tuple) + list(node_2.as_tuple)
        return info.to_result(self.__class__.from_items(new_args))

class IFunction(INode, ABC):

    def with_arg_group(self, group: BaseGroup, info: RunInfo) -> RunInfoResult:
        raise NotImplementedError(self.__class__)

class IRunnable(INode, ABC):

    _cached_run: dict[tuple[INode, RunInfo], RunInfoResult] = dict()

    @classmethod
    def clear_cache(cls):
        cls._cached_run.clear()

    def run(self, info: RunInfo) -> RunInfoResult:
        cached = self._cached_run.get((self, info))
        if cached is not None:
            return cached

        try:
            try:
                result = self._run(info)
            except NodeReturnException as e:
                result = e.result

            assert isinstance(result, RunInfoResult)
            new_info, new_node = result.as_tuple

            if isinstance(self, IOpaqueScope):
                result = info.to_result(new_node)
                new_info, new_node = result.as_tuple
            else:
                new_info = new_info.with_scopes(info)
                result = new_info.to_result(new_node)

            self._cached_run[(self, info)] = result

            try:
                new_result = new_node.as_node.run(info)
            except NodeReturnException as e:
                new_result = e.result
            _, node_aux = new_result.as_tuple
            Eq(new_node, node_aux).raise_on_not_true()

            if new_info.must_return():
                raise NodeReturnException(result)

            return result
        except NodeReturnException as e:
            raise e
        except InvalidNodeException as e:
            exc_info = e.info.add_stack(
                node=self,
                run_info=info,
            )
            raise InvalidNodeException(exc_info) from e

    def _run(self, info: RunInfo) -> RunInfoResult:
        raise NotImplementedError(self.__class__)

class IBoolean(INode):

    _true: IBoolean | None = None
    _false: IBoolean | None = None

    @property
    def as_bool(self) -> bool | None:
        raise NotImplementedError(self.__class__)

    @property
    def strict_bool(self) -> bool:
        value = self.as_bool
        if value is None:
            raise self.to_exception()
        return value

    def raise_on_false(self):
        if self.as_bool is False:
            raise self.to_exception()

    def raise_on_not_true(self):
        if self.as_bool is not True:
            raise self.to_exception()

    def raise_on_undefined(self):
        if self.as_bool is None:
            raise self.to_exception()

    def to_exception_info(self):
        return BooleanExceptionInfo(self)

    def to_exception(self):
        return InvalidNodeException(self.to_exception_info())

    @classmethod
    def true(cls) -> IBoolean:
        t = cls._true
        if t is None:
            t = IntBoolean.create_true()
            cls._true = t
        return t

    @classmethod
    def false(cls) -> IBoolean:
        f = cls._false
        if f is None:
            f = IntBoolean.create()
            cls._false = f
        return f

class ITypedIndex(INodeIndex, typing.Generic[O, T], ABC):

    @classmethod
    def outer_type(cls) -> type[O]:
        raise NotImplementedError(cls)

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError(cls)

    def find_in_node(self, node: INode):
        assert isinstance(node, self.outer_type())
        return self.find_in_outer_node(node)

    def replace_in_target(self, target_node: INode, new_node: INode):
        assert isinstance(target_node, self.outer_type())
        assert isinstance(new_node, self.item_type())
        return self.replace_in_outer_target(target_node, new_node)

    def remove_in_target(self, target_node: INode):
        assert isinstance(target_node, self.outer_type())
        return self.remove_in_outer_target(target_node)

    def find_in_outer_node(self, node: O) -> IOptional[T]:
        raise NotImplementedError(self.__class__)

    def replace_in_outer_target(self, target: O, new_node: T) -> IOptional[O]:
        raise NotImplementedError(self.__class__)

    def remove_in_outer_target(self, target: O) -> IOptional[O]:
        raise NotImplementedError(self.__class__)

class ITypedIntIndex(IInt, ITypedIndex[O, T], typing.Generic[O, T], ABC):

    @classmethod
    def outer_type(cls) -> type[O]:
        raise NotImplementedError

    def find_arg(self, node: INode) -> IOptional[T]:
        result = NodeArgIndex(self.as_int).find_in_node(node)
        if result.value is not None:
            assert isinstance(result.value, self.item_type()), \
                f'{type(result.value)} != {self.item_type()}'
        return typing.cast(IOptional[T], result)

    def replace_arg(self, target: K, new_node: T) -> IOptional[K]:
        return NodeArgIndex(self.as_int).replace_in_target(target, new_node)

    def remove_arg(self, target: K) -> IOptional[K]:
        return NodeArgIndex(self.as_int).remove_in_target(target)

class IInstantiable(INode, ABC):

    @property
    def as_instance(self) -> typing.Self:
        return self

###########################################################
############### TEMPORARY AUXILIAR CLASSES ################
###########################################################

class TmpInnerArg:

    def __init__(self, node: BaseNode, idx: int):
        assert isinstance(node, BaseNode)
        assert isinstance(idx, int)
        self.node = node
        self.idx = idx

    def apply(self) -> BaseNode:
        node = self.node
        idx = self.idx
        node_aux = node.args[idx-1]
        assert isinstance(node_aux, BaseNode)
        node = node_aux
        return node

    def to_node(self) -> InnerArg:
        return InnerArg(self.node, NodeArgIndex(self.idx))

class TmpNestedArg:

    def __init__(self, node: BaseNode, idxs: tuple[int, ...]):
        assert isinstance(node, BaseNode)
        assert all(isinstance(idx, int) for idx in idxs)
        self.node = node
        self.idxs = idxs

    def apply(self) -> BaseNode:
        node = self.node
        idxs = self.idxs
        for idx in idxs:
            node_aux = node.args[idx-1]
            assert isinstance(node_aux, BaseNode)
            node = node_aux
        return node

    def to_node(self) -> NestedArg:
        return NestedArg(
            self.node,
            NestedArgIndexGroup.from_ints(self.idxs)
        )

###########################################################
######################## BASE NODE ########################
###########################################################

class BaseNode(IRunnable, ABC):

    def __init__(self, *args: int | INode | typing.Type[INode]):
        if not isinstance(self, IInstantiable):
            raise NotImplementedError(self.__class__)
        self._args = args
        self._cached_length: int | None = None
        self._cached_hash: int | None = None

    @property
    def args(self) -> tuple[int | INode | typing.Type[INode], ...]:
        return self._args

    @property
    def as_node(self) -> BaseNode:
        return self

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

    def __getitem__(self, index: INodeIndex):
        return index.find_in_node(self)

    def replace_at(self, index: INodeIndex, new_node: INode):
        return index.replace_in_target(self, new_node)

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

    def inner_arg(self, idx: int) -> TmpInnerArg:
        return TmpInnerArg(self, idx)

    def nested_arg(self, idxs: tuple[int, ...]) -> TmpNestedArg:
        return TmpNestedArg(self, idxs)

    def validate(self):
        for arg in self.args:
            if isinstance(arg, INode):
                arg.as_node.validate()

    def strict_validate(self):
        self.validate()

    def cast(self, t: typing.Type[T]) -> T:
        origin = typing.get_origin(t)
        t = origin if origin is not None else t
        assert isinstance(self, t), f'{type(self)} != {t}'
        return typing.cast(T, self)

###########################################################
######################## TYPE NODE ########################
###########################################################

class IType(INode, ABC):

    def accepts(self, inner_type: IType) -> bool | None:
        raise NotImplementedError

    def accepted_by(self, outer_type: IType) -> bool | None:
        raise NotImplementedError

    def valid(self, instance: INode) -> bool | None:
        raise NotImplementedError

    def verify(self, instance: INode):
        if self.valid(instance) is False:
            raise InvalidNodeException(TypeExceptionInfo(self, instance))

class TypeNode(BaseNode, IType, IFunction, ISpecialValue, typing.Generic[T], IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def __init__(self, t: type[T]):
        origin = typing.get_origin(t)
        t = origin if origin is not None else t
        assert isinstance(t, type) and issubclass(t, INode), t
        super().__init__(t)

    @property
    def type(self) -> typing.Type[INode]:
        t = self.args[0]
        return typing.cast(typing.Type[INode], t)

    @property
    def node_value(self) -> INode:
        return self

    def __hash__(self) -> int:
        if self._cached_hash is not None:
            return self._cached_hash
        full_type_name = self.type.__module__ + '.' + self.type.__name__
        hash_value = hash((self.func, full_type_name))
        self._cached_hash = hash_value
        return hash_value

    def with_arg_group(self, group: BaseGroup, info: RunInfo):
        t = self.type
        assert issubclass(t, InheritableNode)
        return t.new(*group.as_tuple).as_node.run(info)

    def accepted_by(self, outer_type: IType) -> bool | None:
        if isinstance(outer_type, TypeNode):
            return issubclass(self.type, outer_type.type)
        return outer_type.accepts(self)

    def accepts(self, inner_type: IType) -> bool | None:
        if isinstance(inner_type, TypeNode):
            return issubclass(inner_type.type, self.type)
        return inner_type.accepted_by(self)

    def valid(self, instance: INode) -> bool | None:
        if not issubclass(self.type, Placeholder):
            if isinstance(instance, Placeholder):
                return None
        return isinstance(instance, self.type)

    def _run(self, info: RunInfo):
        self.validate()
        return info.to_result(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TypeNode):
            return False
        return self.type == other.type

###########################################################
######################## INT NODE #########################
###########################################################

class BaseInt(BaseNode, IInt, ISpecialValue, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def __init__(self, value: int):
        assert isinstance(value, int)
        super().__init__(value)

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(value)

    @property
    def as_int(self) -> int:
        value = self.args[0]
        assert isinstance(value, int)
        return value

    @property
    def node_value(self) -> INode:
        return self

    def _run(self, info: RunInfo):
        self.validate()
        return info.to_result(self)

class Integer(BaseInt, IInstantiable):

    _zero: Integer | None = None
    _one: Integer | None = None

    @classmethod
    def from_int(cls, value: int) -> Integer:
        if value == 0:
            z = cls._zero
            if z is None:
                z = cls(value)
                cls._zero = z
            return z
        if value == 1:
            o = cls._one
            if o is None:
                o = cls(value)
                cls._one = o
            return o
        return cls(value)

    @classmethod
    def zero(cls) -> Integer:
        return cls.from_int(0)

    @classmethod
    def one(cls) -> Integer:
        return cls.from_int(1)


###########################################################
######################## MAIN NODE ########################
###########################################################

class InheritableNode(BaseNode, IInheritableNode, ABC):

    @classmethod
    def new(cls, *args: INode) -> typing.Self:
        return cls(*args)

    def __init__(self, *args: INode):
        assert all(isinstance(arg, INode) for arg in args)
        super().__init__(*args)

    @property
    def args(self) -> tuple[INode, ...]:
        return typing.cast(tuple[INode, ...], self._args)

    def __getitem__(self, index: INodeIndex) -> INode | None:
        if isinstance(index, NodeArgIndex):
            args = self.args
            if index.as_int > 0 and index.as_int <= len(args):
                return args[index.as_int - 1]
            return None
        return super().__getitem__(index)

    def replace_at(self, index: INodeIndex, new_node: INode) -> INode | None:
        assert isinstance(index, INodeIndex)
        assert isinstance(new_node, INode)
        if isinstance(index, NodeArgIndex):
            value = index.as_int
            args = self.args
            if 0 <= value <= len(args):
                return self.func(*[
                    (new_node if i == value - 1 else arg)
                    for i, arg in enumerate(args)
                ])
            return None
        return super().replace_at(index, new_node)

    def validate(self):
        super().validate()
        args = self.args
        type_group = self.arg_type_group().group.apply()
        if isinstance(type_group, OptionalTypeGroup):
            assert len(args) <= 1, \
                f'{type(self)}: {len(args)} > 1'
        elif isinstance(type_group, CountableTypeGroup):
            assert len(args) == len(type_group.args), \
                f'{type(self)}: {len(args)} != {len(type_group.args)}'

    def strict_validate(self):
        self.validate()
        self.arg_type_group().validate_values(DefaultGroup(*self.args))

    def _run(self, info: RunInfo):
        self.validate()
        args: list[INode] = []
        for arg in self.args:
            info, new_arg = arg.as_node.run(info).as_tuple
            args.append(new_arg)
        result = self.func(*args)
        if not info.is_future():
            result.strict_validate()
        return info.to_result(result)

###########################################################
###################### SPECIAL NODES ######################
###########################################################

class Void(InheritableNode, IDefault, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

class UnknownType(InheritableNode, IType, IDefault, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup())

    def accepted_by(self, outer_type: IType) -> bool | None:
        return None

    def accepts(self, inner_type: IType) -> bool | None:
        return None

    def valid(self, instance: INode) -> bool | None:
        return None

class OptionalBase(InheritableNode, IOptional[T], typing.Generic[T], ABC):

    idx_value = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(OptionalTypeGroup(TypeNode(INode)))

    @property
    def value(self) -> T | None:
        if len(self.args) == 0:
            return None
        value = self.inner_arg(self.idx_value).apply()
        return typing.cast(T, value)

    @classmethod
    def from_optional(cls, o: IOptional[T]) -> typing.Self:
        if isinstance(o, cls):
            return o
        return cls(o.value) if o.value is not None else cls()

    @classmethod
    def with_value(cls, value: T | None) -> typing.Self:
        return cls(value) if value is not None else cls()

class Optional(OptionalBase[T], IInstantiable, typing.Generic[T]):

    @classmethod
    def with_int(cls, value: int | None) -> Optional[Integer]:
        return Optional(Integer(value)) if value is not None else Optional()

###########################################################
########################## SCOPE ##########################
###########################################################

class ParentScopeBase(BaseInt, IDefault, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls(1)

    def _index(self, info: RunInfo) -> RunInfoScopeDataIndex:
        raise NotImplementedError

    def _run(self, info: RunInfo):
        node = self._index(info)
        assert isinstance(node, RunInfoScopeDataIndex)
        scope_data_group = info.scope_data_group.apply().cast(ScopeDataGroup)
        amount = len(scope_data_group.as_tuple)
        IsInsideRange(node, Integer.zero(), Integer(amount)).raise_on_not_true()
        return info.to_result(node)

class NearParentScope(ParentScopeBase, IInstantiable):

    def _index(self, info: RunInfo):
        scope_data_group = info.scope_data_group.apply().cast(ScopeDataGroup)
        amount = len(scope_data_group.as_tuple)
        scope_index = amount - self.as_int + 1
        return RunInfoScopeDataIndex(scope_index)

class FarParentScope(ParentScopeBase, IInstantiable):

    def _index(self, info: RunInfo):
        return RunInfoScopeDataIndex(self.as_int)

class IScope(INode, ABC):
    pass

class IOpaqueScope(IScope, ABC):
    pass

class IInnerScope(IScope, ABC):
    pass

class Placeholder(InheritableNode, IFromInt, typing.Generic[T], ABC):

    idx_parent_scope = 1
    idx_index = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            RunInfoScopeDataIndex,
            BaseInt,
        ]))

    @property
    def parent_scope(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_parent_scope)

    @property
    def index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_index)

    def _run(self, info: RunInfo):
        info, node_aux = super()._run(info).as_tuple
        node = node_aux.as_node.cast(self.__class__)

        scope_index = node.parent_scope.apply().cast(RunInfoScopeDataIndex)
        index = node.index.apply().cast(BaseInt)

        scope = scope_index.find_in_outer_node(info).value_or_raise
        assert isinstance(scope, ScopeDataPlaceholderItemGroup), type(scope)

        assert isinstance(node, scope.item_inner_type()), \
            f'{type(node)} != {scope.item_inner_type()} ({scope_index.as_int} - {index.as_int})'

        if isinstance(scope, IScopeDataFutureItemGroup):
            groups = info.scope_data_group.apply().cast(ScopeDataGroup).as_tuple
            group_amount = len(groups)
            near_scope = NearParentScope.from_int(group_amount - scope_index.as_int + 1)
            return info.to_result(node.func(near_scope, index))

        item = NodeArgIndex(index.as_int).find_in_node(scope).value_or_raise
        info, node_aux = item.as_node.run(info).as_tuple
        node_aux = node_aux.as_node.cast(IOptional)

        result = node_aux.value_or_raise

        return info.to_result(result)

class Param(Placeholder[T], IInstantiable, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(
            FarParentScope.create(),
            Integer(value),
        )

class Var(Placeholder[T], IInstantiable, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(
            NearParentScope.create(),
            Integer(value),
        )

###########################################################
######################## NODE IDXS ########################
###########################################################

class NodeIntBaseIndex(BaseInt, INodeIndex, IInt, ABC):
    pass

class NodeMainBaseIndex(NodeIntBaseIndex, ABC):

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

    def find_in_node(self, node: INode):
        item, _ = self._inner_getitem(node, self.as_int)
        return Optional(item) if item is not None else Optional()

    def replace_in_target(self, target_node: INode, new_node: INode):
        new_target, _ = self._replace(target_node, new_node, self.as_int)
        return Optional(new_target) if new_target is not None else Optional()

    def ancestors(self, node: INode) -> tuple[INode, ...]:
        ancestors, _ = self._ancestors(node, self.as_int)
        return tuple(ancestors)

class NodeMainIndex(NodeMainBaseIndex, IInstantiable):
    pass

class NodeArgBaseIndex(NodeIntBaseIndex, ABC):

    def find_in_node(self, node: INode) -> IOptional[INode]:
        index = self.as_int
        if not isinstance(node, InheritableNode):
            return Optional()
        args = node.args
        if 0 < index <= len(args):
            return Optional(args[index - 1])
        return Optional()

    def replace_in_target(self, target_node: T, new_node: INode) -> IOptional[T]:
        if not isinstance(target_node, InheritableNode):
            return Optional()
        index = self.as_int
        args = target_node.args
        if 0 < index <= len(args):
            new_target = target_node.func(*[
                (new_node if i == index - 1 else arg)
                for i, arg in enumerate(args)
            ])
            assert isinstance(new_target, type(target_node))
            return Optional(typing.cast(T, new_target))
        return Optional()

    def remove_in_target(self, target_node: T) -> Optional[T]:
        if not isinstance(target_node, InheritableNode):
            return Optional()
        index = self.as_int
        args = target_node.args
        if 0 < index <= len(args):
            new_args = [
                arg
                for i, arg in enumerate(args)
                if i != index - 1
            ]
            new_target = target_node.func(*new_args)
            assert isinstance(new_target, type(target_node))
            return Optional(typing.cast(T, new_target))
        return Optional()

class NodeArgIndex(NodeArgBaseIndex, IInstantiable):
    pass

###########################################################
####################### ITEMS GROUP #######################
###########################################################

class BaseGroup(InheritableNode, IGroup[T], IDefault, typing.Generic[T], ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(RestTypeGroup(TypeNode(cls.item_type())))

    @property
    def args(self) -> tuple[T, ...]:
        return typing.cast(tuple[T, ...], self._args)

    @property
    def as_tuple(self) -> tuple[T, ...]:
        return self.args

    @classmethod
    def from_items(cls, items: typing.Sequence[T]) -> typing.Self:
        return cls(*items)

    def amount(self) -> int:
        return len(self.args)

    def amount_node(self) -> Integer:
        return Integer(self.amount())

    def strict_validate(self):
        super().strict_validate()
        for arg in self.args:
            t = self.item_type()
            origin = typing.get_origin(t)
            t = origin if origin is not None else t
            assert isinstance(arg, t), f'{type(arg)} != {t}'

class DefaultGroup(BaseGroup[INode], IInstantiable):

    @classmethod
    def item_type(cls):
        return INode

class BaseIntGroup(BaseGroup[IInt], ABC):

    @classmethod
    def item_type(cls):
        return IInt

    @classmethod
    def from_ints(cls, indices: typing.Sequence[int]) -> typing.Self:
        return cls(*[Integer(i) for i in indices])

class IntGroup(BaseIntGroup, IInstantiable):
    pass

class NestedArgIndexGroup(BaseIntGroup, IInstantiable):

    def apply(self, node: BaseNode) -> BaseNode:
        args_indices = [arg.as_int for arg in self.args]
        return node.nested_arg(tuple(args_indices)).apply()

class BaseOptionalValueGroup(BaseGroup[IOptional[T]], IFromInt, typing.Generic[T], ABC):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(*([Optional()]*value))

    @classmethod
    def item_type(cls):
        return Optional[T]

    @classmethod
    def from_optional_items(cls, items: typing.Sequence[T | None]) -> typing.Self:
        return cls(*[Optional.with_value(item) for item in items])

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

    def new_amount(self, amount: int) -> typing.Self:
        items = self.as_tuple
        if amount < len(items):
            return self.from_items(list(items)[:amount])
        elif amount > len(items):
            diff = amount - len(items)
            new_items = list(items) + [Optional()] * diff
            return self.from_items(new_items)
        return self

class OptionalValueGroup(BaseOptionalValueGroup[T], IInstantiable, typing.Generic[T]):
    pass

###########################################################
####################### TYPE GROUPS #######################
###########################################################

class IBaseTypeGroup(INode, ABC):
    pass

class CountableTypeGroup(
    BaseGroup[IType],
    IBaseTypeGroup,
    IFromInt,
    IInstantiable,
    typing.Generic[T],
):

    @classmethod
    def item_type(cls) -> type[IType]:
        return IType

    @classmethod
    def from_types(cls, types: typing.Sequence[type[INode]]) -> typing.Self:
        return cls(*[TypeNode(t) for t in types])

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(*[UnknownType()] * value)

class SingleValueTypeGroup(
    InheritableNode,
    IBaseTypeGroup,
    ISingleChild[IType],
    ABC,
):

    idx_type_node = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IType,
        ]))

    @property
    def type_node(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type_node)

    @property
    def child(self) -> IType:
        return self.type_node.apply().cast(IType)


class RestTypeGroup(SingleValueTypeGroup, IInstantiable):
    pass

class OptionalTypeGroup(SingleValueTypeGroup, IInstantiable):
    pass

class ExtendedTypeGroup(
    InheritableNode,
    IDefault,
    IFromInt,
    IFromSingleNode[IOptional[IInt]],
    IInstantiable,
):

    idx_group = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBaseTypeGroup,
        ]))

    @property
    def group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_group)

    def _valid(self, values: BaseGroup) -> tuple[bool | None, IOptional[IExceptionInfo]]:
        group = self.group.apply().cast(IBaseTypeGroup)
        args = values.args

        if isinstance(group, CountableTypeGroup):
            same_size = Eq(Integer(len(args)), Integer(len(group.args)))
            if not same_size.as_bool:
                return False, Optional(same_size.to_exception_info())
            for i, arg in enumerate(args):
                t_arg = group.args[i]
                if not isinstance(arg, INode):
                    return False, Optional(TypeExceptionInfo(t_arg, values))
                valid = t_arg.valid(arg)
                if not valid:
                    return valid, Optional(TypeExceptionInfo(t_arg, values))
        elif isinstance(group, SingleValueTypeGroup):
            t_arg = group.type_node.apply().cast(IType)
            for arg in args:
                if not isinstance(arg, INode):
                    return False, Optional(TypeExceptionInfo(t_arg, values))
                valid = t_arg.valid(arg)
                if not valid:
                    return valid, Optional(TypeExceptionInfo(t_arg, values))
        else:
            return False, Optional(TypeExceptionInfo(TypeNode(IBaseTypeGroup), group))
        return True, Optional()

    def valid(self, values: BaseGroup) -> bool | None:
        valid, _ = self._valid(values)
        return valid

    def validate_values(self, values: BaseGroup):
        valid, exception_opt = self._valid(values)
        if valid is False:
            exception = exception_opt.value_or_raise
            raise InvalidNodeException(exception)

    @classmethod
    def create(cls) -> typing.Self:
        return cls.rest()

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(CountableTypeGroup.from_int(value))

    @classmethod
    def with_node(cls, node: IOptional[INT]) -> typing.Self:
        value = node.value.as_int if node.value is not None else None
        if value is None:
            return cls.rest()
        return cls.from_int(value)

    @classmethod
    def rest(cls, type_node: IType = UnknownType()) -> typing.Self:
        return cls(RestTypeGroup(type_node))

    def new_amount(self, amount: int) -> typing.Self:
        group = self.group.apply().cast(IBaseTypeGroup)
        if not isinstance(group, CountableTypeGroup):
            return self.with_node(Optional(Integer(amount)))
        items = group.as_tuple
        if amount == len(items):
            return self
        return self.func(
            CountableTypeGroup.from_items([
                (items[i] if i < len(items) else UnknownType())
                for i in range(amount)
            ]))

###########################################################
########################## TYPES ##########################
###########################################################

class UnionType(InheritableNode, IType, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IType.as_type())

    def accepted_by(self, outer_type: IType) -> bool | None:
        return IntersectionType(*self.args).accepts(outer_type)

    def accepts(self, inner_type: IType) -> bool | None:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        assert len(args) == len(self.args)
        items = [t.accepts(inner_type) for t in args]
        if any([item is True for item in items]):
            return True
        if all([item is False for item in items]):
            return False
        return None

    def valid(self, instance: INode) -> bool | None:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        assert len(args) == len(self.args)
        items = [t.valid(instance) for t in args]
        if any([item is True for item in items]):
            return True
        if all([item is False for item in items]):
            return False
        return None

class IntersectionType(InheritableNode, IType, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IType.as_type())

    def accepted_by(self, outer_type: IType) -> bool | None:
        return UnionType(*self.args).accepts(outer_type)

    def accepts(self, inner_type: IType) -> bool | None:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        assert len(args) == len(self.args)
        items = [t.accepts(inner_type) for t in args]
        if any([item is False for item in items]):
            return False
        if all([item is True for item in items]):
            return True
        return None

    def valid(self, instance: INode) -> bool | None:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        assert len(args) == len(self.args)
        items = [t.valid(instance) for t in args]
        if any([item is False for item in items]):
            return False
        if all([item is True for item in items]):
            return True
        return None

class CompositeType(InheritableNode, IType, IInstantiable):

    idx_type = 1
    idx_type_args = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup(
            TypeNode(TypeNode),
            UnionType(ExtendedTypeGroup.as_type(), Void.as_type()),
        ))

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def type_args(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type_args)

    def accepted_by(self, outer_type: IType) -> bool | None:
        if isinstance(outer_type, CompositeType):
            return outer_type.accepts(self)
        if isinstance(outer_type, TypeNode):
            t = self.inner_arg(self.idx_type).apply().as_node.cast(IType)
            return t.accepted_by(outer_type)
        return outer_type.accepts(self)

    def accepts(self, inner_type: IType) -> bool | None:
        if isinstance(inner_type, CompositeType):
            t1 = self.inner_arg(self.idx_type).apply().as_node.cast(IType)
            t2 = inner_type.inner_arg(self.idx_type).apply().as_node.cast(IType)
            if t1 != t2:
                return False
            args1 = self.inner_arg(self.idx_type_args).apply()
            if args1 == Void():
                return True
            assert isinstance(args1, ExtendedTypeGroup)
            args2 = inner_type.inner_arg(self.idx_type_args).apply()
            if not isinstance(args2, ExtendedTypeGroup):
                return False
            args1_group = args1.group.apply().cast(IBaseTypeGroup)
            args2_group = args2.group.apply().cast(IBaseTypeGroup)
            if isinstance(args1_group, CountableTypeGroup):
                if not isinstance(args2_group, CountableTypeGroup):
                    return False
                if len(args1_group.args) != len(args2_group.args):
                    return False
                for t1, t2 in zip(args1_group.args, args2_group.args):
                    if not t1.accepts(t2):
                        return False
                return True
            assert isinstance(args1_group, SingleValueTypeGroup)
            if isinstance(args2_group, CountableTypeGroup):
                for t2 in args2_group.args:
                    if not args1_group.child.accepts(t2):
                        return False
            else:
                assert isinstance(args2_group, SingleValueTypeGroup)
                return args1_group.child.accepts(args2_group.child)
            return True
        if isinstance(inner_type, TypeNode):
            t = self.inner_arg(self.idx_type).apply().as_node.cast(IType)
            return t.accepts(inner_type)
        return inner_type.accepted_by(self)

    def valid(self, instance: INode) -> bool | None:
        t = self.inner_arg(self.idx_type).apply().as_node.cast(IType)
        if not t.valid(instance):
            return False
        args = self.inner_arg(self.idx_type_args).apply()
        if args == Void():
            return True
        assert isinstance(args, ExtendedTypeGroup)
        i_args = instance.as_node.cast(InheritableNode).args
        return args.valid(DefaultGroup(*i_args))

###########################################################
###################### FUNCTION NODE ######################
###########################################################

class ScopedFunctionBase(InheritableNode, IFunction, IScope, ABC):

    idx_param_type_group = 1
    idx_expr = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ExtendedTypeGroup,
            INode,
        ]))

    @property
    def param_type_group(self) -> TmpInnerArg:
        return self.as_node.inner_arg(self.idx_param_type_group)

    @property
    def expr(self) -> TmpInnerArg:
        return self.as_node.inner_arg(self.idx_expr)

    def with_arg_group(self, group: BaseGroup, info: RunInfo):
        if info.is_future():
            scope_data: ScopeDataParamBaseItemGroup = ScopeDataFutureParamItemGroup()
        else:
            info, node_aux = group.run(info).as_tuple
            new_group = node_aux.as_node.cast(BaseGroup)
            param_type_group = self.param_type_group.apply().cast(ExtendedTypeGroup)
            param_type_group.validate_values(new_group)
            scope_data = ScopeDataParamItemGroup.from_optional_items(new_group.as_tuple)
            scope_data.strict_validate()
        new_info = (
            info.create()
            if isinstance(self, IOpaqueScope)
            else info
        ).add_scope(scope_data)

        if isinstance(self, IOpaqueScope):
            try:
                result = self.expr.apply().run(new_info)
            except NodeReturnException as e:
                result = e.result
            _, node = result.as_tuple
            return info.to_result(node)
        else:
            new_info, node = self.expr.apply().run(new_info).as_tuple
            new_info = new_info.with_scopes(info)
            return new_info.to_result(node)

    def prepare_expr(self, info: RunInfo) -> RunInfoResult:
        raise NotImplementedError

    def _run(self, info: RunInfo):
        self.validate()

        info, node_aux = self.param_type_group.apply().run(info).as_tuple
        type_group = node_aux.as_node.cast(ExtendedTypeGroup)

        info, expr = self.prepare_expr(info).as_tuple

        return info.to_result(self.func(type_group, expr))

class FunctionExpr(
    ScopedFunctionBase,
    IFromSingleNode[T],
    IOpaqueScope,
    IInstantiable,
    typing.Generic[T],
):

    @classmethod
    def with_node(cls, node: T) -> typing.Self:
        return cls.new(ExtendedTypeGroup.rest(), node)

    def prepare_expr(self, info: RunInfo):
        return info.to_result(self.expr.apply())

class FunctionWrapper(
    ScopedFunctionBase,
    IFromSingleNode[T],
    IInnerScope,
    IInstantiable,
    typing.Generic[T],
):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ExtendedTypeGroup,
            FunctionCall,
        ]))

    @classmethod
    def with_node(cls, node: T) -> typing.Self:
        return cls.new(ExtendedTypeGroup.rest(), node)

    def prepare_expr(self, info: RunInfo):
        new_info = info.add_scope(ScopeDataFutureParamItemGroup())
        return self.expr.apply().run(new_info)

###########################################################
################### BASIC BOOLEAN NODES ###################
###########################################################

class BaseIntBoolean(BaseInt, IBoolean, IDefault, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls(0)

    @classmethod
    def from_bool(cls, value: bool) -> typing.Self:
        assert value in (True, False)
        return cls.from_int(1 if value else 0)

    @classmethod
    def create_true(cls) -> typing.Self:
        return cls(1)

    @property
    def as_bool(self) -> bool | None:
        if self.as_int == 0:
            return False
        if self.as_int == 1:
            return True
        return None

class IntBoolean(BaseIntBoolean, IInstantiable):
    pass

class RunnableBoolean(InheritableNode, IBoolean, ABC):

    def _run(self, info: RunInfo):
        result = super()._run(info)
        info, node_aux = result.as_tuple
        if info.is_future():
            return result

        node = node_aux.as_node.cast(self.__class__)
        value = node.func(*node.args).strict_bool

        return info.to_result(IntBoolean.from_bool(value))

class BooleanWrapper(
    RunnableBoolean,
    ISingleChild[IBoolean],
    ABC,
):

    idx_value = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBoolean,
        ]))

    @property
    def raw_child(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    @property
    def child(self) -> IBoolean:
        return self.raw_child.apply().cast(IBoolean)

    @property
    def as_bool(self) -> bool | None:
        return self.child.as_bool

###########################################################
######################## EXCEPTION ########################
###########################################################

class NodeReturnException(Exception):

    def __init__(self, result: RunInfoResult):
        super().__init__(result)

    @property
    def result(self) -> RunInfoResult:
        result = self.args[0]
        assert isinstance(result, RunInfoResult)
        return result

class IExceptionInfo(INode, ABC):

    def as_exception(self):
        return InvalidNodeException(self)

    def add_stack(self, node: INode, run_info: RunInfo) -> StackExceptionInfo:
        if isinstance(self, StackExceptionInfo):
            return self._add_stack(node, run_info)

        return StackExceptionInfo(
            StackExceptionInfoGroup(StackExceptionInfoItem(node, run_info)),
            self,
        )

class BooleanExceptionInfo(
    BooleanWrapper,
    IExceptionInfo,
    IInstantiable,
):
    pass

class TypeExceptionInfo(
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_type = 1
    idx_node = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IType,
            INode,
        ]))

    @property
    def type(self):
        return self.inner_arg(self.idx_type)

    @property
    def node(self):
        return self.inner_arg(self.idx_node)

class IStackExceptionInfoItem(INode, ABC):
    pass

class StackExceptionInfoItem(
    InheritableNode,
    IStackExceptionInfoItem,
    IInstantiable,
):

    idx_node = 1
    idx_run_info = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            INode,
            RunInfo,
        ]))

    @property
    def node(self):
        return self.inner_arg(self.idx_node)

    @property
    def run_info(self):
        return self.inner_arg(self.idx_run_info)

    @classmethod
    def with_args(cls, node: INode, run_info: RunInfo) -> typing.Self:
        return cls(node, run_info)

class StackNodeArg(
    InheritableNode,
    IInstantiable,
    IFromSingleNode[INode],
):

    idx_full_arg = 1
    idx_arg_type = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            Optional[INode],
            Optional[TypeNode],
        ]))

    @property
    def full_arg(self):
        return self.inner_arg(self.idx_full_arg)

    @property
    def arg_type(self):
        return self.inner_arg(self.idx_arg_type)

    @classmethod
    def with_full_arg(cls, arg: INode) -> typing.Self:
        return cls(Optional(arg), Optional())

    @classmethod
    def with_arg_type(cls, arg_type: IType) -> typing.Self:
        return cls(Optional(), Optional(arg_type))

    @classmethod
    def with_node(cls, node: INode) -> typing.Self:
        return (
            cls.with_full_arg(node)
            if (
                isinstance(node, ISpecialValue)
                or isinstance(node, BaseIntGroup)
            )
            else cls.with_arg_type(node.as_node.as_type())
        )

class StackExceptionInfoSimplifiedItem(
    InheritableNode,
    IStackExceptionInfoItem,
    IFromSingleNode[InheritableNode],
    IInstantiable,
):

    idx_node = 1
    idx_node_args = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            INode,
            BaseGroup[INode],
        ]))

    @property
    def node(self):
        return self.inner_arg(self.idx_node)

    @property
    def node_args(self):
        return self.inner_arg(self.idx_node_args)

    @classmethod
    def with_node(cls, node: InheritableNode) -> typing.Self:
        return cls(
            node.as_node.as_type(),
            DefaultGroup.from_items([
                StackNodeArg.with_node(arg)
                for arg in node.args
            ]),
        )

class StackExceptionInfoGroup(
    BaseGroup[IStackExceptionInfoItem],
    IInstantiable,
):

    @classmethod
    def item_type(cls):
        return IStackExceptionInfoItem

class StackExceptionInfo(
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_stack_group = 1
    idx_cause = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            StackExceptionInfoGroup,
            IExceptionInfo,
        ]))

    @property
    def stack_group(self):
        return self.inner_arg(self.idx_stack_group)

    @property
    def cause(self):
        return self.inner_arg(self.idx_cause)

    def _add_stack(self, node: INode, run_info: RunInfo) -> typing.Self:
        stack_group = self.stack_group.apply().cast(StackExceptionInfoGroup)
        new_item = (
            StackExceptionInfoItem.with_args(node=node, run_info=run_info)
            if len(stack_group.args) == 0 or not isinstance(node, InheritableNode)
            else StackExceptionInfoSimplifiedItem.with_node(node)
        )
        new_stack_group = stack_group.func(*stack_group.args, new_item)
        result = self.func(new_stack_group, self.cause.apply())
        result.validate()
        return result

class InvalidNodeException(Exception):

    def __init__(self, info: IExceptionInfo):
        super().__init__(info)

    @property
    def info(self) -> IExceptionInfo:
        info = self.args[0]
        return typing.cast(IExceptionInfo, info)

###########################################################
######################## EXCEPTION ########################
###########################################################

class ExceptionInfoWrapper(
    InheritableNode,
    ISingleChild[IExceptionInfo],
    IExceptionInfo,
    IInstantiable,
):

    idx_info = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IExceptionInfo,
        ]))

    @property
    def info(self):
        return self.inner_arg(self.idx_info)

    @property
    def child(self) -> IExceptionInfo:
        return self.info.apply().cast(IExceptionInfo)

###########################################################
################### CORE BOOLEAN NODES ####################
###########################################################

class Not(BooleanWrapper, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        child = self.raw_child.apply()
        if not isinstance(child, IBoolean):
            return None
        if child.as_bool is None:
            return None
        return not child.as_bool

class SingleOptionalBooleanChildWrapper(
    RunnableBoolean,
    ISingleOptionalChild[T],
    typing.Generic[T],
    ABC,
):

    idx_value = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IOptional[T],
        ]))

    @property
    def raw_child(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    @property
    def child(self) -> IOptional[T]:
        return self.raw_child.apply().cast(IOptional[T])

class IsEmpty(SingleOptionalBooleanChildWrapper[INode], IInstantiable):

    def _raise_if_empty(self) -> INode:
        optional = self.inner_arg(self.idx_value).apply()
        if not isinstance(optional, IOptional):
            raise self.to_exception()
        value = optional.value
        if value is None:
            raise self.to_exception()
        return value

    @property
    def value_or_raise(self) -> INode:
        return self._raise_if_empty()

    def raise_if_empty(self):
        self._raise_if_empty()

    @property
    def as_bool(self) -> bool | None:
        value = self.inner_arg(self.idx_value).apply()
        if not isinstance(value, IOptional):
            return None
        return value.value is None

class IsInstance(RunnableBoolean, typing.Generic[T], IInstantiable):

    idx_instance = 1
    idx_type = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types((
            INode,
            TypeNode[T],
        )))

    @property
    def instance(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_instance)

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def as_bool(self) -> bool | None:
        instance = self.instance.apply()
        assert isinstance(instance, INode)
        t = self.type.apply()
        assert isinstance(t, TypeNode)
        valid = t.valid(instance)
        return valid

    @property
    def as_type_or_raise(self) -> T:
        if not self.as_bool:
            raise self.to_exception()
        return typing.cast(T, self.type.apply())

    @classmethod
    def with_args(cls, instance: INode, t: typing.Type[T]) -> IsInstance[T]:
        return typing.cast(IsInstance[T], cls(instance, TypeNode(t)))

    @classmethod
    def assert_type(cls, instance: INode, t: typing.Type[T]) -> T:
        return typing.cast(T, cls(instance, TypeNode(t)).as_type_or_raise)


class Eq(RunnableBoolean, IInstantiable):

    idx_left = 1
    idx_right = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types((
            INode,
            INode,
        )))

    @property
    def left(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_left)

    @property
    def right(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_right)

    @property
    def as_bool(self) -> bool | None:
        left = self.left.apply()
        right = self.right.apply()
        return left == right

    @classmethod
    def from_ints(cls, left: int, right: int) -> Eq:
        return cls(Integer(left), Integer(right))

class MultiArgBooleanNode(RunnableBoolean, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IBoolean.as_type())

class DoubleIntBooleanNode(RunnableBoolean, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IInt,
            IInt,
        ]))

    @classmethod
    def with_args(cls, value_1: int, value_2: int) -> DoubleIntBooleanNode:
        return cls(Integer(value_1), Integer(value_2))

class And(MultiArgBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        args = self.args
        has_none = False
        for arg in args:
            if not isinstance(arg, IBoolean):
                has_none = True
            elif arg.as_bool is None:
                has_none = True
            elif arg.as_bool is False:
                return False
        return None if has_none else True

    def _run(self, info: RunInfo) -> RunInfoResult:
        if info.is_future():
            return super()._run(info)
        run_args: list[INode] = []
        for arg in self.args:
            info, run_arg = arg.as_node.run(info).as_tuple
            if isinstance(run_arg, IBoolean):
                val_1 = run_arg.as_bool
                if val_1 is False:
                    return info.to_result(IBoolean.false())
            run_args.append(run_arg)
        for run_arg in run_args:
            assert isinstance(run_arg, IBoolean)
            run_arg.raise_on_undefined()
        return info.to_result(IBoolean.true())

class Or(MultiArgBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        args = self.args
        has_none = False
        for arg in args:
            if not isinstance(arg, IBoolean):
                has_none = True
            elif arg.as_bool is None:
                has_none = True
            elif arg.as_bool is True:
                return True
        return None if has_none else False

    def _run(self, info: RunInfo) -> RunInfoResult:
        if info.is_future():
            return super()._run(info)
        run_args: list[INode] = []
        for arg in self.args:
            info, run_arg = arg.as_node.run(info).as_tuple
            if isinstance(run_arg, IBoolean):
                val_1 = run_arg.as_bool
                if val_1 is True:
                    return info.to_result(IBoolean.true())
            run_args.append(run_arg)
        for run_arg in run_args:
            assert isinstance(run_arg, IBoolean)
            run_arg.raise_on_undefined()
        return info.to_result(IBoolean.false())

class GreaterThan(DoubleIntBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        args = self.args
        assert len(args) == 2
        a, b = args
        if not isinstance(a, IInt) or not isinstance(b, IInt):
            return None
        return a.as_int > b.as_int

class LessThan(DoubleIntBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        args = self.args
        assert len(args) == 2
        a, b = args
        if not isinstance(a, IInt) or not isinstance(b, IInt):
            return None
        return a.as_int < b.as_int

class IsInsideRange(RunnableBoolean, IInstantiable):

    idx_value = 1
    idx_min_value = 2
    idx_max_value = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IInt,
            IInt,
            IInt,
        ]))

    @classmethod
    def from_raw(cls, value: int | IInt, min_value: int, max_value: int) -> typing.Self:
        v = Integer(value) if isinstance(value, int) else value
        return cls(v, Integer(min_value), Integer(max_value))

    @property
    def raw_value(self):
        return self.inner_arg(self.idx_value)

    @property
    def min_value(self):
        return self.inner_arg(self.idx_min_value)

    @property
    def max_value(self):
        return self.inner_arg(self.idx_max_value)

    @property
    def as_bool(self) -> bool | None:
        value = self.raw_value.apply()
        if not isinstance(value, IInt):
            return None
        min_value = self.min_value.apply()
        if not isinstance(min_value, IInt):
            return None
        max_value = self.max_value.apply()
        if not isinstance(max_value, IInt):
            return None
        return min_value.as_int <= value.as_int <= max_value.as_int

###########################################################
################### CONTROL FLOW NODES ####################
###########################################################

class ScopeDataPlaceholderItemGroup(BaseOptionalValueGroup[Placeholder], ABC):

    @classmethod
    def item_inner_type(cls) -> type[Placeholder]:
        raise NotImplementedError

    @classmethod
    def is_dynamic(cls) -> IBoolean:
        raise NotImplementedError

    def add_item(self, index: Integer, value: INode) -> typing.Self:
        self.is_dynamic().raise_on_not_true()
        items = list(self.as_tuple)
        if index.as_int > len(items):
            items = items + [Optional()] * (index.as_int - len(items))
        current = items[index.as_int - 1]
        current.is_empty().raise_on_not_true()
        items[index.as_int - 1] = Optional(value)
        return self.func(*items)

class ScopeDataParamBaseItemGroup(ScopeDataPlaceholderItemGroup, ABC):

    @classmethod
    def item_inner_type(cls):
        return Param

    @classmethod
    def is_dynamic(cls) -> IBoolean:
        return IBoolean.false()

class IScopeDataActualItemGroup(INode, ABC):
    pass

class IScopeDataFutureItemGroup(INode, ABC):
    pass

class ScopeDataParamItemGroup(
    ScopeDataParamBaseItemGroup,
    IScopeDataActualItemGroup,
    IInstantiable,
):
    pass

class ScopeDataFutureParamItemGroup(
    ScopeDataParamBaseItemGroup,
    IScopeDataFutureItemGroup,
    IInstantiable,
):
    pass

class ScopeDataVarBaseItemGroup(ScopeDataPlaceholderItemGroup, ABC):

    @classmethod
    def item_inner_type(cls):
        return Var

    @classmethod
    def is_dynamic(cls) -> IBoolean:
        return IBoolean.true()

class ScopeDataVarItemGroup(ScopeDataVarBaseItemGroup, IScopeDataActualItemGroup, IInstantiable):
    pass

class ScopeDataFutureVarItemGroup(
    ScopeDataVarBaseItemGroup,
    IScopeDataFutureItemGroup,
    IInstantiable,
):
    pass

class ScopeDataGroup(BaseGroup[ScopeDataPlaceholderItemGroup], IInstantiable):

    @classmethod
    def item_type(cls):
        return ScopeDataPlaceholderItemGroup

    def is_future(self):
        items = self.as_tuple
        return any([isinstance(item, IScopeDataFutureItemGroup) for item in items])

    def add_item(self, item: ScopeDataPlaceholderItemGroup) -> typing.Self:
        items = list(self.as_tuple)
        items.append(item)
        return self.func(*items)

class RunInfo(InheritableNode, IDefault, IInstantiable):

    idx_scope_data_group = 1
    idx_return_after_scope = 2

    @classmethod
    def create(cls) -> typing.Self:
        return cls(ScopeDataGroup(), Optional())

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            ScopeDataGroup,
            Optional[RunInfoScopeDataIndex],
        ]))

    @property
    def scope_data_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_scope_data_group)

    @property
    def return_after_scope(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_return_after_scope)

    def add_scope(self, item: ScopeDataPlaceholderItemGroup) -> typing.Self:
        group = self.scope_data_group.apply().cast(ScopeDataGroup)
        new_group = group.add_item(item)
        return self.with_new_args(
            scope_data_group=new_group,
            return_after_scope=Optional())

    def with_new_args(
        self,
        scope_data_group: ScopeDataGroup | None = None,
        return_after_scope: Optional[RunInfoScopeDataIndex] | None = None,
    ) -> typing.Self:
        scope_data_group = (
            scope_data_group
            if scope_data_group is not None
            else self.scope_data_group.apply().cast(ScopeDataGroup))
        return_after_scope = (
            return_after_scope
            if return_after_scope is not None
            else self.return_after_scope.apply().cast(Optional[RunInfoScopeDataIndex]))
        return self.func(scope_data_group, return_after_scope)

    def add_scope_var(
        self,
        scope_index: RunInfoScopeDataIndex,
        item_index: Integer,
        value: INode,
    ) -> RunInfo:
        assert isinstance(scope_index, RunInfoScopeDataIndex)
        assert isinstance(item_index, Integer)

        item = scope_index.find_in_outer_node(self).value_or_raise
        assert isinstance(item, ScopeDataVarItemGroup)

        new_item = item.add_item(item_index, value)
        result = scope_index.replace_in_outer_target(self, new_item).value_or_raise

        group_1 = self.scope_data_group.apply().cast(ScopeDataGroup).as_tuple
        group_2 = result.scope_data_group.apply().cast(ScopeDataGroup).as_tuple
        Eq(Integer(len(group_1)), Integer(len(group_2))).raise_on_not_true()

        return result

    def is_future(self) -> IBoolean:
        group = self.scope_data_group.apply().cast(ScopeDataGroup)
        return group.is_future()

    def to_result(self, value: INode) -> RunInfoResult:
        return RunInfoResult(self, value)

    def with_scopes(self, info_base: RunInfo) -> typing.Self:
        base_amount = info_base.scope_data_group.apply().cast(ScopeDataGroup).amount()
        group = self.scope_data_group.apply().cast(ScopeDataGroup).as_tuple
        Not(LessThan(Integer(len(group)), Integer(base_amount))).raise_on_not_true()
        new_group = group[:base_amount]
        return_after_scope = self.return_after_scope.apply().cast(Optional[RunInfoScopeDataIndex])
        return_after_val = return_after_scope.value
        if return_after_val is not None:
            assert isinstance(return_after_val, RunInfoScopeDataIndex)
            ret_index = return_after_val.as_int
            if ret_index > base_amount:
                return_after_scope = Optional()
        return self.with_new_args(
            scope_data_group=ScopeDataGroup(*new_group),
            return_after_scope=return_after_scope)

    def must_return(self) -> bool:
        return_after_scope = self.return_after_scope.apply().cast(Optional[RunInfoScopeDataIndex])
        return_after_val = return_after_scope.value
        return return_after_val is not None

class RunInfoResult(InheritableNode, IInstantiable):

    idx_new_info = 1
    idx_return_value = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            RunInfo,
            INode,
        ]))

    @property
    def new_info(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_new_info)

    @property
    def return_value(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_return_value)

    @property
    def as_tuple(self):
        new_info = self.new_info.apply().cast(RunInfo)
        return_value = self.return_value.apply().cast(INode)
        return new_info, return_value

class ControlFlowBaseNode(InheritableNode, ABC):

    def _run(self, info: RunInfo) -> RunInfoResult:
        self.validate()
        if info.is_future():
            args: list[INode] = []
            for arg in self.args:
                info, node_aux = arg.as_node.run(info).as_tuple
                args.append(node_aux)
            return info.to_result(self.func(*args))
        return self._run_control(info)

    def _run_control(self, info: RunInfo) -> RunInfoResult:
        raise NotImplementedError(self.__class__)

class FunctionCall(ControlFlowBaseNode, IInstantiable):

    idx_function = 1
    idx_arg_group = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IFunction,
            BaseGroup,
        ]))

    @property
    def function(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_function)

    @property
    def arg_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_group)

    @classmethod
    def define(cls, fn: IFunction, args: BaseGroup) -> INode:
        return cls(fn, args)

    def _run_control(self, info: RunInfo):
        info, node_aux = self.function.apply().run(info).as_tuple
        fn = node_aux.as_node.cast(IFunction)

        info, node_aux = self.arg_group.apply().run(info).as_tuple
        arg_group = node_aux.as_node.cast(BaseGroup)

        return fn.with_arg_group(group=arg_group, info=info)

class If(ControlFlowBaseNode, IInstantiable):

    idx_condition = 1
    idx_true_expr = 2
    idx_false_expr = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBoolean,
            INode,
            INode,
        ]))

    @property
    def condition(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_condition)

    @property
    def true_expr(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_true_expr)

    @property
    def false_expr(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_false_expr)

    def _run_control(self, info: RunInfo):
        info, node_aux = self.condition.apply().run(info).as_tuple
        condition = node_aux.as_node.cast(IBoolean)

        flag = condition.strict_bool
        if flag:
            return self.true_expr.apply().run(info)
        return self.false_expr.apply().run(info)

class LoopGuard(InheritableNode, IInstantiable):

    idx_condition = 1
    idx_result = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IBoolean,
            INode,
        ]))
    @property
    def condition(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_condition)

    @property
    def result(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_result)

    @classmethod
    def with_args(cls, condition: IBoolean, result: INode) -> LoopGuard:
        return cls(condition, result)

class Loop(ControlFlowBaseNode, IFromSingleNode[IFunction], IInstantiable):

    idx_callback = 1
    idx_initial_data = 2

    @classmethod
    def with_node(cls, node: IFunction) -> typing.Self:
        return cls(node, Optional())

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IFunction,
            Optional[INode],
        ]))

    @property
    def callback(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_callback)

    @property
    def initial_data(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_initial_data)

    def _run_control(self, info: RunInfo):
        info, node_aux = self.callback.apply().run(info).as_tuple
        callback = node_aux.as_node.cast(IFunction)

        info, node_aux = self.initial_data.apply().run(info).as_tuple
        initial_data = node_aux.as_node.cast(Optional[INode])

        condition = True
        data = initial_data
        idx = 0
        while condition:
            idx += 1
            info, node_aux = FunctionCall(callback, DefaultGroup(data)).run(info).as_tuple
            result = node_aux.as_node.cast(LoopGuard)

            info, node_aux = result.condition.apply().run(info).as_tuple
            cond_node = node_aux.as_node.cast(IBoolean)

            info, new_data = result.result.apply().run(info).as_tuple

            data = Optional(new_data)
            condition = cond_node.strict_bool
        return info.to_result(data)

class InstructionGroup(ControlFlowBaseNode, IInnerScope, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(RestTypeGroup(TypeNode(IRunnable)))

    def _run_control(self, info: RunInfo):
        scope_data = (
            ScopeDataFutureVarItemGroup()
            if info.is_future()
            else ScopeDataVarItemGroup()
        )
        info = info.add_scope(scope_data)

        for arg in self.args:
            info, _ = arg.as_node.run(info).as_tuple
        return info.to_result(Void())

class Assign(ControlFlowBaseNode, IInstantiable):

    idx_var_index = 1
    idx_value = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            Integer,
            INode,
        ]))

    @property
    def var_index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_var_index)

    @property
    def value(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    def _run_control(self, info: RunInfo):
        info, node_aux = self.var_index.apply().run(info).as_tuple
        var_index = node_aux.as_node.cast(Integer)

        info, node_aux = self.value.apply().run(info).as_tuple
        value = node_aux

        info, node_aux = NearParentScope.create().run(info).as_tuple
        scope_index = node_aux.as_node.cast(RunInfoScopeDataIndex)

        info = info.add_scope_var(
            scope_index=scope_index,
            item_index=var_index,
            value=value)

        return info.to_result(value)

class Return(ControlFlowBaseNode, IFromSingleNode[INode], IInstantiable):

    idx_parent_scope = 1
    idx_value = 2

    @classmethod
    def with_node(cls, node: INode) -> typing.Self:
        return cls(FarParentScope.create(), node)

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            RunInfoScopeDataIndex,
            INode,
        ]))

    @property
    def parent_scope(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_parent_scope)

    @property
    def value(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    def _run_control(self, info: RunInfo):
        info, node_aux = self.parent_scope.apply().run(info).as_tuple
        scope_index = node_aux.as_node.cast(RunInfoScopeDataIndex)

        info, value = self.value.apply().run(info).as_tuple

        if not info.is_future():
            info = info.with_new_args(
                return_after_scope=Optional(scope_index)
            )

        return info.to_result(value)

class InnerArg(ControlFlowBaseNode, IInstantiable):

    idx_node = 1
    idx_arg_index = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            INode,
            NodeArgIndex,
        ]))

    @property
    def node(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_node)

    @property
    def arg_index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_index)

    def _run_control(self, info: RunInfo):
        info, node_aux = self.node.apply().run(info).as_tuple
        node = node_aux.as_node.cast(INode)

        info, node_aux = self.arg_index.apply().run(info).as_tuple
        arg_index = node_aux.as_node.cast(NodeArgIndex)

        result = arg_index.find_in_node(node).value_or_raise

        return info.to_result(result)

class NestedArg(ControlFlowBaseNode, IInstantiable):

    idx_node = 1
    idx_arg_indices = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            INode,
            NestedArgIndexGroup,
        ]))

    @classmethod
    def from_raw(cls, node: INode, indices: typing.Sequence[int]) -> typing.Self:
        return cls(node, NestedArgIndexGroup.from_ints(indices))

    @property
    def node(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_node)

    @property
    def arg_indices(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_indices)

    def _run_control(self, info: RunInfo):
        info, node_aux = self.node.apply().run(info).as_tuple
        node = node_aux.as_node.cast(INode)

        info, node_aux = self.arg_indices.apply().run(info).as_tuple
        arg_indices = node_aux.as_node.cast(NestedArgIndexGroup)

        result = arg_indices.apply(node)

        return info.to_result(result)

###########################################################
##################### RUN INFO INDEX ######################
###########################################################

class IRunInfoIndex(ITypedIndex[RunInfo, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls) -> type[RunInfo]:
        return RunInfo

    @classmethod
    def item_type(cls) -> type[T]:
        raise NotImplementedError

    def find_in_outer_node(self, node: RunInfo) -> IOptional[T]:
        raise NotImplementedError

    def replace_in_outer_target(self, target: RunInfo, new_node: T) -> IOptional[RunInfo]:
        raise NotImplementedError

    def remove_in_outer_target(self, target: RunInfo) -> IOptional[RunInfo]:
        raise NotImplementedError

class RunInfoIntIndex(
    BaseInt,
    IRunInfoIndex[T],
    ITypedIntIndex[RunInfo, T],
    typing.Generic[T],
    ABC,
):
    pass

class RunInfoScopeDataIndex(RunInfoIntIndex[ScopeDataPlaceholderItemGroup], IInstantiable):

    @classmethod
    def item_type(cls):
        return ScopeDataPlaceholderItemGroup

    @classmethod
    def _outer_group(cls, run_info: RunInfo) -> ScopeDataGroup:
        return run_info.scope_data_group.apply().cast(ScopeDataGroup)

    @classmethod
    def _update(
        cls,
        target: RunInfo,
        group_opt: IOptional[ScopeDataGroup],
    ) -> IOptional[RunInfo]:
        group = group_opt.value
        if group is None:
            return Optional.create()
        assert isinstance(group, ScopeDataGroup)
        return Optional(target.with_new_args(
            scope_data_group=group,
            return_after_scope=Optional(),
        ))

    def find_in_outer_node(
        self,
        node: RunInfo,
    ) -> IOptional[ScopeDataPlaceholderItemGroup]:
        return self.find_arg(self._outer_group(node))

    def replace_in_outer_target(
        self,
        target: RunInfo,
        new_node: ScopeDataPlaceholderItemGroup,
    ) -> IOptional[RunInfo]:
        group_opt = self.replace_arg(self._outer_group(target), new_node)
        return self._update(target, group_opt)

    def remove_in_outer_target(self, target: RunInfo) -> IOptional[RunInfo]:
        group_opt = self.remove_arg(self._outer_group(target))
        return self._update(target, group_opt)

###########################################################
######################## ITERATOR #########################
###########################################################

class IIterator(INode, ABC):

    def next(self, run_info: RunInfo) -> RunInfoResult:
        result = self._next(run_info)
        _, node = result.as_tuple
        assert isinstance(node, IOptional)
        group = node.value
        if group is not None:
            assert isinstance(group, DefaultGroup)
            assert len(group.args) == 2
            new_iter, _ = group.as_tuple
            assert isinstance(new_iter, self.__class__)
        return result

    def _next(self, run_info: RunInfo) -> RunInfoResult:
        raise NotImplementedError

class GroupIterator(InheritableNode, IFromSingleNode[BaseGroup], IIterator, IInstantiable):

    idx_group = 1
    idx_index = 2

    @classmethod
    def with_node(cls, node: INode) -> typing.Self:
        return cls(node, NodeArgIndex(1))

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            BaseGroup,
            NodeArgIndex,
        ]))

    @property
    def group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_group)

    @property
    def index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_index)

    def with_new_args(
        self,
        group: BaseGroup | None = None,
        index: NodeArgIndex | None = None,
    ) -> typing.Self:
        group = group if group is not None else self.group.apply().cast(BaseGroup)
        index = index if index is not None else self.index.apply().cast(NodeArgIndex)
        return self.func(group, index)

    def _next(self, run_info: RunInfo):
        info = run_info

        info, node_aux = self.group.apply().run(info).as_tuple
        group = node_aux.as_node.cast(BaseGroup)

        info, node_aux = self.index.apply().run(info).as_tuple
        index = node_aux.as_node.cast(NodeArgIndex)

        if index.as_int > group.amount():
            return info.to_result(Optional())

        value = index.find_in_node(group).value_or_raise
        new_iter = self.with_new_args(
            group=group,
            index=NodeArgIndex(index.as_int + 1),
        )
        return info.to_result(Optional(DefaultGroup(new_iter, value)))

class Next(ControlFlowBaseNode, IInstantiable):

    idx_iter = 1

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IIterator,
        ]))

    @property
    def iter(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_iter)

    def _run_control(self, info: RunInfo):
        info, node_aux = self.iter.apply().run(info).as_tuple
        iterator = node_aux.as_node.cast(IIterator)
        return iterator.next(info)

class Add(ControlFlowBaseNode, IInstantiable):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IAdditive.as_type())

    def _run_control(self, info: RunInfo):
        node: IAdditive | None = None
        for arg in self.args:
            info, node_aux = arg.as_node.run(info).as_tuple
            if node is None:
                arg_node = node_aux.as_node.cast(IAdditive)
                node = arg_node
            else:
                info, node = node.add(node_aux, run_info=info).as_tuple
        assert node is not None
        return info.to_result(node)
