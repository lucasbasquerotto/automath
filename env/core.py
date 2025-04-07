# pylint: disable=too-many-lines
from __future__ import annotations
from abc import ABC
import typing
from utils.module_utils import type_sorter_key

###########################################################
##################### MAIN INTERFACES #####################
###########################################################

class INode(ABC):

    @classmethod
    def clear_cache(cls):
        BaseNode.clear_actual_cache()

    @classmethod
    def as_type(cls) -> TypeNode[typing.Self]:
        return TypeNode(cls)

    @classmethod
    def as_supertype(cls) -> Type:
        return Type(cls.as_type())

    @classmethod
    def protocol(cls) -> Protocol:
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

    def actual_instance(self) -> BaseNode:
        return self.as_node

    def real(self, t: typing.Type[T]) -> T:
        instance = self.as_node.actual_instance()
        return instance.cast(t)

    def cast(self, t: typing.Type[T]) -> T:
        origin = typing.get_origin(t)
        t = origin if origin is not None else t
        IsInstance.assert_type(self, t)
        assert isinstance(self, t), f'{type(self)} != {t}'
        return typing.cast(T, self)

class IInheritableNode(INode, ABC):

    @classmethod
    def new(cls, *args: INode) -> typing.Self:
        raise NotImplementedError(cls)

    @property
    def node_args(self) -> tuple[INode, ...]:
        raise NotImplementedError(self.__class__)

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

    def add(self, another: INode) -> typing.Self:
        raise NotImplementedError(self.__class__)

class IComparable(INode, ABC):

    def eq(self, another: INode) -> IBoolean:
        return IntBoolean.from_bool(self == another)

    def ne(self, another: INode) -> IBoolean:
        return Not(self.eq(another))

    def lt(self, another: INode) -> IBoolean:
        raise NotImplementedError(self.__class__)

    def le(self, another: INode) -> IBoolean:
        if self.eq(another).as_bool:
            return IBoolean.true()
        return self.lt(another)

    def gt(self, another: INode) -> IBoolean:
        raise NotImplementedError(self.__class__)

    def ge(self, another: INode) -> IBoolean:
        if self.eq(another).as_bool:
            return IBoolean.true()
        return self.gt(another)

class IFromInt(INode, ABC):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        raise NotImplementedError(cls)

class IInt(IFromInt, IAdditive, IComparable, ABC):

    @property
    def as_int(self) -> int:
        raise NotImplementedError(self.__class__)

    def add(self, another: INode):
        new_value = self.as_int + another.real(IInt).as_int
        return self.__class__.from_int(new_value)

    def lt(self, another: INode):
        new_value = self.as_int < another.real(IInt).as_int
        return IntBoolean.from_bool(new_value)

    def gt(self, another: INode):
        new_value = self.as_int > another.real(IInt).as_int
        return IntBoolean.from_bool(new_value)

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
    def item_type(cls) -> TypeNode:
        raise NotImplementedError(cls)

    @property
    def as_tuple(self) -> tuple[T, ...]:
        raise NotImplementedError(self.__class__)

    @classmethod
    def from_items(cls, items: typing.Sequence[T]) -> typing.Self:
        return cls(*items)

    def add(self, another: INode):
        new_args = list(self.as_tuple) + list(another.real(IGroup).as_tuple)
        return self.from_items(new_args)

class IFunction(INode, ABC):

    def fn_protocol(self) -> Protocol:
        raise NotImplementedError(self.__class__)

    def with_arg_group(self, group: BaseGroup, info: RunInfo) -> RunInfoResult:
        raise NotImplementedError(self.__class__)

class IRunnable(INode, ABC):

    def run(self, info_with_stats: RunInfoWithStats) -> RunInfoResult:
        raise NotImplementedError(self.__class__)

    def main_run(self, info: RunInfo) -> tuple[
        RunInfoResult,
        NodeReturnException | None,
    ]:
        try:
            protocol = self.protocol()

            try:
                outer_result = self._run(info)
            except NodeReturnException as e:
                outer_result = e.result

            IsInstance.assert_type(outer_result, RunInfoFullResult)
            assert isinstance(outer_result, RunInfoFullResult)
            result, args_group = outer_result.as_tuple
            IsInstance.assert_type(result, RunInfoResult)
            assert isinstance(result, RunInfoResult)
            new_info, new_node = result.as_tuple

            if isinstance(self, IOpaqueScope):
                result = RunInfoResult.with_args(
                    run_info=info.with_stats(),
                    return_value=new_node,
                )
                new_info, new_node = result.as_tuple
            else:
                new_info = new_info.with_scopes(info)
                result = RunInfoResult.with_args(
                    run_info=new_info,
                    return_value=new_node,
                )

            if (
                (not isinstance(self, IDynamic))
                or
                (
                    info.is_future()
                    and not isinstance(self, Placeholder)
                    and not isinstance(self, ParentScopeBase)
                )
            ):
                Eq(self.as_type(), new_node.as_type()).raise_on_false()

            if not info.is_future():
                if isinstance(self, IDynamic):
                    self.validate_result(result=new_node, args_group=args_group)
                else:
                    protocol.verify(new_node)

            if new_info.must_return():
                outer_result = RunInfoFullResult(result, args_group)
                exception = NodeReturnException(outer_result)
                return result, exception

            return result, None
        except InvalidNodeException as e:
            exc_info = e.info.add_stack(
                node=self,
                run_info=info,
            )
            raise InvalidNodeException(exc_info) from e

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        raise NotImplementedError(self.__class__)

class IDynamic(IRunnable, ABC):

    def validate_result(self, result: INode, args_group: OptionalValueGroup):
        raise NotImplementedError(self.__class__)

class IBoolean(INode):

    @property
    def as_bool(self) -> bool:
        raise NotImplementedError(self.__class__)

    @property
    def strict_bool(self) -> bool:
        value = self.as_bool
        if value is None:
            raise self.to_exception()
        return value

    def raise_on_false(self):
        if not self.as_bool:
            raise self.to_exception()

    def to_exception_info(self):
        return BooleanExceptionInfo(self)

    def to_exception(self):
        return InvalidNodeException(self.to_exception_info())

    @classmethod
    def true(cls) -> 'IntBoolean':
        return IntBoolean.create_true()

    @classmethod
    def false(cls) -> 'IntBoolean':
        return IntBoolean.create()

    @classmethod
    def from_bool(cls, value: bool) -> IntBoolean:
        if value:
            return cls.true()
        return cls.false()

class ITypedIndex(INodeIndex, typing.Generic[O, T], ABC):

    @classmethod
    def outer_type(cls) -> type[O]:
        raise NotImplementedError(cls)

    @classmethod
    def item_type(cls) -> TypeNode:
        raise NotImplementedError(cls)

    def find_in_node(self, node: INode):
        IsInstance.assert_type(node, self.outer_type())
        assert isinstance(node, self.outer_type())
        return self.find_in_outer_node(node)

    def replace_in_target(self, target_node: INode, new_node: INode):
        IsInstance.assert_type(target_node, self.outer_type())
        assert isinstance(target_node, self.outer_type())
        IsInstance.assert_type(new_node, self.item_type().type)
        assert isinstance(new_node, self.item_type().type)
        return self.replace_in_outer_target(target_node, new_node)

    def remove_in_target(self, target_node: INode):
        IsInstance.assert_type(target_node, self.outer_type())
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
            IsInstance.assert_type(result.value, self.item_type().type)
            assert isinstance(result.value, self.item_type().type), \
                f'{type(result.value)} != {self.item_type().type}'
        return result.real(IOptional[T])

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
        IsInstance.assert_type(node, BaseNode)
        assert isinstance(node, BaseNode)
        for idx in idxs:
            Integer(idx).strict_validate()
        assert all(isinstance(idx, int) for idx in idxs)
        self.node = node
        self.idxs = idxs

    def apply(self) -> BaseNode:
        node = self.node
        idxs = self.idxs
        for idx in idxs:
            node_aux = node.args[idx-1]
            IsInstance.assert_type(node_aux, BaseNode)
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

    cache_enabled = True
    fast = False
    _instances: dict[int, BaseNode] = dict()
    _cached_run: dict[int, tuple[RunInfoResult, NodeReturnException | None]] = dict()

    @staticmethod
    def __new__( # type: ignore
        cls: type[BaseNode],
        *args: int | INode | typing.Type[INode],
    ) -> typing.Self: # type: ignore
        h = hash((cls, args)) if BaseNode.cache_enabled else 0
        if BaseNode.cache_enabled:
            instance = cls._instances.get(h)
            if instance is not None:
                assert instance.__class__ == cls, (instance.__class__, cls)
                return typing.cast(typing.Self, instance)
        instance = super().__new__(cls)
        if BaseNode.cache_enabled:
            instance._cached_hash = h
            cls._instances[h] = instance
        else:
            instance._cached_hash = None
        return typing.cast(typing.Self, instance)

    def __init__(self, *args: int | INode | typing.Type[INode]):
        if not isinstance(self, IInstantiable):
            raise NotImplementedError(self.__class__)
        self._args = args
        self._cached_length: int | None = None
        self._cached_hash: int | None = self._cached_hash
        self._cached_valid: bool | None = None
        self._cached_valid_strict: AliasInfo | None = None
        self._cached_result_type: IType | bool | None = None

    @classmethod
    def clear_actual_cache(cls):
        cls._instances.clear()
        cls._cached_run.clear()

    def run(self, info_with_stats: RunInfoWithStats) -> RunInfoResult:
        info = info_with_stats.run_info.apply().real(RunInfo)
        info_hash = hash((self, info)) if BaseNode.cache_enabled else 0

        def fn_result(result: RunInfoResult, exception: NodeReturnException | None):
            if exception is not None:
                exception.add_stats(info_with_stats)
                raise exception
            result.add_stats(info_with_stats)
            return result

        if BaseNode.cache_enabled:
            cached = self._cached_run.get(info_hash)
            if cached is not None:
                result, exception = cached
                return fn_result(result, exception)

        cached_result = self.main_run(info)
        result, exception = cached_result
        _, new_node = result.as_tuple

        if self != new_node:
            new_result, _ = new_node.as_node.main_run(info)

            IsInstance.assert_type(new_result, RunInfoResult)
            assert isinstance(new_result, RunInfoResult)
            _, node_aux = new_result.as_tuple
            Eq(new_node, node_aux).raise_on_false()

        if BaseNode.cache_enabled:
            self._cached_run[info_hash] = cached_result

        return fn_result(result, exception)

    @classmethod
    def default_protocol(cls, args_group: IBaseTypeGroup) -> Protocol:
        return Protocol.with_args(args_group, cls.as_type())

    @classmethod
    def rest_protocol(cls, arg_type: IType) -> Protocol:
        return Protocol.rest(arg_type=arg_type, result_type=cls.as_type())

    @property
    def args(self) -> tuple[int | INode | typing.Type[INode], ...]:
        return self._args

    @property
    def as_node(self) -> BaseNode:
        return self

    def __eq__(self, other) -> bool:
        if not isinstance(other, BaseNode):
            return False
        return hash(self) == hash(other)

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
        return node in self.find(node.__class__)

    def has_until(self, node: INode, until_type: type[INode] | None) -> bool:
        return node in self.find_until(node.__class__, until_type)

    def inner_arg(self, idx: int) -> TmpInnerArg:
        return TmpInnerArg(self, idx)

    def nested_arg(self, idxs: tuple[int, ...]) -> TmpNestedArg:
        return TmpNestedArg(self, idxs)

    def validate(self):
        cached = self._cached_valid
        if cached is True:
            return
        self._validate()
        self._cached_valid = True

    def _validate(self):
        for arg in self.args:
            if isinstance(arg, INode):
                arg.as_node.validate()

    def strict_validate(self) -> AliasInfo:
        cached = self._cached_valid_strict
        if cached is not None:
            return cached
        alias_info = self._strict_validate()
        IsInstance.assert_type(alias_info, AliasInfo)
        assert alias_info is not None
        self._cached_valid_strict = alias_info
        return alias_info

    def _strict_validate(self) -> AliasInfo:
        raise NotImplementedError(self.__class__)

    def result_type(self) -> IType:
        cached = self._cached_result_type
        if cached is not None:
            if isinstance(cached, bool):
                IsInstance.assert_type(self, IType)
                assert isinstance(self, IType)
                return self
            return cached
        instance = self.actual_instance()
        protocol = instance.as_node.protocol()
        alias_info_p = protocol.verify(instance)
        protocol = alias_info_p.apply(protocol)
        Eq.from_ints(len(protocol.find(BaseTypeIndex)), 0).raise_on_false()
        result_type = (
            self.type.apply().real(IType)
            if isinstance(self, TypeEnforcer)
            else protocol.result.apply().real(IType))
        result_type.verify(instance, alias_info=AliasInfo.create())
        self._cached_result_type = result_type if result_type != self else True
        return result_type

###########################################################
######################## TYPE NODE ########################
###########################################################

class IType(INode, ABC):

    _valid_cache: dict[int, tuple[bool, AliasInfo]] = dict()

    def valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        key_hash = hash((self, instance, alias_info)) if BaseNode.cache_enabled else 0
        if BaseNode.cache_enabled:
            cached = self._valid_cache.get(key_hash)
            if cached is not None:
                return cached

        result = self.valid_inner(instance, alias_info)

        if BaseNode.cache_enabled:
            self._valid_cache[key_hash] = result

        return result

    def valid_inner(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        instance = instance.as_node.actual_instance()
        return self._valid(instance, alias_info)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        raise NotImplementedError

    def static_valid(
        self,
        instance: INode,
    ) -> bool:
        valid, _ = self.valid(instance, AliasInfo.create())
        return valid

    def static_valid_node(
        self,
        instance: INode,
    ) -> IntBoolean:
        valid = self.static_valid(instance)
        return IBoolean.from_bool(valid)

    def verify(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> AliasInfo:
        valid, alias_info = self.valid(instance, alias_info)
        if not valid:
            raise InvalidNodeException(TypeExceptionInfo(self, instance, alias_info))
        return alias_info

    @classmethod
    def general_valid_type(
        cls,
        base_type: INode,
        type_to_verify: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        type_stack: list[tuple[INode, INode]] = [(
            base_type,
            type_to_verify,
        )]
        Eq.from_ints(len(type_to_verify.as_node.find(BaseTypeIndex)), 0).raise_on_false()

        while len(type_stack) > 0:
            my_type, type_to_verify = type_stack.pop()
            if isinstance(my_type, ITypeValidator):
                if not isinstance(type_to_verify, IType):
                    return False, alias_info
                valid, alias_info = my_type.valid_type(type_to_verify, alias_info)
                if not valid:
                    return False, alias_info
            elif isinstance(type_to_verify, ITypeValidated):
                if not isinstance(my_type, IType):
                    return False, alias_info
                valid, alias_info = type_to_verify.validated_type(my_type, alias_info)
                if not valid:
                    return False, alias_info
            elif (
                isinstance(my_type, InheritableNode)
                and isinstance(type_to_verify, InheritableNode)
                and len(my_type.as_node.args) > 0
            ):
                if my_type.func != type_to_verify.func:
                    return False, alias_info
                my_args = my_type.args
                fn_args = type_to_verify.args
                if len(my_args) != len(fn_args):
                    return False, alias_info
                for my_arg, fn_arg in list(zip(my_args, fn_args))[::-1]:
                    type_stack.append((my_arg, fn_arg))
            elif my_type != type_to_verify:
                return False, alias_info

        return True, alias_info

    def fill_alias(
        self,
        instance: INode,
        alias_group: TypeAliasGroup,
    ) -> IType:
        alias_info = AliasInfo.with_node(alias_group)
        alias_info = self.verify(instance, alias_info)
        final_type = alias_info.apply_to_node(self).real(IType)
        return final_type

class ITypeValidator(IType, ABC):

    def valid_type(
        self,
        type_to_verify: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        raise NotImplementedError

class ITypeValidated(IType, ABC):

    def validated_type(
        self,
        type_that_verifies: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        raise NotImplementedError

class IBasicType(IType, ABC):
    pass

class TypeNode(
    BaseNode,
    IBasicType,
    ITypeValidator,
    ITypeValidated,
    IFunction,
    ISpecialValue,
    typing.Generic[T],
    IInstantiable,
):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup())

    def fn_protocol(self) -> Protocol:
        return self.type.protocol()

    def __init__(self, t: type[T]):
        origin = typing.get_origin(t)
        t = origin if origin is not None else t
        assert isinstance(t, type)
        assert issubclass(t, INode), t
        super().__init__(t)

    @property
    def type(self) -> typing.Type[T]:
        t = self.args[0]
        return typing.cast(typing.Type[T], t)

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
        InheritableNode.as_supertype().is_subclass(t).raise_on_false()
        assert issubclass(t, InheritableNode)
        return t.new(*group.as_tuple).as_node.run(info.with_stats())

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        return isinstance(instance, self.type), alias_info

    def _strict_validate(self) -> AliasInfo:
        self.validate()
        return AliasInfo.create()

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()
        result = info.with_stats().to_result(self)
        return RunInfoFullResult(result, OptionalValueGroup())

    def __eq__(self, other) -> bool:
        if not isinstance(other, TypeNode):
            return False
        return self.type == other.type

    def valid_type(
        self,
        type_to_verify: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        if not isinstance(type_to_verify, TypeNode):
            if isinstance(type_to_verify, ITypeValidated):
                return type_to_verify.validated_type(self, alias_info)
            return False, alias_info
        if not issubclass(type_to_verify.type, self.type):
            return False, alias_info
        return True, alias_info

    def validated_type(
        self,
        type_that_verifies: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        if not isinstance(type_that_verifies, TypeNode):
            if isinstance(type_that_verifies, ITypeValidator):
                return type_that_verifies.valid_type(self, alias_info)
            return False, alias_info
        if not issubclass(self.type, type_that_verifies.type):
            return False, alias_info
        return True, alias_info

###########################################################
######################## INT NODE #########################
###########################################################

class BaseInt(BaseNode, IInt, ISpecialValue, ABC):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup())

    def __init__(self, value: int):
        assert isinstance(value, int), \
            f'{type(value)} != {int}'
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

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()
        result = info.with_stats().to_result(self)
        return RunInfoFullResult(result, OptionalValueGroup())

    def _strict_validate(self):
        self.validate()
        GreaterOrEqual(self, self.func(0)).raise_on_false()
        return AliasInfo.create()

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

    @property
    def node_args(self) -> tuple[INode, ...]:
        return self.args

    def replace_at(self, index: INodeIndex, new_node: INode) -> INode | None:
        IsInstance.assert_type(index, INodeIndex)
        assert isinstance(index, INodeIndex)
        IsInstance.assert_type(new_node, INode)
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

    def _validate(self):
        super()._validate()
        args = self.args
        type_group = self.protocol().arg_group.apply()
        if isinstance(type_group, OptionalTypeGroup):
            LessOrEqual.with_ints(len(args), 1).raise_on_false()
            assert len(args) <= 1, \
                f'{type(self)}: {len(args)} > 1'
        elif isinstance(type_group, CountableTypeGroup):
            Eq.from_ints(len(args), len(type_group.args)).raise_on_false()
            assert len(args) == len(type_group.args), \
                f'{type(self)}: {len(args)} != {len(type_group.args)}'

    def _thin_strict_validate(self) -> AliasInfo:
        self.validate()
        alias_info = self.protocol().verify_args(DefaultGroup(*self.args))
        for arg in self.args:
            arg.as_node.validate()
        return alias_info

    def _strict_validate(self) -> AliasInfo:
        alias_info = self._thin_strict_validate()
        for arg in self.args:
            arg.as_node.strict_validate()
        return alias_info

    def full_strict_validate(self):
        self.strict_validate()
        for arg in self.args:
            if isinstance(arg, InheritableNode):
                arg.full_strict_validate()
            else:
                arg.as_node.strict_validate()

    def _base_run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()
        args: list[INode] = []
        info_with_stats = info.with_stats()
        for arg in self.args:
            info_with_stats, new_arg = arg.as_node.run(info_with_stats).as_tuple
            args.append(new_arg)
        new_node = self.func(*args)
        if not info_with_stats.is_future():
            new_node.strict_validate()
        result = (
            RunInfoResult.with_args(
                run_info=info_with_stats,
                return_value=new_node,
            )
            if isinstance(self, IDynamic)
            else info_with_stats.to_result(new_node))
        return RunInfoFullResult(result, OptionalValueGroup.from_optional_items(args))

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        return self._base_run(info)

###########################################################
###################### SPECIAL NODES ######################
###########################################################

class Void(InheritableNode, IDefault, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup())

class UnknownType(InheritableNode, IBasicType, IDefault, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup())

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        return True, alias_info

class InvalidType(InheritableNode, IBasicType, IDefault, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup())

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        return False, alias_info

class OptionalBase(InheritableNode, IOptional[T], typing.Generic[T], ABC):

    idx_value = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
            ),
            OptionalTypeGroup(TypeIndex(1)),
            CompositeType(
                cls.as_type(),
                OptionalTypeGroup(TypeIndex(1)),
            )
        )

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

class ParentScopeBase(BaseInt, IDynamic, IDefault, ABC):

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol.with_args(
            CountableTypeGroup(),
            RunInfoScopeDataIndex.as_type(),
        )

    @classmethod
    def create(cls) -> typing.Self:
        return cls(1)

    def _index(self, info: RunInfo) -> RunInfoScopeDataIndex:
        raise NotImplementedError

    def _run(self, info: RunInfo):
        node = self._index(info)
        IsInstance.assert_type(node, RunInfoScopeDataIndex)
        assert isinstance(node, RunInfoScopeDataIndex)
        scope_data_group = info.scope_data_group.apply().real(ScopeDataGroup)
        amount = len(scope_data_group.as_tuple)
        IsInsideRange(node, Integer.zero(), Integer(amount)).raise_on_false()
        result = info.with_stats().to_result(node)
        return RunInfoFullResult(result, OptionalValueGroup())

    def validate_result(self, result: INode, args_group: OptionalValueGroup):
        IsInstance.verify(result, RunInfoScopeDataIndex)

class NearParentScope(ParentScopeBase, IInstantiable):

    def _index(self, info: RunInfo):
        scope_data_group = info.scope_data_group.apply().real(ScopeDataGroup)
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

class Placeholder(InheritableNode, IDynamic, IFromInt, typing.Generic[T], ABC):

    idx_parent_scope = 1
    idx_index = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
            ),
            CountableTypeGroup(
                RunInfoScopeDataIndex.as_type(),
                PlaceholderIndex.as_type(),
            ),
            TypeIndex(1),
        )

    def validate_result(self, result: INode, args_group: OptionalValueGroup):
        IsInstance.verify(result, INode)

    @property
    def parent_scope(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_parent_scope)

    @property
    def index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_index)

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = self._base_run(info).as_tuple
        info_with_stats, node_aux = base_result.as_tuple
        node = node_aux.real(self.__class__)

        scope_index = node.parent_scope.apply().real(RunInfoScopeDataIndex)
        index = node.index.apply().real(BaseInt)

        scope = scope_index.find_in_outer_node(info).value_or_raise
        IsInstance.assert_type(scope, ScopeDataPlaceholderItemGroup)
        assert isinstance(scope, ScopeDataPlaceholderItemGroup), type(scope)

        IsInstance.assert_type(node, scope.item_inner_type())
        assert isinstance(node, scope.item_inner_type()), \
            f'{type(node)} != {scope.item_inner_type()} ({scope_index.as_int} - {index.as_int})'

        if isinstance(scope, IScopeDataFutureItemGroup):
            groups = info.scope_data_group.apply().real(ScopeDataGroup).as_tuple
            group_amount = len(groups)
            near_scope = NearParentScope.from_int(group_amount - scope_index.as_int + 1)
            result = info_with_stats.to_result(node.func(near_scope, index))
            return RunInfoFullResult(result, arg_group)

        item = NodeArgIndex(index.as_int).find_in_node(scope).value_or_raise
        info_with_stats, node_aux = item.as_node.run(info_with_stats).as_tuple
        node_aux = node_aux.real(IOptional)

        new_node = node_aux.value_or_raise

        result = info_with_stats.to_result(new_node)
        return RunInfoFullResult(result, arg_group)

class Param(Placeholder[T], IInstantiable, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(
            FarParentScope.create(),
            PlaceholderIndex(value),
        )

class Var(Placeholder[T], IInstantiable, typing.Generic[T]):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(
            NearParentScope.create(),
            PlaceholderIndex(value),
        )

###########################################################
######################## NODE IDXS ########################
###########################################################

class NodeIntBaseIndex(BaseInt, INodeIndex, IInt, ABC):
    pass

class NodeMainBaseIndex(NodeIntBaseIndex, ABC):

    @classmethod
    def _inner_getitem(cls, node: INode, index: int) -> tuple[INode | None, int]:
        GreaterThan.with_ints(index, 0).raise_on_false()
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
        GreaterThan.with_ints(index, 0).raise_on_false()
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
        GreaterThan.with_ints(index, 0).raise_on_false()
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
        return self._find_in_node_general(node, reverse=False)

    def replace_in_target(self, target_node: T, new_node: INode) -> IOptional[T]:
        return self._replace_in_target_general(target_node, new_node, reverse=False)

    def remove_in_target(self, target_node: T) -> Optional[T]:
        return self._remove_in_target_general(target_node, reverse=False)

    def _index_value(self, node: InheritableNode, reverse: bool) -> int:
        base_index = self.as_int
        args = node.args
        return len(args) - base_index + 1 if reverse else base_index

    def _find_in_node_general(self, node: INode, reverse: bool) -> IOptional[INode]:
        if not isinstance(node, InheritableNode):
            return Optional()
        index = self._index_value(node, reverse)
        args = node.args
        if 0 < index <= len(args):
            return Optional(args[index - 1])
        return Optional()

    def _replace_in_target_general(
        self,
        target_node: T,
        new_node: INode,
        reverse: bool,
    ) -> IOptional[T]:
        if not isinstance(target_node, InheritableNode):
            return Optional()
        index = self._index_value(target_node, reverse)
        args = target_node.args
        if 0 < index <= len(args):
            new_target = target_node.func(*[
                (new_node if i == index - 1 else arg)
                for i, arg in enumerate(args)
            ])
            IsInstance.assert_type(new_target, type(target_node))
            assert isinstance(new_target, type(target_node))
            return Optional(new_target).real(IOptional[T])
        return Optional()

    def _remove_in_target_general(self, target_node: T, reverse: bool) -> Optional[T]:
        if not isinstance(target_node, InheritableNode):
            return Optional()
        index = self._index_value(target_node, reverse)
        args = target_node.args
        if 0 < index <= len(args):
            new_args = [
                arg
                for i, arg in enumerate(args)
                if i != index - 1
            ]
            new_target = target_node.func(*new_args)
            IsInstance.assert_type(new_target, type(target_node))
            assert isinstance(new_target, type(target_node))
            return Optional(new_target).real(Optional[T])
        return Optional()

class NodeArgIndex(NodeArgBaseIndex, IInstantiable):
    pass

class NodeArgReverseIndex(NodeArgBaseIndex, IInstantiable):

    def find_in_node(self, node: INode) -> IOptional[INode]:
        return self._find_in_node_general(node, reverse=True)

    def replace_in_target(self, target_node: T, new_node: INode) -> IOptional[T]:
        return self._replace_in_target_general(target_node, new_node, reverse=True)

    def remove_in_target(self, target_node: T) -> Optional[T]:
        return self._remove_in_target_general(target_node, reverse=True)

###########################################################
####################### ITEMS GROUP #######################
###########################################################

class BaseGroup(InheritableNode, IGroup[T], IDefault, typing.Generic[T], ABC):

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            RestTypeGroup(cls.item_type()),
            CompositeType(
                cls.as_type(),
                RestTypeGroup(cls.item_type()),
            ),
        )

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

    def _strict_validate(self):
        alias_info = super()._strict_validate()
        t = self.item_type().type
        for arg in self.args:
            origin = typing.get_origin(t)
            t = origin if origin is not None else t
            IsInstance.assert_type(arg, t)
            assert isinstance(arg, t), f'{type(arg)} != {t}'
        return alias_info

    def to_optional_group(self) -> OptionalValueGroup[T]:
        return OptionalValueGroup.from_optional_items(self.args)

class DefaultGroup(BaseGroup[INode], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return INode.as_type()

class BaseIntGroup(BaseGroup[IInt], ABC):

    @classmethod
    def item_type(cls) -> TypeNode:
        return IInt.as_type()

    @classmethod
    def create_item(cls, value: int):
        return Integer(value)

    @classmethod
    def from_ints(cls, indices: typing.Sequence[int]) -> typing.Self:
        return cls(*[cls.create_item(i) for i in indices])

class IntGroup(BaseIntGroup, IInstantiable):
    pass

class NestedArgIndexGroup(BaseIntGroup, IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return NodeArgIndex.as_type()

    @classmethod
    def create_item(cls, value: int):
        return NodeArgIndex(value)

    def apply(self, node: BaseNode) -> BaseNode:
        args_indices = [arg.as_int for arg in self.args]
        return node.nested_arg(tuple(args_indices)).apply()

class BaseOptionalValueGroup(BaseGroup[IOptional[T]], IFromInt, typing.Generic[T], ABC):

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(*([Optional()]*value))

    @classmethod
    def item_type(cls) -> TypeNode:
        return Optional.as_type()

    @classmethod
    def from_optional_items(cls, items: typing.Sequence[T | None]) -> typing.Self:
        return cls(*[Optional.with_value(item) for item in items])

    def strict(self) -> DefaultGroup:
        values = [item.value for item in self.args if item.value is not None]
        Eq.from_ints(len(values), len(self.args)).raise_on_false()
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

class ITypeGroupValidator(INode, ABC):

    def valid_info(
        self,
        values: OptionalValueGroup,
        alias_info: AliasInfo,
        raise_on_invalid: bool = False,
    ) -> tuple[bool, AliasInfo]:
        raise NotImplementedError

    def validate_values(
        self,
        values: OptionalValueGroup,
        alias_info: AliasInfo,
    ) -> AliasInfo:
        _, alias_info = self.valid_info(
            values,
            alias_info=alias_info,
            raise_on_invalid=True)
        return alias_info

class IBaseTypeGroup(ITypeGroupValidator, ABC):
    pass

class CountableTypeGroup(
    BaseGroup[IType],
    IBaseTypeGroup,
    IFromInt,
    IInstantiable,
):

    @classmethod
    def item_type(cls) -> TypeNode:
        return IType.as_type()

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls(*[UnknownType()] * value)

    def valid_info(
        self,
        values: OptionalValueGroup,
        alias_info: AliasInfo,
        raise_on_invalid: bool = False,
    ) -> tuple[bool, AliasInfo]:
        args = values.args
        same_size = SameArgsAmount(values, self)

        if not same_size.as_bool:
            if raise_on_invalid:
                raise InvalidNodeException(same_size.to_exception_info())
            return False, alias_info

        for i, arg_opt in enumerate(args):
            arg = arg_opt.value
            if arg is not None:
                t_arg = self.args[i]
                if not isinstance(arg, INode):
                    if raise_on_invalid:
                        raise InvalidNodeException(TypeExceptionInfo(t_arg, values, alias_info))
                    return False, alias_info
                valid, alias_info = t_arg.valid(arg, alias_info=alias_info)
                if not valid:
                    if raise_on_invalid:
                        raise InvalidNodeException(TypeExceptionInfo(t_arg, arg, alias_info))
                    return valid, alias_info

        if len(args) != len(self.args):
            if raise_on_invalid:
                raise SameArgsAmount(values, self).to_exception()
            return False, alias_info

        return True, alias_info

class SingleValueTypeGroup(
    InheritableNode,
    IBaseTypeGroup,
    ISingleChild[IType],
    ABC,
):

    idx_type_node = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(TypeAlias(IType.as_type())),
            CountableTypeGroup(TypeIndex(1)),
            CompositeType(
                cls.as_type(),
                CountableTypeGroup(TypeIndex(1)),
            ),
        )

    @property
    def type_node(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type_node)

    @property
    def child(self) -> IType:
        return self.type_node.apply().real(IType)

    def valid_info(
        self,
        values: OptionalValueGroup,
        alias_info: AliasInfo,
        raise_on_invalid: bool = False,
    ) -> tuple[bool, AliasInfo]:
        t_arg = self.type_node.apply().real(IType)

        for arg_opt in values.args:
            arg = arg_opt.value
            if arg is not None:
                if not isinstance(arg, INode):
                    if raise_on_invalid:
                        raise InvalidNodeException(TypeExceptionInfo(t_arg, values, alias_info))
                    return False, alias_info
                valid, alias_info = t_arg.valid(arg, alias_info=alias_info)
                if not valid:
                    if raise_on_invalid:
                        raise InvalidNodeException(TypeExceptionInfo(t_arg, arg, alias_info))
                    return valid, alias_info

        return True, alias_info

class RestTypeGroup(SingleValueTypeGroup, IInstantiable):
    pass

class OptionalTypeGroup(SingleValueTypeGroup, IInstantiable):
    pass

class TypeAlias(InheritableNode, ISingleChild[IType], IInstantiable):

    idx_type = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(TypeAlias(IType.as_type())),
            CountableTypeGroup(TypeIndex(1)),
            CompositeType(
                cls.as_type(),
                CountableTypeGroup(TypeIndex(1)),
            ),
        )

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def child(self) -> IType:
        return self.type.apply().real(IType)

class TypeAliasGroup(
    BaseGroup[TypeAlias],
    IInstantiable,
):

    @classmethod
    def item_type(cls) -> TypeNode:
        return TypeAlias.as_type()

    def _strict_validate(self):
        alias_info = super()._strict_validate()
        for i, arg_aux in enumerate(self.args):
            index = i + 1
            arg = arg_aux.real(TypeAlias)
            inner_idxs = sorted(arg.as_node.find(BaseTypeIndex), key=lambda t: t.as_int)
            invalid_idxs = [idx for idx in inner_idxs if idx.as_int >= index]
            if len(invalid_idxs) > 0:
                invalid_group = DefaultGroup(*invalid_idxs)
                raise InvalidNodeException(
                    TypeAliasIndexExceptionInfo(TypeIndex(index), arg, invalid_group))
        return alias_info

class TypeAliasOptionalGroup(
    BaseOptionalValueGroup[IType],
    IInstantiable,
):

    @classmethod
    def init(cls, origin_group: TypeAliasGroup) -> typing.Self:
        return cls(*[Optional()] * len(origin_group.args))

    def replace_aliases(self, node: INode) -> INode:
        for i, actual in enumerate(self.as_tuple):
            t = (
                InvalidType()
                if actual.is_empty().as_bool
                else actual.value_or_raise)
            node = TypeIndex(i + 1).replace_in_node(node, t)
        Eq(Integer(len(node.as_node.find(BaseTypeIndex))), Integer.zero()).raise_on_false()
        return node

class AliasInfo(
    InheritableNode,
    IDefault,
    IFromSingleNode[TypeAliasGroup],
    IInstantiable,
):

    idx_alias_group_base = 1
    idx_alias_group_actual = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            TypeAliasGroup.as_type(),
            TypeAliasOptionalGroup.as_type(),
        ))

    def _validate(self):
        super()._validate()
        base_group = self.alias_group_base.apply().real(TypeAliasGroup)
        actual_group = self.alias_group_actual.apply().real(TypeAliasOptionalGroup)
        SameArgsAmount(base_group, actual_group).raise_on_false()

    @property
    def alias_group_base(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_alias_group_base)

    @property
    def alias_group_actual(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_alias_group_actual)

    @classmethod
    def create(cls) -> typing.Self:
        return cls.with_node(TypeAliasGroup())

    @classmethod
    def with_node(cls, node: TypeAliasGroup) -> typing.Self:
        return cls(node, TypeAliasOptionalGroup.init(node))

    def define(self, type_index: BaseTypeIndex, new_type: IType) -> AliasInfo:
        base_group = self.alias_group_base.apply().real(TypeAliasGroup)
        alias = type_index.find_in_node(base_group).value_or_raise.real(TypeAlias)
        base_type = alias.child
        actual_group = self.alias_group_actual.apply().real(TypeAliasOptionalGroup)
        full_type = IntersectionType(base_type, new_type)

        old_type_opt = type_index.find_in_node(actual_group).value_or_raise.real(IOptional)
        new_type_opt: Optional[INode] = Optional(full_type)

        if old_type_opt.is_empty().as_bool:
            new_group = type_index.replace_in_target(
                actual_group,
                Optional(full_type),
            ).value_or_raise
            actual_group = new_group
        elif isinstance(type_index, LazyTypeIndex):
            IBoolean.from_bool(not type_index.present_in_node(old_type_opt)).raise_on_false()
            IBoolean.from_bool(not type_index.present_in_node(new_type_opt)).raise_on_false()
            valid, _ = IType.general_valid_type(
                base_type=new_type_opt,
                type_to_verify=old_type_opt,
                alias_info=self,
            )
            if not valid:
                raise InvalidNodeException(
                    TypeAcceptExceptionInfo(
                        new_type_opt,
                        old_type_opt,
                    )
                )
        else:
            Eq(new_type_opt, old_type_opt).raise_on_false()

        return self.func(base_group, actual_group)

    def apply(self, protocol: Protocol) -> Protocol:
        self_group = self.alias_group_base.apply().real(TypeAliasGroup)
        p_alias_group = protocol.alias_group.apply().real(TypeAliasGroup)
        Eq(self_group, p_alias_group).raise_on_false()
        Eq(Integer(len(p_alias_group.as_node.find(BaseTypeIndex))), Integer.zero()).raise_on_false()

        actual_group = self.alias_group_actual.apply().real(TypeAliasOptionalGroup)
        args_group = actual_group.replace_aliases(protocol.arg_group.apply()).real(IBaseTypeGroup)
        result_group = actual_group.replace_aliases(protocol.result.apply()).real(IType)

        return Protocol.with_args(
            alias_group=TypeAliasGroup(),
            args_type_group=args_group,
            result_type=result_group,
        )

    def apply_to_node(self, node: INode) -> INode:
        actual_group = self.alias_group_actual.apply().real(TypeAliasOptionalGroup)
        node = actual_group.replace_aliases(node)
        return node

class BaseTypeIndex(NodeArgBaseIndex, ITypeValidator, ABC):

    def define(
        self,
        instance: INode,
        result_type: IType,
        alias_info: AliasInfo,
    ) -> AliasInfo:
        raise NotImplementedError

    def valid_inner(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        return self._valid(instance, alias_info)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        base_group = alias_info.alias_group_base.apply().real(TypeAliasGroup)
        type_alias = self.find_in_node(base_group).value_or_raise.real(TypeAlias)
        original = instance
        instance = instance.as_node.actual_instance()
        valid, alias_info = type_alias.child.valid(instance, alias_info=alias_info)
        if not valid:
            return valid, alias_info
        result_type = original.as_node.result_type()
        result_type.verify(instance, alias_info=AliasInfo.create())
        alias_info = self.define(
            instance=instance,
            result_type=result_type,
            alias_info=alias_info)
        return True, alias_info

    def actual_type(self, alias_info: AliasInfo) -> IOptional[IType]:
        actual_group = alias_info.alias_group_actual.apply().real(TypeAliasOptionalGroup)
        actual_opt = self.find_in_node(actual_group).value_or_raise.real(IOptional[IType])
        return actual_opt

    def valid_type(
        self,
        type_to_verify: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        if not isinstance(type_to_verify, IType):
            return False, alias_info
        alias_info = alias_info.define(self, type_to_verify)
        return True, alias_info

    def _occurences(self, node: INode) -> list[BaseTypeIndex]:
        items: list[BaseTypeIndex] = sorted([
            item
            for item in node.as_node.find(BaseTypeIndex)
            if item.as_int == self.as_int
        ], key=lambda t: type_sorter_key(t.__class__))
        return items

    def present_in_node(self, node: INode) -> bool:
        return len(self._occurences(node)) > 0

    def replace_in_node(self, target: INode, new_node: INode) -> INode:
        items = self._occurences(target)
        for item in items:
            target = target.as_node.replace(item, new_node)
        return target

class TypeIndex(BaseTypeIndex, IInstantiable):

    def define(
        self,
        instance: INode,
        result_type: IType,
        alias_info: AliasInfo,
    ) -> AliasInfo:
        alias_info = alias_info.define(self, result_type)
        return alias_info

class LazyTypeIndex(BaseTypeIndex, IInstantiable):

    def define(
        self,
        instance: INode,
        result_type: IType,
        alias_info: AliasInfo,
    ) -> AliasInfo:
        actual_type = self.actual_type(alias_info).value_or_raise.real(IType)
        actual_type.verify(instance, alias_info=alias_info)
        return alias_info

class Protocol(
    InheritableNode,
    IDefault,
    IFromInt,
    IFromSingleNode[IBaseTypeGroup],
    ITypeGroupValidator,
    IInstantiable,
):

    idx_alias_group = 1
    idx_arg_group = 2
    idx_result = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            TypeAliasGroup.as_type(),
            IBaseTypeGroup.as_type(),
            IType.as_type(),
        ))

    @property
    def alias_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_alias_group)

    @property
    def arg_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_group)

    @property
    def result(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_result)

    def valid_info(
        self,
        values: OptionalValueGroup,
        alias_info: AliasInfo,
        raise_on_invalid: bool = False,
    ) -> tuple[bool, AliasInfo]:
        group = self.arg_group.apply().real(IBaseTypeGroup)
        return group.valid_info(
            values,
            alias_info=alias_info,
            raise_on_invalid=raise_on_invalid)

    @classmethod
    def create(cls) -> typing.Self:
        return cls.rest()

    @classmethod
    def from_int(cls, value: int) -> typing.Self:
        return cls.with_node(CountableTypeGroup.from_int(value))

    @classmethod
    def rest(
        cls,
        arg_type: IType = UnknownType(),
        result_type: IType = UnknownType(),
    ) -> typing.Self:
        return cls.with_args(RestTypeGroup(arg_type), result_type)

    @classmethod
    def with_node(cls, node: IBaseTypeGroup) -> typing.Self:
        return cls.with_args(node, UnknownType())

    @classmethod
    def with_args(
        cls,
        args_type_group: IBaseTypeGroup,
        result_type: IType,
        alias_group: TypeAliasGroup | None = None,
    ) -> typing.Self:
        alias_group = (
            alias_group
            if alias_group is not None
            else TypeAliasGroup())
        return cls(alias_group, args_type_group, result_type)

    def with_new_args(
        self,
        args_type_group: IBaseTypeGroup | None = None,
        result_type: IType | None = None,
        alias_group: TypeAliasGroup | None = None,
    ) -> typing.Self:
        alias_group = (
            alias_group
            if alias_group is not None
            else self.alias_group.apply().real(TypeAliasGroup))
        args_type_group = (
            args_type_group
            if args_type_group is not None
            else self.arg_group.apply().real(IBaseTypeGroup))
        result_type = (
            result_type
            if result_type is not None
            else self.result.apply().real(IType))
        return self.func(alias_group, args_type_group, result_type)

    def new_amount(self, amount: int) -> typing.Self:
        group = self.arg_group.apply().real(IBaseTypeGroup)
        if not isinstance(group, CountableTypeGroup):
            return self.from_int(amount)
        items = group.as_tuple
        if amount == len(items):
            return self
        return self.with_new_args(
            args_type_group=CountableTypeGroup.from_items([
                (items[i] if i < len(items) else UnknownType())
                for i in range(amount)
            ])
        )

    def verify(self, instance: INode) -> AliasInfo:
        if isinstance(instance, InheritableNode):
            args = DefaultGroup(*instance.args)
            alias_info = self.verify_args(args)
            self.verify_result(instance, alias_info)
            return alias_info
        else:
            instance.as_node.strict_validate()
            return AliasInfo.create()

    def fill_aliases(self, instance: INode) -> 'Protocol':
        alias_info = self.verify(instance)
        return alias_info.apply(self)

    def verify_args(self, args: BaseGroup) -> AliasInfo:
        return self.verify_optional_args(args.to_optional_group())

    def verify_optional_args(self, args_group: OptionalValueGroup) -> AliasInfo:
        original_group = self.arg_group.apply().real(IBaseTypeGroup)
        aliases_base = self.alias_group.apply().real(TypeAliasGroup)
        alias_info = AliasInfo.with_node(aliases_base)
        alias_info = original_group.validate_values(args_group, alias_info=alias_info)
        return alias_info

    def verify_result(self, result: INode, alias_info: AliasInfo) -> AliasInfo:
        result_type = self.result.apply().real(IType)
        alias_info = result_type.verify(result, alias_info=alias_info)
        return alias_info

    def with_default_aliases(self) -> 'Protocol':
        alias_group = self.alias_group.apply().real(TypeAliasGroup)
        alias_info = AliasInfo.with_node(alias_group)
        return alias_info.apply(self)

    def final_result_type(self, instance: INode) -> IType:
        result_type = self.result.apply().real(IType)
        alias_group = self.alias_group.apply().real(TypeAliasGroup)
        final_result_type = result_type.fill_alias(instance, alias_group)
        return final_result_type

    def valid_protocol(self):
        self.full_strict_validate()
        alias_group = self.alias_group.apply().real(TypeAliasGroup)
        alias_amount = len(alias_group.as_tuple)
        alias_idxs = sorted(list(self.find(BaseTypeIndex)), key=lambda index: index.as_int)
        for index in alias_idxs:
            if index.as_int <= 0 or index.as_int > alias_amount:
                raise InvalidNodeException(
                    TypeIndexProtocolExceptionInfo(index, self))
        for i in range(alias_amount):
            index = TypeIndex(i + 1)
            if not index.present_in_node(self):
                raise InvalidNodeException(
                    TypeIndexProtocolExceptionInfo(index, self))

###########################################################
########################## TYPES ##########################
###########################################################

class Type(InheritableNode, IBasicType, ISingleChild[IType], IInstantiable):

    idx_type = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(IType.as_type()))

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def child(self) -> IType:
        return self.type.apply().real(IType)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        my_type = self.type.apply().real(IType)
        type_to_verify = instance.real(IType)
        return IType.general_valid_type(my_type, type_to_verify, alias_info)

    def is_subclass(
        self,
        t: typing.Type,
    ) -> IntBoolean:
        if not issubclass(t, INode):
            return IBoolean.false()
        return self.static_valid_node(t.as_type())

class NotType(InheritableNode, IBasicType, ISingleChild[IType], IInstantiable):

    idx_type = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(IType.as_type()))

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def child(self) -> IType:
        return self.type.apply().real(IType)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        my_type = self.type.apply().real(IType)
        Eq.from_ints(len(my_type.as_node.find(TypeIndex)), 0).raise_on_false()
        valid, alias_info = my_type.valid(instance, alias_info)
        return not valid, alias_info

class AliasType(InheritableNode, IBasicType, IInstantiable):

    idx_type = 1
    idx_alias = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IType.as_type(),
            BaseTypeIndex.as_type(),
        ))

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def alias(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_alias)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        my_type = self.type.apply().real(IType)
        type_index = self.alias.apply().real(BaseTypeIndex)
        valid, alias_info = my_type.valid(instance, alias_info)
        if not valid:
            return False, alias_info
        alias_info = alias_info.define(type_index, my_type)
        return valid, alias_info

class InstanceType(InheritableNode, IBasicType, ISingleChild[INode], IInstantiable):

    idx_instance = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(INode.as_type()))

    @property
    def instance(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_instance)

    @property
    def child(self) -> IType:
        return self.instance.apply().real(IType)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        return instance == self.instance.apply(), alias_info

class FunctionType(InheritableNode, IBasicType, IInstantiable):

    idx_arg_group = 1
    idx_result = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IBaseTypeGroup.as_type(),
            IType.as_type(),
        ))

    @property
    def arg_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_group)

    @property
    def result(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_result)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        if not isinstance(instance, IFunction):
            return False, alias_info

        arg_group = self.arg_group.apply().real(IBaseTypeGroup)
        result_type = self.result.apply().real(IType)

        fn_protocol = instance.fn_protocol()
        fn_alias_group = fn_protocol.alias_group.apply().real(TypeAliasGroup)

        # Inner function must not have aliases
        Eq(fn_alias_group, TypeAliasGroup()).raise_on_false()

        my_protocol = Protocol(
            TypeAliasGroup(),
            arg_group,
            result_type,
        )

        return IType.general_valid_type(
            base_type=my_protocol,
            type_to_verify=fn_protocol,
            alias_info=alias_info,
        )

class DynamicType(InheritableNode, IBasicType, IInstantiable):

    idx_transformer = 1
    idx_type = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
                TypeAlias(INode.as_type()),
                TypeAlias(IType.as_type()),
            ),
            CountableTypeGroup(
                FunctionType(
                    CountableTypeGroup(TypeIndex(1)),
                    TypeIndex(2),
                ),
                TypeIndex(3),
            ),
            cls.as_type(),
        )

    @property
    def transformer(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_transformer)

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        transformer = self.transformer.apply().real(IFunction)
        t = self.type.apply().real(IType)

        arg_group = DefaultGroup(instance)

        run_info = RunInfo.with_args(
            scope_data_group=ScopeDataGroup(),
            return_after_scope=Optional(),
        )
        _, result = FunctionCall.define(
            fn=transformer,
            args=arg_group,
        ).as_node.run(run_info.with_stats()).as_tuple

        return t.valid(result, alias_info=alias_info)

class TypeEnforcer(InheritableNode, IInstantiable):

    idx_type = 1
    idx_node = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
            ),
            CountableTypeGroup(
                Type(TypeIndex(1)),
                LazyTypeIndex(1),
            ),
            TypeIndex(1),
        )

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def node(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_node)

    def _strict_validate(self):
        alias_info = super()._strict_validate()
        t = self.type.apply().real(IType)
        node = self.node.apply()
        t.verify(node, alias_info=AliasInfo.create())
        return alias_info

    def actual_instance(self) -> BaseNode:
        instance: BaseNode = self
        while isinstance(instance, TypeEnforcer):
            instance = instance.node.apply()
        return instance

class IComplexType(IType, ABC):

    def multi_bool(
        self,
        items: typing.Sequence[bool],
        any_case: bool,
        all_case: bool,
    ) -> bool:
        if any([item is any_case for item in items]):
            return any_case
        if all([item is all_case for item in items]):
            return all_case
        return False

class UnionType(
    InheritableNode,
    IComplexType,
    ITypeValidator,
    ITypeValidated,
    IInstantiable,
):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.rest_protocol(IType.as_type())

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        Eq.from_ints(len(args), len(self.args)).raise_on_false()
        for t in args:
            valid, alias_info = t.valid(instance, alias_info=alias_info)
            if valid:
                return True, alias_info
        return False, alias_info

    def valid_type(
        self,
        type_to_verify: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        Eq.from_ints(len(args), len(self.args)).raise_on_false()
        for t in args:
            valid, alias_info = IType.general_valid_type(
                base_type=t,
                type_to_verify=type_to_verify,
                alias_info=alias_info)
            if valid:
                return True, alias_info
        return False, alias_info

    def validated_type(
        self,
        type_that_verifies: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        Eq.from_ints(len(args), len(self.args)).raise_on_false()
        for t in args:
            valid, alias_info = IType.general_valid_type(
                base_type=type_that_verifies,
                type_to_verify=t,
                alias_info=alias_info)
            if not valid:
                return False, alias_info
        return True, alias_info

class IntersectionType(
    InheritableNode,
    IComplexType,
    ITypeValidator,
    ITypeValidated,
    IInstantiable,
):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.rest_protocol(IType.as_type())

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        Eq.from_ints(len(args), len(self.args)).raise_on_false()
        for t in args:
            valid, alias_info = t.valid(instance, alias_info=alias_info)
            if not valid:
                return False, alias_info
        return True, alias_info

    def valid_type(
        self,
        type_to_verify: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        Eq.from_ints(len(args), len(self.args)).raise_on_false()
        for t in args:
            valid, alias_info = IType.general_valid_type(
                base_type=t,
                type_to_verify=type_to_verify,
                alias_info=alias_info)
            if not valid:
                return False, alias_info
        return True, alias_info

    def validated_type(
        self,
        type_that_verifies: IType,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        args = [arg for arg in self.args if isinstance(arg, IType)]
        Eq.from_ints(len(args), len(self.args)).raise_on_false()
        for t in args:
            valid, alias_info = IType.general_valid_type(
                base_type=type_that_verifies,
                type_to_verify=t,
                alias_info=alias_info)
            if valid:
                return True, alias_info
        return False, alias_info

class CompositeType(InheritableNode, IComplexType, IInstantiable):

    idx_type = 1
    idx_type_args = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IType.as_type(),
            UnionType(IBaseTypeGroup.as_type(), Void.as_type()),
        ))

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def type_args(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type_args)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        t = self.type.apply().real(IType)
        valid, alias_info = t.valid(instance, alias_info=alias_info)
        if not valid:
            return False, alias_info
        args = self.type_args.apply()
        if args == Void():
            return True, alias_info
        IsInstance.verify(args, IBaseTypeGroup)
        assert isinstance(args, IBaseTypeGroup)
        i_args = instance.real(InheritableNode).args
        return args.valid_info(
            DefaultGroup(*i_args).to_optional_group(),
            alias_info=alias_info)

###########################################################
###################### FUNCTION NODE ######################
###########################################################

class ScopedFunctionBase(
    InheritableNode,
    IFunction,
    IScope,
    ABC,
):

    idx_protocol_arg = 1
    idx_expr = 2

    @classmethod
    def expr_type(cls) -> IType:
        raise NotImplementedError

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(
            CountableTypeGroup(
                Protocol.as_type(),
                cls.expr_type(),
            ),
        )

    @property
    def protocol_arg(self) -> TmpInnerArg:
        return self.as_node.inner_arg(self.idx_protocol_arg)

    @property
    def expr(self) -> TmpInnerArg:
        return self.as_node.inner_arg(self.idx_expr)

    def _strict_validate(self):
        return self._thin_strict_validate()

    def fn_protocol(self) -> Protocol:
        return self.protocol_arg.apply().real(Protocol)

    def with_arg_group(self, group: BaseGroup, info: RunInfo):
        alias_info: AliasInfo | None = None
        protocol_arg: Protocol | None = None

        if info.is_future():
            scope_data: ScopeDataParamBaseItemGroup = ScopeDataFutureParamItemGroup()
        else:
            protocol_arg = self.protocol_arg.apply().real(Protocol)
            alias_info = protocol_arg.verify_args(group)
            scope_data = ScopeDataParamItemGroup.from_optional_items(group.as_tuple)
            scope_data.strict_validate()

        new_main_info = (
            info.with_new_args(
                scope_data_group=ScopeDataGroup(),
                return_after_scope=Optional(),
            )
            if isinstance(self, IOpaqueScope)
            else info
        ).add_scope(scope_data)
        new_info = new_main_info.with_stats()

        try:
            result = self.expr.apply().run(new_info)
        except NodeReturnException as e:
            result, _ = e.result.as_tuple

        if isinstance(self, IOpaqueScope):
            info_with_stats, node = result.as_tuple
            new_info = info.with_stats().add_stats(info_with_stats)
        else:
            new_info, node = result.as_tuple
            new_info = new_info.with_scopes(info)

        if not info.is_future():
            Optional.with_value(alias_info).raise_if_empty()
            assert alias_info is not None
            Optional.with_value(protocol_arg).raise_if_empty()
            assert protocol_arg is not None
            protocol_arg.verify_result(node, alias_info=alias_info)

        return RunInfoResult.with_args(
            run_info=new_info,
            return_value=node,
        )

    def prepare_expr(self, info_with_stats: RunInfoWithStats) -> RunInfoResult:
        raise NotImplementedError

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()

        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.protocol_arg.apply().run(info_with_stats).as_tuple
        type_group = node_aux.real(Protocol)

        info_with_stats, expr = self.prepare_expr(info_with_stats).as_tuple

        result = info_with_stats.to_result(self.func(type_group, expr))
        args_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [type_group, expr]
        )
        return RunInfoFullResult(result, args_group)

class FunctionExpr(
    ScopedFunctionBase,
    IOpaqueScope,
    IInstantiable,
):

    @classmethod
    def expr_type(cls) -> IType:
        return INode.as_type()

    def prepare_expr(self, info_with_stats: RunInfoWithStats):
        expr = self.expr.apply()
        return RunInfoResult.with_args(
            run_info=info_with_stats,
            return_value=expr,
        )

class FunctionWrapper(
    ScopedFunctionBase,
    IInnerScope,
    IInstantiable,
):

    @classmethod
    def expr_type(cls) -> IType:
        return FunctionCall.as_type()

    def prepare_expr(self, info_with_stats: RunInfoWithStats):
        new_info = info_with_stats.add_scope(ScopeDataFutureParamItemGroup())
        return self.expr.apply().run(new_info)

###########################################################
################### BASIC BOOLEAN NODES ###################
###########################################################

class BaseIntBoolean(BaseInt, IBoolean, IDefault, ABC):

    @classmethod
    def create(cls) -> typing.Self:
        return cls(0)

    @classmethod
    def create_true(cls) -> typing.Self:
        return cls(1)

    @property
    def as_bool(self) -> bool:
        if self.as_int == 0:
            return False
        if self.as_int == 1:
            return True
        raise InvalidNodeException(BooleanExceptionInfo(self))

    def _strict_validate(self):
        alias_info = super()._strict_validate()
        Or(Eq(self, self.func(0)), Eq(self, self.func(1))).raise_on_false()
        return alias_info

class IntBoolean(BaseIntBoolean, IInstantiable):
    pass

class RunnableBoolean(InheritableNode, IDynamic, IBoolean, ABC):

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol.with_args(
            cls.args_type_group(),
            IntBoolean.as_type(),
        )

    def validate_result(self, result: INode, args_group: OptionalValueGroup):
        IsInstance.verify(result, IntBoolean)

    @classmethod
    def args_type_group(cls) -> IBaseTypeGroup:
        raise NotImplementedError

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = self._base_run(info).as_tuple
        info_with_stats, node_aux = base_result.as_tuple
        if info_with_stats.is_future():
            return base_result

        node = node_aux.real(self.__class__)
        value = node.func(*node.args).strict_bool

        result = info_with_stats.to_result(IntBoolean.from_bool(value))
        return RunInfoFullResult(result, arg_group)

class BooleanWrapper(
    RunnableBoolean,
    ISingleChild[IBoolean],
    ABC,
):

    idx_value = 1

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(IBoolean.as_type())

    @property
    def raw_child(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    @property
    def child(self) -> IBoolean:
        return self.raw_child.apply().real(IBoolean)

    @property
    def as_bool(self) -> bool:
        return self.child.as_bool

###########################################################
######################## EXCEPTION ########################
###########################################################

class NodeReturnException(Exception):

    def __init__(self, result: RunInfoFullResult):
        super().__init__(result)

    @property
    def result(self) -> RunInfoFullResult:
        result = self.args[0]
        IsInstance.verify(result, RunInfoFullResult)
        assert isinstance(result, RunInfoFullResult)
        return result

    def add_stats(self, info_with_stats: RunInfoWithStats) -> typing.Self:
        new_result = self.result.add_stats(info_with_stats)
        return self.__class__(new_result)

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
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_value = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(
            CountableTypeGroup(IBoolean.as_type()),
        )

    @property
    def value(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

class TypeExceptionInfo(
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_type = 1
    idx_node = 2
    idx_alias_info = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IType.as_type(),
            INode.as_type(),
            AliasInfo.as_type(),
        ))

    @property
    def type(self):
        return self.inner_arg(self.idx_type)

    @property
    def node(self):
        return self.inner_arg(self.idx_node)

    @property
    def alias_info(self):
        return self.inner_arg(self.idx_alias_info)

class TypeAcceptExceptionInfo(
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_type_that_accepts = 1
    idx_type_to_accept = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IType.as_type(),
            IType.as_type(),
        ))

    @property
    def type_that_accepts(self):
        return self.inner_arg(self.idx_type_that_accepts)

    @property
    def type_to_accept(self):
        return self.inner_arg(self.idx_type_to_accept)

class TypeAliasIndexExceptionInfo(
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_type_index = 1
    idx_type_alias = 2
    idx_invalid_inner_type_indices = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            BaseTypeIndex.as_type(),
            TypeAlias.as_type(),
            CompositeType(
                DefaultGroup.as_type(),
                RestTypeGroup.with_node(BaseTypeIndex.as_type()),
            )
        ))

    @property
    def type_index(self):
        return self.inner_arg(self.idx_type_index)

    @property
    def type_alias(self):
        return self.inner_arg(self.idx_type_alias)

    @property
    def invalid_inner_type_indices(self):
        return self.inner_arg(self.idx_invalid_inner_type_indices)

class TypeIndexProtocolExceptionInfo(
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_type_index = 1
    idx_protocol_arg = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            BaseTypeIndex.as_type(),
            Protocol.as_type(),
        ))

    @property
    def type_index(self):
        return self.inner_arg(self.idx_type_index)

    @property
    def protocol_arg(self):
        return self.inner_arg(self.idx_protocol_arg)

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
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            INode.as_type(),
            RunInfo.as_type(),
        ))

    @property
    def node(self):
        return self.inner_arg(self.idx_node)

    @property
    def run_info(self):
        return self.inner_arg(self.idx_run_info)

    @classmethod
    def with_args(cls, node: INode, run_info: RunInfo) -> typing.Self:
        return cls(node, run_info)

class StackExceptionInfoRunItem(
    InheritableNode,
    IStackExceptionInfoItem,
    ISingleChild[INode],
    IInstantiable,
):

    idx_node = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            INode.as_type(),
        ))

    @property
    def node(self):
        return self.inner_arg(self.idx_node)

class IStackNodeArg(INode, ABC):
    pass

class StackNodeFullArg(
    InheritableNode,
    IStackNodeArg,
    IFromSingleNode[INode],
    IInstantiable,
):

    idx_full_arg = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            INode.as_type(),
        ))

    @property
    def full_arg(self):
        return self.inner_arg(self.idx_full_arg)

class StackNodeSimplifiedArg(
    InheritableNode,
    IStackNodeArg,
    IFromSingleNode[INode],
    IInstantiable,
):

    idx_arg_type = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IType.as_type(),
        ))

    @property
    def arg_type(self):
        return self.inner_arg(self.idx_arg_type)

class StackExceptionInfoSimplifiedItem(
    InheritableNode,
    IStackExceptionInfoItem,
    IFromSingleNode[InheritableNode],
    IInstantiable,
):

    idx_node = 1
    idx_inner_node_args = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            INode.as_type(),
            BaseGroup.as_type(),
        ))

    @property
    def node(self):
        return self.inner_arg(self.idx_node)

    @property
    def inner_node_args(self):
        return self.inner_arg(self.idx_inner_node_args)

    @classmethod
    def arg_wrapper(cls, arg: INode) -> IStackNodeArg:
        return (
            StackNodeFullArg(arg)
            if (isinstance(arg, (BaseIntGroup, ISpecialValue)))
            else StackNodeSimplifiedArg(arg.as_node.as_type())
        )

    @classmethod
    def with_node(cls, node: InheritableNode) -> typing.Self:
        return cls(
            node.as_node.as_type(),
            DefaultGroup.from_items([
                cls.arg_wrapper(arg)
            for arg in node.args
            ]),
        )

class StackExceptionInfoGroup(
    BaseGroup[IStackExceptionInfoItem],
    IInstantiable,
):

    @classmethod
    def item_type(cls) -> TypeNode:
        return IStackExceptionInfoItem.as_type()

class StackExceptionInfo(
    InheritableNode,
    IExceptionInfo,
    IInstantiable,
):

    idx_stack_group = 1
    idx_cause = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            StackExceptionInfoGroup.as_type(),
            IExceptionInfo.as_type(),
        ))

    @property
    def stack_group(self):
        return self.inner_arg(self.idx_stack_group)

    @property
    def cause(self):
        return self.inner_arg(self.idx_cause)

    def _add_stack(self, node: INode, run_info: RunInfo) -> typing.Self:
        stack_group = self.stack_group.apply().real(StackExceptionInfoGroup)
        new_item = node if isinstance(node, StackExceptionInfoRunItem) else (
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
        IsInstance.verify(info, IExceptionInfo)
        assert isinstance(info, IExceptionInfo)
        return info

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
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(IExceptionInfo.as_type()))

    @property
    def info(self):
        return self.inner_arg(self.idx_info)

    @property
    def child(self) -> IExceptionInfo:
        return self.info.apply().real(IExceptionInfo)

###########################################################
################### CORE BOOLEAN NODES ####################
###########################################################

class Not(BooleanWrapper, IInstantiable):

    @property
    def as_bool(self) -> bool:
        child = self.raw_child.apply()
        if not isinstance(child, IBoolean):
            return False
        return not child.as_bool

class SingleOptionalBooleanChildWrapper(
    RunnableBoolean,
    ISingleOptionalChild[INode],
    ABC,
):

    idx_value = 1

    @classmethod
    def item_type(cls) -> TypeNode:
        return INode.as_type()

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(cls.item_type()),
            ),
        )

    @property
    def raw_child(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    @property
    def child(self) -> Optional[INode]:
        return self.raw_child.apply().real(Optional[INode])

class IsEmpty(SingleOptionalBooleanChildWrapper, IInstantiable):

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
    def as_bool(self) -> bool:
        value = self.inner_arg(self.idx_value).apply()
        if not isinstance(value, IOptional):
            return False
        return value.value is None

class IsInstance(RunnableBoolean, typing.Generic[T], IInstantiable):

    idx_instance = 1
    idx_type = 2

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            INode.as_type(),
            TypeNode.as_type(),
        )

    @property
    def instance(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_instance)

    @property
    def type(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_type)

    @property
    def as_bool(self) -> bool:
        instance = self.instance.apply()
        if not isinstance(instance, INode):
            return False
        t = self.type.apply()
        if not isinstance(t, TypeNode):
            return False
        alias_info = AliasInfo.with_node(TypeAliasGroup())
        valid, _ = t.valid(instance, alias_info=alias_info)
        return valid

    @property
    def as_type_or_raise(self) -> T:
        if not self.as_bool:
            raise self.to_exception()
        return typing.cast(T, self.type.apply())

    @classmethod
    def with_args(cls, instance: INode, t: typing.Type[T]) -> IsInstance[T]:
        return cls(instance, TypeNode(t)).real(IsInstance[T])

    @classmethod
    def assert_type(cls, instance: typing.Any, t: typing.Type[T]) -> T:
        return cls(instance, TypeNode(t)).as_type_or_raise

    @classmethod
    def verify(cls, instance: typing.Any, t: typing.Type[INode]):
        cls(instance, t.as_type()).raise_on_false()

class Eq(RunnableBoolean, IInstantiable):

    idx_left = 1
    idx_right = 2

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            INode.as_type(),
            INode.as_type(),
        )

    @property
    def left(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_left)

    @property
    def right(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_right)

    @property
    def as_bool(self) -> bool:
        left = self.left.apply()
        right = self.right.apply()
        return left == right

    @classmethod
    def from_ints(cls, left: int, right: int) -> Eq:
        return cls(Integer(left), Integer(right))

class SameArgsAmount(RunnableBoolean, IInstantiable):

    idx_left = 1
    idx_right = 2

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            INode.as_type(),
            INode.as_type(),
        )

    @property
    def left(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_left)

    @property
    def right(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_right)

    @property
    def as_bool(self) -> bool:
        left = self.left.apply()
        right = self.right.apply()
        l_amount = len(left.as_node.args)
        r_amount = len(right.as_node.args)
        return l_amount == r_amount

class MultiArgBooleanNode(RunnableBoolean, ABC):

    @classmethod
    def args_type_group(cls):
        return RestTypeGroup(IBoolean.as_type())

class DoubleIntBooleanNode(RunnableBoolean, ABC):

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            IInt.as_type(),
            IInt.as_type(),
        )

    @classmethod
    def with_args(cls, value_1: int, value_2: int) -> DoubleIntBooleanNode:
        return cls(Integer(value_1), Integer(value_2))

class And(MultiArgBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool:
        args = self.args
        for arg in args:
            if not isinstance(arg, IBoolean):
                return False
            elif not arg.as_bool:
                return False
        return True

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        if info.is_future():
            return self._base_run(info)

        run_args: list[INode] = []

        def fn_return(result: RunInfoResult) -> RunInfoFullResult:
            all_run_args = run_args + [None] * (len(self.args) - len(run_args))
            arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
                all_run_args)
            return RunInfoFullResult(result, arg_group)

        info_with_stats = info.with_stats()

        for arg in self.args:
            info_with_stats, run_arg = arg.as_node.run(info_with_stats).as_tuple
            if isinstance(run_arg, IBoolean):
                if not run_arg.as_bool:
                    result = info_with_stats.to_result(IBoolean.false())
                    return fn_return(result)
            run_args.append(run_arg)

        for run_arg in run_args:
            IsInstance.verify(run_arg, IBoolean)
            assert isinstance(run_arg, IBoolean)

        result = info_with_stats.to_result(IBoolean.true())

        return fn_return(result)

class Or(MultiArgBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool:
        args = self.args
        for arg in args:
            if isinstance(arg, IBoolean) and arg.as_bool:
                return True
        return False

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        if info.is_future():
            return self._base_run(info)

        run_args: list[INode] = []

        def fn_return(result: RunInfoResult) -> RunInfoFullResult:
            all_run_args = run_args + [None] * (len(self.args) - len(run_args))
            arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
                all_run_args)
            return RunInfoFullResult(result, arg_group)

        info_with_stats = info.with_stats()

        for arg in self.args:
            info_with_stats, run_arg = arg.as_node.run(info_with_stats).as_tuple
            if isinstance(run_arg, IBoolean):
                val_1 = run_arg.as_bool
                if val_1 is True:
                    result = info_with_stats.to_result(IBoolean.true())
                    return fn_return(result)
            run_args.append(run_arg)

        for run_arg in run_args:
            IsInstance.verify(run_arg, IBoolean)
            assert isinstance(run_arg, IBoolean)

        result = info_with_stats.to_result(IBoolean.false())

        return fn_return(result)

class ComparableEq(RunnableBoolean, IInstantiable):

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            IComparable.as_type(),
            IComparable.as_type(),
        )

    @property
    def as_bool(self) -> bool:
        args = self.args
        Eq.from_ints(len(args), 2).raise_on_false()
        assert len(args) == 2
        a, b = args
        IsInstance.assert_type(a, IComparable)
        assert isinstance(a, IComparable)
        IsInstance.assert_type(b, IComparable)
        assert isinstance(b, IComparable)
        return a.eq(b).as_bool

    @classmethod
    def with_ints(cls, value1: int, value2: int) -> typing.Self:
        return cls(Integer(value1), Integer(value2))

class GreaterThan(RunnableBoolean, IInstantiable):

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            IComparable.as_type(),
            IComparable.as_type(),
        )

    @property
    def as_bool(self) -> bool:
        args = self.args
        Eq.from_ints(len(args), 2).raise_on_false()
        assert len(args) == 2
        a, b = args
        IsInstance.assert_type(a, IComparable)
        assert isinstance(a, IComparable)
        IsInstance.assert_type(b, IComparable)
        assert isinstance(b, IComparable)
        return a.gt(b).as_bool

    @classmethod
    def with_ints(cls, value1: int, value2: int) -> typing.Self:
        return cls(Integer(value1), Integer(value2))

class LessThan(RunnableBoolean, IInstantiable):

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            IComparable.as_type(),
            IComparable.as_type(),
        )

    @property
    def as_bool(self) -> bool:
        args = self.args
        Eq.from_ints(len(args), 2).raise_on_false()
        assert len(args) == 2
        a, b = args
        IsInstance.assert_type(a, IComparable)
        assert isinstance(a, IComparable)
        IsInstance.assert_type(b, IComparable)
        assert isinstance(b, IComparable)
        return a.lt(b).as_bool

    @classmethod
    def with_ints(cls, value1: int, value2: int) -> typing.Self:
        return cls(Integer(value1), Integer(value2))

class GreaterOrEqual(RunnableBoolean, IInstantiable):

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            IComparable.as_type(),
            IComparable.as_type(),
        )

    @property
    def as_bool(self) -> bool:
        args = self.args
        Eq.from_ints(len(args), 2).raise_on_false()
        assert len(args) == 2
        a, b = args
        IsInstance.assert_type(a, IComparable)
        assert isinstance(a, IComparable)
        IsInstance.assert_type(b, IComparable)
        assert isinstance(b, IComparable)
        return a.ge(b).as_bool

    @classmethod
    def with_ints(cls, value1: int, value2: int) -> typing.Self:
        return cls(Integer(value1), Integer(value2))

class LessOrEqual(RunnableBoolean, IInstantiable):

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            IComparable.as_type(),
            IComparable.as_type(),
        )

    @property
    def as_bool(self) -> bool:
        args = self.args
        Eq.from_ints(len(args), 2).raise_on_false()
        assert len(args) == 2
        a, b = args
        IsInstance.assert_type(a, IComparable)
        assert isinstance(a, IComparable)
        IsInstance.assert_type(b, IComparable)
        assert isinstance(b, IComparable)
        return a.le(b).as_bool

    @classmethod
    def with_ints(cls, value1: int, value2: int) -> typing.Self:
        return cls(Integer(value1), Integer(value2))

class IsInsideRange(RunnableBoolean, IInstantiable):

    idx_value = 1
    idx_min_value = 2
    idx_max_value = 3

    @classmethod
    def args_type_group(cls):
        return CountableTypeGroup(
            IInt.as_type(),
            IInt.as_type(),
            IInt.as_type(),
        )

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
    def as_bool(self) -> bool:
        value = self.raw_value.apply()
        IsInstance.assert_type(value, IInt)
        assert isinstance(value, IInt)
        min_value = self.min_value.apply()
        IsInstance.assert_type(min_value, IInt)
        assert isinstance(min_value, IInt)
        max_value = self.max_value.apply()
        IsInstance.assert_type(max_value, IInt)
        assert isinstance(max_value, IInt)
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

    @classmethod
    def is_future(cls) -> bool:
        raise NotImplementedError

    def define_item(self, index: PlaceholderIndex, value: INode) -> typing.Self:
        self.is_dynamic().raise_on_false()
        items = list(self.as_tuple)
        if index.as_int > len(items):
            items = items + [Optional()] * (index.as_int - len(items))
        items[index.as_int - 1] = Optional(value)
        return self.func(*items)

class PlaceholderIndex(
    NodeArgBaseIndex,
    ITypedIndex[ScopeDataPlaceholderItemGroup, Placeholder],
    IInstantiable,
):

    @classmethod
    def outer_type(cls):
        return ScopeDataPlaceholderItemGroup

    @classmethod
    def item_type(cls) -> TypeNode:
        return Placeholder.as_type()

    def find_in_outer_node(self, node: ScopeDataPlaceholderItemGroup):
        IsInstance.assert_type(node, ScopeDataPlaceholderItemGroup)
        assert isinstance(node, ScopeDataPlaceholderItemGroup)
        return self.find_in_node(node)

    def replace_in_outer_target(self, target: ScopeDataPlaceholderItemGroup, new_node: Placeholder):
        IsInstance.assert_type(target, ScopeDataPlaceholderItemGroup)
        assert isinstance(target, ScopeDataPlaceholderItemGroup)
        IsInstance.assert_type(new_node, Placeholder)
        assert isinstance(new_node, Placeholder)
        return self.replace_in_target(target, new_node)

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

    @classmethod
    def is_future(cls) -> bool:
        return False

class ScopeDataFutureParamItemGroup(
    ScopeDataParamBaseItemGroup,
    IScopeDataFutureItemGroup,
    IInstantiable,
):

    @classmethod
    def is_future(cls) -> bool:
        return True

class ScopeDataVarBaseItemGroup(ScopeDataPlaceholderItemGroup, ABC):

    @classmethod
    def item_inner_type(cls):
        return Var

    @classmethod
    def is_dynamic(cls) -> IBoolean:
        return IBoolean.true()

class ScopeDataVarItemGroup(ScopeDataVarBaseItemGroup, IScopeDataActualItemGroup, IInstantiable):

    @classmethod
    def is_future(cls) -> bool:
        return False

class ScopeDataFutureVarItemGroup(
    ScopeDataVarBaseItemGroup,
    IScopeDataFutureItemGroup,
    IInstantiable,
):

    @classmethod
    def is_future(cls) -> bool:
        return True

class ScopeDataGroup(BaseGroup[ScopeDataPlaceholderItemGroup], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return ScopeDataPlaceholderItemGroup.as_type()

    def is_future(self):
        items = self.as_tuple
        return any([isinstance(item, IScopeDataFutureItemGroup) for item in items])

    def add_item(self, item: ScopeDataPlaceholderItemGroup) -> typing.Self:
        items = list(self.as_tuple)
        items.append(item)
        return self.func(*items)
class RunInfo(InheritableNode, IInstantiable):

    idx_scope_data_group = 1
    idx_return_after_scope = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            ScopeDataGroup.as_type(),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(RunInfoScopeDataIndex.as_type()),
            ),
        ))

    @property
    def scope_data_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_scope_data_group)

    @property
    def return_after_scope(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_return_after_scope)

    def add_scope(self, item: ScopeDataPlaceholderItemGroup) -> typing.Self:
        group = self.scope_data_group.apply().real(ScopeDataGroup)
        new_group = group.add_item(item)
        return self.with_new_args(
            scope_data_group=new_group,
            return_after_scope=Optional())

    @classmethod
    def with_args(
        cls,
        scope_data_group: ScopeDataGroup,
        return_after_scope: Optional[RunInfoScopeDataIndex],
    ) -> typing.Self:
        return cls(scope_data_group, return_after_scope)

    def with_new_args(
        self,
        scope_data_group: ScopeDataGroup | None = None,
        return_after_scope: Optional[RunInfoScopeDataIndex] | None = None,
    ) -> typing.Self:
        scope_data_group = (
            scope_data_group
            if scope_data_group is not None
            else self.scope_data_group.apply().real(ScopeDataGroup))
        return_after_scope = (
            return_after_scope
            if return_after_scope is not None
            else self.return_after_scope.apply().real(Optional[RunInfoScopeDataIndex]))
        return self.func(scope_data_group, return_after_scope)

    def add_scope_var(
        self,
        scope_index: RunInfoScopeDataIndex,
        item_index: PlaceholderIndex,
        value: INode,
    ) -> RunInfo:
        IsInstance.assert_type(scope_index, RunInfoScopeDataIndex)
        assert isinstance(scope_index, RunInfoScopeDataIndex)
        IsInstance.assert_type(item_index, PlaceholderIndex)
        assert isinstance(item_index, PlaceholderIndex)

        item = scope_index.find_in_outer_node(self).value_or_raise
        IsInstance.assert_type(item, ScopeDataVarItemGroup)
        assert isinstance(item, ScopeDataVarItemGroup)

        new_item = item.define_item(item_index, value)
        result = scope_index.replace_in_outer_target(self, new_item).value_or_raise

        group_1 = self.scope_data_group.apply().real(ScopeDataGroup)
        group_2 = result.scope_data_group.apply().real(ScopeDataGroup)
        SameArgsAmount(group_1, group_2).raise_on_false()

        return result

    def is_future(self) -> IBoolean:
        group = self.scope_data_group.apply().real(ScopeDataGroup)
        return group.is_future()

    def with_scopes(self, info_base: RunInfo) -> typing.Self:
        base_group = info_base.scope_data_group.apply().real(ScopeDataGroup)
        base_amount = len(base_group.as_tuple)
        group = self.scope_data_group.apply().real(ScopeDataGroup).as_tuple
        Not(LessThan(Integer(len(group)), Integer(base_amount))).raise_on_false()
        if len(base_group.as_tuple) == 0:
            new_group = base_group
        else:
            last_base = base_group.as_tuple[-1]
            last_after = group[base_amount-1]
            Eq(
                last_base.item_inner_type().as_type(),
                last_after.item_inner_type().as_type(),
            ).raise_on_false()
            if not last_base.is_future() and last_after.is_future():
                new_group = base_group
            else:
                Eq(last_base.func.as_type(), last_after.func.as_type()).raise_on_false()
                new_items = list(base_group.as_tuple[:-1]) + [last_after]
                new_group = base_group.func(*new_items)
        return_after_scope = self.return_after_scope.apply().real(Optional[RunInfoScopeDataIndex])
        return_after_val = return_after_scope.value
        if return_after_val is not None:
            IsInstance.assert_type(return_after_val, RunInfoScopeDataIndex)
            assert isinstance(return_after_val, RunInfoScopeDataIndex)
            ret_index = return_after_val.as_int
            if ret_index > base_amount:
                return_after_scope = Optional()
        return self.with_new_args(
            scope_data_group=new_group,
            return_after_scope=return_after_scope)

    def must_return(self) -> bool:
        return_after_scope = self.return_after_scope.apply().real(Optional[RunInfoScopeDataIndex])
        return_after_val = return_after_scope.value
        return return_after_val is not None

    def with_stats(self) -> RunInfoWithStats:
        return RunInfoWithStats.with_args(
            run_info=self,
            stats=RunInfoStats.create(),
        )

class RunInfoStats(InheritableNode, IDefault, IAdditive, IInstantiable):

    idx_instructions = 1
    idx_memory = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Integer.as_type(),
            Integer.as_type(),
        ))

    @property
    def instructions(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_instructions)

    @property
    def memory(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_memory)

    @classmethod
    def create(cls) -> typing.Self:
        return cls.with_args()

    @classmethod
    def with_args(
        cls,
        instructions: int = 0,
        memory: int = 0,
    ) -> typing.Self:
        return cls(Integer(instructions), Integer(memory))

    def add(self, another: INode) -> typing.Self:
        IsInstance.assert_type(another, RunInfoStats)
        assert isinstance(another, RunInfoStats)
        instructions = self.instructions.apply().real(Integer)
        memory = self.memory.apply().real(Integer)
        another_instructions = another.instructions.apply().real(Integer)
        another_memory = another.memory.apply().real(Integer)
        new_instructions = instructions.as_int + another_instructions.as_int
        new_memory = memory.as_int + another_memory.as_int
        return self.with_args(
            instructions=new_instructions,
            memory=new_memory,
        )

class RunInfoWithStats(InheritableNode, IInstantiable):

    idx_run_info = 1
    idx_stats = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            RunInfo.as_type(),
            RunInfoStats.as_type(),
        ))

    @property
    def run_info(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_run_info)

    @property
    def stats(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_stats)

    @classmethod
    def with_args(
        cls,
        run_info: RunInfo,
        stats: RunInfoStats,
    ) -> typing.Self:
        return cls(run_info, stats)

    def is_future(self) -> IBoolean:
        run_info = self.run_info.apply().real(RunInfo)
        group = run_info.scope_data_group.apply().real(ScopeDataGroup)
        return group.is_future()

    def to_result(self, result: INode) -> RunInfoResult:
        return RunInfoResult.with_args(
            run_info=self,
            return_value=result)

    def add_stats(self, info_with_stats: typing.Self) -> typing.Self:
        stats_to_add = info_with_stats.stats.apply().real(RunInfoStats)
        return self.add_inner_stats(stats_to_add)

    def add_inner_stats(self, stats_to_add: RunInfoStats) -> typing.Self:
        run_info = self.run_info.apply().real(RunInfo)
        stats = self.stats.apply().real(RunInfoStats)
        new_stats = stats.add(stats_to_add)
        return self.with_args(run_info, new_stats)

    def with_scopes(self, info_base: RunInfo) -> typing.Self:
        run_info = self.run_info.apply().real(RunInfo)
        new_run_info = run_info.with_scopes(info_base)
        return self.with_args(
            run_info=new_run_info,
            stats=self.stats.apply().real(RunInfoStats),
        )

    def add_scope(
        self,
        item: ScopeDataPlaceholderItemGroup,
    ) -> typing.Self:
        run_info = self.run_info.apply().real(RunInfo)
        new_run_info = run_info.add_scope(item)
        return self.with_args(
            run_info=new_run_info,
            stats=self.stats.apply().real(RunInfoStats),
        )

    def add_scope_var(
        self,
        scope_index: RunInfoScopeDataIndex,
        item_index: PlaceholderIndex,
        value: INode,
    ) -> typing.Self:
        run_info = self.run_info.apply().real(RunInfo)
        new_run_info = run_info.add_scope_var(scope_index, item_index, value)
        return self.with_args(
            run_info=new_run_info,
            stats=self.stats.apply().real(RunInfoStats),
        )

    def with_return(self, return_after_scope: Optional[RunInfoScopeDataIndex]) -> typing.Self:
        run_info = self.run_info.apply().real(RunInfo)
        new_run_info = run_info.with_new_args(
            return_after_scope=return_after_scope)
        return self.with_args(
            run_info=new_run_info,
            stats=self.stats.apply().real(RunInfoStats),
        )

    def must_return(self) -> bool:
        run_info = self.run_info.apply().real(RunInfo)
        return run_info.must_return()

    def get_stats(self) -> RunInfoStats:
        return self.stats.apply().real(RunInfoStats)

class RunInfoScopeDataIndex(
    BaseInt,
    ITypedIntIndex[RunInfo, ScopeDataPlaceholderItemGroup],
    IInstantiable,
):

    @classmethod
    def outer_type(cls) -> type[RunInfo]:
        return RunInfo

    @classmethod
    def item_type(cls) -> TypeNode:
        return ScopeDataPlaceholderItemGroup.as_type()

    @classmethod
    def _outer_group(cls, run_info: RunInfo) -> ScopeDataGroup:
        return run_info.scope_data_group.apply().real(ScopeDataGroup)

    @classmethod
    def _update(
        cls,
        target: RunInfo,
        group_opt: IOptional[ScopeDataGroup],
    ) -> IOptional[RunInfo]:
        group = group_opt.value
        if group is None:
            return Optional.create()
        IsInstance.assert_type(group, ScopeDataGroup)
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

class RunInfoResult(InheritableNode, IInstantiable):

    idx_run_info = 1
    idx_return_value = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            RunInfoWithStats.as_type(),
            INode.as_type(),
        ))

    @property
    def run_info(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_run_info)

    @property
    def return_value(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_return_value)

    @property
    def as_tuple(self):
        new_info = self.run_info.apply().real(RunInfoWithStats)
        return_value = self.return_value.apply().cast(INode)
        return new_info, return_value

    @classmethod
    def with_args(
        cls,
        run_info: RunInfoWithStats,
        return_value: INode,
    ) -> RunInfoResult:
        return cls(run_info, return_value)

    def add_stats(self, info_with_stats: RunInfoWithStats) -> RunInfoResult:
        run_info = self.run_info.apply().real(RunInfoWithStats)
        new_run_info = run_info.add_stats(info_with_stats)
        return self.with_args(
            run_info=new_run_info,
            return_value=self.return_value.apply().cast(INode),
        )

class RunInfoFullResult(InheritableNode, IInstantiable):

    idx_info_result = 1
    idx_run_args = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            RunInfoResult.as_type(),
            OptionalValueGroup.as_type(),
        ))

    @property
    def info_result(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_info_result)

    @property
    def run_args(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_run_args)

    @property
    def as_tuple(self):
        info_result = self.info_result.apply().real(RunInfoResult)
        return_value = self.run_args.apply().real(OptionalValueGroup)
        return info_result, return_value

    @classmethod
    def with_args(
        cls,
        info_result: RunInfoResult,
        run_args: OptionalValueGroup[INode],
    ) -> typing.Self:
        return cls(info_result, run_args)

    def with_new_args(
        self,
        info_result: RunInfoResult | None = None,
        run_args: OptionalValueGroup[INode] | None = None,
    ) -> typing.Self:
        info_result = (
            info_result
            if info_result is not None
            else self.info_result.apply().real(RunInfoResult))
        run_args = (
            run_args
            if run_args is not None
            else self.run_args.apply().real(OptionalValueGroup[INode]))
        return self.func(info_result, run_args)

    def add_stats(self, info_with_stats: RunInfoWithStats) -> typing.Self:
        info_result, _ = self.as_tuple
        info_result = info_result.add_stats(info_with_stats)
        return self.with_new_args(
            info_result=info_result,
        )

class ControlFlowBaseNode(InheritableNode, IDynamic, ABC):

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()
        if info.is_future():
            args: list[INode] = []
            info_with_stats = info.with_stats()
            for arg in self.args:
                info_with_stats, node_aux = arg.as_node.run(info_with_stats).as_tuple
                args.append(node_aux)
            result = info_with_stats.to_result(self.func(*args))
            arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(args)
            return RunInfoFullResult(result, arg_group)
        return self._run_control(info)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        raise NotImplementedError(self.__class__)

    def validate_result(self, result: INode, args_group: OptionalValueGroup):
        protocol = self.protocol()
        alias_info = protocol.verify_optional_args(args_group)
        alias_group_actual = alias_info.alias_group_actual.apply().real(TypeAliasOptionalGroup)
        p_arg_group = protocol.arg_group.apply().real(IBaseTypeGroup)
        p_result = protocol.result.apply().real(IType)
        for alias in p_arg_group.as_node.find(BaseTypeIndex):
            value = alias_group_actual.as_tuple[alias.as_int-1]
            if value.is_empty().as_bool:
                p_result = alias.replace_in_node(p_result, InvalidType()).real(IType)
        protocol = protocol.with_new_args(result_type=p_result)
        protocol.verify_result(result, alias_info=alias_info)

class BaseNormalizer(ControlFlowBaseNode, ABC):

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        default_result = self._base_run(info)
        if info.is_future():
            return default_result
        result_aux, arg_group = default_result.as_tuple
        info_with_stats, node_aux = result_aux.as_tuple
        node = node_aux.real(self.__class__)
        result = info_with_stats.to_result(node.normalize())
        return RunInfoFullResult(result, arg_group)

    def normalize(self) -> INode:
        raise NotImplementedError(self.__class__)

    def validate_result(self, result: INode, args_group: OptionalValueGroup):
        IsInstance.verify(result, BaseNode)

class BaseNormalizerGroup(BaseNormalizer, IGroup[T], typing.Generic[T], ABC):

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            RestTypeGroup(cls.item_type()),
            CompositeType(
                cls.as_type(),
                RestTypeGroup(cls.item_type()),
            ),
        )

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

    def _strict_validate(self):
        alias_info = super()._strict_validate()
        t = self.item_type().type
        for arg in self.args:
            origin = typing.get_origin(t)
            t = origin if origin is not None else t
            IsInstance.assert_type(arg, t)
            assert isinstance(arg, t), f'{type(arg)} != {t}'
        return alias_info

    def to_optional_group(self) -> OptionalValueGroup[T]:
        return OptionalValueGroup.from_optional_items(self.args)

class FunctionCall(ControlFlowBaseNode, IInstantiable):

    idx_function = 1
    idx_arg_group = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol.with_args(
            CountableTypeGroup(
                IFunction.as_type(),
                BaseGroup.as_type(),
            ),
            INode.as_type(),
        )

    @property
    def function(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_function)

    @property
    def arg_group(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_group)

    @classmethod
    def define(cls, fn: IFunction, args: BaseGroup) -> INode:
        return cls(fn, args)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.function.apply().run(info_with_stats).as_tuple
        fn = node_aux.real(IFunction)

        info_with_stats, node_aux = self.arg_group.apply().run(info_with_stats).as_tuple
        fn_arg_group = node_aux.real(BaseGroup)

        result = fn.with_arg_group(group=fn_arg_group, info=info)
        result = result.add_stats(info_with_stats)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [fn, fn_arg_group])
        return RunInfoFullResult(result, arg_group)

class If(ControlFlowBaseNode, IInstantiable):

    idx_condition = 1
    idx_true_expr = 2
    idx_false_expr = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
                TypeAlias(INode.as_type()),
            ),
            CountableTypeGroup(
                IBoolean.as_type(),
                TypeIndex(1),
                TypeIndex(2),
            ),
            UnionType(TypeIndex(1), TypeIndex(2)),
        )

    @property
    def condition(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_condition)

    @property
    def true_expr(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_true_expr)

    @property
    def false_expr(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_false_expr)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.condition.apply().run(info_with_stats).as_tuple
        condition = node_aux.real(IBoolean)

        flag = condition.strict_bool

        if flag:
            true_result = self.true_expr.apply().run(info_with_stats)
            _, true_node = true_result.as_tuple
            arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
                [condition, true_node, None])
            return RunInfoFullResult(true_result, arg_group)

        false_result = self.false_expr.apply().run(info_with_stats)
        _, false_node = false_result.as_tuple
        arg_group = OptionalValueGroup.from_optional_items(
            [condition, None, false_node])
        return RunInfoFullResult(false_result, arg_group)

class LoopGuard(InheritableNode, IInstantiable):

    idx_condition = 1
    idx_result = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IBoolean.as_type(),
            INode.as_type(),
        ))

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
    def protocol(cls) -> Protocol:
        return Protocol.with_args(
            CountableTypeGroup(
                IFunction.as_type(),
                Optional.as_type(),
            ),
            INode.as_type(),
        )

    @property
    def callback(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_callback)

    @property
    def initial_data(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_initial_data)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.callback.apply().run(info_with_stats).as_tuple
        callback = node_aux.real(IFunction)

        info_with_stats, node_aux = self.initial_data.apply().run(info_with_stats).as_tuple
        initial_data = node_aux.real(Optional[INode])

        condition = True
        data = initial_data
        idx = 0
        while condition:
            idx += 1
            info_with_stats, node_aux = FunctionCall(
                callback,
                DefaultGroup(data),
            ).run(info_with_stats).as_tuple
            result = node_aux.real(LoopGuard)

            info_with_stats, node_aux = result.condition.apply().run(
                info_with_stats).as_tuple
            cond_node = node_aux.real(IBoolean)

            info_with_stats, new_data = result.result.apply().run(
                info_with_stats).as_tuple

            data = Optional(new_data)
            condition = cond_node.strict_bool

        result = info_with_stats.to_result(result=data)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [callback, initial_data]
        )
        return RunInfoFullResult(result, arg_group)

class InstructionGroup(ControlFlowBaseNode, IInnerScope, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol.with_args(
            alias_group=TypeAliasGroup(
                TypeAlias(INode.as_type()),
            ),
            args_type_group=RestTypeGroup(IRunnable.as_type()),
            result_type=TypeIndex(1),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        scope_data = (
            ScopeDataFutureVarItemGroup()
            if info.is_future()
            else ScopeDataVarItemGroup()
        )
        info = info.add_scope(scope_data)
        info_with_stats = info.with_stats()

        run_args: list[INode] = []
        for i, arg in enumerate(self.args):
            try:
                info_with_stats, node = arg.as_node.run(info_with_stats).as_tuple
            except InvalidNodeException as e:
                exc_info = e.info.add_stack(
                    node=StackExceptionInfoRunItem(arg),
                    run_info=info,
                )
                exc_info = exc_info.add_stack(
                    node=InnerArg(self, NodeArgIndex(i+1)),
                    run_info=info,
                )
                raise InvalidNodeException(exc_info) from e
            run_args.append(node)

        result = info_with_stats.to_result(Void())
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(run_args)
        return RunInfoFullResult(result, arg_group)

class Assign(ControlFlowBaseNode, IInstantiable):

    idx_var_index = 1
    idx_value = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
            ),
            CountableTypeGroup(
                PlaceholderIndex.as_type(),
                TypeIndex(1),
            ),
            TypeIndex(1),
        )

    @property
    def var_index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_var_index)

    @property
    def value(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.var_index.apply().run(info_with_stats).as_tuple
        var_index = node_aux.real(PlaceholderIndex)

        info_with_stats, node_aux = self.value.apply().run(info_with_stats).as_tuple
        value = node_aux

        info_with_stats, node_aux = NearParentScope.create().run(info_with_stats).as_tuple
        scope_index = node_aux.real(RunInfoScopeDataIndex)

        info_with_stats = info_with_stats.add_scope_var(
            scope_index=scope_index,
            item_index=var_index,
            value=value)

        result = info_with_stats.to_result(value)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [var_index, value]
        )
        return RunInfoFullResult(result, arg_group)

class Return(ControlFlowBaseNode, IFromSingleNode[INode], IInstantiable):

    idx_parent_scope = 1
    idx_value = 2

    @classmethod
    def with_node(cls, node: INode) -> typing.Self:
        return cls(FarParentScope.create(), node)

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
            ),
            CountableTypeGroup(
                RunInfoScopeDataIndex.as_type(),
                TypeIndex(1),
            ),
            TypeIndex(1),
        )

    @property
    def parent_scope(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_parent_scope)

    @property
    def value(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.parent_scope.apply().run(info_with_stats).as_tuple
        scope_index = node_aux.real(RunInfoScopeDataIndex)

        info_with_stats, value = self.value.apply().run(info_with_stats).as_tuple

        if not info_with_stats.is_future():
            info_with_stats = info_with_stats.with_return(Optional(scope_index))

        result = info_with_stats.to_result(value)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [scope_index, value]
        )
        return RunInfoFullResult(result, arg_group)

class InnerArg(ControlFlowBaseNode, IInstantiable):

    idx_node = 1
    idx_arg_index = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
            ),
            CountableTypeGroup(
                INode.as_type(),
                NodeArgIndex.as_type(),
            ),
            TypeIndex(1),
        )

    @property
    def node(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_node)

    @property
    def arg_index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_index)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.node.apply().run(info_with_stats).as_tuple
        node = node_aux.cast(INode)

        info_with_stats, node_aux = self.arg_index.apply().run(info_with_stats).as_tuple
        arg_index = node_aux.real(NodeArgIndex)

        result = arg_index.find_in_node(node).value_or_raise

        result = RunInfoResult.with_args(
            run_info=info_with_stats,
            return_value=result)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [node, arg_index]
        )
        return RunInfoFullResult(result, arg_group)

class NestedArg(ControlFlowBaseNode, IInstantiable):

    idx_node = 1
    idx_arg_indices = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INode.as_type()),
            ),
            CountableTypeGroup(
                INode.as_type(),
                NestedArgIndexGroup.as_type(),
            ),
            TypeIndex(1),
        )

    @classmethod
    def from_raw(cls, node: INode, indices: typing.Sequence[int]) -> typing.Self:
        return cls(node, NestedArgIndexGroup.from_ints(indices))

    @property
    def node(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_node)

    @property
    def arg_indices(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg_indices)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, root_node = self.node.apply().run(info_with_stats).as_tuple
        node = root_node

        info_with_stats, node_aux = self.arg_indices.apply().run(info_with_stats).as_tuple
        arg_indices = node_aux.real(NestedArgIndexGroup)

        for arg_index in arg_indices.as_tuple:
            info_with_stats, node = InnerArg(node, arg_index).run(info_with_stats).as_tuple

        result = RunInfoResult.with_args(
            run_info=info_with_stats,
            return_value=node)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [node, arg_indices]
        )
        return RunInfoFullResult(result, arg_group)

###########################################################
######################## ITERATOR #########################
###########################################################

class IIterator(INode, ABC):

    def next(self, info_with_stats: RunInfoWithStats) -> RunInfoResult:
        run_info = info_with_stats.run_info.apply().real(RunInfo)
        result = self._next(run_info)
        result.add_stats(info_with_stats)
        _, node = result.as_tuple
        IsInstance.assert_type(node, IOptional)
        assert isinstance(node, IOptional)
        group = node.value
        if group is not None:
            IsInstance.assert_type(group, DefaultGroup)
            assert isinstance(group, DefaultGroup)
            Eq.from_ints(len(group.args), 2).raise_on_false()
            assert len(group.args) == 2
            new_iter, _ = group.as_tuple
            IsInstance.assert_type(new_iter, self.__class__)
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
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            BaseGroup.as_type(),
            NodeArgIndex.as_type(),
        ))

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
        group = group if group is not None else self.group.apply().real(BaseGroup)
        index = index if index is not None else self.index.apply().real(NodeArgIndex)
        return self.func(group, index)

    def _next(self, run_info: RunInfo):
        info_with_stats = run_info.with_stats()

        info_with_stats, node_aux = self.group.apply().run(info_with_stats).as_tuple
        group = node_aux.real(BaseGroup)

        info_with_stats, node_aux = self.index.apply().run(info_with_stats).as_tuple
        index = node_aux.real(NodeArgIndex)

        if index.as_int > group.amount():
            result: Optional[INode] = Optional()
            return info_with_stats.to_result(result)

        value = index.find_in_node(group).value_or_raise
        new_iter = self.with_new_args(
            group=group,
            index=NodeArgIndex(index.as_int + 1),
        )
        result = Optional(DefaultGroup(new_iter, value))
        return info_with_stats.to_result(result)

class Next(ControlFlowBaseNode, IInstantiable):

    idx_iter = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol.with_args(
            CountableTypeGroup(IIterator.as_type()),
            INode.as_type(),
        )

    @property
    def iter(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_iter)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()
        info_with_stats, node_aux = self.iter.apply().run(info_with_stats).as_tuple
        iterator = node_aux.real(IIterator)
        result = iterator.next(info_with_stats)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [iterator]
        )
        return RunInfoFullResult(result, arg_group)

class Add(ControlFlowBaseNode, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(IAdditive.as_type()),
            ),
            RestTypeGroup(IAdditive.as_type()),
            TypeIndex(1),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        node: IAdditive | None = None
        run_args: list[INode] = []
        info_with_stats = info.with_stats()
        for arg in self.args:
            info_with_stats, node_aux = arg.as_node.run(info_with_stats).as_tuple
            run_args.append(node_aux)
            if node is None:
                arg_node = node_aux.real(IAdditive)
                node = arg_node
            else:
                node = node.add(node_aux)
        Optional.with_value(node).raise_if_empty()
        assert node is not None
        result = info_with_stats.to_result(node)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            run_args)
        return RunInfoFullResult(result, arg_group)

###########################################################
#################### ARITHMETIC NODES #####################
###########################################################

class INumber(IAdditive, ABC):

    @classmethod
    def zero(cls) -> BinaryInt:
        return BinaryInt(IBoolean.false())

    @classmethod
    def one(cls) -> BinaryInt:
        return BinaryInt(IBoolean.true())

    @classmethod
    def minus_one(cls) -> SignedInt:
        return SignedInt(NegativeSign(IBoolean.true()), cls.one())

    def subtract(self, value: INumber) -> INumber:
        raise NotImplementedError

    def multiply(self, value: INumber) -> INumber:
        raise NotImplementedError

class IComparableNumber(INumber, IComparable, ABC):
    pass

class ISign(INode, ABC):
    pass

class IComparableSign(ISign, ABC):
    pass

class ISignedNumber(INumber, ABC):

    @property
    def abs(self) -> BinaryInt:
        raise NotImplementedError

    @property
    def sign(self) -> ISign:
        raise NotImplementedError

class IComparableSignedNumber(ISignedNumber, IComparableNumber, ABC):

    @property
    def abs(self) -> BinaryInt:
        raise NotImplementedError

    @property
    def sign(self) -> IComparableSign:
        raise NotImplementedError

class NegativeSign(InheritableNode, IComparableSign, IDefault, IInstantiable):

    idx_negative = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IntBoolean.as_type(),
        ))

    @classmethod
    def create(cls):
        return cls(IBoolean.true())

    @property
    def negative(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_negative)

class IDivisible(INumber, ABC):

    def divide(self, value: IDivisible) -> IDivisible:
        raise NotImplementedError

    def divided_by(self, value: IDivisible) -> IDivisible:
        raise NotImplementedError

class BaseSignedInt(BaseNormalizer, IComparableSignedNumber, IDivisible, ABC):

    @property
    def abs(self) -> BinaryInt:
        raise NotImplementedError

    @property
    def sign(self) -> NegativeSign:
        raise NotImplementedError

    def precision(self) -> int:
        return len(self.abs.as_tuple)

    def with_new_abs(self, abs_value: BinaryInt) -> BaseSignedInt:
        raise NotImplementedError

    def normalize(self) -> BaseSignedInt:
        raise NotImplementedError

    def validate_result(self, result: INode, args_group: OptionalValueGroup):
        IsInstance.verify(result, BaseSignedInt)

    def subtract(self, value: INumber) -> BaseSignedInt:
        IsInstance.assert_type(value, BaseSignedInt)
        assert isinstance(value, BaseSignedInt)
        return self.add(
            SignedInt(
                NegativeSign.create(),
                value,
            ).normalize()
        )

    def multiply(self, value: INumber) -> BaseSignedInt:
        raise NotImplementedError

    def divide(self, value: IDivisible) -> IDivisible:
        raise NotImplementedError

    def divided_by(self, value: IDivisible) -> IDivisible:
        raise NotImplementedError

    def divide_int(self, value: BaseSignedInt) -> BaseSignedInt:
        raise NotImplementedError

    def modulo(self, value: BaseSignedInt) -> BinaryInt:
        raise NotImplementedError

    def to_rational(self) -> BaseSignedRational:
        return SignedRational(
            self.sign,
            Rational(self.abs, INumber.one()).normalize(),
        ).normalize()

class BinaryInt(BaseSignedInt, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.rest_protocol(IntBoolean.as_type())

    @property
    def abs(self):
        return self

    @property
    def sign(self):
        return NegativeSign(IBoolean.false())

    def with_new_abs(self, abs_value: BinaryInt) -> BaseSignedInt:
        return abs_value

    @classmethod
    def from_items(cls, items: typing.Sequence[IntBoolean]) -> typing.Self:
        return cls(*items).normalize()

    @property
    def as_tuple(self) -> tuple[IntBoolean, ...]:
        return typing.cast(tuple[IntBoolean, ...], self.args)

    def normalize(self) -> typing.Self:
        if self == INumber.zero():
            return self

        bits = self.as_tuple
        GreaterThan.with_ints(len(bits), 0).raise_on_false()
        assert len(bits) > 0
        if bits[0] == IBoolean.true():
            return self

        new_bits_list: list[IntBoolean] = []
        for i, bit in enumerate(bits):
            if bit == IBoolean.true():
                new_bits_list = list(bits[i:])
                break

        if len(new_bits_list) == 0:
            new_bits_list.append(IBoolean.false())

        return self.from_items(new_bits_list)

    def add(self, another: INode):
        if isinstance(another, BinaryInt):
            my_bits: tuple[IntBoolean, ...] = self.as_tuple
            other_bits: tuple[IntBoolean, ...] = another.real(BinaryInt).as_tuple

            new_bits_reverse: list[IntBoolean] = []
            one_to_add = False

            for i in range(max(len(my_bits), len(other_bits))):
                my_idx = len(my_bits) - i - 1
                my_bit = my_bits[my_idx] if my_idx >= 0 else IBoolean.false()
                other_idx = len(other_bits) - i - 1
                other_bit = other_bits[other_idx] if other_idx >= 0 else IBoolean.false()
                prev_one_to_add = one_to_add
                one_to_add = False
                current = IBoolean.false()
                if my_bit.as_bool:
                    current = IBoolean.true()
                if other_bit.as_bool:
                    if current.as_bool:
                        current = IBoolean.false()
                        one_to_add = True
                    else:
                        current = IBoolean.true()
                if prev_one_to_add:
                    if current.as_bool:
                        current = IBoolean.false()
                        one_to_add = True
                    else:
                        current = IBoolean.true()
                new_bits_reverse.append(current)

            if one_to_add:
                new_bits_reverse.append(IBoolean.true())

            new_bits = BinaryInt.from_items(new_bits_reverse[::-1]).normalize()

            return new_bits

        other = another.real(INumber)
        return other.add(self)

    def subtract(self, value: INumber):
        another = value
        if isinstance(another, BinaryInt):
            my_bits: tuple[IntBoolean, ...] = self.as_tuple
            other_bits: tuple[IntBoolean, ...] = another.as_tuple

            new_bits_reverse: list[IntBoolean] = []
            one_to_subtract = False

            for i in range(max(len(my_bits), len(other_bits))):
                my_idx = len(my_bits) - i - 1
                my_bit = my_bits[my_idx] if my_idx >= 0 else IBoolean.false()
                other_idx = len(other_bits) - i - 1
                other_bit = other_bits[other_idx] if other_idx >= 0 else IBoolean.false()

                if one_to_subtract:
                    if my_bit == IBoolean.true():
                        one_to_subtract = False
                        my_bit = IBoolean.false()
                    else:
                        my_bit = IBoolean.true()

                if my_bit == other_bit:
                    new_bits_reverse.append(IBoolean.false())
                elif my_bit == IBoolean.true():
                    new_bits_reverse.append(IBoolean.true())
                else:
                    Not(IBoolean.from_bool(one_to_subtract)).raise_on_false()
                    assert not one_to_subtract
                    Eq(my_bit, IBoolean.false()).raise_on_false()
                    assert my_bit == IBoolean.false()
                    Eq(other_bit, IBoolean.true()).raise_on_false()
                    assert other_bit == IBoolean.true()
                    one_to_subtract = True
                    new_bits_reverse.append(IBoolean.true())

            new_bits: BaseSignedInt = BinaryInt.from_items(
                new_bits_reverse[::-1]
            ).normalize()

            if one_to_subtract:
                new_bits = SignedInt(
                    NegativeSign(IBoolean.true()),
                    BinaryInt
                        .from_items(
                            [IBoolean.true()]
                            + ([IBoolean.false()] * len(new_bits_reverse))
                        )
                        .subtract(new_bits)
                )

            return new_bits

        other = another.real(SignedInt)
        return self.add(other.abs)

    def lt(self, another: INode):
        if not isinstance(another, BinaryInt):
            IsInstance.assert_type(another, IComparable)
            assert isinstance(another, IComparable)
            return another.gt(self)
        my_bits = self.as_tuple
        other_bits = another.as_tuple
        max_len = max(len(my_bits), len(other_bits))
        for i in range(max_len):
            my_idx = len(my_bits) - max_len + i
            my_bit = my_bits[my_idx] if my_idx >= 0 else IBoolean.false()
            other_idx = len(other_bits) - max_len + i
            other_bit = other_bits[other_idx] if other_idx >= 0 else IBoolean.false()
            if my_bit == other_bit:
                continue
            if my_bit == IBoolean.true():
                Eq(other_bit, IBoolean.false()).raise_on_false()
                assert other_bit == IBoolean.false()
                return IntBoolean.false()
            Eq(my_bit, IBoolean.false()).raise_on_false()
            assert my_bit == IBoolean.false()
            Eq(other_bit, IBoolean.true()).raise_on_false()
            assert other_bit == IBoolean.true()
            return IntBoolean.true()
        return IntBoolean.false()

    def gt(self, another: INode):
        if not isinstance(another, BinaryInt):
            IsInstance.assert_type(another, IComparable)
            assert isinstance(another, IComparable)
            return another.lt(self)
        my_bits = self.as_tuple
        other_bits = another.as_tuple
        max_len = max(len(my_bits), len(other_bits))
        for i in range(max_len):
            my_idx = len(my_bits) - max_len + i
            my_bit = my_bits[my_idx] if my_idx >= 0 else IBoolean.false()
            other_idx = len(other_bits) - max_len + i
            other_bit = other_bits[other_idx] if other_idx >= 0 else IBoolean.false()
            if my_bit == other_bit:
                continue
            if my_bit == IBoolean.true():
                Eq(other_bit, IBoolean.false()).raise_on_false()
                assert other_bit == IBoolean.false()
                return IntBoolean.true()
            Eq(my_bit, IBoolean.false()).raise_on_false()
            assert my_bit == IBoolean.false()
            Eq(other_bit, IBoolean.true()).raise_on_false()
            assert other_bit == IBoolean.true()
            return IntBoolean.false()
        return IntBoolean.false()

    def multiply(self, value: INumber) -> BaseSignedInt:
        IsInstance.assert_type(value, BaseSignedInt)
        assert isinstance(value, BaseSignedInt)
        sign = value.sign
        tail: list[IntBoolean] = []
        current = INumber.zero()
        for bit in value.abs.as_tuple[::-1]:
            if bit == IBoolean.true():
                current = current.add(
                    BinaryInt.from_items(list(self.as_tuple) + tail))
            tail.append(IBoolean.false())
        return SignedInt(sign, current).normalize()

    def divide(self, value: IDivisible) -> IDivisible:
        if isinstance(value, BinaryInt):
            return Rational(self, value)
        return value.divided_by(self)

    def divided_by(self, value: IDivisible) -> IDivisible:
        if isinstance(value, BinaryInt):
            return Rational(value, self)
        return value.divide(self)

    def _divide_int(self, value: INumber) -> tuple[BaseSignedInt, BinaryInt]:
        IsInstance.assert_type(value, BaseSignedInt)
        assert isinstance(value, BaseSignedInt)
        sign = value.sign
        my_bits = self.abs.as_tuple
        other_abs = value.abs
        quotient_bits: list[IntBoolean] = []
        remainder = BinaryInt.from_items(my_bits[:1]).normalize()
        remaining_bits = list(my_bits[1:])
        while len(remaining_bits) > 0:
            if remainder.ge(other_abs).as_bool:
                remainder = remainder.subtract(other_abs).real(BinaryInt)
                quotient_bits.append(IBoolean.true())
            else:
                quotient_bits.append(IBoolean.false())
            next_bit = remaining_bits.pop(0)
            remainder = BinaryInt.from_items(list(remainder.abs.as_tuple) + [next_bit])
        if remainder.ge(other_abs).as_bool:
            remainder = remainder.subtract(other_abs).real(BinaryInt)
            quotient_bits.append(IBoolean.true())
        else:
            quotient_bits.append(IBoolean.false())
        remainder.lt(other_abs).raise_on_false()
        quotient = BinaryInt.from_items(quotient_bits).normalize()
        return SignedInt(sign, quotient).normalize(), remainder

    def divide_int(self, value: INumber) -> BaseSignedInt:
        quotient, _ = self._divide_int(value)
        return quotient

    def modulo(self, value: BaseSignedInt) -> BinaryInt:
        _, remainder = self._divide_int(value)
        return remainder

class SignedInt(BaseSignedInt, IInstantiable):

    idx_sign = 1
    idx_abs = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                NegativeSign.as_type(),
                BaseSignedInt.as_type(),
            ),
            UnionType(
                BinaryInt.as_type(),
                CompositeType(
                    SignedInt.as_type(),
                    CountableTypeGroup(
                        InstanceType(NegativeSign(IBoolean.true())),
                        BinaryInt.as_type(),
                    ),
                ),
            ),
        )

    @property
    def raw_sign(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_sign)

    @property
    def raw_abs(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_abs)

    @property
    def sign(self):
        return self.raw_sign.apply().real(NegativeSign)

    @property
    def abs(self):
        return self.raw_abs.apply().real(BinaryInt)

    def with_new_abs(self, abs_value: BinaryInt) -> BaseSignedInt:
        return self.func(self.sign, abs_value).normalize()

    def normalize(self) -> BaseSignedInt:
        sign = self.raw_sign.apply().real(NegativeSign)
        abs_value = self.raw_abs.apply().real(BaseSignedInt).normalize()
        zero = INumber.zero()
        if abs_value == zero:
            return zero
        if isinstance(abs_value, SignedInt):
            inner_sign = abs_value.sign
            Eq(inner_sign, NegativeSign(IBoolean.true())).raise_on_false()
            assert inner_sign == NegativeSign(IBoolean.true())
            inner_abs_value = abs_value.abs
            IsInstance.assert_type(inner_abs_value, BinaryInt)
            assert isinstance(inner_abs_value, BinaryInt)
            if sign == NegativeSign(IBoolean.true()):
                return inner_abs_value
            else:
                return abs_value
        IsInstance.assert_type(abs_value, BinaryInt)
        assert isinstance(abs_value, BinaryInt)
        node = (
            self.func(sign, abs_value)
            if sign == NegativeSign(IBoolean.true())
            else abs_value)
        return node

    def add(self, another: INode):
        my_abs = self.abs

        if isinstance(another, SignedInt):
            other_abs = another.abs
            Eq(self.sign, another.sign).raise_on_false()
            return self.func(
                self.sign,
                my_abs.add(other_abs),
            ).normalize()

        node_2 = another.real(BinaryInt)
        return node_2.subtract(my_abs)

    def lt(self, another: INode):
        if isinstance(another, BinaryInt):
            return IBoolean.true()
        node_2 = another.real(SignedInt)
        my_abs = self.abs
        other_abs = node_2.abs
        return my_abs.gt(other_abs)

    def gt(self, another: INode):
        if isinstance(another, BinaryInt):
            return IBoolean.false()
        node_2 = another.real(SignedInt)
        my_abs = self.abs
        other_abs = node_2.abs
        return my_abs.lt(other_abs)

    def _mul_sign(self, another: BaseSignedInt) -> NegativeSign:
        my_sign = self.sign
        other_sign = another.sign
        sign = (
            NegativeSign.create()
            if my_sign != other_sign
            else NegativeSign(IBoolean.false())
        )
        return sign

    def multiply(self, value: INumber) -> BaseSignedInt:
        IsInstance.assert_type(value, BaseSignedInt)
        assert isinstance(value, BaseSignedInt)
        sign = self._mul_sign(value)
        product = self.abs.multiply(value.abs)
        return SignedInt(sign, product).normalize()

    def divide(self, value: IDivisible) -> IDivisible:
        if isinstance(value, BaseSignedInt):
            same_sign = Eq(self.sign, value.sign).as_bool
            return SignedRational(
                NegativeSign(IBoolean.from_bool(not same_sign)),
                Rational(self.abs, value.abs),
            ).normalize()
        return value.divided_by(self)

    def divided_by(self, value: IDivisible) -> IDivisible:
        if isinstance(value, BaseSignedInt):
            same_sign = Eq(self.sign, value.sign).as_bool
            return SignedRational(
                NegativeSign(IBoolean.from_bool(not same_sign)),
                Rational(value.abs, self.abs),
            ).normalize()
        return value.divide(self)

    def divide_int(self, value: BaseSignedInt) -> BaseSignedInt:
        IsInstance.assert_type(value, BaseSignedInt)
        assert isinstance(value, BaseSignedInt)
        sign = self._mul_sign(value)
        quotient = self.abs.divide_int(value.abs)
        return SignedInt(sign, quotient).normalize()

    def modulo(self, value: BaseSignedInt) -> BinaryInt:
        IsInstance.assert_type(value, BaseSignedInt)
        assert isinstance(value, BaseSignedInt)
        remainder = self.abs.modulo(value.abs)
        return remainder

class BinaryToInt(BaseNormalizer, IInstantiable):

    idx_binary = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(BinaryInt.as_type()),
            Integer.as_type(),
        )

    @property
    def binary(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_binary)

    def normalize(self) -> Integer:
        binary = self.binary.apply().real(BinaryInt)
        exp = 0
        value = 0
        for bit in binary.as_tuple[::-1]:
            if bit == IBoolean.true():
                value += 2 ** exp
            exp += 1
        return Integer(value)

class IntToBinary(BaseNormalizer, IInstantiable):

    idx_integer = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(Integer.as_type()),
            BinaryInt.as_type(),
        )

    @property
    def integer(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_integer)

    def normalize(self) -> BinaryInt:
        integer = self.integer.apply().real(Integer)
        value = integer.as_int
        bits_reverse: list[IntBoolean] = []
        while value > 0:
            bit = IBoolean.from_bool(value % 2 == 1)
            bits_reverse.append(bit)
            value //= 2
        if len(bits_reverse) == 0:
            bits_reverse.append(IBoolean.false())
        return BinaryInt.from_items(bits_reverse[::-1]).normalize()

class BaseSignedRational(BaseNormalizer, IComparableSignedNumber, IDivisible, ABC):

    @property
    def sign(self) -> NegativeSign:
        raise NotImplementedError

    @property
    def value(self) -> Rational:
        raise NotImplementedError

    @property
    def numerator(self) -> BinaryInt:
        raise NotImplementedError

    @property
    def denominator(self) -> BinaryInt:
        raise NotImplementedError

    def precision(self) -> int:
        return self.abs.precision()

    def with_new_value(self, value: Rational) -> BaseSignedRational:
        raise NotImplementedError

    def normalize(self) -> BaseSignedRational:
        raise NotImplementedError

    def validate_result(self, result: INode, args_group: OptionalValueGroup):
        IsInstance.verify(result, BaseSignedRational)

    def add(self, another: INode):
        if isinstance(another, BaseSignedInt):
            another = another.to_rational()

        if isinstance(another, BaseSignedRational):
            my_sign = self.sign
            my_numerator = self.numerator
            my_denominator = self.denominator

            other_sign = another.sign
            other_numerator = another.numerator
            other_denominator = another.denominator

            my_full_numerator = SignedInt(
                my_sign,
                my_numerator.multiply(other_denominator),
            ).normalize()
            IsInstance.assert_type(my_full_numerator, BaseSignedInt)
            assert isinstance(my_full_numerator, BaseSignedInt)
            other_full_numerator = SignedInt(
                other_sign,
                other_numerator.multiply(my_denominator),
            ).normalize()
            IsInstance.assert_type(other_full_numerator, BaseSignedInt)
            assert isinstance(other_full_numerator, BaseSignedInt)
            new_numerator = my_full_numerator.add(other_full_numerator)
            IsInstance.assert_type(new_numerator, BaseSignedInt)
            assert isinstance(new_numerator, BaseSignedInt)
            new_denominator = my_denominator.multiply(other_denominator)
            IsInstance.assert_type(new_denominator, BinaryInt)
            assert isinstance(new_denominator, BinaryInt)
            return SignedRational(
                new_numerator.sign,
                Rational(new_numerator.abs, new_denominator).normalize(),
            ).normalize()

        other = another.real(INumber)
        return other.add(self)

    def subtract(self, value: INumber) -> BaseSignedRational:
        IsInstance.assert_type(value, BaseSignedRational)
        assert isinstance(value, BaseSignedRational)
        return self.add(
            SignedRational(
                NegativeSign.create(),
                value,
            ).normalize()
        )

    def multiply(self, value: INumber) -> BaseSignedRational:
        another = value

        if isinstance(another, BaseSignedInt):
            another = another.to_rational()

        IsInstance.assert_type(another, BaseSignedRational)
        assert isinstance(another, BaseSignedRational)

        my_sign = self.sign
        my_numerator = self.numerator
        my_denominator = self.denominator

        other_sign = another.sign
        other_numerator = another.numerator
        other_denominator = another.denominator

        same_sign = Eq(my_sign, other_sign).as_bool
        new_sign = NegativeSign(IBoolean.from_bool(not same_sign))
        new_numerator = my_numerator.multiply(other_numerator)
        IsInstance.assert_type(new_numerator, BinaryInt)
        assert isinstance(new_numerator, BinaryInt)
        new_denominator = my_denominator.multiply(other_denominator)
        IsInstance.assert_type(new_denominator, BinaryInt)
        assert isinstance(new_denominator, BinaryInt)
        return SignedRational(
            new_sign,
            Rational(new_numerator, new_denominator).normalize(),
        ).normalize()

    def divide(self, value: IDivisible) -> IDivisible:
        another = value

        if isinstance(another, BaseSignedInt):
            another = another.to_rational()

        IsInstance.assert_type(another, BaseSignedRational)
        assert isinstance(another, BaseSignedRational)

        my_sign = self.sign
        my_numerator = self.numerator
        my_denominator = self.denominator

        other_sign = another.sign
        other_numerator = another.numerator
        other_denominator = another.denominator

        same_sign = Eq(my_sign, other_sign).as_bool
        new_sign = NegativeSign(IBoolean.from_bool(not same_sign))
        new_numerator = my_numerator.multiply(other_denominator)
        IsInstance.assert_type(new_numerator, BinaryInt)
        assert isinstance(new_numerator, BinaryInt)
        new_denominator = my_denominator.multiply(other_numerator)
        IsInstance.assert_type(new_denominator, BinaryInt)
        assert isinstance(new_denominator, BinaryInt)
        return SignedRational(
            new_sign,
            Rational(new_numerator, new_denominator).normalize(),
        ).normalize()

    def divided_by(self, value: IDivisible) -> IDivisible:
        another = value

        if isinstance(another, BaseSignedInt):
            another = another.to_rational()

        return another.divide(self)

    def lt(self, another: INode):
        if isinstance(another, BaseSignedInt):
            another = another.to_rational()

        if not isinstance(another, BaseSignedRational):
            IsInstance.assert_type(another, IComparable)
            assert isinstance(another, IComparable)
            return another.gt(self)

        my_sign = self.sign
        my_numerator = self.numerator
        my_denominator = self.denominator
        other_sign = another.sign
        other_numerator = another.numerator
        other_denominator = another.denominator
        same_sign = Eq(my_sign, other_sign).as_bool
        if not same_sign:
            return Eq(my_sign, NegativeSign.create())
        my_full_numerator = my_numerator.multiply(other_denominator)
        other_full_numerator = other_numerator.multiply(my_denominator)
        if my_sign == NegativeSign.create():
            return my_full_numerator.gt(other_full_numerator)
        return my_full_numerator.lt(other_full_numerator)

    def gt(self, another: INode):
        if isinstance(another, BaseSignedInt):
            another = another.to_rational()
        IsInstance.assert_type(another, IComparable)
        assert isinstance(another, IComparable)
        return another.lt(self)

    def eq(self, another: INode) -> IBoolean:
        if isinstance(another, BaseSignedInt):
            another = another.to_rational()
        if isinstance(another, BaseSignedRational):
            new_self = SignedInt(
                self.sign,
                self.numerator.multiply(another.denominator),
            ).normalize()
            new_other = SignedInt(
                another.sign,
                another.numerator.multiply(self.denominator),
            ).normalize()
            return Eq(new_self, new_other)
        return IBoolean.false()

    def to_int(self) -> Optional[BaseSignedInt]:
        sign = self.sign
        numerator = self.numerator
        denominator = self.denominator
        zero = INumber.zero()
        if numerator == zero:
            return Optional(zero)
        one = INumber.one()
        if denominator == one:
            return Optional(SignedInt(sign, numerator).normalize())
        if numerator == denominator:
            return Optional(SignedInt(sign, one).normalize())
        if numerator.modulo(denominator) == zero:
            return Optional(SignedInt(
                sign,
                numerator.divide_int(denominator),
            ).normalize())
        return Optional()

    def irreductible(self) -> BaseSignedRational | BaseSignedInt:
        sign = self.sign
        numerator = self.numerator
        denominator = self.denominator
        zero = INumber.zero()
        if numerator.modulo(denominator) == zero:
            return SignedInt(
                sign,
                numerator.divide_int(denominator),
            ).normalize()
        one = INumber.one()
        new_n: BinaryInt = numerator
        new_d: BinaryInt = denominator
        current_factor = denominator.subtract(one)
        while current_factor.gt(one).as_bool:
            while new_n.modulo(current_factor) == zero and new_d.modulo(current_factor) == zero:
                new_n = new_n.divide_int(current_factor).real(BinaryInt)
                new_d = new_d.divide_int(current_factor).real(BinaryInt)
            current_factor = current_factor.subtract(one)
        return SignedRational(
            sign,
            Rational(new_n, new_d).normalize(),
        ).normalize()

class IntToRational(BaseNormalizer, IInstantiable):

    idx_signed_int = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(BaseSignedInt.as_type()),
            BaseSignedRational.as_type(),
        )

    @property
    def signed_int(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_signed_int)

    def normalize(self) -> BaseSignedRational:
        signed_int = self.signed_int.apply().real(BaseSignedInt)
        return signed_int.to_rational()

class RationalToInt(BaseNormalizer, IInstantiable):

    idx_rational = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(BaseSignedRational.as_type()),
            BaseSignedInt.as_type(),
        )

    @property
    def rational(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_rational)

    def normalize(self) -> BaseSignedInt:
        rational = self.rational.apply().real(BaseSignedRational)
        return rational.to_int().value_or_raise

class ReduceRational(BaseNormalizer, IInstantiable):

    idx_rational = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(UnionType(
                BaseSignedRational.as_type(),
                BaseSignedInt.as_type(),
            )),
            UnionType(
                BaseSignedRational.as_type(),
                BaseSignedInt.as_type(),
            ),
        )

    @property
    def rational(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_rational)

    def normalize(self) -> BaseSignedRational | BaseSignedInt:
        rational = self.rational.apply().real(BaseSignedRational)
        return rational.irreductible()

class Rational(BaseSignedRational, IInstantiable):

    idx_numerator = 1
    idx_denominator = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            BinaryInt.as_type(),
            BinaryInt.as_type(),
        ))

    @property
    def numerator(self) -> BinaryInt:
        return self.inner_arg(self.idx_numerator).apply().real(BinaryInt)

    @property
    def denominator(self) -> BinaryInt:
        return self.inner_arg(self.idx_denominator).apply().real(BinaryInt)

    @property
    def sign(self) -> NegativeSign:
        return NegativeSign(IBoolean.false())

    @property
    def value(self) -> Rational:
        return self

    def with_new_value(self, value: Rational) -> BaseSignedRational:
        return value.normalize()

    def normalize(self) -> BaseSignedRational:
        denominator = self.denominator
        Not(Eq(denominator, INumber.zero())).raise_on_false()
        return self

class SignedRational(BaseSignedRational, IInstantiable):

    idx_sign = 1
    idx_value = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                NegativeSign.as_type(),
                BaseSignedRational.as_type(),
            ),
            UnionType(
                Rational.as_type(),
                CompositeType(
                    SignedRational.as_type(),
                    CountableTypeGroup(
                        InstanceType(NegativeSign(IBoolean.true())),
                        Rational.as_type(),
                    ),
                ),
            ),
        )

    @property
    def raw_sign(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_sign)

    @property
    def raw_value(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_value)

    @property
    def sign(self):
        return self.raw_sign.apply().real(NegativeSign)

    @property
    def value(self) -> Rational:
        return self.raw_value.apply().real(Rational)

    @property
    def numerator(self) -> BinaryInt:
        return self.value.numerator

    @property
    def denominator(self) -> BinaryInt:
        return self.value.denominator

    def with_new_value(self, value: Rational) -> BaseSignedRational:
        return self.func(self.sign, value).normalize()

    def normalize(self) -> BaseSignedRational:
        sign = self.raw_sign.apply().real(NegativeSign)
        value = self.raw_value.apply().real(BaseSignedRational)
        if isinstance(value, SignedRational):
            inner_sign = value.sign
            Eq(inner_sign, NegativeSign(IBoolean.true())).raise_on_false()
            assert inner_sign == NegativeSign(IBoolean.true())
            inner_value = value.value
            IsInstance.assert_type(inner_value, Rational)
            assert isinstance(inner_value, Rational)
            if sign == NegativeSign(IBoolean.true()):
                return inner_value
            else:
                return value
        IsInstance.assert_type(value, Rational)
        assert isinstance(value, Rational)
        node = (
            self.func(sign, value)
            if sign == NegativeSign(IBoolean.true())
            else value)
        return node

class Float(BaseNormalizer, IComparableNumber, IInstantiable):

    idx_base = 1
    idx_exponent = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(
            CountableTypeGroup(
                BaseSignedInt.as_type(),
                BaseSignedInt.as_type(),
            ),
        )

    @property
    def base(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_base)

    @property
    def exponent(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exponent)

    def precision(self) -> int:
        base = self.base.apply().real(BaseSignedInt)
        return len(base.abs.as_tuple)

    def numeric(self) -> float:
        base = self.base.apply().real(BaseSignedInt)
        exponent = self.exponent.apply().real(BaseSignedInt)
        sign = base.sign
        abs_value = base.abs
        precision = len(abs_value.as_tuple)
        abs_int = BinaryToInt(abs_value).normalize()
        exp_int = BinaryToInt(exponent).normalize()
        IsInstance.assert_type(abs_int, Integer)
        assert isinstance(abs_int, Integer)
        IsInstance.assert_type(exp_int, Integer)
        assert isinstance(exp_int, Integer)
        value = abs_int * (2 ** (exp_int.as_int - precision))
        if sign == NegativeSign(IBoolean.true()):
            value = -value
        return float(value)

    def normalize(self) -> Float:
        base = self.base.apply().real(BaseSignedInt)
        zero = INumber.zero()
        if base == zero:
            return self.func(zero, zero)
        return self

    def same_exponents(self, other: Float) -> tuple[
        BaseSignedInt,
        BaseSignedInt,
        BaseSignedInt,
        int,
    ]:
        my_base = self.base.apply().real(BaseSignedInt)
        my_exponent = self.exponent.apply().real(BaseSignedInt)
        other_base = other.base.apply().real(BaseSignedInt)
        other_exponent = other.exponent.apply().real(BaseSignedInt)
        my_precision = self.precision()
        other_precision = other.precision()
        self_higher = False

        if my_exponent.lt(other_exponent).as_bool:
            higher_precision = other_precision
            higher_exp = other_exponent
            higher_base = other_base
            lower_precision = my_precision
            lower_exp = my_exponent
            lower_base = my_base
        else:
            self_higher = True
            higher_precision = my_precision
            higher_exp = my_exponent
            higher_base = my_base
            lower_precision = other_precision
            lower_exp = other_exponent
            lower_base = other_base

        diff_precision = higher_precision - lower_precision

        if diff_precision > 0:
            lower_bits = list(lower_base.abs.as_tuple)
            lower_bits = lower_bits + [IBoolean.false()] * diff_precision
            lower_base = lower_base.with_new_abs(BinaryInt.from_items(lower_bits))
        elif diff_precision < 0:
            higher_bits = list(higher_base.abs.as_tuple)
            higher_bits = higher_bits + [IBoolean.false()] * -diff_precision
            higher_base = higher_base.with_new_abs(BinaryInt.from_items(higher_bits))

        precision = higher_base.precision()
        Eq.from_ints(precision, lower_base.precision()).raise_on_false()
        assert precision == lower_base.precision()

        higher_bits = list(higher_base.abs.as_tuple)
        current_exp = higher_exp
        one = INumber.one()
        while lower_exp.lt(current_exp).as_bool:
            higher_bits.append(IBoolean.false())
            current_exp = current_exp.subtract(one).real(BaseSignedInt)
        higher_base = SignedInt(
            higher_base.sign,
            BinaryInt.from_items(higher_bits)
        ).normalize()
        higher_exp = current_exp
        Eq(higher_exp, lower_exp).raise_on_false()

        my_new_base = higher_base if self_higher else lower_base
        other_new_base = lower_base if self_higher else higher_base

        return my_new_base, other_new_base, current_exp, precision

    def same_precisions(self, other: Float) -> tuple[
        BaseSignedInt,
        BaseSignedInt,
        int,
    ]:
        my_base = self.base.apply().real(BaseSignedInt)
        other_base = other.base.apply().real(BaseSignedInt)
        my_precision = self.precision()
        other_precision = other.precision()
        self_higher = False

        if my_precision < other_precision:
            higher_precision = other_precision
            higher_base = other_base
            lower_precision = my_precision
            lower_base = my_base
        else:
            self_higher = True
            higher_precision = my_precision
            higher_base = my_base
            lower_precision = other_precision
            lower_base = other_base

        diff_precision = higher_precision - lower_precision
        lower_bits = list(lower_base.abs.as_tuple)
        lower_bits = lower_bits + [IBoolean.false()] * diff_precision
        lower_base = lower_base.with_new_abs(BinaryInt.from_items(lower_bits))

        precision = higher_base.precision()
        Eq.from_ints(precision, lower_base.precision()).raise_on_false()
        assert precision == lower_base.precision()

        my_new_base = higher_base if self_higher else lower_base
        other_new_base = lower_base if self_higher else higher_base

        return my_new_base, other_new_base, precision

    def add(self, another: INode):
        (
            my_new_base,
            other_new_base,
            current_exp,
            precision,
        ) = self.same_exponents(another.real(Float))

        new_base = my_new_base.add(other_new_base).normalize()
        IsInstance.assert_type(new_base, BaseSignedInt)
        assert isinstance(new_base, BaseSignedInt)
        diff = new_base.precision() - precision
        one = INumber.one()
        while diff > 0:
            current_exp = current_exp.add(one).real(BaseSignedInt)
            diff -= 1
        while diff < 0:
            current_exp = current_exp.subtract(one).real(BaseSignedInt)
            diff += 1
        return Float(new_base, current_exp)

    def subtract(self, value: INumber) -> Float:
        if isinstance(value, BaseSignedInt):
            value = AsFloat(value).normalize()
        IsInstance.assert_type(value, Float)
        assert isinstance(value, Float)
        return self.add(
            Float(
                SignedInt(
                    NegativeSign.create(),
                    value.base.apply().real(BaseSignedInt),
                ).normalize(),
                value.exponent.apply()
            )
        ).normalize()

    def multiply(self, value: INumber) -> Float:
        other = value.real(Float)
        my_base = self.base.apply().real(BaseSignedInt)
        my_exponent = self.exponent.apply().real(BaseSignedInt)
        other_base = other.base.apply().real(BaseSignedInt)
        other_exponent = other.exponent.apply().real(BaseSignedInt)

        new_base = my_base.multiply(other_base)
        new_exp = my_exponent.add(other_exponent)
        diff = my_base.precision() + other_base.precision() - new_base.precision()
        while diff > 0:
            new_exp = new_exp.subtract(INumber.one())
            diff -= 1
        while diff < 0:
            new_exp = new_exp.add(INumber.one())
            diff += 1
        return Float(new_base, new_exp).normalize()

    def eq(self, another: INode) -> IBoolean:
        other = another.real(Float)
        my_base = self.base.apply().real(BaseSignedInt)
        my_exponent = self.exponent.apply().real(BaseSignedInt)
        other_base = other.base.apply().real(BaseSignedInt)
        other_exponent = other.exponent.apply().real(BaseSignedInt)
        my_sign = my_base.sign.real(NegativeSign)
        other_sign = other_base.sign.real(NegativeSign)
        same_sign = Eq(my_sign, other_sign).as_bool
        same_exp = Eq(my_exponent, other_exponent).as_bool
        if same_sign and same_exp:
            my_new_base, other_new_base, _ = self.same_precisions(other)
            return Eq(my_new_base, other_new_base)
        return IBoolean.false()

    def lt(self, another: INode) -> IBoolean:
        other = another.real(Float)
        my_base = self.base.apply().real(BaseSignedInt)
        my_exponent = self.exponent.apply().real(BaseSignedInt)
        other_base = other.base.apply().real(BaseSignedInt)
        other_exponent = other.exponent.apply().real(BaseSignedInt)
        my_sign = my_base.sign.real(NegativeSign)
        other_sign = other_base.sign.real(NegativeSign)
        if my_sign != other_sign:
            return IBoolean.from_bool(my_sign == NegativeSign(IBoolean.true()))
        if my_exponent != other_exponent:
            if my_sign == NegativeSign(IBoolean.true()):
                return my_exponent.gt(other_exponent)
            return my_exponent.lt(other_exponent)
        my_new_base, other_new_base, _ = self.same_precisions(other)
        return my_new_base.lt(other_new_base)

    def gt(self, another: INode):
        other = another.real(Float)
        return other.lt(self)

class AsFloat(BaseNormalizer, IInstantiable):

    idx_binary = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(BaseSignedInt.as_type()),
            Float.as_type(),
        )

    @property
    def binary(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_binary)

    def normalize(self) -> Float:
        binary = self.binary.apply().real(BaseSignedInt)
        zero = INumber.zero()
        if binary == zero:
            return Float(zero, zero)
        bits_amount = len(binary.abs.as_tuple)
        return Float(
            binary,
            IntToBinary(Integer(bits_amount)).normalize()
        ).normalize()

###########################################################
################## ARITHMETIC OPERATIONS ##################
###########################################################

class Subtract(ControlFlowBaseNode, IInstantiable):

    idx_minuend = 1
    idx_subtrahend = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INumber.as_type()),
            ),
            CountableTypeGroup(
                INumber.as_type(),
                INumber.as_type(),
            ),
            TypeIndex(1),
        )

    @property
    def minuend(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_minuend)

    @property
    def subtrahend(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_subtrahend)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.minuend.apply().run(info_with_stats).as_tuple
        minuend = node_aux.real(INumber)

        info_with_stats, node_aux = self.subtrahend.apply().run(info_with_stats).as_tuple
        subtrahend = node_aux.real(INumber)

        node = minuend.subtract(subtrahend)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [minuend, subtrahend])
        return RunInfoFullResult(info_with_stats.to_result(node), arg_group)

class Multiply(ControlFlowBaseNode, IInstantiable):

    idx_factor_1 = 1
    idx_factor_2 = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(INumber.as_type()),
            ),
            CountableTypeGroup(
                INumber.as_type(),
                INumber.as_type(),
            ),
            TypeIndex(1),
        )

    @property
    def factor_1(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_factor_1)

    @property
    def factor_2(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_factor_2)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.factor_1.apply().run(info_with_stats).as_tuple
        factor_1 = node_aux.real(INumber)

        info_with_stats, node_aux = self.factor_2.apply().run(info_with_stats).as_tuple
        factor_2 = node_aux.real(INumber)

        node = factor_1.multiply(factor_2)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [factor_1, factor_2])
        return RunInfoFullResult(info_with_stats.to_result(node), arg_group)

class Divide(ControlFlowBaseNode, IInstantiable):

    idx_dividend = 1
    idx_divisor = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(IDivisible.as_type()),
            ),
            CountableTypeGroup(
                IDivisible.as_type(),
                IDivisible.as_type(),
            ),
            TypeIndex(1),
        )

    @property
    def dividend(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_dividend)

    @property
    def divisor(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_divisor)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.dividend.apply().run(info_with_stats).as_tuple
        dividend = node_aux.real(IDivisible)

        info_with_stats, node_aux = self.divisor.apply().run(info_with_stats).as_tuple
        divisor = node_aux.real(IDivisible)

        node = dividend.divide(divisor)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [dividend, divisor])
        return RunInfoFullResult(info_with_stats.to_result(node), arg_group)

class DivideInt(ControlFlowBaseNode, IInstantiable):

    idx_dividend = 1
    idx_divisor = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(BaseSignedInt.as_type()),
            ),
            CountableTypeGroup(
                BaseSignedInt.as_type(),
                BaseSignedInt.as_type(),
            ),
            TypeIndex(1),
        )

    @property
    def dividend(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_dividend)

    @property
    def divisor(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_divisor)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.dividend.apply().run(info_with_stats).as_tuple
        dividend = node_aux.real(BaseSignedInt)

        info_with_stats, node_aux = self.divisor.apply().run(info_with_stats).as_tuple
        divisor = node_aux.real(BaseSignedInt)

        node = dividend.divide_int(divisor)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [dividend, divisor])
        return RunInfoFullResult(info_with_stats.to_result(node), arg_group)

class Modulo(ControlFlowBaseNode, IInstantiable):

    idx_dividend = 1
    idx_divisor = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(
                TypeAlias(BaseSignedInt.as_type()),
            ),
            CountableTypeGroup(
                BaseSignedInt.as_type(),
                BaseSignedInt.as_type(),
            ),
            TypeIndex(1),
        )

    @property
    def dividend(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_dividend)

    @property
    def divisor(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_divisor)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        info_with_stats = info.with_stats()

        info_with_stats, node_aux = self.dividend.apply().run(info_with_stats).as_tuple
        dividend = node_aux.real(BaseSignedInt)

        info_with_stats, node_aux = self.divisor.apply().run(info_with_stats).as_tuple
        divisor = node_aux.real(BaseSignedInt)

        node = dividend.modulo(divisor)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [dividend, divisor])
        return RunInfoFullResult(info_with_stats.to_result(node), arg_group)
