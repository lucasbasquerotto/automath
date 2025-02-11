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
    def as_type(cls) -> TypeNode[typing.Self]:
        return TypeNode(cls)

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
        origin = typing.get_origin(t)
        t = origin if origin is not None else t
        instance: BaseNode = self.as_node
        if not issubclass(t, TypeEnforcer):
            instance = instance.actual_instance()
        return instance.cast(t)

    def cast(self, t: typing.Type[T]) -> T:
        origin = typing.get_origin(t)
        t = origin if origin is not None else t
        assert isinstance(self, t), f'{type(self)} != {t}'
        return typing.cast(T, self)

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
        node_1 = node_aux.real(IInt)

        info, another_aux = another.as_node.run(info).as_tuple
        node_2 = another_aux.real(IInt)

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
        node_1 = node_aux.real(IGroup)

        info, another_aux = another.as_node.run(info).as_tuple
        node_2 = another_aux.real(IGroup)

        new_args = list(node_1.as_tuple) + list(node_2.as_tuple)
        return info.to_result(self.__class__.from_items(new_args))

class IFunction(INode, ABC):

    def fn_protocol(self) -> Protocol:
        raise NotImplementedError(self.__class__)

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
            protocol = self.protocol()

            try:
                outer_result = self._run(info)
            except NodeReturnException as e:
                outer_result = e.result

            assert isinstance(outer_result, RunInfoFullResult)
            result, args_group = outer_result.as_tuple
            assert isinstance(result, RunInfoResult)
            new_info, new_node = result.as_tuple

            if isinstance(self, IOpaqueScope):
                result = RunInfoResult.with_args(
                    run_info=info,
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

            self._cached_run[(self, info)] = result

            try:
                new_result = new_node.as_node.run(info)
            except NodeReturnException as e:
                new_outer_result = e.result
                assert isinstance(new_outer_result, RunInfoFullResult)
                new_result, _ = new_outer_result.as_tuple

            assert isinstance(new_result, RunInfoResult)
            _, node_aux = new_result.as_tuple
            Eq(new_node, node_aux).raise_on_false()

            if not info.is_future():
                if isinstance(self, IDynamic):
                    self.verify_result(result=new_node, args_group=args_group)
                else:
                    protocol.verify(new_node)

            if new_info.must_return():
                outer_result = RunInfoFullResult(result, args_group)
                raise NodeReturnException(outer_result)

            return result
        except NodeReturnException as e:
            raise e
        except InvalidNodeException as e:
            exc_info = e.info.add_stack(
                node=self,
                run_info=info,
            )
            raise InvalidNodeException(exc_info) from e

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        raise NotImplementedError(self.__class__)

class IDynamic(IRunnable, ABC):

    def verify_result(self, result: INode, args_group: OptionalValueGroup):
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
    def true(cls) -> IBoolean:
        return IntBoolean.create_true()

    @classmethod
    def false(cls) -> IBoolean:
        return IntBoolean.create()

    @classmethod
    def from_bool(cls, value: bool) -> IBoolean:
        if value:
            return cls.true()
        return cls.false()

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

    def has_node(self, node: INode) -> bool:
        return node in self.find(node.__class__)

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

    def inner_arg(self, idx: int) -> TmpInnerArg:
        return TmpInnerArg(self, idx)

    def nested_arg(self, idxs: tuple[int, ...]) -> TmpNestedArg:
        return TmpNestedArg(self, idxs)

    def validate(self):
        for arg in self.args:
            if isinstance(arg, INode):
                arg.as_node.validate()

    def strict_validate(self) -> AliasInfo:
        raise NotImplementedError(self.__class__)

    def result_type(self) -> IType:
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
        return result_type

###########################################################
######################## TYPE NODE ########################
###########################################################

class IType(INode, ABC):

    def valid(
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
        assert isinstance(t, type) and issubclass(t, INode), t
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
        assert issubclass(t, InheritableNode)
        return t.new(*group.as_tuple).as_node.run(info)

    def _valid(
        self,
        instance: INode,
        alias_info: AliasInfo,
    ) -> tuple[bool, AliasInfo]:
        return isinstance(instance, self.type), alias_info

    def strict_validate(self) -> AliasInfo:
        self.validate()
        return AliasInfo.create()

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()
        result = info.to_result(self)
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

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()
        result = info.to_result(self)
        return RunInfoFullResult(result, OptionalValueGroup())

    def strict_validate(self) -> AliasInfo:
        self.validate()
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
        type_group = self.protocol().arg_group.apply()
        if isinstance(type_group, OptionalTypeGroup):
            assert len(args) <= 1, \
                f'{type(self)}: {len(args)} > 1'
        elif isinstance(type_group, CountableTypeGroup):
            assert len(args) == len(type_group.args), \
                f'{type(self)}: {len(args)} != {len(type_group.args)}'

    def strict_validate(self) -> AliasInfo:
        self.validate()
        alias_info = self.protocol().verify_args(DefaultGroup(*self.args))
        return alias_info

    def full_strict_validate(self):
        self.strict_validate()
        for arg in self.args:
            if isinstance(arg, InheritableNode):
                arg.full_strict_validate()
            else:
                arg.as_node.strict_validate()

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()
        args: list[INode] = []
        for arg in self.args:
            info, new_arg = arg.as_node.run(info).as_tuple
            args.append(new_arg)
        new_node = self.func(*args)
        if not info.is_future():
            new_node.strict_validate()
        result = (
            RunInfoResult.with_args(
                run_info=info,
                return_value=new_node,
            )
            if isinstance(self, IDynamic)
            else info.to_result(new_node))
        return RunInfoFullResult(result, OptionalValueGroup.from_optional_items(args))

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

    _empty: Optional[T] | None = None

    @staticmethod
    def __new__(cls: type[Optional[T]], *args: INode):
        if len(args) == 0 and cls._empty is not None:
            opt = cls._empty
            if opt is not None:
                return opt
        instance = super().__new__(cls)
        instance.__class__.__init__(instance, *args)
        if len(args) == 0:
            cls._empty = instance
        return instance

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
        assert isinstance(node, RunInfoScopeDataIndex)
        scope_data_group = info.scope_data_group.apply().real(ScopeDataGroup)
        amount = len(scope_data_group.as_tuple)
        IsInsideRange(node, Integer.zero(), Integer(amount)).raise_on_false()
        result = info.to_result(node)
        return RunInfoFullResult(result, OptionalValueGroup())

    def verify_result(self, result: INode, args_group: OptionalValueGroup):
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

    def verify_result(self, result: INode, args_group: OptionalValueGroup):
        IsInstance.verify(result, INode)

    @property
    def parent_scope(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_parent_scope)

    @property
    def index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_index)

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info, node_aux = base_result.as_tuple
        node = node_aux.real(self.__class__)

        scope_index = node.parent_scope.apply().real(RunInfoScopeDataIndex)
        index = node.index.apply().real(BaseInt)

        scope = scope_index.find_in_outer_node(info).value_or_raise
        assert isinstance(scope, ScopeDataPlaceholderItemGroup), type(scope)

        assert isinstance(node, scope.item_inner_type()), \
            f'{type(node)} != {scope.item_inner_type()} ({scope_index.as_int} - {index.as_int})'

        if isinstance(scope, IScopeDataFutureItemGroup):
            groups = info.scope_data_group.apply().real(ScopeDataGroup).as_tuple
            group_amount = len(groups)
            near_scope = NearParentScope.from_int(group_amount - scope_index.as_int + 1)
            result = info.to_result(node.func(near_scope, index))
            return RunInfoFullResult(result, arg_group)

        item = NodeArgIndex(index.as_int).find_in_node(scope).value_or_raise
        info, node_aux = item.as_node.run(info).as_tuple
        node_aux = node_aux.real(IOptional)

        new_node = node_aux.value_or_raise

        result = info.to_result(new_node)
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
            return Optional(new_target).real(IOptional[T])
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
            return Optional(new_target).real(Optional[T])
        return Optional()

class NodeArgIndex(NodeArgBaseIndex, IInstantiable):
    pass

###########################################################
####################### ITEMS GROUP #######################
###########################################################

class BaseGroup(InheritableNode, IGroup[T], IDefault, typing.Generic[T], ABC):

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            RestTypeGroup(TypeNode(cls.item_type())),
            CompositeType(
                cls.as_type(),
                RestTypeGroup(TypeNode(cls.item_type())),
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

    def strict_validate(self):
        super().strict_validate()
        t = self.item_type()
        for arg in self.args:
            origin = typing.get_origin(t)
            t = origin if origin is not None else t
            assert isinstance(arg, t), f'{type(arg)} != {t}'

    def to_optional_group(self) -> OptionalValueGroup[T]:
        return OptionalValueGroup.from_optional_items(self.args)

class DefaultGroup(BaseGroup[INode], IInstantiable):

    @classmethod
    def item_type(cls):
        return INode

class BaseIntGroup(BaseGroup[IInt], ABC):

    @classmethod
    def item_type(cls):
        return IInt

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
    def item_type(cls):
        return NodeArgIndex

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
    def item_type(cls) -> type[IType]:
        return IType

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
    def item_type(cls) -> type[TypeAlias]:
        return TypeAlias

    def strict_validate(self):
        super().strict_validate()
        for i, arg_aux in enumerate(self.args):
            index = i + 1
            arg = arg_aux.real(TypeAlias)
            inner_idxs = sorted(arg.as_node.find(BaseTypeIndex), key=lambda t: t.as_int)
            invalid_idxs = [idx for idx in inner_idxs if idx.as_int >= index]
            if len(invalid_idxs) > 0:
                invalid_group = DefaultGroup(*invalid_idxs)
                raise InvalidNodeException(
                    TypeAliasIndexExceptionInfo(TypeIndex(index), arg, invalid_group))

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

    def validate(self):
        super().validate()
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
            new_group.strict_validate()
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

    def valid(
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

    def strict_validate(self):
        super().strict_validate()
        t = self.type.apply().real(IType)
        node = self.node.apply()
        t.verify(node, alias_info=AliasInfo.create())

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
        items: list[bool] = []
        for t in args:
            valid, alias_info = t.valid(instance, alias_info=alias_info)
            items.append(valid)
        return self.multi_bool(
            items,
            any_case=True,
            all_case=False,
        ), alias_info

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
        items: list[bool] = []
        for t in args:
            valid, alias_info = t.valid(instance, alias_info=alias_info)
            items.append(valid)
        return self.multi_bool(
            items,
            any_case=False,
            all_case=True,
        ), alias_info

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

    def fn_protocol(self) -> Protocol:
        return self.protocol_arg.apply().real(Protocol)

    def with_arg_group(self, group: BaseGroup, info: RunInfo):
        alias_info: AliasInfo | None = None
        protocol_arg: Protocol = self.protocol_arg.apply().real(Protocol)

        if info.is_future():
            scope_data: ScopeDataParamBaseItemGroup = ScopeDataFutureParamItemGroup()
        else:
            info, node_aux = group.run(info).as_tuple
            new_group = node_aux.real(BaseGroup)
            alias_info = protocol_arg.verify_args(new_group)
            scope_data = ScopeDataParamItemGroup.from_optional_items(new_group.as_tuple)
            scope_data.strict_validate()

        new_info = (
            info.create()
            if isinstance(self, IOpaqueScope)
            else info
        ).add_scope(scope_data)

        try:
            result = self.expr.apply().run(new_info)
        except NodeReturnException as e:
            result, _ = e.result.as_tuple

        if isinstance(self, IOpaqueScope):
            _, node = result.as_tuple
            new_info = info
        else:
            new_info, node = result.as_tuple
            new_info = new_info.with_scopes(info)

        if not info.is_future():
            assert alias_info is not None
            protocol_arg.verify_result(node, alias_info=alias_info)

        return RunInfoResult.with_args(
            run_info=new_info,
            return_value=node,
        )

    def prepare_expr(self, info: RunInfo) -> RunInfoResult:
        raise NotImplementedError

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()

        info, node_aux = self.protocol_arg.apply().run(info).as_tuple
        type_group = node_aux.real(Protocol)

        info, expr = self.prepare_expr(info).as_tuple

        result = info.to_result(self.func(type_group, expr))
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

    def prepare_expr(self, info: RunInfo):
        expr = self.expr.apply()
        return RunInfoResult.with_args(
            run_info=info,
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
    def as_bool(self) -> bool:
        if self.as_int == 0:
            return False
        if self.as_int == 1:
            return True
        raise InvalidNodeException(BooleanExceptionInfo(self))

class IntBoolean(BaseIntBoolean, IInstantiable):

    _true: IntBoolean | None = None
    _false: IntBoolean | None = None

    @staticmethod
    def __new__(cls: type[IntBoolean], value: int) -> IntBoolean:
        if value == 0:
            if cls._false is None:
                instance = super().__new__(cls)
                instance.__class__.__init__(instance, value)
                cls._false = instance
            return cls._false
        if value == 1:
            if cls._true is None:
                instance = super().__new__(cls)
                instance.__class__.__init__(instance, value)
                cls._true = instance
            return cls._true
        instance = super().__new__(cls)
        instance.__class__.__init__(instance, value)
        raise InvalidNodeException(BooleanExceptionInfo(instance))

class RunnableBoolean(InheritableNode, IDynamic, IBoolean, ABC):

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol.with_args(
            cls.args_type_group(),
            IntBoolean.as_type(),
        )

    def verify_result(self, result: INode, args_group: OptionalValueGroup):
        IsInstance.verify(result, IntBoolean)

    @classmethod
    def args_type_group(cls) -> IBaseTypeGroup:
        raise NotImplementedError

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info, node_aux = base_result.as_tuple
        if info.is_future():
            return base_result

        node = node_aux.real(self.__class__)
        value = node.func(*node.args).strict_bool

        result = info.to_result(IntBoolean.from_bool(value))
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
        assert isinstance(result, RunInfoFullResult)
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

class StackNodeArg(
    InheritableNode,
    IInstantiable,
    IFromSingleNode[INode],
):

    idx_full_arg = 1
    idx_arg_type = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Optional.as_type(),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IType.as_type()),
            ),
        ))

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
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            INode.as_type(),
            BaseGroup.as_type(),
        ))

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
    def item_type(cls) -> IType:
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
        assert isinstance(instance, INode)
        t = self.type.apply()
        assert isinstance(t, TypeNode)
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
    def assert_type(cls, instance: INode, t: typing.Type[T]) -> T:
        return cls(instance, TypeNode(t)).as_type_or_raise

    @classmethod
    def verify(cls, instance: INode, t: typing.Type[INode]):
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
            return super()._run(info)

        run_args: list[INode] = []

        def fn_return(result: RunInfoResult) -> RunInfoFullResult:
            all_run_args = run_args + [None] * (len(self.args) - len(run_args))
            arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
                all_run_args)
            return RunInfoFullResult(result, arg_group)

        for arg in self.args:
            info, run_arg = arg.as_node.run(info).as_tuple
            if isinstance(run_arg, IBoolean):
                if not run_arg.as_bool:
                    result = info.to_result(IBoolean.false())
                    return fn_return(result)
            run_args.append(run_arg)

        for run_arg in run_args:
            assert isinstance(run_arg, IBoolean)

        result = info.to_result(IBoolean.true())

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
            return super()._run(info)

        run_args: list[INode] = []

        def fn_return(result: RunInfoResult) -> RunInfoFullResult:
            all_run_args = run_args + [None] * (len(self.args) - len(run_args))
            arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
                all_run_args)
            return RunInfoFullResult(result, arg_group)

        for arg in self.args:
            info, run_arg = arg.as_node.run(info).as_tuple
            if isinstance(run_arg, IBoolean):
                val_1 = run_arg.as_bool
                if val_1 is True:
                    result = info.to_result(IBoolean.true())
                    return fn_return(result)
            run_args.append(run_arg)

        for run_arg in run_args:
            assert isinstance(run_arg, IBoolean)

        result = info.to_result(IBoolean.false())

        return fn_return(result)

class GreaterThan(DoubleIntBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool:
        args = self.args
        assert len(args) == 2
        a, b = args
        assert isinstance(a, IInt)
        assert isinstance(b, IInt)
        return a.as_int > b.as_int

class LessThan(DoubleIntBooleanNode, IInstantiable):

    @property
    def as_bool(self) -> bool:
        args = self.args
        assert len(args) == 2
        a, b = args
        assert isinstance(a, IInt)
        assert isinstance(b, IInt)
        return a.as_int < b.as_int

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
        assert isinstance(value, IInt)
        min_value = self.min_value.apply()
        assert isinstance(min_value, IInt)
        max_value = self.max_value.apply()
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
    def item_type(cls):
        return Placeholder

    def find_in_outer_node(self, node: ScopeDataPlaceholderItemGroup):
        assert isinstance(node, ScopeDataPlaceholderItemGroup)
        return self.find_in_node(node)

    def replace_in_outer_target(self, target: ScopeDataPlaceholderItemGroup, new_node: Placeholder):
        assert isinstance(target, ScopeDataPlaceholderItemGroup)
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
        assert isinstance(scope_index, RunInfoScopeDataIndex)
        assert isinstance(item_index, PlaceholderIndex)

        item = scope_index.find_in_outer_node(self).value_or_raise
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
        base_amount = base_group.amount()
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
                Not(LessThan(
                    Integer(len(last_after.as_tuple)),
                    Integer(len(last_base.as_tuple)),
                )).raise_on_false()
                last_inner_group = last_base.func(
                    *last_base.as_tuple,
                    *last_after.as_tuple[last_base.amount():],
                )
                new_group = base_group.func(
                    *list(base_group.as_tuple[:-1]) + [last_inner_group]
                )
        return_after_scope = self.return_after_scope.apply().real(Optional[RunInfoScopeDataIndex])
        return_after_val = return_after_scope.value
        if return_after_val is not None:
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

    def to_result(self, result: INode) -> RunInfoResult:
        return RunInfoResult.with_args(
            run_info=self,
            return_value=result)

class RunInfoScopeDataIndex(
    BaseInt,
    ITypedIntIndex[RunInfo, ScopeDataPlaceholderItemGroup],
    IInstantiable,
):

    @classmethod
    def outer_type(cls) -> type[RunInfo]:
        return RunInfo

    @classmethod
    def item_type(cls):
        return ScopeDataPlaceholderItemGroup

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
            RunInfo.as_type(),
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
        new_info = self.run_info.apply().real(RunInfo)
        return_value = self.return_value.apply().cast(INode)
        return new_info, return_value

    @classmethod
    def with_args(
        cls,
        run_info: RunInfo,
        return_value: INode,
    ) -> RunInfoResult:
        return cls(run_info, return_value)

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

class ControlFlowBaseNode(InheritableNode, IDynamic, ABC):

    def _run(self, info: RunInfo) -> RunInfoFullResult:
        self.validate()
        if info.is_future():
            args: list[INode] = []
            for arg in self.args:
                info, node_aux = arg.as_node.run(info).as_tuple
                args.append(node_aux)
            result = info.to_result(self.func(*args))
            arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(args)
            return RunInfoFullResult(result, arg_group)
        return self._run_control(info)

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        raise NotImplementedError(self.__class__)

    def verify_result(self, result: INode, args_group: OptionalValueGroup):
        protocol = self.protocol()
        alias_info = protocol.verify_optional_args(args_group)
        alias_group_actual = alias_info.alias_group_actual.apply().real(TypeAliasOptionalGroup)
        p_arg_group = protocol.arg_group.apply().real(IBaseTypeGroup)
        p_result = protocol.result.apply().real(IType)
        for alias in p_arg_group.as_node.find(BaseTypeIndex):
            value = alias_group_actual.as_tuple[alias.as_int-1]
            if value.is_empty().as_bool:
                p_result = p_result.as_node.replace(alias, InvalidType()).real(IType)
        protocol = protocol.with_new_args(result_type=p_result)
        protocol.verify_result(result, alias_info=alias_info)

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
        info, node_aux = self.function.apply().run(info).as_tuple
        fn = node_aux.real(IFunction)

        info, node_aux = self.arg_group.apply().run(info).as_tuple
        fn_arg_group = node_aux.real(BaseGroup)

        result = fn.with_arg_group(group=fn_arg_group, info=info)
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
        info, node_aux = self.condition.apply().run(info).as_tuple
        condition = node_aux.real(IBoolean)

        flag = condition.strict_bool

        if flag:
            true_result = self.true_expr.apply().run(info)
            _, true_node = true_result.as_tuple
            arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
                [condition, true_node, None])
            return RunInfoFullResult(true_result, arg_group)

        false_result = self.false_expr.apply().run(info)
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
        info, node_aux = self.callback.apply().run(info).as_tuple
        callback = node_aux.real(IFunction)

        info, node_aux = self.initial_data.apply().run(info).as_tuple
        initial_data = node_aux.real(Optional[INode])

        condition = True
        data = initial_data
        idx = 0
        while condition:
            idx += 1
            info, node_aux = FunctionCall(callback, DefaultGroup(data)).run(info).as_tuple
            result = node_aux.real(LoopGuard)

            info, node_aux = result.condition.apply().run(info).as_tuple
            cond_node = node_aux.real(IBoolean)

            info, new_data = result.result.apply().run(info).as_tuple

            data = Optional(new_data)
            condition = cond_node.strict_bool

        result = info.to_result(result=data)
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

        run_args: list[INode] = []
        for arg in self.args:
            info, node = arg.as_node.run(info).as_tuple
            run_args.append(node)

        result = info.to_result(Void())
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
        info, node_aux = self.var_index.apply().run(info).as_tuple
        var_index = node_aux.real(PlaceholderIndex)

        info, node_aux = self.value.apply().run(info).as_tuple
        value = node_aux

        info, node_aux = NearParentScope.create().run(info).as_tuple
        scope_index = node_aux.real(RunInfoScopeDataIndex)

        info = info.add_scope_var(
            scope_index=scope_index,
            item_index=var_index,
            value=value)

        result = info.to_result(value)
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
        info, node_aux = self.parent_scope.apply().run(info).as_tuple
        scope_index = node_aux.real(RunInfoScopeDataIndex)

        info, value = self.value.apply().run(info).as_tuple

        if not info.is_future():
            info = info.with_new_args(
                return_after_scope=Optional(scope_index)
            )

        result = info.to_result(value)
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
        info, node_aux = self.node.apply().run(info).as_tuple
        node = node_aux.cast(INode)

        info, node_aux = self.arg_index.apply().run(info).as_tuple
        arg_index = node_aux.real(NodeArgIndex)

        result = arg_index.find_in_node(node).value_or_raise

        result = RunInfoResult.with_args(
            run_info=info,
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
        info, root_node = self.node.apply().run(info).as_tuple
        node = root_node

        info, node_aux = self.arg_indices.apply().run(info).as_tuple
        arg_indices = node_aux.real(NestedArgIndexGroup)

        for arg_index in arg_indices.as_tuple:
            info, node = InnerArg(node, arg_index).run(info).as_tuple

        result = RunInfoResult.with_args(
            run_info=info,
            return_value=node)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            [node, arg_indices]
        )
        return RunInfoFullResult(result, arg_group)

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
        info = run_info

        info, node_aux = self.group.apply().run(info).as_tuple
        group = node_aux.real(BaseGroup)

        info, node_aux = self.index.apply().run(info).as_tuple
        index = node_aux.real(NodeArgIndex)

        if index.as_int > group.amount():
            result: Optional[INode] = Optional()
            return info.to_result(result)

        value = index.find_in_node(group).value_or_raise
        new_iter = self.with_new_args(
            group=group,
            index=NodeArgIndex(index.as_int + 1),
        )
        result = Optional(DefaultGroup(new_iter, value))
        return info.to_result(result)

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
        info, node_aux = self.iter.apply().run(info).as_tuple
        iterator = node_aux.real(IIterator)
        result = iterator.next(info)
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
            RestTypeGroup(TypeIndex(1)),
            TypeIndex(1),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        node: IAdditive | None = None
        run_args: list[INode] = []
        for arg in self.args:
            info, node_aux = arg.as_node.run(info).as_tuple
            run_args.append(node_aux)
            if node is None:
                arg_node = node_aux.real(IAdditive)
                node = arg_node
            else:
                info, node = node.add(node_aux, run_info=info).as_tuple
        assert node is not None
        result = info.to_result(node)
        arg_group: OptionalValueGroup[INode] = OptionalValueGroup.from_optional_items(
            run_args)
        return RunInfoFullResult(result, arg_group)
