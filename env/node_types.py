from abc import ABC
from env.core import (
    INode,
    InheritableNode,
    TmpNestedArg,
    IBoolean,
    BaseInt,
    ExtendedTypeGroup,
    CountableTypeGroup,
    IInstantiable)
from env.state import State, StateDefinitionGroup
from env.meta_env import GoalNode

class HaveDefinition(GoalNode, IInstantiable):

    idx_definition_expr = 0

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            INode,
        ]))

    @property
    def definition_expr(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_definition_expr)

    def evaluate(self, state: State) -> bool:
        definition = self.args[0]
        group = state.definition_group.apply().cast(StateDefinitionGroup)
        definitions = [d.definition_expr.apply() for d in group.as_tuple]
        return definition in definitions

class IntBooleanNode(BaseInt, IBoolean, IInstantiable):

    @property
    def as_bool(self) -> bool | None:
        if self.as_int == 0:
            return False
        if self.as_int == 1:
            return True
        return None

class MultiArgBooleanNode(InheritableNode, IBoolean, ABC):

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup.rest(IBoolean.as_type())

    @property
    def as_bool(self) -> bool | None:
        raise NotImplementedError

class AndNode(MultiArgBooleanNode, IInstantiable):

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

class OrNode(MultiArgBooleanNode, IInstantiable):

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
