from env.core import (
    INode,
    TmpNestedArg,
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
