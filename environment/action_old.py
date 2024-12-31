# pylint: disable=C0302
from environment.core import (
    BaseNodeMainIndex,
    Param,
    Function,
    FunctionParams,
    ParamsGroup,
    ArgsGroup,
    InheritableNode,
    Integer,
    IntValueGroup,
    IntTypeGroup,
    BaseNodeIndex,
    BaseNode)
from environment.state import (
    State,
    FunctionDefinitionGroup,
    FunctionDefinitionNode,
    PartialDefinitionGroup,
    ParamsArgsGroup,
    PartialArgsOuterGroup)

###########################################################
######################## CONSTANTS ########################
###########################################################

ACTION_ARG_TYPE_PARTIAL_DEFINITION = 1
ACTION_ARG_TYPE_ARG_GROUP = 2
ACTION_ARG_TYPE_ARG_IDX = 3
ACTION_ARG_TYPE_DEFINITION = 4
ACTION_ARG_TYPE_EXPRESSION_ORIGIN = 5
ACTION_ARG_TYPE_EXPRESSION_TARGET = 6
ACTION_ARG_TYPE_PARTIAL_NODE = 7
ACTION_ARG_TYPE_INT = 8
ACTION_ARG_TYPE_VALUE_TYPE = 9

ARG_TYPES = [
    ACTION_ARG_TYPE_PARTIAL_DEFINITION,
    ACTION_ARG_TYPE_ARG_GROUP,
    ACTION_ARG_TYPE_ARG_IDX,
    ACTION_ARG_TYPE_DEFINITION,
    ACTION_ARG_TYPE_EXPRESSION_ORIGIN,
    ACTION_ARG_TYPE_EXPRESSION_TARGET,
    ACTION_ARG_TYPE_PARTIAL_NODE,
    ACTION_ARG_TYPE_INT,
    ACTION_ARG_TYPE_VALUE_TYPE,
]

VALUE_TYPE_INT = 1
VALUE_TYPE_PARAM = 2
VALUE_TYPE_META_DEFINITION = 3
VALUE_TYPE_META_PROPOSITION = 4
VALUE_TYPE_DEFINITION_KEY = 5
VALUE_TYPE_DEFINITION_EXPR = 6
VALUE_TYPE_PARTIAL_DEFINITION = 7
VALUE_TYPE_PROPOSITION = 8

VALUE_TYPES = [
    VALUE_TYPE_INT,
    VALUE_TYPE_PARAM,
    VALUE_TYPE_META_DEFINITION,
    VALUE_TYPE_META_PROPOSITION,
    VALUE_TYPE_DEFINITION_KEY,
    VALUE_TYPE_DEFINITION_EXPR,
    VALUE_TYPE_PARTIAL_DEFINITION,
    VALUE_TYPE_PROPOSITION,
]


def _validate_indexes(idx_list: list[int]):
    for i, arg in enumerate(idx_list):
        assert arg == i + 1
_validate_indexes(ARG_TYPES)
_validate_indexes(VALUE_TYPES)

class NewActionArg(Integer):
    pass

class ActionArgPartialDefinition(NewActionArg):
    pass

class ActionArgArgGroup(NewActionArg):
    pass

class ActionArgArgIdx(NewActionArg):
    pass

class ActionArgDefinition(NewActionArg):
    pass

class ActionArgExpressionOrigin(NewActionArg):
    pass

class ActionArgExpressionTarget(NewActionArg):
    pass

class ActionArgPartialNode(NewActionArg):
    pass

class ActionArgInt(NewActionArg):
    pass

class ActionArgValueType(NewActionArg):
    pass

class NewActionInput(InheritableNode):
    def __init__(self, *args: NewActionArg):
        assert all(isinstance(arg, NewActionArg) for arg in args)
        super().__init__(*args)

NEW_ACTION_ARG_TYPES = [
    ActionArgPartialDefinition,
    ActionArgArgGroup,
    ActionArgArgIdx,
    ActionArgDefinition,
    ActionArgExpressionOrigin,
    ActionArgExpressionTarget,
    ActionArgPartialNode,
    ActionArgInt,
    ActionArgValueType,
]

class ActionInfo(InheritableNode):
    pass

###########################################################
###################### ACTION INPUT #######################
###########################################################

class ActionInput(IntValueGroup):
    pass

###########################################################
###################### ACTION OUTPUT ######################
###########################################################

class NewPartialDefinitionActionOutput:
    def __init__(self, partial_definition_idx: int):
        self._partial_definition_idx = partial_definition_idx

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

class PartialActionOutput:
    def __init__(
        self,
        partial_definition_idx: int,
        node_idx: int,
        new_function_info: Function,
        new_expr_arg_group: ParamsArgsGroup | None,
    ):
        self._partial_definition_idx = partial_definition_idx
        self._node_idx = node_idx
        self._new_function_info = new_function_info
        self._new_expr_arg_group = new_expr_arg_group

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

    @property
    def node_idx(self) -> int:
        return self._node_idx

    @property
    def new_function_info(self) -> Function:
        return self._new_function_info

    @property
    def new_expr_arg_group(self) -> ParamsArgsGroup | None:
        return self._new_expr_arg_group

class RemovePartialDefinitionActionOutput:
    def __init__(self, partial_definition_idx: int):
        self._partial_definition_idx = partial_definition_idx

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

class NewArgGroupActionOutput:
    def __init__(self, arg_group_idx: int, params_amount: int, args_amount: int):
        self._arg_group_idx = arg_group_idx
        self._params_amount = params_amount
        self._args_amount = args_amount

    @property
    def arg_group_idx(self) -> int:
        return self._arg_group_idx

    @property
    def params_amount(self) -> int:
        return self._params_amount

    @property
    def args_amount(self) -> int:
        return self._args_amount

class ArgFromExprActionOutput:
    def __init__(
        self,
        arg_group_idx: int,
        arg_idx: int,
        new_function_info: Function,
    ):
        self._arg_group_idx = arg_group_idx
        self._arg_idx = arg_idx
        self._new_function_info = new_function_info

    @property
    def arg_group_idx(self) -> int:
        return self._arg_group_idx

    @property
    def arg_idx(self) -> int:
        return self._arg_idx

    @property
    def new_function_info(self) -> Function:
        return self._new_function_info

class RemoveArgGroupActionOutput:
    def __init__(self, arg_group_idx: int):
        self._arg_group_idx = arg_group_idx

    @property
    def arg_group_idx(self) -> int:
        return self._arg_group_idx

class NewDefinitionFromPartialActionOutput:
    def __init__(self, definition_idx: int, partial_definition_idx: int):
        self._definition_idx = definition_idx
        self._partial_definition_idx = partial_definition_idx

    @property
    def definition_idx(self) -> int:
        return self._definition_idx

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

class ReplaceByDefinitionActionOutput:
    def __init__(self, definition_idx: int, expr_id: int):
        self._definition_idx = definition_idx
        self._expr_id = expr_id

    @property
    def definition_idx(self) -> int:
        return self._definition_idx

    @property
    def expr_id(self) -> int:
        return self._expr_id

class ExpandDefinitionActionOutput:
    def __init__(self, definition_idx: int, expr_id: int):
        self._definition_idx = definition_idx
        self._expr_id = expr_id

    @property
    def definition_idx(self) -> int:
        return self._definition_idx

    @property
    def expr_id(self) -> int:
        return self._expr_id

class ReformulationActionOutput:
    def __init__(self, expr_id: int, new_function_info: Function):
        self._expr_id = expr_id
        self._new_function_info = new_function_info

    @property
    def expr_id(self) -> int:
        return self._expr_id

    @property
    def new_function_info(self) -> Function:
        return self._new_function_info

ActionOutput = (
    NewPartialDefinitionActionOutput |
    PartialActionOutput |
    RemovePartialDefinitionActionOutput |
    NewArgGroupActionOutput |
    ArgFromExprActionOutput |
    RemoveArgGroupActionOutput |
    NewDefinitionFromPartialActionOutput |
    ReplaceByDefinitionActionOutput |
    ExpandDefinitionActionOutput |
    ReformulationActionOutput)

###########################################################
####################### EXCEPTIONS ########################
###########################################################

class InvalidActionException(Exception):
    pass

class InvalidActionArgException(InvalidActionException):
    pass

class InvalidActionArgsException(InvalidActionException):
    pass


###########################################################
######################### ACTION ##########################
###########################################################

class Action:

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        raise NotImplementedError

    @classmethod
    def validate_args(cls, input: ActionInput, state: State | None) -> None:
        arg_types = cls.metadata().as_tuple

        if len(input.args) != len(arg_types):
            raise InvalidActionArgsException(f"Invalid action length: {len(input.args)}")

        for i, arg_type in enumerate(arg_types):
            arg_info = input.args[i]

            if not isinstance(arg_info, arg_type.type):
                raise InvalidActionArgException(
                    f"Invalid action arg type: {arg_type.type} != {type(arg_info)}")

            if state is not None:
                if isinstance(arg_info, ActionArgPartialDefinition):
                    partial_definition_idx = arg_info.value
                    if partial_definition_idx <= 0:
                        raise InvalidActionArgException(
                            f"Invalid partial definition: {partial_definition_idx}")
                    if partial_definition_idx > len(state.partial_definitions or []):
                        raise InvalidActionArgException(
                            f"Invalid partial definition: {partial_definition_idx}")
                elif isinstance(arg_info, ActionArgArgGroup):
                    arg_group_idx = arg_info.value
                    if arg_group_idx <= 0:
                        raise InvalidActionArgException(f"Invalid arg group: {arg_group_idx}")
                    if arg_group_idx > len(state.arg_groups or []):
                        raise InvalidActionArgException(f"Invalid arg group: {arg_group_idx}")
                elif isinstance(arg_info, ActionArgArgIdx):
                    arg_idx = arg_info.value
                    if arg_idx <= 0:
                        raise InvalidActionArgException(f"Invalid arg index: {arg_idx}")

                    if i == 0:
                        raise InvalidActionArgException(
                            "The argument index must be right after the group " + \
                            "(expected another argument before)")

                    expected_group = arg_types[i - 1]
                    if expected_group != ACTION_ARG_TYPE_ARG_GROUP:
                        raise InvalidActionArgException(
                            f"Invalid argument order: {expected_group} != " + \
                            f"{ACTION_ARG_TYPE_ARG_GROUP}")

                    last_arg = input.args[i - 1]
                    assert isinstance(last_arg, ActionArgArgGroup)
                    arg_group_idx = last_arg.value
                    arg_groups = state.arg_groups.as_tuple
                    assert arg_group_idx > 0
                    assert arg_group_idx <= len(arg_groups)

                    arg_group = arg_groups[arg_group_idx - 1]
                    if arg_idx > len(arg_group.inner_args):
                        raise InvalidActionArgException(
                            f"Invalid arg index: {arg_idx} (group={arg_group_idx}, " + \
                            f"max={len(arg_group.inner_args)})")
                elif isinstance(arg_info, ActionArgDefinition):
                    definition_idx = arg_info.value
                    if definition_idx <= 0:
                        raise InvalidActionArgException(f"Invalid definition: {definition_idx}")
                    if definition_idx > len(state.definitions or []):
                        raise InvalidActionArgException(f"Invalid definition: {definition_idx}")
                elif isinstance(arg_info, ActionArgExpressionOrigin):
                    expr_id = arg_info.value
                    if expr_id <= 0:
                        raise InvalidActionArgException(f"Invalid origin expr: {expr_id}")
                    function_info = state.get_expr(expr_id)
                    if not function_info:
                        raise InvalidActionArgException(f"Invalid origin expr: {expr_id}")
                elif isinstance(arg_info, ActionArgExpressionTarget):
                    expr_id = arg_info.value
                    if expr_id <= 0:
                        raise InvalidActionArgException(f"Invalid target expr: {expr_id}")
                    function_info = state.get_expr(expr_id)
                    if function_info and function_info.readonly:
                        raise InvalidActionArgException(f"Target expr is readonly: {expr_id}")
                elif isinstance(arg_info, ActionArgPartialNode):
                    node_idx = arg_info.value
                    if node_idx <= 0:
                        raise InvalidActionArgException(f"Invalid node index: {expr_id}")

                    if i == 0:
                        raise InvalidActionArgException(
                            "The argument index must be right after a partial definition " + \
                            "(expected another argument before)")

                    expected_partial_definition = arg_types[i - 1]
                    if expected_partial_definition != ACTION_ARG_TYPE_PARTIAL_DEFINITION:
                        raise InvalidActionArgException(
                            f"Invalid argument order: {expected_partial_definition} != " + \
                            f"{ACTION_ARG_TYPE_PARTIAL_DEFINITION}")

                    last_arg = input.args[i - 1]
                    assert isinstance(last_arg, ActionArgPartialDefinition)
                    partial_definition_idx = last_arg.value
                    partial_definitions = state.partial_definitions.as_tuple
                    if partial_definition_idx <= 0:
                        raise InvalidActionArgException(
                            f"Invalid partial definition: {partial_definition_idx}")
                    if partial_definition_idx > len(partial_definitions or []):
                        raise InvalidActionArgException(
                            f"Invalid partial definition: {partial_definition_idx}")

                    if node_idx > 1:
                        partial_definition = partial_definitions[partial_definition_idx - 1]
                        if partial_definition is None:
                            raise InvalidActionArgException(
                                f"Invalid node index: {node_idx} " + \
                                f"(partial={partial_definition_idx})")
                        node = partial_definition.expr[BaseNodeMainIndex.from_int(node_idx)]
                        if node is None:
                            raise InvalidActionArgException(
                                f"Invalid node index: {node_idx} " + \
                                f"(partial={partial_definition_idx})")
                elif isinstance(arg_info, ActionArgInt):
                    pass
                else:
                    pass


    @classmethod
    def input_from_raw(cls, action: tuple[int, ...]) -> ActionInput:
        arg_types = cls.metadata().as_tuple

        if len(action) != len(arg_types):
            raise InvalidActionArgsException(f"Invalid action length: {len(action)}")

        args = [
            arg_type.type(value)
            for arg_type, value in zip(arg_types, action)
        ]
        return ActionInput(*args)

    @classmethod
    def create(cls, input: ActionInput) -> 'Action':
        input.validate(cls.metadata())
        cls.validate_args(input=input, state=None)
        return cls._create(input)

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        raise NotImplementedError

    def from_index(self, index: BaseNodeIndex, node: BaseNode) -> BaseNode:
        result = index.from_node(node)
        if result is None:
            raise InvalidActionArgException(f"Empty result from index: {index}")
        return result

    def replace_target(
        self,
        index: BaseNodeIndex,
        target_node: BaseNode,
        new_node: BaseNode,
    ) -> BaseNode:
        result = index.replace_target(target_node, new_node)
        if result is None:
            raise InvalidActionArgException(f"Target with index [{index}] not replaced")
        return result

    @property
    def input(self) -> ActionInput:
        raise NotImplementedError

    def output(self, state: State) -> ActionOutput:
        self.validate_args(input=self.input, state=state)
        return self._output(state)

    def _output(self, state: State) -> ActionOutput:
        raise NotImplementedError

    def apply(self, state: State) -> tuple[State, ActionOutput]:
        output = self.output(state)
        new_state = self._apply(state, output)
        return new_state, output

    def _apply(self, state: State, output: ActionOutput) -> State:
        if isinstance(output, NewPartialDefinitionActionOutput):
            partial_definition_idx = output.partial_definition_idx

            assert partial_definition_idx == len(state.partial_definitions or []) + 1

            partial_definitions = list(state.partial_definitions.as_tuple)
            partial_definitions.append(None)

            return State(
                function_group=state.definitions,
                partial_definitions=PartialDefinitionGroup.from_items(partial_definitions),
                arg_groups=state.arg_groups)
        elif isinstance(output, PartialActionOutput):
            partial_definition_idx = output.partial_definition_idx
            node_idx = output.node_idx
            new_function_info = output.new_function_info
            new_expr_args = output.new_expr_arg_group
            partial_definitions_list = list(state.partial_definitions.as_tuple)

            assert partial_definition_idx is not None
            assert partial_definition_idx > 0
            assert partial_definition_idx <= len(partial_definitions_list)

            if new_expr_args is not None:
                assert len(new_function_info.params) <= len(new_expr_args.inner_args)

                new_expr = new_function_info.expr.subs({
                    old_param.param: new_arg
                    for i, old_param in enumerate(new_function_info.params.as_tuple)
                    if (new_arg := new_expr_args.inner_args[i]) is not None
                })

                old_params = new_expr.find(Param).intersection(
                    new_function_info.params)
                assert len(old_params) == 0

                return state.change_partial_definition(
                    partial_definition_idx=partial_definition_idx,
                    node_idx=node_idx,
                    new_function_info=Function(
                        new_expr,
                        FunctionParams(*new_expr_args.outer_params)))
            else:
                return state.change_partial_definition(
                    partial_definition_idx=partial_definition_idx,
                    node_idx=node_idx,
                    new_function_info=new_function_info)
        elif isinstance(output, RemovePartialDefinitionActionOutput):
            partial_definition_idx = output.partial_definition_idx
            partial_definitions_list = list(state.partial_definitions or [])

            assert partial_definition_idx is not None
            assert partial_definition_idx > 0
            assert partial_definition_idx <= len(partial_definitions_list)

            partial_definitions_list = [
                expr
                for i, expr in enumerate(partial_definitions_list)
                if i != partial_definition_idx - 1
            ]

            return State.from_raw(
                definitions=state.definitions,
                partial_definitions=tuple(partial_definitions_list),
                arg_groups=state.arg_groups)
        elif isinstance(output, NewArgGroupActionOutput):
            arg_group_idx = output.arg_group_idx
            params_amount = output.params_amount
            args_amount = output.args_amount

            assert arg_group_idx == len(state.arg_groups or []) + 1

            arg_groups = list(state.arg_groups or [])
            arg_groups.append(ParamsArgsGroup(
                ParamsGroup(*[
                    Param(i + 1)
                    for i in range(params_amount)
                ]),
                ArgsGroup.from_items([None] * args_amount)))

            return State.from_raw(
                definitions=state.definitions,
                partial_definitions=state.partial_definitions,
                arg_groups=tuple(arg_groups))
        elif isinstance(output, ArgFromExprActionOutput):
            arg_group_idx = output.arg_group_idx
            arg_idx = output.arg_idx
            new_function_info = output.new_function_info

            arg_groups = list(state.arg_groups.as_tuple)
            assert len(arg_groups) > 0
            assert arg_group_idx > 0
            assert arg_group_idx <= len(arg_groups)

            arg_group = arg_groups[arg_group_idx - 1]
            assert arg_idx > 0
            assert arg_idx <= len(arg_group.inner_args)

            assert len(new_function_info.params) <= len(arg_group.outer_params)

            new_expr = new_function_info.expr.subs({
                old_param.param: arg_group.outer_params[i]
                for i, old_param in enumerate(new_function_info.params.as_tuple)
            })

            arg_groups_list = list(arg_groups)
            arg_group_args = list(arg_group.inner_args)
            arg_group_args[arg_idx - 1] = new_expr
            arg_groups_list[arg_group_idx - 1] = ParamsArgsGroup(
                ParamsGroup(*arg_group.outer_params),
                ArgsGroup.from_items(arg_group_args))

            return State(
                function_group=state.definitions,
                partial_definitions=state.partial_definitions,
                arg_groups=PartialArgsOuterGroup.from_items(arg_groups_list))
        elif isinstance(output, RemoveArgGroupActionOutput):
            arg_group_idx = output.arg_group_idx
            arg_groups = list(state.arg_groups.as_tuple)

            assert arg_group_idx > 0
            assert arg_group_idx <= len(arg_groups)

            arg_groups_list = [
                arg_group
                for i, arg_group in enumerate(arg_groups)
                if i != arg_group_idx - 1
            ]

            return State(
                function_group=state.definitions,
                partial_definitions=state.partial_definitions,
                arg_groups=PartialArgsOuterGroup.from_items(arg_groups_list))
        elif isinstance(output, NewDefinitionFromPartialActionOutput):
            definition_idx = output.definition_idx
            assert definition_idx == len(state.definitions or []) + 1

            partial_definition_idx = output.partial_definition_idx
            assert partial_definition_idx is not None
            assert partial_definition_idx > 0
            assert partial_definition_idx <= len(state.partial_definitions or [])

            partial_definitions_list = list(state.partial_definitions.as_tuple)
            function_info = partial_definitions_list[partial_definition_idx - 1]
            assert function_info is not None

            definitions_list = list(state.definitions.as_tuple)
            definitions_list.append(FunctionDefinitionNode(
                FunctionKey(),
                function_info))

            partial_definitions_list = [
                expr
                for i, expr in enumerate(partial_definitions_list)
                if i != partial_definition_idx - 1
            ]

            return State(
                function_group=FunctionDefinitionGroup.from_items(definitions_list),
                partial_definitions=PartialDefinitionGroup.from_items(partial_definitions_list),
                arg_groups=state.arg_groups)
        elif isinstance(output, ReplaceByDefinitionActionOutput):
            definition_idx = output.definition_idx
            expr_id = output.expr_id
            definitions = state.definitions.as_tuple

            assert definitions is not None
            assert definition_idx is not None
            assert definition_idx > 0
            assert definition_idx <= len(definitions)
            assert expr_id is not None

            definition_node = definitions[definition_idx - 1]
            assert definition_node is not None

            key = definition_node.definition_key
            definition_info = definition_node.function_info
            target = state.get_expr(expr_id)

            assert target is not None
            assert not target.readonly
            assert len(definition_info.params) <= len(target.function_info.params)
            assert definition_info == target

            return state.apply_new_expr(
                expr_id=expr_id,
                new_function_info=Function(key, FunctionParams()))
        elif isinstance(output, ExpandDefinitionActionOutput):
            definition_idx = output.definition_idx
            expr_id = output.expr_id
            definitions = state.definitions.as_tuple

            assert len(definitions) > 0
            assert definition_idx is not None
            assert definition_idx > 0
            assert definition_idx <= len(definitions)
            assert expr_id is not None

            definition_node = definitions[definition_idx - 1]
            assert definition_node is not None

            key = definition_node.definition_key
            definition_info = definition_node.function_info
            target = state.get_expr(expr_id)

            assert target is not None
            assert not target.readonly
            target_expr = target.function_info.expr
            assert key(target_expr.args) == target_expr

            return state.apply_new_expr(expr_id=expr_id, new_function_info=definition_info)
        elif isinstance(output, ReformulationActionOutput):
            expr_id = output.expr_id
            new_function_info = output.new_function_info
            return state.apply_new_expr(expr_id=expr_id, new_function_info=new_function_info)
        else:
            raise ValueError(f"Invalid action output: {output}")

###########################################################
################### BASE IMPLEMENTATION ###################
###########################################################

class EmptyArgsBaseAction(Action):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        return IntTypeGroup()

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        cls.validate_args(input=input, state=None)
        return cls(input=input)

    def __init__(self, input: ActionInput):
        self._input = input

    @property
    def input(self) -> ActionInput:
        return self._input

    def _output(self, state: State) -> ActionOutput:
        raise NotImplementedError

class DoubleIntInputBaseAction(Action):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        return IntTypeGroup.from_types(
            ActionArgInt,
            ActionArgInt,
        )

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            value1=input.args[0].value,
            value2=input.args[1].value,
        )

    def __init__(self, input: ActionInput, value1: int, value2: int):
        self._input = input
        self._value1 = value1
        self._value2 = value2

    @property
    def value1(self) -> int:
        return self._value1

    @property
    def value2(self) -> int:
        return self._value2

    @property
    def input(self) -> ActionInput:
        return self._input

    def _output(self, state: State) -> ActionOutput:
        raise NotImplementedError

class ValueTypeBaseAction(Action):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        raise NotImplementedError

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        raise NotImplementedError

    def __init__(
        self,
        input: ActionInput,
        value_type: int,
        value: int,
    ):
        self._input = input
        self._value_type = value_type
        self._value = value

    @property
    def value_type(self) -> int:
        return self._value_type

    @property
    def value(self) -> int:
        return self._value

    @property
    def input(self) -> ActionInput:
        return self._input

    def _output(self, state: State) -> ActionOutput:
        raise NotImplementedError

    def get_function_info(self, state: State) -> Function:
        if self.value_type == VALUE_TYPE_INT:
            return Function(Param(self.value), FunctionParams())
        elif self.value_type == VALUE_TYPE_PARAM:
            return Function(Param(self.value), FunctionParams())
        elif self.value_type == VALUE_TYPE_META_DEFINITION:
            raise NotImplementedError
        elif self.value_type == VALUE_TYPE_META_PROPOSITION:
            raise NotImplementedError
        elif self.value_type == VALUE_TYPE_DEFINITION_KEY:
            definition_idx = self.value
            definitions = state.definitions.as_tuple
            if definition_idx <= 0:
                raise InvalidActionArgException(f"Invalid definition: {definition_idx}")
            if definition_idx > len(definitions or []):
                raise InvalidActionArgException(f"Invalid definition: {definition_idx}")
            definition_node = definitions[definition_idx]
            definition_key = definition_node.definition_key
            function_info = definition_node.function_info
            # TODO instantiate the type of key (use wrap)
            # when replacing the key, also replace the wrapper
            return Function(definition_key, function_info.params)
        elif self.value_type == VALUE_TYPE_DEFINITION_EXPR:
            definition_idx = self.value
            definitions = state.definitions.as_tuple
            if definition_idx <= 0:
                raise InvalidActionArgException(f"Invalid definition: {definition_idx}")
            if definition_idx > len(definitions or []):
                raise InvalidActionArgException(f"Invalid definition: {definition_idx}")
            definition_node = definitions[definition_idx]
            return definition_node.function_info
        elif self.value_type == VALUE_TYPE_PARTIAL_DEFINITION:
            definition_idx = self.value
            partial_definitions = state.partial_definitions.as_tuple
            if definition_idx <= 0:
                raise InvalidActionArgException(f"Invalid partial definition: {definition_idx}")
            if definition_idx > len(partial_definitions):
                raise InvalidActionArgException(f"Invalid partial definition: {definition_idx}")
            definition_expr_p = partial_definitions[definition_idx]
            if definition_expr_p is None:
                raise InvalidActionArgException(f"Empty partial definition: {definition_idx}")
            return definition_expr_p
        elif self.value_type == VALUE_TYPE_PROPOSITION:
            raise NotImplementedError
        else:
            raise InvalidActionArgException(
                f"Invalid value type: {self.value_type}")

class ArgValueTypeBaseAction(ValueTypeBaseAction):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        return IntTypeGroup.from_types(
            ActionArgArgGroup,
            ActionArgArgIdx,
            ActionArgValueType,
            ActionArgInt,
        )

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            arg_group_idx=input.args[0].value,
            arg_idx=input.args[1].value,
            value_type=input.args[2].value,
            value=input.args[3].value,
        )

    def __init__(
        self,
        input: ActionInput,
        arg_group_idx: int,
        arg_idx: int,
        value_type: int,
        value: int,
    ):
        super().__init__(
            input=input,
            value_type=value_type,
            value=value,
        )

        self._arg_group_idx = arg_group_idx
        self._arg_idx = arg_idx

    @property
    def arg_group_idx(self) -> int:
        return self._arg_group_idx

    @property
    def arg_idx(self) -> int:
        return self._arg_idx

    def _output(self, state: State) -> ArgFromExprActionOutput:
        raise NotImplementedError

class PartialDefinitionValueTypeBaseAction(ValueTypeBaseAction):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        return IntTypeGroup.from_types(
            ActionArgPartialDefinition,
            ActionArgValueType,
            ActionArgInt,
        )

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            partial_definition_idx=input.args[0].value,
            value_type=input.args[1].value,
            value=input.args[2].value,
        )

    def __init__(
        self,
        input: ActionInput,
        partial_definition_idx: int,
        value_type: int,
        value: int,
    ):
        super().__init__(
            input=input,
            value_type=value_type,
            value=value,
        )

        self._partial_definition_idx = partial_definition_idx

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

    def _output(self, state: State) -> PartialActionOutput:
        raise NotImplementedError

class DefinitionExprBaseAction(Action):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        return IntTypeGroup.from_types(
            ActionArgDefinition,
            ActionArgExpressionTarget,
        )

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            definition_idx=input.args[0].value,
            target_expr_id=input.args[1].value,
        )

    def __init__(self, input: ActionInput, definition_idx: int, target_expr_id: int):
        self._input = input
        self._definition_idx = definition_idx
        self._target_expr_id = target_expr_id

    @property
    def definition_idx(self) -> int:
        return self._definition_idx

    @property
    def target_expr_id(self) -> int:
        return self._target_expr_id

    @property
    def input(self) -> ActionInput:
        return self._input

    def _output(self, state: State) -> ActionOutput:
        raise NotImplementedError

class PartialDefinitionBaseAction(Action):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        raise NotImplementedError

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        raise NotImplementedError

    def __init__(
        self,
        input: ActionInput,
        partial_definition_idx: int,
        node_idx: int,
    ):
        self._input = input
        self._partial_definition_idx = partial_definition_idx
        self._node_idx = node_idx

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

    @property
    def node_idx(self) -> int:
        return self._node_idx

    @property
    def input(self) -> ActionInput:
        return self._input

    def _output(self, state: State) -> PartialActionOutput:
        raise NotImplementedError

    def get_partial_definition_info(
        self,
        state: State,
    ) -> tuple[Function | None, Function | None, int | None]:
        partial_definition_idx = self.partial_definition_idx
        partial_definitions_list = list(state.partial_definitions.as_tuple)

        if partial_definition_idx < 0:
            raise InvalidActionArgException(
                f"Invalid partial definition: {partial_definition_idx}")
        if partial_definition_idx >= len(partial_definitions_list):
            raise InvalidActionArgException(
                f"Invalid partial definition: {partial_definition_idx}")

        root_info = partial_definitions_list[partial_definition_idx]

        if self.node_idx == 1:
            return root_info, None, None

        if root_info is None:
            raise InvalidActionArgException(
                f"Invalid node index: {self.node_idx}" \
                + f" (empty partial definition: {partial_definition_idx})")

        expr, parent_expr, child_idx = state.get_expr_full_info(
            root=root_info.expr,
            node_idx=self.node_idx)
        function_info = (
            Function(expr, root_info.params)
            if expr is not None
            else None)
        parent_function_info = (
            Function(parent_expr, root_info.params)
            if parent_expr is not None
            else None)

        return function_info, parent_function_info, child_idx

class PartialNodeFromExprOuterParamsBaseAction(PartialDefinitionBaseAction):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        return IntTypeGroup.from_types(
            ActionArgPartialDefinition,
            ActionArgPartialNode,
            ActionArgExpressionOrigin,
        )

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            partial_definition_idx=input.args[0].value,
            node_idx=input.args[1].value,
            origin_expr_id=input.args[2].value,
        )

    def __init__(
        self,
        input: ActionInput,
        partial_definition_idx: int,
        node_idx: int,
        origin_expr_id: int,
    ):
        super().__init__(
            input=input,
            partial_definition_idx=partial_definition_idx,
            node_idx=node_idx,
        )

        self._origin_expr_id = origin_expr_id

    @property
    def origin_expr_id(self) -> int:
        return self._origin_expr_id

    def _output(self, state: State) -> PartialActionOutput:
        raise NotImplementedError

class PartialNodeFromExprWithArgsBaseAction(PartialDefinitionBaseAction):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        return IntTypeGroup.from_types(
            ActionArgPartialDefinition,
            ActionArgPartialNode,
            ActionArgExpressionOrigin,
            ActionArgArgGroup,
        )

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            partial_definition_idx=input.args[0].value,
            node_idx=input.args[1].value,
            origin_expr_id=input.args[2].value,
            expr_arg_group_idx=input.args[3].value,
        )

    def __init__(
        self,
        input: ActionInput,
        partial_definition_idx: int,
        node_idx: int,
        origin_expr_id: int,
        expr_arg_group_idx: int,
    ):
        super().__init__(
            input=input,
            partial_definition_idx=partial_definition_idx,
            node_idx=node_idx,
        )

        self._origin_expr_id = origin_expr_id
        self._expr_arg_group_idx = expr_arg_group_idx

    @property
    def origin_expr_id(self) -> int:
        return self._origin_expr_id

    @property
    def expr_arg_group_idx(self) -> int:
        return self._expr_arg_group_idx

    def _output(self, state: State) -> PartialActionOutput:
        raise NotImplementedError

###########################################################
################## IMPLEMENTATION (MAIN) ##################
###########################################################

class NewPartialDefinitionAction(EmptyArgsBaseAction):

    def _output(self, state: State) -> ActionOutput:
        partial_definition_idx = len(state.partial_definitions or []) + 1
        return NewPartialDefinitionActionOutput(partial_definition_idx=partial_definition_idx)

class PartialDefinitionValueTypeAction(PartialDefinitionValueTypeBaseAction):

    def _output(self, state: State) -> PartialActionOutput:
        partial_definition_idx = self.partial_definition_idx
        new_function_info = self.get_function_info(state)

        return PartialActionOutput(
            partial_definition_idx=partial_definition_idx,
            node_idx=1,
            new_function_info=new_function_info,
            new_expr_arg_group=None)

class PartialNodeFromExprAction(PartialNodeFromExprOuterParamsBaseAction):

    def _output(self, state: State) -> PartialActionOutput:
        partial_definition_idx = self.partial_definition_idx
        node_idx = self.node_idx
        expr_id = self.origin_expr_id

        new_info = state.get_expr(expr_id)
        if new_info is None:
            raise InvalidActionArgException(
                f"Invalid node index: {expr_id}" \
                + f" (partial_definition_idx: {partial_definition_idx})")

        if node_idx == 1:
            return PartialActionOutput(
                partial_definition_idx=partial_definition_idx,
                node_idx=node_idx,
                new_function_info=new_info.function_info,
                new_expr_arg_group=None)

        function_info, _, _ = self.get_partial_definition_info(state)

        if function_info is None:
            raise InvalidActionArgException(
                f"Invalid node index: {node_idx}" \
                + f" (partial_definition_idx: {partial_definition_idx})")

        return PartialActionOutput(
            partial_definition_idx=partial_definition_idx,
            node_idx=node_idx,
            new_function_info=new_info.function_info,
            new_expr_arg_group=None)

class PartialNodeFromExprWithArgsAction(PartialNodeFromExprWithArgsBaseAction):

    def _output(self, state: State) -> PartialActionOutput:
        partial_definition_idx = self.partial_definition_idx
        node_idx = self.node_idx
        expr_id = self.origin_expr_id
        expr_arg_group_idx = self.expr_arg_group_idx

        arg_groups = state.arg_groups.as_tuple

        new_info = state.get_expr(expr_id)
        if new_info is None:
            raise InvalidActionArgException(f"Invalid node index: {expr_id}")

        if expr_arg_group_idx <= 0 or expr_arg_group_idx > len(arg_groups):
            raise InvalidActionArgException(f"Invalid expr arg group index: {expr_arg_group_idx}")

        new_expr_args = arg_groups[expr_arg_group_idx - 1]
        if len(new_info.function_info.params) > len(new_expr_args.inner_args):
            raise InvalidActionArgException(
                "New expression amount of params invalid: " \
                + f"{len(new_info.function_info.params)} > {len(new_expr_args.inner_args)}")

        dependencies = new_info.function_info.expr.find(Param).intersection(
            new_info.function_info.params)
        dependencies_idxs = sorted([p.value for p in dependencies])
        for i, param in enumerate(new_info.function_info.params):
            arg_expr = new_expr_args.inner_args[i]
            if arg_expr is None and param in dependencies:
                raise InvalidActionArgException(
                    f"Missing param: {param.value} (need all of {dependencies_idxs})")

        return PartialActionOutput(
            partial_definition_idx=partial_definition_idx,
            node_idx=node_idx,
            new_function_info=new_info.function_info,
            new_expr_arg_group=new_expr_args)

class NewArgGroupAction(DoubleIntInputBaseAction):

    def _output(self, state: State) -> ActionOutput:
        arg_group_idx = len(state.arg_groups or []) + 1
        return NewArgGroupActionOutput(
            arg_group_idx=arg_group_idx,
            params_amount=self.value1,
            args_amount=self.value2)

class ArgValueTypeAction(ArgValueTypeBaseAction):

    def _output(self, state: State) -> ArgFromExprActionOutput:
        arg_group_idx = self.arg_group_idx
        arg_idx = self.arg_idx
        arg_groups = state.arg_groups.as_tuple

        if len(arg_groups) == 0:
            raise InvalidActionArgException("No arg groups yet")
        if arg_group_idx <= 0 or arg_group_idx > len(arg_groups):
            raise InvalidActionArgException(f"Invalid arg group index: {arg_group_idx}")

        arg_group = arg_groups[arg_group_idx - 1]
        if arg_idx <= 0 or arg_idx > len(arg_group.inner_args):
            raise InvalidActionArgException(f"Invalid arg index: {arg_idx}")

        new_function_info = self.get_function_info(state)

        return ArgFromExprActionOutput(
            arg_group_idx=arg_group_idx,
            arg_idx=arg_idx,
            new_function_info=new_function_info)

class NewDefinitionFromPartialAction(Action):

    @classmethod
    def metadata(cls) -> IntTypeGroup:
        return IntTypeGroup.from_types(ActionArgPartialDefinition)

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            partial_definition_idx=input.args[0].value,
        )

    def __init__(self, input: ActionInput, partial_definition_idx: int):
        self._input = input
        self._partial_definition_idx = partial_definition_idx

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

    @property
    def input(self) -> ActionInput:
        return self._input

    def _output(self, state: State) -> ActionOutput:
        self.validate_args(input=self.input, state=state)

        partial_definition_idx = self.partial_definition_idx
        partial_definitions = list(state.partial_definitions.as_tuple)
        if partial_definition_idx <= 0 or partial_definition_idx > len(partial_definitions):
            raise InvalidActionArgException(
                f"Invalid partial definition index: {partial_definition_idx}")
        partial_definition = partial_definitions[partial_definition_idx - 1]
        if not partial_definition:
            raise InvalidActionArgException(
                f"Partial definition {partial_definition_idx} has no expression")
        definition_idx = len(state.definitions or []) + 1
        return NewDefinitionFromPartialActionOutput(
            definition_idx=definition_idx,
            partial_definition_idx=partial_definition_idx)

class ReplaceByDefinitionAction(DefinitionExprBaseAction):

    def _output(self, state: State) -> ActionOutput:
        definition_idx = self.definition_idx
        target_expr_id = self.target_expr_id
        definitions = state.definitions.as_tuple
        if len(definitions) == 0:
            raise InvalidActionArgException("No definitions yet")
        if definition_idx <= 0 or definition_idx > len(state.definitions or []):
            raise InvalidActionArgException(f"Invalid definition index: {definition_idx}")
        definition_node = definitions[definition_idx - 1]
        key = definition_node.definition_key
        definition_info = definition_node.function_info
        target = state.get_expr(target_expr_id)
        if not target:
            raise InvalidActionArgException(f"Invalid target node index: {target_expr_id}")
        if target.readonly:
            raise InvalidActionArgException(f"Target {target_expr_id} is readonly")
        if len(definition_info.params) > len(target.function_info.params):
            raise InvalidActionArgException(
                f"Invalid amount of params: {len(definition_info.params)} > "
                + f"(expected {len(target.function_info.params)})")
        if definition_info != target:
            raise InvalidActionArgException(
                f"Invalid target node: {target} "
                + f"(expected {definition_info} from definition {key})")
        return ReplaceByDefinitionActionOutput(
            definition_idx=definition_idx,
            expr_id=target_expr_id)

class ExpandDefinitionAction(DefinitionExprBaseAction):

    def _output(self, state: State) -> ActionOutput:
        definition_idx = self.definition_idx
        target_expr_id = self.target_expr_id
        definitions = state.definitions.as_tuple
        if len(definitions) == 0:
            raise InvalidActionArgException("No definitions yet")
        if definition_idx <= 0 or definition_idx > len(state.definitions or []):
            raise InvalidActionArgException(f"Invalid definition index: {definition_idx}")
        definition_key, definition_function_info = definitions[definition_idx - 1]
        if not definition_function_info:
            raise InvalidActionArgException(f"Definition {definition_idx} has no expression")
        target_function_info = state.get_expr(target_expr_id)
        if target_function_info is None:
            raise InvalidActionArgException(f"No target node found with index: {target_expr_id}")
        if target_function_info.readonly:
            raise InvalidActionArgException(f"Target {target_expr_id} is readonly")
        if definition_key != target_function_info:
            raise InvalidActionArgException(
                f"Invalid target node: {target_function_info} (expected {definition_key})")
        return ExpandDefinitionActionOutput(definition_idx=definition_idx, expr_id=target_expr_id)

BASIC_ACTION_TYPES = (
    NewPartialDefinitionAction,
    PartialDefinitionValueTypeAction,
    PartialNodeFromExprAction,
    PartialNodeFromExprWithArgsAction,
    NewArgGroupAction,
    ArgValueTypeAction,
    NewDefinitionFromPartialAction,
    ReplaceByDefinitionAction,
    ExpandDefinitionAction,
)
