# pylint: disable=C0302
from utils.types import FunctionDefinition, ParamVar, ExprInfo
from environment.state import State, ArgGroup

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

ARG_TYPES = [
    ACTION_ARG_TYPE_PARTIAL_DEFINITION,
    ACTION_ARG_TYPE_ARG_GROUP,
    ACTION_ARG_TYPE_ARG_IDX,
    ACTION_ARG_TYPE_DEFINITION,
    ACTION_ARG_TYPE_EXPRESSION_ORIGIN,
    ACTION_ARG_TYPE_EXPRESSION_TARGET,
    ACTION_ARG_TYPE_PARTIAL_NODE,
    ACTION_ARG_TYPE_INT,
]

def _validate_args():
    for i, arg in enumerate(ARG_TYPES):
        assert arg == i + 1, f"Invalid arg type: {arg} (expected: {i + 1})"
_validate_args()

###########################################################
###################### ACTION INPUT #######################
###########################################################

ActionArgType = int

class ActionArgsMetaInfo:
    def __init__(
        self,
        arg_types: tuple[ActionArgType, ...],
    ):
        self._arg_types = arg_types

    @property
    def arg_types(self) -> tuple[ActionArgType, ...]:
        return self._arg_types

class ActionMetaInfo(ActionArgsMetaInfo):
    def __init__(
        self,
        type_idx: int,
        arg_types: tuple[ActionArgType, ...],
    ):
        super().__init__(arg_types=arg_types)
        self._type_idx = type_idx

    @property
    def type_idx(self) -> int:
        return self._type_idx

class ActionArg:
    def __init__(self, type: ActionArgType, value: int):
        if type not in ARG_TYPES:
            raise InvalidActionArgException(f"Invalid action arg type: {type}")
        if not isinstance(value, int):
            raise InvalidActionArgException(f"Invalid action arg value: {value}")
        self._type = type
        self._value = value

    @property
    def type(self) -> ActionArgType:
        return self._type

    @property
    def value(self) -> int:
        return self._value

class ActionInput:
    def __init__(self, args: tuple[ActionArg, ...]):
        self.args = args

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
        new_expr_info: ExprInfo,
        new_expr_args: ArgGroup | None,
    ):
        self._partial_definition_idx = partial_definition_idx
        self._node_idx = node_idx
        self._new_expr_info = new_expr_info
        self._new_expr_args = new_expr_args

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

    @property
    def node_idx(self) -> int:
        return self._node_idx

    @property
    def new_expr_info(self) -> ExprInfo:
        return self._new_expr_info

    @property
    def new_expr_args(self) -> ArgGroup | None:
        return self._new_expr_args

class RemovePartialDefinitionActionOutput:
    def __init__(self, partial_definition_idx: int):
        self._partial_definition_idx = partial_definition_idx

    @property
    def partial_definition_idx(self) -> int:
        return self._partial_definition_idx

class NewArgGroupActionOutput:
    def __init__(self, arg_group_idx: int, amount: int):
        self._arg_group_idx = arg_group_idx
        self._amount = amount

    @property
    def arg_group_idx(self) -> int:
        return self._arg_group_idx

    @property
    def amount(self) -> int:
        return self._amount

class ArgFromExprActionOutput:
    def __init__(
        self,
        arg_group_idx: int,
        arg_idx: int,
        new_expr_info: ExprInfo,
    ):
        self._arg_group_idx = arg_group_idx
        self._arg_idx = arg_idx
        self._new_expr_info = new_expr_info

    @property
    def arg_group_idx(self) -> int:
        return self._arg_group_idx

    @property
    def arg_idx(self) -> int:
        return self._arg_idx

    @property
    def new_expr_info(self) -> ExprInfo:
        return self._new_expr_info

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
    def __init__(self, expr_id: int, new_expr_info: ExprInfo):
        self._expr_id = expr_id
        self._new_expr_info = new_expr_info

    @property
    def expr_id(self) -> int:
        return self._expr_id

    @property
    def new_expr_info(self) -> ExprInfo:
        return self._new_expr_info

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
    def metadata(cls) -> ActionArgsMetaInfo:
        raise NotImplementedError()

    @classmethod
    def validate_args(cls, input: ActionInput, state: State | None) -> None:
        arg_types = cls.metadata().arg_types
        if len(input.args) != len(arg_types):
            raise InvalidActionArgsException(f"Invalid action length: {len(input.args)}")
        for i, arg_type in enumerate(arg_types):
            arg_info = input.args[i]
            if not arg_type in ARG_TYPES:
                raise InvalidActionArgException(f"Invalid action arg type: {arg_type}")
            if arg_type != arg_info.type:
                raise InvalidActionArgException(
                    f"Invalid action arg type: {arg_type} != {input.args[i].type}")
            if state is not None:
                if arg_type == ACTION_ARG_TYPE_PARTIAL_DEFINITION:
                    partial_definition_idx = arg_info.value
                    if partial_definition_idx <= 0:
                        raise InvalidActionArgException(
                            f"Invalid partial definition: {partial_definition_idx}")
                    if partial_definition_idx > len(state.partial_definitions or []):
                        raise InvalidActionArgException(
                            f"Invalid partial definition: {partial_definition_idx}")
                elif arg_type == ACTION_ARG_TYPE_ARG_GROUP:
                    arg_group_idx = arg_info.value
                    if arg_group_idx <= 0:
                        raise InvalidActionArgException(f"Invalid arg group: {arg_group_idx}")
                    if arg_group_idx > len(state.arg_groups or []):
                        raise InvalidActionArgException(f"Invalid arg group: {arg_group_idx}")
                elif arg_type == ACTION_ARG_TYPE_ARG_IDX:
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

                    arg_group_idx = input.args[i - 1].value
                    arg_group = state.arg_groups[arg_group_idx - 1]
                    if arg_idx > arg_group.amount:
                        raise InvalidActionArgException(
                            f"Invalid arg index: {arg_idx} (group={arg_group_idx}, " + \
                            f"max={arg_group.amount})")
                elif arg_type == ACTION_ARG_TYPE_DEFINITION:
                    definition_idx = arg_info.value
                    if definition_idx <= 0:
                        raise InvalidActionArgException(f"Invalid definition: {definition_idx}")
                    if definition_idx > len(state.definitions or []):
                        raise InvalidActionArgException(f"Invalid definition: {definition_idx}")
                elif arg_type == ACTION_ARG_TYPE_EXPRESSION_ORIGIN:
                    expr_id = arg_info.value
                    if expr_id <= 0:
                        raise InvalidActionArgException(f"Invalid origin expr: {expr_id}")
                    expr_info = state.get_expr(expr_id)
                    if not expr_info:
                        raise InvalidActionArgException(f"Invalid origin expr: {expr_id}")
                elif arg_type == ACTION_ARG_TYPE_EXPRESSION_TARGET:
                    expr_id = arg_info.value
                    if expr_id <= 0:
                        raise InvalidActionArgException(f"Invalid target expr: {expr_id}")
                    expr_info = state.get_expr(expr_id)
                    if expr_info and expr_info.readonly:
                        raise InvalidActionArgException(f"Target expr is readonly: {expr_id}")
                elif arg_type == ACTION_ARG_TYPE_PARTIAL_NODE:
                    node_idx = arg_info.value
                    if node_idx <= 0:
                        raise InvalidActionArgException(f"Invalid node index: {expr_id}")

                    # Must be right after a partial definition
                    if i == 0:
                        raise InvalidActionArgException(
                            "The argument index must be right after a partial definition " + \
                            "(expected another argument before)")

                    expected_partial_definition = arg_types[i - 1]
                    if expected_partial_definition != ACTION_ARG_TYPE_PARTIAL_DEFINITION:
                        raise InvalidActionArgException(
                            f"Invalid argument order: {expected_partial_definition} != " + \
                            f"{ACTION_ARG_TYPE_PARTIAL_DEFINITION}")

                    partial_definition_idx = input.args[i - 1].value
                    if partial_definition_idx <= 0:
                        raise InvalidActionArgException(
                            f"Invalid partial definition: {partial_definition_idx}")
                    if partial_definition_idx > len(state.partial_definitions or []):
                        raise InvalidActionArgException(
                            f"Invalid partial definition: {partial_definition_idx}")

                    if node_idx > 1:
                        partial_definition = state.partial_definitions[partial_definition_idx - 1]
                        node = State.get_partial_definition_node(partial_definition, node_idx)
                        if node is None:
                            raise InvalidActionArgException(
                                f"Invalid node index: {node_idx} " + \
                                f"(partial={partial_definition_idx})")
                elif arg_type == ACTION_ARG_TYPE_INT:
                    pass
                else:
                    raise InvalidActionArgException(f"Invalid action arg type: {arg_type}")


    @classmethod
    def to_input(cls, action: tuple[int, ...]) -> ActionInput:
        if len(action) != len(cls.metadata().arg_types):
            raise InvalidActionArgsException(f"Invalid action length: {len(action)}")

        args = [
            ActionArg(type, value)
            for type, value in zip(cls.metadata().arg_types, action)
        ]
        return ActionInput(tuple(args))

    @classmethod
    def create(cls, input: ActionInput) -> 'Action':
        cls.validate_args(input=input, state=None)
        return cls._create(input)

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        raise NotImplementedError()

    @property
    def input(self) -> ActionInput:
        raise NotImplementedError()

    def output(self, state: State) -> ActionOutput:
        self.validate_args(input=self.input, state=state)
        return self._output(state)

    def _output(self, state: State) -> ActionOutput:
        raise NotImplementedError()


    def apply(self, state: State) -> 'State':
        output = self.output(state)

        if isinstance(output, NewPartialDefinitionActionOutput):
            partial_definition_idx = output.partial_definition_idx

            assert partial_definition_idx == len(state.partial_definitions or []) + 1, \
                f"Invalid partial definition index: {partial_definition_idx}"

            partial_definitions = list(state.partial_definitions or [])
            partial_definitions.append(None)

            return State(
                definitions=state.definitions,
                partial_definitions=tuple(partial_definitions),
                arg_groups=state.arg_groups)
        elif isinstance(output, PartialActionOutput):
            partial_definition_idx = output.partial_definition_idx
            node_idx = output.node_idx
            new_expr_info = output.new_expr_info
            new_expr_args = output.new_expr_args
            partial_definitions_list = list(state.partial_definitions or [])

            assert partial_definition_idx is not None, "Empty partial definition index"
            assert partial_definition_idx > 0, \
                f"Invalid partial definition index: {partial_definition_idx}"
            assert partial_definition_idx <= len(partial_definitions_list), \
                f"Invalid partial definition index: {partial_definition_idx}"

            if new_expr_args is not None:
                assert len(new_expr_info.params) <= len(new_expr_args.params), \
                    "New expression amount of params invalid: " \
                    + f"{len(new_expr_info.params)} > {len(new_expr_args.params)}"

                new_expr = new_expr_info.expr.subs({
                    old_param: new_expr_args.expressions[i]
                    for i, old_param in enumerate(new_expr_info.params)
                    if new_expr_args.expressions[i] is not None
                })

                old_params: set[ParamVar] = new_expr.atoms(ParamVar).intersection(
                    new_expr_info.params)
                old_params_idxs = sorted([p.index for p in old_params])
                assert len(old_params) == 0, f"Old params not replaced: {old_params_idxs}"

                return state.change_partial_definition(
                    partial_definition_idx=partial_definition_idx,
                    node_idx=node_idx,
                    new_expr_info=ExprInfo(
                        expr=new_expr,
                        params=new_expr_args.params))
            else:
                return state.change_partial_definition(
                    partial_definition_idx=partial_definition_idx,
                    node_idx=node_idx,
                    new_expr_info=new_expr_info)
        elif isinstance(output, RemovePartialDefinitionActionOutput):
            partial_definition_idx = output.partial_definition_idx
            partial_definitions_list = list(state.partial_definitions or [])

            assert partial_definition_idx is not None, "Empty partial definition index"
            assert partial_definition_idx > 0, \
                f"Invalid partial definition index: {partial_definition_idx}"
            assert partial_definition_idx <= len(partial_definitions_list), \
                f"Invalid partial definition index: {partial_definition_idx}"

            partial_definitions_list = [
                expr
                for i, expr in enumerate(partial_definitions_list)
                if i != partial_definition_idx - 1
            ]

            return State(
                definitions=state.definitions,
                partial_definitions=tuple(partial_definitions_list),
                arg_groups=state.arg_groups)
        elif isinstance(output, NewArgGroupActionOutput):
            arg_group_idx = output.arg_group_idx
            amount = output.amount

            assert arg_group_idx == len(state.arg_groups or []) + 1, \
                f"Invalid arg group index: {arg_group_idx}"

            arg_groups = list(state.arg_groups or [])
            arg_groups.append(ArgGroup(
                amount=amount,
                params=tuple([
                    ParamVar(i + 1)
                    for i in range(amount)
                ]),
                expressions=tuple([None] * amount)))

            return State(
                definitions=state.definitions,
                partial_definitions=state.partial_definitions,
                arg_groups=tuple(arg_groups))
        elif isinstance(output, ArgFromExprActionOutput):
            arg_group_idx = output.arg_group_idx
            arg_idx = output.arg_idx
            new_expr_info = output.new_expr_info

            arg_groups = list(state.arg_groups or [])
            assert len(arg_groups) > 0, "No arg groups yet"
            assert arg_group_idx > 0, f"Invalid arg group index: {arg_group_idx}"
            assert arg_group_idx <= len(arg_groups), \
                f"Invalid arg group index: {arg_group_idx} (max={len(arg_groups)})"

            arg_group = arg_groups[arg_group_idx - 1]
            assert arg_idx > 0, f"Invalid arg index: {arg_idx}"
            assert arg_idx <= arg_group.amount, \
                f"Invalid arg index: {arg_idx} (max={arg_group.amount})"

            assert len(new_expr_info.params) <= len(arg_group.params), \
                "New expression amount of params invalid: " \
                + f"{len(new_expr_info.params)} > {len(arg_group.params)}"

            new_expr = new_expr_info.expr.subs({
                old_param: arg_group.params[i]
                for i, old_param in enumerate(new_expr_info.params)
            })

            arg_groups_list = list(arg_groups)
            arg_group_list = list(arg_group.expressions)
            arg_group_list[arg_idx - 1] = new_expr
            arg_groups_list[arg_group_idx - 1] = ArgGroup(
                amount=arg_group.amount,
                params=arg_group.params,
                expressions=tuple(arg_group_list))

            return State(
                definitions=state.definitions,
                partial_definitions=state.partial_definitions,
                arg_groups=tuple(arg_groups_list))
        elif isinstance(output, RemoveArgGroupActionOutput):
            arg_group_idx = output.arg_group_idx
            arg_groups = list(state.arg_groups or [])

            assert arg_group_idx > 0, f"Invalid arg group index: {arg_group_idx}"
            assert arg_group_idx <= len(arg_groups), \
                f"Invalid arg group index: {arg_group_idx} (max={len(arg_groups)})"

            arg_groups_list = [
                arg_group
                for i, arg_group in enumerate(arg_groups)
                if i != arg_group_idx - 1
            ]

            return State(
                definitions=state.definitions,
                partial_definitions=state.partial_definitions,
                arg_groups=tuple(arg_groups_list))
        elif isinstance(output, NewDefinitionFromPartialActionOutput):
            definition_idx = output.definition_idx
            assert definition_idx == len(state.definitions or []) + 1, \
                f"Invalid definition index: {definition_idx}"

            partial_definition_idx = output.partial_definition_idx
            assert partial_definition_idx is not None, "Empty partial definition index"
            assert partial_definition_idx > 0, \
                f"Invalid partial definition index: {partial_definition_idx}"
            assert partial_definition_idx <= len(state.partial_definitions or []), \
                f"Invalid partial definition index: {partial_definition_idx}"

            partial_definitions_list = list(state.partial_definitions or [])
            expr = partial_definitions_list[partial_definition_idx - 1]
            assert expr is not None, "Empty expression for partial definition"

            definitions_list = list(state.definitions or [])
            definition_idx = len(definitions_list) + 1
            definitions_list.append((FunctionDefinition(definition_idx), expr))

            partial_definitions_list = [
                expr
                for i, expr in enumerate(partial_definitions_list)
                if i != partial_definition_idx - 1
            ]

            return State(
                definitions=tuple(definitions_list),
                partial_definitions=tuple(partial_definitions_list),
                arg_groups=state.arg_groups)
        elif isinstance(output, ReplaceByDefinitionActionOutput):
            definition_idx = output.definition_idx
            expr_id = output.expr_id
            definitions = state.definitions

            assert definitions is not None, "No definitions yet"
            assert definition_idx is not None, "Empty definition index"
            assert definition_idx > 0, f"Invalid definition index: {definition_idx}"
            assert definition_idx <= len(definitions), \
                f"Invalid definition index: {definition_idx}"
            assert expr_id is not None, "Empty expression id"

            key, definition_info = definitions[definition_idx - 1]
            target_expr_info = state.get_expr(expr_id)

            assert definition_info is not None, "Empty definition node"
            assert target_expr_info is not None, "Empty target node"
            assert not target_expr_info.readonly, "Target node is readonly"
            assert len(definition_info.params) <= len(target_expr_info.params), \
                "Definition amount of params invalid: " \
                + f"{len(definition_info.params)} > {len(target_expr_info.params)}"
            assert definition_info == target_expr_info, \
                f"Invalid definition node: {definition_info.expr}" + \
                f" (expected {target_expr_info.expr})"

            return state.apply_new_expr(
                expr_id=expr_id,
                new_expr_info=ExprInfo(expr=key, params=()))
        elif isinstance(output, ExpandDefinitionActionOutput):
            definition_idx = output.definition_idx
            expr_id = output.expr_id
            definitions = state.definitions

            assert len(definitions) > 0, "No definitions yet"
            assert definition_idx is not None, "Empty definition index"
            assert definition_idx > 0, f"Invalid definition index: {definition_idx}"
            assert definition_idx <= len(definitions), \
                f"Invalid definition index: {definition_idx}"
            assert expr_id is not None, "Empty expression id"

            key, definition_info = definitions[definition_idx - 1]
            target_expr_info = state.get_expr(expr_id)
            assert target_expr_info is not None, f"Target node not found (expr_id={expr_id})"
            assert not target_expr_info.readonly, "Target node is readonly"
            assert key == target_expr_info.expr, \
                f"Invalid target node: {target_expr_info.expr} (expected {key})"

            return state.apply_new_expr(expr_id=expr_id, new_expr_info=definition_info)
        elif isinstance(output, ReformulationActionOutput):
            expr_id = output.expr_id
            new_expr_info = output.new_expr_info
            return state.apply_new_expr(expr_id=expr_id, new_expr_info=new_expr_info)
        else:
            raise ValueError(f"Invalid action output: {output}")

###########################################################
################### BASE IMPLEMENTATION ###################
###########################################################

class EmptyArgsBaseAction(Action):

    @classmethod
    def metadata(cls) -> ActionArgsMetaInfo:
        return ActionArgsMetaInfo(tuple())

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
        raise NotImplementedError()

class IntInputBaseAction(Action):

    @classmethod
    def metadata(cls) -> ActionArgsMetaInfo:
        return ActionArgsMetaInfo((ACTION_ARG_TYPE_INT,))

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            value=input.args[0].value,
        )

    def __init__(self, input: ActionInput, value: int):
        self._input = input
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    @property
    def input(self) -> ActionInput:
        return self._input

    def _output(self, state: State) -> ActionOutput:
        raise NotImplementedError()

class ArgFromExprBaseAction(Action):

    @classmethod
    def metadata(cls) -> ActionArgsMetaInfo:
        return ActionArgsMetaInfo((
            ACTION_ARG_TYPE_ARG_GROUP,
            ACTION_ARG_TYPE_ARG_IDX,
            ACTION_ARG_TYPE_EXPRESSION_ORIGIN,
        ))

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        return cls(
            input=input,
            arg_group_idx=input.args[0].value,
            arg_idx=input.args[1].value,
            origin_expr_id=input.args[1].value,
        )

    def __init__(self, input: ActionInput, arg_group_idx: int, arg_idx: int, origin_expr_id: int):
        self._input = input
        self._arg_group_idx = arg_group_idx
        self._arg_idx = arg_idx
        self._origin_expr_id = origin_expr_id

    @property
    def arg_group_idx(self) -> int:
        return self._arg_group_idx

    @property
    def arg_idx(self) -> int:
        return self._arg_idx

    @property
    def origin_expr_id(self) -> int:
        return self._origin_expr_id

    @property
    def input(self) -> ActionInput:
        return self._input

    def _output(self, state: State) -> ArgFromExprActionOutput:
        raise NotImplementedError()

class DefinitionExprBaseAction(Action):

    @classmethod
    def metadata(cls) -> ActionArgsMetaInfo:
        return ActionArgsMetaInfo((
            ACTION_ARG_TYPE_DEFINITION,
            ACTION_ARG_TYPE_EXPRESSION_TARGET,
        ))

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
        raise NotImplementedError()

class PartialDefinitionBaseAction(Action):

    @classmethod
    def metadata(cls) -> ActionArgsMetaInfo:
        raise NotImplementedError()

    @classmethod
    def _create(cls, input: ActionInput) -> 'Action':
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_partial_definition_info(
        self,
        state: State,
    ) -> tuple[ExprInfo | None, ExprInfo | None, int | None]:
        partial_definition_idx = self.partial_definition_idx
        partial_definitions_list = list(state.partial_definitions or [])

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
        expr_info = (
            ExprInfo(expr=expr, params=root_info.params)
            if expr is not None
            else None)
        parent_expr_info = (
            ExprInfo(expr=parent_expr, params=root_info.params)
            if parent_expr is not None
            else None)

        return expr_info, parent_expr_info, child_idx

class PartialNodeFromExprOuterParamsBaseAction(PartialDefinitionBaseAction):

    @classmethod
    def metadata(cls) -> ActionArgsMetaInfo:
        return ActionArgsMetaInfo((
            ACTION_ARG_TYPE_PARTIAL_DEFINITION,
            ACTION_ARG_TYPE_PARTIAL_NODE,
            ACTION_ARG_TYPE_EXPRESSION_ORIGIN,
        ))

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
        raise NotImplementedError()

class PartialNodeFromExprWithArgsBaseAction(PartialDefinitionBaseAction):

    @classmethod
    def metadata(cls) -> ActionArgsMetaInfo:
        return ActionArgsMetaInfo((
            ACTION_ARG_TYPE_PARTIAL_DEFINITION,
            ACTION_ARG_TYPE_PARTIAL_NODE,
            ACTION_ARG_TYPE_EXPRESSION_ORIGIN,
            ACTION_ARG_TYPE_ARG_GROUP,
        ))

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
        raise NotImplementedError()

###########################################################
################## IMPLEMENTATION (MAIN) ##################
###########################################################

class NewPartialDefinitionAction(EmptyArgsBaseAction):

    def _output(self, state: State) -> ActionOutput:
        partial_definition_idx = len(state.partial_definitions or []) + 1
        return NewPartialDefinitionActionOutput(partial_definition_idx=partial_definition_idx)

class PartialNodeFromExprAction(PartialNodeFromExprOuterParamsBaseAction):

    def _output(self, state: State) -> PartialActionOutput:
        partial_definition_idx = self.partial_definition_idx
        node_idx = self.node_idx
        expr_id = self.origin_expr_id

        new_expr_info = state.get_expr(expr_id)
        if new_expr_info is None:
            raise InvalidActionArgException(
                f"Invalid node index: {expr_id}" \
                + f" (partial_definition_idx: {partial_definition_idx})")

        if node_idx == 1:
            return PartialActionOutput(
                partial_definition_idx=partial_definition_idx,
                node_idx=node_idx,
                new_expr_info=new_expr_info,
                new_expr_args=None)

        expr_info, _, _ = self.get_partial_definition_info(state)

        if expr_info is None:
            raise InvalidActionArgException(
                f"Invalid node index: {node_idx}" \
                + f" (partial_definition_idx: {partial_definition_idx})")

        return PartialActionOutput(
            partial_definition_idx=partial_definition_idx,
            node_idx=node_idx,
            new_expr_info=new_expr_info,
            new_expr_args=None)

class PartialNodeFromExprWithArgsAction(PartialNodeFromExprWithArgsBaseAction):

    def _output(self, state: State) -> PartialActionOutput:
        partial_definition_idx = self.partial_definition_idx
        node_idx = self.node_idx
        expr_id = self.origin_expr_id
        expr_arg_group_idx = self.expr_arg_group_idx

        new_expr_info = state.get_expr(expr_id)
        if new_expr_info is None:
            raise InvalidActionArgException(f"Invalid node index: {expr_id}")

        if expr_arg_group_idx <= 0 or expr_arg_group_idx > len(state.arg_groups):
            raise InvalidActionArgException(f"Invalid expr arg group index: {expr_arg_group_idx}")

        new_expr_args = state.arg_groups[expr_arg_group_idx - 1]
        if len(new_expr_info.params) > len(new_expr_args.params):
            raise InvalidActionArgException(
                "New expression amount of params invalid: " \
                + f"{len(new_expr_args.params)} > {len(new_expr_info.params)}")

        dependencies: set[ParamVar] = new_expr_info.expr.atoms(ParamVar).intersection(
            new_expr_info.params)
        dependencies_idxs = sorted([p.index for p in dependencies])
        for i, param in enumerate(new_expr_info.params):
            arg_expr = new_expr_args.expressions[i]
            if arg_expr is None and param in dependencies:
                raise InvalidActionArgException(
                    f"Missing param: {param.index} (need all of {dependencies_idxs})")

        return PartialActionOutput(
            partial_definition_idx=partial_definition_idx,
            node_idx=node_idx,
            new_expr_info=new_expr_info,
            new_expr_args=new_expr_args)

class NewArgGroupAction(IntInputBaseAction):

    def _output(self, state: State) -> ActionOutput:
        arg_group_idx = len(state.arg_groups or []) + 1
        return NewArgGroupActionOutput(arg_group_idx=arg_group_idx, amount=self.value)

class ArgFromExprAction(ArgFromExprBaseAction):

    def _output(self, state: State) -> ArgFromExprActionOutput:
        arg_group_idx = self.arg_group_idx
        arg_idx = self.arg_idx
        expr_id = self.origin_expr_id
        arg_groups = state.arg_groups

        if len(arg_groups) == 0:
            raise InvalidActionArgException("No arg groups yet")
        if arg_group_idx <= 0 or arg_group_idx > len(arg_groups):
            raise InvalidActionArgException(f"Invalid arg group index: {arg_group_idx}")

        arg_group = arg_groups[arg_group_idx - 1]
        if arg_idx <= 0 or arg_idx > arg_group.amount:
            raise InvalidActionArgException(f"Invalid arg index: {arg_idx}")

        new_expr_info = state.get_expr(expr_id)
        if new_expr_info is None:
            raise InvalidActionArgException(f"Node with index not found: {expr_id}")

        return ArgFromExprActionOutput(
            arg_group_idx=arg_group_idx,
            arg_idx=arg_idx,
            new_expr_info=new_expr_info)

class NewDefinitionFromPartialAction(Action):

    @classmethod
    def metadata(cls) -> ActionArgsMetaInfo:
        return ActionArgsMetaInfo((ACTION_ARG_TYPE_PARTIAL_DEFINITION,))

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
        partial_definitions = list(state.partial_definitions or [])
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
        definitions = state.definitions
        if len(definitions) == 0:
            raise InvalidActionArgException("No definitions yet")
        if definition_idx <= 0 or definition_idx > len(state.definitions or []):
            raise InvalidActionArgException(f"Invalid definition index: {definition_idx}")
        key, definition_expr_info = definitions[definition_idx - 1]
        target_expr_info = state.get_expr(target_expr_id)
        if not target_expr_info:
            raise InvalidActionArgException(f"Invalid target node index: {target_expr_id}")
        if target_expr_info.readonly:
            raise InvalidActionArgException(f"Target {target_expr_id} is readonly")
        if len(definition_expr_info.params) > len(target_expr_info.params):
            raise InvalidActionArgException(
                f"Invalid amount of params: {len(definition_expr_info.params)} > "
                + f"(expected {len(target_expr_info.params)})")
        if definition_expr_info != target_expr_info:
            raise InvalidActionArgException(
                f"Invalid target node: {target_expr_info} "
                + f"(expected {definition_expr_info} from definition {key})")
        return ReplaceByDefinitionActionOutput(
            definition_idx=definition_idx,
            expr_id=target_expr_id)

class ExpandDefinitionAction(DefinitionExprBaseAction):

    def _output(self, state: State) -> ActionOutput:
        definition_idx = self.definition_idx
        target_expr_id = self.target_expr_id
        definitions = state.definitions
        if len(definitions) == 0:
            raise InvalidActionArgException("No definitions yet")
        if definition_idx <= 0 or definition_idx > len(state.definitions or []):
            raise InvalidActionArgException(f"Invalid definition index: {definition_idx}")
        definition_key, definition_expr_info = definitions[definition_idx - 1]
        if not definition_expr_info:
            raise InvalidActionArgException(f"Definition {definition_idx} has no expression")
        target_expr_info = state.get_expr(target_expr_id)
        if target_expr_info is None:
            raise InvalidActionArgException(f"No target node found with index: {target_expr_id}")
        if target_expr_info.readonly:
            raise InvalidActionArgException(f"Target {target_expr_id} is readonly")
        if definition_key != target_expr_info:
            raise InvalidActionArgException(
                f"Invalid target node: {target_expr_info} (expected {definition_key})")
        return ExpandDefinitionActionOutput(definition_idx=definition_idx, expr_id=target_expr_id)

BASIC_ACTIONS = (
    NewPartialDefinitionAction,
    PartialNodeFromExprAction,
    PartialNodeFromExprWithArgsAction,
    NewArgGroupAction,
    ArgFromExprAction,
    NewDefinitionFromPartialAction,
    ReplaceByDefinitionAction,
    ExpandDefinitionAction,
)
