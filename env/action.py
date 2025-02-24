from __future__ import annotations
import typing
from abc import ABC
from utils.env_logger import env_logger
from env.core import (
    InheritableNode,
    IExceptionInfo,
    Optional,
    IsInsideRange,
    InvalidNodeException,
    IOptional,
    Protocol,
    CountableTypeGroup,
    IInt,
    Integer,
    GreaterThan,
    IWrapper,
    CompositeType,
    OptionalTypeGroup,
    TmpInnerArg,
    IInstantiable)
from env.state import State
from env.meta_env import (
    MetaData,
    MetaInfo,
    IActionOutput,
    IAction,
    IBasicAction,
    IRawAction,
    GeneralTypeGroup,
    MetaInfoOptions)
from env.full_state import (
    FullState,
    HistoryNode,
    HistoryGroupNode,
    BaseActionData,
    RawActionErrorActionData,
    ActionTypeErrorActionData,
    ActionErrorActionData,
    ActionOutputErrorActionData,
    SuccessActionData,
    MetaAllowedBasicActionsTypeIndex,
)
from env.symbol import Symbol

###########################################################
########################## MAIN ###########################
###########################################################

O = typing.TypeVar('O', bound=IActionOutput)

class FullActionOutput(InheritableNode, IWrapper, IInstantiable):

    idx_raw_action = 1
    idx_action = 2
    idx_output = 3
    idx_new_state = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            IAction.as_type(),
            IActionOutput.as_type(),
            State.as_type(),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action)

    @property
    def output(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_output)

    @property
    def new_state(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_new_state)

    @classmethod
    def with_args(
        cls,
        raw_action: Optional[IRawAction],
        action: IAction,
        output: IActionOutput,
        new_state: State,
    ) -> FullActionOutput:
        return cls(
            raw_action,
            action,
            output,
            new_state,
        )

###########################################################
#################### ACTION EXCEPTION #####################
###########################################################

class IActionExceptionInfo(IExceptionInfo, ABC):

    def to_action_data(self) -> BaseActionData:
        raise NotImplementedError

    def as_exception(self):
        return InvalidActionException(self)

class InvalidActionException(InvalidNodeException):

    @property
    def info(self) -> IActionExceptionInfo:
        info = self.args[0]
        return typing.cast(IActionExceptionInfo, info)

    def to_action_data(self) -> BaseActionData:
        return self.info.to_action_data()

class RawActionExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_raw_action = 1
    idx_exception = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            RawAction.as_type(),
            IExceptionInfo.as_type(),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    def to_action_data(self) -> RawActionErrorActionData:
        return RawActionErrorActionData.from_args(
            raw_action=Optional(self.raw_action.apply()),
            action=Optional(),
            output=Optional(),
            exception=Optional(self.exception.apply().cast(IExceptionInfo)),
        )

class ActionTypeExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_raw_action = 1
    idx_action = 2
    idx_exception = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            IAction.as_type(),
            IExceptionInfo.as_type(),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    def to_action_data(self) -> ActionTypeErrorActionData:
        return ActionTypeErrorActionData.from_args(
            raw_action=self.raw_action.apply().real(Optional[IRawAction]),
            action=Optional(self.action.apply()),
            output=Optional(),
            exception=Optional(self.exception.apply().cast(IExceptionInfo)),
        )

class ActionInputExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_raw_action = 1
    idx_action = 2
    idx_exception = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            IAction.as_type(),
            IExceptionInfo.as_type(),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    def to_action_data(self) -> ActionErrorActionData:
        return ActionErrorActionData.from_args(
            raw_action=self.raw_action.apply().real(Optional[IRawAction]),
            action=Optional(self.action.apply()),
            output=Optional(),
            exception=Optional(self.exception.apply().cast(IExceptionInfo)),
        )

class ActionOutputExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_raw_action = 1
    idx_action = 2
    idx_output = 3
    idx_exception = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            IAction.as_type(),
            IActionOutput.as_type(),
            IExceptionInfo.as_type(),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action)

    @property
    def output(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_output)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    def to_action_data(self) -> ActionOutputErrorActionData:
        return ActionOutputErrorActionData.from_args(
            raw_action=self.raw_action.apply().real(Optional[IRawAction]),
            action=Optional(self.action.apply().cast(BaseAction)),
            output=Optional(self.output.apply().cast(IActionOutput)),
            exception=Optional(self.exception.apply().cast(IExceptionInfo)),
        )

###########################################################
##################### IMPLEMENTATION ######################
###########################################################

class BaseAction(InheritableNode, IAction[FullState], typing.Generic[O], ABC):

    def as_action(self) -> typing.Self:
        return self

    def _run_action(self, full_state: FullState) -> O:
        raise NotImplementedError

    def inner_run(self, full_state: FullState) -> FullActionOutput:
        action: BaseAction = self
        raw_action: RawAction | None = None

        if isinstance(action, RawAction):
            raw_action = action

            try:
                raw_action.strict_validate()
                action_aux = raw_action.to_action(full_state)
                assert isinstance(action_aux, BaseAction)
                action = action_aux
            except InvalidNodeException as e:
                raise RawActionExceptionInfo(raw_action, e.info).as_exception() from e

        try:
            action.strict_validate()
            meta = full_state.meta.apply().cast(MetaInfo)
            allowed_actions = meta.allowed_actions.apply().cast(GeneralTypeGroup[IAction])
            min_index = 1
            max_index = len(allowed_actions.as_tuple)
            action_type = allowed_actions.as_tuple.index(action.as_type()) + 1
            IsInsideRange.from_raw(
                value=action_type,
                min_value=min_index,
                max_value=max_index,
            ).raise_on_false()
        except InvalidNodeException as e:
            raise ActionTypeExceptionInfo(
                Optional.with_value(raw_action),
                action.as_type(),
                e.info,
            ).as_exception() from e

        try:
            # pylint: disable=protected-access
            output = action._run_action(full_state)
            assert isinstance(output, IActionOutput)
        except InvalidNodeException as e:
            raise ActionInputExceptionInfo(
                Optional.with_value(raw_action),
                action,
                e.info,
            ).as_exception() from e

        try:
            new_state = output.run_output(full_state)
            result = FullActionOutput.with_args(
                raw_action=Optional.with_value(raw_action),
                action=action,
                output=output,
                new_state=new_state)
            result.strict_validate()
            return result
        except InvalidNodeException as e:
            raise ActionOutputExceptionInfo(
                Optional.with_value(raw_action),
                action,
                output,
                e.info,
            ).as_exception() from e

    def run_action_details(self, full_state: FullState) -> tuple[FullState, BaseActionData]:
        meta = full_state.meta.apply().cast(MetaInfo)
        options = meta.options.apply().cast(MetaInfoOptions)
        max_history_state_size = options.max_history_state_size.apply().cast(
            IOptional[IInt]
        ).value
        current = full_state.current.apply().cast(HistoryNode)
        meta_data = current.meta_data.apply().cast(MetaData)
        remaining_steps_opt = meta_data.remaining_steps.apply().cast(Optional[Integer])
        remaining_steps = (
            (remaining_steps_opt.value.as_int)
            if remaining_steps_opt.value is not None
            else None)

        try:
            if remaining_steps is not None:
                GreaterThan.with_args(remaining_steps, 0).raise_on_false()
            full_output = self.inner_run(full_state)
            raw_action_opt = full_output.raw_action.apply().cast(Optional[IRawAction])
            actual_action = full_output.action.apply().cast(BaseAction)
            output = full_output.output.apply().cast(IActionOutput)
            next_state = full_output.new_state.apply().cast(State)
            next_state.strict_validate()
            action_data: BaseActionData = SuccessActionData.from_args(
                raw_action=raw_action_opt,
                action=Optional(actual_action),
                output=Optional(output),
                exception=Optional.create(),
            )
        except InvalidActionException as e:
            symbol = Symbol(
                node=e.info.as_node,
                node_types=full_state.node_types(),
            )
            env_logger.debug(str(symbol), exc_info=e)
            next_state = current.state.apply().cast(State)
            action_data = e.to_action_data()

        action_data.strict_validate()
        remaining_steps = (
            remaining_steps - 1
            if remaining_steps is not None
            else None)
        meta_data = meta_data.with_new_args(
            remaining_steps
        )

        current = current.with_new_args(
            action_data=Optional(action_data),
        )

        history = list(full_state.history.apply().cast(HistoryGroupNode).as_tuple)
        history.append(current)

        current = HistoryNode.with_args(
            state=next_state,
            meta_data=meta_data,
        )

        if max_history_state_size is not None:
            history = history[-max_history_state_size.as_int:]

        return FullState.with_args(
            meta=meta,
            current=current,
            history=HistoryGroupNode.from_items(history),
        ), action_data

    def run_action(self, full_state: FullState) -> FullState:
        full_state, _ = self.run_action_details(full_state)
        return full_state

class BasicAction(
    BaseAction[O],
    IBasicAction[FullState],
    typing.Generic[O],
    ABC,
):
    pass

class GeneralAction(
    BaseAction[typing.Self], # type: ignore[misc]
    IActionOutput[FullState],
    ABC,
):
    def _run_action(self, full_state: FullState) -> typing.Self:
        return self

class RawAction(BaseAction[IActionOutput], IRawAction[FullState], IInstantiable):

    idx_action_index = 1
    idx_arg1 = 2
    idx_arg2 = 3
    idx_arg3 = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            MetaAllowedBasicActionsTypeIndex.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
        ))

    @property
    def action_index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action_index)

    @property
    def arg1(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg1)

    @property
    def arg2(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg2)

    @property
    def arg3(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg3)

    def to_action(self, full_state: FullState) -> IBasicAction[FullState]:
        action_index = self.action_index.apply().real(MetaAllowedBasicActionsTypeIndex)
        arg1 = self.arg1.apply().real(Integer)
        arg2 = self.arg2.apply().real(Integer)
        arg3 = self.arg3.apply().real(Integer)

        action_type = action_index.find_in_outer_node(full_state).value_or_raise

        basic_action = action_type.type.from_raw(arg1.as_int, arg2.as_int, arg3.as_int)

        return basic_action
