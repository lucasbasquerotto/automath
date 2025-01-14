from __future__ import annotations
import typing
from abc import ABC
from utils.env_logger import env_logger
from env.core import (
    InheritableNode,
    IExceptionInfo,
    Optional,
    TypeNode,
    IsInsideRange,
    InvalidNodeException,
    IOptional,
    ExtendedTypeGroup,
    CountableTypeGroup,
    IInt,
    TmpNestedArg,
    IInstantiable)
from env.state import State
from env.meta_env import (
    IMetaData,
    MetaInfo,
    IActionOutput,
    IAction,
    IBasicAction,
    GeneralTypeGroup,
    MetaInfoOptions)
from env.full_state import FullState, HistoryNode, HistoryGroupNode

###########################################################
########################## MAIN ###########################
###########################################################

O = typing.TypeVar('O', bound=IActionOutput)

class FullActionOutput(InheritableNode, IInstantiable):

    idx_output = 1
    idx_new_state = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IActionOutput,
            State,
        ]))

    @property
    def output(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_output)

    @property
    def new_state(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_new_state)

###########################################################
####################### ACTION DATA #######################
###########################################################

class ActionData(InheritableNode, IMetaData, IInstantiable):

    idx_action = 1
    idx_output = 2
    idx_exception = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IOptional[IAction],
            IOptional[IActionOutput],
            IOptional[IExceptionInfo],
        ]))

    @property
    def action(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_action)

    @property
    def output(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_output)

    @property
    def exception(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_exception)

    @classmethod
    def from_args(
        cls,
        action: IOptional[IAction],
        output: IOptional[IActionOutput],
        exception: IOptional[IExceptionInfo],
    ) -> typing.Self:
        return cls(action, output, exception)

###########################################################
#################### ACTION EXCEPTION #####################
###########################################################

class IActionExceptionInfo(IExceptionInfo, ABC):

    def to_action_data(self) -> ActionData:
        raise NotImplementedError

    def as_exception(self):
        return InvalidActionException(self)

class InvalidActionException(InvalidNodeException):

    @property
    def info(self) -> IActionExceptionInfo:
        info = self.args[0]
        return typing.cast(IActionExceptionInfo, info)

    def to_action_data(self) -> ActionData:
        return self.info.to_action_data()

class ActionTypeExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_action_type = 1
    idx_exception = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            TypeNode[IAction],
            IExceptionInfo,
        ]))

    def to_action_data(self) -> ActionData:
        return ActionData.from_args(
            action=Optional.create(),
            output=Optional.create(),
            exception=Optional(self),
        )

class ActionInputExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_action = 1
    idx_exception = 2

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IAction,
            IExceptionInfo,
        ]))

    @property
    def action(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_action)

    @property
    def exception(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_exception)

    def to_action_data(self) -> ActionData:
        return ActionData.from_args(
            action=Optional(self.action.apply().cast(BaseAction)),
            output=Optional.create(),
            exception=Optional(self.exception.apply().cast(IExceptionInfo)),
        )

class ActionOutputExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_action = 1
    idx_output = 2
    idx_exception = 3

    @classmethod
    def arg_type_group(cls) -> ExtendedTypeGroup:
        return ExtendedTypeGroup(CountableTypeGroup.from_types([
            IAction,
            IActionOutput,
            IExceptionInfo,
        ]))

    @property
    def action(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_action)

    @property
    def output(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_output)

    @property
    def exception(self) -> TmpNestedArg:
        return self.nested_arg(self.idx_exception)

    def to_action_data(self) -> ActionData:
        return ActionData.from_args(
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

    def _run(self, full_state: FullState) -> O:
        raise NotImplementedError

    def inner_run(self, full_state: FullState) -> FullActionOutput:
        try:
            meta = full_state.meta.apply().cast(MetaInfo)
            allowed_actions = meta.allowed_actions.apply().cast(GeneralTypeGroup[IAction])
            min_index = 1
            max_index = len(allowed_actions.as_tuple)
            action_type = allowed_actions.as_tuple.index(self.as_type()) + 1
            IsInsideRange.from_raw(
                value=action_type,
                min_value=min_index,
                max_value=max_index,
            ).raise_on_false()
        except InvalidNodeException as e:
            raise ActionTypeExceptionInfo(self.as_type(), e.info).as_exception() from e

        try:
            output = self._run(full_state)
            assert isinstance(output, IActionOutput)
        except InvalidNodeException as e:
            raise ActionInputExceptionInfo(self, e.info).as_exception() from e

        try:
            new_state = output.apply(full_state)
            return FullActionOutput(output, new_state)
        except InvalidNodeException as e:
            # make a new exception from the above, following the comment bellow
            raise ActionOutputExceptionInfo(self, output, e.info).as_exception() from e


    def run_action(self, full_state: FullState) -> FullState:
        meta = full_state.meta.apply().cast(MetaInfo)
        options = meta.options.apply().cast(MetaInfoOptions)
        max_history_state_size = options.max_history_state_size.apply().cast(IOptional[IInt]).value
        current = full_state.current.apply().cast(HistoryNode)

        try:
            full_output = self.inner_run(full_state)
            output = full_output.output.apply().cast(IActionOutput)
            next_state = full_output.new_state.apply().cast(State)
            action_data = ActionData.from_args(
                action=Optional(self),
                output=Optional(output),
                exception=Optional.create(),
            )
        except InvalidActionException as e:
            env_logger.error(e, exc_info=True)
            next_state = current.state.apply().cast(State)
            action_data = e.to_action_data()

        current = HistoryNode(next_state, Optional(action_data))

        history = list(full_state.history.apply().cast(HistoryGroupNode).as_tuple)
        history.append(current)

        if max_history_state_size is not None:
            history = history[-max_history_state_size.as_int:]

        return FullState(
            full_state.meta.apply(),
            current,
            HistoryGroupNode.from_items(history),
        )

class BasicAction(
    BaseAction[O],
    IBasicAction[FullState],
    typing.Generic[O],
    ABC,
):
    pass

class GeneralAction(
    BaseAction[typing.Self], # type: ignore[misc]
    IActionOutput,
    ABC,
):
    def _run(self, full_state: FullState) -> typing.Self:
        return self
