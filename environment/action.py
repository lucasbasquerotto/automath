from __future__ import annotations
import typing
from abc import ABC
from utils.logger import logger
from environment.core import (
    INode,
    InheritableNode,
    IExceptionInfo,
    Optional,
    TypeNode,
    IsInsideRange,
    InvalidNodeException,
    IOptional,
    IInstantiable)
from environment.state import State
from environment.meta_env import IMetaData
from environment.full_state import FullState, HistoryNode, HistoryGroupNode

###########################################################
########################## MAIN ###########################
###########################################################

class IActionOutput(INode, ABC):

    def apply(self, full_state: FullState) -> State:
        raise NotImplementedError

T = typing.TypeVar('T', bound=INode)
O = typing.TypeVar('O', bound=IActionOutput)

class IAction(INode, typing.Generic[O], ABC):

    def as_action(self) -> 'BaseAction[O]':
        raise NotImplementedError

class FullActionOutput(InheritableNode, typing.Generic[O], IInstantiable):

    def __init__(
        self,
        output: O,
        new_state: State,
    ):
        assert isinstance(output, IActionOutput)
        assert isinstance(new_state, State)
        super().__init__(output, new_state)

    @property
    def output(self) -> O:
        output = self.args[0]
        return typing.cast(O, output)

    @property
    def new_state(self) -> State:
        new_state = self.args[1]
        return typing.cast(State, new_state)

###########################################################
####################### ACTION DATA #######################
###########################################################

class ActionData(InheritableNode, IMetaData, IInstantiable):
    def __init__(
        self,
        action: IOptional['BaseAction'],
        output: IOptional[IActionOutput],
        exception: IOptional[IExceptionInfo],
    ):
        super().__init__(action, output, exception)

    @property
    def action(self) -> IOptional['BaseAction']:
        action = self.args[0]
        return typing.cast(IOptional[BaseAction], action)

    @property
    def output(self) -> IOptional[IActionOutput]:
        output = self.args[1]
        return typing.cast(IOptional[IActionOutput], output)

    @property
    def exception(self) -> IOptional[IExceptionInfo]:
        exception = self.args[2]
        return typing.cast(IOptional[IExceptionInfo], exception)

###########################################################
#################### ACTION EXCEPTION #####################
###########################################################

class IActionExceptionInfo(IExceptionInfo, ABC):

    def to_action_data(self) -> ActionData:
        raise NotImplementedError

class InvalidActionException(InvalidNodeException):

    def __init__(self, info: IActionExceptionInfo):
        super().__init__(info)

    @property
    def info(self) -> IActionExceptionInfo:
        info = self.args[0]
        return typing.cast(IActionExceptionInfo, info)

    def to_action_data(self) -> ActionData:
        return self.info.to_action_data()

class ActionTypeExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    def __init__(
        self,
        action_type: TypeNode['BaseAction'],
        exception: IExceptionInfo,
    ):
        super().__init__(action_type, exception)

    def to_action_data(self) -> ActionData:
        return ActionData(
            action=Optional.create(),
            output=Optional.create(),
            exception=Optional(self),
        )

class ActionInputExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    def __init__(
        self,
        action: 'BaseAction',
        exception: IExceptionInfo,
    ):
        assert isinstance(action, BaseAction)
        assert isinstance(exception, IExceptionInfo)
        super().__init__(action, exception)

    def to_action_data(self) -> ActionData:
        return ActionData(
            action=Optional(typing.cast(BaseAction, self.args[0])),
            output=Optional.create(),
            exception=Optional(typing.cast(IExceptionInfo, self.args[1])),
        )

class ActionOutputExceptionInfo(InheritableNode, IExceptionInfo, IInstantiable):

    def __init__(
        self,
        action: 'BaseAction',
        output: IActionOutput,
        exception: IExceptionInfo,
    ):
        assert isinstance(action, BaseAction)
        assert isinstance(output, IActionOutput)
        assert isinstance(exception, IExceptionInfo)
        super().__init__(action, output, exception)

    def to_action_data(self) -> ActionData:
        return ActionData(
            action=Optional(typing.cast(BaseAction, self.args[0])),
            output=Optional(typing.cast(IActionOutput, self.args[1])),
            exception=Optional(typing.cast(IExceptionInfo, self.args[2])),
        )

###########################################################
##################### IMPLEMENTATION ######################
###########################################################

class BaseAction(InheritableNode, IAction[O], typing.Generic[O], ABC):

    def as_action(self) -> typing.Self:
        return self

    def _run(self, full_state: FullState) -> O:
        raise NotImplementedError

    def inner_run(self, full_state: FullState) -> FullActionOutput[O]:
        try:
            allowed_actions = full_state.meta.allowed_actions
            min_index = 1
            max_index = len(allowed_actions.subtypes.as_tuple)
            action_type = allowed_actions.subtypes.as_tuple.index(self.wrap_type()) + 1
            IsInsideRange.from_raw(
                value=action_type,
                min_value=min_index,
                max_value=max_index,
            ).raise_on_false()
        except InvalidNodeException as e:
            raise ActionTypeExceptionInfo(self.wrap_type(), e.info).as_exception()

        try:
            output = self._run(full_state)
            assert isinstance(output, IActionOutput)
        except InvalidNodeException as e:
            raise ActionInputExceptionInfo(self, e.info).as_exception()

        try:
            new_state = output.apply(full_state)
            return FullActionOutput(output, new_state)
        except InvalidNodeException as e:
            raise ActionOutputExceptionInfo(self, output, e.info).as_exception()

    def run_action(self, full_state: FullState) -> FullState:
        try:
            full_output = self.inner_run(full_state)
            output = full_output.output
            next_state = full_output.new_state
            action_data = ActionData(
                action=Optional(self),
                output=Optional(output),
                exception=Optional.create(),
            )
        except InvalidActionException as e:
            logger.debug(f"Invalid action: {e}")
            next_state = full_state.current.state
            action_data = e.to_action_data()

        current = HistoryNode(next_state, Optional(action_data))

        history = list(full_state.history.as_tuple)
        history.append(full_state.current)

        max_history_state_size = full_state.meta.options.max_history_state_size.value
        if max_history_state_size is not None:
            history = history[-max_history_state_size.to_int:]

        return FullState(
            full_state.meta,
            current=current,
            history=HistoryGroupNode.from_items(history),
        )

class GeneralAction(
    BaseAction[typing.Self], # type: ignore[misc]
    IActionOutput,
    ABC,
):
    pass
