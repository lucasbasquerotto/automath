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
    IInstantiable)
from env.state import State
from env.meta_env import (
    IMetaData,
    MetaInfo,
    IActionOutput,
    IAction,
    IBasicAction,
    SubtypeOuterGroup,
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

    def to_action_data(self) -> ActionData:
        return ActionData.from_args(
            action=Optional(typing.cast(BaseAction, self.args[self.idx_action])),
            output=Optional.create(),
            exception=Optional(typing.cast(IExceptionInfo, self.args[self.idx_exception])),
        )

class ActionOutputExceptionInfo(InheritableNode, IExceptionInfo, IInstantiable):

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

    def to_action_data(self) -> ActionData:
        return ActionData.from_args(
            action=Optional(typing.cast(BaseAction, self.args[self.idx_action])),
            output=Optional(typing.cast(IActionOutput, self.args[self.idx_output])),
            exception=Optional(typing.cast(IExceptionInfo, self.args[self.idx_exception])),
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
            meta = typing.cast(MetaInfo, full_state.args[full_state.idx_meta])
            allowed_actions = typing.cast(
                SubtypeOuterGroup[IAction],
                meta.args[meta.idx_allowed_actions])
            min_index = 1
            subtypes = typing.cast(
                GeneralTypeGroup[IAction],
                allowed_actions.args[allowed_actions.idx_subtypes])
            max_index = len(subtypes.as_tuple)
            action_type = subtypes.as_tuple.index(self.as_type()) + 1
            IsInsideRange.from_raw(
                value=action_type,
                min_value=min_index,
                max_value=max_index,
            ).raise_on_false()
        except InvalidNodeException as e:
            raise ActionTypeExceptionInfo(self.as_type(), e.info).as_exception()

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
        meta = typing.cast(MetaInfo, full_state.args[full_state.idx_meta])
        options = typing.cast(MetaInfoOptions, meta.args[meta.idx_options])
        max_history_state_size = typing.cast(
            IOptional[IInt],
            options.args[options.idx_max_history_state_size],
        ).value
        current = typing.cast(HistoryNode, full_state.args[full_state.idx_current])

        try:
            full_output = self.inner_run(full_state)
            output = typing.cast(IActionOutput, full_output.args[full_output.idx_output])
            next_state = typing.cast(State, full_output.args[full_output.idx_new_state])
            action_data = ActionData.from_args(
                action=Optional(self),
                output=Optional(output),
                exception=Optional.create(),
            )
        except InvalidActionException as e:
            env_logger.debug(f"Invalid action: {e}")
            next_state = typing.cast(State, current.args[current.idx_state])
            action_data = e.to_action_data()

        current = HistoryNode(next_state, Optional(action_data))

        history = list(typing.cast(
            HistoryGroupNode,
            full_state.args[full_state.idx_history],
        ).as_tuple)
        history.append(current)

        if max_history_state_size is not None:
            history = history[-max_history_state_size.as_int:]

        return FullState(
            full_state.args[full_state.idx_meta],
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
    pass
