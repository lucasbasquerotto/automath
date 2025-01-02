import typing
from utils.logger import logger
from environment.core import Optional, InheritableNode, BaseGroup, Integer
from environment.state import State
from environment.action import (
    BaseAction,
    ActionOutput,
    InvalidActionException,
    IExceptionInfo,
    EmptyExceptionInfo,
    OutsideRangeExceptionInfo)
from environment.meta_env import MetaInfo, ActionData

class HistoryNode(InheritableNode):
    def __init__(self, state: State, action_data: Optional[ActionData]):
        super().__init__(state, action_data)

    @property
    def state(self) -> State:
        state = self.args[0]
        return typing.cast(State, state)

    @property
    def action_data(self) -> Optional[ActionData]:
        action_data = self.args[1]
        return typing.cast(Optional[ActionData], action_data)

class HistoryGroupNode(BaseGroup[HistoryNode]):

    @classmethod
    def item_type(cls) -> type[HistoryNode]:
        return HistoryNode

class FullStateNode(InheritableNode):
    def __init__(self, meta: MetaInfo, current: HistoryNode, history: HistoryGroupNode):
        super().__init__(meta, current, history)

    @property
    def meta(self) -> MetaInfo:
        meta = self.args[0]
        return typing.cast(MetaInfo, meta)

    @property
    def current(self) -> HistoryNode:
        current = self.args[1]
        return typing.cast(HistoryNode, current)

    @property
    def history(self) -> HistoryGroupNode:
        history = self.args[2]
        return typing.cast(HistoryGroupNode, history)

    def run_action(self, action_wrapper: Optional[BaseAction]) -> typing.Self:
        last_state = self.current.state
        assert isinstance(last_state, State)

        action: BaseAction | None = None
        output: ActionOutput | None = None
        exception: IExceptionInfo | None = None

        try:
            action = action_wrapper.value
            action = EmptyExceptionInfo.verify(action)

            allowed_actions = self.meta.allowed_actions
            min_index = 1
            max_index = len(allowed_actions.subtypes.as_tuple)
            action_type = allowed_actions.subtypes.as_tuple.index(action.wrap_type()) + 1
            OutsideRangeExceptionInfo.verify(
                value=Integer(action_type),
                min_value=min_index,
                max_value=max_index,
            )

            full_output = action.run(last_state)
            output = full_output.output
            next_state = full_output.new_state
        except InvalidActionException as e:
            logger.debug(f"Invalid action: {e}")
            next_state = last_state
            exception = e.info

        action_data = ActionData(
            action=Optional(action),
            output=Optional(output),
            exception=Optional(exception),
        )

        current = HistoryNode(next_state, Optional(action_data))

        history = list(self.history.as_tuple)
        history.append(self.current)

        max_history_state_size = self.meta.options.max_history_state_size.value
        if max_history_state_size is not None:
            history = history[-max_history_state_size.to_int:]

        return self.func(
            self.meta,
            current=current,
            history=HistoryGroupNode.from_items(history),
        )
