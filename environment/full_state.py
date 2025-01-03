import typing
from environment.core import (
    InheritableNode,
    BaseGroup,
    IOptional)
from environment.state import State
from environment.meta_env import MetaInfo, IMetaData

class HistoryNode(InheritableNode):
    def __init__(self, state: State, meta_data: IOptional[IMetaData]):
        super().__init__(state, meta_data)

    @property
    def state(self) -> State:
        state = self.args[0]
        return typing.cast(State, state)

    @property
    def meta_data(self) -> IOptional[IMetaData]:
        meta_data = self.args[1]
        return typing.cast(IOptional[IMetaData], meta_data)

class HistoryGroupNode(BaseGroup[HistoryNode]):

    @classmethod
    def item_type(cls) -> type[HistoryNode]:
        return HistoryNode

class FullState(InheritableNode):
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

###########################################################
######################### INDICES #########################
###########################################################

# class DefaultTypeIndex
