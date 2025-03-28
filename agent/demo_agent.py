import typing
from env.action import RawAction
from env.base_agent import BaseAgent
from env.full_state import FullState


class DemoAgent(BaseAgent):
    """
    A simple agent that executes a predefined sequence of actions.
    This agent is useful for testing and demonstration purposes.
    """

    def __init__(self, actions: typing.Sequence[RawAction]):
        """
        Initialize the DemoAgent with a sequence of actions.

        Args:
            actions: A list of RawAction objects to be executed in order
        """
        self.actions = actions
        self.action_index = 0
        self.total_actions = len(actions)

    def select_action(self, state: FullState) -> RawAction:
        """
        Select the next action in the sequence based on the current state.

        Args:
            state: Current environment state (not used in this implementation)

        Returns:
            The next RawAction in the sequence

        Raises:
            IndexError: If all actions have already been executed
        """
        if self.action_index >= self.total_actions:
            raise IndexError(
                f"All {self.total_actions} actions have been executed. No more actions available.")

        action = self.actions[self.action_index]
        self.action_index += 1
        return action

    def train(
        self,
        state: FullState,
        action: RawAction,
        reward: float,
        next_state: FullState,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        This agent doesn't learn, so the train method is a no-op.
        """

    def reset(self) -> None:
        """
        Reset the agent to start executing actions from the beginning of the sequence.
        """
        self.action_index = 0
