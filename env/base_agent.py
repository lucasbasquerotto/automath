from abc import ABC
from typing import TypeVar, Generic
from env.action import RawAction
from env.full_state import FullState

T = TypeVar('T')

class BaseAgent(Generic[T], ABC):
    """Base agent interface that defines the core functionality needed for training."""

    def select_action(self, state: FullState) -> RawAction:
        """Select an action based on the current state.

        Args:
            state: Current environment state

        Returns:
            Selected action to perform
        """
        raise NotImplementedError()

    def train(
        self,
        state: FullState,
        action: RawAction,
        reward: float,
        next_state: FullState,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Train the agent using an experience tuple.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            terminated: Whether episode ended naturally (e.g. goal achieved)
            truncated: Whether episode was cut short (e.g. max steps reached)
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset the agent's internal state between episodes."""
