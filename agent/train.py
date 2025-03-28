import time
from typing import Optional
import torch
from agent.smart_agent import SmartAgent
from env.full_state import FullState
from env.goal_env import GoalEnv
from env.node_types import HaveScratch
from env.trainer import Trainer
from env import core
from test_suite import test_root
from utils.env_logger import env_logger

def train_smart_agent(
    action_space_size: int = 10,
    feature_dim: int = 256,
    hidden_dim: int = 128,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.99,
    replay_buffer_capacity: int = 10000,
    batch_size: int = 64,
    target_update_frequency: int = 100,
    episodes_per_state: int = 10,
    max_steps_per_episode: int = 10000,
    save_interval: int = 100,
    model_path: str = "tmp/trained_model.pt",
    device: Optional[torch.device] = None,
    fast: bool = True,
) -> SmartAgent:
    """
    Train the SmartAgent using test cases from the test suite.

    Args:
        action_space_size: Number of possible action indices
        feature_dim: Dimension of feature vector produced by node feature extractor
        hidden_dim: Size of hidden layers
        learning_rate: Learning rate for optimizer
        gamma: Discount factor for future rewards
        epsilon_start: Starting value for epsilon (exploration probability)
        epsilon_end: Minimum value for epsilon
        epsilon_decay: Decay rate for epsilon
        replay_buffer_capacity: Capacity of experience replay buffer
        batch_size: Batch size for training
        target_update_frequency: How often to update target network
        episodes_per_state: Number of episodes to train on each state
        max_steps_per_episode: Maximum steps per episode
        save_interval: How often to save the model (in states)
        model_path: Path to save the trained model
        device: Device to use for tensor operations
        fast: Whether to use fast mode for testing (fewer test cases)

    Returns:
        Trained SmartAgent
    """
    # Create the SmartAgent
    agent = SmartAgent(
        action_space_size=action_space_size,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        replay_buffer_capacity=replay_buffer_capacity,
        batch_size=batch_size,
        target_update_frequency=target_update_frequency,
        device=device
    )

    # Function to run training on a list of full_states
    def run_agent_training(full_states: list[FullState]) -> tuple[int, int]:
        total_trained = 0
        total_skipped = 0
        amount = 0

        for i, full_state_case in enumerate(full_states):
            env_logger.info(f"Training on case {i+1}/{len(full_states)}")

            # Get the history of states from this case
            history_amount = full_state_case.history_amount()

            # For each state in history, train the agent
            for history_idx in range(history_amount):
                completed = run_agent_case(
                    history_idx=history_idx,
                    history_amount=history_amount,
                    current_fs=full_state_case
                )
                if completed:
                    total_trained += 1
                else:
                    total_skipped += 1

                # Save the model at regular intervals
                amount += 1
                if (amount + 1) % save_interval == 0:
                    datetime = time.strftime("%Y%m%d_%H%M%S")
                    agent.save(model_path)
                    env_logger.info(f"{datetime} [{amount}] Model saved to {model_path}")

        return total_trained, total_skipped

    # Function to run training on a list of full_states
    def run_agent_case(
        history_idx: int,
        history_amount: int,
        current_fs: FullState,
    ) -> bool:
        # Create a goal environment with the current state
        env = GoalEnv(
            goal=HaveScratch.with_goal(core.Void()),
            fn_initial_state=lambda _: current_fs
        )

        # Create a trainer and run training
        trainer = Trainer(
            env=env,
            agent=agent,
            episodes=episodes_per_state,
            max_steps_per_episode=max_steps_per_episode,
            inception_level=0,
            max_inception_level=0,  # No inception training for now
        )

        try:
            env_logger.info(f"  Training on history state {history_idx+1}/{history_amount}")
            results = trainer.train()

            # Print training statistics
            rewards = [r for r, _, _ in results]
            steps = [s for _, s, _ in results]
            successes = sum(1 for _, _, success in results if success)

            env_logger.info(
                f"    Avg reward: {sum(rewards) / len(rewards):.2f}, "
                f"Avg steps: {sum(steps) / len(steps):.2f}, "
                f"Success rate: {successes / len(results):.2%}")

            return True
        except Exception as e:
            env_logger.error(f"Error training on state {history_idx+1}: {e}")
            return False

    # Additional info function for logging
    def fn_additional_info_agent(values: tuple[int, int]) -> str:
        total_trained, total_skipped = values
        return f"States trained: {total_trained}, States skipped: {total_skipped}"

    # Run training on all test cases
    start_time = time.time()
    env_logger.info("Starting SmartAgent training...")

    # Use the test_root's run_with_agent to get all the test states
    test_root.run_with_agent(
        fast=fast,
        fn_run_agent=run_agent_training,
        fn_additional_info_agent=fn_additional_info_agent,
    )

    duration = time.time() - start_time
    env_logger.info(f"Training completed in {duration:.2f} seconds")

    # Save the final model
    agent.save(model_path)
    env_logger.info(f"Final model saved to {model_path}")

    return agent
