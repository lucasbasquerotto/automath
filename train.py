#!/usr/bin/env python
"""
Main training script that trains the SmartAgent using settings from the config directory.
"""
import os
import time
from typing import Dict, Any

from agent.train import train_smart_agent
from config import agent_settings
from utils.env_logger import env_logger


def main() -> None:
    """
    Main function to run the agent training process.

    Loads settings from config/agent_settings.py and passes them to the
    train_smart_agent function in agent/train.py.
    """
    start_time = time.time()
    env_logger.info("Starting training using settings from config/agent_settings.py")

    # Create parameters dictionary from agent_settings
    params: Dict[str, Any] = {
        "action_space_size": agent_settings.ACTION_SPACE_SIZE,
        "feature_dim": agent_settings.FEATURE_DIM,
        "hidden_dim": agent_settings.HIDDEN_DIM,
        "learning_rate": agent_settings.LEARNING_RATE,
        "gamma": agent_settings.GAMMA,
        "epsilon_start": agent_settings.EPSILON_START,
        "epsilon_end": agent_settings.EPSILON_END,
        "epsilon_decay": agent_settings.EPSILON_DECAY,
        "replay_buffer_capacity": agent_settings.REPLAY_BUFFER_CAPACITY,
        "batch_size": agent_settings.BATCH_SIZE,
        "target_update_frequency": agent_settings.TARGET_UPDATE_FREQUENCY,
        "episodes_per_state": agent_settings.EPISODES_PER_STATE,
        "max_steps_per_episode": agent_settings.MAX_STEPS_PER_EPISODE,
        "save_interval": agent_settings.SAVE_INTERVAL,
        "model_path": agent_settings.MODEL_PATH,
        "device": agent_settings.DEVICE,
        "fast": agent_settings.FAST,
    }

    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(params["model_path"]), exist_ok=True)

    # Train the agent
    try:
        train_smart_agent(**params)
        env_logger.info(
            f"Training completed successfully in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        env_logger.error(f"Training failed: {e}")
        raise e


if __name__ == "__main__":
    main()
