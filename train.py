#!/usr/bin/env python
"""
Main training script that trains the SmartAgent using settings from the config directory.
"""
import os
import time

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

    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(agent_settings.MODEL_PATH), exist_ok=True)

    # Train the agent
    try:
        train_smart_agent(
            input_dim=agent_settings.INPUT_DIM,
            feature_dim=agent_settings.FEATURE_DIM,
            hidden_dim=agent_settings.HIDDEN_DIM,
            hidden_amount=agent_settings.HIDDEN_AMOUNT,
            learning_rate=agent_settings.LEARNING_RATE,
            gamma=agent_settings.GAMMA,
            epsilon_start=agent_settings.EPSILON_START,
            epsilon_end=agent_settings.EPSILON_END,
            epsilon_decay=agent_settings.EPSILON_DECAY,
            replay_buffer_capacity=agent_settings.REPLAY_BUFFER_CAPACITY,
            batch_size=agent_settings.BATCH_SIZE,
            target_update_frequency=agent_settings.TARGET_UPDATE_FREQUENCY,
            episodes_per_state=agent_settings.EPISODES_PER_STATE,
            max_steps_per_episode=agent_settings.MAX_STEPS_PER_EPISODE,
            dropout_rate=agent_settings.DROPOUT_RATE,
            force_exploration_interval=agent_settings.FORCE_EXPLORATION_INTERVAL,
            save_interval=agent_settings.SAVE_INTERVAL,
            model_path=agent_settings.MODEL_PATH,
            device=agent_settings.DEVICE,
            seed=agent_settings.SEED,
        )
        env_logger.info(
            f"Training completed successfully in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        env_logger.error(f"Training failed: {e}")
        raise e


if __name__ == "__main__":
    # ActionErrorActionData<59>{4}(
    #     Optional<44>{1}(RawAction<297>{4}(
    #         MetaAllowedBasicActionsTypeIndex<181>[5]
    #         Integer<155>[0]
    #         Integer<155>[8]
    #         Integer<155>[0]
    #     ))
    #     Optional<44>{1}(DefineArgsGroup<280>{3}(
    #         StateArgsGroupIndex<195>[0]
    #         NodeArgIndex<158>[8]
    #         StateScratchIndex<198>[0]
    #     ))
    #     Optional<44>{0}
    #     Optional<44>{1}(BooleanExceptionInfo<39>{1}(IsEmpty<137>{1}(Optional<44>{0})))
    # )
    # from env.core import Optional, IsEmpty, NodeArgIndex, BooleanExceptionInfo, Integer
    # from env.full_state import MetaAllowedBasicActionsTypeIndex, ActionErrorActionData
    # from env.action import RawAction
    # from env.action_impl import DefineArgsGroup
    # from env.state import StateArgsGroupIndex, StateScratchIndex
    # node = ActionErrorActionData(
    #     Optional(RawAction(
    #         MetaAllowedBasicActionsTypeIndex(5),
    #         Integer(0),
    #         Integer(8),
    #         Integer(0)
    #     )),
    #     Optional(DefineArgsGroup(
    #         StateArgsGroupIndex(0),
    #         NodeArgIndex(8),
    #         StateScratchIndex(0)
    #     )),
    #     Optional(),
    #     Optional(BooleanExceptionInfo(IsEmpty(Optional()))),
    # )
    # print('before validation')
    # node.strict_validate()
    # print('after validation')

    # ActionOutputErrorActionData<60>{4}(
    #     Optional<44>{1}(RawAction<297>{4}(
    #         MetaAllowedBasicActionsTypeIndex<181>[20]
    #         Integer<155>[0]
    #         Integer<155>[3]
    #         Integer<155>[20]
    #     ))
    #     Optional<44>{1}(VerifyGoal<295>{3}(
    #         Optional<44>{0}
    #         MetaFromIntTypeIndex<185>[3]
    #         Integer<155>[20]
    #     ))
    #     Optional<44>{1}(VerifyGoalOutput<320>{2}(
    #         Optional<44>{0}
    #         IntBoolean<153>[20]
    #     ))
    #     Optional<44>{1}(BooleanExceptionInfo<39>{1}(IsInstance<129>{2}(
    #         IntBoolean<153>[20]
    #         TypeNode<224>[StateScratchIndex<198>]
    #     )))
    # )
    # from env.core import Optional, IsEmpty, NodeArgIndex, BooleanExceptionInfo, Integer
    # from env.full_state import MetaAllowedBasicActionsTypeIndex, ActionOutputErrorActionData, MetaFromIntTypeIndex
    # from env.action import RawAction
    # from env.action_impl import VerifyGoal, VerifyGoalOutput
    # # from env.state import MetaFromIntTypeIndex
    # node = ActionOutputErrorActionData(
    #     Optional(RawAction(
    #         MetaAllowedBasicActionsTypeIndex(20),
    #         Integer(0),
    #         Integer(3),
    #         Integer(20)
    #     )),
    #     Optional(VerifyGoal(
    #         Optional(),
    #         MetaFromIntTypeIndex(3),
    #         Integer(20)
    #     )),
    #     Optional(VerifyGoalOutput(
    #         Optional(),
    #         IntBoolean(20)
    #     )),
    #     Optional(BooleanExceptionInfo(IsEmpty(Optional()))),
    # )
    # print('before validation')
    # node.strict_validate()
    # print('after validation')

    # from env.core import Optional, IntBoolean
    # from env.action_impl import VerifyGoalOutput
    # node = VerifyGoalOutput(
    #     Optional(),
    #     IntBoolean(20)
    # )
    # node.strict_validate()

    # raise RuntimeError('Test')
    main()
