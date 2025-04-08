import typing
import time
from env.core import Optional, IExceptionInfo, IRunnable, IsInstance
from env.goal_env import GoalEnv
from env.base_agent import BaseAgent
from env.full_state import BaseActionData
from env.action import BaseAction, IActionOutput, RawAction
from env import node_types

class Trainer:

    def __init__(
        self,
        env: GoalEnv,
        agent: BaseAgent,
        max_steps_per_episode: int | None = None,
        inception_level: int = 0,
        max_inception_level: int = 0,
        static_actions: typing.Sequence[RawAction] | None = None,
    ):
        self.env = env
        self.agent = agent
        self.max_steps_per_episode = max_steps_per_episode
        self.inception_level = inception_level
        self.max_inception_level = max_inception_level
        self.static_actions = static_actions

    def train(self) -> tuple[float, int, bool]:
        """Run a single training episode.

        Returns:
            Tuple of (total_reward, steps_taken, goal_achieved)
        """
        state = self.env.reset()
        self.agent.reset()

        total_reward = 0.0
        steps = 0
        static_actions = self.static_actions
        terminated = False

        while True:
            if static_actions is not None:
                if len(static_actions) <= steps:
                    break

            start = time.time()
            start_select = start

            # Select and execute action
            raw_action = (
                static_actions[steps]
                if static_actions is not None
                else self.agent.select_action(state)
            )

            end_select = time.time()
            start_step = end_select

            next_state, reward, terminated, truncated = self.env.step(raw_action)
            done = terminated or truncated

            end_step = time.time()
            start_sub_train = end_step

            self.sub_train(raw_action)

            end_sub_train = time.time()
            start_train = end_sub_train

            # Train agent
            self.agent.train(
                state=state,
                action=raw_action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                truncated=truncated)

            end_train = time.time()
            end = end_train

            static = static_actions is not None
            print(
                time.strftime('%H:%M:%S'),
                f"[Trainer{' (static)' if static else ''}]",
                f"Total: {end - start:.2f}s,",
                f"Select: {end_select - start_select:.2f}s,",
                f"Step: {end_step - start_step:.2f}s,",
                f"Sub-train: {end_sub_train - start_sub_train:.2f}s,",
                f"Train: {end_train - start_train:.2f}s",
            )

            total_reward += reward
            steps += 1

            if done:
                break

            if (
                self.max_steps_per_episode is not None
                and steps >= self.max_steps_per_episode
            ):
                break

            state = next_state

        return total_reward, steps, terminated

    def sub_train(self, raw_action: RawAction):
        inception_level = self.inception_level
        max_inception_level = self.max_inception_level

        if inception_level < max_inception_level:
            full_state = self.env.full_state

            _, action_data, __ = raw_action.run_action_details(full_state)
            IsInstance.assert_type(action_data, BaseActionData)
            assert isinstance(action_data, BaseActionData)

            action_opt = action_data.action.apply().real(Optional[BaseAction])
            output_opt = action_data.output.apply().real(Optional[IActionOutput])
            exception_opt = action_data.exception.apply().real(Optional[IExceptionInfo])

            action = action_opt.value
            output = output_opt.value
            exception = exception_opt.value

            self.sub_train_with_goal(node_types.CorrectActionValidator(
                raw_action,
                full_state,
            ))

            if action is not None:
                self.sub_train_with_goal(node_types.ActionFromRawAction(
                    raw_action,
                    full_state,
                ))

            if (action is not None) and (output is not None):
                self.sub_train_with_goal(node_types.ActionOutputFromAction(
                    action,
                    full_state,
                ))

                self.sub_train_with_goal(node_types.ActionOutputFromRawAction(
                    raw_action,
                    full_state,
                ))

            if (action is not None) and (output is not None) and (exception is None):
                self.sub_train_with_goal(node_types.NewStateFromActionOutput(
                    output,
                    full_state,
                ))

                self.sub_train_with_goal(node_types.NewStateFromAction(
                    action,
                    full_state,
                ))

                self.sub_train_with_goal(node_types.NewStateFromRawAction(
                    raw_action,
                    full_state,
                ))

    def sub_train_with_goal(self, goal_expr: IRunnable):
        sub_env = GoalEnv(
            goal=node_types.HaveResultScratch.with_goal(goal_expr),
            reward_evaluator=self.env.reward_evaluator,
            max_history_state_size=self.env.max_history_state_size,
            max_steps=self.env.max_steps,
            allowed_actions=node_types.ESSENTIAL_ACTIONS,
        )
        sub_trainer = Trainer(
            env=sub_env,
            agent=self.agent,
            max_steps_per_episode=self.max_steps_per_episode,
            inception_level=self.inception_level + 1,
            max_inception_level=self.max_inception_level,
        )
        sub_trainer.train()
