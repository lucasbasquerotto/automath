from typing import List, Tuple
from env.core import Optional, IExceptionInfo, IRunnable
from env.goal_env import GoalEnv
from env.base_agent import BaseAgent
from env.full_state import BaseActionData
from env.action import BaseAction, IActionOutput, RawAction
from env import node_types, action_impl

class Trainer:
    def __init__(
        self,
        env: GoalEnv,
        agent: BaseAgent,
        episodes: int = 1000,
        max_steps_per_episode: int | None = None,
        inception_level: int = 0,
        max_inception_level: int = 0,
    ):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.inception_level = inception_level
        self.max_inception_level = max_inception_level

    def train_episode(self) -> Tuple[float, int, bool]:
        """Run a single training episode.

        Returns:
            Tuple of (total_reward, steps_taken, goal_achieved)
        """
        state = self.env.reset()
        self.agent.reset()

        total_reward = 0.0
        steps = 0

        while True:
            # Select and execute action
            raw_action = self.agent.select_action(state)
            next_state, reward, terminated, truncated = self.env.step(raw_action)
            done = terminated or truncated

            self.sub_train(raw_action)

            # Train agent
            self.agent.train(
                state=state,
                action=raw_action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                truncated=truncated)

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

    def train(self) -> List[Tuple[float, int, bool]]:
        """Run the full training process for max_episodes.

        Returns:
            List of (episode_reward, episode_steps, goal_achieved) tuples
        """
        results = []

        for _ in range(self.episodes):
            episode_result = self.train_episode()
            results.append(episode_result)

        return results

    def sub_train(self, raw_action: RawAction):
        inception_level = self.inception_level
        max_inception_level = self.max_inception_level

        if inception_level < max_inception_level:
            full_state = self.env.full_state

            _, action_data = raw_action.run_action_details(full_state)
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
            allowed_actions=(
                action_impl.RestoreHistoryStateOutput,
                action_impl.VerifyGoal,
                action_impl.CreateDynamicGoal,
                action_impl.VerifyDynamicGoal,
                action_impl.DeleteDynamicGoalOutput,
                action_impl.ResetStateHiddenInfo,
                action_impl.DefineStateHiddenInfo,
                action_impl.CreateScratch,
                action_impl.DeleteScratchOutput,
                action_impl.ClearScratch,
                action_impl.DefineScratchFromDefault,
                action_impl.DefineScratchFromInt,
                action_impl.DefineScratchFromSingleArg,
                action_impl.DefineScratchFromIntIndex,
                action_impl.DefineScratchFromFunctionWithIntArg,
                action_impl.DefineScratchFromFunctionWithSingleArg,
                action_impl.DefineScratchFromFunctionWithArgs,
                action_impl.DefineScratchFromScratchNode,
                action_impl.UpdateScratchFromAnother,
                action_impl.CreateArgsGroup,
                action_impl.DeleteArgsGroupOutput,
                action_impl.DefineArgsGroup,
            ),
        )
        sub_trainer = Trainer(
            env=sub_env,
            agent=self.agent,
            episodes=self.episodes,
            max_steps_per_episode=self.max_steps_per_episode,
            inception_level=self.inception_level + 1,
            max_inception_level=self.max_inception_level,
        )
        sub_trainer.train_episode()
