from env.core import (
    INode,
    IRunnable,
    RunInfo,
    ScopeDataGroup,
    Optional,
    Eq,
    Void,
    IBoolean,
    IDefault,
    Protocol,
    ControlFlowBaseNode,
    TypeAliasGroup,
    CountableTypeGroup,
    RunInfoFullResult,
    IntBoolean,
    Integer,
    RunInfoStats,
    IInstantiable,
)
from env.state import (
    State,
    StateScratchIndex,
    IGoal,
    Goal,
    StateDynamicGoalIndex,
)
from env.action import IBasicAction, IActionOutput, BaseAction, RawAction
from env.full_state import FullState, SuccessActionData
from env import action_impl

ESSENTIAL_ACTIONS = (
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
)

class HaveScratch(Goal[INode, StateScratchIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return INode

    @classmethod
    def eval_param_type(cls):
        return StateScratchIndex

    def evaluate(self, state: State, eval_param: StateScratchIndex):
        goal_inner_expr = self.goal_inner_expr.apply()
        assert isinstance(eval_param, StateScratchIndex)
        scratch = eval_param.find_in_node(state).value_or_raise
        content = scratch.value_or_raise
        return Eq(content, goal_inner_expr)

class HaveResultScratch(Goal[IRunnable, StateScratchIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return IRunnable

    @classmethod
    def eval_param_type(cls):
        return StateScratchIndex

    def evaluate(self, state: State, eval_param: StateScratchIndex):
        runnable = self.goal_inner_expr.apply().real(IRunnable)
        run_info = RunInfo.with_args(
            scope_data_group=ScopeDataGroup(),
            return_after_scope=Optional(),
        )
        eval_result = runnable.run(run_info.with_stats())
        _, goal_inner_expr = eval_result.as_tuple
        assert isinstance(eval_param, StateScratchIndex)
        scratch = eval_param.find_in_node(state).value_or_raise
        content = scratch.value_or_raise
        return Eq(content, goal_inner_expr)

class HaveDynamicGoal(Goal[IGoal, StateDynamicGoalIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return IGoal

    @classmethod
    def eval_param_type(cls):
        return StateDynamicGoalIndex

    def evaluate(self, state: State, eval_param: StateDynamicGoalIndex):
        goal_inner_expr = self.goal_inner_expr.apply()
        assert isinstance(eval_param, StateDynamicGoalIndex)
        dynamic_goal = eval_param.find_in_node(state).value_or_raise
        content = dynamic_goal.goal_expr.apply()
        return Eq(content, goal_inner_expr)

class HaveDynamicGoalAchieved(Goal[Void, StateDynamicGoalIndex], IDefault, IInstantiable):

    @classmethod
    def goal_type(cls):
        return Void

    @classmethod
    def eval_param_type(cls):
        return StateDynamicGoalIndex

    @classmethod
    def create(cls):
        return cls.with_goal(Void())

    def evaluate(self, state: State, eval_param: StateDynamicGoalIndex):
        assert isinstance(eval_param, StateDynamicGoalIndex)
        dynamic_goal = eval_param.find_in_node(state).value_or_raise
        content = dynamic_goal.goal_achieved.apply().cast(IBoolean)
        return content

class CorrectActionValidator(ControlFlowBaseNode, IInstantiable):

    idx_action = 1
    idx_full_state = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                BaseAction.as_type(),
                FullState.as_type(),
            ),
            IntBoolean.as_type(),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info_with_stats, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action = final_self.inner_arg(self.idx_action).apply().real(BaseAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        _, action_data = action.run_action_details(full_state)

        info_with_stats, result = IBoolean.from_bool(
            isinstance(action_data, SuccessActionData)
        ).as_node.run(info_with_stats).as_tuple

        return RunInfoFullResult(info_with_stats.to_result(result), arg_group)

class ActionFromRawAction(ControlFlowBaseNode, IInstantiable):

    idx_raw_action = 1
    idx_full_state = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                RawAction.as_type(),
                FullState.as_type(),
            ),
            IBasicAction.as_type(),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info_with_stats, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionFromRawAction)

        raw_action = final_self.inner_arg(self.idx_raw_action).apply().real(RawAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        action = raw_action.to_action(full_state)

        return RunInfoFullResult(info_with_stats.to_result(action), arg_group)

class ActionOutputFromAction(ControlFlowBaseNode, IInstantiable):

    idx_action = 1
    idx_full_state = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                BaseAction.as_type(),
                FullState.as_type(),
            ),
            IActionOutput.as_type(),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info_with_stats, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action = final_self.inner_arg(self.idx_action).apply().real(BaseAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        full_out, action_info = action.inner_run(full_state)
        output = full_out.output.apply().cast(IActionOutput)
        info_with_stats = info_with_stats.add_inner_stats(action_info.to_stats())

        return RunInfoFullResult(info_with_stats.to_result(output), arg_group)

class ActionOutputFromRawAction(ControlFlowBaseNode, IInstantiable):

    idx_raw_action = 1
    idx_full_state = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                RawAction.as_type(),
                FullState.as_type(),
            ),
            IActionOutput.as_type(),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info_with_stats, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action = final_self.inner_arg(self.idx_raw_action).apply().real(RawAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        full_out, action_info = action.inner_run(full_state)
        output = full_out.output.apply().cast(IActionOutput)
        info_with_stats = info_with_stats.add_inner_stats(action_info.to_stats())

        return RunInfoFullResult(info_with_stats.to_result(output), arg_group)

class NewStateFromActionOutput(ControlFlowBaseNode, IInstantiable):

    idx_action_output = 1
    idx_full_state = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                IActionOutput.as_type(),
                FullState.as_type(),
            ),
            State.as_type(),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info_with_stats, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action_output = final_self.inner_arg(self.idx_action_output).apply().real(IActionOutput)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        new_state, output_info = action_output.run_output(full_state)
        info_with_stats = info_with_stats.add_inner_stats(output_info.to_stats())

        return RunInfoFullResult(info_with_stats.to_result(new_state), arg_group)

class NewStateFromAction(ControlFlowBaseNode, IInstantiable):

    idx_action = 1
    idx_full_state = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                BaseAction.as_type(),
                FullState.as_type(),
            ),
            State.as_type(),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info_with_stats, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action = final_self.inner_arg(self.idx_action).apply().real(BaseAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        full_out, action_info = action.inner_run(full_state)
        new_state = full_out.new_state.apply().real(State)
        info_with_stats = info_with_stats.add_inner_stats(action_info.to_stats())

        return RunInfoFullResult(info_with_stats.to_result(new_state), arg_group)

class NewStateFromRawAction(ControlFlowBaseNode, IInstantiable):

    idx_raw_action = 1
    idx_full_state = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                RawAction.as_type(),
                FullState.as_type(),
            ),
            State.as_type(),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info_with_stats, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action = final_self.inner_arg(self.idx_raw_action).apply().real(RawAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        full_out, action_info = action.inner_run(full_state)
        new_state = full_out.new_state.apply().real(State)
        info_with_stats = info_with_stats.add_inner_stats(action_info.to_stats())

        return RunInfoFullResult(info_with_stats.to_result(new_state), arg_group)
