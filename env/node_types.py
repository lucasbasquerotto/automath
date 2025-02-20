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
    Type,
    ControlFlowBaseNode,
    TypeAliasGroup,
    CountableTypeGroup,
    RunInfoFullResult,
    Integer,
    IntBoolean,
    TypeNode,
    IInstantiable,
)
from env.state import (
    State,
    StateScratchIndex,
    IGoal,
    Goal,
    StateDynamicGoalIndex,
)
from env.action import IBasicAction, IActionOutput, BaseAction
from env.full_state import FullState, SuccessActionData

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
        goal_inner_expr = runnable.run(run_info)
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

class ActionFromBasicArgs(ControlFlowBaseNode, IInstantiable):

    idx_action_type = 1
    idx_arg1 = 2
    idx_arg2 = 3
    idx_arg3 = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return Protocol(
            TypeAliasGroup(),
            CountableTypeGroup(
                Type(IBasicAction.as_type()),
                Integer.as_type(),
                Integer.as_type(),
                Integer.as_type(),
            ),
            IBasicAction.as_type(),
        )

    def _run_control(self, info: RunInfo) -> RunInfoFullResult:
        base_result, arg_group = super()._run(info).as_tuple
        info, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionFromBasicArgs)

        action_type = final_self.inner_arg(
            self.idx_action_type
        ).apply().real(TypeNode[IBasicAction]).type
        assert issubclass(action_type, IBasicAction)

        arg1 = final_self.inner_arg(self.idx_arg1).apply().real(Integer)
        arg2 = final_self.inner_arg(self.idx_arg2).apply().real(Integer)
        arg3 = final_self.inner_arg(self.idx_arg3).apply().real(Integer)

        action = action_type.from_raw(arg1, arg2, arg3)

        return RunInfoFullResult(info.to_result(action), arg_group)

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
        info, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action = final_self.inner_arg(self.idx_action).apply().real(BaseAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        _, action_data = action.run_action_details(full_state)

        info, result = IBoolean.from_bool(
            isinstance(action_data, SuccessActionData)
        ).as_node.run(info).as_tuple

        return RunInfoFullResult(info.to_result(result), arg_group)

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
        info, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action = final_self.inner_arg(self.idx_action).apply().real(BaseAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        output = action.inner_run(full_state).output.apply().cast(IActionOutput)

        return RunInfoFullResult(info.to_result(output), arg_group)

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
        info, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action_output = final_self.inner_arg(self.idx_action_output).apply().real(IActionOutput)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        new_state = action_output.run_output(full_state)

        return RunInfoFullResult(info.to_result(new_state), arg_group)

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
        info, inner_result = base_result.as_tuple

        final_self = inner_result.real(ActionOutputFromAction)

        action = final_self.inner_arg(self.idx_action).apply().real(BaseAction)
        full_state = final_self.inner_arg(self.idx_full_state).apply().real(FullState)

        new_state = action.inner_run(full_state).new_state.apply().real(State)

        return RunInfoFullResult(info.to_result(new_state), arg_group)
