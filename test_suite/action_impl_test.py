from env import core
from env import action_impl
from env import state
from env import action
from env import meta_env
from env import full_state
from env import node_types as node_types_module
from env.goal_env import GoalEnv

def get_current_state(env: GoalEnv):
    return env.full_state.nested_args(
        (full_state.FullState.idx_current, full_state.HistoryNode.idx_state)
    ).apply().cast(state.State)

def get_last_history_meta(env: GoalEnv):
    history = env.full_state.history.apply().cast(full_state.HistoryGroupNode)
    last = history.as_tuple[-1]
    return last.meta_data.apply().nested_arg(
        core.Optional.idx_value
    ).apply().cast(action.ActionData)

def get_from_int_type_index(node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_args((
        meta_env.MetaInfo.idx_from_int_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def test_goal_output():
    params = (core.Param.from_int(1), core.Param.from_int(2), core.Param.from_int(3))
    p1, p2, p3 = params
    scratch_goal_1 = core.FunctionExpr.with_child(
        core.Or(
            core.And(p1, p2, core.IntBoolean(1)),
            core.And(p2, p3),
        ),
    )
    goal_1 = node_types_module.HaveScratch.with_goal(scratch_goal_1)
    scratch_goal_2 = core.Not(core.Eq(core.IntBoolean.create_true(), core.IntBoolean.create()))
    goal_2 = node_types_module.HaveScratch.with_goal(scratch_goal_2)
    scratch_goal_3 = core.FunctionCall.define(
        state.FunctionId(1),
        core.DefaultGroup(
            core.DefaultGroup(core.Integer(3), core.Integer(4)),
            core.Eq(p1, core.Integer(27)),
        )
    )
    goal_3 = node_types_module.HaveScratch.with_goal(scratch_goal_3)

    def has_goal(env: GoalEnv, goal: meta_env.IGoal):
        selected_goal = env.full_state.nested_args(
            (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
        ).apply()
        return selected_goal == goal

    def fn_before_final_state(
        meta: meta_env.MetaInfo,
        goal: node_types_module.HaveScratch,
        scratch_goal: core.INode | None,
    ) -> full_state.FullState:
        state_meta = state.StateMetaInfo.create_with_goal(
            full_state.HistoryNode.create_goal_achieved_with_goal(goal)
        )
        return full_state.FullState.with_args(
            meta=meta,
            current=full_state.HistoryNode.with_args(
                state.State.from_raw(
                    meta_info=state_meta,
                    scratchs=[
                        scratch_goal
                        if scratch_goal is not None
                        else goal.definition_expr.apply()
                    ],
                )
            )
        )

    def fn_before_final_env(
        goal: node_types_module.HaveScratch,
        scratch_goal: core.INode | None,
    ) -> GoalEnv:
        env = GoalEnv(
            goal=goal,
            fn_initial_state=lambda meta: fn_before_final_state(
                meta=meta,
                goal=goal,
                scratch_goal=scratch_goal,
            ))
        assert has_goal(env=env, goal=goal)
        return env

    def test_goal_output_with_goal(
        goal: node_types_module.HaveScratch,
        scratch_goal: core.INode,
        direct: bool,
        error=False,
    ):
        env = fn_before_final_env(goal=goal, scratch_goal=scratch_goal)
        current_state = get_current_state(env)
        initial_state = current_state

        assert current_state == state.State.from_raw(
            scratchs=[scratch_goal],
        )
        assert env.full_state.goal_achieved() is False

        meta_idx = get_from_int_type_index(state.StateScratchIndex, env)
        raw_action = action_impl.VerifyGoal.from_raw(0, meta_idx, 1)
        full_action = action_impl.VerifyGoal(
            core.Optional(),
            full_state.MetaFromIntTypeIndex(meta_idx),
            core.Integer(1),
        )
        output = action_impl.VerifyGoalOutput(
            core.Optional(),
            state.StateScratchIndex(1),
        )
        act = output if direct else raw_action
        env.step(act)
        current_state = get_current_state(env)
        last_history_meta = get_last_history_meta(env)

        if not error:
            state_meta = state.StateMetaInfo.create_with_goal(state.GoalAchieved.achieved())
            assert current_state.meta_info.apply() == state_meta
            assert current_state == state.State.from_raw(
                meta_info=state_meta,
                scratchs=[scratch_goal],
            )
            assert last_history_meta == action.ActionData.from_args(
                action=core.Optional(output if direct else full_action),
                output=core.Optional(output),
                exception=core.Optional(),
            )
            assert env.full_state.goal_achieved() is True
        else:
            assert current_state == initial_state
            expected_action: core.Optional[core.INode] = core.Optional(
                output if direct else full_action
            )
            actual_action = last_history_meta.action.apply().cast(core.IOptional)
            assert expected_action == actual_action
            actual_output_opt = last_history_meta.output.apply().cast(core.IOptional)
            assert actual_output_opt == core.Optional(output)
            core.Not(
                last_history_meta.exception.apply().cast(core.IOptional).is_empty()
            ).raise_on_not_true()
            assert env.full_state.goal_achieved() is False

    def main():
        test_goal_output_with_goal(goal_1, scratch_goal_1, direct=True)
        test_goal_output_with_goal(goal_2, scratch_goal_2, direct=False)
        test_goal_output_with_goal(goal_3, scratch_goal_3, direct=True)
        test_goal_output_with_goal(goal_1, scratch_goal_2, direct=True, error=True)
        test_goal_output_with_goal(goal_2, scratch_goal_3, direct=False, error=True)

    main()

def action_impl_test():
    test_goal_output()

def test():
    action_impl_test()
