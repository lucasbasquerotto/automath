import numpy as np
from environment.core import FunctionExpr, FunctionParams, Param
from environment.full_state_old import (
    UNDEFINED_OR_EMPTY_FIELD,
    HISTORY_TYPE_META,
    HISTORY_TYPE_STATE,
    HISTORY_TYPE_ACTION,
    CONTEXT_HISTORY_TYPE,
    CONTEXT_META_MAIN,
    CONTEXT_STATE_DEFINITION,
    CONTEXT_STATE_PARTIAL_DEFINITION,
    CONTEXT_STATE_ARG_GROUP,
    CONTEXT_ACTION_TYPE,
    CONTEXT_ACTION_INPUT,
    CONTEXT_ACTION_OUTPUT,
    CONTEXT_ACTION_STATUS,
    CONTEXT_META_TRANSITION,
    SUBCONTEXT_ACTION_INPUT_AMOUNT,
    SUBCONTEXT_ACTION_INPUT_ARG,
    SUBCONTEXT_ACTION_OUTPUT_PARTIAL_DEFINITION_IDX,
    SUBCONTEXT_ACTION_OUTPUT_DEFINITION_IDX,
    SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_IDX,
    SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_PARAMS_AMOUNT,
    SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP_ARGS_AMOUNT,
    SUBCONTEXT_ACTION_OUTPUT_ARG_GROUP,
    SUBCONTEXT_ACTION_OUTPUT_EXPR_ID,
    SUBCONTEXT_ACTION_OUTPUT_NODE_IDX,
    SUBCONTEXT_ACTION_OUTPUT_NODE_EXPR,
    SUBCONTEXT_META_ACTION_TYPE,
    GROUP_CONTEXT_ARG_GROUP,
    GROUP_CONTEXT_ACTION_TYPE,
    GROUP_SUBCONTEXT_ARG_GROUP_MAIN,
    GROUP_SUBCONTEXT_ARG_GROUP_EXPR,
    GROUP_SUBCONTEXT_ACTION_TYPE_MAIN,
    GROUP_SUBCONTEXT_ACTION_TYPE_ARG,
    ITEM_CONTEXT_SYMBOL_IDX,
    ITEM_CONTEXT_TYPE_IDX,
    ITEM_CONTEXT_PARAM_IDX,
    ITEM_CONTEXT_EXPR_IDX,
    ACTION_STATUS_SUCCESS_ID,
    ACTION_STATUS_FAIL_ID,
    NodeItemData)
from environment.action_old import BASIC_ACTION_TYPES
from impl.goal_env import GoalEnv
from impl.node_types import HaveDefinition, AndNode, OrNode, TrueNode


def test_env():
    def same_data(env: GoalEnv, test_data_list: list[NodeItemData]):
        env_data = env.current_state.raw_data()
        test_data = np.array([d.to_array() for d in test_data_list])
        if not np.array_equal(env_data, test_data):
            print('='*80)
            print('env_data')
            print('-'*80)
            print(env_data)
            print('='*80)
            print('test_data')
            print('-'*80)
            print(test_data)
            print('='*80)
        assert np.array_equal(env_data, test_data)

    params = (Param(1), Param(2), Param(3))
    p1, p2, p3 = params
    goal = HaveDefinition(
        FunctionExpr(
            OrNode(
                AndNode(p1, p2, TrueNode()),
                AndNode(p2, p3),
            ),
            FunctionParams(*params),
        )
    )

    env = GoalEnv(goal)

    history_0_meta = [
        NodeItemData.with_defaults(
            history_number=0,
            history_type=HISTORY_TYPE_META,
            context=CONTEXT_HISTORY_TYPE,
            node_value=HISTORY_TYPE_META,
        )
    ]

    for i, action_type in enumerate(BASIC_ACTION_TYPES):
        arg_types = action_type.metadata().arg_types

        history_0_meta.append(
            NodeItemData.with_defaults(
                history_number=0,
                history_type=HISTORY_TYPE_META,
                context=CONTEXT_META_MAIN,
                subcontext=SUBCONTEXT_META_ACTION_TYPE,
                group_idx=i+1,
                group_context=GROUP_CONTEXT_ACTION_TYPE,
                group_subcontext=GROUP_SUBCONTEXT_ACTION_TYPE_MAIN,
                node_value=len(arg_types),
            )
        )

        for j, arg_type in enumerate(arg_types):
            history_0_meta.append(
                NodeItemData.with_defaults(
                    history_number=0,
                    history_type=HISTORY_TYPE_META,
                    context=CONTEXT_META_MAIN,
                    subcontext=SUBCONTEXT_META_ACTION_TYPE,
                    group_idx=i+1,
                    group_context=GROUP_CONTEXT_ACTION_TYPE,
                    group_subcontext=GROUP_SUBCONTEXT_ACTION_TYPE_ARG,
                    item_idx=j+1,
                    node_value=arg_type,
                )
            )


    history_1_meta = [
        NodeItemData.with_defaults(
            history_number=1,
            history_type=HISTORY_TYPE_STATE,
            context=CONTEXT_HISTORY_TYPE,
            node_value=HISTORY_TYPE_STATE,
        )
    ]

    same_data(env=env, test_data_list=history_0_meta + history_1_meta)

def test():
    test_env()
