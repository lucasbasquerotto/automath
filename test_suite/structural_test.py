from utils.types import FunctionInfo, ParamVar
from impl.goal_env import GoalEnv
from impl.node_types import HaveDefinition, AndNode, OrNode, TrueNode

def test_env():
    params = (ParamVar(1), ParamVar(2), ParamVar(3))
    p1, p2, p3 = params
    goal = HaveDefinition(
        FunctionInfo(
            params=params,
            expr=OrNode(
                AndNode(p1, p2, TrueNode()),
                AndNode(p2, p3),
            ),
        ).to_expr()
    )
    print('goal:', goal)

    env = GoalEnv(goal)
    print(env.current_state.raw_data())

def test():
    test_env()
