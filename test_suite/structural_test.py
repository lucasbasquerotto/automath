from env.core import FunctionExpr, Param, And, Or, IntBoolean
from env.goal_env import GoalEnv
from env.node_types import HaveScratch


def test_env():
    # def same_data(env: Environment, test_node: INode):
    #     env_data = env.current_state.raw_data()
    #     test_data = np.array([d.to_array() for d in test_data_list])
    #     if not np.array_equal(env_data, test_data):
    #         print('='*80)
    #         print('env_data')
    #         print('-'*80)
    #         print(env_data)
    #         print('='*80)
    #         print('test_data')
    #         print('-'*80)
    #         print(test_data)
    #         print('='*80)
    #     assert np.array_equal(env_data, test_data)

    params = (Param.from_int(1), Param.from_int(2), Param.from_int(3))
    p1, p2, p3 = params
    goal = HaveScratch(
        FunctionExpr.with_child(
            Or(
                And(p1, p2, IntBoolean(1)),
                And(p2, p3),
            ),
        )
    )

    env = GoalEnv(goal)

    print(env.to_symbolic())

def test():
    test_env()
