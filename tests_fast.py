import time
from env.core import BaseNode
from test_suite import test_root

def test():
    start_time = time.time()
    BaseNode.fast = True
    final_states = test_root.test(fast=True)
    amount = len(final_states)
    action_amount = sum([fs.history_amount() for fs in final_states])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Amount of completed tests: {amount} ({action_amount} actions)")
    print(f"Time taken to run all tests (fast): {elapsed_time:.2f} seconds")
