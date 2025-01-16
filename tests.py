from test_suite import test_root

def test():
    final_states = test_root.test()
    amount = len(final_states)
    print(f"Amount of completed tests: {amount}")
