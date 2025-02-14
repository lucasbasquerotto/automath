import tests
from env.core import BaseNode

def test():
    BaseNode.cache_enabled = False
    tests.test()
