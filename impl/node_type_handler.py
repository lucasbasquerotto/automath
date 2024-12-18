import sympy
from utils.types import IndexedElem
from environment.meta_env import NodeTypeHandler, BaseNode

class DefaultNodeTypeHandler(NodeTypeHandler):
    def get_value(self, node: BaseNode) -> int:
        if isinstance(node, IndexedElem):
            return node.index
        if isinstance(node, sympy.Integer):
            return int(node)
        return 0
