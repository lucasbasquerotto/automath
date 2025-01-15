import sympy
from sympy.printing.latex import LatexPrinter
from env import core

class SympyShared(sympy.Basic):

    def __init__(self, node_id: sympy.Integer, name: sympy.Symbol, *args: sympy.Basic):
        super().__init__()
        self._args = (node_id, name, *args)

    def _data(self) -> tuple[str, tuple[sympy.Basic, ...]]:
        node_id = self.args[0]
        assert isinstance(node_id, sympy.Integer)
        name = self.args[1]
        assert isinstance(name, sympy.Symbol)
        args = self.args[2:]
        amount = len(args)
        name_str = r"\text{" + name.name + r"<" + str(node_id) + r">}"
        amount_str = r"\{" + str(amount) + r"\}"
        node_name = f"{name_str}{amount_str}"
        return node_name, args

class SympyWrapper(SympyShared):

    def _latex(self, printer: LatexPrinter) -> str:
        node_name, args = self._data()

        if len(args) == 0:
            return node_name
        args_latex = r" \\ ".join(
            r"\{" + str(i+1) + r"\}\text{ }" + printer.doprint(arg)
            for i, arg in enumerate(args))
        begin = r"\begin{cases}"
        end = r"\end{cases}"

        return f"{node_name} {begin} {args_latex} {end}"

class SympyFunction(SympyShared):

    def _latex(self, printer: LatexPrinter) -> str:
        node_name, args = self._data()

        newline = r" \\ \text{} "
        args_latex = newline.join(printer.doprint(arg) for arg in args)
        if len(args) == 0:
            return node_name
        if len(args) <= 1:
            return f"{node_name}({args_latex})"
        args_latex = args_latex.replace(newline, newline + r" \quad ")

        return f"{node_name}( {newline} \\quad {args_latex} {newline} )"

class Symbol:

    def __init__(
        self,
        node: core.BaseNode,
        node_types: tuple[type[core.INode], ...],
    ):
        self._node = node
        self._node_types = node_types
        self._symbol = self._symbolic(self._node, self._node_types)

    @property
    def symbol(self) -> sympy.Basic:
        return self._symbol

    def latex(self) -> str:
        return str(sympy.latex(self.symbol))

    def to_str(self) -> str:
        return self.latex().replace(
            r"\\", "\n"
        ).replace(
            r"\quad", "    "
        ).replace(
            r"\text{}", ""
        ).replace(
            r"\text{ }", " "
        )

    @classmethod
    def _symbolic(
        cls,
        node: core.BaseNode,
        node_types: tuple[type[core.INode], ...],
        raw=False,
    ) -> sympy.Basic:
        assert isinstance(node, core.BaseNode)
        node_id = node_types.index(node.func) + 1
        name = node.func.__name__

        if isinstance(node, core.ISpecialValue):
            value_aux = node.node_value

            if isinstance(value_aux, core.IInt):
                value = value_aux.as_int
                name_str = r"\text{" + name + r"<" + str(node_id) + r">}"
                return sympy.Symbol(f'{name_str}[{value}]')
            elif isinstance(value_aux, core.TypeNode):
                name_str = r"\text{" + name + r"<" + str(node_id) + r">}"
                type_id = node_types.index(value_aux.type) + 1
                type_name = value_aux.type.__name__
                type_name_str = r"\text{" + type_name + r"<" + str(type_id) + r">}"
                return sympy.Symbol(f'{name_str}[{type_name_str}]')
            else:
                raise ValueError(f'Invalid value type: {type(value_aux)}')

        if isinstance(node, core.Placeholder):
            name_str = r"\text{" + name + r"<" + str(node_id) + r">}"
            scope_id = node.parent_scope.apply()
            scope_id_str = r"{" + sympy.latex(cls._symbolic(scope_id, node_types, raw=raw)) + r"}"
            index = node.index.apply()
            index_str = r"{" + sympy.latex(cls._symbolic(index, node_types, raw=raw)) + r"}"
            type_node = node.type_node.apply()
            type_node_str = (
                r"[" + sympy.latex(cls._symbolic(type_node, node_types, raw=raw)) + r"]"
                if not isinstance(type_node, core.UnknownType)
                else ''
            )
            return sympy.Symbol(f'{name_str}_{index_str}^{scope_id_str}{type_node_str}')

        raw_args = [arg.as_node for arg in node.args if isinstance(arg, core.INode)]
        assert len(raw_args) == len(node.args)

        args: tuple[sympy.Basic, ...] = tuple([
            cls._symbolic(arg, node_types, raw=raw) for arg in raw_args
        ])

        outer_args = (sympy.Integer(node_id), sympy.Symbol(name), *args)

        if isinstance(node, core.IWrapper):
            return SympyWrapper(*outer_args)

        return SympyFunction(*outer_args)
