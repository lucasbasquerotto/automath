import typing
import sympy
from sympy.printing.latex import LatexPrinter
from env import core

class SympyLeaf(sympy.Basic):

    def __init__(
        self,
        node_id: sympy.Integer,
        name: sympy.Symbol,
        sub: sympy.Basic,
        sup: sympy.Basic,
        value: sympy.Basic,
    ):
        super().__init__()
        self._args = (node_id, name, sub, sup, value)

    @classmethod
    def with_args(
        cls,
        node_id: sympy.Integer,
        name: sympy.Symbol,
        sub: sympy.Basic = sympy.Symbol(''),
        sup: sympy.Basic = sympy.Symbol(''),
        value: sympy.Basic = sympy.Symbol(''),
    ) -> typing.Self:
        return cls(node_id, name, sub, sup, value)

    def __str__(self) -> str:
        node_id = self.args[0]
        assert isinstance(node_id, sympy.Integer)
        name = self.args[1]
        assert isinstance(name, sympy.Symbol)
        sub = self.args[2]
        assert isinstance(sub, sympy.Basic)
        sup = self.args[3]
        assert isinstance(sup, sympy.Basic)
        value = self.args[4]
        assert isinstance(value, sympy.Basic)
        name_str = name.name + f"<{node_id}>"
        sub_str = f"_{sub}" if sub != sympy.Symbol('') else ''
        sup_str = f"^{sup}" if sup != sympy.Symbol('') else ''
        value_str = f"[{value}]" if value != sympy.Symbol('') else ''
        return name_str + sub_str + sup_str + value_str

    def _latex(self, printer: LatexPrinter) -> str:
        node_id = self.args[0]
        assert isinstance(node_id, sympy.Integer)
        name = self.args[1]
        assert isinstance(name, sympy.Symbol)
        sub = self.args[2]
        assert isinstance(sub, sympy.Basic)
        sup = self.args[3]
        assert isinstance(sup, sympy.Basic)
        value = self.args[4]
        assert isinstance(value, sympy.Basic)
        name_str = r"\text{" + printer.doprint(name) + r"<" + str(node_id) + r">}"
        sub_str = (
            r"_\text{" + printer.doprint(sub) + r"}"
            if sub != sympy.Symbol('')
            else '')
        sup_str = (
            r"^\text{" + printer.doprint(sup) + r"}"
            if sup != sympy.Symbol('')
            else '')
        value_str = (
            r"[\text{" + printer.doprint(value) + r"}]"
            if value != sympy.Symbol('')
            else '')
        return name_str + sub_str + sup_str + value_str

class SympyShared(sympy.Basic):

    def __init__(self, node_id: sympy.Integer, name: sympy.Symbol, *args: sympy.Basic):
        super().__init__()
        self._args = (node_id, name, *args)

    def _data(self) -> tuple[str, str, tuple[sympy.Basic, ...]]:
        node_id = self.args[0]
        assert isinstance(node_id, sympy.Integer)
        name = self.args[1]
        assert isinstance(name, sympy.Symbol)
        args = self.args[2:]
        amount = len(args)
        name_str = name.name + r"<" + str(node_id) + r">"
        node_name = name_str + '{' + str(amount) + '}'
        node_name_latex = r"\text{" + name_str + r"}\{" + str(amount) + r"\}"
        return node_name, node_name_latex, args

    def __str__(self) -> str:
        node_name, _, args = self._data()

        args_str = '\n'.join(str(arg) for arg in args)

        if len(args) == 0:
            return node_name

        if len(args) == 1:
            return f"{node_name}({args_str})"

        separator = '\n    '
        args_str = args_str.replace('\n', separator)

        return f"{node_name}({separator}{args_str}\n)"

class SympyWrapper(SympyShared):

    def _latex(self, printer: LatexPrinter) -> str:
        _, node_name, args = self._data()
        if len(args) == 0:
            return node_name
        newline = r" \\ "
        args_latex = newline.join(
            r"\{" + str(i+1) + r"\}\text{ }" + printer.doprint(arg)
            for i, arg in enumerate(args))
        begin = r"\begin{cases}"
        end = r"\end{cases}"
        return f"{node_name} {begin} {args_latex} {end}"

class SympyFunction(SympyShared):

    def _latex(self, printer: LatexPrinter) -> str:
        _, node_name, args = self._data()

        newline = r" \\ "
        args_latex = newline.join(printer.doprint(arg) for arg in args)

        if len(args) == 0:
            return node_name

        if len(args) == 1:
            return f"{node_name}({args_latex})"

        args_latex = args_latex.replace(newline, newline + r" \quad ")

        return f"{node_name}({newline} \\quad {args_latex} {newline})"

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

    def __str__(self) -> str:
        return str(self.symbol)

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
                return SympyLeaf.with_args(
                    node_id=sympy.Integer(node_id),
                    name=sympy.Symbol(name),
                    value=sympy.Integer(value),
                )
            elif isinstance(value_aux, core.TypeNode):
                type_id = node_types.index(value_aux.type) + 1
                type_name = value_aux.type.__name__
                type_name_str = type_name + r"<" + str(type_id) + r">"
                return SympyLeaf.with_args(
                    node_id=sympy.Integer(node_id),
                    name=sympy.Symbol(name),
                    value=sympy.Symbol(type_name_str),
                )
            else:
                raise ValueError(f'Invalid value type: {type(value_aux)}')

        if isinstance(node, core.Placeholder):
            scope_id = node.parent_scope.apply()
            index = node.index.apply()
            return SympyLeaf.with_args(
                node_id=sympy.Integer(node_id),
                name=sympy.Symbol(name),
                sub=cls._symbolic(index, node_types, raw=raw),
                sup=cls._symbolic(scope_id, node_types, raw=raw),
            )

        raw_args = [arg.as_node for arg in node.args if isinstance(arg, core.INode)]
        assert len(raw_args) == len(node.args)

        args: tuple[sympy.Basic, ...] = tuple([
            cls._symbolic(arg, node_types, raw=raw) for arg in raw_args
        ])

        outer_args = (sympy.Integer(node_id), sympy.Symbol(name), *args)

        if isinstance(node, core.IWrapper):
            return SympyWrapper(*outer_args)

        return SympyFunction(*outer_args)
