import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class ChainIndexingRule(BaseRule):
    """
    Detects chained indexing in Pandas DataFrames that leads to inefficient energy use.
    """

    id = "chain_indexing"
    name = "Chain Indexing"
    description = (
        "Chained indexing (e.g., df['one']['two']) triggers multiple Pandas operations, "
        "increasing memory and CPU usage."
    )
    optimization = "Use df.loc[:, ('one', 'two')] or a single indexing call for efficiency."

    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )

    def should_apply(self, node: ast.AST) -> bool:
        """
        Run this rule on the entire module node so the visitor can see:
          - Assignments of 'df' to pd.DataFrame(...)
          - Subscript nodes for chained indexing
        """
        return isinstance(node, ast.Module)

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Creates a ChainIndexingVisitor to walk the entire module's AST,
        identifying any chain indexing in recognized Pandas DataFrames.
        """
        visitor = self.ChainIndexingVisitor()
        visitor.visit(node)
        return visitor.smells

    class ChainIndexingVisitor(ast.NodeVisitor):
        """
        AST Visitor that:
          - Identifies variables that are likely Pandas DataFrames.
          - Flags chained indexing (df['A']['B']) on those DataFrame variables.
        """

        def __init__(self):
            super().__init__()
            self.df_candidates = set()  # Variable names recognized as DataFrames
            self.smells = []

        def visit_Assign(self, node: ast.Assign):
            """
            Attempts to identify DataFrame variables by checking if the right-hand side is:
              - A call to pd.DataFrame(...)
              - A call to pd.read_* (e.g. pd.read_csv, pd.read_excel, etc.)
            """
            if isinstance(node.value, ast.Call):
                func_chain = self._get_attr_chain(node.value.func)

                if len(func_chain) >= 2 and func_chain[0] == "pd":
                    if func_chain[1] == "DataFrame" or func_chain[1].startswith("read_"):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.df_candidates.add(target.id)

            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript):
            """
            Detects chain indexing by checking if node.value is also a Subscript.
            """
            if isinstance(node.value, ast.Subscript):
                # Walk up to find the ultimate base name
                base_name = self._get_subscript_root_name(node.value)
                if base_name in self.df_candidates:
                    # Flag a smell for chained indexing
                    self.smells.append(
                        Smell(
                            rule_id=ChainIndexingRule.id,
                            rule_name=ChainIndexingRule.name,
                            description=ChainIndexingRule.description,
                            optimization=ChainIndexingRule.optimization,
                            start_line=node.lineno
                        )
                    )
            self.generic_visit(node)
            
        def _get_attr_chain(self, node: ast.AST) -> list[str]:
            """
            Builds a list representing the fully qualified name of an attribute chain.
            """
            chain = []
            current = node
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                chain.append(current.id)
            chain.reverse()
            return chain

        def _get_subscript_root_name(self, node: ast.Subscript) -> str:
            """
            Walks up a chain of Subscripts to find the ultimate Name node.
            Returns '' if it doesn't find a Name.
            """
            current = node
            while isinstance(current, ast.Subscript):
                current = current.value
            if isinstance(current, ast.Name):
                return current.id
            return ""
