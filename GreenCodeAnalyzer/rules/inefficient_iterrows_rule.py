import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class InefficientIterationWithIterrows(BaseRule):
    """
    Detects inefficient use of Pandas iterrows for row-by-row data manipulation.
    """
    
    id = "inefficient_iterrows"
    name = "InefficientIterationWithIterrows"
    description = "Using iterrows for row-by-row Pandas operations is slow and energy-intensive due to Python overhead."
    optimization = "Replace with vectorized Pandas operations (e.g., apply, vector arithmetic, or groupby)."

    def __init__(self):
        super().__init__(id=self.id,
                        name=self.name,
                        description=self.description,
                        optimization=self.optimization)
    
    def should_apply(self, node: ast.AST) -> bool:
        """
        Applies to For loops that potentially use iterrows.
        """
        return isinstance(node, ast.For)

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Checks if the For loop uses Pandas iterrows for inefficient row-by-row data manipulation.
        """
        smells = []
        
        # Check if the loop's iterator is a method call to iterrows
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
            attr = node.iter.func
            if attr.attr == "iterrows" and isinstance(attr.value, (ast.Name, ast.Attribute)):
                # Analyze the loop body for inefficient manipulation patterns
                for stmt in node.body:
                    # Look for assignments, arithmetic, or accumulator patterns involving row variables
                    if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                        # Check if the statement uses the row variable
                        if len(node.target.elts) >= 2:  # e.g., for index, row in ...
                            row_var = node.target.elts[1]  # Typically 'row'
                            if self._uses_variable(stmt, row_var.id):
                                smells.append(Smell(
                                    rule_id=self.id,
                                    rule_name=self.name,
                                    description=self.description,
                                    penalty=self.penalty,
                                    optimization=self.optimization,
                                    start_line=node.lineno
                                ))
                                break
                    # Check for function calls that might manipulate row data
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        if len(node.target.elts) >= 2:
                            row_var = node.target.elts[1]
                            if self._uses_variable(stmt.value, row_var.id):
                                smells.append(Smell(
                                    rule_id=self.id,
                                    rule_name=self.name,
                                    description=self.description,
                                    penalty=self.penalty,
                                    optimization=self.optimization,
                                    start_line=node.lineno
                                ))
                                break
        
        return smells

    def _uses_variable(self, node: ast.AST, var_name: str) -> bool:
        """
        Helper method to check if a given AST node uses a specific variable name.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == var_name:
                return True
        return False