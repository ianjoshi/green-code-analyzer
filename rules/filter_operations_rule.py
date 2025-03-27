import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class FilterOperationsRule(BaseRule):
    """
    Detects inefficient filtering operations that use loops instead of vectorized operations.
    """
    
    id = "filter_operations"
    name = "Inefficient Filter Operations"
    description = "Using loops for filtering elements instead of vectorized operations causes unnecessary iterations and is energy-intensive."
    optimization = "Replace with vectorized operations (e.g., np.where or boolean indexing tensor[tensor > 0.5], df[df['values'] > 0.5])."   

    def __init__(self):
        super().__init__(id=self.id,
                        name=self.name,
                        description=self.description,
                        optimization=self.optimization)
    
    def should_apply(self, node: ast.AST) -> bool:
        """
        Applies to For loops that potentially contain filtering operations.
        """
        return isinstance(node, ast.For)
    
    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Checks if the For loop contains filtering patterns where elements are conditionally
        appended to a list based on a condition.
        """
        smells = []
        
        # Check if we have a typical filter pattern
        if self._is_filter_pattern(node):
            smells.append(Smell(
                rule_id=self.id,
                rule_name=self.name,
                description=self.description,
                optimization=self.optimization,
                start_line=node.lineno
            ))
        
        return smells
    
    def _is_filter_pattern(self, node: ast.For) -> bool:
        """
        Detects if the for loop contains a filtering pattern:
        - Iterates over a sequence
        - Has an if condition inside
        - Appends elements to a list within the if-block
        """
        # Check if the loop has a body
        if not node.body:
            return False
        
        # Look for if statements within the loop body
        for stmt in node.body:
            if isinstance(stmt, ast.If):
                # Check if the if-block contains an append call
                for if_stmt in stmt.body:
                    if isinstance(if_stmt, ast.Expr) and isinstance(if_stmt.value, ast.Call):
                        call = if_stmt.value
                        # Check if it's a list.append() call
                        if isinstance(call.func, ast.Attribute) and call.func.attr == 'append':
                            return True
        
        return False
