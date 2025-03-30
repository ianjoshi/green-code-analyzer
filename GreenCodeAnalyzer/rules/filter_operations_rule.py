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
    optimization = "Replace with boolean indexing (array[array > 0.5], tensor[tensor > 0.5], df[df['values'] > 0.5]) or tensor masking."   

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
        - Does NOT have an else block with append operations (that would be a conditional operation)
        """
        # Check if the loop has a body
        if not node.body:
            return False
        
        # Look for if statements within the loop body
        for stmt in node.body:
            if isinstance(stmt, ast.If):
                # Look for append calls in the if-block
                if_append_calls = self._find_append_calls(stmt.body)
                
                # Check if there are append calls in the else block
                else_append_calls = self._find_append_calls(stmt.orelse)
                
                # If we have appends in both if and else blocks, this is a conditional operation, not filtering
                if if_append_calls and else_append_calls:
                    return False
                
                # If we have different operations in if and else, this is a conditional operation
                if self._is_conditional_operation(stmt):
                    return False
                
                # If we only have append calls in the if-block (true filtering), return True
                if if_append_calls and not else_append_calls:
                    return True
        
        return False
    
    def _find_append_calls(self, stmt_list) -> list:
        """
        Finds all append() calls in a statement list.
        """
        append_calls = []
        for stmt in stmt_list:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'append':
                    append_calls.append(call)
        return append_calls
    
    def _is_conditional_operation(self, if_node: ast.If) -> bool:
        """
        Determines if an if statement represents a conditional operation instead of filtering.
        A conditional operation typically:
        1. Has different operations in if and else blocks
        2. Modifies the same target in different ways
        """
        # If there's no else block, it's not a conditional operation
        if not if_node.orelse:
            return False
        
        # Look for array assignments in both branches
        if_assigns = self._extract_assignments(if_node.body)
        else_assigns = self._extract_assignments(if_node.orelse)
        
        # If the same variable is being assigned in both branches, it's a conditional operation
        for target in if_assigns:
            if target in else_assigns:
                return True
                
        return False
    
    def _extract_assignments(self, stmt_list) -> set:
        """
        Extracts all assignment targets from a statement list.
        """
        targets = set()
        for stmt in stmt_list:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        targets.add(target.id)
                    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        targets.add(target.value.id)
        return targets
