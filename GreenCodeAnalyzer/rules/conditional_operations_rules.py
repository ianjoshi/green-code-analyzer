import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class ConditionalOperationsRule(BaseRule):
    """
    Detects inefficient conditional operations that should be vectorized.
    This rule identifies loops that perform element-wise conditional operations
    on arrays/tensors that could be vectorized using library functions.
    """
    
    id = "conditional_operations"
    name = "Conditional Operations"
    description = (
        "Element-wise conditional operations performed in loops are inefficient and "
        "waste computational resources, increasing energy consumption."
    )
    optimization = (
        "Use vectorized operations like np.where(), torch.where(), or DataFrame.loc with conditions "
        "instead of iterating through elements with loops."
    )

    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )
    
    def should_apply(self, node) -> bool:
        """
        Checks if the node is a For or While loop that iterates over a sequence.
        """
        if isinstance(node, ast.For):
            # Check for range(len(df))
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == 'range':
                    return True
            # Check for df.index or df.iterrows() iteration
            if isinstance(node.iter, ast.Attribute) and node.iter.attr in {"index", "iterrows"}:
                return True
            
        return isinstance(node, ast.While)

    
    def apply_rule(self, node) -> list[Smell]:
        """
        Analyzes the loop for conditional operations specifically where different 
        operations are performed based on a condition.
        """
        smells = []
        
        try:
            # Make sure this is truly a conditional operation and not just filtering
            if self._is_conditional_branch_operation(node):
                smell = Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description=self.description,
                    start_line=node.lineno,
                    end_line=getattr(node, 'end_lineno', node.lineno),
                    optimization=self.optimization,
                )
                smells.append(smell)
        except Exception as e:
            # Handle any exceptions that might occur during analysis
            pass
        
        return smells
    
    def _is_conditional_branch_operation(self, node) -> bool:
        """
        Identifies true conditional branching operations in a loop, where different
        actions are taken based on a condition.
        """
        if not hasattr(node, 'body'):
            return False

        for stmt in node.body:
            if isinstance(stmt, ast.If):
                # Ensure condition accesses a DataFrame element
                if not self._accesses_array_element(stmt.test):
                    continue

                if not stmt.body or not stmt.orelse:
                    continue

                body_ops = self._extract_operation_type(stmt.body)
                orelse_ops = self._extract_operation_type(stmt.orelse)

                # Check if same column is modified with different values
                if body_ops and orelse_ops and any(b[1] == o[1] for b in body_ops for o in orelse_ops):
                    return True

        return False

    
    def _accesses_array_element(self, node) -> bool:
        """
        Check if the node accesses an array or DataFrame element using subscript notation
        or Pandas .loc/.iloc notation.
        """
        for subnode in ast.walk(node):
            # Standard subscript check (for lists, NumPy arrays, etc.)
            if isinstance(subnode, ast.Subscript):
                return True
            # Check for Pandas .loc or .iloc usage
            if isinstance(subnode, ast.Attribute) and subnode.attr in {"loc", "iloc"}:
                return True
        return False
        
    def _extract_operation_type(self, body) -> set:
        """
        Extracts the type of operations being performed in a branch.
        Detects assignment to DataFrame elements via .loc or .iloc.
        """
        operations = set()

        for stmt in body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Subscript) or (
                        isinstance(target, ast.Attribute) and target.attr in {"loc", "iloc"}
                    ):
                        if isinstance(target.value, ast.Name):
                            var_name = target.value.id

                            if isinstance(stmt.value, ast.BinOp):
                                op_type = type(stmt.value.op).__name__
                                operations.add((op_type, var_name))
                            else:
                                operations.add(('assign', var_name))
        
        return operations

