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
                # Ensure condition accesses an array element
                if not self._accesses_array_element(stmt.test):
                    continue

                # For NumPy and Pandas, we need more specific checks
                if not stmt.body:
                    continue
                
                # Extract operations from both branches
                body_ops = self._extract_operation_type(stmt.body)
                orelse_ops = self._extract_operation_type(stmt.orelse) if stmt.orelse else set()
                
                # First, check for NumPy operations - distinguish from simple filtering
                numpy_operation = self._has_numpy_conditional_operation(stmt, body_ops, orelse_ops)
                if numpy_operation:
                    return True

                # Special case for Pandas operations
                pandas_ops = any('pandas' in str(op) for op in body_ops)
                if pandas_ops:
                    return True
                
                # Check if same variable is modified with different operations
                if body_ops and orelse_ops and any(b[1] == o[1] for b in body_ops for o in orelse_ops):
                    return True

        return False
    
    def _has_numpy_conditional_operation(self, if_stmt, body_ops, orelse_ops):
        """
        Determines if this is a NumPy conditional operation (not just filtering).
        A conditional operation typically modifies the same variable differently in each branch
        or appends different values to the same list.
        """
        # For NumPy specifically, we look for patterns like:
        # if array[i] > 0.5: 
        #     modified_elements.append(array[i] + 1) 
        # else:
        #     modified_elements.append(array[i] - 1)
        
        # Check if both branches contain append operations
        body_appends = self._get_append_targets(if_stmt.body)
        orelse_appends = self._get_append_targets(if_stmt.orelse) if if_stmt.orelse else []
        
        # If there are appends to the same list in both branches,
        # with different operations, this is a conditional operation
        common_targets = set(body_appends).intersection(orelse_appends)
        if common_targets:
            return True
            
        # Check if both branches modify array elements
        body_assigns = {op[1] for op in body_ops if op[0] != 'append'}
        orelse_assigns = {op[1] for op in orelse_ops if op[0] != 'append'}
        
        # If the same array is modified in both branches, this is a conditional operation
        if body_assigns.intersection(orelse_assigns):
            return True
            
        return False
        
    def _get_append_targets(self, stmt_list):
        """
        Gets the targets of append() calls in the statement list.
        """
        append_targets = []
        for stmt in stmt_list:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'append':
                    if isinstance(call.func.value, ast.Name):
                        append_targets.append(call.func.value.id)
        return append_targets
    
    def _accesses_array_element(self, node) -> bool:
        """
        Check if the node accesses an array or DataFrame element using subscript notation
        or Pandas .loc/.iloc notation.
        """
        for subnode in ast.walk(node):
            # Standard subscript check (for lists, NumPy arrays, etc.)
            if isinstance(subnode, ast.Subscript):
                # Check for pandas df.loc[i, "col"] pattern
                if isinstance(subnode.value, ast.Attribute) and subnode.value.attr in {"loc", "iloc"}:
                    return True
                return True
                
            # Check for Pandas .loc or .iloc usage
            if isinstance(subnode, ast.Attribute) and subnode.attr in {"loc", "iloc"}:
                return True
                
        return False
        
    def _extract_operation_type(self, body) -> set:
        """
        Extracts the type of operations being performed in a branch.
        Detects assignment to elements and append operations.
        """
        operations = set()

        for stmt in body:
            # Handle append operations (common in NumPy patterns)
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'append':
                    if isinstance(call.func.value, ast.Name):
                        target = call.func.value.id
                        # Check for arithmetic operations in the append argument
                        if len(call.args) > 0 and isinstance(call.args[0], ast.BinOp):
                            op_type = type(call.args[0].op).__name__
                            operations.add(('append_' + op_type, target))
                        else:
                            operations.add(('append', target))

            # Handle assignments
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    # Handle direct subscript access like arr[i]
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name):
                            var_name = target.value.id
                            if isinstance(stmt.value, ast.BinOp):
                                op_type = type(stmt.value.op).__name__
                                operations.add((op_type, var_name))
                            else:
                                operations.add(('assign', var_name))
                        
                        # Handle Pandas df.loc[i, "col"] pattern
                        elif isinstance(target.value, ast.Attribute) and target.value.attr in {"loc", "iloc"}:
                            if isinstance(target.value.value, ast.Name):
                                var_name = f"pandas:{target.value.value.id}"
                                if isinstance(stmt.value, ast.BinOp):
                                    op_type = type(stmt.value.op).__name__
                                    operations.add((op_type, var_name))
                                else:
                                    operations.add(('assign', var_name))
                    
                    # Handle attribute access like df.loc[...]
                    elif isinstance(target, ast.Attribute) and target.attr in {"loc", "iloc"}:
                        if isinstance(target.value, ast.Name):
                            var_name = f"pandas:{target.value.id}"
                            if isinstance(stmt.value, ast.BinOp):
                                op_type = type(stmt.value.op).__name__
                                operations.add((op_type, var_name))
                            else:
                                operations.add(('assign', var_name))
        
        return operations

