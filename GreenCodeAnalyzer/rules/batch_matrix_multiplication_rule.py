import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class BatchMatrixMultiplicationRule(BaseRule):
    """
    Detects sequential matrix multiplications inside loops that can be optimized with batch operations.
    """
    id = "batch_matrix_mult"
    name = "Sequential Matrix Multiplications"
    description = "Sequential matrix multiplications detected inside loops that could be batched. Batching reduces energy use by leveraging parallel GPU execution."
    optimization = (
        "Replace loop with batch operations like numpy.matmul(A, B), torch.bmm(A, B), or tf.linalg.matmul(A, B), "
        "where A and B are higher-dimensional arrays."
    )

    def __init__(self):
        super().__init__(id=self.id, name=self.name, description=self.description, optimization=self.optimization)

    def should_apply(self, node: ast.AST) -> bool:
        """
        Apply the rule to both for and while loops.
        """
        return isinstance(node, (ast.For, ast.While))

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Analyze the loop body for matrix multiplications that can be batched.
        """
        smells = []
        loop_vars = self._get_loop_vars(node)

        # Only proceed if loop variables are identified
        if loop_vars:
            # Find matrix multiplication calls in the loop body
            matrix_ops = self._detect_matrix_multiplications(node.body, loop_vars)
            
            for call in matrix_ops:
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description=self.description,
                    optimization=self.optimization,
                    start_line=call.lineno
                ))

        return smells

    def _get_loop_vars(self, node: ast.AST) -> set[str]:
        """
        Extract loop variables from for or while loops.
        """
        loop_vars = set()

        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            # For loops explicitly define the loop variable in the target
            loop_vars.add(node.target.id)
        elif isinstance(node, ast.While):
            # For while loops, look for variables modified in the body
            for stmt in node.body:
                if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                    # Detect augmented assignments like 'i += 1'
                    loop_vars.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    # Check each assignment target
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and isinstance(stmt.value, ast.BinOp):
                            # Look for patterns like 'i = i + 1'
                            if isinstance(stmt.value.left, ast.Name) and stmt.value.left.id == target.id:
                                loop_vars.add(target.id)
        return loop_vars

    def _detect_matrix_multiplications(self, nodes: list[ast.AST], loop_vars: set[str]) -> list[ast.Call]:
        """
        Detect matrix multiplication calls where arguments contain subscripts indexed by loop variables.
        """
        matrix_calls = []

        for node in nodes:
            # Check if the node is a standalone expression with a call
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                # Verify it’s a matrix multiplication and arguments contain loop-variable-indexed subscripts
                if self._is_matrix_mult_call(call) and any(self._contains_subscript_with_loop_vars(arg, loop_vars) for arg in call.args):
                    matrix_calls.append(call)
            # Check assignments
            elif isinstance(node, ast.Assign):
                matrix_calls.extend(self._find_calls(node.value, loop_vars))
            # Handle other expressions not tied to assignments
            elif isinstance(node, ast.Expr):
                matrix_calls.extend(self._find_calls(node.value, loop_vars))

        return matrix_calls

    def _find_calls(self, node, loop_vars: set[str]) -> list[ast.Call]:
        """
        Recursively searches an AST node for matrix multiplication calls with arguments containing loop-variable-indexed subscripts.
        """
        calls = []

        if isinstance(node, ast.Call):
            # Check if this is a matrix multiplication call with relevant subscripts
            if self._is_matrix_mult_call(node) and any(self._contains_subscript_with_loop_vars(arg, loop_vars) for arg in node.args):
                calls.append(node)
        elif isinstance(node, (ast.Expr, ast.Assign, ast.BinOp)):
            # Recursively explore all fields of the node
            for child in ast.iter_fields(node):
                if isinstance(child[1], (ast.AST, list)):
                    if isinstance(child[1], list):
                        for subnode in child[1]:
                            if isinstance(subnode, ast.AST):
                                calls.extend(self._find_calls(subnode, loop_vars))
                    else:
                        calls.extend(self._find_calls(child[1], loop_vars))
        return calls

    def _is_matrix_mult_call(self, call: ast.Call) -> bool:
        """
        Check if a call is a matrix multiplication function (e.g., np.matmul, torch.bmm).
        """
        # Ensure the function being called is an attribute or name
        if not isinstance(call.func, (ast.Attribute, ast.Name)):
            return False
        if isinstance(call.func, ast.Attribute):
            func_name = call.func.attr

            # Check if call matches 'np.matmul', 'torch.bmm' or 'tf.linalg.matmul'
            if func_name in ("matmul", "bmm"):
                value = call.func.value
                if isinstance(value, ast.Name) and value.id in ("np", "torch"):
                    return True
                elif (isinstance(value, ast.Attribute) and 
                      isinstance(value.value, ast.Name) and 
                      value.value.id == "tf" and value.attr == "linalg"):
                    return True
                
        return False

    def _contains_subscript_with_loop_vars(self, node, loop_vars: set[str]) -> bool:
        """
        Recursively checks if a node or its subnodes contain a subscript using a loop variable.
        """
        # Check if the node is a subscript (e.g., A[i])
        if isinstance(node, ast.Subscript):
            if self._slice_uses_loop_var(node.slice, loop_vars):
                return True
            return self._contains_subscript_with_loop_vars(node.value, loop_vars)
        # Handle method calls (e.g., tensor[i].unsqueeze(0))
        elif isinstance(node, ast.Call):
            # Check the function being called for subscripts
            if self._contains_subscript_with_loop_vars(node.func, loop_vars):
                return True
            # Check each argument of the call
            for arg in node.args:
                if self._contains_subscript_with_loop_vars(arg, loop_vars):
                    return True
        # Handle attribute access (e.g., tensor[i].method)
        elif isinstance(node, ast.Attribute):
            return self._contains_subscript_with_loop_vars(node.value, loop_vars)
        # Handle operations like addition, multiplication, etc.
        elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp)):
            # Iterate over possible operand fields
            for field in ('left', 'right', 'operand', 'values'):
                if hasattr(node, field):
                    operand = getattr(node, field)
                    if isinstance(operand, list):
                        for op in operand:
                            if self._contains_subscript_with_loop_vars(op, loop_vars):
                                return True
                    else:
                        if self._contains_subscript_with_loop_vars(operand, loop_vars):
                            return True
                        
        # If no relevant subscripts are found, return False
        return False

    def _slice_uses_loop_var(self, slice_node, loop_vars: set[str]) -> bool:
        """
        Checks if a slice uses a loop variable.
        """
        # Check if the slice is a single variable (e.g., i in A[i])
        if isinstance(slice_node, ast.Name):
            return slice_node.id in loop_vars
        # Handle multi-dimensional slices
        elif isinstance(slice_node, ast.Tuple):
            return any(self._slice_uses_loop_var(elt, loop_vars) for elt in slice_node.elts)
        
        return False