import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class BatchMatrixMultiplicationRule(BaseRule):
    """
    Detects sequential matrix multiplications inside loops that can be optimized with batch operations.

    This rule identifies cases where matrix multiplications (e.g., np.matmul, torch.bmm) are performed
    repeatedly within for or while loops on indexed slices of arrays. Batching these operations into a single call
    can improve performance by leveraging parallel execution, especially on GPUs.

    Example:
        Before:
            i = 0
            while i < n:
                result[i] = np.matmul(A[i], B[i])
                i += 1
        After:
            result = np.matmul(A, B)  # Batched operation
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
        """Apply the rule to both for and while loops."""
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
                    loop_vars.add(stmt.target.id)  # e.g., i += 1
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
        Detect matrix multiplication calls where arguments are indexed by loop variables.
        """
        matrix_calls = []

        for node in nodes:
            # Check if the node is a standalone expression with a call
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                # Verify it’s a matrix multiplication and uses a loop variable
                if self._is_matrix_mult_call(call) and any(self._is_indexed_with_loop_vars(arg, loop_vars) for arg in call.args):
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
        Recursively searches an AST node for matrix multiplication calls with indexed arguments.
        """
        calls = []

        if isinstance(node, ast.Call):
             # Check if this is a matrix multiplication call with indexed args
            if self._is_matrix_mult_call(node) and any(self._is_indexed_with_loop_vars(arg, loop_vars) for arg in node.args):
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

    def _is_indexed_with_loop_vars(self, arg, loop_vars: set[str]) -> bool:
        """
        Check if an argument is subscripted with any loop variables.
        """
        if isinstance(arg, ast.Subscript):
            slice_val = arg.slice
            if isinstance(slice_val, ast.Name) and slice_val.id in loop_vars:
                return True  # Single index, e.g., A[i]
            elif isinstance(slice_val, ast.Tuple):
                for elt in slice_val.elts:
                    if isinstance(elt, ast.Name) and elt.id in loop_vars:
                        return True  # Multi-dimensional index, e.g., A[i, j]
        return False