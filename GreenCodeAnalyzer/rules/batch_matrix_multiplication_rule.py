import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class BatchMatrixMultiplicationRule(BaseRule):
    id = "batch_matrix_mult"
    name = "Sequential Batch Matrix Multiplications"
    description = "Consecutive batch matrix multiplications detected. Batching reduces energy use by leveraging parallel GPU execution."
    optimization = "Use batch operations like numpy.matmul(batch1, batch2), torch.bmm(batch1, batch2), or tf.linalg.matmul(batch1, batch2)."

    def __init__(self):
        super().__init__(id=self.id, name=self.name, description=self.description, optimization=self.optimization)

    def should_apply(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.For, ast.While))

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        smells = []
        matrix_ops = self._detect_matrix_multiplications(node.body)
        for call in matrix_ops:
            smells.append(Smell(
                rule_id=self.id,
                rule_name=self.name,
                description=self.description,
                optimization=self.optimization,
                start_line=call.lineno
            ))
        return smells
    
    def _detect_matrix_multiplications(self, nodes: list[ast.AST]) -> list[ast.Call]:
        """
        Detects matrix multiplication calls, including those within expressions.
        """
        matrix_calls = []

        def _find_calls(node):
            """Recursively find Call nodes within an AST node."""
            calls = []
            if isinstance(node, ast.Call) and self._is_matrix_mult_call(node):
                calls.append(node)
            elif isinstance(node, (ast.Expr, ast.Assign, ast.BinOp)):
                # Recursively search sub-nodes
                for child in ast.iter_fields(node):
                    if isinstance(child[1], (ast.AST, list)):
                        if isinstance(child[1], list):
                            for subnode in child[1]:
                                if isinstance(subnode, ast.AST):
                                    calls.extend(_find_calls(subnode))
                        else:
                            calls.extend(_find_calls(child[1]))
            return calls

        for node in nodes:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if self._is_matrix_mult_call(node.value):
                    matrix_calls.append(node.value)
            elif isinstance(node, ast.Assign):
                # Look inside the assigned value (e.g., BinOp)
                matrix_calls.extend(_find_calls(node.value))
            elif isinstance(node, ast.Expr):
                # Handle standalone expressions
                matrix_calls.extend(_find_calls(node.value))

        return matrix_calls

    def _is_matrix_mult_call(self, call: ast.Call) -> bool:
        if not isinstance(call.func, (ast.Attribute, ast.Name)):
            return False
        if isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
            if func_name in ("matmul", "bmm"):
                value = call.func.value
                if isinstance(value, ast.Name) and value.id in ("np", "torch"):
                    return True
                elif (isinstance(value, ast.Attribute) and 
                      isinstance(value.value, ast.Name) and 
                      value.value.id == "tf" and value.attr == "linalg"):
                    return True
        return False