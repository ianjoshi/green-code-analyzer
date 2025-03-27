import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class BroadcastingRule(BaseRule):
    """
    Detects inefficient use of tile when broadcasting could be used instead in TensorFlow code.
    """
    
    id = "broadcasting"
    name = "Broadcasting"
    description = "Use of tile where broadcasting would be more memory-efficient. Broadcasting avoids storing intermediate tiled results."
    optimization = "Leverage implicit broadcasting to perform operations directly, avoiding explicit tiling. For example, use 'a + b' instead of 'a + tf.tile(b, [1, 2])'."

    def __init__(self):
        super().__init__(id=self.id,
                         name=self.name,
                         description=self.description,
                         optimization=self.optimization)

    def should_apply(self, node) -> bool:
        """
        Applies to binary operations involving a tile call (e.g., tf.tile, tensorflow.tile).
        """
        if not isinstance(node, ast.BinOp):
            return False
        
        # Check if either operand involves a tile call
        return self._is_tile_call(node.left) or self._is_tile_call(node.right)

    def apply_rule(self, node) -> list[Smell]:
        """
        Flags binary operations using tile where broadcasting could be applied instead.
        """
        smells = []

        # Extract the tile call
        tile_node = None
        if self._is_tile_call(node.left):
            tile_node = node.left
        elif self._is_tile_call(node.right):
            tile_node = node.right

        if tile_node:
            # Verify it's a TensorFlow tile call
            func_value = tile_node.func.value
            is_tensorflow_tile = False

            if isinstance(func_value, ast.Name):
                # Case 1: Alias like tf.tile, tflow.tile, etc.
                # Assume it's TensorFlow if it’s a tile call in a binary op
                is_tensorflow_tile = True
            elif isinstance(func_value, ast.Attribute):
                # Case 2: tensorflow.tile
                if func_value.attr == "tensorflow" and isinstance(func_value.value, ast.Name) and func_value.value.id == "tensorflow":
                    is_tensorflow_tile = True

            if is_tensorflow_tile and len(tile_node.args) >= 2:  # Ensure tile has input tensor and multiples
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description=self.description,
                    penalty=self.penalty,
                    optimization=self.optimization,
                    start_line=node.lineno
                ))

        return smells
    
    def _is_tile_call(self, expr: ast.AST) -> bool:
        """
        Checks if the given expression is a TensorFlow tile call (e.g., tf.tile, tensorflow.tile).
        """
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute) or expr.func.attr != "tile":
            return False
        
        # Could be tf.tile (with alias) or tensorflow.tile (no alias)
        func_value = expr.func.value
        if isinstance(func_value, ast.Name):  # e.g., tf.tile
            return True
        elif isinstance(func_value, ast.Attribute):  # e.g., tensorflow.tile
            return func_value.attr == "tensorflow" and isinstance(func_value.value, ast.Name) and func_value.value.id == "tensorflow"
        
        return False