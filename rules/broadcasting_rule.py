import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class BroadcastingRule(BaseRule):
    """
    Detects inefficient use of tf.tile when broadcasting could be used instead.
    """
    
    id = "broadcasting"
    name = "Broadcasting"
    description = "Use of tf.tile where broadcasting would be more memory-efficient. Broadcasting avoids storing intermediate tiled results."
    optimization = "Leverage implicit broadcasting to perform operations directly, avoiding explicit tiling with tf.tile. For example, use 'a + b' instead of 'a + tf.tile(b, [1, 2])'."

    def __init__(self):
        super().__init__(id=self.id,
                         name=self.name,
                         description=self.description,
                         optimization=self.optimization)

    def should_apply(self, node) -> bool:
        """
        Applies to binary operations involving a tf.tile call.
        """
        if not isinstance(node, ast.BinOp):
            return False
        
        # Check if either operand involves a tf.tile call
        def has_tf_tile(expr):
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
                return expr.func.attr == "tile" and isinstance(expr.func.value, ast.Name) and expr.func.value.id == "tf"
            return False

        return has_tf_tile(node.left) or has_tf_tile(node.right)

    def apply_rule(self, node) -> list[Smell]:
        """
        Flags binary operations using tf.tile where broadcasting could be applied instead.
        """
        smells = []

        # Extract the tf.tile call and its arguments
        tile_node = None
        if isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Attribute) and node.left.func.attr == "tile":
            tile_node = node.left
        elif isinstance(node.right, ast.Call) and isinstance(node.right.func, ast.Attribute) and node.right.func.attr == "tile":
            tile_node = node.right

        if tile_node:
            # Verify it's a tf.tile call
            if (isinstance(tile_node.func, ast.Attribute) and 
                tile_node.func.attr == "tile" and 
                isinstance(tile_node.func.value, ast.Name) and 
                tile_node.func.value.id == "tf"):
                
                # Check that tf.tile has at least two arguments (input tensor and multiples)
                if len(tile_node.args) >= 2:
                    smells.append(Smell(
                        rule_id=self.id,
                        rule_name=self.name,
                        description=self.description,
                        penalty=self.penalty,
                        optimization=self.optimization,
                        start_line=node.lineno
                    ))

        return smells