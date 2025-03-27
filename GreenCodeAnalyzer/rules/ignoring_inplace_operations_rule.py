import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class IgnoringInplaceOperationsRule(BaseRule):
    """
    Detects operations in PyTorch, TensorFlow, NumPy, or Pandas that could use in-place variants to reduce memory allocations.
    """
    
    id = "ignoring_inplace_ops"
    name = "Ignoring Inplace Operations"
    description = "Using non-in-place operations (e.g., add instead of add_) in PyTorch, TensorFlow, NumPy, or Pandas increases memory allocations, raising energy consumption."
    optimization = "Replace with in-place operations (e.g., add_(), inplace=True) where safe to reduce memory overhead."
    
    def __init__(self):
        super().__init__(id=self.id,
                         name=self.name, 
                         description=self.description, 
                         optimization=self.optimization)
    
    def should_apply(self, node) -> bool:
        """
        Applies to Call nodes that might represent library operations.
        """
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    
    def apply_rule(self, node) -> list[Smell]:
        """
        Checks if an operation in PyTorch, TensorFlow, NumPy, or Pandas lacks an in-place variant.
        """
        # Library-specific inplaceable operations
        library_ops = {
            "torch": {"add", "mul", "div", "sub", "relu", "clamp", "sigmoid", "tanh"},
            "tf": {"add", "multiply", "divide", "subtract"},
            "np": {"add", "multiply", "divide", "subtract"},
            "pandas": {"add", "mul", "div", "sub"}
        }
        
        # Extract the function name and its parent
        func_name = node.func.attr
        parent = node.func.value
        
        # Track if a smell has already been created to prevent duplicates
        smell_created = False
        smells = []
        
        for lib, ops in library_ops.items():
            if func_name in ops and not smell_created:
                # Verify library context (e.g., torch.add, tf.add, np.add, df.add)
                is_library_call = (
                    (isinstance(parent, ast.Name) and parent.id == lib) or
                    (isinstance(parent, ast.Attribute) and lib in str(ast.unparse(parent))) or
                    isinstance(parent, ast.Name)
                )
                
                if is_library_call:
                    # For Pandas, check if inplace=True is missing
                    if lib == "pandas" and not any(
                        isinstance(kw, ast.keyword) and kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value
                        for kw in node.keywords
                    ):
                        smells.append(Smell(
                            rule_id=self.id,
                            rule_name=self.name,
                            description=self.description,
                            penalty=self.penalty,
                            optimization=self.optimization,
                            start_line=node.lineno
                        ))
                        smell_created = True
                    
                    # For other libraries, flag non-in-place ops directly
                    elif lib != "pandas":
                        smells.append(Smell(
                            rule_id=self.id,
                            rule_name=self.name,
                            description=self.description,
                            penalty=self.penalty,
                            optimization=self.optimization,
                            start_line=node.lineno
                        ))
                        smell_created = True
        
        return smells