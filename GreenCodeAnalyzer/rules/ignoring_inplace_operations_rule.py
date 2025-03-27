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
        # Library-specific inplaceable operations (non-in-place names)
        library_ops = {
            "torch": {"add", "mul", "div", "sub", "relu", "clamp", "sigmoid", "tanh"},
            "tf": {"add", "multiply", "divide", "subtract"},
            "np": {"add", "multiply", "divide", "subtract"},
            "pandas": {"add", "mul", "div", "sub"}
        }
        
        func_name = node.func.attr
        parent = node.func.value
        smells = []
        
        # Check each library, but only create one smell per call
        for lib, ops in library_ops.items():
            # For PyTorch, also consider in-place variants (e.g., add_)
            if lib == "torch" and (func_name in ops or func_name.rstrip("_") in ops):
                is_library_call = (
                    (isinstance(parent, ast.Name) and parent.id == lib) or
                    (isinstance(parent, ast.Attribute) and lib in str(ast.unparse(parent))) or
                    isinstance(parent, ast.Name)
                )
                
                if is_library_call and not func_name.endswith("_"):
                    smells.append(Smell(
                        rule_id=self.id,
                        rule_name=self.name,
                        description=self.description,
                        penalty=self.penalty,
                        optimization=self.optimization,
                        start_line=node.lineno
                    ))
                    break  # Stop after first match to avoid duplicates
            
            # For other libraries
            elif func_name in ops:
                is_library_call = (
                    (isinstance(parent, ast.Name) and parent.id == lib) or
                    (isinstance(parent, ast.Attribute) and lib in str(ast.unparse(parent))) or
                    isinstance(parent, ast.Name)
                )
                
                if is_library_call:
                    if lib == "pandas":
                        # Check for inplace=True
                        if not any(
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
                            break
                    
                    elif lib == "np":
                        # Check for out= parameter
                        if not any(
                            isinstance(kw, ast.keyword) and kw.arg == "out"
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
                            break
                    
                    elif lib == "tf":
                        # Flag TensorFlow directly (no direct in-place equivalent)
                        smells.append(Smell(
                            rule_id=self.id,
                            rule_name=self.name,
                            description=self.description,
                            penalty=self.penalty,
                            optimization=self.optimization,
                            start_line=node.lineno
                        ))
                        break
        
        return smells