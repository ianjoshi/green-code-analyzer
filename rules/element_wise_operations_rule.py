import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class ElementWiseOperartionsRule(BaseRule):
    """
    Detects inefficient element-wise operations using loops instead of vectorized operations or mapping.
    in NumPy and Pytorch. TensorFlow tensors require to be turned into NumPy arrays before applying this 
    rule so we assume they are also accounted for.
    """
    
    id = "element_wise_operations"
    name = "Element-wise Operations"
    description = "Using loops for element-wise operations instead of vectorized operations wastes CPU/GPU cycles and memory."
    optimization = "Replace loops with vectorized operations (e.g., array + 1, tensor**2)."
    
    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )
        self.array_vars = set()
        
    def should_apply(self, node) -> bool:
        """
        Applies to For loops and assignments that might create arrays/tensors.
        """
        return isinstance(node, (ast.For, ast.Assign))
    
    def apply_rule(self, node) -> list[Smell]:
        """
        Detects if the For loops perform element-wise operations on arrays/tensors.
        """
        smells = []
        
        # Track array/tensor variable creation
        if isinstance(node, ast.Assign):
            self._track_array_variables(node)
            return smells
            
        # Check for element-wise operations in loops
        if isinstance(node, ast.For):
            # Look for specific patterns in loop body that indicate element-wise operations
            if self._is_indexed_assignment_loop(node):
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description=self.description,
                    penalty=self.penalty,
                    optimization=self.optimization,
                    start_line=node.lineno
                ))
                
        return smells
    
    def _track_array_variables(self, node):
        """
        Tracks variables that might be arrays or tensors based on their initialization.
        """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            value = node.value
            
            # Check if it's initialized with a known tensor/array constructor
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Attribute):
                    # Handle methods like torch.zeros_like, np.zeros, etc.
                    if func.attr in {'zeros', 'ones', 'zeros_like', 'ones_like', 'empty', 'empty_like', 
                                    'rand', 'randn', 'random', 'arange', 'linspace', 'array', 'tensor'}:
                        self.array_vars.add(var_name)
                elif isinstance(func, ast.Name):
                    # Handle direct constructor calls
                    if func.id in {'array', 'tensor'}:
                        self.array_vars.add(var_name)
    
    def _is_indexed_assignment_loop(self, node: ast.For) -> bool:
        """
        Checks if a for loop is performing element-wise operations on arrays/tensors.
        """
        # Check if it's iterating over a range (common pattern in element-wise operations)
        is_range_iteration = False
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
            is_range_iteration = True
            
            # Also check if the range argument involves "len" of an array variable
            if len(node.iter.args) >= 1:
                if (isinstance(node.iter.args[0], ast.Call) and 
                    isinstance(node.iter.args[0].func, ast.Name) and 
                    node.iter.args[0].func.id == 'len' and
                    len(node.iter.args[0].args) == 1 and
                    isinstance(node.iter.args[0].args[0], ast.Name)):
                    # It's range(len(something))
                    array_name = node.iter.args[0].args[0].id
                    if array_name in self.array_vars:
                        is_range_iteration = True
                    
        if not is_range_iteration:
            return False
        
        # Look for indexed assignments in loop body
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                
                # Check for array[i] = ... pattern
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    array_name = target.value.id
                    
                    # Check if we're doing assignment to a previously identified array
                    if array_name in self.array_vars:
                        # Check if we're also using the array in the right-hand side (common in element-wise ops)
                        rhs = stmt.value
                        if self._contains_same_array_access(rhs, array_name, node.target):
                            return True
                        
                        # Look for operations that could be vectorized (arithmetic, power, etc.)
                        if self._is_vectorizable_operation(rhs):
                            return True
        
        return False
    
    def _contains_same_array_access(self, node, array_name, loop_var) -> bool:
        """
        Checks if an expression contains access to the same array with the loop variable.
        Example: `array[i] = array[i] + 1`
        """
        for child in ast.walk(node):
            if (isinstance(child, ast.Subscript) and
                isinstance(child.value, ast.Name) and
                child.value.id == array_name and
                isinstance(child.slice, ast.Name) and
                child.slice.id == loop_var.id):
                return True
        return False
    
    def _is_vectorizable_operation(self, node) -> bool:
        """
        Checks if an operation could be easily vectorized.
        """
        # Check for arithmetic operations
        if isinstance(node, ast.BinOp):
            return isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow))
            
        # Check for functions like math.sin, math.exp, etc.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return node.func.attr in {
                'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
                'square', 'abs', 'pow', 'floor', 'ceil'
            }
            
        return False
