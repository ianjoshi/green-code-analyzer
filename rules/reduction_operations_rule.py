import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class ReductionOperationsRule(BaseRule):
    """
    Detects inefficient implementations of reduction operations (sum, mean, min, max)
    using loops instead of vectorized operations in NumPy, Pandas, PyTorch, and TensorFlow.
    """
    
    id = "reduction_operations"
    name = "Inefficient Reduction Operations"
    description = "Using loops for reduction operations instead of vectorized methods consumes more energy."
    optimization = "Replace with built-in reduction methods."
    
    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )
        # Track array/tensor variables for detection
        self.array_vars = set()
        # Track known reduction variables and their operation type
        self.reduction_vars = {}  # {var_name: operation_type}
        
    def should_apply(self, node) -> bool:
        """
        Applies to For loops and assignments that might involve reduction operations.
        """
        return isinstance(node, (ast.For, ast.Assign, ast.AugAssign))
    
    def apply_rule(self, node) -> list[Smell]:
        """
        Detects loops that perform manual reduction operations that could be vectorized.
        """
        smells = []
        
        # Track array/tensor variable creation
        if isinstance(node, ast.Assign):
            self._track_array_variables(node)
            # Also track initialization of reduction accumulators
            self._track_accumulator_init(node)
            return smells
            
        # Track augmented assignments that might be reduction operations
        if isinstance(node, ast.AugAssign):
            self._track_augmented_assignments(node)
            return smells
            
        # Check for reduction operations in loops
        if isinstance(node, ast.For):
            # Look for specific patterns in loop body that indicate reduction operations
            reduction_type = self._identify_reduction_pattern(node)
            if reduction_type:
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description=f"{self.description} Found inefficient {reduction_type} operation.",
                    penalty=self.penalty,
                    optimization=self._get_specific_optimization(reduction_type),
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
                    # Handle methods like torch.zeros, np.random.rand, etc.
                    if func.attr in {'zeros', 'ones', 'zeros_like', 'ones_like', 'empty', 'empty_like', 
                                    'rand', 'randn', 'random', 'arange', 'linspace', 'array', 'tensor',
                                    'uniform', 'normal', 'randint'}:
                        self.array_vars.add(var_name)
                        
                    # Handle DataFrame creation
                    elif func.attr == 'DataFrame':
                        self.array_vars.add(var_name)
                elif isinstance(func, ast.Name):
                    # Handle direct constructor calls
                    if func.id in {'array', 'tensor', 'DataFrame'}:
                        self.array_vars.add(var_name)
                        
            # Handle pandas column indexing (treat as array)
            elif isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
                if value.value.id in self.array_vars:  # If it's a known dataframe
                    self.array_vars.add(var_name)  # Column access is also an array
    
    def _track_accumulator_init(self, node):
        """
        Tracks initialization of variables that might be reduction accumulators.
        """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            value = node.value
            
            # Check for typical accumulator initialization patterns
            if isinstance(value, ast.Constant):
                if value.value == 0:
                    # Possible sum or mean accumulator
                    self.reduction_vars[var_name] = "sum"
                elif (isinstance(value.value, (int, float)) and 
                      (value.value == float('inf') or value.value == -float('inf'))):
                    # Potential min/max accumulator
                    if value.value == float('inf'):
                        self.reduction_vars[var_name] = "min"
                    else:
                        self.reduction_vars[var_name] = "max"
    
    def _track_augmented_assignments(self, node):
        """
        Tracks augmented assignments that might be part of reduction operations.
        """
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            
            # Check for += pattern which is common in sum or mean
            if isinstance(node.op, ast.Add):
                if var_name in self.reduction_vars:
                    if self.reduction_vars[var_name] == "sum":
                        # Update to mean if division is detected
                        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Div):
                            self.reduction_vars[var_name] = "mean"
    
    def _identify_reduction_pattern(self, node: ast.For) -> str:
        """
        Identifies various reduction patterns in loops and returns the type of reduction.
        Returns: "sum", "mean", "min", "max", or None if no pattern is detected.
        """
        # Check if iterating over array or range
        is_array_iteration = False
        array_name = None
        
        # Case 1: Iterating directly over an array (for x in array)
        if isinstance(node.iter, ast.Name) and node.iter.id in self.array_vars:
            is_array_iteration = True
            array_name = node.iter.id
            
        # Case 2: Iterating over range (common pattern for array indexing)
        elif (isinstance(node.iter, ast.Call) and 
              isinstance(node.iter.func, ast.Name) and 
              node.iter.func.id == 'range'):
            is_array_iteration = True
            
            # Get array name from range argument if it's range(len(array))
            if len(node.iter.args) >= 1:
                if (isinstance(node.iter.args[0], ast.Call) and 
                    isinstance(node.iter.args[0].func, ast.Name) and 
                    node.iter.args[0].func.id == 'len' and
                    len(node.iter.args[0].args) == 1 and
                    isinstance(node.iter.args[0].args[0], ast.Name)):
                    array_name = node.iter.args[0].args[0].id
        
        if not is_array_iteration:
            return None
            
        # Check if we can find a mean pattern directly
        mean_pattern = self._detect_direct_mean_pattern(node, array_name)
        if mean_pattern:
            return "mean"
            
        # Detect sum pattern: accumulator += array[i]
        sum_pattern = self._detect_sum_pattern(node, array_name)
        if sum_pattern:
            return "sum"
            
        # Detect min pattern: if array[i] < min_val: min_val = array[i]
        if self._detect_min_pattern(node, array_name):
            return "min"
            
        # Detect max pattern: if array[i] > max_val: max_val = array[i]
        if self._detect_max_pattern(node, array_name):
            return "max"
            
        return None
    
    def _detect_direct_mean_pattern(self, node: ast.For, array_name: str) -> bool:
        """
        Detects if the loop is computing a mean by checking for a sum pattern followed by division.
        """
        # Find the sum accumulator
        sum_var = self._detect_sum_pattern(node, array_name)
        if not sum_var:
            return False
        
        # Get parent context
        parent = self._get_parent_node(node)
        if not parent or not isinstance(parent, (ast.Module, ast.FunctionDef)):
            return False
        
        # Find the loop's position and check subsequent statements
        loop_pos = -1
        for i, stmt in enumerate(parent.body):
            if stmt == node:
                loop_pos = i
                break
        
        if loop_pos == -1 or loop_pos + 1 >= len(parent.body):
            return False
        
        # Check the next statement for division
        next_stmt = parent.body[loop_pos + 1]
        if (isinstance(next_stmt, ast.Assign) and 
            isinstance(next_stmt.value, ast.BinOp) and 
            isinstance(next_stmt.value.op, ast.Div)):
            left = next_stmt.value.left
            right = next_stmt.value.right
            if isinstance(left, ast.Name) and left.id == sum_var:
                # Check if dividing by length of the array or a constant
                if (isinstance(right, ast.Call) and 
                    isinstance(right.func, ast.Name) and 
                    right.func.id == 'len' and 
                    len(right.args) == 1 and 
                    isinstance(right.args[0], ast.Name) and 
                    (array_name is None or right.args[0].id == array_name)):
                    return True
                elif isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
                    return True
        
        return False
    
    def _detect_sum_pattern(self, node: ast.For, array_name: str) -> str:
        """
        Detects sum reduction pattern and returns the accumulator variable name if found.
        """
        for stmt in node.body:
            # Look for augmented assignment (+=)
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Add):
                if isinstance(stmt.target, ast.Name):
                    accumulator = stmt.target.id
                    # Check if the value is array[i] or a reference to the iterated item
                    if self._is_array_element_access(stmt.value, array_name, node.target):
                        return accumulator
        return None
    
    def _detect_mean_pattern(self, node: ast.For, sum_var: str) -> bool:
        """
        Detects if a sum pattern is followed by division to compute mean.
        """
        # Check if there's a statement after the loop that divides the sum
        parent = self._get_parent_node(node)
        if parent and isinstance(parent, ast.Module):
            # Find the loop's position in the parent
            for i, stmt in enumerate(parent.body):
                if stmt == node and i + 1 < len(parent.body):
                    next_stmt = parent.body[i + 1]
                    # Check if next statement is an assignment with division
                    if isinstance(next_stmt, ast.Assign) and isinstance(next_stmt.value, ast.BinOp):
                        if isinstance(next_stmt.value.op, ast.Div):
                            if (isinstance(next_stmt.value.left, ast.Name) and 
                                next_stmt.value.left.id == sum_var):
                                return True
        return False
    
    def _detect_min_pattern(self, node: ast.For, array_name: str) -> bool:
        """
        Detects min reduction pattern.
        """
        for stmt in node.body:
            # Look for if array[i] < min_val: min_val = array[i]
            if isinstance(stmt, ast.If):
                # Check comparison condition
                if (isinstance(stmt.test, ast.Compare) and 
                    len(stmt.test.ops) == 1 and 
                    isinstance(stmt.test.ops[0], ast.Lt)):
                    
                    # Check if one side is array access
                    left_is_array = self._is_array_element_access(stmt.test.left, array_name, node.target)
                    right_is_min = isinstance(stmt.test.comparators[0], ast.Name)
                    
                    if left_is_array and right_is_min:
                        # Check for assignment in body
                        for substmt in stmt.body:
                            if (isinstance(substmt, ast.Assign) and 
                                isinstance(substmt.targets[0], ast.Name) and 
                                substmt.targets[0].id == stmt.test.comparators[0].id):
                                return True
                    
                    # Check reverse condition: min_val > array[i]
                    right_is_array = self._is_array_element_access(stmt.test.comparators[0], array_name, node.target)
                    left_is_min = isinstance(stmt.test.left, ast.Name)
                    if right_is_array and left_is_min:
                        # Check for assignment in body
                        for substmt in stmt.body:
                            if (isinstance(substmt, ast.Assign) and 
                                isinstance(substmt.targets[0], ast.Name) and 
                                substmt.targets[0].id == stmt.test.left.id):
                                return True
        return False
    
    def _detect_max_pattern(self, node: ast.For, array_name: str) -> bool:
        """
        Detects max reduction pattern.
        """
        for stmt in node.body:
            # Look for if array[i] > max_val: max_val = array[i]
            if isinstance(stmt, ast.If):
                # Check comparison condition
                if (isinstance(stmt.test, ast.Compare) and 
                    len(stmt.test.ops) == 1 and 
                    isinstance(stmt.test.ops[0], ast.Gt)):
                    
                    # Check if one side is array access
                    left_is_array = self._is_array_element_access(stmt.test.left, array_name, node.target)
                    right_is_max = isinstance(stmt.test.comparators[0], ast.Name)
                    
                    if left_is_array and right_is_max:
                        # Check for assignment in body
                        for substmt in stmt.body:
                            if (isinstance(substmt, ast.Assign) and 
                                isinstance(substmt.targets[0], ast.Name) and 
                                substmt.targets[0].id == stmt.test.comparators[0].id):
                                return True
                    
                    # Check reverse condition: max_val < array[i]
                    right_is_array = self._is_array_element_access(stmt.test.comparators[0], array_name, node.target)
                    left_is_max = isinstance(stmt.test.left, ast.Name)
                    if right_is_array and left_is_max:
                        # Check for assignment in body
                        for substmt in stmt.body:
                            if (isinstance(substmt, ast.Assign) and 
                                isinstance(substmt.targets[0], ast.Name) and 
                                substmt.targets[0].id == stmt.test.left.id):
                                return True
        return False
    
    def _is_array_element_access(self, node, array_name, loop_var) -> bool:
        """
        Checks if the node represents access to array elements using the loop variable.
        Handles both direct loop var use and indexing with the loop variable.
        """
        # Case 1: Direct array[i] indexing 
        if isinstance(node, ast.Subscript):
            # Handle pandas df['column'][i] pattern
            if isinstance(node.value, ast.Subscript):
                # This is likely a df['column'][i] pattern
                if isinstance(node.value.value, ast.Name):
                    # Check if it's a known DataFrame
                    if node.value.value.id in self.array_vars:
                        # Check if indexing with loop variable
                        if isinstance(node.slice, ast.Name) and isinstance(loop_var, ast.Name):
                            return node.slice.id == loop_var.id
                
            # Handle array[i] indexing
            elif isinstance(node.value, ast.Name):
                if array_name is None or node.value.id == array_name:
                    # Check if indexing with loop variable
                    if isinstance(node.slice, ast.Name) and isinstance(loop_var, ast.Name):
                        return node.slice.id == loop_var.id
            
            # Handle pandas slice pattern like df[('column', i)]
            elif isinstance(node.value, ast.Name) and node.value.id in self.array_vars:
                if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                    # Check if second element is the loop variable
                    if isinstance(node.slice.elts[1], ast.Name) and isinstance(loop_var, ast.Name):
                        return node.slice.elts[1].id == loop_var.id
                        
                # Handle pandas slice pattern like df[:('column', i)]
                elif isinstance(node.slice, ast.Slice) and isinstance(node.slice.upper, ast.Tuple):
                    if len(node.slice.upper.elts) == 2:
                        # Check if second element is the loop variable
                        if isinstance(node.slice.upper.elts[1], ast.Name) and isinstance(loop_var, ast.Name):
                            return node.slice.upper.elts[1].id == loop_var.id
        
        # Case 2: Direct reference to loop var (when iterating over the array directly)
        elif isinstance(node, ast.Name) and isinstance(loop_var, ast.Name):
            return node.id == loop_var.id
            
        return False
    
    def _get_parent_node(self, node):
        """
        Helper method to get the parent of a node (simple implementation).
        Note: This is a simplified approximation and would need AST parent tracking for accuracy.
        """
        # Return the closest ancestor Module or FunctionDef as the parent
        # In a real implementation, you'd need a proper AST parent tracking mechanism
        return None  # Simplified
    
    def _get_specific_optimization(self, reduction_type: str) -> str:
        """
        Returns specific optimization suggestion based on reduction type.
        """
        base_optimization = self.optimization
        if reduction_type == "sum":
            return f"{base_optimization} Use numpy.sum(), torch.sum(), pandas.DataFrame.sum(), or tf.reduce_sum()."
        elif reduction_type == "mean":
            return f"{base_optimization} Use numpy.mean(), torch.mean(), pandas.DataFrame.mean(), or tf.reduce_mean()."
        elif reduction_type == "min":
            return f"{base_optimization} Use numpy.min(), torch.min(), pandas.DataFrame.min(), or tf.reduce_min()."
        elif reduction_type == "max":
            return f"{base_optimization} Use numpy.max(), torch.max(), pandas.DataFrame.max(), or tf.reduce_max()."
        return base_optimization
