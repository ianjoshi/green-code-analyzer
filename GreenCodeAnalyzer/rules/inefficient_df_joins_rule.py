import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class InefficientDataFrameJoinsRule(BaseRule):
    """
    Detects inefficient join operations on Pandas DataFrames that can lead to excessive memory usage and computation.
    """
    
    id = "inefficient_df_joins"
    name = "Inefficient DataFrame Joins"
    description = "Inefficient DataFrame join operations found, such as repeated joins or joins without proper indexing."
    optimization = "Set indexes before joins with set_index() and store join results in variables to avoid repeating the same joins."
    
    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization,
        )
        # Store seen merge operations per function to detect redundant joins
        self.merge_operations_per_function = {}
        # Current function being analyzed
        self.current_function = None
        # Track DataFrames that have indices set
        self.indexed_dataframes = set()
    
    def should_apply(self, node: ast.AST) -> bool:
        """
        Applies to function definitions that might contain DataFrame operations
        and to method calls that might be DataFrame merges/joins or set_index.
        """
        # Track current function being analyzed
        if isinstance(node, ast.FunctionDef):
            self.current_function = node.name
            self.merge_operations_per_function[node.name] = []
            self.indexed_dataframes = set()
            return False
        
        # Track dataframes that have set_index called on them
        if (isinstance(node, ast.Call) and 
            isinstance(node.func, ast.Attribute) and 
            node.func.attr == 'set_index'):
            try:
                # Get the DataFrame variable name
                df_name = ast.unparse(node.func.value).strip()
                self.indexed_dataframes.add(df_name)
            except:
                pass
            return False
        
        # Check if this is an os.path.join call (which we should ignore)
        if self._is_os_path_join(node):
            return False
            
        # Check for DataFrame merge or join calls
        return (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and 
                node.func.attr in ('merge', 'join'))
    
    def _is_os_path_join(self, node: ast.AST) -> bool:
        """
        Determines if the given call node is an os.path.join call.
        """
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return False
            
        # Check if it's specifically os.path.join
        if node.func.attr != 'join':
            return False
            
        # Check if it's from os.path module
        try:
            value = node.func.value
            if isinstance(value, ast.Attribute) and value.attr == 'path':
                if isinstance(value.value, ast.Name) and value.value.id == 'os':
                    return True
        except Exception:
            pass
            
        return False
    
    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Identifies inefficient join patterns:
        1. Repeated joins of the same DataFrames
        2. Joins without proper indexing
        """
        smells = []
        
        if not isinstance(node, ast.Call):
            return smells
        
        # Check if it's a merge or join call
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('merge', 'join'):
            call_signature = self._extract_merge_signature(node)
            
            # Check for violations
            # Violation 1: No set_index before merge
            if not self._has_preceding_set_index(node):
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=f"{self.name}: Missing Index",
                    description="Merging DataFrames without setting indexes first causes inefficient lookups.",
                    optimization="Call set_index() on DataFrames before merging to speed up join operations.",
                    start_line=node.lineno
                ))
            
            # Violation 2: Redundant joins of the same DataFrames
            if self.current_function and call_signature:
                # Check if this merge operation signature was seen before in this function
                if call_signature in self.merge_operations_per_function.get(self.current_function, []):
                    smells.append(Smell(
                        rule_id=self.id,
                        rule_name=f"{self.name}: Redundant Join",
                        description="Repeating the same join operation multiple times wastes computation.",
                        optimization="Store the result of the join and reuse it instead of repeating the same join.",
                        start_line=node.lineno
                    ))
                elif self.current_function:
                    self.merge_operations_per_function[self.current_function].append(call_signature)
        
        return smells
    
    def _extract_merge_signature(self, node: ast.Call) -> str:
        """
        Extracts a simplified signature of the merge operation to detect redundant joins.
        """
        # Try to get the DataFrame variable names and join keys
        try:
            # Get the left DataFrame name
            left_df = ast.unparse(node.func.value).strip()
            
            # Extract right DataFrame and join key from arguments
            right_df = ""
            join_key = ""
            
            # Look for positional args (first arg is the right DataFrame)
            if node.args:
                right_df = ast.unparse(node.args[0]).strip()
            
            # Look for 'on' or 'left_on'/'right_on' in keyword args
            for kw in node.keywords:
                if kw.arg == 'on' and isinstance(kw.value, ast.Constant):
                    join_key = kw.value.value
                    break
                elif kw.arg == 'left_on' and isinstance(kw.value, ast.Constant):
                    join_key = f"{kw.value.value}::"
                    # Try to find matching right_on
                    for kw2 in node.keywords:
                        if kw2.arg == 'right_on' and isinstance(kw2.value, ast.Constant):
                            join_key += kw2.value.value
                            break
                    break
            
            # Create a signature string
            return f"{left_df}::{right_df}::{join_key}" if right_df else ""
            
        except Exception:
            # If any error in parsing, return empty string
            return ""
    
    def _has_preceding_set_index(self, node: ast.AST) -> bool:
        """
        Check if there's a set_index call before this merge or if the DataFrames used in the merge
        have had set_index called on them previously in the function.
        """
        # First, check if the direct method chaining includes set_index
        try:
            if isinstance(node.func.value, ast.Call):
                if (isinstance(node.func.value.func, ast.Attribute) and 
                        node.func.value.func.attr == 'set_index'):
                    return True
        except Exception:
            pass
            
        # Next, check if either DataFrame in the merge has been indexed
        try:
            # Get the left DataFrame name (the one merge/join is called on)
            left_df = ast.unparse(node.func.value).strip()
            
            # If the left DataFrame has been indexed previously
            if left_df in self.indexed_dataframes:
                return True
                
            # Check the right DataFrame (first argument to merge/join)
            if node.args:
                right_df = ast.unparse(node.args[0]).strip()
                if right_df in self.indexed_dataframes:
                    return True
        except Exception:
            pass
            
        # If using merge with 'left_index' or 'right_index' set to True
        try:
            for kw in node.keywords:
                if (kw.arg in ('left_index', 'right_index') and 
                    isinstance(kw.value, ast.Constant) and 
                    kw.value.value is True):
                    return True
        except Exception:
            pass
            
        return False
