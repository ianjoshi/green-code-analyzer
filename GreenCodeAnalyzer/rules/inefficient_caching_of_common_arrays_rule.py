import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class IneffectiveCachingOfCommonArrays(BaseRule):
    """
    Detects inefficient recreation of identical arrays or tensors inside loops.
    """
    
    id = "ineffective_array_caching"
    name = "Ineffective Caching Of Common Arrays"
    description = (
        "Recreating identical arrays or tensors inside loops wastes CPU/GPU cycles "
        "and memory, increasing energy consumption."
    )
    optimization = "Cache the array outside the loop to eliminate repeated creation."

    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )
    
    def should_apply(self, node) -> bool:
        """
        Determines if the rule applies to the given AST node.
        """
        return isinstance(node, (ast.For, ast.While))

    def apply_rule(self, node) -> list[Smell]:
        """
        Analyzes the loop for array creation calls with loop-invariant arguments.
        """
        smells = []
        
        # Set of deterministic array creation function names from common libraries
        array_funcs = {
            'arange', 'zeros', 'ones', 'empty', 'full', 'linspace', 'meshgrid',
            'eye', 'identity', 'tri', 'vander'
        }
        
        # Collect variables assigned within the loop
        assigned_vars = set()
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            assigned_vars.add(node.target.id)  # Include loop variable
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
        
        # Identify array creation calls within the loop
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Call):
                func = stmt.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr
                
                if func_name in array_funcs:
                    # Collect variables used in the function arguments
                    dependent_vars = set()
                    for arg in stmt.args:
                        dependent_vars.update(self._get_variables(arg))
                    for kw in stmt.keywords:
                        dependent_vars.update(self._get_variables(kw.value))
                    
                    # If no dependent variables are assigned in the loop, it's a smell
                    if not dependent_vars.intersection(assigned_vars):
                        smells.append(Smell(
                            rule_id=self.id,
                            rule_name=self.name,
                            description=self.description,
                            penalty=self.penalty,
                            optimization=self.optimization,
                            start_line=stmt.lineno
                        ))
        
        return smells
    
    def _get_variables(self, node):
        """
        Extracts variable names used in an expression.
        """
        variables = set()
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            variables.add(node.id)
        else:
            for child in ast.iter_child_nodes(node):
                variables.update(self._get_variables(child))
        return variables