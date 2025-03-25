import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class InefficientIterationWithIterrows(BaseRule):
    """
    Detects inefficient use of Pandas iterrows for row-by-row data manipulation.
    """
    
    id = "inefficient_iterrows"
    name = "InefficientIterationWithIterrows"
    description = "Using iterrows for row-by-row Pandas operations is slow and energy-intensive due to Python overhead."
    optimization = "Replace with vectorized Pandas operations (e.g., apply, vector arithmetic, or groupby)."

    def __init__(self):
        super().__init__(id=self.id,
                        name=self.name,
                        description=self.description,
                        optimization=self.optimization)
    
    def should_apply(self, node: ast.AST) -> bool:
        """
        Applies to For loops that potentially use iterrows.
        """
        return isinstance(node, ast.For)

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Checks if the For loop uses Pandas iterrows and flags it as an energy smell.
        """
        smells = []
        
        # Check if the loop's iterator is a method call
        if isinstance(node.iter, ast.Call):
            call = node.iter
            
            # Check if the function being called is an attribute (e.g., df.iterrows)
            if isinstance(call.func, ast.Attribute):
                attr = call.func
                
                # Look for 'iterrows' as the attribute name
                if attr.attr == "iterrows":
                    # Verify it's being called (has parentheses)
                    if isinstance(attr.value, (ast.Name, ast.Attribute)):
                        smells.append(Smell(
                            rule_id=self.id,
                            rule_name=self.name,
                            description=self.description,
                            penalty=self.penalty,
                            optimization=self.optimization,
                            start_line=node.lineno
                        ))
        
        return smells