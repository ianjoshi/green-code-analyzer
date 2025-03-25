import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class LongLoopRule(BaseRule):
    """
    Detects long-running loops that may cause excessive energy consumption.
    """
    
    id = "long_loop"
    name = "LongLoopRule"
    description = "Loop may be inefficient due to excessive iterations."
    optimization = "Consider using a more efficient algorithm or data structure to reduce the number of iterations."

    def __init__(self):
        super().__init__(id=self.id,
                         name=self.name, 
                         description=self.description, 
                         optimization=self.optimization)
    
    def should_apply(self, node) -> bool:
        """
        Applies to 'for' and 'while' loop nodes.
        """
        return isinstance(node, (ast.For, ast.While))

    def apply_rule(self, node) -> list[Smell]:
        """
        Checks if a loop has a large number of iterations and flags it as an energy smell.
        """
        smells = []
        
        if isinstance(node, ast.For):
            # If iterating over a range, check the loop bounds
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                if len(node.iter.args) == 1:  # range(n)
                    upper_bound = node.iter.args[0]
                elif len(node.iter.args) == 2:  # range(start, stop)
                    upper_bound = node.iter.args[1]
                elif len(node.iter.args) == 3:  # range(start, stop, step)
                    upper_bound = node.iter.args[1]
                else:
                    upper_bound = None
                
                if isinstance(upper_bound, ast.Constant) and isinstance(upper_bound.value, int):
                    if upper_bound.value > 10000:  # Threshold for large loops
                        smells.append(Smell(
                            rule_id=self.id,
                            rule_name=self.name,
                            description=self.description,
                            penalty=self.penalty,
                            optimization=self.optimization,
                            start_line=node.lineno
                        ))
        
        return smells