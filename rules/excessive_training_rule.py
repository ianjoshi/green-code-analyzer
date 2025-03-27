import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class ExcessiveTrainingRule(BaseRule):
    """
    Detects excessive training patterns in machine learning code that waste energy.
    
    This rule identifies training loops without proper early stopping mechanisms.
    """
    
    id = "excessive_training"
    name = "Excessive Training"
    description = "Training loop without proper early stopping mechanism detected."
    optimization = "Implement early stopping by monitoring validation metrics and stopping when no improvement is seen for a number of epochs."
    
    def __init__(self):
        super().__init__(id=self.id,
                         name=self.name,
                         description=self.description,
                         optimization=self.optimization)
        
        # Key training terms to look for
        self.training_terms = {"train", "fit", "epoch", "backward", "optimizer"}
        # Early stopping related terms
        self.stopping_terms = {"early", "stop", "patience", "monitor", "callback", "convergence"}
        
    def should_apply(self, node: ast.AST) -> bool:
        """
        Applies to loops or training functions.
        """
        # Look for loops or functions with training-related names
        if isinstance(node, ast.FunctionDef):
            return any(term in node.name.lower() for term in self.training_terms)
        return isinstance(node, (ast.For, ast.While))
    
    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Identifies training code without early stopping logic.
        """
        smells = []
        
        # Check if this is likely a training loop or function
        has_training = False
        
        # For function definitions, check the name and body
        if isinstance(node, ast.FunctionDef):
            # Already checked name in should_apply
            has_training = True
        
        # For loops, check if they contain training-related code
        elif isinstance(node, (ast.For, ast.While)):
            # Check if the loop has a large number of iterations (for For loops)
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == "range" and len(node.iter.args) > 0:
                    # For range(X), check if X is large
                    if isinstance(node.iter.args[0], ast.Constant) and node.iter.args[0].value > 20:
                        has_training = self._contains_training_terms(node)
            
            # Check if the loop contains training-related terms in any case
            has_training = has_training or self._contains_training_terms(node)
        
        # If it looks like training code, check for early stopping mechanisms
        if has_training and not self._contains_early_stopping(node):
            smells.append(Smell(
                rule_id=self.id,
                rule_name=self.name,
                description=self.description,
                optimization=self.optimization,
                start_line=node.lineno,
                end_line=getattr(node, 'end_lineno', node.lineno)
            ))
        
        return smells
    
    def _contains_training_terms(self, node: ast.AST) -> bool:
        """
        Simple check for training-related terms in AST node text.
        """
        node_src = ast.unparse(node)
        return any(term in node_src.lower() for term in self.training_terms)
    
    def _contains_early_stopping(self, node: ast.AST) -> bool:
        """
        Simple check for early stopping mechanisms.
        """
        # Look for early stopping terms in the node source
        node_src = ast.unparse(node)
        has_stopping_terms = any(term in node_src.lower() for term in self.stopping_terms)
        
        # Also look for 'break' statements inside loops, which might indicate early stopping
        has_break = "break" in node_src
        
        # Check for early stopping callbacks (common in frameworks)
        has_callback = "callback" in node_src.lower() and "early" in node_src.lower()
        
        return has_stopping_terms or has_break or has_callback
