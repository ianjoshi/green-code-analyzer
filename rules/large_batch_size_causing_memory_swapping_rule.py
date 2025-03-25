import ast
from models.smell import Smell
from rules.base_rule import BaseRule
from typing import Optional

class LargeBatchSizesCausingMemorySwapping(BaseRule):
    """
    Detects overly large batch sizes in PyTorch or TensorFlow that may cause memory swapping.
    """
    
    id = "large_batch_size"
    name = "Overly Large Batch Sizes May Cause Memory Swapping"
    description = "Overly large batch sizes may exceed GPU memory, causing swapping and increasing energy usage."
    optimization = "Experiment with smaller batch sizes or use gradient accumulation to optimize memory use."
    
    # Threshold for what constitutes a "large" batch size
    THRESHOLD = 1024

    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization,
            penalty=None  # Penalty ignored as per requirements
        )
    
    def should_apply(self, node: ast.AST) -> bool:
        """
        Determines if the rule should be applied to the given AST node.
        Only applies to function calls where batch sizes are typically set.
        """
        return isinstance(node, ast.Call)

    def apply_rule(self, node: ast.Call) -> list[Smell]:
        """
        Analyzes function call nodes to detect large batch sizes passed as keyword or positional arguments.
        """
        smells = []
        
        # Check keyword arguments for 'batch_size'
        for keyword in node.keywords:
            if keyword.arg == 'batch_size':
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, int):
                    if value.value > self.THRESHOLD:
                        smells.append(self._create_smell(node.lineno))
        
        # Check positional arguments for specific functions (DataLoader or batch)
        func_name = self._get_func_name(node.func)
        if func_name == 'DataLoader' and len(node.args) >= 2:
            # batch_size is typically the second positional argument in DataLoader
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                if arg.value > self.THRESHOLD:
                    smells.append(self._create_smell(node.lineno))
        elif func_name == 'batch' and len(node.args) >= 1:
            # batch_size is typically the first positional argument in Dataset.batch()
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                if arg.value > self.THRESHOLD:
                    smells.append(self._create_smell(node.lineno))
        
        return smells
    
    def _create_smell(self, lineno: int) -> Smell:
        """
        Creates a Smell object with the rule's details.
        """
        return Smell(
            rule_id=self.id,
            rule_name=self.name,
            description=self.description,
            penalty=None,
            optimization=self.optimization,
            start_line=lineno
        )
    
    @staticmethod
    def _get_func_name(func: ast.AST) -> Optional[str]:
        """
        Extracts the function or method name from a Call node's func attribute.
        """
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return func.attr
        return None