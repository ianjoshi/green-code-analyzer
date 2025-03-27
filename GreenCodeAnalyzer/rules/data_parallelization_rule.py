import os
import sys

# Add project root to Python path for absolute imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import ast
from models.smell import Smell
from rules.base_rule import BaseRule


class DataParallelizationRule(BaseRule):
    id = "data_parallel"
    name = "Suboptimal Data Parallelization"
    description = "Usage of torch.nn.DataParallel detected. This can be less efficient than DistributedDataParallel."
    optimization = ("Consider using torch.nn.parallel.DistributedDataParallel instead of torch.nn.DataParallel. "
                   "DDP is more efficient and scales better, even on a single node with multiple GPUs. "
                   "It provides better performance through more efficient communication and gradient synchronization.")

    def __init__(self):
        super().__init__(id=self.id, name=self.name, description=self.description, optimization=self.optimization)
        self.data_parallel_imports = set()  # Track DataParallel imports

    def should_apply(self, node: ast.AST) -> bool:
        # Check imports first
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            self._check_imports(node)
        return isinstance(node, ast.Call)

    def _check_imports(self, node: ast.AST) -> None:
        """Track imports of DataParallel from torch"""
        if isinstance(node, ast.ImportFrom):
            if node.module in ('torch.nn', 'torch.nn.parallel'):
                for name in node.names:
                    if name.name == 'DataParallel':
                        if name.asname:
                            self.data_parallel_imports.add(name.asname)
                        else:
                            self.data_parallel_imports.add('DataParallel')
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.name in ('torch.nn.DataParallel', 'torch.nn.parallel.DataParallel'):
                    if name.asname:
                        self.data_parallel_imports.add(name.asname)
                    else:
                        self.data_parallel_imports.add('DataParallel')

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        if not self._is_data_parallel_usage(node):
            return []
        
        return [Smell(
            rule_id=self.id,
            rule_name=self.name,
            description=self.description,
            optimization=self.optimization,
            start_line=node.lineno
        )]

    def _is_data_parallel_usage(self, node: ast.Call) -> bool:
        """
        Detects if the node represents usage of torch.nn.DataParallel
        """
        if not isinstance(node.func, (ast.Attribute, ast.Name)):
            return False

        # Check for direct DataParallel usage (only if it was imported from torch)
        if isinstance(node.func, ast.Name):
            return node.func.id in self.data_parallel_imports

        # Check for torch.nn.DataParallel or nn.DataParallel usage
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "DataParallel":
                # Get the base object (e.g., torch.nn or nn)
                value = node.func.value
                if isinstance(value, ast.Name) and value.id in ("nn",):
                    return True
                elif (isinstance(value, ast.Attribute) and 
                      isinstance(value.value, ast.Name) and 
                      value.value.id == "torch" and 
                      value.attr == "nn"):
                    return True

        return False

if __name__ == "__main__":
    from engines.smell_engine import SmellEngine
    engine = SmellEngine("data/inefficient_data_parallelization.py")
    smells = engine.collect()
    print(smells)

