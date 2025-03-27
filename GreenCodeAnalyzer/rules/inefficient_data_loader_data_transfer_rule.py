import os
import sys

# Add project root to Python path for absolute imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class InefficientDataLoaderDataTransferRule(BaseRule):
    """
    Detects PyTorch DataLoader configurations that may cause inefficient data transfers
    to GPU due to not using pinned memory.
    """
    
    id = "inefficient_data_transfer"
    name = "Inefficient Data Transfer Configuration"
    description = "Refrain from using standard (pageable) CPU memory for large data loads when transferring to GPU. When transferring data from CPU to GPU, pinned (page-locked) memory can speed up and streamline transfers in CUDA. Non-pinned memory can cause additional overhead, stalling the GPU."
    optimization = "Enable pin_memory=True in the PyTorch DataLoader, which can significantly reduce latency for GPU-bound training."
    
    def __init__(self):
        super().__init__(id=self.id,
                         name=self.name, 
                         description=self.description, 
                         optimization=self.optimization)
        self.dataloader_imports = set()  # Track DataLoader imports

    def should_apply(self, node: ast.AST) -> bool:
        # Check imports first
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            self._check_imports(node)
        return isinstance(node, ast.Call)

    def _check_imports(self, node: ast.AST) -> None:
        """Track imports of DataLoader from torch"""
        if isinstance(node, ast.ImportFrom):
            if node.module == 'torch.utils.data':
                for name in node.names:
                    if name.name == 'DataLoader':
                        self.dataloader_imports.add(name.asname or 'DataLoader')
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.name == 'torch.utils.data.DataLoader':
                    self.dataloader_imports.add(name.asname or 'DataLoader')

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        if not self._is_dataloader_usage(node):
            return []

        # Initialize flags for checking parameters
        pin_memory_found = False
        pin_memory_value = False
        
        # Check all keyword arguments
        for keyword in node.keywords:
            if keyword.arg == "pin_memory":
                pin_memory_found = True
                if isinstance(keyword.value, ast.Constant):
                    pin_memory_value = keyword.value.value
        
        # Create smell if pin_memory is missing or False
        if not pin_memory_found or not pin_memory_value:
            return [Smell(
                rule_id=self.id,
                rule_name=self.name,
                description=self.description,
                penalty=self.penalty,
                optimization=self.optimization,
                start_line=node.lineno
            )]
        
        return []

    def _is_dataloader_usage(self, node: ast.Call) -> bool:
        """
        Detects if the node represents usage of torch.utils.data.DataLoader
        """
        if not isinstance(node.func, (ast.Attribute, ast.Name)):
            return False

        # Check for direct DataLoader usage (only if it was imported from torch)
        if isinstance(node.func, ast.Name):
            return node.func.id in self.dataloader_imports

        # Check for torch.utils.data.DataLoader usage
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "DataLoader":
                value = node.func.value
                if (isinstance(value, ast.Attribute) and 
                    value.attr == "data" and 
                    isinstance(value.value, ast.Attribute) and 
                    value.value.attr == "utils" and 
                    isinstance(value.value.value, ast.Name) and 
                    value.value.value.id == "torch"):
                    return True

        return False

if __name__ == "__main__":
    from engines.smell_engine import SmellEngine
    engine = SmellEngine("data/inefficient_data_loader_data_transfer.py")
    smells = engine.collect()
    print(smells)
