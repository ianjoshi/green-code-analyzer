import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class ExcessiveGPUTensorTransfersRule(BaseRule):
    """
    Detects frequent unnecessary data transfers between CPU and GPU in PyTorch code.
    """
    id = "excessive_gpu_transfers"
    name = "Excessive GPU Tensor Transfers"
    description = "Frequent CPU-GPU tensor transfers detected. This increases energy consumption due to high data movement overhead."
    optimization = "Minimize transfers by keeping tensors on the GPU for consecutive operations or batching transfers when possible."

    def __init__(self):
        super().__init__(id=self.id,
                         name=self.name,
                         description=self.description,
                         optimization=self.optimization)

    def should_apply(self, node) -> bool:
        """
        Applies to function definitions where tensor transfers might occur.
        """
        return isinstance(node, ast.FunctionDef)

    def apply_rule(self, node: ast.FunctionDef) -> list[Smell]:
        """
        Analyzes a function for excessive CPU-GPU tensor transfers by tracking tensor data flow.
        
        Returns:
            List of Smell objects representing detected violations.
        """
        smells = []
        parent = {} # Maps variable names to their source variable (parent in the tensor lineage)
        tensor_states = {} # Tracks device state for each origin tensor: {origin_name: {'last_device': str, 'last_line': int}}

        def find_origin(var_name: str) -> str:
            """
            Finds the root origin of a variable by following the parent chain.
            """
            origin = var_name
            while origin in parent:
                origin = parent[origin]
            return origin

        # Single pass through the AST
        for child in ast.walk(node):
            if (isinstance(child, ast.Assign) and 
                len(child.targets) == 1 and 
                isinstance(child.targets[0], ast.Name)):
                target_name = child.targets[0].id
                value = child.value

                # Case 1: Handle direct device transfers (e.g., x.cpu(), x.cuda())
                if (isinstance(value, ast.Call) and 
                    isinstance(value.func, ast.Attribute) and 
                    value.func.attr in ('cpu', 'cuda')):
                    if isinstance(value.func.value, ast.Name):
                        source_name = value.func.value.id
                        parent[target_name] = source_name
                        origin_name = find_origin(source_name)
                        current_device = value.func.attr

                        # Initialize state if origin is new
                        if origin_name not in tensor_states:
                            tensor_states[origin_name] = {'last_device': None, 'last_line': None}
                        
                        last_device = tensor_states[origin_name]['last_device']

                        # Detect a device switch (excluding the first transfer)
                        if last_device is not None and last_device != current_device:
                            smells.append(Smell(
                                rule_id=self.id,
                                rule_name=self.name,
                                description=self.description,
                                optimization=self.optimization,
                                start_line=child.lineno
                            ))

                        # Update device state
                        tensor_states[origin_name] = {
                            'last_device': current_device,
                            'last_line': child.lineno
                        }

                # Case 2: Handle operations that propagate a tensor’s identity without changing its device
                elif isinstance(value, ast.BinOp) and isinstance(value.left, ast.Name):
                    source_name = value.left.id
                    # Propagate origin if source is a known tensor
                    if source_name in parent or source_name in tensor_states:
                        parent[target_name] = source_name

        return smells