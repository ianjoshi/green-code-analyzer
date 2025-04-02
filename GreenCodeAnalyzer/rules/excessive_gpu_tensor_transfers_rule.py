import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class ExcessiveGPUTensorTransfersRule(BaseRule):
    """
    Detects frequent unnecessary data transfers between CPU and GPU in PyTorch code,
    such as oscillating between .cpu() and .cuda() or frequent calls to .to(device=...) 
    with different devices.
    """

    id = "excessive_gpu_transfers"
    name = "Excessive GPU Tensor Transfers"
    description = (
        "Frequent CPU-GPU tensor transfers detected. This increases energy consumption "
        "due to high data movement overhead."
    )
    optimization = (
        "Minimize transfers by keeping tensors on the GPU for consecutive operations "
        "or batching transfers when possible."
    )

    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )

    def should_apply(self, node: ast.AST) -> bool:
        """
        Applies this rule to function definitions, where tensor transfers are likely to occur.
        """
        return isinstance(node, ast.FunctionDef)

    def apply_rule(self, node: ast.FunctionDef) -> list[Smell]:
        """
        Analyzes the body of a function for excessive CPU-GPU tensor transfers by 
        tracking variable lineage and device states.
        """
        smells = []

        # Maps variable names to a "parent" variable that it was derived from
        var_lineage = {}

        # Tracks device states for the origin of each variable lineage
        origin_device_state = {}

        def find_root_origin(var_name: str) -> str:
            """
            Follows lineage backward to find the original ancestor variable.
            """
            origin = var_name
            while origin in var_lineage:
                origin = var_lineage[origin]
            return origin

        for child in ast.walk(node):
            # Only process single assignments
            if (
                isinstance(child, ast.Assign) 
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
            ):
                target_name = child.targets[0].id
                value = child.value

                # Case A: direct device transfers (x.cpu(), x.cuda(), x.to(...))
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                ):
                    source_name = value.func.value.id
                    attr_name = value.func.attr

                    # Mark target's parent
                    var_lineage[target_name] = source_name
                    origin = find_root_origin(source_name)

                    # Initialize state if this origin is new
                    if origin not in origin_device_state:
                        origin_device_state[origin] = {
                            'last_device': None,
                            'last_line': None
                        }

                    # Attempt to parse current device from the call
                    current_device = self._parse_device_call(attr_name, value)

                    # Check device switching
                    last_device = origin_device_state[origin]['last_device']

                    # If there is a known device, and the device has changed, flag a smell
                    if last_device is not None and current_device is not None and last_device != current_device:
                        smells.append(
                            Smell(
                                rule_id=self.id,
                                rule_name=self.name,
                                description=self.description,
                                optimization=self.optimization,
                                start_line=child.lineno
                            )
                        )

                    # Update the device state for this origin
                    if current_device is not None:
                        origin_device_state[origin]['last_device'] = current_device
                        origin_device_state[origin]['last_line'] = child.lineno

                # Case B: operations that propagate lineage
                elif isinstance(value, ast.BinOp):
                    # If it's a binary operation and the left side is a Name, check lineage
                    if isinstance(value.left, ast.Name):
                        src_name = value.left.id
                        if src_name in var_lineage or src_name in origin_device_state:
                            var_lineage[target_name] = src_name

        return smells

    def _parse_device_call(self, attr_name: str, call_node: ast.Call) -> str | None:
        """
        Attempts to parse the device string ('cpu', 'cuda', etc.) from a call:
            - x.cpu()
            - x.cuda()
            - x.to('cuda')
            - x.to(device='cuda')
            - x.to(torch.device('cuda'))
        """
        if attr_name == 'cpu':
            return 'cpu'
        elif attr_name == 'cuda':
            return 'cuda'
        elif attr_name == 'to':
            # Attempt to parse arguments
            recognized_device = self._extract_device_from_to(call_node)
            return recognized_device
        return None

    def _extract_device_from_to(self, call_node: ast.Call) -> str | None:
        """
        Extracts a device string from x.to(...) calls. 
        """
        # Positional arguments
        if call_node.args:
            first_arg = call_node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                # x.to('cuda') or x.to('cpu')
                return first_arg.value
            # x.to(torch.device('cuda'))
            if isinstance(first_arg, ast.Call) and isinstance(first_arg.func, ast.Attribute):
                # Check if it's torch.device('cuda' or 'cpu')
                if first_arg.func.attr == 'device' and isinstance(first_arg.func.value, ast.Name):
                    if first_arg.func.value.id == 'torch' and first_arg.args:
                        if isinstance(first_arg.args[0], ast.Constant) and isinstance(first_arg.args[0].value, str):
                            return first_arg.args[0].value

        # Keyword arguments
        for kw in call_node.keywords:
            if kw.arg == 'device':
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    # device='cuda' or 'cpu'
                    return kw.value.value
                # device=torch.device('cuda')
                if isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Attribute):
                    if (kw.value.func.attr == 'device'
                            and isinstance(kw.value.func.value, ast.Name)
                            and kw.value.func.value.id == 'torch'):
                        if kw.value.args and isinstance(kw.value.args[0], ast.Constant):
                            return kw.value.args[0].value
                        
        return None
