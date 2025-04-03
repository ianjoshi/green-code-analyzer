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
        var_lineage = {}  # Maps variable names to their parent variable
        origin_device_state = {}  # Tracks device states for variable origins

        def find_root_origin(var_name: str) -> str:
            """
            Follows lineage backward to find the original ancestor variable, with cycle detection.
            """
            origin = var_name
            visited = set()
            while origin in var_lineage:
                if origin in visited:
                    return origin  # Break cycle and return last valid origin
                visited.add(origin)
                origin = var_lineage[origin]
            return origin

        for child in ast.walk(node):
            # Process only single-target assignments
            if (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
            ):
                target_name = child.targets[0].id
                value = child.value

                # Case A: Detect PyTorch device transfer calls
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                ):
                    source_name = value.func.value.id
                    attr_name = value.func.attr

                    # Only process known PyTorch transfer methods
                    current_device = self._parse_device_call(attr_name, value)
                    if current_device is None:
                        continue  # Skip if not a recognized device transfer

                    # Update lineage
                    var_lineage[target_name] = source_name
                    origin = find_root_origin(source_name)

                    # Initialize device state for new origins
                    if origin not in origin_device_state:
                        origin_device_state[origin] = {
                            'last_device': None,
                            'last_line': None
                        }

                    # Check for device switching
                    last_device = origin_device_state[origin]['last_device']
                    if (
                        last_device is not None
                        and current_device is not None
                        and last_device != current_device
                    ):
                        smells.append(
                            Smell(
                                rule_id=self.id,
                                rule_name=self.name,
                                description=self.description,
                                optimization=self.optimization,
                                start_line=child.lineno
                            )
                        )

                    # Update device state
                    origin_device_state[origin]['last_device'] = current_device
                    origin_device_state[origin]['last_line'] = child.lineno

                # Case B: Propagate lineage for binary operations
                elif isinstance(value, ast.BinOp):
                    if isinstance(value.left, ast.Name):
                        src_name = value.left.id
                        if src_name in var_lineage or src_name in origin_device_state:
                            var_lineage[target_name] = src_name

        return smells

    def _parse_device_call(self, attr_name: str, call_node: ast.Call) -> str | None:
        """
        Parses the device string ('cpu', 'cuda', etc.) from a PyTorch call:
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
            return self._extract_device_from_to(call_node)
        return None

    def _extract_device_from_to(self, call_node: ast.Call) -> str | None:
        """
        Extracts a device string from x.to(...) calls.
        """
        # Positional arguments
        if call_node.args:
            first_arg = call_node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                return first_arg.value
            if (
                isinstance(first_arg, ast.Call)
                and isinstance(first_arg.func, ast.Attribute)
                and first_arg.func.attr == 'device'
                and isinstance(first_arg.func.value, ast.Name)
                and first_arg.func.value.id == 'torch'
                and first_arg.args
                and isinstance(first_arg.args[0], ast.Constant)
                and isinstance(first_arg.args[0].value, str)
            ):
                return first_arg.args[0].value

        # Keyword arguments
        for kw in call_node.keywords:
            if kw.arg == 'device':
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    return kw.value.value
                if (
                    isinstance(kw.value, ast.Call)
                    and isinstance(kw.value.func, ast.Attribute)
                    and kw.value.func.attr == 'device'
                    and isinstance(kw.value.func.value, ast.Name)
                    and kw.value.func.value.id == 'torch'
                    and kw.value.args
                    and isinstance(kw.value.args[0], ast.Constant)
                ):
                    return kw.value.args[0].value
        return None