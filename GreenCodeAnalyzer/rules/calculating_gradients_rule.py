import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class CalculatingGradientsRule(BaseRule):
    """
    Detects unnecessary gradient tracking during inference in PyTorch and TensorFlow.
    """

    id = "calculating_gradients"
    name = "Calculating Gradients"
    description = "Unnecessary gradient tracking during inference increases computational cost."
    optimization = "Disable gradient tracking for inference to improve energy efficiency."

    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )

        # Declare model bases to detect custom classes that inherit from them
        self.pytorch_model_bases = {'nn.Module'}
        self.tensorflow_model_bases = {'tf.keras.Model', 'tf.keras.layers.Layer'}

    def should_apply(self, node: ast.AST) -> bool:
        """
        Applies this rule to FunctionDef nodes and at the Module level.
        """
        return isinstance(node, (ast.FunctionDef, ast.Module))

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Creates a GradientTrackingVisitor to walk the AST, then collects and returns any identified smells.
        """
        visitor = self.GradientTrackingVisitor(is_module=isinstance(node, ast.Module))
        visitor.visit(node)
        smells = []

        # PyTorch: If .backward() is never used, but there are model calls outside torch.no_grad()
        if not visitor.has_backward and visitor.pytorch_model_calls_not_in_no_grad:
            for call in visitor.pytorch_model_calls_not_in_no_grad:
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description="Potential unnecessary gradient tracking in PyTorch inference.",
                    optimization="Use `with torch.no_grad():` for model inference in PyTorch.",
                    start_line=call.lineno
                ))

        # TensorFlow: If tape.gradient() is never used, but there are model calls inside tf.GradientTape()
        if not visitor.has_tape_gradient and visitor.tf_model_calls_in_tape:
            for call in visitor.tf_model_calls_in_tape:
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description="Potential unnecessary gradient tracking in TensorFlow inference using tf.GradientTape().",
                    optimization="Avoid using tf.GradientTape() for model inference in TensorFlow.",
                    start_line=call.lineno
                ))

        return smells

    class GradientTrackingVisitor(ast.NodeVisitor):
        """
        AST Visitor that tracks:
          - PyTorch or TF model instantiations (including user-defined classes that inherit).
          - Entry and exit of torch.no_grad() and tf.GradientTape() contexts.
          - Calls to recognized models, and whether they are inside or outside relevant contexts.
          - Whether .backward() or tape.gradient() is called.
        """

        def __init__(self, is_module: bool):
            """
            :param is_module: True if analyzing the top-level module node. 
                              Otherwise, analyzing a function node.
            """
            self.is_module = is_module

            # Sets of variable or attribute names recognized as PyTorch/TF models
            self.pytorch_models = set()
            self.tensorflow_models = set()

            # Stack counters for context managers
            self.inside_no_grad = 0
            self.inside_gradient_tape = 0

            # Calls tracked based on relevant contexts
            self.pytorch_model_calls_not_in_no_grad = []
            self.tf_model_calls_in_tape = []

            # Flags to indicate if .backward() or tape.gradient() has appeared
            self.has_backward = False
            self.has_tape_gradient = False

            super().__init__()

        def visit_Module(self, node: ast.Module):
            """
            Scans the module body for class definitions that inherit from PyTorch/TF bases,
            then visits child nodes.
            """
            for item in node.body:
                if isinstance(item, ast.ClassDef):
                    self.check_class_definition(item)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            """
            Checks if this class inherits from nn.Module, tf.keras.Model, or tf.keras.layers.Layer.
            """
            self.check_class_definition(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            """
            Visits a function definition. 
            Continues visiting children for assignments, calls, etc.
            """
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            """
            Detects PyTorch or TF model creation assigned to variables.
            """
            if isinstance(node.value, ast.Call):
                # Check if the call references a recognized framework or a custom class
                framework = self.infer_framework_from_call(node.value)
                
                # Store the target name or attribute in the appropriate set
                for target in node.targets:
                    if isinstance(target, ast.Name) and framework:
                        if framework == 'pytorch':
                            self.pytorch_models.add(target.id)
                        else:
                            self.tensorflow_models.add(target.id)
                    elif isinstance(target, ast.Attribute) and framework:
                        attr_str = self.get_full_attr_name(target)
                        if framework == 'pytorch':
                            self.pytorch_models.add(attr_str)
                        else:
                            self.tensorflow_models.add(attr_str)

            self.generic_visit(node)

        def visit_With(self, node: ast.With):
            """
            Detects entering and exiting:
              - with torch.no_grad():
              - with tf.GradientTape() as tape:
            Increments counters to track these contexts.
            """
            if self.is_no_grad(node):
                self.inside_no_grad += 1
                self.generic_visit(node)
                self.inside_no_grad -= 1
            elif self.is_gradient_tape(node):
                self.inside_gradient_tape += 1
                self.generic_visit(node)
                self.inside_gradient_tape -= 1
            else:
                self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            """
            Detects calls to:
              - recognized PyTorch model variables (outside no_grad).
              - recognized TF model variables (inside GradientTape).
              - .backward() or tape.gradient().
            """
            if self.is_pytorch_model_call(node) and self.inside_no_grad == 0:
                self.pytorch_model_calls_not_in_no_grad.append(node)

            if self.is_tf_model_call(node) and self.inside_gradient_tape > 0:
                self.tf_model_calls_in_tape.append(node)

            if self.is_backward_call(node):
                self.has_backward = True

            if self.is_tape_gradient_call(node):
                self.has_tape_gradient = True

            self.generic_visit(node)

        def check_class_definition(self, node: ast.ClassDef):
            """
            Checks if a class inherits from known PyTorch/TF bases (e.g. nn.Module),
            then marks that class name as a recognized model class.
            """
            for base in node.bases:
                base_chain = self.get_attribute_chain(base)
                joined = ".".join(base_chain)
                if joined in self.pytorch_model_bases:
                    self.pytorch_models.add(node.name) 
                elif joined in self.tensorflow_model_bases:
                    self.tensorflow_models.add(node.name)

        def infer_framework_from_call(self, call_node: ast.Call) -> str | None:
            """
            Infers if a call is creating a PyTorch or TF model (or a subclass).
            Returns 'pytorch', 'tensorflow', or None if not recognized.
            """
            func_chain = self.get_attribute_chain(call_node.func)
            joined = ".".join(func_chain)

            # If it's a user-defined class recognized as PyTorch or TF
            if any(cls in self.pytorch_models for cls in func_chain):
                return 'pytorch'
            if any(cls in self.tensorflow_models for cls in func_chain):
                return 'tensorflow'

            # If it's part of torch.nn.*
            if "torch" in func_chain or "nn" in func_chain:
                return 'pytorch'

            # If it's part of tf.keras.*
            if func_chain[:2] == ["tf", "keras"]:
                return 'tensorflow'

            return None

        def is_no_grad(self, node: ast.With) -> bool:
            """
            Checks if this 'with' statement is 'with torch.no_grad():'
            """
            if node.items and isinstance(node.items[0].context_expr, ast.Call):
                func_chain = self.get_attribute_chain(node.items[0].context_expr.func)
                return func_chain == ["torch", "no_grad"]
            return False

        def is_gradient_tape(self, node: ast.With) -> bool:
            """
            Checks if this 'with' statement is 'with tf.GradientTape() as tape:'
            """
            if node.items and isinstance(node.items[0].context_expr, ast.Call):
                func_chain = self.get_attribute_chain(node.items[0].context_expr.func)
                return func_chain == ["tf", "GradientTape"]
            return False

        def is_pytorch_model_call(self, node: ast.Call) -> bool:
            """
            Checks if this call is to a known PyTorch model variable/attribute.
            """
            func_name = self.get_full_func_name(node.func)
            return func_name in self.pytorch_models

        def is_tf_model_call(self, node: ast.Call) -> bool:
            """
            Checks if this call is to a known TF model variable/attribute.
            """
            func_name = self.get_full_func_name(node.func)
            return func_name in self.tensorflow_models

        def is_backward_call(self, node: ast.Call) -> bool:
            """
            Checks if the call is something like 'x.backward()'.
            """
            return (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'backward'
            )

        def is_tape_gradient_call(self, node: ast.Call) -> bool:
            """
            Checks if the call is 'tape.gradient(...)'.
            """
            return (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'gradient'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'tape'
            )
        
        def get_attribute_chain(self, node: ast.AST) -> list[str]:
            """
            Gets the full dotted path as a list, from left to right.
            """
            chain = []
            cur = node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.append(cur.id)
            chain.reverse()
            return chain

        def get_full_func_name(self, func_node: ast.AST) -> str:
            """
            Gets a simplified name for a function call.
            """
            if isinstance(func_node, ast.Name):
                return func_node.id
            elif isinstance(func_node, ast.Attribute):
                return self.get_full_attr_name(func_node)
            return ""

        def get_full_attr_name(self, attr_node: ast.Attribute) -> str:
            """
            Gets a simplified name for an attribute.
            """
            chain = []
            cur = attr_node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.append(cur.id)
            chain.reverse()
            return ".".join(chain)
