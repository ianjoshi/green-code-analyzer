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
        self.pytorch_model_bases = {'nn.Module'}
        self.tensorflow_model_bases = {'tf.keras.Model', 'tf.keras.layers.Layer'}

    def should_apply(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.FunctionDef, ast.Module))

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        visitor = self.GradientTrackingVisitor(is_module=isinstance(node, ast.Module))
        visitor.visit(node)
        smells = []

        # PyTorch: Flag model calls outside torch.no_grad() without backward()
        if not visitor.has_backward and visitor.pytorch_model_calls_not_in_no_grad:
            for call in visitor.pytorch_model_calls_not_in_no_grad:
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description="Potential unnecessary gradient tracking in PyTorch inference.",
                    optimization="Use `with torch.no_grad():` for model inference in PyTorch.",
                    start_line=call.lineno
                ))

        # TensorFlow: Flag model calls in tf.GradientTape() without tape.gradient()
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
        def __init__(self, is_module: bool):
            self.is_module = is_module
            self.pytorch_models = set()
            self.tensorflow_models = set()
            self.inside_no_grad = 0
            self.pytorch_model_calls_not_in_no_grad = []
            self.has_backward = False
            self.inside_gradient_tape = 0
            self.tf_model_calls_in_tape = []
            self.has_tape_gradient = False

        def visit_Assign(self, node: ast.Assign):
            """
            Detects PyTorch or TF models assigned to variables, e.g.
                pytorch_model = nn.Linear(...)
                tf_model = tf.keras.Sequential(...)
                tf_model = tf.keras.models.Sequential(...)
            so we can later track calls to them.
            """
            if isinstance(node.value, ast.Call):
                func = node.value.func

                # Detect PyTorch models
                if isinstance(func, ast.Attribute):
                    if (
                        func.attr in {'Linear', 'Conv2d', 'Module'}
                        and isinstance(func.value, ast.Name)
                        and func.value.id == 'nn'
                    ):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.pytorch_models.add(target.id)

                # Detect TensorFlow Keras models
                if self.is_tf_keras_model_creation(func):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.tensorflow_models.add(target.id)

            self.generic_visit(node)

        def visit_With(self, node: ast.With):
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
            if self.is_pytorch_model_call(node) and self.inside_no_grad == 0:
                self.pytorch_model_calls_not_in_no_grad.append(node)
            if self.is_tensorflow_model_call(node) and self.inside_gradient_tape > 0:
                self.tf_model_calls_in_tape.append(node)
            if self.is_backward_call(node):
                self.has_backward = True
            if self.is_tape_gradient_call(node):
                self.has_tape_gradient = True
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            # If it's a nested function (not top-level), only recurse for children
            # but the top-level logic is handled at the module or class level.
            if not self.is_module:
                self.generic_visit(node)

        def is_no_grad(self, node: ast.With) -> bool:
            if node.items and isinstance(node.items[0].context_expr, ast.Call):
                func = node.items[0].context_expr.func
                return (
                    isinstance(func, ast.Attribute)
                    and func.attr == 'no_grad'
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'torch'
                )
            return False

        def is_gradient_tape(self, node: ast.With) -> bool:
            if node.items and isinstance(node.items[0].context_expr, ast.Call):
                func = node.items[0].context_expr.func
                return (
                    isinstance(func, ast.Attribute)
                    and func.attr == 'GradientTape'
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'tf'
                )
            return False

        def is_pytorch_model_call(self, node: ast.Call) -> bool:
            # If node.func is a simple Name, check if it's in our stored model names
            return isinstance(node.func, ast.Name) and node.func.id in self.pytorch_models

        def is_tensorflow_model_call(self, node: ast.Call) -> bool:
            # Same approach for TF: if node.func is a Name, check if in our stored model names
            return isinstance(node.func, ast.Name) and node.func.id in self.tensorflow_models

        def is_backward_call(self, node: ast.Call) -> bool:
            return (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'backward'
            )

        def is_tape_gradient_call(self, node: ast.Call) -> bool:
            return (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'gradient'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'tape'
            )

        def is_tf_keras_model_creation(self, func: ast.AST) -> bool:
            """
            Returns True if the given ast node creates a TF Keras model
            either via tf.keras.Sequential or tf.keras.models.Sequential or .Model.
            We'll walk up the attribute chain looking for 'tf' -> 'keras' -> ...
            and then 'Sequential' / 'Model' as the final attribute.
            """
            # Must be something like: tf.keras.(models.)Sequential(...) or tf.keras.(models.)Model(...)
            # So final attribute is 'Sequential' or 'Model' and the chain must include tf and keras.

            chain = []
            cur = func
            # Walk up the attribute chain, collecting .attr as we go:
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.append(cur.id)
            chain.reverse()  # so the chain is in natural reading order: [tf, keras, models, Sequential]

            if len(chain) < 3:
                return False
            # We need at least: tf, keras, and something like Sequential/Model at the end
            if chain[0] != 'tf' or chain[1] != 'keras':
                return False

            # The final item is either 'Sequential' or 'Model'
            return chain[-1] in {'Sequential', 'Model'}
