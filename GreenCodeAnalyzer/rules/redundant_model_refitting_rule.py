import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class RedundantModelRefittingRule(BaseRule):
    """
    Detects redundant calls to .fit() on the same dataset without changes in data or hyperparameters.
    """
    
    id = "redundant_model_refitting"
    name = "Redundant Model Refitting"
    description = "Multiple .fit() calls detected on unchanged data, wasting CPU/memory resources."
    optimization = "Reuse the fitted model or use partial_fit() for incremental training."

    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )

        # Track fit calls: {model_var: [(lineno, args_key, data_vars)]}
        self.fit_calls = {}
        
        # Track variables that might be modified: {var_name: last_modified_line}
        self.modified_vars = {}

    def should_apply(self, node) -> bool:
        """
        Applies to attribute calls (e.g., model.fit()) and assignments that might modify data.
        """
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)) or isinstance(node, ast.Assign)

    def apply_rule(self, node) -> list[Smell]:
        """
        Identifies redundant .fit() calls on the same model instance with unchanged data.
        """
        smells = []

        # Handle assignments to track potential data modifications
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.modified_vars[target.id] = node.lineno
            return smells

        # Check if this is a .fit() call
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == "fit" and 
            isinstance(node.func.value, ast.Name)):
            
            model_var = node.func.value.id  # e.g., 'model' in 'model.fit()'
            
            if not node.args:
                return smells
                
            # Extract data variables and normalize args
            data_args = self._args_to_key(node.args)
            data_vars = self._extract_data_vars(node.args)
            
            # Initialize tracking for this model variable
            if model_var not in self.fit_calls:
                self.fit_calls[model_var] = []
            
            # Check previous fit calls for this model
            for prev_call in self.fit_calls[model_var]:
                prev_line, prev_args, _ = prev_call
                if prev_args == data_args:  # Syntactically same args
                    # Check if any data variables were modified since the last fit
                    data_unmodified = all(
                        var not in self.modified_vars or 
                        self.modified_vars[var] < prev_line 
                        for var in data_vars
                    )
                    if data_unmodified:
                        smells.append(Smell(
                            rule_id=self.id,
                            rule_name=self.name,
                            description=self.description,
                            penalty=self.penalty,
                            optimization=self.optimization,
                            start_line=node.lineno
                        ))
                        break
            
            # Record this fit call with data variables
            self.fit_calls[model_var].append((node.lineno, data_args, data_vars))
        
        return smells

    def _args_to_key(self, args: list[ast.AST]) -> str:
        """
        Converts fit() arguments to a normalized string key for comparison.
        """
        key_parts = []
        for arg in args:
            if isinstance(arg, ast.Name):
                key_parts.append(f"var_{arg.id}")
            elif isinstance(arg, ast.Constant):
                key_parts.append(f"const_{arg.value}")
            elif isinstance(arg, ast.Call):
                if isinstance(arg.func, ast.Name):
                    key_parts.append(f"call_{arg.func.id}")
                else:
                    key_parts.append("call_unknown")
            else:
                key_parts.append("unknown")
        return "|".join(key_parts)

    def _extract_data_vars(self, args: list[ast.AST]) -> set[str]:
        """
        Extracts variable names used in the arguments that represent the data.
        """
        data_vars = set()
        for arg in args:
            if isinstance(arg, ast.Name):
                data_vars.add(arg.id)
            elif isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                # For calls like X.copy(), conservatively include the base variable if available
                for sub_arg in arg.args:
                    if isinstance(sub_arg, ast.Name):
                        data_vars.add(sub_arg.id)
        return data_vars

    def process_node(self, node: ast.AST) -> list[Smell]:
        """
        Override to ensure assignments are processed before fit calls in the same scope.
        """
        return super().process_node(node)