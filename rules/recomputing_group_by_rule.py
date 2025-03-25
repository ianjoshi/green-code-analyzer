import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class RecomputingGroupByRule(BaseRule):
    """
    Detects inefficient multiple groupby calls on the same DataFrame with identical keys.
    """
    
    id = "recomputing_groupby"
    name = "RecomputingGroupByRule"
    description = "Multiple groupby calls on the same DataFrame with identical keys cause inefficient recomputation of groupings."
    optimization = "Compute all required aggregations in a single groupby call using agg() or store the GroupBy object for reuse."
    aggregation_methods = ['sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var']

    def __init__(self):
        super().__init__(
            id=self.id,
            name=self.name,
            description=self.description,
            optimization=self.optimization
        )

        # Dictionary to track seen (DataFrame_name, keys_tuple) and their first line numbers
        self.seen = {}

    def should_apply(self, node: ast.AST) -> bool:
        """
        Determines if the node is a call to an aggregation method on a groupby operation.
        """
        # Check if the node is a Call to an aggregation method (e.g., sum(), mean())
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr not in self.aggregation_methods:
            return False

        # Check if the aggregation is called on a groupby() result
        value = node.func.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
            return False
        if value.func.attr != 'groupby':
            return False

        return True

    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Applies the rule to detect redundant groupby calls on the same DataFrame with identical keys.
        """
        smells = []

        # Extract the groupby call (e.g., df.groupby('key'))
        groupby_call = node.func.value
        df_expr = groupby_call.func.value

        # For simplicity, handle only direct DataFrame names (e.g., 'df'), not complex expressions
        if not isinstance(df_expr, ast.Name):
            return smells
        df_name = df_expr.id

        # Extract grouping keys from groupby arguments
        args = groupby_call.args
        keys_tuple = self._get_groupby_keys(args)
        if keys_tuple is None:
            return smells  # Skip if keys are not simple strings or lists of strings

        # Create a unique key for tracking this groupby operation
        groupby_key = (df_name, keys_tuple)

        # Check if this (DataFrame, keys) combination has been seen before
        if groupby_key in self.seen:
            first_line = self.seen[groupby_key]
            smell_description = (
                f"{self.description} This call repeats a groupby operation "
                f"first used on line {first_line}."
            )
            smells.append(Smell(
                rule_id=self.id,
                rule_name=self.name,
                description=smell_description,
                optimization=self.optimization,
                start_line=node.lineno
            ))
        else:
            # Record the first occurrence with its line number
            self.seen[groupby_key] = node.lineno

        return smells

    def _get_groupby_keys(self, args: list[ast.AST]) -> tuple[str, ...] | None:
        """
        Extracts and normalizes grouping keys from groupby call arguments.
        """
        if not args:
            return None
        arg = args[0]  # Typically, the first argument is the grouping key(s)

        # Case 1: Single string key (e.g., groupby('key'))
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return (arg.value,)

        # Case 2: List of string keys (e.g., groupby(['key1', 'key2']))
        elif isinstance(arg, ast.List):
            keys = []
            for elt in arg.elts:
                if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                    return None  # Skip if any element isn't a string constant
                keys.append(elt.value)
            return tuple(keys)

        return None  # Unsupported key type