import ast
from models.smell import Smell
from rules.base_rule import BaseRule

class ChainIndexingRule(BaseRule):
    """
    Detects chained indexing in Pandas DataFrames that leads to inefficient energy use.
    """
    
    id = "chain_indexing"
    name = "Chain Indexing"
    description = "Chained indexing (e.g., df['one']['two']) triggers multiple Pandas operations, increasing memory and CPU usage."
    optimization = "Use df.loc[:, ('one', 'two')] for a single, efficient operation."

    def __init__(self):
        super().__init__(id=self.id,
                         name=self.name,
                         description=self.description,
                         optimization=self.optimization)
    
    def should_apply(self, node: ast.AST) -> bool:
        """
        Applies to subscription nodes that might represent indexing operations.
        """
        return isinstance(node, ast.Subscript)
    
    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Detects chained indexing patterns in Pandas DataFrame operations.
        """
        smells = []
        
        # Check if this is a chained indexing operation (e.g., df['one']['two'])
        if (isinstance(node, ast.Subscript) and 
            isinstance(node.value, ast.Subscript)):
            
            # Extract the base variable (e.g., 'df' in df['one']['two'])
            current_node = node.value
            while isinstance(current_node, ast.Subscript):
                current_node = current_node.value
            
            # Check if the base is a variable (e.g., 'df', 'dataframe', etc.)
            if isinstance(current_node, ast.Name):
                # Assume it's a DataFrame if it’s followed by chained indexing
                smells.append(Smell(
                    rule_id=self.id,
                    rule_name=self.name,
                    description=self.description,
                    penalty=self.penalty,
                    optimization=self.optimization,
                    start_line=node.lineno
                ))
        
        return smells