import ast
from abc import ABC, abstractmethod
from models.smell import Smell

class BaseRule(ABC):
    """
    Abstract base class for all energy efficiency static analysis rules.
    
    Attributes:
    - id (str): A unique identifier for the rule.
    - name (str): The unique name of the rule.
    - message (str): A default message describing the rule.
    - penalty (float): The penalty applied to the energy score due to the smell, which starts at 100.
    """
    def __init__(self, id: str, name: str, message: str, penalty: float) -> None:
        self.id: str = id
        self.name: str = name
        self.message: str = message
        self.penalty: float = penalty 
    
    @abstractmethod
    def should_apply(self, node: ast.AST) -> bool:
        """
        Determines whether the rule should be applied to the given AST node.
        
        :param node: An individual AST node.
        :return: True if the rule applies to the node, False otherwise.
        """
        pass
    
    @abstractmethod
    def apply_rule(self, node: ast.AST) -> list[Smell]:
        """
        Apply the rule to a single AST node and return a list of detected smells.
        
        :param node: An individual AST node.
        :return: A list of Smell objects representing detected issues.
        """
        pass
    
    def process_node(self, node: ast.AST) -> list[Smell]:
        """
        Checks if the rule should be applied to the node, and if so, applies it.
        
        :param node: An individual AST node.
        :return: A list of Smell objects if the rule is applied, otherwise an empty list.
        """
        if self.should_apply(node):
            return self.apply_rule(node)
        return []
