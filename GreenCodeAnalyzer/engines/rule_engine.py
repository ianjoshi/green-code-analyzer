import ast
from typing import List, Type
from rules.base_rule import BaseRule
from models.smell import Smell

class RuleEngine:
    """
    A modular engine for detecting energy-related code smells of Python source code.
    """
    def __init__(self, rules: List[Type[BaseRule]] = None):
        """
        Initializes the engine with a list of rules.
        
        :param rules: A list of rules.
        """
        self.rules = rules if rules else []
    
    def add_rule(self, rule: Type[BaseRule]):
        """
        Adds a new rule to the engine.
        
        :param rule: A rule that inherits from BaseRule.
        """
        self.rules.append(rule)
    
    def analyze(self, source_code: str) -> List[Smell]:
        """
        Parses the source code into an AST and applies all injected rules.
        
        :param source_code: The Python source code to analyze.
        :return: A list of detected Smell objects.
        """
        tree = ast.parse(source_code)
        detected_smells = []
        
        # Traverse the AST nodes
        for node in ast.walk(tree):
            # Apply rules to detect smells
            for rule in self.rules:
                smells = rule.process_node(node)
                detected_smells.extend(smells)
        
        return detected_smells