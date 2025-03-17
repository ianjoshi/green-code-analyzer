from typing import List
from engines.rule_engine import RuleEngine
from models.smell import Smell
from rules.basic.long_loop_rule import LongLoopRule

class SmellEngine:
    """
    An engine that analyzes a given Python source file and collects energy-related code smells, based on injected rules.

    Attributes:
        filepath (str): The path to the Python source file to be analyzed.
        engine (RuleEngine): The rule engine that processes the AST and applies rules.
    """

    def __init__(self, filepath: str):
        """
        Initializes the class with the given source file path.

        :param filepath: Path to the Python source file to be analyzed.
        """
        self.filepath = filepath
        self.engine = RuleEngine()

        # Add basic rules
        self.engine.add_rule(LongLoopRule())

    def collect(self) -> List[Smell]:
        """
        Reads and parses the source file, then applies registered rules to detect code smells.

        :return: A list of Smell objects representing detected inefficiencies.
        """
        with open(self.filepath, "r") as file:
            source_code = file.read()
        return self.engine.analyze(source_code)

# Example usage
if __name__ == "__main__":
    collector = SmellEngine("data/long_loop.py")
    smells = collector.collect()
    
    for smell in smells:
        print(smell)