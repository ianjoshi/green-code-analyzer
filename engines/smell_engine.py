from collections import OrderedDict
from typing import List
from engines.rule_engine import RuleEngine
from models.smell import Smell

from rules.basic.long_loop_rule import LongLoopRule

from rules.broadcasting_rule import BroadcastingRule
from rules.chain_indexing_rule import ChainIndexingRule
from rules.batch_matrix_multiplication_rule import BatchMatrixMultiplicationRule

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

        # Add data science rules
        self.engine.add_rule(BroadcastingRule())
        self.engine.add_rule(ChainIndexingRule())
        self.engine.add_rule(BatchMatrixMultiplicationRule())

    def collect(self) -> List[Smell]:
        """
        Reads and parses the source file, then applies registered rules to detect code smells.

        :return: A list of Smell objects representing detected inefficiencies.
        """
        with open(self.filepath, "r") as file:
            source_code = file.read()
        
        # Collect all detected smells
        smells = self.engine.analyze(source_code)

        return self.organize_smells_by_line(smells)

    def organize_smells_by_line(self, smells: List[Smell]) -> OrderedDict:
        """
        Reorganizes a list of Smell objects into an OrderedDict where keys are line numbers
        and values are lists of all smells affecting that line.
        
        :param smells: A list of Smell objects from the smell engine.
        :return: An OrderedDict mapping line numbers to lists of Smell objects.
        """
        # Temporary dict to collect smells by line
        line_dict = {}

        for smell in smells:
            end_line = smell.end_line if smell.end_line is not None else smell.start_line
            for line in range(smell.start_line, end_line + 1):
                line_dict.setdefault(line, []).append(smell)

        # Convert the dictionary to an OrderedDict, sorted by line number
        return OrderedDict(sorted(line_dict.items(), key=lambda item: item[0]))