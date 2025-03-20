from dataclasses import dataclass
from typing import Optional

@dataclass
class Smell:
    """
    Represents a detected energy code smell.

    Attributes:
        - rule_id (str): The ID of the rule that detected the smell.
        - rule_name (str): The name of the rule that detected the smell.
        - description (str): A description of the detected smell.
        - penalty (float): The penalty applied to the energy score due to the smell, which starts at 100.
        - start_line (int): The starting line number where the smell occurs.
        - end_line (Optional[int]): The ending line number of the smell occurrence (if applicable).
          If null, assume smell only covers single line.
        - optimization (Optional[str]): Possible solution or solutions for the energy code smell, if available.
    """
    rule_id: str
    rule_name: str
    description: str
    penalty: float
    start_line: int
    end_line: Optional[int] = None
    optimization: Optional[str] = None

    def __str__(self):
        """String representation."""
        line_info = f"Lines {self.start_line}-{self.end_line}" if self.end_line else f"Line {self.start_line}"
        optimization_info = f", Optimization: {self.optimization}" if self.optimization else ""
        return (f"Rule ID: {self.rule_id}, Rule Name: {self.rule_name}, "
                f"Description: {self.description}, Penalty: {self.penalty:.2f}"
                f"{optimization_info}, "
                f"Affected Line(s): {line_info}")