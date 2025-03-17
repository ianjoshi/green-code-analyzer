from dataclasses import dataclass
from typing import Optional

@dataclass
class Smell:
    """
    Represents a detected energy code smell.

    Attributes:
    - rule_id (str): The ID of the rule that detected the smell.
    - rule_name (str): The name of the rule that detected the smell.
    - message (str): A description of the detected smell.
    - penalty (float): The penalty applied to the energy score due to the smell, which starts at 100.
    - start_line (int): The starting line number where the smell occurs.
    - end_line (Optional[int]): The ending line number of the smell occurrence (if applicable). If null, assume smell only covers single line.
    """
    rule_id: str
    rule_name: str
    message: str
    penalty: float
    start_line: int
    end_line: Optional[int] = None

    def __str__(self):
        """String representation."""
        line_info = f"Lines {self.start_line}-{self.end_line}" if self.end_line else f"Line {self.start_line}"
        return f"[-{self.penalty}] {self.rule_id} {self.rule_name}: {self.message} ({line_info})"

