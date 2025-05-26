from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class SingleBehaviorRedTeamingMethod(ABC):
    """Base class for red teaming methods that generate test cases for a single behavior."""

    @abstractmethod
    def generate_test_cases_single_behavior(
        self, behavior_dict: Dict, verbose: bool = False, **kwargs
    ) -> Tuple[str, List]:
        """
        Generate test cases for a single behavior.

        Args:
            behavior_dict: Dictionary containing behavior specification
            verbose: Whether to print progress
            **kwargs: Additional arguments

        Returns:
            Tuple containing:
                - Generated test case
                - List of logs/metadata about generation process
        """
        pass
