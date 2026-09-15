"""
ToolPort — abstract contract for any simulation tool an NPC can invoke.

Interface Segregation: kept deliberately minimal so each tool
only needs to implement execute() and get_schema().
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class ToolPort(ABC):
    """
    Abstract interface for NPC-accessible simulation tools.
    Each tool (KPI Calculator, JIRA Mock, etc.) implements this contract.
    """

    @property
    @abstractmethod
    def tool_id(self) -> str:
        """Unique identifier matching the 'tools_allowed' list in Persona."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the tool with given parameters.

        Args:
            parameters: Tool-specific input dict.

        Returns:
            Tool output dict (always serializable).
        """
        ...

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return the JSON Schema of this tool for LLM function-calling.
        Format compatible with OpenAI function spec.
        """
        ...
