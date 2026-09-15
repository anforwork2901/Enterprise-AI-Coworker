"""
LLMPort — abstract contract for any LLM provider.

Concrete implementations (OpenAI, Gemini, etc.) live in infrastructure/llm/.
Domain and Application layers ONLY depend on this ABC — never on LangChain directly.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class LLMPort(ABC):
    """
    Abstract interface for language model interactions.
    All LLM adapters MUST implement this contract (Liskov Substitution Principle).
    """

    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        history: List[Tuple[str, str]],
        user_message: str,
    ) -> str:
        """
        Generate a plain-text response from the LLM.

        Args:
            system_prompt:  The persona + context instruction.
            history:        List of (role, content) prior turns.
            user_message:   The current user input.

        Returns:
            Assistant response text.
        """
        ...

    @abstractmethod
    async def chat_with_tools(
        self,
        system_prompt: str,
        history: List[Tuple[str, str]],
        user_message: str,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a response with optional tool/function calls.

        Args:
            system_prompt:  The persona + context instruction.
            history:        Prior conversation turns.
            user_message:   The current user input.
            tools:          List of tool schemas available to the LLM.

        Returns:
            Dict with keys: "content" (str), "tool_calls" (list | None).
        """
        ...
