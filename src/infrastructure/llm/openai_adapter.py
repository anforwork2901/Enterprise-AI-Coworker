"""
OpenAIAdapter — LangChain-powered implementation of LLMPort.

This is the ONLY file in the entire project that imports langchain-openai.
It wraps ChatOpenAI behind the LLMPort interface, keeping the domain clean.
"""
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMPort):
    """
    Concrete implementation of LLMPort using LangChain's ChatOpenAI.

    Satisfies Liskov Substitution: swappable with any other LLMPort implementation
    (e.g., GeminiAdapter) without any upstream code changes.
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        self._client = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("OpenAIAdapter initialized with model='%s'", model)

    async def chat(
        self,
        system_prompt: str,
        history: List[Tuple[str, str]],
        user_message: str,
    ) -> str:
        """Generate a plain text response using ChatOpenAI."""
        messages = self._build_messages(system_prompt, history, user_message)
        response: AIMessage = await self._client.ainvoke(messages)
        return response.content

    async def chat_with_tools(
        self,
        system_prompt: str,
        history: List[Tuple[str, str]],
        user_message: str,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a response with optional function/tool calling."""
        messages = self._build_messages(system_prompt, history, user_message)
        client_with_tools = self._client.bind_tools(tools)
        response: AIMessage = await client_with_tools.ainvoke(messages)

        return {
            "content": response.content,
            "tool_calls": [
                {
                    "name": tc["name"],
                    "arguments": tc["args"],
                }
                for tc in (response.tool_calls or [])
            ],
        }

    # ─── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        system_prompt: str,
        history: List[Tuple[str, str]],
        user_message: str,
    ) -> list:
        """Convert our internal format to LangChain message objects."""
        messages = [SystemMessage(content=system_prompt)]
        for role, content in history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))
        return messages
