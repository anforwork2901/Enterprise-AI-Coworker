"""
GeminiAdapter — LangChain-powered implementation of LLMPort for Google Gemini.

It wraps ChatGoogleGenerativeAI behind the LLMPort interface, keeping the domain clean.
"""
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class GeminiAdapter(LLMPort):
    """
    Concrete implementation of LLMPort using Google Gemini via LangChain.
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        if not api_key:
            raise ValueError("Gemini API Key is missing. Please set it in .env")
            
        self._client = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        logger.info("GeminiAdapter initialized with model='%s'", model)

    async def chat(
        self,
        system_prompt: str,
        history: List[Tuple[str, str]],
        user_message: str,
    ) -> str:
        """Generate a plain text response using Gemini."""
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

        parsed_tool_calls = []
        for tc in (response.tool_calls or []):
            parsed_tool_calls.append({
                "name": tc["name"],
                "arguments": tc["args"]
            })

        return {
            "content": response.content,
            "tool_calls": parsed_tool_calls,
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
