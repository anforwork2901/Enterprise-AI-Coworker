"""
Unit tests for SupervisorService — loop detection logic.
LLM dependency is mocked.
"""
import pytest
from unittest.mock import AsyncMock

from src.domain.entities.conversation import Conversation
from src.domain.entities.message import Message
from src.domain.services.supervisor import SupervisorService


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat.return_value = "Have you considered approaching this from the talent strategy angle?"
    return llm


@pytest.fixture
def supervisor(mock_llm) -> SupervisorService:
    return SupervisorService(llm=mock_llm)


@pytest.fixture
def conversation() -> Conversation:
    return Conversation(session_id="test-session", persona_id="gucci_chro")


class TestLoopDetection:
    @pytest.mark.asyncio
    async def test_no_hint_on_fresh_conversation(self, supervisor, conversation):
        hint = await supervisor.evaluate(conversation, "Test goal")
        assert hint is None  # turn_count < minimum

    @pytest.mark.asyncio
    async def test_no_hint_when_messages_vary(self, supervisor, conversation):
        messages = [
            "Tell me about the competency framework.",
            "How does inter-brand mobility work?",
            "What are the APAC challenges?",
        ]
        for i, content in enumerate(messages):
            conversation.add_message(Message(role="user", content=content))
            conversation.add_message(Message(role="assistant", content=f"Response {i}"))

        hint = await supervisor.evaluate(conversation, "Practice HRM strategies")
        assert hint is None

    @pytest.mark.asyncio
    async def test_hint_generated_when_stuck(self, supervisor, conversation, mock_llm):
        repeated = "I don't understand. Can you explain?"
        for _ in range(4):
            conversation.add_message(Message(role="user", content=repeated))
            conversation.add_message(Message(role="assistant", content="Let me clarify..."))

        hint = await supervisor.evaluate(conversation, "Practice HRM strategies")
        assert hint is not None
        assert conversation.is_stuck is True
        mock_llm.chat.assert_called_once()
