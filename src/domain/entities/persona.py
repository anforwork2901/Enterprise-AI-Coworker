"""
Persona entity — the core identity definition of an NPC.

This is a pure data class with no external dependencies.
Loaded from YAML files via PersonaLoader (Open/Closed Principle:
new NPC = new YAML, no code changes required).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Persona:
    """
    Immutable NPC identity definition.

    Attributes:
        persona_id:          Unique identifier (matches YAML filename stem).
        name:                Display name of the NPC.
        role:                Job title / role in the simulation.
        company:             Company or organization context.
        system_prompt:       Core LLM system instruction for this persona.
        personality_traits:  List of personality descriptors (e.g., "decisive", "reserved").
        hidden_constraints:  Behavioral constraints invisible to user
                             (e.g., "never discusses competitor brands").
        knowledge_domains:   Topics this NPC has authority over (used for RAG filtering).
        tone:                Communication style (e.g., "formal", "direct", "empathetic").
        tools_allowed:       Tool IDs this NPC is permitted to call.
    """
    persona_id: str
    name: str
    role: str
    company: str
    system_prompt: str
    personality_traits: List[str] = field(default_factory=list)
    hidden_constraints: List[str] = field(default_factory=list)
    knowledge_domains: List[str] = field(default_factory=list)
    tone: str = "professional"
    tools_allowed: List[str] = field(default_factory=list)
