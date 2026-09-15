"""
SafetyLevel enum — classifies content safety of user input and NPC output.
"""
from enum import Enum


class SafetyLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"      # Off-topic, mildly inappropriate
    BLOCKED = "blocked"      # Jailbreak attempt, harmful content
