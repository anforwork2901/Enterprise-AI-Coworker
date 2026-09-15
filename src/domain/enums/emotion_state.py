"""
EmotionState enum — tracks the NPC's emotional disposition toward the user.
Changes based on user behavior across turns (respects State Management requirement).
"""
from enum import Enum


class EmotionState(str, Enum):
    NEUTRAL = "neutral"
    ENGAGED = "engaged"        # User is collaborative and on-track
    PLEASED = "pleased"        # User demonstrates competence
    IMPATIENT = "impatient"    # User is going off-topic or repeating
    GUARDED = "guarded"        # User attempted jailbreak / disrespect
    FRUSTRATED = "frustrated"  # Persistent bad behavior
