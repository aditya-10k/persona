"""
Persona modeling package for global and situational styles,
LLM inference interfaces, and persona packaging.
"""

from src.persona.style import GlobalStyleProfile, SituationalStyleProfile, StyleProfileBuilder

__all__ = [
    "GlobalStyleProfile",
    "SituationalStyleProfile",
    "StyleProfileBuilder",
]
