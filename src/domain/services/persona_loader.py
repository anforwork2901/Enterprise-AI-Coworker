"""
PersonaLoader — loads and validates NPC persona definitions from YAML files.

Follows Open/Closed Principle: adding a new NPC requires only a new YAML file,
no changes to this loader or any other code.
"""
import logging
from pathlib import Path
from typing import Dict

import yaml

from src.domain.entities.persona import Persona

logger = logging.getLogger(__name__)


class PersonaLoader:
    """
    Responsible solely for loading Persona entities from YAML files.
    Caches loaded personas to avoid repeated disk reads.
    """

    def __init__(self, personas_dir: str) -> None:
        self._dir = Path(personas_dir)
        self._cache: Dict[str, Persona] = {}

    def load(self, persona_id: str) -> Persona:
        """
        Load a Persona by ID. Uses cache on subsequent calls.

        Args:
            persona_id: Matches the YAML filename stem (e.g., "gucci_ceo").

        Returns:
            Persona entity populated from the YAML file.

        Raises:
            FileNotFoundError: If no YAML file exists for the given persona_id.
            ValueError:        If the YAML is missing required fields.
        """
        if persona_id in self._cache:
            return self._cache[persona_id]

        yaml_path = self._dir / f"{persona_id}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Persona file not found: {yaml_path}. "
                f"Expected a YAML file at '{yaml_path}'."
            )

        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        persona = self._parse(persona_id, data)
        self._cache[persona_id] = persona
        logger.info("Loaded persona '%s' from %s", persona_id, yaml_path)
        return persona

    def load_all(self) -> Dict[str, Persona]:
        """Load all YAML personas from the personas directory."""
        personas: Dict[str, Persona] = {}
        for yaml_file in self._dir.glob("*.yaml"):
            pid = yaml_file.stem
            personas[pid] = self.load(pid)
        return personas

    # ─── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(persona_id: str, data: dict) -> Persona:
        required = {"name", "role", "company", "system_prompt"}
        missing = required - data.keys()
        if missing:
            raise ValueError(
                f"Persona '{persona_id}' YAML is missing required fields: {missing}"
            )
        return Persona(
            persona_id=persona_id,
            name=data["name"],
            role=data["role"],
            company=data["company"],
            system_prompt=data["system_prompt"],
            personality_traits=data.get("personality_traits", []),
            hidden_constraints=data.get("hidden_constraints", []),
            knowledge_domains=data.get("knowledge_domains", []),
            tone=data.get("tone", "professional"),
            tools_allowed=data.get("tools_allowed", []),
        )
