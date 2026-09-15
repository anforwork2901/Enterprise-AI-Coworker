"""
JIRAMock — simulated JIRA ticket lookup tool for NPC use.

Implements ToolPort. Returns fake but realistic JIRA data to keep
the simulation grounded and support "tool use" demonstration.
"""
from typing import Any, Dict

from src.domain.ports.tool_port import ToolPort

# ─── Static fixture data ──────────────────────────────────────────────────────
_MOCK_TICKETS: Dict[str, Dict[str, Any]] = {
    "HRM-001": {
        "id": "HRM-001",
        "title": "Define Gucci Group Competency Framework v2.0",
        "status": "In Progress",
        "assignee": "CHRO Team",
        "priority": "High",
        "due_date": "2025-Q2",
        "description": "Roll out Vision, Entrepreneurship, Passion, Trust competency framework across all brands.",
    },
    "HRM-002": {
        "id": "HRM-002",
        "title": "Inter-Brand Mobility Program Launch",
        "status": "Planned",
        "assignee": "Talent Acquisition",
        "priority": "Medium",
        "due_date": "2025-Q3",
        "description": "Enable structured talent mobility across Gucci, Balenciaga, Saint Laurent.",
    },
    "HRM-003": {
        "id": "HRM-003",
        "title": "Regional Training Needs Assessment — APAC",
        "status": "Blocked",
        "assignee": "Regional EB Manager",
        "priority": "High",
        "due_date": "2025-Q1",
        "description": "Blocked pending alignment on mandatory vs optional training modules.",
    },
}


class JIRAMock(ToolPort):
    """
    Simulated JIRA ticket lookup. Returns fixture data keyed by ticket ID.
    Demonstrates tool use capability without requiring real JIRA credentials.
    """

    @property
    def tool_id(self) -> str:
        return "jira_lookup"

    @property
    def description(self) -> str:
        return "Look up a JIRA ticket by ID to check status, assignee, and details."

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        ticket_id = parameters.get("ticket_id", "").upper()
        ticket = _MOCK_TICKETS.get(ticket_id)
        if ticket:
            return ticket
        return {"error": f"Ticket '{ticket_id}' not found.", "available": list(_MOCK_TICKETS.keys())}

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.tool_id,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The JIRA ticket ID to look up (e.g., HRM-001).",
                    }
                },
                "required": ["ticket_id"],
            },
        }
