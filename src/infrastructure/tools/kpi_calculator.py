"""
KPICalculator — simulation tool for the NPC to compute basic HR/talent KPIs.

Implements ToolPort. The NPC can invoke this when the user asks about metrics.
"""
from typing import Any, Dict

from src.domain.ports.tool_port import ToolPort


class KPICalculator(ToolPort):
    """
    Calculates basic Talent & Leadership KPIs.
    Supports: talent_coverage_ratio, mobility_rate, competency_adoption_rate.
    """

    @property
    def tool_id(self) -> str:
        return "kpi_calculator"

    @property
    def description(self) -> str:
        return "Calculate HR talent and leadership KPIs from input metrics."

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        metric = parameters.get("metric")
        inputs = parameters.get("inputs", {})

        if metric == "talent_coverage_ratio":
            filled = inputs.get("filled_critical_roles", 0)
            total = inputs.get("total_critical_roles", 1)
            ratio = round(filled / total * 100, 2)
            return {"metric": metric, "result": f"{ratio}%", "status": "green" if ratio >= 80 else "amber"}

        if metric == "mobility_rate":
            moved = inputs.get("employees_moved", 0)
            total = inputs.get("total_employees", 1)
            rate = round(moved / total * 100, 2)
            return {"metric": metric, "result": f"{rate}%"}

        if metric == "competency_adoption_rate":
            adopted = inputs.get("employees_trained", 0)
            total = inputs.get("total_employees", 1)
            rate = round(adopted / total * 100, 2)
            return {"metric": metric, "result": f"{rate}%", "target": "85%"}

        return {"error": f"Unknown metric: {metric}"}

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.tool_id,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["talent_coverage_ratio", "mobility_rate", "competency_adoption_rate"],
                        "description": "The KPI to calculate.",
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Numeric inputs required for the metric formula.",
                    },
                },
                "required": ["metric", "inputs"],
            },
        }
