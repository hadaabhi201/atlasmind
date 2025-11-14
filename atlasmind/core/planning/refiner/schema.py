from typing import Any, Dict, List


class PlanRefinementSchema:
    """Validates and provides structure for LLM plan refinement output."""

    # Example schema for prompt embedding
    SCHEMA_EXAMPLE = {
        "entity": "Main subject or entity (string)",
        "intent": "Task objective (string)",
        "filters": {"start_year": 0, "end_year": 0, "category": "string"},
        "topic": "Domain or topic (string)",
        "answer_type": "Expected answer type (e.g., list, count, explanation)",
        "language": "Language of question (string)",
        "confidence": 0.0,
        "specialized_steps": ["Rewritten, context-specific steps"]
    }

    @staticmethod
    def validate(data: Any) -> Dict[str, Any]:
        """Ensures LLM output is structured, safe, and complete."""
        if not isinstance(data, dict):
            return {}

        def clean_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(v).strip() for v in value if v]
            if isinstance(value, str):
                return [value.strip()]
            return []

        validated = {
            "entity": str(data.get("entity", "")).strip(),
            "intent": str(data.get("intent", "")).strip(),
            "filters": data.get("filters", {}),
            "topic": str(data.get("topic", "")).strip(),
            "answer_type": str(data.get("answer_type", "")).strip(),
            "language": str(data.get("language", "English")).strip(),
            "confidence": float(data.get("confidence", 0.0)),
            "specialized_steps": clean_list(data.get("specialized_steps", [])),
        }

        # Fallback if no steps provided
        if not validated["specialized_steps"]:
            validated["specialized_steps"] = ["No refined steps provided."]

        return validated
