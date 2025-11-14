import json
from typing import Any, Dict
from atlasmind.core.planning.base_plan import PlanTemplate
from atlasmind.core.planning.refiner.prompt_templates import build_plan_prompt
from atlasmind.core.planning.refiner.schema import PlanRefinementSchema


class LLMPlanRefiner:
    """Uses an LLM to enrich a base PlanTemplate with metadata and context-specific steps."""

    def __init__(self, llm, logger):
        """
        Args:
            llm: Async LLM client implementing .acomplete(prompt: str)
            logger: Logger instance for structured tracing.
        """
        self.llm = llm
        self.logger = logger

    async def refine(self, question: str, base_plan: PlanTemplate) -> PlanTemplate:
        """Refine plan using LLM: extract metadata, specialize steps, validate output."""

        prompt = build_plan_prompt(
            question=question,
            plan_steps=base_plan.plan_steps,
            tool_name=base_plan.tool.value
        )

        try:
            # --- Step 1: Query the LLM ---
            result = await self.llm.acomplete(prompt)
            response_text = result.text.strip()
            self.logger.info(f"[LLMPlanRefiner] Raw LLM output: {response_text}")

            # --- Step 2: Parse JSON output ---
            data = self._parse_json(response_text)
            if not data:
                self.logger.warning("[LLMPlanRefiner] No valid JSON returned by LLM, skipping enrichment.")
                return base_plan

            # --- Step 3: Validate & normalize fields ---
            validated = PlanRefinementSchema.validate(data)

            # --- Step 4: Merge specialized steps ---
            if validated.get("specialized_steps"):
                base_plan.plan_steps = validated["specialized_steps"]

            # --- Step 5: Merge metadata ---
            metadata_updates = {
                "entity": validated.get("entity"),
                "intent": validated.get("intent"),
                "filters": validated.get("filters"),
                "topic": validated.get("topic"),
                "answer_type": validated.get("answer_type"),
                "language": validated.get("language"),
                "confidence": validated.get("confidence"),
                "question": question,
            }
            
            if base_plan.metadata is None or not isinstance(base_plan.metadata, dict):
                base_plan.metadata = {}

            # Merge valid values only (no None)
            base_plan.metadata.update({k: v for k, v in metadata_updates.items() if v is not None})

            return base_plan

        except Exception as e:
            self.logger.error(f"[LLMPlanRefiner] Failed to refine plan for {base_plan.question}: {e}")
            return base_plan

    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and safely parse JSON block from LLM output."""
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            if start == -1 or end == -1:
                return {}
            json_str = text[start:end]
            return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"[LLMPlanRefiner] JSON parsing error: {e}")
            return {}
