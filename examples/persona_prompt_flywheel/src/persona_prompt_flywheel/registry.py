from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PromptPatchRegistry:
    def __init__(self, patches: list[dict[str, Any]]) -> None:
        self.patches = patches

    @classmethod
    def load(cls, path: str | Path) -> "PromptPatchRegistry":
        registry_path = Path(path)
        if registry_path.suffix == ".jsonl":
            patches = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            patches = payload.get("patches", payload if isinstance(payload, list) else [])
        return cls(patches)

    @classmethod
    def from_patches(cls, patches: list[dict[str, Any]]) -> "PromptPatchRegistry":
        return cls(list(patches))

    def get_candidate_patches(self, profile: dict[str, Any], intent: str) -> list[dict[str, Any]]:
        profile_composite = profile.get("profile_composite")
        results = []
        for patch in self.patches:
            if patch.get("profile_intent") not in {intent, "unknown"}:
                continue
            if profile_composite and patch.get("profile_composite") not in {profile_composite, "unknown"}:
                continue
            if patch.get("status") in {"candidate", "low_confidence_candidate"}:
                results.append(patch)
        return results

    def select_best_patch(
        self,
        profile: dict[str, Any],
        intent: str,
        status: str = "candidate",
    ) -> dict[str, Any] | None:
        candidates = [
            patch for patch in self.get_candidate_patches(profile, intent)
            if patch.get("status") == status
        ]
        if not candidates and status == "candidate":
            candidates = self.get_candidate_patches(profile, intent)
        if not candidates:
            return None
        return max(candidates, key=lambda patch: float(patch.get("confidence", 0)))

    def render_runtime_prompt(self, base_prompt: str, patch: dict[str, Any] | None) -> str:
        if not patch:
            return base_prompt
        return (
            f"{base_prompt}\n\n"
            "[Personalized Prompt Patch]\n"
            f"{patch['prompt_patch']}\n\n"
            "Do:\n"
            + "\n".join(f"- {rule}" for rule in patch.get("do_rules", []))
            + "\n\nDon't:\n"
            + "\n".join(f"- {rule}" for rule in patch.get("dont_rules", []))
        )

