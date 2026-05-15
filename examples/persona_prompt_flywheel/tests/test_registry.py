from __future__ import annotations

from persona_prompt_flywheel.registry import PromptPatchRegistry


def test_registry_selects_and_renders_patch() -> None:
    registry = PromptPatchRegistry.from_patches(
        [
            {
                "profile_intent": "yiliaoqita",
                "profile_composite": "yiliaoqita_edit_voice_high_mid",
                "prompt_patch": "先给方向，再问问题。",
                "do_rules": ["先给方向"],
                "dont_rules": ["不要连续追问"],
                "status": "candidate",
                "confidence": 0.7,
            }
        ]
    )

    patch = registry.select_best_patch(
        {"profile_composite": "yiliaoqita_edit_voice_high_mid"},
        intent="yiliaoqita",
    )
    rendered = registry.render_runtime_prompt("base", patch)

    assert patch is not None
    assert "先给方向" in rendered

