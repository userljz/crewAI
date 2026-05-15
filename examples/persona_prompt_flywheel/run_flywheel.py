#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persona_prompt_flywheel.api_client import ConfigurableAgentAPI, MockAgentAPI, UnifiedAgentAPI
from persona_prompt_flywheel.crew import PersonaPromptFlywheelControlledCrew
from persona_prompt_flywheel.human_review import merge_human_review_labels


def main() -> None:
    args = _parse_args()
    if args.merge_human_labels:
        result = merge_human_review_labels(args.merge_human_labels, args.rules)
        print(
            f"Merged {result['merged_labels']} human label updates into "
            f"{result['label_config_path']} (version {result['version']})."
        )
        return
    run_pipeline(args)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run the flywheel through CrewAI-style orchestration and controlled tools."""

    api: UnifiedAgentAPI = MockAgentAPI() if args.mock_api else ConfigurableAgentAPI()
    crew_runner = PersonaPromptFlywheelControlledCrew(
        api=api,
        input_path=args.input,
        output_dir=args.output,
        pipeline_config_path=args.config,
        label_config_path=args.rules,
        feature_config_path=args.feature_config,
        modeling_config_path=args.modeling_config,
    )
    return crew_runner.kickoff()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run persona prompt flywheel demo.")
    parser.add_argument("--input", default=PROJECT_ROOT / "data" / "sample_interactions.tsv")
    parser.add_argument("--output", default=PROJECT_ROOT.parent.parent / "runs" / "persona_prompt_flywheel_demo")
    parser.add_argument("--config", default=PROJECT_ROOT / "config" / "pipeline_config.yaml")
    parser.add_argument("--rules", default=PROJECT_ROOT / "config" / "feedback_labeling_rules.yaml")
    parser.add_argument("--feature-config", default=PROJECT_ROOT / "config" / "feature_config.yaml")
    parser.add_argument("--modeling-config", default=PROJECT_ROOT / "config" / "modeling_config.yaml")
    parser.add_argument("--mock-api", action="store_true")
    parser.add_argument("--merge-human-labels")
    return parser.parse_args()


if __name__ == "__main__":
    main()
