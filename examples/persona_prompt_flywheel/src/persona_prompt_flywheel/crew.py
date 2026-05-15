from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from persona_prompt_flywheel.controlled_tools import (
    FeatureExtractionTool,
    FeedbackLabelingTool,
    ModelingTool,
    PromptPatchGenerationTool,
    RunSummaryTool,
    build_initial_state,
)
from persona_prompt_flywheel.io_utils import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CREWAI_SRC = REPO_ROOT / "lib" / "crewai" / "src"
if CREWAI_SRC.exists() and str(CREWAI_SRC) not in sys.path:
    sys.path.insert(0, str(CREWAI_SRC))

try:  # pragma: no cover - local smoke tests use the controlled runner directly.
    from crewai import Agent, Crew, Process, Task
    from crewai.project import CrewBase, agent, crew, task
except Exception:  # pragma: no cover - keep the offline pipeline usable without CrewAI deps.
    Agent = Crew = Process = Task = None  # type: ignore[assignment]

    def CrewBase(cls: type[Any]) -> type[Any]:  # type: ignore[misc]
        return cls

    def agent(func: Any) -> Any:
        return func

    def task(func: Any) -> Any:
        return func

    def crew(func: Any) -> Any:
        return func


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PersonaPromptFlywheelControlledCrew:
    """CrewAI-style orchestrator that executes only approved deterministic tools."""

    agents_config_path = PROJECT_ROOT / "config" / "agents.yaml"
    tasks_config_path = PROJECT_ROOT / "config" / "tasks.yaml"

    def __init__(
        self,
        *,
        api: Any,
        input_path: str | Path,
        output_dir: str | Path,
        pipeline_config_path: str | Path,
        label_config_path: str | Path,
        feature_config_path: str | Path,
        modeling_config_path: str | Path,
    ) -> None:
        self.state = build_initial_state(
            input_path=input_path,
            output_dir=output_dir,
            pipeline_config_path=pipeline_config_path,
            label_config_path=label_config_path,
            feature_config_path=feature_config_path,
            modeling_config_path=modeling_config_path,
            api=api,
        )
        self.agents_config = read_yaml(self.agents_config_path)
        self.tasks_config = read_yaml(self.tasks_config_path)
        self.steps = [
            ("feedback_labeling_task", "feedback_labeling_agent", FeedbackLabelingTool()),
            ("feature_extraction_task", "feature_discovery_agent", FeatureExtractionTool()),
            ("modeling_task", "modeling_code_agent", ModelingTool()),
            ("prompt_patch_generation_task", "prompt_patch_generator_agent", PromptPatchGenerationTool()),
        ]

    def kickoff(self) -> dict[str, Any]:
        total = len(self.steps)
        for index, (task_name, agent_name, tool) in enumerate(self.steps, start=1):
            task_config = dict(self.tasks_config.get(task_name, {}))
            task_config.update(
                {
                    "name": task_name,
                    "agent": agent_name,
                    "controlled_tool": tool.name,
                    "agent_config": self.agents_config.get(agent_name, {}),
                }
            )
            tool.run(self.state, index=index, total=total, task_config=task_config)
        return RunSummaryTool().run(self.state)


@CrewBase
class PersonaPromptFlywheelCrew:
    """CrewAI semantic wrapper plus controlled-tool execution entrypoint."""

    agents_config = str(PROJECT_ROOT / "config" / "agents.yaml")
    tasks_config = str(PROJECT_ROOT / "config" / "tasks.yaml")

    @agent
    def feedback_labeling_agent(self) -> Any:
        return Agent(config=self.agents_config["feedback_labeling_agent"], verbose=True) if Agent else None

    @agent
    def feature_discovery_agent(self) -> Any:
        return Agent(config=self.agents_config["feature_discovery_agent"], verbose=True) if Agent else None

    @agent
    def modeling_code_agent(self) -> Any:
        return Agent(config=self.agents_config["modeling_code_agent"], verbose=True) if Agent else None

    @agent
    def prompt_patch_generator_agent(self) -> Any:
        return Agent(config=self.agents_config["prompt_patch_generator_agent"], verbose=True) if Agent else None

    @task
    def feedback_labeling_task(self) -> Any:
        return Task(config=self.tasks_config["feedback_labeling_task"]) if Task else None

    @task
    def feature_extraction_task(self) -> Any:
        return Task(config=self.tasks_config["feature_extraction_task"]) if Task else None

    @task
    def modeling_task(self) -> Any:
        return Task(config=self.tasks_config["modeling_task"]) if Task else None

    @task
    def prompt_patch_generation_task(self) -> Any:
        return Task(config=self.tasks_config["prompt_patch_generation_task"]) if Task else None

    @crew
    def crew(self) -> Any:
        if not Crew:
            return None
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=True)
