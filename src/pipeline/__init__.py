"""
CVE Difficulty Grading & Scenario Expansion Pipeline.

This package implements the pipeline described in docs/cve_difficulty_and_expansion.md:
  1. CVE Classifier  — grades each CVE with difficulty_score and difficulty_tier
  2. CVE Selector    — picks CVEs matching template slots + difficulty tier
  3. Overlay Generator — creates overlay YAML from selected CVEs
  4. Scenario Compiler — merges template + overlay → full PenGym YAML
  5. Curriculum Controller — manages phase transitions during training
  6. Extensible Registry — service/process registry + CVE addition pipeline
"""

from .cve_classifier import CVEClassifier
from .scenario_compiler import (
    ScenarioCompiler, ScenarioPipeline,
    CVESelector, OverlayGenerator,
    generate_template_from_yaml,
)
from .curriculum_controller import (
    CurriculumController, CurriculumConfig, PhaseConfig, FlatController
)
from .extensible_registry import (
    ServiceRegistry, ServiceDefinition, ProcessDefinition,
    CVEAdditionPipeline, TemplateExpander,
)

# SimpleDQNAgent requires torch — import lazily
def _get_simple_dqn_agent():
    from .simple_dqn_agent import SimpleDQNAgent
    return SimpleDQNAgent

__all__ = [
    'CVEClassifier',
    'ScenarioCompiler',
    'ScenarioPipeline',
    'CVESelector',
    'OverlayGenerator',
    'generate_template_from_yaml',
    'CurriculumController',
    'CurriculumConfig',
    'PhaseConfig',
    'FlatController',
    'ServiceRegistry',
    'ServiceDefinition',
    'ProcessDefinition',
    'CVEAdditionPipeline',
    'TemplateExpander',
]
