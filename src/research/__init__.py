"""Read-only research synthesis utilities for PortfolioLab."""

from src.research.synthesis import (
    ArtifactDefinition,
    ArtifactRegistry,
    PublicRepositoryContract,
    StrategyDefinition,
    StrategyRegistry,
    SynthesisEngine,
    build_research_synthesis,
    default_artifact_registry,
    default_strategy_registry,
    evaluate_public_repo_contract,
)

__all__ = [
    "ArtifactDefinition",
    "ArtifactRegistry",
    "PublicRepositoryContract",
    "StrategyDefinition",
    "StrategyRegistry",
    "SynthesisEngine",
    "build_research_synthesis",
    "default_artifact_registry",
    "default_strategy_registry",
    "evaluate_public_repo_contract",
]
