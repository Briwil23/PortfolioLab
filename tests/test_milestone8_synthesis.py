from __future__ import annotations

from pathlib import Path

import pytest

from src.research.synthesis import (
    ArtifactDefinition,
    ArtifactRegistry,
    PublicRepositoryContract,
    StrategyDefinition,
    StrategyRegistry,
    SynthesisEngine,
    build_research_synthesis,
)


def test_public_repo_contract_ignores_local_only_snapshot_files():
    contract = PublicRepositoryContract(root=Path(__file__).resolve().parents[1])
    outcome = contract.validate()

    assert outcome["read_only"] is True
    assert outcome["requires_frozen_snapshot"] is False
    assert outcome["status"] == "PUBLIC_CORE_OK"
    assert "data/canonical_factors/canonical_factors_daily.csv" in outcome["local_only_exclusions"]
    assert outcome["is_valid"] is True


def test_artifact_registry_validates_required_public_outputs():
    public_registry = ArtifactRegistry(
        (
            ArtifactDefinition(
                milestone="milestone7",
                label="portfolio_metrics_gross",
                relative_path="results/milestone7_canonical/portfolio_metrics_gross.csv",
                kind="csv",
                required=True,
                description="Gross performance metrics",
            ),
            ArtifactDefinition(
                milestone="milestone7",
                label="stress_analysis",
                relative_path="results/milestone7_canonical/stress_analysis.csv",
                kind="csv",
                required=True,
                description="Stress-period diagnostics",
            ),
        )
    )

    root = Path(__file__).resolve().parents[1]
    validation = public_registry.validate(root)
    assert validation["ok"] is True
    assert validation["artifacts"]["portfolio_metrics_gross"]["exists"] is True
    assert validation["artifacts"]["stress_analysis"]["exists"] is True


def test_strategy_registry_rejects_duplicate_strategy_names():
    registry = StrategyRegistry()
    registry.register(StrategyDefinition("Minimum CVaR", "M7", "primary", "tail-risk"))

    with pytest.raises(ValueError):
        registry.register(StrategyDefinition("Minimum CVaR", "M7", "secondary", "different objective"))


def test_synthesis_engine_is_read_only_and_deterministic(tmp_path: Path):
    engine = SynthesisEngine(root=Path(__file__).resolve().parents[1])
    first = engine.summarize()
    second = engine.summarize()

    assert first == second
    assert first["read_only"] is True
    assert first["primary_anchor"] == "M7"
    assert first["public_repo_contract"]["status"] == "PUBLIC_CORE_OK"
    assert "M7" in first["supplemental_context"] or "M4" in first["supplemental_context"]

    output_path = tmp_path / "synthesis_report.json"
    with pytest.raises(PermissionError):
        engine.write_report(output_path)
    assert not output_path.exists()


def test_build_research_synthesis_returns_public_summary():
    summary = build_research_synthesis(Path(__file__).resolve().parents[1])

    assert summary["status"] == "ready"
    assert summary["metrics"]["primary_strategy"]
    assert summary["public_repo_contract"]["status"] == "PUBLIC_CORE_OK"
    assert summary["artifacts"]["portfolio_metrics_gross"]["exists"] is True
    assert summary["artifacts"]["stress_analysis"]["exists"] is True
