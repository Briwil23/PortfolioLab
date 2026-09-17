"""Read-only synthesis layer for combining certified PortfolioLab milestones.

The M8 design intentionally treats the repository as a read-only evidence
platform. The synthesis engine reads certified artifact files, validates the
public repo contract, and reports provenance without modifying tracked
canonical outputs or requiring the excluded local-only factor snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LOCAL_ONLY_EXCLUSIONS = (
    "data/canonical_factors/canonical_factors_daily.csv",
    "data/canonical_factors/raw",
    "results/milestone5_validation",
    "results/milestone7_validation",
    "results/backtest/HISTORICAL_LIVE_DATA_NOTE.txt",
)

DEFAULT_PUBLIC_REQUIRED_PATHS = (
    "data/canonical",
    "results/milestone4_canonical",
    "results/milestone7_canonical",
    "src",
    "tests",
)


@dataclass(frozen=True)
class ArtifactDefinition:
    """Definition for a public milestone artifact."""

    milestone: str
    label: str
    relative_path: str
    kind: str = "csv"
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class StrategyDefinition:
    """Definition for a research strategy in the synthesis registry."""

    name: str
    milestone: str
    category: str
    objective: str
    source: str = "canonical"
    metrics: tuple[str, ...] = ()


class ArtifactRegistry:
    """Registry of milestone artifacts used by the read-only synthesis engine."""

    def __init__(self, artifacts: Iterable[ArtifactDefinition] | None = None) -> None:
        self._artifacts: dict[str, ArtifactDefinition] = {}
        for artifact in artifacts or ():
            self.register(artifact)

    def register(self, artifact: ArtifactDefinition) -> ArtifactDefinition:
        if artifact.label in self._artifacts:
            raise ValueError(f"Artifact label already registered: {artifact.label}")
        self._artifacts[artifact.label] = artifact
        return artifact

    def __iter__(self):
        return iter(self._artifacts.values())

    def as_dict(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for artifact in self._artifacts.values():
            result[artifact.label] = {
                "milestone": artifact.milestone,
                "path": artifact.relative_path,
                "kind": artifact.kind,
                "required": artifact.required,
                "description": artifact.description,
            }
        return result

    def validate(self, root: str | Path | None = None) -> dict[str, Any]:
        target_root = Path(root) if root is not None else ROOT
        payload: dict[str, Any] = {"ok": True, "missing": [], "artifacts": {}}
        for artifact in sorted(self._artifacts.values(), key=lambda item: item.label):
            path = target_root / artifact.relative_path
            exists = path.exists()
            payload["artifacts"][artifact.label] = {
                "milestone": artifact.milestone,
                "path": artifact.relative_path,
                "exists": exists,
                "required": artifact.required,
            }
            if artifact.required and not exists:
                payload["missing"].append(artifact.relative_path)
        payload["ok"] = not payload["missing"]
        return payload


class StrategyRegistry:
    """Registry of strategy definitions used in synthesis metadata."""

    def __init__(self, strategies: Iterable[StrategyDefinition] | None = None) -> None:
        self._strategies: dict[str, StrategyDefinition] = {}
        for strategy in strategies or ():
            self.register(strategy)

    def register(self, strategy: StrategyDefinition) -> StrategyDefinition:
        normalized = strategy.name.strip()
        if not normalized:
            raise ValueError("Strategy name cannot be empty.")
        if normalized in self._strategies:
            raise ValueError(f"Strategy already registered: {normalized}")
        self._strategies[normalized] = strategy
        return strategy

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))

    def get(self, name: str) -> StrategyDefinition:
        try:
            return self._strategies[name]
        except KeyError as exc:
            raise KeyError(f"Unknown strategy: {name}") from exc

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "milestone": strategy.milestone,
                "category": strategy.category,
                "objective": strategy.objective,
                "source": strategy.source,
                "metrics": list(strategy.metrics),
            }
            for name, strategy in sorted(self._strategies.items())
        }


class PublicRepositoryContract:
    """Public-core contract that intentionally ignores excluded local-only snapshots."""

    def __init__(
        self,
        root: str | Path | None = None,
        required_paths: Iterable[str] | None = None,
        local_only_exclusions: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else ROOT
        self.required_paths = tuple(required_paths or DEFAULT_PUBLIC_REQUIRED_PATHS)
        self.local_only_exclusions = tuple(local_only_exclusions or DEFAULT_LOCAL_ONLY_EXCLUSIONS)

    def validate(self) -> dict[str, Any]:
        missing = [path for path in self.required_paths if not (self.root / path).exists()]
        present_local_only = [path for path in self.local_only_exclusions if (self.root / path).exists()]
        ok = not missing
        return {
            "status": "PUBLIC_CORE_OK" if ok else "PUBLIC_CORE_MISSING",
            "public_repo_status": "PUBLIC_CORE_OK" if ok else "PUBLIC_CORE_MISSING",
            "read_only": True,
            "requires_public_core": True,
            "requires_frozen_snapshot": False,
            "required_paths": list(self.required_paths),
            "missing": missing,
            "local_only_exclusions": list(self.local_only_exclusions),
            "local_only_present": present_local_only,
            "is_valid": ok,
            "repo_root": str(self.root),
        }


def default_artifact_registry() -> ArtifactRegistry:
    return ArtifactRegistry(
        (
            ArtifactDefinition(
                milestone="milestone4",
                label="walk_forward_returns_gross",
                relative_path="results/milestone4_canonical/walk_forward_returns_gross.csv",
                kind="csv",
                required=True,
                description="Gross walk-forward benchmark output used as historical comparison context.",
            ),
            ArtifactDefinition(
                milestone="milestone7",
                label="portfolio_metrics_gross",
                relative_path="results/milestone7_canonical/portfolio_metrics_gross.csv",
                kind="csv",
                required=True,
                description="Primary M7 return, risk, and tail-risk metrics for gross portfolio performance.",
            ),
            ArtifactDefinition(
                milestone="milestone7",
                label="portfolio_metrics_net_10bps",
                relative_path="results/milestone7_canonical/portfolio_metrics_net_10bps.csv",
                kind="csv",
                required=True,
                description="Primary M7 net-of-cost metrics at 10 bps per rebalance.",
            ),
            ArtifactDefinition(
                milestone="milestone7",
                label="stress_analysis",
                relative_path="results/milestone7_canonical/stress_analysis.csv",
                kind="csv",
                required=True,
                description="COVID and 2022 stress-period diagnostics.",
            ),
            ArtifactDefinition(
                milestone="milestone7",
                label="verification",
                relative_path="results/milestone7_canonical/milestone7_empirical_verification.json",
                kind="json",
                required=True,
                description="Certified empirical verification bundle and artifact hash manifest.",
            ),
        )
    )


def default_strategy_registry() -> StrategyRegistry:
    return StrategyRegistry(
        (
            StrategyDefinition(
                name="Minimum CVaR",
                milestone="M7",
                category="primary",
                objective="minimize historical tail risk while preserving diversification",
                source="canonical",
                metrics=("annualized_return", "max_drawdown", "realized_95_cvar_loss"),
            ),
            StrategyDefinition(
                name="SPY",
                milestone="M7",
                category="benchmark",
                objective="single-asset market benchmark",
                source="canonical",
                metrics=("annualized_return", "annualized_volatility", "sharpe"),
            ),
            StrategyDefinition(
                name="Equal Weight",
                milestone="M7",
                category="benchmark",
                objective="naive diversified baseline",
                source="canonical",
                metrics=("annualized_return", "annualized_volatility", "max_drawdown"),
            ),
            StrategyDefinition(
                name="Minimum Variance",
                milestone="M7",
                category="benchmark",
                objective="variance-minimizing baseline",
                source="canonical",
                metrics=("annualized_volatility", "realized_95_var_loss"),
            ),
            StrategyDefinition(
                name="Maximum Sharpe",
                milestone="M7",
                category="benchmark",
                objective="return-seeking risk-adjusted benchmark",
                source="canonical",
                metrics=("sharpe", "annualized_return"),
            ),
            StrategyDefinition(
                name="Combined Robust Max Sharpe λ=0.50 γ=0.10",
                milestone="M7",
                category="benchmark",
                objective="robust max-sharpe baseline with turnover-aware regularization",
                source="canonical",
                metrics=("annualized_return", "sharpe", "max_drawdown"),
            ),
            StrategyDefinition(
                name="Inverse Volatility",
                milestone="M7",
                category="benchmark",
                objective="inverse-volatility benchmark",
                source="canonical",
                metrics=("annualized_return", "annualized_volatility", "sharpe"),
            ),
            StrategyDefinition(
                name="Equal Risk Contribution",
                milestone="M7",
                category="benchmark",
                objective="equal-risk-contribution benchmark",
                source="canonical",
                metrics=("annualized_return", "annualized_volatility", "realized_95_cvar_loss"),
            ),
        )
    )


def _strategy_order(frame: pd.DataFrame) -> list[str]:
    values = frame["strategy"].dropna().astype(str).unique().tolist()
    return sorted(values, key=lambda item: item.lower())


def _read_optional_csv(root: Path, relative_path: str) -> pd.DataFrame | None:
    path = root / relative_path
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_tradeoff_table(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Construct a deterministic trade-off table from compatible M7 certified evidence."""
    target_root = Path(root) if root is not None else ROOT
    gross = pd.read_csv(target_root / "results/milestone7_canonical/portfolio_metrics_gross.csv")
    net = pd.read_csv(target_root / "results/milestone7_canonical/portfolio_metrics_net_10bps.csv")
    turnover = _read_optional_csv(target_root, "results/milestone7_canonical/turnover_analysis.csv")

    strategy_names = _strategy_order(gross)
    rows: list[dict[str, Any]] = []
    for strategy_name in strategy_names:
        gross_row = gross.loc[gross["strategy"].astype(str) == strategy_name].iloc[0]
        net_row = net.loc[net["strategy"].astype(str) == strategy_name].iloc[0]

        row: dict[str, Any] = {
            "strategy": strategy_name,
            "annualized_return_gross": float(gross_row["annualized_return"]),
            "annualized_return_net_10bps": float(net_row["annualized_return"]),
            "return_cost_drag": float(float(gross_row["annualized_return"]) - float(net_row["annualized_return"])),
            "sharpe_gross": float(gross_row["sharpe"]),
            "sharpe_net_10bps": float(net_row["sharpe"]),
            "sharpe_cost_drag": float(float(gross_row["sharpe"]) - float(net_row["sharpe"])),
            "annualized_volatility_gross": float(gross_row["annualized_volatility"]),
            "max_drawdown_gross": float(gross_row["max_drawdown"]),
            "realized_95_var_loss": float(gross_row["realized_95_var_loss"]),
            "realized_95_cvar_loss": float(gross_row["realized_95_cvar_loss"]),
        }

        if turnover is not None and "strategy" in turnover.columns:
            turnover_row = turnover.loc[turnover["strategy"].astype(str) == strategy_name]
            if not turnover_row.empty:
                row["turnover"] = float(turnover_row.iloc[0]["annualized_turnover"])

        rows.append(row)
    return rows


def build_personality_feature_table(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Construct objective feature rows using only compatible M7 evidence."""
    target_root = Path(root) if root is not None else ROOT
    gross = pd.read_csv(target_root / "results/milestone7_canonical/portfolio_metrics_gross.csv")
    turnover = _read_optional_csv(target_root, "results/milestone7_canonical/turnover_analysis.csv")
    stress = _read_optional_csv(target_root, "results/milestone7_canonical/stress_analysis.csv")

    rows: list[dict[str, Any]] = []
    for strategy_name in _strategy_order(gross):
        gross_row = gross.loc[gross["strategy"].astype(str) == strategy_name].iloc[0]
        row: dict[str, Any] = {
            "strategy": strategy_name,
            "annualized_return": float(gross_row["annualized_return"]),
            "annualized_volatility": float(gross_row["annualized_volatility"]),
            "sharpe": float(gross_row["sharpe"]),
            "max_drawdown": float(gross_row["max_drawdown"]),
            "realized_95_var_loss": float(gross_row["realized_95_var_loss"]),
            "realized_95_cvar_loss": float(gross_row["realized_95_cvar_loss"]),
        }

        if turnover is not None and "strategy" in turnover.columns:
            turnover_row = turnover.loc[turnover["strategy"].astype(str) == strategy_name]
            if not turnover_row.empty:
                row["turnover"] = float(turnover_row.iloc[0]["annualized_turnover"])

        if stress is not None and "strategy" in stress.columns:
            covid = stress.loc[(stress["strategy"].astype(str) == strategy_name) & (stress["stress_period"].astype(str) == "COVID")]
            if not covid.empty:
                covid_row = covid.iloc[0]
                row["COVID_cumulative_return"] = float(covid_row["cumulative_return"])
                row["COVID_max_drawdown"] = float(covid_row["maximum_drawdown"])
            two = stress.loc[(stress["strategy"].astype(str) == strategy_name) & (stress["stress_period"].astype(str) == "2022")]
            if not two.empty:
                two_row = two.iloc[0]
                row["2022_cumulative_return"] = float(two_row["cumulative_return"])
                row["2022_max_drawdown"] = float(two_row["maximum_drawdown"])

        rows.append(row)
    return rows


def build_supplemental_evidence(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Build supplemental evidence records from certified public M3/M4/M5 artifacts."""
    target_root = Path(root) if root is not None else ROOT
    records: list[dict[str, Any]] = []

    def append_record(milestone: str, strategy: str, evidence_type: str, metric: str, value: Any, source_artifact: str, comparison_limitation: str, status: str = "CERTIFIED") -> None:
        records.append(
            {
                "milestone": milestone,
                "strategy": strategy,
                "evidence_type": evidence_type,
                "metric": metric,
                "value": value,
                "source_artifact": source_artifact,
                "status": status,
                "comparison_limitation": comparison_limitation,
            }
        )

    m3_turnover = _read_optional_csv(target_root, "results/milestone3_canonical/turnover_analysis.csv")
    if m3_turnover is not None and "strategy" in m3_turnover.columns and "annualized_approx_turnover" in m3_turnover.columns:
        for _, row in m3_turnover.iterrows():
            append_record(
                "M3",
                str(row["strategy"]),
                "turnover",
                "annualized_approx_turnover",
                float(row["annualized_approx_turnover"]),
                "results/milestone3_canonical/turnover_analysis.csv",
                "Supplemental only; not merged into M7 primary metric columns.",
            )

    m3_concentration = _read_optional_csv(target_root, "results/milestone3_canonical/concentration_analysis.csv")
    if m3_concentration is not None and "strategy" in m3_concentration.columns and "mean_hhi" in m3_concentration.columns:
        for _, row in m3_concentration.iterrows():
            append_record(
                "M3",
                str(row["strategy"]),
                "concentration",
                "mean_hhi",
                float(row["mean_hhi"]),
                "results/milestone3_canonical/concentration_analysis.csv",
                "Supplemental only; not merged into M7 primary metric columns.",
            )

    m4_turnover = _read_optional_csv(target_root, "results/milestone4_canonical/turnover_analysis.csv")
    if m4_turnover is not None and "strategy" in m4_turnover.columns and "annualized_approx_turnover" in m4_turnover.columns:
        for _, row in m4_turnover.iterrows():
            append_record(
                "M4",
                str(row["strategy"]),
                "turnover",
                "annualized_approx_turnover",
                float(row["annualized_approx_turnover"]),
                "results/milestone4_canonical/turnover_analysis.csv",
                "Supplemental only; not merged into M7 primary metric columns.",
            )

    m4_stress = _read_optional_csv(target_root, "results/milestone4_canonical/stress_period_analysis.csv")
    if m4_stress is not None and "strategy" in m4_stress.columns and "stress_period" in m4_stress.columns:
        for _, row in m4_stress.iterrows():
            append_record(
                "M4",
                str(row["strategy"]),
                "stress",
                "cumulative_return",
                float(row["cumulative_return"]),
                "results/milestone4_canonical/stress_period_analysis.csv",
                "Supplemental only; not merged into M7 primary metric columns.",
            )

    m5_exposure = _read_optional_csv(target_root, "results/milestone5_canonical/rolling_exposure_summary.csv")
    if m5_exposure is not None and not m5_exposure.empty:
        for _, row in m5_exposure.head(12).iterrows():
            append_record(
                "M5",
                str(row["strategy"]),
                "factor_exposure",
                str(row["factor"]),
                float(row["mean"]),
                "results/milestone5_canonical/rolling_exposure_summary.csv",
                "Public M5 factor evidence only; not required by the public synthesis contract.",
                status="PUBLIC_EXPLANATORY",
            )
    else:
        records.append(
            {
                "milestone": "M5",
                "strategy": "N/A",
                "evidence_type": "public_synthesis_status",
                "metric": "M5_SUPPLEMENTAL_PUBLIC_SYNTHESIS",
                "value": "UNAVAILABLE",
                "source_artifact": "results/milestone5_canonical/rolling_exposure_summary.csv",
                "status": "UNAVAILABLE",
                "comparison_limitation": "Public M5 factor evidence not safely consumable without the excluded frozen factor snapshot.",
            }
        )

    return records


def build_provenance_map(root: str | Path | None = None) -> dict[str, Any]:
    """Provide field-level provenance metadata for trade-off, personality, and supplemental evidence."""
    target_root = Path(root) if root is not None else ROOT
    gross = pd.read_csv(target_root / "results/milestone7_canonical/portfolio_metrics_gross.csv")
    net = pd.read_csv(target_root / "results/milestone7_canonical/portfolio_metrics_net_10bps.csv")
    stress = _read_optional_csv(target_root, "results/milestone7_canonical/stress_analysis.csv")
    turnover = _read_optional_csv(target_root, "results/milestone7_canonical/turnover_analysis.csv")

    provenance: dict[str, Any] = {
        "tradeoff_fields": {
            "annualized_return_gross": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "annualized_return"},
            "annualized_return_net_10bps": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_net_10bps.csv", "source_column": "annualized_return"},
            "return_cost_drag": {"status": "DERIVED", "formula": "annualized_return_gross - annualized_return_net_10bps", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv + results/milestone7_canonical/portfolio_metrics_net_10bps.csv"},
            "sharpe_gross": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "sharpe"},
            "sharpe_net_10bps": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_net_10bps.csv", "source_column": "sharpe"},
            "sharpe_cost_drag": {"status": "DERIVED", "formula": "sharpe_gross - sharpe_net_10bps", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv + results/milestone7_canonical/portfolio_metrics_net_10bps.csv"},
            "annualized_volatility_gross": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "annualized_volatility"},
            "max_drawdown_gross": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "max_drawdown"},
            "realized_95_var_loss": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "realized_95_var_loss"},
            "realized_95_cvar_loss": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "realized_95_cvar_loss"},
        },
        "personality_fields": {
            "annualized_return": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "annualized_return"},
            "annualized_volatility": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "annualized_volatility"},
            "sharpe": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "sharpe"},
            "max_drawdown": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "max_drawdown"},
            "realized_95_var_loss": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "realized_95_var_loss"},
            "realized_95_cvar_loss": {"status": "DIRECT", "source_artifact": "results/milestone7_canonical/portfolio_metrics_gross.csv", "source_column": "realized_95_cvar_loss"},
        },
        "supplemental_fields": {
            "M3_turnover": {"status": "CONTEXTUAL", "source_artifact": "results/milestone3_canonical/turnover_analysis.csv"},
            "M3_concentration": {"status": "CONTEXTUAL", "source_artifact": "results/milestone3_canonical/concentration_analysis.csv"},
            "M4_turnover": {"status": "CONTEXTUAL", "source_artifact": "results/milestone4_canonical/turnover_analysis.csv"},
            "M4_stress": {"status": "CONTEXTUAL", "source_artifact": "results/milestone4_canonical/stress_period_analysis.csv"},
            "M5_factor_exposure": {"status": "CONTEXTUAL", "source_artifact": "results/milestone5_canonical/rolling_exposure_summary.csv"},
        },
        "repo_root": str(target_root),
        "gross_rows": int(len(gross)),
        "net_rows": int(len(net)),
        "stress_rows": int(len(stress)) if stress is not None else 0,
        "turnover_rows": int(len(turnover)) if turnover is not None else 0,
    }
    return provenance


class SynthesisEngine:
    """Read-only synthesis engine for certified research summaries."""

    def __init__(
        self,
        root: str | Path | None = None,
        artifact_registry: ArtifactRegistry | None = None,
        strategy_registry: StrategyRegistry | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else ROOT
        self.artifact_registry = artifact_registry or default_artifact_registry()
        self.strategy_registry = strategy_registry or default_strategy_registry()
        self.contract = PublicRepositoryContract(self.root)

    def read_csv(self, relative_path: str) -> pd.DataFrame:
        path = self.root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {relative_path}")
        return pd.read_csv(path)

    def read_json(self, relative_path: str) -> dict[str, Any]:
        path = self.root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {relative_path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_report(self, target: str | Path) -> Path:
        raise PermissionError("SynthesisEngine is read-only and cannot write artifact outputs.")

    def summarize(self) -> dict[str, Any]:
        public_contract = self.contract.validate()
        registry_state = self.artifact_registry.validate(self.root)
        metric_frame = self.read_csv("results/milestone7_canonical/portfolio_metrics_gross.csv")
        stress_frame = self.read_csv("results/milestone7_canonical/stress_analysis.csv")

        if "strategy" not in metric_frame.columns:
            raise ValueError("portfolio_metrics_gross.csv is missing the required 'strategy' column.")

        best_row = metric_frame.sort_values(["annualized_return", "sharpe"], ascending=[False, False]).iloc[0].to_dict()
        primary_anchor = {
            "milestone": "M7",
            "strategy": str(best_row["strategy"]),
            "annualized_return": float(best_row["annualized_return"]),
            "sharpe": float(best_row["sharpe"]),
            "max_drawdown": float(best_row["max_drawdown"]),
            "realized_95_cvar_loss": float(best_row["realized_95_cvar_loss"]),
        }

        tradeoff_table = build_tradeoff_table(self.root)
        personality_features = build_personality_feature_table(self.root)
        supplemental_evidence = build_supplemental_evidence(self.root)
        provenance = build_provenance_map(self.root)

        stress_periods = sorted(stress_frame["stress_period"].unique().tolist()) if not stress_frame.empty else []
        available_strategies = [str(value) for value in metric_frame["strategy"].dropna().unique().tolist()]
        summary = {
            "read_only": True,
            "primary_anchor": "M7",
            "supplemental_context": ["M4", "M3", "M5"],
            "provenance": {
                "repo_root": str(self.root),
                "artifact_registry": self.artifact_registry.as_dict(),
                "strategy_registry": self.strategy_registry.as_dict(),
                "public_contract": public_contract,
                "tradeoff_fields": provenance["tradeoff_fields"],
                "personality_fields": provenance["personality_fields"],
                "supplemental_fields": provenance["supplemental_fields"],
            },
            "metrics": {
                "primary_strategy": primary_anchor["strategy"],
                "primary_anchor": primary_anchor,
                "stress_periods": stress_periods,
                "available_strategies": available_strategies,
            },
            "tradeoff_table": tradeoff_table,
            "personality_features": personality_features,
            "supplemental_evidence": supplemental_evidence,
            "m5_public_data_dependency": "PUBLIC_EXPLANATORY_AVAILABLE" if any(item["milestone"] == "M5" and item["status"] == "PUBLIC_EXPLANATORY" for item in supplemental_evidence) else "UNAVAILABLE",
            "artifacts": registry_state["artifacts"],
            "public_repo_contract": public_contract,
            "strategy_registry": self.strategy_registry.as_dict(),
            "read_only_guard": "no writes to tracked canonical outputs are permitted",
            "status": "ready",
        }
        return summary


def evaluate_public_repo_contract(root: str | Path | None = None) -> dict[str, Any]:
    """Public convenience wrapper for repo validation without requiring frozen snapshots."""
    contract = PublicRepositoryContract(root=root)
    return contract.validate()


def build_research_synthesis(root: str | Path | None = None) -> dict[str, Any]:
    """Construct a deterministic read-only synthesis summary for the public project."""
    engine = SynthesisEngine(root=root)
    return engine.summarize()


def _default_engine(root: str | Path | None = None) -> SynthesisEngine:
    return SynthesisEngine(root=root)


__all__ = [
    "ArtifactDefinition",
    "ArtifactRegistry",
    "PublicRepositoryContract",
    "StrategyDefinition",
    "StrategyRegistry",
    "SynthesisEngine",
    "build_research_synthesis",
    "build_tradeoff_table",
    "build_personality_feature_table",
    "build_supplemental_evidence",
    "build_provenance_map",
    "default_artifact_registry",
    "default_strategy_registry",
    "evaluate_public_repo_contract",
]
