"""Deterministic factor-model registry and validation for Milestone 5 regression mechanics."""

from __future__ import annotations

FACTOR_MODEL_REGISTRY = {
    "CAPM": ["MKT_RF"],
    "FF3": ["MKT_RF", "SMB", "HML"],
    "FF5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
    "FF5_MOM": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


def get_model_factor_labels(model_name: str) -> list[str]:
    """Return the fixed, label-based factor order for a known model."""
    if model_name not in FACTOR_MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Expected one of {sorted(FACTOR_MODEL_REGISTRY)}.")
    return list(FACTOR_MODEL_REGISTRY[model_name])


def validate_model_spec(model_name: str, labels: list[str] | tuple[str, ...]) -> list[str]:
    """Validate a candidate model against its locked specification in label order.

    This function uses label-based alignment only. Metadata columns such as `date`,
    `RF`, and target columns are ignored; required model factors must still be
    present and unique, and any duplicate required labels fail closed.
    """
    expected = get_model_factor_labels(model_name)
    provided = list(labels)
    required_provided = [label for label in provided if label in expected]
    if len(set(required_provided)) != len(required_provided):
        raise ValueError(f"Duplicate factor labels are not allowed for model {model_name!r}: {required_provided!r}.")
    missing = [label for label in expected if label not in required_provided]
    if missing:
        raise ValueError(f"Missing required factors for model {model_name!r}: {missing!r}.")
    return [label for label in expected if label in required_provided]


__all__ = ["FACTOR_MODEL_REGISTRY", "get_model_factor_labels", "validate_model_spec"]
