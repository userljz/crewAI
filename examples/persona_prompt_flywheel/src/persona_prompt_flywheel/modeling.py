from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from persona_prompt_flywheel.io_utils import read_yaml, write_json
from persona_prompt_flywheel.modeling_tools import (
    build_design_matrix,
    build_training_dataset,
    train_logistic_regression,
)


def run_modeling(
    feature_matrix: list[dict[str, Any]],
    feature_catalog: dict[str, Any],
    *,
    modeling_config_path: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = Path(output_dir)
    modeling_config = read_yaml(modeling_config_path)
    training_plan = build_training_plan(feature_catalog, modeling_config)
    write_json(output / "training_plan.json", training_plan)

    training_rows = build_training_dataset(
        feature_matrix,
        output / "training_dataset.csv",
        modeling_config.get("training_dataset", {}),
    )
    x, y, feature_names, design_metadata = build_design_matrix(
        training_rows,
        training_plan,
        output / "design_matrix.csv",
    )
    report, coefficients = train_logistic_regression(
        x,
        y,
        feature_names,
        output,
        random_state=int(modeling_config.get("random_state", 42)),
        model_config=modeling_config.get("logistic_regression", {}),
    )
    excluded = Counter(str(row.get("semantic_feedback_label")) for row in feature_matrix if not row.get("usable_for_training"))
    report.update(
        {
            "excluded_label_counts": dict(excluded),
            "design_metadata": design_metadata,
        }
    )
    write_json(output / "model_report.json", report)
    insights = build_segment_insights(feature_matrix, training_rows, coefficients, report, modeling_config)
    write_json(output / "segment_insights.json", insights)
    model_insights = build_model_insights(report, insights, coefficients, feature_catalog, modeling_config)
    write_json(output / "model_insights.json", model_insights)
    _write_model_insights_markdown(output / "model_insights.md", model_insights)
    _write_model_card(output / "model_card.md", report, insights)
    return report, insights


def build_training_plan(feature_catalog: dict[str, Any], modeling_config: dict[str, Any]) -> dict[str, Any]:
    objective = feature_catalog.get("objective_features", {})
    subjective = feature_catalog.get("subjective_features", {})
    plan_config = modeling_config.get("training_plan", {})
    numeric_features = list(objective) + list(subjective)
    return {
        "label_policy": plan_config.get("label_policy"),
        "numeric_features": numeric_features,
        "categorical_features": plan_config.get("categorical_features", []),
        "interaction_terms_enabled": bool(plan_config.get("interaction_terms_enabled", True)),
        "max_interaction_features": int(plan_config.get("max_interaction_features", 5000)),
        "min_category_count_for_interaction": int(plan_config.get("min_category_count_for_interaction", 20)),
        "excluded_labels": plan_config.get("excluded_labels", []),
        "model": plan_config.get("model"),
    }


def build_segment_insights(
    feature_matrix: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    coefficients: list[dict[str, Any]],
    report: dict[str, Any],
    modeling_config: dict[str, Any],
) -> dict[str, Any]:
    insights_config = modeling_config.get("insights", {})
    top_global = int(insights_config.get("top_global_drivers", 5))
    positive_drivers = [_driver(row, insights_config) for row in coefficients if row["effect_direction"] == "positive"][:top_global]
    negative_drivers = [_driver(row, insights_config) for row in coefficients if row["effect_direction"] == "negative"][:top_global]
    min_segment_samples = int(insights_config.get("min_segment_samples_for_insight", 3))
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        by_segment[str(row.get("profile_composite", "unknown"))].append(row)

    segment_insights: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for segment, rows in sorted(by_segment.items()):
        positive = sum(int(row["training_target_positive"]) for row in rows)
        sample_size = len(rows)
        base = {
            "segment": segment,
            "profile_intent": rows[0].get("profile_intent", "unknown") if rows else "unknown",
            "sample_size": sample_size,
            "positive_rate": round(positive / sample_size, 4) if sample_size else 0,
            "model_metrics": report.get("metrics", {}),
        }
        if sample_size < min_segment_samples:
            reason = str(insights_config.get("skipped_segment_reason", "insufficient_samples"))
            skipped.append({"segment": segment, "reason": reason})
            segment_insights.append(
                {**base, "positive_drivers": [], "negative_drivers": [], "skipped": True, "skip_reason": reason}
            )
        else:
            segment_insights.append({**base, "positive_drivers": positive_drivers, "negative_drivers": negative_drivers, "skipped": False, "skip_reason": None})

    if not segment_insights:
        seen_segments = sorted({str(row.get("profile_composite", "unknown")) for row in feature_matrix})
        for segment in seen_segments:
            reason = str(insights_config.get("no_training_rows_reason", "no_training_rows"))
            skipped.append({"segment": segment, "reason": reason})
            segment_insights.append(
                {
                    "segment": segment,
                    "profile_intent": "unknown",
                    "sample_size": 0,
                    "positive_rate": 0,
                    "model_metrics": report.get("metrics", {}),
                    "positive_drivers": [],
                    "negative_drivers": [],
                    "skipped": True,
                    "skip_reason": reason,
                }
            )

    return {
        "global_insights": {
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers,
        },
        "segment_insights": segment_insights,
        "skipped_segments": skipped,
    }


def _driver(row: dict[str, Any], insights_config: dict[str, Any]) -> dict[str, Any]:
    feature = row["feature_name"]
    direction_text = insights_config.get("direction_text", {})
    direction = direction_text.get(row["effect_direction"], row["effect_direction"])
    confidence_labels = insights_config.get("confidence_labels", {})
    low_support = int(insights_config.get("low_support_threshold", 30))
    return {
        "feature": feature,
        "effect": row["coefficient"],
        "confidence": confidence_labels.get("low" if row["support_count"] < low_support else "medium"),
        "interpretation": f"{feature}: {direction}",
    }


def build_model_insights(
    report: dict[str, Any],
    segment_insights: dict[str, Any],
    coefficients: list[dict[str, Any]],
    feature_catalog: dict[str, Any],
    modeling_config: dict[str, Any],
) -> dict[str, Any]:
    """Convert model artifacts into reviewable human-facing insights."""

    n_positive = int(report.get("n_positive") or 0)
    n_non_positive = int(report.get("n_non_positive") or 0)
    n_samples = int(report.get("n_samples") or n_positive + n_non_positive)
    excluded_counts = {str(label): int(count) for label, count in report.get("excluded_label_counts", {}).items()}
    thresholds = modeling_config.get("thresholds", {})
    max_abs_effect = max((float(row.get("abs_coefficient", 0)) for row in coefficients), default=0.0)

    insights_config = modeling_config.get("insights", {})
    positive_drivers = _driver_visual_rows(coefficients, "positive", max_abs_effect, insights_config)
    negative_drivers = _driver_visual_rows(coefficients, "negative", max_abs_effect, insights_config)
    skipped_segments = segment_insights.get("skipped_segments", [])
    top_segment_drivers = int(insights_config.get("top_segment_drivers", 3))
    segment_rows = [
        {
            "segment": item.get("segment", "unknown"),
            "sample_size": item.get("sample_size", 0),
            "positive_rate": item.get("positive_rate", 0),
            "skipped": bool(item.get("skipped")),
            "skip_reason": item.get("skip_reason"),
            "top_positive_drivers": [driver["feature"] for driver in item.get("positive_drivers", [])[:top_segment_drivers]],
            "top_negative_drivers": [driver["feature"] for driver in item.get("negative_drivers", [])[:top_segment_drivers]],
        }
        for item in segment_insights.get("segment_insights", [])
    ]

    return {
        "status": report.get("status"),
        "headline": _model_headline(report, n_positive, n_non_positive, modeling_config),
        "training_readiness": {
            "n_samples": n_samples,
            "n_positive": n_positive,
            "n_non_positive": n_non_positive,
            "positive_rate": round(n_positive / n_samples, 4) if n_samples else 0,
            "class_balance": {
                "positive_feedback": _count_bar(n_positive, n_samples),
                "non_positive_feedback": _count_bar(n_non_positive, n_samples),
            },
            "excluded": {
                "by_label": excluded_counts,
            },
            "thresholds": thresholds,
            "warnings": report.get("warnings", []),
        },
        "model_metrics": report.get("metrics", {}),
        "feature_summary": {
            "features_used": len(report.get("features_used", [])),
            "objective_feature_count": len(feature_catalog.get("objective_features", {})),
            "subjective_feature_count": len(feature_catalog.get("subjective_features", {})),
            "candidate_feature_count": len(feature_catalog.get("candidate_features", [])),
            "interaction_terms_enabled": bool(report.get("interaction_terms_enabled")),
        },
        "driver_insights": {
            "positive": positive_drivers,
            "negative": negative_drivers,
        },
        "segment_checks": {
            "segments": segment_rows,
            "skipped_segments": skipped_segments,
        },
        "review_recommendations": _review_recommendations(
            report=report,
            n_positive=n_positive,
            n_non_positive=n_non_positive,
            excluded_counts=excluded_counts,
            skipped_segments=skipped_segments,
            thresholds=thresholds,
            modeling_config=modeling_config,
        ),
    }


def _model_headline(
    report: dict[str, Any],
    n_positive: int,
    n_non_positive: int,
    modeling_config: dict[str, Any],
) -> str:
    status = report.get("status")
    headlines = modeling_config.get("insights", {}).get("headlines", {})
    template = headlines.get(status) or headlines.get("unknown", "")
    return str(template).format(n_positive=n_positive, n_non_positive=n_non_positive, status=status)


def _driver_visual_rows(
    coefficients: list[dict[str, Any]],
    direction: str,
    max_abs_effect: float,
    insights_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direction_text = insights_config.get("direction_text", {})
    top_visual = int(insights_config.get("top_visual_drivers", 5))
    for row in coefficients:
        if row.get("effect_direction") != direction:
            continue
        coefficient = float(row.get("coefficient", 0))
        feature = str(row.get("feature_name", "unknown"))
        interpretation = direction_text.get(direction, direction)
        rows.append(
            {
                "feature": feature,
                "coefficient": coefficient,
                "support_count": int(row.get("support_count", 0)),
                "feature_type": row.get("feature_type"),
                "effect_bar": _effect_bar(abs(coefficient), max_abs_effect),
                "interpretation": f"{feature}: {interpretation}",
            }
        )
        if len(rows) >= top_visual:
            break
    return rows


def _review_recommendations(
    *,
    report: dict[str, Any],
    n_positive: int,
    n_non_positive: int,
    excluded_counts: dict[str, int],
    skipped_segments: list[dict[str, Any]],
    thresholds: dict[str, Any],
    modeling_config: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    templates = modeling_config.get("insights", {}).get("review_recommendations", {})
    min_positive = int(thresholds.get("global_train_min_positive", 30))
    min_non_positive = int(thresholds.get("global_train_min_non_positive", 30))
    if report.get("status") != "trained":
        recommendations.append(str(templates.get("not_trained", "")).format())
    if n_positive < min_positive or n_non_positive < min_non_positive:
        recommendations.append(
            str(templates.get("need_more_samples", "")).format(
                min_positive=min_positive,
                min_non_positive=min_non_positive,
            )
        )
    excluded_template = str(templates.get("excluded_label", ""))
    for label, count in sorted(excluded_counts.items()):
        if count and excluded_template:
            recommendations.append(excluded_template.format(label=label, count=count))
    if skipped_segments:
        recommendations.append(str(templates.get("skipped_segments", "")).format())
    if not recommendations:
        recommendations.append(str(templates.get("ready", "")).format())
    return [item for item in recommendations if item]


def _count_bar(value: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "-" * width
    filled = round((value / total) * width)
    return "#" * filled + "-" * (width - filled)


def _effect_bar(value: float, max_value: float, width: int = 20) -> str:
    if max_value <= 0:
        return "-" * width
    filled = max(1, round((value / max_value) * width))
    return "#" * filled + "-" * (width - filled)


def _write_model_insights_markdown(path: str | Path, insights: dict[str, Any]) -> None:
    readiness = insights["training_readiness"]
    metrics = insights["model_metrics"]
    feature_summary = insights["feature_summary"]
    lines = [
        "# Model Insights",
        "",
        f"Status: {insights.get('status')}",
        "",
        insights.get("headline", ""),
        "",
        "## Training Data",
        "",
        f"- Samples: {readiness['n_samples']} total, {readiness['n_positive']} positive, {readiness['n_non_positive']} non-positive.",
        f"- Positive rate: {readiness['positive_rate']}",
        f"- Positive feedback: `{readiness['class_balance']['positive_feedback']}`",
        f"- Non-positive feedback: `{readiness['class_balance']['non_positive_feedback']}`",
        "- Excluded labels: "
        + (
            ", ".join(f"{label}={count}" for label, count in sorted(readiness["excluded"]["by_label"].items()))
            or "none"
        ),
        "",
        "## Metrics",
        "",
        *(_metric_lines(metrics) or ["- No metrics available."]),
        "",
        "## Feature Coverage",
        "",
        f"- Features used: {feature_summary['features_used']}",
        f"- Objective / subjective / candidate features: {feature_summary['objective_feature_count']} / {feature_summary['subjective_feature_count']} / {feature_summary['candidate_feature_count']}",
        f"- Interaction terms enabled: {feature_summary['interaction_terms_enabled']}",
        "",
        "## Positive Drivers",
        "",
        *_driver_lines(insights["driver_insights"]["positive"]),
        "",
        "## Negative Drivers",
        "",
        *_driver_lines(insights["driver_insights"]["negative"]),
        "",
        "## Segment Checks",
        "",
        *_segment_lines(insights["segment_checks"]["segments"]),
        "",
        "## Review Recommendations",
        "",
        *[f"- {item}" for item in insights["review_recommendations"]],
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _metric_lines(metrics: dict[str, Any]) -> list[str]:
    return [f"- {name}: {round(float(value), 4)}" for name, value in sorted(metrics.items())]


def _driver_lines(drivers: list[dict[str, Any]]) -> list[str]:
    if not drivers:
        return ["- No driver available."]
    return [
        f"- `{driver['feature']}` coef={driver['coefficient']} support={driver['support_count']} `{driver['effect_bar']}`: {driver['interpretation']}"
        for driver in drivers
    ]


def _segment_lines(segments: list[dict[str, Any]]) -> list[str]:
    if not segments:
        return ["- No segment insight available."]
    return [
        f"- `{segment['segment']}` samples={segment['sample_size']} positive_rate={segment['positive_rate']} skipped={segment['skipped']}"
        for segment in segments
    ]


def _write_model_card(path: str | Path, report: dict[str, Any], insights: dict[str, Any]) -> None:
    positive = ", ".join(driver["feature"] for driver in insights["global_insights"]["positive_drivers"][:3]) or "无"
    negative = ", ".join(driver["feature"] for driver in insights["global_insights"]["negative_drivers"][:3]) or "无"
    text = f"""# Persona Prompt Flywheel Model Card

Status: {report.get("status")}

Samples: {report.get("n_samples")} total, {report.get("n_positive")} positive, {report.get("n_non_positive")} non-positive.

Label policy: semantic relabeling only. `raw_is_positive_feedback` is retained for audit but is not used as the training target.

Positive drivers: {positive}

Negative drivers: {negative}

Warnings: {", ".join(report.get("warnings", [])) or "none"}

Detailed human-readable insights: `model_insights.md`
"""
    Path(path).write_text(text, encoding="utf-8")

