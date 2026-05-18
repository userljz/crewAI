from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    from crewai.tools import BaseTool
except ImportError:  # pragma: no cover - local smoke tests do not require CrewAI installed.
    BaseTool = object  # type: ignore[assignment]


def build_training_dataset(
    feature_matrix: list[dict[str, Any]],
    output_path: str | Path,
    dataset_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_fields = [str(field) for field in dataset_config.get("metadata_fields", [])]
    for record in feature_matrix:
        target = record.get("training_target_positive")
        if not record.get("usable_for_training") or target not in {0, 1}:
            continue
        row = {field: record.get(field) for field in metadata_fields}
        row.update(
            {
                "training_target_positive": int(target),
                "label_source": record.get("label_source", dataset_config.get("default_label_source")),
                "label_confidence": record.get("label_confidence", 0),
            }
        )
        row.update(record.get("features", {}))
        rows.append(row)
    _write_csv(output_path, rows)
    return rows


def build_design_matrix(
    training_rows: list[dict[str, Any]],
    training_plan: dict[str, Any],
    output_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    numeric_features = training_plan["numeric_features"]
    categorical_features = training_plan["categorical_features"]
    metadata: dict[str, Any] = {"numeric_scalers": {}, "categorical_levels": {}}

    numeric_columns: list[list[float]] = []
    for feature in numeric_features:
        values = np.array([_to_float(row.get(feature)) for row in training_rows], dtype=float)
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) or 1.0
        metadata["numeric_scalers"][feature] = {"mean": mean, "std": std}
        numeric_columns.append(((values - mean) / std).tolist())

    one_hot_columns: list[list[float]] = []
    one_hot_names: list[str] = []
    one_hot_supports: list[int] = []
    for feature in categorical_features:
        levels = sorted({str(row.get(feature, "unknown")) for row in training_rows})
        metadata["categorical_levels"][feature] = levels
        for level in levels:
            one_hot_names.append(f"{feature}={level}")
            column = [1.0 if str(row.get(feature, "unknown")) == level else 0.0 for row in training_rows]
            one_hot_columns.append(column)
            one_hot_supports.append(int(sum(column)))

    columns = numeric_columns + one_hot_columns
    feature_names = list(numeric_features) + one_hot_names

    if training_plan.get("interaction_terms_enabled"):
        max_interactions = int(training_plan.get("max_interaction_features", 5000))
        min_category_count = int(training_plan.get("min_category_count_for_interaction", 1))
        for num_name, num_col in zip(numeric_features, numeric_columns, strict=False):
            for cat_name, cat_col, support in zip(one_hot_names, one_hot_columns, one_hot_supports, strict=False):
                if support < min_category_count:
                    continue
                if len(feature_names) >= max_interactions:
                    break
                columns.append([a * b for a, b in zip(num_col, cat_col, strict=False)])
                feature_names.append(f"{num_name} x {cat_name}")

    x = np.array(columns, dtype=float).T if columns else np.empty((len(training_rows), 0))
    y = np.array([int(row["training_target_positive"]) for row in training_rows], dtype=int)
    _write_design_matrix(output_path, training_rows, x, y, feature_names)
    return x, y, feature_names, metadata


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    *,
    random_state: int = 42,
    model_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_config = model_config or {}
    output = Path(output_dir)
    n_positive = int(y.sum()) if len(y) else 0
    n_non_positive = int(len(y) - n_positive)
    status_labels = model_config.get("status_labels", {})
    if len(y) < 2 or n_positive == 0 or n_non_positive == 0 or x.shape[1] == 0:
        report = {
            "status": status_labels.get("insufficient_data"),
            "label_source_policy": model_config.get("label_source_policy"),
            "n_samples": int(len(y)),
            "n_positive": n_positive,
            "n_non_positive": n_non_positive,
            "features_used": feature_names,
            "interaction_terms_enabled": any(" x " in name for name in feature_names),
            "metrics": {},
            "warnings": [str(model_config.get("insufficient_data_warning", "Need at least two classes and one feature to train."))],
        }
        _write_csv(output / "coefficients.csv", [])
        return report, []

    model = LogisticRegression(
        penalty=str(model_config.get("penalty", "l2")),
        class_weight=model_config.get("class_weight", "balanced"),
        max_iter=int(model_config.get("max_iter", 1000)),
        solver=str(model_config.get("solver", "liblinear")),
        random_state=random_state,
    )
    metrics: dict[str, float] = {}
    min_total = int(model_config.get("min_total_samples_for_trained", 60))
    min_class = int(model_config.get("min_class_samples_for_trained", 30))
    status = (
        status_labels.get("trained_low_confidence")
        if len(y) < min_total or min(n_positive, n_non_positive) < min_class
        else status_labels.get("trained")
    )
    use_holdout = len(y) >= int(model_config.get("min_samples_for_holdout", 20)) and min(
        n_positive, n_non_positive
    ) >= int(
        model_config.get("min_class_samples_for_holdout", 2)
    )
    if use_holdout:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=float(model_config.get("test_size", 0.2)),
            random_state=random_state,
            stratify=y,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        eval_y = y_test
        metrics["accuracy"] = float(accuracy_score(eval_y, pred))
        metrics["precision"] = float(precision_score(eval_y, pred, zero_division=0))
        metrics["recall"] = float(recall_score(eval_y, pred, zero_division=0))
    else:
        model.fit(x, y)

    joblib.dump({"model": model, "feature_names": feature_names}, output / "model.joblib")
    coefficients = interpret_coefficients(
        model.coef_[0],
        feature_names,
        x,
        near_zero_threshold=float(model_config.get("near_zero_coefficient_threshold", 0.05)),
    )
    _write_csv(output / "coefficients.csv", coefficients)
    warnings = []
    if status == status_labels.get("trained_low_confidence"):
        warnings.append(str(model_config.get("low_confidence_warning", "Low sample count; treat results as directional only.")))
    if not use_holdout:
        warnings.append("样本量不足以划分 holdout，未报告泛化指标；系数仅用于离线排查。")

    report = {
        "status": status,
        "label_source_policy": model_config.get("label_source_policy"),
        "n_samples": int(len(y)),
        "n_positive": n_positive,
        "n_non_positive": n_non_positive,
        "features_used": feature_names,
        "interaction_terms_enabled": any(" x " in name for name in feature_names),
        "metrics": metrics,
        "warnings": warnings,
    }
    return report, coefficients


def interpret_coefficients(
    coefficients: np.ndarray,
    feature_names: list[str],
    x: np.ndarray,
    *,
    near_zero_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, coefficient in enumerate(coefficients):
        name = feature_names[index]
        abs_value = abs(float(coefficient))
        if abs_value < near_zero_threshold:
            direction = "near_zero"
        else:
            direction = "positive" if coefficient > 0 else "negative"
        if " x " in name:
            feature_type = "interaction"
        elif "=" in name:
            feature_type = "categorical"
        else:
            feature_type = "numeric"
        rows.append(
            {
                "feature_name": name,
                "feature_type": feature_type,
                "coefficient": round(float(coefficient), 6),
                "effect_direction": direction,
                "abs_coefficient": round(abs_value, 6),
                "standardized": feature_type == "numeric",
                "support_count": int(np.count_nonzero(x[:, index])) if x.size else 0,
            }
        )
    return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)


class SklearnLogisticRegressionTool(BaseTool):  # type: ignore[misc, valid-type]
    name: str = "Sklearn Logistic Regression Sandbox"
    description: str = (
        "Runs a fixed, controlled sklearn logistic regression template. "
        "It does not execute generated code, shell commands, or network calls."
    )

    def _run(self, training_plan_path: str, feature_matrix_path: str, output_dir: str) -> str:
        with Path(training_plan_path).open("r", encoding="utf-8") as file:
            training_plan = json.load(file)
        with Path(feature_matrix_path).open("r", encoding="utf-8") as file:
            feature_matrix = [json.loads(line) for line in file if line.strip()]
        rows = build_training_dataset(feature_matrix, Path(output_dir) / "training_dataset.csv", {})
        x, y, feature_names, _metadata = build_design_matrix(rows, training_plan, Path(output_dir) / "design_matrix.csv")
        report, _coefficients = train_logistic_regression(x, y, feature_names, output_dir)
        return json.dumps(report, ensure_ascii=False)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_design_matrix(
    path: str | Path,
    training_rows: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(training_rows):
        row = {"trace_id": source.get("trace_id"), "training_target_positive": int(y[index])}
        row.update({name: float(x[index, col]) for col, name in enumerate(feature_names)})
        rows.append(row)
    _write_csv(path, rows)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
        else:
            file.write("")

