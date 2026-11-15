from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42


@dataclass
class ModelResult:
    name: str
    probabilities: np.ndarray
    predictions: np.ndarray
    threshold: float
    metrics: dict[str, float]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={"Unnamed: 0": "customer_id"})

    required = {"default", "student", "balance", "income"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "customer_id" not in df.columns:
        df.insert(0, "customer_id", np.arange(1, len(df) + 1))

    cleaned = df[["customer_id", "default", "student", "balance", "income"]].copy()
    cleaned["default"] = cleaned["default"].astype(str).str.strip()
    cleaned["student"] = cleaned["student"].astype(str).str.strip()
    cleaned["default_flag"] = (cleaned["default"].str.lower() == "yes").astype(int)
    cleaned["student_flag"] = (cleaned["student"].str.lower() == "yes").astype(int)
    cleaned["balance"] = pd.to_numeric(cleaned["balance"], errors="coerce")
    cleaned["income"] = pd.to_numeric(cleaned["income"], errors="coerce")
    cleaned = cleaned.dropna(subset=["balance", "income", "default_flag", "student_flag"])
    return cleaned.reset_index(drop=True)


def stratified_split(y: np.ndarray, test_size: float = 0.25, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, l2: float = 1e-4, max_iter: int = 100) -> np.ndarray:
    xb = add_intercept(x)
    beta = np.zeros(xb.shape[1])
    reg = np.eye(xb.shape[1]) * l2
    reg[0, 0] = 0.0

    for _ in range(max_iter):
        p = sigmoid(xb @ beta)
        w = np.clip(p * (1.0 - p), 1e-8, None)
        gradient = xb.T @ (p - y) + reg @ beta
        hessian = (xb.T * w) @ xb + reg
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sigmoid(add_intercept(x) @ beta)


def fit_lda(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    classes = np.array([0, 1])
    means = np.array([x[y == cls].mean(axis=0) for cls in classes])
    priors = np.array([np.mean(y == cls) for cls in classes])
    pooled = np.cov(x.T, bias=False)
    if pooled.ndim == 0:
        pooled = np.array([[float(pooled)]])
    pooled += np.eye(pooled.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(pooled)
    return {"means": means, "priors": priors, "inv_cov": inv_cov}


def predict_lda(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    means = model["means"]
    priors = model["priors"]
    inv_cov = model["inv_cov"]
    scores = []
    for cls_idx in range(2):
        mean = means[cls_idx]
        score = x @ inv_cov @ mean - 0.5 * mean @ inv_cov @ mean + math.log(float(priors[cls_idx]))
        scores.append(score)
    scores = np.column_stack(scores)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs[:, 1]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def auc_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(probabilities)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probabilities) + 1)

    sorted_probs = probabilities[order]
    start = 0
    while start < len(sorted_probs):
        end = start + 1
        while end < len(sorted_probs) and sorted_probs[end] == sorted_probs[start]:
            end += 1
        if end - start > 1:
            avg_rank = ranks[order[start:end]].mean()
            ranks[order[start:end]] = avg_rank
        start = end

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (probabilities >= threshold).astype(int)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc_score(y_true, probabilities),
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.01, 0.99, 99)))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        f1 = evaluate(y_true, probabilities, float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def roc_points(y_true: np.ndarray, probabilities: np.ndarray, steps: int = 80) -> list[tuple[float, float]]:
    thresholds = np.linspace(1.0, 0.0, steps)
    points = []
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        tp, tn, fp, fn = confusion_counts(y_true, pred)
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        points.append((fpr, tpr))
    return points


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_bar_chart(path: Path, title: str, labels: list[str], values: list[float], colors: list[str]) -> None:
    width, height = 760, 460
    left, right, top, bottom = 80, 40, 70, 70
    max_value = max(values) * 1.1
    chart_h = height - top - bottom
    chart_w = width - left - right
    bar_w = chart_w / len(values) * 0.55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
    ]
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + (i + 0.5) * chart_w / len(values) - bar_w / 2
        h = value / max_value * chart_h
        y = height - bottom - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial" font-size="16">{value:,.0f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="16">{svg_escape(label)}</text>')
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def save_scatter(path: Path, df: pd.DataFrame) -> None:
    width, height = 820, 560
    left, right, top, bottom = 85, 45, 65, 80
    sample = df.sample(min(len(df), 1800), random_state=RANDOM_SEED)
    x = sample["balance"].to_numpy()
    y = sample["income"].to_numpy()
    x_min, x_max = 0, df["balance"].max()
    y_min, y_max = 0, df["income"].max()

    def sx(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * (width - left - right)

    def sy(v: float) -> float:
        return height - bottom - (v - y_min) / (y_max - y_min) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Balance vs Income by Default Status</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="15">Balance</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="Arial" font-size="15">Income</text>',
        f'<text x="{width-200}" y="78" font-family="Arial" font-size="14" fill="#2f6f9f">Non-default</text>',
        f'<text x="{width-200}" y="100" font-family="Arial" font-size="14" fill="#c43c39">Default</text>',
    ]
    for _, row in sample.iterrows():
        color = "#c43c39" if row["default_flag"] == 1 else "#2f6f9f"
        radius = 3.2 if row["default_flag"] == 1 else 2.1
        opacity = 0.75 if row["default_flag"] == 1 else 0.28
        parts.append(f'<circle cx="{sx(row["balance"]):.2f}" cy="{sy(row["income"]):.2f}" r="{radius}" fill="{color}" opacity="{opacity}"/>')
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def save_boxplot(path: Path, df: pd.DataFrame, column: str, title: str) -> None:
    width, height = 760, 460
    left, right, top, bottom = 90, 50, 70, 70
    groups = [("No", df.loc[df["default"] == "No", column].to_numpy()), ("Yes", df.loc[df["default"] == "Yes", column].to_numpy())]
    max_value = max(float(np.percentile(values, 99)) for _, values in groups) * 1.05
    min_value = 0.0

    def sy(v: float) -> float:
        return height - bottom - (v - min_value) / (max_value - min_value) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
    ]
    for i, (label, values) in enumerate(groups):
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        low, high = np.percentile(values, [5, 95])
        x = left + (i + 1) * (width - left - right) / 3
        box_w = 105
        parts.append(f'<line x1="{x}" y1="{sy(low):.1f}" x2="{x}" y2="{sy(high):.1f}" stroke="#333" stroke-width="2"/>')
        parts.append(f'<line x1="{x-35}" y1="{sy(low):.1f}" x2="{x+35}" y2="{sy(low):.1f}" stroke="#333" stroke-width="2"/>')
        parts.append(f'<line x1="{x-35}" y1="{sy(high):.1f}" x2="{x+35}" y2="{sy(high):.1f}" stroke="#333" stroke-width="2"/>')
        parts.append(f'<rect x="{x-box_w/2}" y="{sy(q3):.1f}" width="{box_w}" height="{sy(q1)-sy(q3):.1f}" fill="#79a7c8" stroke="#222"/>')
        parts.append(f'<line x1="{x-box_w/2}" y1="{sy(med):.1f}" x2="{x+box_w/2}" y2="{sy(med):.1f}" stroke="#c43c39" stroke-width="3"/>')
        parts.append(f'<text x="{x}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="16">Default: {label}</text>')
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def save_roc(path: Path, model_points: dict[str, list[tuple[float, float]]], aucs: dict[str, float]) -> None:
    width, height = 760, 560
    left, right, top, bottom = 80, 180, 65, 75

    def sx(v: float) -> float:
        return left + v * (width - left - right)

    def sy(v: float) -> float:
        return height - bottom - v * (height - top - bottom)

    colors = ["#2f6f9f", "#c43c39", "#4d8b57"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">ROC Curve Comparison</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="#999" stroke-dasharray="5,5"/>',
        f'<text x="{left + (width-left-right)/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="15">False Positive Rate</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="Arial" font-size="15">True Positive Rate</text>',
    ]
    for i, (name, points) in enumerate(model_points.items()):
        color = colors[i % len(colors)]
        d = " ".join(("M" if j == 0 else "L") + f" {sx(fpr):.1f} {sy(tpr):.1f}" for j, (fpr, tpr) in enumerate(points))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-right+20}" y="{top+24+i*28}" font-family="Arial" font-size="14" fill="{color}">{svg_escape(name)} AUC={aucs[name]:.3f}</text>')
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_float(value: float) -> str:
    return f"{value:.4f}"


def build_report(
    path: Path,
    df: pd.DataFrame,
    metrics_rows: list[dict[str, object]],
    feature_summary: pd.DataFrame,
    best_model: str,
) -> None:
    default_count = int(df["default_flag"].sum())
    default_rate = default_count / len(df)
    lines = [
        "# Credit Card Default Prediction - Final Project Report",
        "",
        "## 1. Project Objective",
        "",
        "The goal of this project is to predict whether a credit card customer will default on a monthly payment using annual income, current balance, and student status. The output is an interpretable risk model that can support credit-risk monitoring and customer prioritization.",
        "",
        "## 2. Dataset Overview",
        "",
        f"- Records analyzed: {len(df):,}",
        f"- Default customers: {default_count:,} ({format_pct(default_rate)})",
        f"- Non-default customers: {len(df) - default_count:,} ({format_pct(1 - default_rate)})",
        f"- Student customers: {int(df['student_flag'].sum()):,}",
        f"- Non-student customers: {int((1 - df['student_flag']).sum()):,}",
        "",
        "The low default rate means the dataset is imbalanced. Accuracy should therefore be interpreted with care because a model can be highly accurate while still missing many default cases.",
        "",
        "## 3. Data Preparation",
        "",
        "- Renamed the index column to `customer_id`.",
        "- Converted `default` into binary target `default_flag`.",
        "- Converted `student` into binary predictor `student_flag`.",
        "- Checked and dropped missing numeric values for `balance` and `income`.",
        "- Used a stratified 75% training / 25% testing split to preserve the default-rate distribution.",
        "",
        "## 4. Exploratory Data Analysis",
        "",
        "| Segment | Count | Avg Balance | Median Balance | Avg Income | Median Income | Default Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in feature_summary.iterrows():
        lines.append(
            f"| {row['segment']} | {int(row['count']):,} | {row['avg_balance']:,.2f} | {row['median_balance']:,.2f} | "
            f"{row['avg_income']:,.2f} | {row['median_income']:,.2f} | {format_pct(row['default_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Chart files generated in `outputs/`:",
            "",
            "- `class_distribution.svg`",
            "- `balance_by_default.svg`",
            "- `income_by_default.svg`",
            "- `balance_income_scatter.svg`",
            "- `roc_comparison.svg`",
            "",
            "## 5. Models Built",
            "",
            "Three models were trained and evaluated:",
            "",
            "- Logistic Regression using balance only",
            "- Multiple Logistic Regression using balance, income, and student status",
            "- Linear Discriminant Analysis using balance, income, and student status",
            "",
            "Thresholds were selected on the training set to maximize F1 score, then evaluated on the held-out test set.",
            "",
            "## 6. Model Performance",
            "",
            "| Model | Threshold | Accuracy | Recall | Specificity | Precision | F1 | AUC | TP | FP | TN | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics_rows:
        lines.append(
            f"| {row['model']} | {row['threshold']:.4f} | {format_pct(row['accuracy'])} | {format_pct(row['sensitivity_recall'])} | "
            f"{format_pct(row['specificity'])} | {format_pct(row['precision'])} | {format_float(row['f1'])} | {format_float(row['auc'])} | "
            f"{int(row['tp'])} | {int(row['fp'])} | {int(row['tn'])} | {int(row['fn'])} |"
        )
    lines.extend(
        [
            "",
            "## 7. Recommended Model",
            "",
            f"The recommended model is **{best_model}**, based on the best combination of AUC and F1 score on the test data.",
            "",
            "For business use, the probability score should be treated as a risk-ranking signal rather than an automatic rejection decision. The final threshold should be adjusted based on operational capacity and the cost tradeoff between false positives and missed defaults.",
            "",
            "## 8. Key Findings",
            "",
            "- Balance is the strongest practical predictor of default risk.",
            "- Default is rare in this dataset, so recall and precision are more informative than accuracy alone.",
            "- Income contributes less signal than balance in this dataset.",
            "- Student status matters, but its interpretation should be considered alongside balance and income.",
            "",
            "## 9. Deliverables",
            "",
            "- Cleaned analysis workflow: `src/credit_card_default_project.py`",
            "- Model comparison: `outputs/metrics.csv`",
            "- Customer predictions: `outputs/predictions.csv`",
            "- Visual outputs: `outputs/*.svg`",
            "- Machine-readable summary: `outputs/summary.json`",
            "",
            "## 10. Next Steps",
            "",
            "- Validate the model on newer customer data.",
            "- Tune the risk threshold according to the financial cost of false positives and false negatives.",
            "- Add more predictors if available, such as payment history, credit utilization, delinquency history, age of account, and bureau variables.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit card default prediction project pipeline")
    parser.add_argument("--data", type=Path, default=Path(r"C:\Users\User\Downloads\Default.xlsx"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    y = df["default_flag"].to_numpy()
    train_idx, test_idx = stratified_split(y)

    feature_sets = {
        "Logistic Regression - Balance": ["balance"],
        "Logistic Regression - Balance Income Student": ["balance", "income", "student_flag"],
        "Linear Discriminant Analysis": ["balance", "income", "student_flag"],
    }

    results: list[ModelResult] = []
    all_probabilities: dict[str, np.ndarray] = {}

    for name, cols in feature_sets.items():
        x = df[cols].to_numpy(dtype=float)
        x_train_raw = x[train_idx]
        x_test_raw = x[test_idx]
        x_train, mean, std = standardize_fit(x_train_raw)
        x_test = standardize_apply(x_test_raw, mean, std)
        x_all = standardize_apply(x, mean, std)
        y_train = y[train_idx]
        y_test = y[test_idx]

        if name.startswith("Logistic"):
            beta = fit_logistic_regression(x_train, y_train)
            train_prob = predict_logistic(beta, x_train)
            test_prob = predict_logistic(beta, x_test)
            all_prob = predict_logistic(beta, x_all)
        else:
            model = fit_lda(x_train, y_train)
            train_prob = predict_lda(model, x_train)
            test_prob = predict_lda(model, x_test)
            all_prob = predict_lda(model, x_all)

        threshold = choose_threshold(y_train, train_prob)
        metrics = evaluate(y_test, test_prob, threshold)
        results.append(ModelResult(name, test_prob, (test_prob >= threshold).astype(int), threshold, metrics))
        all_probabilities[name] = all_prob

    metrics_rows = []
    for result in results:
        row = {"model": result.name}
        row.update(result.metrics)
        metrics_rows.append(row)

    metrics_rows = sorted(metrics_rows, key=lambda r: (r["f1"], r["auc"]), reverse=True)
    best_model = metrics_rows[0]["model"]

    write_csv(output_dir / "metrics.csv", metrics_rows)

    predictions = df[["customer_id", "default", "student", "balance", "income"]].copy()
    for name, probs in all_probabilities.items():
        safe = name.lower().replace(" - ", "_").replace(" ", "_")
        predictions[f"{safe}_probability"] = probs
    best_threshold = next(row["threshold"] for row in metrics_rows if row["model"] == best_model)
    best_prob_col = best_model.lower().replace(" - ", "_").replace(" ", "_") + "_probability"
    predictions["recommended_default_probability"] = predictions[best_prob_col]
    predictions["recommended_prediction"] = np.where(predictions["recommended_default_probability"] >= best_threshold, "Yes", "No")
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    feature_summary = pd.DataFrame(
        [
            {
                "segment": "All customers",
                "count": len(df),
                "avg_balance": df["balance"].mean(),
                "median_balance": df["balance"].median(),
                "avg_income": df["income"].mean(),
                "median_income": df["income"].median(),
                "default_rate": df["default_flag"].mean(),
            },
            {
                "segment": "Default = No",
                "count": int((df["default_flag"] == 0).sum()),
                "avg_balance": df.loc[df["default_flag"] == 0, "balance"].mean(),
                "median_balance": df.loc[df["default_flag"] == 0, "balance"].median(),
                "avg_income": df.loc[df["default_flag"] == 0, "income"].mean(),
                "median_income": df.loc[df["default_flag"] == 0, "income"].median(),
                "default_rate": 0.0,
            },
            {
                "segment": "Default = Yes",
                "count": int((df["default_flag"] == 1).sum()),
                "avg_balance": df.loc[df["default_flag"] == 1, "balance"].mean(),
                "median_balance": df.loc[df["default_flag"] == 1, "balance"].median(),
                "avg_income": df.loc[df["default_flag"] == 1, "income"].mean(),
                "median_income": df.loc[df["default_flag"] == 1, "income"].median(),
                "default_rate": 1.0,
            },
            {
                "segment": "Student = Yes",
                "count": int((df["student_flag"] == 1).sum()),
                "avg_balance": df.loc[df["student_flag"] == 1, "balance"].mean(),
                "median_balance": df.loc[df["student_flag"] == 1, "balance"].median(),
                "avg_income": df.loc[df["student_flag"] == 1, "income"].mean(),
                "median_income": df.loc[df["student_flag"] == 1, "income"].median(),
                "default_rate": df.loc[df["student_flag"] == 1, "default_flag"].mean(),
            },
            {
                "segment": "Student = No",
                "count": int((df["student_flag"] == 0).sum()),
                "avg_balance": df.loc[df["student_flag"] == 0, "balance"].mean(),
                "median_balance": df.loc[df["student_flag"] == 0, "balance"].median(),
                "avg_income": df.loc[df["student_flag"] == 0, "income"].mean(),
                "median_income": df.loc[df["student_flag"] == 0, "income"].median(),
                "default_rate": df.loc[df["student_flag"] == 0, "default_flag"].mean(),
            },
        ]
    )
    feature_summary.to_csv(output_dir / "eda_summary.csv", index=False)

    save_bar_chart(
        output_dir / "class_distribution.svg",
        "Default Class Distribution",
        ["No Default", "Default"],
        [int((df["default_flag"] == 0).sum()), int((df["default_flag"] == 1).sum())],
        ["#2f6f9f", "#c43c39"],
    )
    save_boxplot(output_dir / "balance_by_default.svg", df, "balance", "Balance Distribution by Default")
    save_boxplot(output_dir / "income_by_default.svg", df, "income", "Income Distribution by Default")
    save_scatter(output_dir / "balance_income_scatter.svg", df)

    roc_source = {}
    aucs = {}
    for result in results:
        roc_source[result.name] = roc_points(y[test_idx], result.probabilities)
        aucs[result.name] = result.metrics["auc"]
    save_roc(output_dir / "roc_comparison.svg", roc_source, aucs)

    build_report(output_dir / "final_report.md", df, metrics_rows, feature_summary, str(best_model))

    summary = {
        "rows": int(len(df)),
        "default_rate": float(df["default_flag"].mean()),
        "best_model": best_model,
        "metrics": metrics_rows,
        "outputs": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Project complete. Best model: {best_model}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
