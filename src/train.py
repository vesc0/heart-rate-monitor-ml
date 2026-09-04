"""Train and evaluate the binary stress classifier.

Evaluation is Leave-One-Subject-Out only. Random k-fold is not reported: windows
overlap by 50% and subjects repeat across folds, which inflates scores without
measuring what the app actually does — generalize to an unseen person.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURES

LABELS = ["Not Stressed", "Stressed"]
# class_weight matters here: stress is a minority of windows, so an unweighted
# model scores well simply by rarely predicting it.
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    # Leaf size is set for a deployable artifact: deeper forests are ~8x larger
    # for the same AUC and worse stress recall.
    "Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, min_samples_leaf=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "SVM (RBF)": CalibratedClassifierCV(
        SVC(C=10, class_weight="balanced", random_state=42), ensemble=False
    ),
}
TREE_MODELS = {"Random Forest", "Extra Trees"}


def pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])


def loso(model, X, y, groups):
    """Out-of-fold predictions and stress probabilities, one subject held out."""
    shared = dict(cv=LeaveOneGroupOut(), groups=groups, n_jobs=-1)
    predicted = cross_val_predict(pipeline(model), X, y, **shared)
    proba = cross_val_predict(pipeline(model), X, y, method="predict_proba", **shared)
    return predicted, proba[:, 1]


def score(y, predicted, proba):
    return {
        "accuracy": accuracy_score(y, predicted),
        "stress_recall": recall_score(y, predicted),
        "macro_f1": f1_score(y, predicted, average="macro"),
        "roc_auc": roc_auc_score(y, proba),
    }


def plot_comparison(results, path):
    frame = pd.DataFrame(results).set_index("model")[["roc_auc", "stress_recall", "macro_f1"]]
    axis = frame.plot.barh(figsize=(9, 4.5), color=["#4c72b0", "#dd8452", "#55a868"])
    axis.set(xlabel="score", xlim=(0, 1))
    axis.set_title("Leave-One-Subject-Out performance", fontweight="bold")
    axis.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_diagnostics(y, predicted, proba, groups, name, outputs):
    _, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y, predicted, normalize="true"), display_labels=LABELS
    ).plot(ax=axes[0], cmap="Blues", colorbar=False, values_format=".2f")
    axes[0].set_title(f"{name} — normalized confusion matrix", fontweight="bold")

    fpr, tpr, _ = roc_curve(y, proba)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, proba):.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate")
    axes[1].set_title("ROC (out-of-fold)", fontweight="bold")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outputs / "confusion_matrix.png", dpi=150)
    plt.close()

    accuracy = accuracy_score(y, predicted)
    per_subject = (
        pd.DataFrame({"subject": groups, "correct": y == predicted})
        .groupby("subject")["correct"].mean().sort_values(ascending=False)
    )
    axis = per_subject.plot.bar(
        figsize=(11, 4.5), color=sns.color_palette("Set2", len(per_subject))
    )
    axis.axhline(accuracy, color="red", linestyle="--", label=f"overall {accuracy:.3f}")
    axis.set(ylabel="accuracy", ylim=(0, 1))
    axis.set_title(f"Per-subject accuracy — {name}", fontweight="bold")
    axis.legend()
    plt.tight_layout()
    plt.savefig(outputs / "per_subject_accuracy.png", dpi=150)
    plt.close()


def plot_importance(model, X, y, path):
    importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring="roc_auc"
    )
    order = np.argsort(importance.importances_mean)
    plt.figure(figsize=(8, 7))
    plt.barh([FEATURES[i] for i in order], importance.importances_mean[order], color="#4c72b0")
    plt.xlabel("drop in ROC AUC when shuffled")
    plt.title("Permutation importance (in-sample)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the stress classifier")
    parser.add_argument("--data", type=Path, default=Path("data/hrv_features.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    X, y, groups = frame[list(FEATURES)].values, frame.stress.values, frame.subject.values
    majority = max(np.mean(y == 0), np.mean(y == 1))
    print(f"{len(frame)} windows, {frame.subject.nunique()} subjects, "
          f"{y.mean():.1%} stress (majority-class accuracy {majority:.3f})\n")

    results, folds = [], {}
    for name, model in MODELS.items():
        predicted, proba = loso(model, X, y, groups)
        folds[name] = (predicted, proba)
        results.append({"model": name, **score(y, predicted, proba)})
        row = results[-1]
        print(f"{name:22s} auc={row['roc_auc']:.3f} recall={row['stress_recall']:.3f} "
              f"macroF1={row['macro_f1']:.3f} acc={row['accuracy']:.3f}")

    best = max(results, key=lambda r: r["roc_auc"])
    name = best["model"]
    predicted, proba = folds[name]
    print(f"\nBest by ROC AUC: {name}\n")
    print(classification_report(y, predicted, target_names=LABELS, digits=3))

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, args.outputs_dir / "model_comparison.png")
    plot_diagnostics(y, predicted, proba, groups, name, args.outputs_dir)

    final = pipeline(MODELS[name]).fit(X, y)
    plot_importance(final, X, y, args.outputs_dir / "feature_importance.png")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "model": final,
        "feature_columns": list(FEATURES),
        "hrv_columns": list(FEATURES),
        "demo_columns": [],
        "demo_defaults": {},
        "binary_labels": {0: LABELS[0], 1: LABELS[1]},
        "model_type": "tree" if name in TREE_MODELS else "other",
        "training_info": {
            "model": name,
            "n_samples": len(frame),
            "n_subjects": int(frame.subject.nunique()),
            "evaluation": "leave-one-subject-out",
            "majority_class_accuracy": float(majority),
            **{k: float(v) for k, v in best.items() if k != "model"},
        },
    }
    joblib.dump(artifacts, args.models_dir / "all_artifacts.joblib")
    print(json.dumps(artifacts["training_info"], indent=2))
    print(f"\nSaved model to {args.models_dir}/ and plots to {args.outputs_dir}/")


if __name__ == "__main__":
    main()
