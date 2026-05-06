"""Train and evaluate text classification models for the Kaggle project."""

from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


# Define the main project folders once so the script can be run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Keep the random state fixed so results are reproducible across runs.
RANDOM_STATE = 42


def validate_input_files() -> tuple[Path, Path, Path]:
    """Check that the expected CSV files exist before training starts."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    sample_path = DATA_DIR / "sample_submission.csv"

    missing_files = [path for path in [train_path, test_path, sample_path] if not path.exists()]
    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "The following required files are missing from the data directory:\n"
            f"{missing_text}"
        )

    return train_path, test_path, sample_path


def validate_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, sample_df: pd.DataFrame
) -> None:
    """Confirm that the dataset columns match the required Kaggle schema."""
    expected_train_columns = ["ID", "TEXT", "LABEL"]
    expected_test_columns = ["ID", "TEXT"]
    expected_sample_columns = ["ID", "LABEL"]

    if list(train_df.columns) != expected_train_columns:
        raise ValueError(
            f"train.csv columns must be {expected_train_columns}, "
            f"but found {list(train_df.columns)}"
        )

    if list(test_df.columns) != expected_test_columns:
        raise ValueError(
            f"test.csv columns must be {expected_test_columns}, "
            f"but found {list(test_df.columns)}"
        )

    if list(sample_df.columns) != expected_sample_columns:
        raise ValueError(
            f"sample_submission.csv columns must be {expected_sample_columns}, "
            f"but found {list(sample_df.columns)}"
        )


def clean_text(text: str, stemmer: PorterStemmer) -> str:
    """Normalize raw review text into a cleaner representation for TF-IDF."""
    # Convert missing values to an empty string so the cleaning steps do not fail.
    text = "" if pd.isna(text) else str(text)

    # Decode HTML entities and remove HTML tags that appear in some rows.
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Lowercase the text and keep only letters, numbers, and spaces.
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse repeated whitespace to keep tokenization consistent.
    text = re.sub(r"\s+", " ", text).strip()

    # Apply stemming so related word forms are treated more similarly.
    tokens = [stemmer.stem(token) for token in text.split()]
    return " ".join(tokens)


def prepare_datasets(
    train_path: Path, test_path: Path, sample_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CSV files and add a cleaned text column to train and test sets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_path)

    validate_columns(train_df, test_df, sample_df)

    stemmer = PorterStemmer()

    # Keep both stemmed and non-stemmed views of the text because some classifiers
    # work better when the original word forms are preserved.
    train_df["clean_text"] = train_df["TEXT"].apply(lambda value: clean_text(value, stemmer))
    test_df["clean_text"] = test_df["TEXT"].apply(lambda value: clean_text(value, stemmer))
    train_df["clean_text_no_stem"] = train_df["TEXT"].apply(
        lambda value: clean_text_without_stemming(value)
    )
    test_df["clean_text_no_stem"] = test_df["TEXT"].apply(
        lambda value: clean_text_without_stemming(value)
    )
    train_df["clean_text_mild"] = train_df["TEXT"].apply(lambda value: clean_text_mild(value))
    test_df["clean_text_mild"] = test_df["TEXT"].apply(lambda value: clean_text_mild(value))

    return train_df, test_df, sample_df


def clean_text_without_stemming(text: str) -> str:
    """Normalize raw review text but keep original word forms intact."""
    text = "" if pd.isna(text) else str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_mild(text: str) -> str:
    """Normalize text while preserving punctuation cues that help sentiment."""
    text = "" if pd.isna(text) else str(text)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipelines() -> dict[str, Pipeline]:
    """Create the model pipelines that share the same TF-IDF vectorization step."""
    stemmed_tfidf_settings = {
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.95,
        "max_features": 60000,
        "sublinear_tf": True,
        "dtype": np.float32,
    }
    word_tfidf_settings = {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_df": 0.95,
        "max_features": 120000,
        "sublinear_tf": True,
        "dtype": np.float32,
    }
    char_tfidf_settings = {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "min_df": 3,
        "max_features": 80000,
        "sublinear_tf": True,
        "dtype": np.float32,
    }

    return {
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**stemmed_tfidf_settings)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        solver="saga",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**stemmed_tfidf_settings)),
                ("model", MultinomialNB(alpha=0.5)),
            ]
        ),
        "Linear SVM (word TF-IDF)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**word_tfidf_settings)),
                ("model", LinearSVC(C=1.0)),
            ]
        ),
        "Linear SVM (word + char TF-IDF)": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(**word_tfidf_settings)),
                            ("char", TfidfVectorizer(**char_tfidf_settings)),
                        ]
                    ),
                ),
                ("model", LinearSVC(C=0.8)),
            ]
        ),
        "Linear SVM (word + char TF-IDF, mild cleaning)": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "word",
                                TfidfVectorizer(
                                    analyzer="word",
                                    ngram_range=(1, 2),
                                    min_df=3,
                                    max_df=0.98,
                                    max_features=160000,
                                    sublinear_tf=True,
                                    dtype=np.float32,
                                ),
                            ),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=3,
                                    max_features=120000,
                                    sublinear_tf=True,
                                    dtype=np.float32,
                                ),
                            ),
                        ]
                    ),
                ),
                ("model", LinearSVC(C=0.8)),
            ]
        ),
    }


def evaluate_models(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, str]]:
    """Train each candidate model, score it on a validation split, and compare results."""
    train_split, valid_split, y_train, y_valid = train_test_split(
        train_df[["clean_text", "clean_text_no_stem", "clean_text_mild"]],
        train_df["LABEL"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_df["LABEL"],
    )

    model_pipelines = build_pipelines()
    metrics_rows: list[dict[str, float | str]] = []
    reports: dict[str, str] = {}

    for model_name, pipeline in model_pipelines.items():
        print(f"Training {model_name}...", flush=True)

        if "mild cleaning" in model_name:
            X_train = train_split["clean_text_mild"]
            X_valid = valid_split["clean_text_mild"]
        elif "Linear SVM" in model_name:
            X_train = train_split["clean_text_no_stem"]
            X_valid = valid_split["clean_text_no_stem"]
        else:
            X_train = train_split["clean_text"]
            X_valid = valid_split["clean_text"]

        # Train the pipeline on the training split.
        pipeline.fit(X_train, y_train)

        # Generate predictions on the validation split for comparison.
        predictions = pipeline.predict(X_valid)

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_valid, predictions),
                "precision": precision_score(y_valid, predictions, average="weighted", zero_division=0),
                "recall": recall_score(y_valid, predictions, average="weighted", zero_division=0),
                "f1_score": f1_score(y_valid, predictions, average="weighted", zero_division=0),
            }
        )

        # Keep the full class-wise report for the text summary file.
        reports[model_name] = classification_report(y_valid, predictions, digits=4, zero_division=0)

    results_df = pd.DataFrame(metrics_rows).sort_values(
        by=["f1_score", "accuracy"], ascending=False
    ).reset_index(drop=True)

    return results_df, model_pipelines, reports


def save_model_comparison_chart(results_df: pd.DataFrame) -> Path:
    """Save a chart that compares the validation metrics across models."""
    chart_path = OUTPUT_DIR / "model_comparison.png"

    plot_df = results_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="metric",
        value_name="score",
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Validation Metric Comparison")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

    return chart_path


def save_results_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    results_df: pd.DataFrame,
    reports: dict[str, str],
    best_model_name: str,
) -> Path:
    """Write the model comparison results and best-model summary to a text file."""
    results_path = OUTPUT_DIR / "model_results.txt"

    lines = [
        "NLP Classification Project Results",
        "=" * 40,
        "",
        f"Training rows: {len(train_df)}",
        f"Test rows: {len(test_df)}",
        f"Sample submission rows: {len(sample_df)}",
        "",
        "Preprocessing steps:",
        "- HTML entity decoding and tag removal",
        "- Lowercasing",
        "- Removal of non-alphanumeric characters",
        "- Whitespace normalization",
        "- Porter stemming for the baseline word models",
        "- Non-stemmed text preserved for the stronger SVM variants",
        "- Mild-cleaning variant preserves punctuation cues for sentiment-heavy examples",
        "",
        "Validation setup:",
        "- 80/20 train/validation split",
        f"- Random state: {RANDOM_STATE}",
        "- Stratified split using LABEL",
        "",
        "Model comparison:",
    ]

    for _, row in results_df.iterrows():
        lines.extend(
            [
                f"- {row['model']}",
                f"  Accuracy : {row['accuracy']:.4f}",
                f"  Precision: {row['precision']:.4f}",
                f"  Recall   : {row['recall']:.4f}",
                f"  F1-score : {row['f1_score']:.4f}",
                "",
                "  Classification report:",
                reports[row["model"]],
                "",
            ]
        )

    lines.extend(
        [
            f"Best model selected: {best_model_name}",
            "",
            "Saved outputs:",
            f"- {results_path}",
            f"- {OUTPUT_DIR / 'submission.csv'}",
            f"- {OUTPUT_DIR / 'model_comparison.png'}",
        ]
    )

    results_path.write_text("\n".join(lines), encoding="utf-8")
    return results_path


def train_best_model_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    model_pipelines: dict[str, Pipeline],
    best_model_name: str,
) -> Path:
    """Retrain the best pipeline on all training data and create the Kaggle submission file."""
    best_pipeline = model_pipelines[best_model_name]

    # Fit the best model on the full training set before creating test predictions.
    print(f"Retraining best model on full training data: {best_model_name}", flush=True)
    if "mild cleaning" in best_model_name:
        training_text = train_df["clean_text_mild"]
        test_text = test_df["clean_text_mild"]
    elif "Linear SVM" in best_model_name:
        training_text = train_df["clean_text_no_stem"]
        test_text = test_df["clean_text_no_stem"]
    else:
        training_text = train_df["clean_text"]
        test_text = test_df["clean_text"]

    best_pipeline.fit(training_text, train_df["LABEL"])
    test_predictions = best_pipeline.predict(test_text)

    # Build the submission with the exact required column names and order.
    submission_df = pd.DataFrame({"ID": test_df["ID"], "LABEL": test_predictions})
    submission_df = submission_df[["ID", "LABEL"]]

    if list(submission_df.columns) != list(sample_df.columns):
        raise ValueError("The submission columns do not match sample_submission.csv")

    submission_path = OUTPUT_DIR / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    return submission_path


def main() -> None:
    """Run the complete training, evaluation, and submission workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path, test_path, sample_path = validate_input_files()
    train_df, test_df, sample_df = prepare_datasets(train_path, test_path, sample_path)

    results_df, model_pipelines, reports = evaluate_models(train_df)
    best_model_name = results_df.loc[0, "model"]

    save_model_comparison_chart(results_df)
    submission_path = train_best_model_and_predict(
        train_df, test_df, sample_df, model_pipelines, best_model_name
    )
    results_path = save_results_summary(
        train_df, test_df, sample_df, results_df, reports, best_model_name
    )

    # Print a short summary so the user can confirm the pipeline finished successfully.
    print("Training complete.")
    print(f"Best model: {best_model_name}")
    print(f"Results saved to: {results_path}")
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
