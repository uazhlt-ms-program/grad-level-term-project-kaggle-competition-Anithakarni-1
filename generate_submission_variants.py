"""Generate stronger Kaggle submission variants for the LING 539 competition."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    """Aggressively normalize raw text for a traditional TF-IDF baseline."""
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Kaggle train and test files and add cleaned text variants."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_df["clean_text"] = train_df["TEXT"].map(clean_text)
    test_df["clean_text"] = test_df["TEXT"].map(clean_text)
    train_df["clean_text_mild"] = train_df["TEXT"].map(clean_text_mild)
    test_df["clean_text_mild"] = test_df["TEXT"].map(clean_text_mild)

    return train_df, test_df


def build_submission_models() -> dict[str, tuple[str, Pipeline]]:
    """Return the strongest traditional text models found so far."""
    return {
        "svm_word": (
            "clean_text",
            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=3,
                            max_df=0.95,
                            max_features=120000,
                            sublinear_tf=True,
                        ),
                    ),
                    ("model", LinearSVC(C=1.0)),
                ]
            ),
        ),
        "svm_word_char": (
            "clean_text",
            Pipeline(
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
                                        max_df=0.95,
                                        max_features=120000,
                                        sublinear_tf=True,
                                    ),
                                ),
                                (
                                    "char",
                                    TfidfVectorizer(
                                        analyzer="char_wb",
                                        ngram_range=(3, 5),
                                        min_df=3,
                                        max_features=80000,
                                        sublinear_tf=True,
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("model", LinearSVC(C=0.8)),
                ]
            ),
        ),
        "svm_word_char_mild": (
            "clean_text_mild",
            Pipeline(
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
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("model", LinearSVC(C=0.8)),
                ]
            ),
        ),
    }


def main() -> None:
    """Score the strong variants on validation and write Kaggle-ready CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df[["clean_text", "clean_text_mild"]],
        train_df["LABEL"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_df["LABEL"],
    )

    models = build_submission_models()
    results_lines = []

    for model_name, (text_column, pipeline) in models.items():
        print(f"Training {model_name}...", flush=True)
        pipeline.fit(X_train[text_column], y_train)
        valid_predictions = pipeline.predict(X_valid[text_column])

        accuracy = accuracy_score(y_valid, valid_predictions)
        weighted_f1 = f1_score(y_valid, valid_predictions, average="weighted")
        results_lines.append(
            f"{model_name}\taccuracy={accuracy:.5f}\tweighted_f1={weighted_f1:.5f}"
        )

        pipeline.fit(train_df[text_column], train_df["LABEL"])
        test_predictions = pipeline.predict(test_df[text_column])

        submission_df = pd.DataFrame({"ID": test_df["ID"], "LABEL": test_predictions})
        submission_path = OUTPUT_DIR / f"submission_{model_name}.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Wrote {submission_path}", flush=True)

    results_path = OUTPUT_DIR / "tuning_results.txt"
    results_path.write_text("\n".join(results_lines) + "\n", encoding="utf-8")
    print(f"Saved validation summary to {results_path}", flush=True)


if __name__ == "__main__":
    main()
