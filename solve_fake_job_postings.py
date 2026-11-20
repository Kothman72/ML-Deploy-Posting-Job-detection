"""Train and evaluate a fake-job-posting classifier from local CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
TEXT_FIELDS = {
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
}


def existing_path(explicit_path: str | None, filename: str) -> Path | None:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    for path in (ROOT / filename, ROOT / "data" / filename):
        if path.is_file():
            return path
    return None


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if feature_columns is not None:
        frame = frame.reindex(columns=feature_columns)
    else:
        feature_columns = list(frame.columns)

    text_columns = [
        column for column in feature_columns if column.lower() in TEXT_FIELDS
    ]
    if not text_columns:
        text_columns = [
            column
            for column in feature_columns
            if pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
        ]

    category_columns = [
        column
        for column in feature_columns
        if column not in text_columns
        and (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]
    numeric_columns = [
        column
        for column in feature_columns
        if column not in text_columns
        and column not in category_columns
        and pd.api.types.is_numeric_dtype(frame[column])
    ]

    prepared = frame.copy()
    if text_columns:
        prepared["_combined_text"] = (
            prepared[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
        )
    for column in category_columns:
        prepared[column] = prepared[column].fillna("__missing__").astype(str)

    selected_columns = (
        (["_combined_text"] if text_columns else [])
        + category_columns
        + numeric_columns
    )
    return prepared[selected_columns], text_columns, category_columns, numeric_columns


def build_model(
    text_columns: list[str],
    category_columns: list[str],
    numeric_columns: list[str],
) -> Pipeline:
    transformers = []
    if text_columns:
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=250_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                "_combined_text",
            )
        )
    if category_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                category_columns,
            )
        )
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable text, categorical, or numeric features were found.")

    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            (
                "classifier",
                LogisticRegression(
                    C=4,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", help="Path to job_postings_train.csv")
    parser.add_argument("--test", help="Optional path to job_postings_test.csv")
    parser.add_argument("--output", default="submission.csv", help="Prediction CSV path")
    args = parser.parse_args()

    train_path = existing_path(args.train, "job_postings_train.csv")
    if train_path is None:
        raise FileNotFoundError(
            f"Place job_postings_train.csv in {ROOT} or {ROOT / 'data'}, "
            "or pass its path with --train."
        )

    train = pd.read_csv(train_path)
    target_column = next(
        (column for column in train.columns if column.lower() == "fraudulent"),
        None,
    )
    if target_column is None:
        raise ValueError(
            f"Training data must include a 'fraudulent' column; found {train.columns.tolist()}."
        )

    id_column = next(
        (column for column in ("job_id", "id") if column in train.columns),
        None,
    )
    feature_columns = [
        column
        for column in train.columns
        if column not in {target_column, id_column}
    ]
    X, text_columns, category_columns, numeric_columns = prepare_features(
        train[feature_columns]
    )
    y = train[target_column].astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Training data needs at least two examples of each class.")

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model = build_model(text_columns, category_columns, numeric_columns)
    model.fit(X_fit, y_fit)
    valid_probabilities = model.predict_proba(X_valid)[:, 1]
    valid_predictions = (valid_probabilities >= 0.5).astype(int)

    print(f"Training file: {train_path}")
    print(f"Rows: {len(train):,}; fraudulent rate: {y.mean():.2%}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_valid, valid_probabilities):.4f}")
    print(f"Validation F1 (threshold 0.5): {f1_score(y_valid, valid_predictions):.4f}")
    print("Validation confusion matrix:\n", confusion_matrix(y_valid, valid_predictions))
    print(
        classification_report(
            y_valid,
            valid_predictions,
            target_names=["legitimate", "fraudulent"],
            zero_division=0,
        )
    )

    test_path = existing_path(args.test, "job_postings_test.csv")
    if test_path is None:
        print(
            "No competition test CSV found; validation is complete. "
            "Add job_postings_test.csv to create a submission."
        )
        return

    test = pd.read_csv(test_path)
    test_ids = test[id_column].copy() if id_column and id_column in test.columns else None
    test_features, _, _, _ = prepare_features(
        test.reindex(columns=feature_columns), feature_columns
    )
    model.fit(X, y)
    test_probabilities = model.predict_proba(test_features)[:, 1]

    submission = pd.DataFrame({target_column: test_probabilities})
    if test_ids is not None:
        submission.insert(0, id_column, test_ids.to_numpy())
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
