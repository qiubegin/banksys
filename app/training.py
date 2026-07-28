from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "subscription_model.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
TARGET_COLUMN = "subscribe"
DROP_COLUMNS = ["id", TARGET_COLUMN]
RANDOM_STATE = 42
TEST_SIZE = 0.2
PREDICTION_FEATURE_ORDER = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]


def normalize_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = normalize_target(df[TARGET_COLUMN])
    if target.isna().any():
        raise ValueError("Target column contains unsupported values.")
    features = df.drop(columns=DROP_COLUMNS)
    return features, target.astype(int)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ]
    )


def build_training_pipeline(features: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def train_model(df: pd.DataFrame) -> dict[str, object]:
    features, target = prepare_training_frame(df)
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipeline = build_training_pipeline(features)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_valid)

    metrics = calculate_metrics(y_valid, predictions)
    metadata = {
        "model_type": "LogisticRegression",
        "train_size": int(len(x_train)),
        "validation_size": int(len(x_valid)),
        "feature_count": int(features.shape[1]),
        "feature_names": features.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
    }
    return {"pipeline": pipeline, "metrics": metrics, "metadata": metadata}


def build_prediction_frame(inputs: dict[str, object]) -> pd.DataFrame:
    missing_fields = [field for field in PREDICTION_FEATURE_ORDER if field not in inputs]
    if missing_fields:
        raise ValueError(f"Missing prediction fields: {', '.join(missing_fields)}")
    return pd.DataFrame([{field: inputs[field] for field in PREDICTION_FEATURE_ORDER}])


def predict_subscription(
    inputs: dict[str, object],
    *,
    model_path: Path = MODEL_ARTIFACT_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, object]:
    artifacts = load_training_artifacts(model_path=model_path, metrics_path=metrics_path)
    prediction_frame = build_prediction_frame(inputs)
    pipeline = artifacts["pipeline"]
    predicted_label = int(pipeline.predict(prediction_frame)[0])
    probability = float(pipeline.predict_proba(prediction_frame)[0][1])
    return {
        "label": "yes" if predicted_label == 1 else "no",
        "probability": round(probability, 4),
        "metadata": artifacts["metadata"],
    }


def save_training_artifacts(
    training_result: dict[str, object],
    *,
    model_path: Path = MODEL_ARTIFACT_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": training_result["pipeline"],
            "metadata": training_result["metadata"],
        },
        model_path,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": training_result["metrics"],
                "metadata": training_result["metadata"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_training_artifacts(
    *, model_path: Path = MODEL_ARTIFACT_PATH, metrics_path: Path = METRICS_PATH
) -> dict[str, object]:
    artifact = joblib.load(model_path)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "pipeline": artifact["pipeline"],
        "metadata": artifact["metadata"],
        "metrics": metrics_payload["metrics"],
    }


def train_from_csv(
    path: Path = TRAIN_DATA_PATH,
    *,
    model_path: Path = MODEL_ARTIFACT_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, object]:
    df = pd.read_csv(path)
    training_result = train_model(df)
    save_training_artifacts(training_result, model_path=model_path, metrics_path=metrics_path)
    return training_result


def main() -> None:
    training_result = train_from_csv()
    payload = {
        "metrics": training_result["metrics"],
        "metadata": training_result["metadata"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
