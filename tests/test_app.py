from pathlib import Path

import pandas as pd
import pytest

from app.app import (
    CATEGORICAL_PREDICTION_FIELDS,
    NUMERIC_PREDICTION_FIELDS,
    PREDICTION_TARGET_LABELS,
    TARGET_LABELS,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    apply_filters,
    build_job_subscription_rate,
    build_monthly_subscription_rate,
    build_overview_metrics,
    build_prediction_input_options,
    build_subscription_summary,
    collect_prediction_inputs,
    get_filter_options,
    load_dataset,
    main,
    normalize_subscribe,
    read_training_summary,
)
from app.training import (
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    PREDICTION_FEATURE_ORDER,
    build_prediction_frame,
    calculate_metrics,
    load_training_artifacts,
    normalize_target,
    predict_subscription,
    prepare_training_frame,
    train_from_csv,
    train_model,
)


class SidebarStub:
    def __init__(self) -> None:
        self.headers: list[str] = []
        self.slider_calls: list[tuple[str, int, int, tuple[int, int]]] = []
        self.multiselect_calls: list[tuple[str, list[str]]] = []
        self.slider_value = (18, 95)
        self.multiselect_values: dict[str, list[str]] = {
            "职业": [],
            "婚姻状态": [],
            "联系月份": [],
            "认购结果": [],
        }

    def header(self, value: str) -> None:
        self.headers.append(value)

    def slider(
        self,
        label: str,
        min_value: int,
        max_value: int,
        value: tuple[int, int],
    ) -> tuple[int, int]:
        self.slider_calls.append((label, min_value, max_value, value))
        return self.slider_value

    def multiselect(self, label: str, options: list[str], format_func=None) -> list[str]:
        self.multiselect_calls.append((label, options))
        return self.multiselect_values[label]


class StreamlitStub:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, object]] = []
        self.page_config: dict[str, str] = {}
        self.title_text = ""
        self.caption_texts: list[str] = []
        self.subheaders: list[str] = []
        self.dataframes: list[pd.DataFrame] = []
        self.bar_charts: list[pd.DataFrame] = []
        self.line_charts: list[pd.DataFrame] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.info_messages: list[str] = []
        self.success_messages: list[str] = []
        self.sidebar = SidebarStub()
        self.number_input_calls: list[tuple[str, object]] = []
        self.selectbox_calls: list[tuple[str, list[str]]] = []
        self.button_calls: list[str] = []
        self.button_value = False

    def set_page_config(self, **kwargs: str) -> None:
        self.page_config = kwargs

    def title(self, value: str) -> None:
        self.title_text = value

    def caption(self, value: str) -> None:
        self.caption_texts.append(value)

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def columns(self, count: int):
        return [MetricColumn(self) for _ in range(count)]

    def dataframe(self, value: pd.DataFrame, use_container_width: bool) -> None:
        self.dataframes.append(value)

    def bar_chart(self, value: pd.DataFrame) -> None:
        self.bar_charts.append(value)

    def line_chart(self, value: pd.DataFrame) -> None:
        self.line_charts.append(value)

    def warning(self, value: str) -> None:
        self.warnings.append(value)

    def error(self, value: str) -> None:
        self.errors.append(value)

    def info(self, value: str) -> None:
        self.info_messages.append(value)

    def success(self, value: str) -> None:
        self.success_messages.append(value)

    def metric(self, label: str, value: object) -> None:
        self.metrics.append((label, value))

    def number_input(self, label: str, **kwargs) -> object:
        self.number_input_calls.append((label, kwargs))
        return kwargs["value"]

    def selectbox(self, label: str, options: list[str]) -> str:
        self.selectbox_calls.append((label, options))
        return options[0]

    def button(self, label: str) -> bool:
        self.button_calls.append(label)
        return self.button_value


class MetricColumn:
    def __init__(self, parent: StreamlitStub) -> None:
        self.parent = parent

    def metric(self, label: str, value: object) -> None:
        self.parent.metrics.append((label, value))


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DATA_PATH)


@pytest.fixture
def temp_artifact_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "model.joblib", tmp_path / "metrics.json"


@pytest.fixture
def trained_artifacts(temp_artifact_paths: tuple[Path, Path]) -> tuple[Path, Path]:
    model_path, metrics_path = temp_artifact_paths
    train_from_csv(model_path=model_path, metrics_path=metrics_path)
    return model_path, metrics_path


def test_load_dataset_reads_training_data():
    df = load_dataset("data/train.csv")

    assert not df.empty
    assert "subscribe" in df.columns


def test_load_dataset_raises_for_missing_file(tmp_path: Path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(missing_file)


def test_normalize_subscribe_lowercases_values():
    series = pd.Series(["Yes", " no ", "YES"])

    assert normalize_subscribe(series).tolist() == ["yes", "no", "yes"]


def test_get_filter_options_returns_sorted_unique_values(train_df: pd.DataFrame):
    options = get_filter_options(train_df, "marital")

    assert options == sorted(train_df["marital"].dropna().astype(str).unique().tolist())


def test_apply_filters_filters_by_multiple_conditions(train_df: pd.DataFrame):
    filtered_df = apply_filters(
        train_df,
        jobs=["admin."],
        marital_statuses=["married", "single", "divorced"],
        months=["may", "aug"],
        age_range=(30, 60),
        subscribe_values=["yes"],
    )

    assert not filtered_df.empty
    assert set(filtered_df["job"]) == {"admin."}
    assert set(filtered_df["month"]).issubset({"may", "aug"})
    assert filtered_df["age"].between(30, 60).all()
    assert normalize_subscribe(filtered_df["subscribe"]).eq("yes").all()


def test_build_overview_metrics_returns_expected_keys(train_df: pd.DataFrame):
    metrics = build_overview_metrics(train_df.head(10))

    assert metrics["sample_count"] == 10
    assert 0 <= metrics["subscribe_rate"] <= 100
    assert metrics["average_age"] > 0
    assert metrics["average_duration"] > 0


def test_build_subscription_summary_uses_chinese_labels(train_df: pd.DataFrame):
    summary = build_subscription_summary(train_df)

    assert set(summary["subscribe"]).issubset(set(TARGET_LABELS.values()))
    assert summary["count"].sum() == len(train_df)


def test_build_job_subscription_rate_sorts_descending(train_df: pd.DataFrame):
    summary = build_job_subscription_rate(train_df)

    assert list(summary.columns) == ["job", "subscription_rate", "sample_count"]
    assert summary.iloc[0]["subscription_rate"] >= summary.iloc[-1]["subscription_rate"]


def test_build_monthly_subscription_rate_uses_expected_month_order(train_df: pd.DataFrame):
    summary = build_monthly_subscription_rate(train_df)

    month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    expected_order = [month for month in month_order if month in set(train_df["month"])]
    assert summary["month"].tolist() == expected_order


def test_build_prediction_input_options_returns_numeric_and_categorical_fields(
    train_df: pd.DataFrame,
):
    options = build_prediction_input_options(train_df)

    assert set(NUMERIC_PREDICTION_FIELDS).issubset(options)
    assert set(CATEGORICAL_PREDICTION_FIELDS).issubset(options)
    assert options["age"]["type"] == "numeric"
    assert options["job"]["type"] == "categorical"


def test_collect_prediction_inputs_uses_defaults(
    monkeypatch: pytest.MonkeyPatch, train_df: pd.DataFrame
):
    stub = StreamlitStub()

    monkeypatch.setattr("app.app.st", stub)

    prediction_inputs = collect_prediction_inputs(train_df)

    assert list(prediction_inputs.keys()) == PREDICTION_FEATURE_ORDER
    assert len(stub.number_input_calls) == len(NUMERIC_PREDICTION_FIELDS)
    assert len(stub.selectbox_calls) == len(CATEGORICAL_PREDICTION_FIELDS)


def test_normalize_target_maps_labels_to_binary():
    series = pd.Series(["yes", "no", " yes "])

    assert normalize_target(series).tolist() == [1, 0, 1]


def test_prepare_training_frame_returns_features_and_target(train_df: pd.DataFrame):
    features, target = prepare_training_frame(train_df)

    assert "subscribe" not in features.columns
    assert "id" not in features.columns
    assert target.isin([0, 1]).all()
    assert len(features) == len(target)


def test_train_model_returns_pipeline_metrics_and_metadata(train_df: pd.DataFrame):
    result = train_model(train_df)

    assert {"pipeline", "metrics", "metadata"}.issubset(result.keys())
    assert result["metadata"]["model_type"] == "LogisticRegression"
    assert result["metadata"]["feature_count"] > 0
    assert 0 <= result["metrics"]["accuracy"] <= 1


def test_calculate_metrics_returns_rounded_scores():
    metrics = calculate_metrics(pd.Series([1, 0, 1, 0]), pd.Series([1, 0, 0, 0]))

    assert metrics == {
        "accuracy": 0.75,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
    }


def test_build_prediction_frame_preserves_expected_order():
    inputs = {field: index for index, field in enumerate(PREDICTION_FEATURE_ORDER)}

    frame = build_prediction_frame(inputs)

    assert list(frame.columns) == PREDICTION_FEATURE_ORDER


def test_build_prediction_frame_raises_for_missing_fields():
    with pytest.raises(ValueError, match="Missing prediction fields"):
        build_prediction_frame({"age": 30})


def test_train_from_csv_persists_artifacts(temp_artifact_paths: tuple[Path, Path]):
    model_path, metrics_path = temp_artifact_paths

    result = train_from_csv(model_path=model_path, metrics_path=metrics_path)

    assert model_path.exists()
    assert metrics_path.exists()
    loaded = load_training_artifacts(model_path=model_path, metrics_path=metrics_path)
    assert loaded["metadata"]["model_type"] == result["metadata"]["model_type"]
    assert loaded["metrics"] == result["metrics"]


def test_predict_subscription_returns_label_and_probability(
    train_df: pd.DataFrame, trained_artifacts: tuple[Path, Path]
):
    model_path, metrics_path = trained_artifacts
    sample_inputs = train_df.drop(columns=["id", "subscribe"]).iloc[0].to_dict()

    prediction = predict_subscription(
        sample_inputs, model_path=model_path, metrics_path=metrics_path
    )

    assert prediction["label"] in PREDICTION_TARGET_LABELS
    assert 0 <= prediction["probability"] <= 1


def test_read_training_summary_returns_none_when_artifacts_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.app.MODEL_ARTIFACT_PATH", Path("missing-model.joblib"))
    monkeypatch.setattr("app.app.METRICS_PATH", Path("missing-metrics.json"))

    assert read_training_summary() is None


def test_main_renders_analysis_page(monkeypatch: pytest.MonkeyPatch):
    stub = StreamlitStub()
    mock_summary = {
        "metrics": {"accuracy": 0.9, "precision": 0.88, "recall": 0.87, "f1": 0.875},
        "metadata": {
            "model_type": "LogisticRegression",
            "train_size": 100,
            "validation_size": 20,
            "feature_count": 20,
        },
    }

    monkeypatch.setattr("app.app.st", stub)
    monkeypatch.setattr("app.app.read_training_summary", lambda: mock_summary)

    main()

    assert stub.page_config == {"page_title": "banksys", "layout": "wide"}
    assert stub.title_text == "banksys"
    assert stub.caption_texts[0] == "银行营销数据分析与认购预测系统"
    assert stub.sidebar.headers == ["分析筛选"]
    assert stub.subheaders[:2] == ["数据集概览", "交互式数据分析"]
    assert ("训练集样本数", len(pd.read_csv(TRAIN_DATA_PATH))) in stub.metrics
    assert ("测试集样本数", len(pd.read_csv(TEST_DATA_PATH))) in stub.metrics
    assert any(label == "认购率" for label, _ in stub.metrics)
    assert any(label == "Accuracy" for label, _ in stub.metrics)
    assert "在线预测" in stub.subheaders
    assert len(stub.bar_charts) == 2
    assert len(stub.line_charts) == 1
    assert len(stub.dataframes) >= 5
    assert not stub.errors


def test_main_shows_training_hint_when_artifacts_missing(monkeypatch: pytest.MonkeyPatch):
    stub = StreamlitStub()
    expected_info = "当前还没有训练产物。请先运行 `python -m app.training` 生成模型和指标文件。"
    expected_warning = "当前没有可用模型，无法进行在线预测。请先运行 `python -m app.training`。"

    monkeypatch.setattr("app.app.st", stub)
    monkeypatch.setattr("app.app.read_training_summary", lambda: None)

    main()

    assert stub.info_messages == [expected_info]
    assert stub.warnings[-1] == expected_warning


def test_main_renders_prediction_result(monkeypatch: pytest.MonkeyPatch):
    stub = StreamlitStub()
    stub.button_value = True
    mock_summary = {
        "metrics": {"accuracy": 0.9, "precision": 0.88, "recall": 0.87, "f1": 0.875},
        "metadata": {
            "model_type": "LogisticRegression",
            "train_size": 100,
            "validation_size": 20,
            "feature_count": 20,
        },
    }
    mock_prediction = {"label": "yes", "probability": 0.8234, "metadata": mock_summary["metadata"]}

    monkeypatch.setattr("app.app.st", stub)
    monkeypatch.setattr("app.app.read_training_summary", lambda: mock_summary)
    monkeypatch.setattr("app.app.predict_subscription", lambda inputs: mock_prediction)

    main()

    assert stub.success_messages == [PREDICTION_TARGET_LABELS["yes"]]
    assert ("认购概率", "82.34%") in stub.metrics


def test_main_shows_error_when_prediction_fails(monkeypatch: pytest.MonkeyPatch):
    stub = StreamlitStub()
    stub.button_value = True
    mock_summary = {
        "metrics": {"accuracy": 0.9, "precision": 0.88, "recall": 0.87, "f1": 0.875},
        "metadata": {
            "model_type": "LogisticRegression",
            "train_size": 100,
            "validation_size": 20,
            "feature_count": 20,
        },
    }

    monkeypatch.setattr("app.app.st", stub)
    monkeypatch.setattr("app.app.read_training_summary", lambda: mock_summary)

    def fake_predict_subscription(inputs: dict[str, object]) -> dict[str, object]:
        raise ValueError("mock prediction error")

    monkeypatch.setattr("app.app.predict_subscription", fake_predict_subscription)

    main()

    assert stub.errors[-1] == "预测失败: mock prediction error"


def test_main_shows_error_when_dataset_missing(monkeypatch: pytest.MonkeyPatch):
    stub = StreamlitStub()

    def fake_load_dataset(path: str | Path) -> pd.DataFrame:
        raise FileNotFoundError(f"Dataset not found: {path}")

    monkeypatch.setattr("app.app.st", stub)
    monkeypatch.setattr("app.app.load_dataset", fake_load_dataset)

    main()

    assert stub.errors == [f"Dataset not found: {TRAIN_DATA_PATH}"]


def test_default_training_artifact_paths_are_gitignored():
    assert MODEL_ARTIFACT_PATH.suffix == ".joblib"
    assert METRICS_PATH.suffix == ".json"
