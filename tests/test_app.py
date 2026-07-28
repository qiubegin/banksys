from pathlib import Path

import pandas as pd
import pytest

from app.app import (
    TARGET_LABELS,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    apply_filters,
    build_job_subscription_rate,
    build_monthly_subscription_rate,
    build_overview_metrics,
    build_subscription_summary,
    get_filter_options,
    load_dataset,
    main,
    normalize_subscribe,
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
        self.sidebar = SidebarStub()

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


class MetricColumn:
    def __init__(self, parent: StreamlitStub) -> None:
        self.parent = parent

    def metric(self, label: str, value: object) -> None:
        self.parent.metrics.append((label, value))


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DATA_PATH)


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


def test_main_renders_analysis_page(monkeypatch: pytest.MonkeyPatch):
    stub = StreamlitStub()

    monkeypatch.setattr("app.app.st", stub)

    main()

    assert stub.page_config == {"page_title": "banksys", "layout": "wide"}
    assert stub.title_text == "banksys"
    assert stub.caption_texts[0] == "银行营销数据分析与认购预测系统"
    assert stub.sidebar.headers == ["分析筛选"]
    assert stub.subheaders[:2] == ["数据集概览", "交互式数据分析"]
    assert ("训练集样本数", len(pd.read_csv(TRAIN_DATA_PATH))) in stub.metrics
    assert ("测试集样本数", len(pd.read_csv(TEST_DATA_PATH))) in stub.metrics
    assert any(label == "认购率" for label, _ in stub.metrics)
    assert len(stub.bar_charts) == 2
    assert len(stub.line_charts) == 1
    assert len(stub.dataframes) >= 4
    assert not stub.errors


def test_main_shows_error_when_dataset_missing(monkeypatch: pytest.MonkeyPatch):
    stub = StreamlitStub()

    def fake_load_dataset(path: str | Path) -> pd.DataFrame:
        raise FileNotFoundError(f"Dataset not found: {path}")

    monkeypatch.setattr("app.app.st", stub)
    monkeypatch.setattr("app.app.load_dataset", fake_load_dataset)

    main()

    assert stub.errors == [f"Dataset not found: {TRAIN_DATA_PATH}"]
