from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.training import (
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    PREDICTION_FEATURE_ORDER,
    load_training_artifacts,
    predict_subscription,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
MONTH_ORDER = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
TARGET_LABELS = {"yes": "认购", "no": "未认购"}
PREDICTION_TARGET_LABELS = {"yes": "预测会认购", "no": "预测不会认购"}
NUMERIC_PREDICTION_FIELDS = {
    "age": "年龄",
    "duration": "通话时长",
    "campaign": "本次营销联系次数",
    "pdays": "距上次联系的天数",
    "previous": "历史联系次数",
    "emp_var_rate": "就业变化率",
    "cons_price_index": "消费者价格指数",
    "cons_conf_index": "消费者信心指数",
    "lending_rate3m": "3个月贷款利率",
    "nr_employed": "就业人数指标",
}
CATEGORICAL_PREDICTION_FIELDS = {
    "job": "职业",
    "marital": "婚姻状态",
    "education": "教育程度",
    "default": "是否违约",
    "housing": "是否有房贷",
    "loan": "是否有个人贷款",
    "contact": "联系方式",
    "month": "联系月份",
    "day_of_week": "联系星期",
    "poutcome": "上次营销结果",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def normalize_subscribe(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def get_filter_options(df: pd.DataFrame, column: str) -> list[str]:
    return sorted(df[column].dropna().astype(str).unique().tolist())


def apply_filters(
    df: pd.DataFrame,
    *,
    jobs: list[str],
    marital_statuses: list[str],
    months: list[str],
    age_range: tuple[int, int],
    subscribe_values: list[str],
) -> pd.DataFrame:
    filtered_df = df.copy()

    if jobs:
        filtered_df = filtered_df[filtered_df["job"].isin(jobs)]
    if marital_statuses:
        filtered_df = filtered_df[filtered_df["marital"].isin(marital_statuses)]
    if months:
        filtered_df = filtered_df[filtered_df["month"].isin(months)]

    min_age, max_age = age_range
    filtered_df = filtered_df[filtered_df["age"].between(min_age, max_age)]

    if subscribe_values:
        subscribe_series = normalize_subscribe(filtered_df["subscribe"])
        filtered_df = filtered_df[subscribe_series.isin(subscribe_values)]

    return filtered_df


def build_overview_metrics(df: pd.DataFrame) -> dict[str, float]:
    subscribe_rate = normalize_subscribe(df["subscribe"]).eq("yes").mean() if not df.empty else 0.0

    return {
        "sample_count": int(len(df)),
        "subscribe_rate": round(subscribe_rate * 100, 2),
        "average_age": round(df["age"].mean(), 1) if not df.empty else 0.0,
        "average_duration": round(df["duration"].mean(), 1) if not df.empty else 0.0,
    }


def build_subscription_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        normalize_subscribe(df["subscribe"])
        .map(TARGET_LABELS)
        .value_counts()
        .rename_axis("subscribe")
        .reset_index(name="count")
    )
    if summary.empty:
        return pd.DataFrame(columns=["subscribe", "count"])
    return summary.sort_values("count", ascending=False).reset_index(drop=True)


def build_job_subscription_rate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["job", "subscription_rate"])

    summary = (
        df.assign(subscribe_flag=normalize_subscribe(df["subscribe"]).eq("yes").astype(float))
        .groupby("job", as_index=False)
        .agg(subscription_rate=("subscribe_flag", "mean"), sample_count=("subscribe_flag", "size"))
        .sort_values(["subscription_rate", "sample_count", "job"], ascending=[False, False, True])
    )
    summary["subscription_rate"] = (summary["subscription_rate"] * 100).round(2)
    return summary


def build_monthly_subscription_rate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "subscription_rate"])

    summary = (
        df.assign(subscribe_flag=normalize_subscribe(df["subscribe"]).eq("yes").astype(float))
        .groupby("month", as_index=False)
        .agg(subscription_rate=("subscribe_flag", "mean"))
    )
    summary["month"] = pd.Categorical(summary["month"], categories=MONTH_ORDER, ordered=True)
    summary = summary.sort_values("month")
    summary["subscription_rate"] = (summary["subscription_rate"] * 100).round(2)
    summary["month"] = summary["month"].astype(str)
    return summary.reset_index(drop=True)


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["metric", "value"])

    summary = pd.DataFrame(
        {
            "metric": ["平均年龄", "平均通话时长", "平均营销次数"],
            "value": [
                round(df["age"].mean(), 2),
                round(df["duration"].mean(), 2),
                round(df["campaign"].mean(), 2),
            ],
        }
    )
    return summary


def read_training_summary() -> dict[str, object] | None:
    if not MODEL_ARTIFACT_PATH.exists() or not METRICS_PATH.exists():
        return None
    return load_training_artifacts()


def build_prediction_input_options(train_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    options: dict[str, dict[str, object]] = {}
    for field in PREDICTION_FEATURE_ORDER:
        if field in NUMERIC_PREDICTION_FIELDS:
            series = train_df[field]
            options[field] = {
                "label": NUMERIC_PREDICTION_FIELDS[field],
                "type": "numeric",
                "min": float(series.min()),
                "max": float(series.max()),
                "value": float(round(series.median(), 2)),
                "step": 1.0 if pd.api.types.is_integer_dtype(series) else 0.01,
            }
        else:
            options[field] = {
                "label": CATEGORICAL_PREDICTION_FIELDS[field],
                "type": "categorical",
                "options": get_filter_options(train_df, field),
            }
    return options


def collect_prediction_inputs(train_df: pd.DataFrame) -> dict[str, object]:
    options = build_prediction_input_options(train_df)
    inputs: dict[str, object] = {}

    st.subheader("在线预测")
    st.caption("通过点选输入客户特征，调用离线训练模型预测该客户是否会认购。")

    for field in PREDICTION_FEATURE_ORDER:
        config = options[field]
        if config["type"] == "numeric":
            if config["step"] == 1.0:
                inputs[field] = st.number_input(
                    config["label"],
                    min_value=int(config["min"]),
                    max_value=int(config["max"]),
                    value=int(config["value"]),
                    step=1,
                )
            else:
                inputs[field] = st.number_input(
                    config["label"],
                    min_value=float(config["min"]),
                    max_value=float(config["max"]),
                    value=float(config["value"]),
                    step=float(config["step"]),
                    format="%.2f",
                )
        else:
            inputs[field] = st.selectbox(config["label"], options=config["options"])
    return inputs


def render_analysis_page(train_df: pd.DataFrame) -> None:
    st.subheader("交互式数据分析")
    st.caption("通过筛选条件观察不同客户群体的认购情况与营销特征。")

    age_range = st.sidebar.slider(
        "年龄范围",
        min_value=int(train_df["age"].min()),
        max_value=int(train_df["age"].max()),
        value=(int(train_df["age"].min()), int(train_df["age"].max())),
    )
    selected_jobs = st.sidebar.multiselect("职业", options=get_filter_options(train_df, "job"))
    selected_marital = st.sidebar.multiselect(
        "婚姻状态", options=get_filter_options(train_df, "marital")
    )
    selected_months = st.sidebar.multiselect(
        "联系月份",
        options=[month for month in MONTH_ORDER if month in set(train_df["month"])],
    )
    selected_subscribe = st.sidebar.multiselect(
        "认购结果",
        options=["yes", "no"],
        format_func=lambda value: TARGET_LABELS[value],
    )

    filtered_df = apply_filters(
        train_df,
        jobs=selected_jobs,
        marital_statuses=selected_marital,
        months=selected_months,
        age_range=age_range,
        subscribe_values=selected_subscribe,
    )

    if filtered_df.empty:
        st.warning("当前筛选条件下没有数据，请调整筛选条件后重试。")
        return

    metrics = build_overview_metrics(filtered_df)
    metric_columns = st.columns(4)
    metric_columns[0].metric("筛选后样本数", metrics["sample_count"])
    metric_columns[1].metric("认购率", f"{metrics['subscribe_rate']}%")
    metric_columns[2].metric("平均年龄", metrics["average_age"])
    metric_columns[3].metric("平均通话时长", metrics["average_duration"])

    st.subheader("认购结果分布")
    subscription_summary = build_subscription_summary(filtered_df)
    st.bar_chart(subscription_summary.set_index("subscribe"))
    st.dataframe(subscription_summary, use_container_width=True)

    st.subheader("不同职业的认购率")
    job_summary = build_job_subscription_rate(filtered_df)
    st.bar_chart(job_summary.set_index("job")[["subscription_rate"]])
    st.dataframe(job_summary, use_container_width=True)

    st.subheader("不同月份的认购率")
    monthly_summary = build_monthly_subscription_rate(filtered_df)
    st.line_chart(monthly_summary.set_index("month")[["subscription_rate"]])
    st.dataframe(monthly_summary, use_container_width=True)

    st.subheader("数值特征摘要")
    st.dataframe(build_numeric_summary(filtered_df), use_container_width=True)

    st.subheader("筛选后样本预览")
    st.dataframe(filtered_df.head(30), use_container_width=True)


def render_training_status() -> None:
    st.subheader("离线训练状态")

    training_summary = read_training_summary()
    if training_summary is None:
        st.info("当前还没有训练产物。请先运行 `python -m app.training` 生成模型和指标文件。")
        return

    metrics = training_summary["metrics"]
    metadata = training_summary["metadata"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    metric_columns[1].metric("Precision", f"{metrics['precision']:.3f}")
    metric_columns[2].metric("Recall", f"{metrics['recall']:.3f}")
    metric_columns[3].metric("F1", f"{metrics['f1']:.3f}")

    st.dataframe(
        pd.DataFrame(
            {
                "字段": ["模型类型", "训练样本数", "验证样本数", "特征数", "模型文件"],
                "值": [
                    metadata["model_type"],
                    metadata["train_size"],
                    metadata["validation_size"],
                    metadata["feature_count"],
                    str(MODEL_ARTIFACT_PATH),
                ],
            }
        ),
        use_container_width=True,
    )


def render_prediction_page(train_df: pd.DataFrame) -> None:
    training_summary = read_training_summary()
    if training_summary is None:
        st.warning("当前没有可用模型，无法进行在线预测。请先运行 `python -m app.training`。")
        return

    prediction_inputs = collect_prediction_inputs(train_df)
    if not st.button("开始预测"):
        return

    try:
        prediction_result = predict_subscription(prediction_inputs)
    except Exception as exc:
        st.error(f"预测失败: {exc}")
        return

    st.success(PREDICTION_TARGET_LABELS[prediction_result["label"]])
    st.metric("认购概率", f"{prediction_result['probability'] * 100:.2f}%")
    st.dataframe(
        pd.DataFrame(
            {
                "输入字段": list(prediction_inputs.keys()),
                "输入值": list(prediction_inputs.values()),
            }
        ),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="banksys", layout="wide")
    st.title("banksys")
    st.caption("银行营销数据分析与认购预测系统")

    st.sidebar.header("分析筛选")

    try:
        train_df = load_dataset(TRAIN_DATA_PATH)
        test_df = load_dataset(TEST_DATA_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.subheader("数据集概览")
    col1, col2, col3 = st.columns(3)
    col1.metric("训练集样本数", len(train_df))
    col2.metric("测试集样本数", len(test_df))
    col3.metric("训练集字段数", train_df.shape[1])

    render_analysis_page(train_df)
    render_training_status()
    render_prediction_page(train_df)


if __name__ == "__main__":
    main()
