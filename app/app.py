from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
MONTH_ORDER = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
TARGET_LABELS = {"yes": "认购", "no": "未认购"}


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


if __name__ == "__main__":
    main()
