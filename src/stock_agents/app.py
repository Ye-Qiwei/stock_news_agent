from __future__ import annotations

from datetime import date, timedelta
import logging
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

from stock_agents.config import RAG_DIR
from stock_agents.industry import infer_industry
from stock_agents.news_graph import GraphState, build_graph
from stock_agents.news_service import NewsService
from stock_agents.price_service import PriceService
from stock_agents.rag_store import clear_rag, get_rag_size_bytes
from stock_agents.ticker import infer_company_name, infer_ticker, normalize_ticker_for_market


logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
if not (ROOT_DIR / "mcp_servers").exists():
    ROOT_DIR = Path(__file__).resolve().parents[3]
PRICE_SERVER = str(ROOT_DIR / "mcp_servers" / "price_server.py")
NEWS_SERVER = str(ROOT_DIR / "mcp_servers" / "news_server.py")


def _bytes_to_mb(value: int) -> float:
    return value / (1024 * 1024)


def _filter_range(frame: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    today = frame["date"].max().date()
    if range_key == "3m":
        start = today - timedelta(days=90)
    elif range_key == "1y":
        start = today - timedelta(days=365)
    else:
        start = today - timedelta(days=365 * 5)
    return frame[frame["date"].dt.date >= start]


def _weekly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        frame.set_index("date")["close"]
        .resample("W-FRI")
        .last()
        .dropna()
        .reset_index()
    )
    weekly.rename(columns={"close": "price"}, inplace=True)
    return weekly


def _build_chart(weekly: pd.DataFrame) -> go.Figure:
    weekly = weekly.copy()
    weekly["price"] = pd.to_numeric(weekly["price"], errors="coerce")
    weekly = weekly.dropna(subset=["price"]).sort_values("date")
    x_vals = weekly["date"].dt.to_pydatetime()
    y_vals = weekly["price"].astype(float).tolist()

    y_min = min(y_vals) if y_vals else 0.0
    y_max = max(y_vals) if y_vals else 1.0
    pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Weekly",
            marker={"size": 7},
            line={"width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        margin={"l": 70, "r": 30, "t": 20, "b": 70},
        xaxis_title="Week",
        yaxis_title="Close",
        yaxis={"range": [y_min - pad, y_max + pad]},
        hovermode="closest",
        autosize=True,
    )
    fig.update_xaxes(type="date", title_standoff=12, ticklabelposition="outside")
    fig.update_yaxes(
        tickformat=".2f",
        type="linear",
        title_standoff=12,
        ticklabelposition="outside",
    )
    logger.info("chart y_min=%s y_max=%s sample=%s", y_min, y_max, y_vals[:5])
    return fig


def _grey_fig(fig: go.Figure) -> go.Figure:
    faded = go.Figure(fig.to_dict())
    faded.update_traces(marker={"color": "#9ca3af"}, line={"color": "#9ca3af"}, opacity=0.4)
    faded.update_layout(
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f3f4f6",
    )
    return faded


@st.cache_data(ttl=3600, show_spinner=False)
def _get_price_data(ticker: str, market: str, server_path: str) -> tuple[pd.DataFrame, str]:
    price_service = PriceService(price_server_script=server_path)
    return price_service.fetch_price(ticker=ticker.strip(), market=market)


def _render_news(items, title: str) -> None:
    scores = [item["score"] for item in items]
    avg_score = sum(scores) / len(scores) if scores else 0
    sentiment = "利好" if avg_score > 0 else "不利" if avg_score < 0 else "中性"
    for item in items:
        logger.info("news %s title=%s score=%s", title, item.get("title", ""), item.get("score"))
    st.subheader(f"{title} (平均: {avg_score:.2f}, {sentiment})")
    if not items:
        st.info("未检索到相关新闻。")
        return
    for item in items:
        st.markdown(f"**{item['title']}**")
        st.markdown("<br/>".join(item["summary"]))
        st.caption(item["link"])
        st.write("---")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    st.set_page_config(page_title="Stock News Agents", layout="wide")
    st.title("股票股价 + 历史新闻多智能体系统")

    with st.sidebar:
        st.header("查询参数")
        ticker = st.text_input("Ticker (用于股价/可留空自动识别)", value="AAPL")
        company_name = st.text_input("公司名(用于新闻搜索/可留空自动识别)", value="")
        market = st.selectbox("市场", ["US", "JP"], index=0)
        range_options = {"3个月": "3m", "1年": "1y", "5年": "5y"}
        range_label = st.radio("时间范围", list(range_options.keys()), index=1, horizontal=True)
        range_key = range_options[range_label]
        if st.button("清理 RAG"):
            clear_rag()
            st.success("RAG 已清理")
        rag_size = _bytes_to_mb(get_rag_size_bytes())
        st.metric("RAG 占用 (MB)", f"{rag_size:.2f}")
        st.progress(min(rag_size / 500, 1.0))

    if not ticker and company_name:
        inferred = infer_ticker(company_name, market)
        if inferred:
            ticker = inferred
            st.sidebar.success(f"已根据公司名识别 ticker: {ticker}")
    if ticker:
        normalized = normalize_ticker_for_market(ticker, market)
        if normalized and normalized != ticker:
            ticker = normalized
            st.sidebar.info(f"已根据市场调整 ticker: {ticker}")
    if not ticker:
        st.info("请输入股票或基金的 ticker，或填写公司名自动识别。")
        return

    st.sidebar.caption(f"MCP price server: {PRICE_SERVER}")
    st.sidebar.caption(f"存在: {Path(PRICE_SERVER).exists()}")

    fetch_start = time.perf_counter()
    frame, source = _get_price_data(ticker=ticker.strip(), market=market, server_path=PRICE_SERVER)
    logger.info("price fetch elapsed=%.3fs", time.perf_counter() - fetch_start)
    if frame.empty:
        st.error("未获取到股价数据，请检查 ticker 与市场。")
        return
    logger.info(
        "price frame rows=%d date_min=%s date_max=%s close_min=%s close_max=%s",
        len(frame),
        frame["date"].min(),
        frame["date"].max(),
        frame["close"].min(),
        frame["close"].max(),
    )

    frame = _filter_range(frame, range_key)
    weekly = _weekly_frame(frame)
    weekly["price"] = pd.to_numeric(weekly["price"], errors="coerce")
    weekly = weekly.dropna(subset=["price"])
    logger.info(
        "weekly rows=%d price_nunique=%d price_min=%s price_max=%s",
        len(weekly),
        weekly["price"].nunique(),
        weekly["price"].min(),
        weekly["price"].max(),
    )
    logger.info("weekly dtypes=%s", weekly.dtypes.to_dict())
    logger.info("weekly sample=%s", weekly.head(3).to_dict(orient="records"))
    logger.info("daily rows=%d close_min=%s close_max=%s", len(frame), frame["close"].min(), frame["close"].max())
    if weekly.empty:
        st.error("时间范围内无有效数据。")
        return

    st.caption(f"股价来源: {source}")
    chart_placeholder = st.empty()
    current_key = f"{ticker}-{market}-{range_key}"
    if "last_fig" in st.session_state and st.session_state.get("last_key") != current_key:
        chart_placeholder.plotly_chart(_grey_fig(st.session_state["last_fig"]), use_container_width=True)

    with st.spinner("正在更新股价图表..."):
        fig = _build_chart(weekly)
        chart_placeholder.empty()
        selected = plotly_events(
            fig,
            click_event=True,
            select_event=False,
            override_height=520,
            override_width="100%",
            key=current_key,
        )
        st.session_state["last_fig"] = fig
        st.session_state["last_key"] = current_key

    if selected:
        selected_date = pd.to_datetime(selected[0]["x"]).date()
        week_start = selected_date - timedelta(days=6)
        st.info(f"已选择周: {week_start} ~ {selected_date}")

        with st.spinner("正在获取新闻并总结..."):
            industry = infer_industry(ticker)
            effective_company = company_name.strip()
            if not effective_company:
                inferred_name = infer_company_name(ticker, market)
                if inferred_name:
                    effective_company = inferred_name
                    st.info(f"已根据 ticker 推断公司名: {effective_company}")
            news_service = NewsService(news_server_script=NEWS_SERVER)
            graph = build_graph(news_service)
            state = GraphState(
                ticker=ticker,
                week_start=week_start,
                industry=industry,
                company_name=effective_company,
            )
            result = graph.invoke(state)

        col1, col2 = st.columns(2)
        with col1:
            _render_news(result["company_news"], "公司相关新闻")
        with col2:
            _render_news(result["industry_news"], "行业相关新闻")


if __name__ == "__main__":
    main()
