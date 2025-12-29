import io
import json
import math
import os
import base64
import re
import ast
import codecs
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI


# ------------------------------
# Streamlit Cloud Secrets Support
# ------------------------------
def load_secrets():
    """Load API keys from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        if hasattr(st, 'secrets'):
            if 'OPENAI_API_KEY' in st.secrets:
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
            if 'E2B_API_KEY' in st.secrets:
                os.environ['E2B_API_KEY'] = st.secrets['E2B_API_KEY']
    except Exception:
        pass  # Secrets not available, fall back to env vars


# Load secrets at module import
load_secrets()

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - optional dependency
    IsolationForest = None  # type: ignore


# ------------------------------
# Configuration and constants
# ------------------------------
DEFAULT_TIME_GRAIN = "W"  # weekly
ANOMALY_Z_THRESHOLD = 2.5
MAX_PREVIEW_ROWS = 25
REPORT_FILENAME = "insights_report.txt"


@dataclass
class Schema:
    date_col: Optional[str]
    metric_col: Optional[str]
    group_col: Optional[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]


@dataclass
class Insight:
    text: str
    severity: str = "info"  # info | positive | negative | anomaly


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _decode_str_to_bytes(text: str) -> Optional[bytes]:
    if not isinstance(text, str):
        return None
    s = text.strip()

    # 1) data URI: data:image/png;base64,XXXXX
    if s.startswith("data:image/"):
        try:
            comma = s.find(",")
            if comma != -1:
                b64 = s[comma + 1 :].strip()
                data = base64.b64decode(b64, validate=False)
                if data:
                    return data
        except Exception:
            pass

    # 2) Plain base64
    try:
        data = base64.b64decode(s, validate=False)
        if data and (data.startswith(PNG_MAGIC) or data.startswith(JPEG_MAGIC)):
            return data
    except Exception:
        pass

    # 3) Python bytes literal string, e.g., "b'...'" or "b\"...\""
    if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (bytes, bytearray)):
                b = bytes(val)
                if b:
                    return b
        except Exception:
            pass

    # 4) Escaped hex sequences like "\\x89PNG\\r\\n..."
    try:
        unescaped = codecs.decode(s, "unicode_escape")
        b2 = unescaped.encode("latin1", errors="ignore")
        if b2 and (b2.startswith(PNG_MAGIC) or b2.startswith(JPEG_MAGIC)):
            return b2
    except Exception:
        pass

    # 5) Last resort: treat as latin1-encoded bytes
    try:
        b3 = s.encode("latin1", errors="ignore")
        if b3 and (b3.startswith(PNG_MAGIC) or b3.startswith(JPEG_MAGIC)):
            return b3
    except Exception:
        pass

    return None


def _coerce_bytes(image_like) -> Optional[bytes]:
    if image_like is None:
        return None
    if isinstance(image_like, (bytes, bytearray, memoryview)):
        return bytes(image_like)
    if isinstance(image_like, str):
        return _decode_str_to_bytes(image_like)
    return None


def _to_streamlit_image(image_like):
    b = _coerce_bytes(image_like)
    if b is None:
        return None
    return io.BytesIO(b)


def _to_bytes(image_like) -> Optional[bytes]:
    return _coerce_bytes(image_like)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Trim column names and deduplicate by appending suffixes
    df = df.copy()
    raw_cols = list(df.columns)
    cleaned = []
    for idx, c in enumerate(raw_cols):
        name = "Unnamed" if c is None or (isinstance(c, str) and c.strip() == "") else str(c).strip()
        cleaned.append(name)
    counts: Dict[str, int] = {}
    unique_cols = []
    for name in cleaned:
        if name not in counts:
            counts[name] = 1
            unique_cols.append(name)
        else:
            counts[name] += 1
            unique_cols.append(f"{name}__{counts[name]}")
    df.columns = unique_cols
    return df


def safe_get_series(df: pd.DataFrame, col: str) -> pd.Series:
    # Handles duplicate-named columns by picking the first occurrence as Series
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def set_page() -> None:
    st.set_page_config(
        page_title="Data-to-Insight AI Analyst",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📈 Data-to-Insight AI Analyst")
    st.caption(
        "Upload a CSV or Excel file to automatically generate trends, anomalies, and business recommendations."
    )


# ------------------------------
# Data loading and schema
# ------------------------------
def safe_read(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    # Try CSV as a fallback
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)


def guess_datetime_columns(df: pd.DataFrame) -> List[str]:
    dt_cols = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            dt_cols.append(col)
            continue
        if any(tok in col.lower() for tok in ["date", "time", "timestamp", "week", "month", "year"]):
            # Attempt parse; tolerate errors
            try:
                parsed = pd.to_datetime(series, errors="raise")
                if parsed.notna().mean() > 0.6:
                    df[col] = pd.to_datetime(series, errors="coerce")
                    dt_cols.append(col)
            except Exception:
                pass
    return dt_cols


def infer_schema(df: pd.DataFrame) -> Schema:
    numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_columns = [
        c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    datetime_columns = guess_datetime_columns(df)

    date_col = datetime_columns[0] if datetime_columns else None

    # choose metric with largest sum as default
    metric_col = None
    if numeric_columns:
        totals = {c: pd.to_numeric(df[c], errors="coerce").fillna(0).sum() for c in numeric_columns}
        metric_col = max(totals, key=totals.get)

    # choose first categorical as group
    group_col = categorical_columns[0] if categorical_columns else None

    return Schema(
        date_col=date_col,
        metric_col=metric_col,
        group_col=group_col,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )


# ------------------------------
# Analysis helpers
# ------------------------------
def aggregate_time_series(
    df: pd.DataFrame, date_col: str, metric_col: str, time_grain: str = DEFAULT_TIME_GRAIN
) -> pd.DataFrame:
    # Ensure datetime
    ts = pd.DataFrame(
        {
            date_col: safe_get_series(df, date_col),
            metric_col: pd.to_numeric(safe_get_series(df, metric_col), errors="coerce"),
        }
    )
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    ts = ts.sort_values(date_col).set_index(date_col)
    ts_agg = ts.resample(time_grain)[metric_col].sum().to_frame(name=metric_col)
    ts_agg["period"] = ts_agg.index
    ts_agg = ts_agg.reset_index(drop=True)
    return ts_agg


def compute_wow_change(ts_agg: pd.DataFrame, metric_col: str) -> Optional[float]:
    if len(ts_agg) < 2:
        return None
    last, prev = ts_agg[metric_col].iloc[-1], ts_agg[metric_col].iloc[-2]
    if prev == 0:
        return None
    return (last - prev) / abs(prev)


def compute_linear_trend(ts_agg: pd.DataFrame, metric_col: str) -> Optional[float]:
    if len(ts_agg) < 3:
        return None
    y = ts_agg[metric_col].values.astype(float)
    x = np.arange(len(y))
    slope, _intercept = np.polyfit(x, y, 1)
    # Relative slope per period compared to mean scale
    denom = np.mean(y) if np.mean(y) != 0 else 1.0
    return slope / denom


def top_group_movers(
    df: pd.DataFrame, date_col: str, metric_col: str, group_col: str, time_grain: str
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    if date_col is None or group_col is None:
        return pd.DataFrame(), None, None
    try:
        tmp = pd.DataFrame(
            {
                date_col: safe_get_series(df, date_col),
                group_col: safe_get_series(df, group_col),
                metric_col: pd.to_numeric(safe_get_series(df, metric_col), errors="coerce"),
            }
        )
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        tmp = tmp.sort_values(date_col)
        grp = (
            tmp.set_index(date_col)
            .groupby(group_col)[metric_col]
            .resample(time_grain)
            .sum()
            .reset_index()
        )
        # For each group, calc WoW change for last 2 periods
        last_two = grp.groupby(group_col).tail(2)
        pivot = last_two.pivot_table(index=group_col, columns=date_col, values=metric_col, aggfunc="sum").dropna(
            axis=0, how="any"
        )
        if pivot.shape[1] < 2:
            return grp, None, None
        cols = sorted(pivot.columns)
        prev_col, last_col = cols[-2], cols[-1]
        change = (pivot[last_col] - pivot[prev_col]).div(pivot[prev_col].replace(0, np.nan)).dropna()
        if change.empty:
            return grp, None, None
        best = change.sort_values(ascending=False).head(1)
        worst = change.sort_values(ascending=True).head(1)
        return grp, best, worst
    except Exception:
        return pd.DataFrame(), None, None


def detect_anomalies_zscore(ts_agg: pd.DataFrame, metric_col: str, threshold: float) -> pd.DataFrame:
    values = ts_agg[metric_col].astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        ts_agg["z"] = 0.0
    else:
        ts_agg["z"] = (values - mean) / std
    anomalies = ts_agg[ts_agg["z"].abs() >= threshold].copy()
    return anomalies


def detect_anomalies_iforest(ts_agg: pd.DataFrame, metric_col: str, contamination: float = 0.1) -> pd.DataFrame:
    if IsolationForest is None:
        return pd.DataFrame(columns=list(ts_agg.columns) + ["score"])
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    X = ts_agg[[metric_col]].astype(float).values
    model.fit(X)
    score = model.decision_function(X)
    pred = model.predict(X)  # -1 anomaly
    out = ts_agg.copy()
    out["iforest_score"] = score
    out["iforest_label"] = pred
    return out[out["iforest_label"] == -1]


def humanize_pct(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "n/a"
    return f"{x*100:.1f}%"


def generate_insights_and_recs(
    df: pd.DataFrame,
    schema: Schema,
    time_grain: str,
    enable_anomaly: bool,
    anomaly_method: str,
    z_threshold: float,
    contamination: float,
) -> Tuple[List[Insight], List[str], Dict]:
    debug: Dict = {}
    insights: List[Insight] = []
    recs: List[str] = []

    date_col, metric_col, group_col = schema.date_col, schema.metric_col, schema.group_col
    if not metric_col:
        insights.append(Insight("No numeric metric column detected. Please select one.", "negative"))
        return insights, recs, debug

    # Overall summary
    total = pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum()
    insights.append(Insight(f"Total {metric_col}: {total:,.0f}", "info"))

    # Time series analysis if date available
    if date_col:
        ts_agg = aggregate_time_series(df, date_col, metric_col, time_grain)
        debug["ts_points"] = len(ts_agg)
        if len(ts_agg) >= 2:
            wow = compute_wow_change(ts_agg, metric_col)
            trend = compute_linear_trend(ts_agg, metric_col)
            if wow is not None:
                sev = "positive" if wow >= 0 else "negative"
                direction = "increased" if wow >= 0 else "decreased"
                insights.append(Insight(f"{metric_col} {direction} {humanize_pct(wow)} vs prior period.", sev))
            if trend is not None and abs(trend) > 0.01:
                sev = "positive" if trend > 0 else "negative"
                direction = "upward" if trend > 0 else "downward"
                insights.append(Insight(f"Underlying {direction} trend detected over time (slope≈{trend:.2f} per period).", sev))

            # Group movers
            if group_col:
                grp, best, worst = top_group_movers(df, date_col, metric_col, group_col, time_grain)
                if best is not None and not best.empty:
                    g = best.index[0]
                    insights.append(Insight(f"Top mover: '{g}' {humanize_pct(float(best.iloc[0]))} WoW.", "positive"))
                if worst is not None and not worst.empty:
                    g = worst.index[0]
                    insights.append(Insight(f"Biggest decline: '{g}' {humanize_pct(float(worst.iloc[0]))} WoW.", "negative"))

            # Anomalies
            if enable_anomaly and len(ts_agg) >= 6:
                if anomaly_method == "Z-score":
                    anomalies = detect_anomalies_zscore(ts_agg, metric_col, z_threshold)
                else:
                    anomalies = detect_anomalies_iforest(ts_agg, metric_col, contamination)
                debug["anomaly_count"] = len(anomalies)
                for _, row in anomalies.tail(3).iterrows():
                    dt = row.get("period", None)
                    val = row[metric_col]
                    insights.append(Insight(f"Anomaly on {pd.to_datetime(dt).date() if pd.notna(dt) else 'period'}: {metric_col}={val:,.0f}", "anomaly"))

    else:
        # No date column: show top categories by metric
        if schema.categorical_columns:
            first_cat = schema.categorical_columns[0]
            ranking = (
                df.groupby(first_cat)[metric_col]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            top_items = ", ".join([f"{idx} ({val:,.0f})" for idx, val in ranking.items()])
            insights.append(Insight(f"Top {first_cat} by {metric_col}: {top_items}", "info"))

    # Recommendations based on signals
    # Heuristic: if negative WoW or downward trend or anomalies, propose actions
    neg_signals = any(i.severity in ("negative", "anomaly") for i in insights)
    pos_signals = any(i.severity == "positive" for i in insights)
    if neg_signals:
        recs.append("Investigate drivers for declining segments; review pricing, promotions, and inventory constraints.")
        recs.append("Drill into affected regions/categories to isolate root causes (channel mix, product availability).")
    if pos_signals and not neg_signals:
        recs.append("Double down on growth areas (expand stock, amplify campaigns, or replicate playbooks).")
    if not recs:
        recs.append("Establish a weekly monitoring cadence with alerts for significant week-over-week changes.")

    return insights, recs, debug


def build_report_text(
    file_name: str,
    schema: Schema,
    insights: List[Insight],
    recs: List[str],
) -> str:
    lines = []
    lines.append(f"Data-to-Insight Report: {file_name}")
    lines.append("-" * 60)
    lines.append("Schema:")
    lines.append(f"  Date column:   {schema.date_col}")
    lines.append(f"  Metric column: {schema.metric_col}")
    lines.append(f"  Group column:  {schema.group_col}")
    lines.append("")
    lines.append("Insights:")
    for i, ins in enumerate(insights, 1):
        lines.append(f"  {i}. {ins.text}")
    lines.append("")
    lines.append("Recommendations:")
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. {r}")
    return "\n".join(lines)


# ------------------------------
# UI
# ------------------------------
from helper import load_env  # type: ignore
from lib.utils import create_sandbox  # type: ignore
from lib.tools import tools as sandbox_tools  # type: ignore
from lib.tools_schemas import execute_code_schema, execute_bash_schema  # type: ignore
from lib.coding_agent import coding_agent, log  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# CHAT WITH DATA - Conversational AI for follow-up questions and visualizations
# ══════════════════════════════════════════════════════════════════════════════

def initialize_chat_session():
    """Initialize chat session state with memory."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = {}
    if "chat_visualizations" not in st.session_state:
        st.session_state.chat_visualizations = []
    if "chat_sandbox" not in st.session_state:
        st.session_state.chat_sandbox = None


def get_data_context_summary(data: dict, df_summary: Optional[str] = None) -> str:
    """Build context summary from analysis results for chat."""
    context_parts = []
    
    # Business summary
    if data.get("business_summary"):
        context_parts.append(f"Business Summary: {data['business_summary']}")
    
    # Overview metrics
    overview = data.get("overview", {})
    if overview:
        context_parts.append(f"Main Metric: {overview.get('main_metric', 'unknown')}")
        context_parts.append(f"Total Value: {overview.get('total_value', 'N/A')}")
        context_parts.append(f"WoW Change: {overview.get('latest_wow_change', 'N/A')}%")
        context_parts.append(f"Trend: {overview.get('trend_direction', 'N/A')}")
        context_parts.append(f"Date Range: {overview.get('date_range', 'N/A')}")
    
    # Key insights
    insights = data.get("insights", [])
    if insights:
        context_parts.append(f"Key Insights: {'; '.join(insights[:5])}")
    
    # Patterns detected
    patterns = data.get("detected_patterns", [])
    if patterns:
        pattern_strs = [str(p) if isinstance(p, str) else p.get("pattern", str(p)) for p in patterns[:5]]
        context_parts.append(f"Detected Patterns: {'; '.join(pattern_strs)}")
    
    # Recommendations
    recs = data.get("recommendations", [])
    if recs:
        context_parts.append(f"Recommendations: {'; '.join(recs[:3])}")
    
    # DataFrame summary if available
    if df_summary:
        context_parts.append(f"Data Shape: {df_summary}")
    
    return "\n".join(context_parts)


def process_chat_query(query: str, data_context: str, dataset_url: Optional[str], 
                       uploaded_file, sbx) -> Tuple[str, Optional[bytes], Optional[str]]:
    """Process a chat query and return response with optional visualization."""
    load_env()
    client = OpenAI()
    
    # Build the chat system prompt
    chat_system = f"""You are an AI Data Analyst assistant. You have access to the analysis results and can:
1. Answer questions about the data, insights, and patterns
2. Generate new visualizations on request
3. Provide deeper analysis on specific segments or metrics
4. Explain patterns and recommend actions

## Current Analysis Context:
{data_context}

## Your Capabilities:
- Answer questions about the data using the context above
- If the user asks for a visualization, write Python code to generate it
- If you need to run code, output it in a ```python code block
- Be specific with numbers and insights from the context
- When asked to visualize, always save the chart as 'chat_chart.png' using matplotlib with Agg backend

## Response Format:
- For questions: Provide clear, concise answers based on the analysis
- For visualizations: Generate Python code and explain what it shows
- Always be helpful and data-driven
"""

    # Get conversation history for context
    history = st.session_state.get("chat_messages", [])[-10:]  # Last 10 messages for context
    
    messages = [{"role": "system", "content": chat_system}]
    
    # Add conversation history
    for msg in history:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        
        # Check if response contains Python code for visualization
        chart_bytes = None
        code_output = None
        
        if "```python" in assistant_response and sbx is not None:
            # Extract Python code
            import re
            code_match = re.search(r'```python\n(.*?)```', assistant_response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                
                # Ensure matplotlib uses Agg backend
                if "matplotlib" in code and "Agg" not in code:
                    code = "import matplotlib\nmatplotlib.use('Agg')\n" + code
                
                # Add dataset loading if needed
                if dataset_url and "read_csv" not in code and "read_excel" not in code:
                    if dataset_url.lower().endswith('.csv'):
                        code = f"import pandas as pd\ndf = pd.read_csv('{dataset_url}')\n" + code
                    else:
                        code = f"import pandas as pd\ndf = pd.read_excel('{dataset_url}', engine='openpyxl')\n" + code
                
                try:
                    # Execute code in sandbox
                    exec_result = sbx.run_code(code, language="python")
                    if exec_result and exec_result.logs:
                        code_output = "\n".join(exec_result.logs.stdout) if exec_result.logs.stdout else ""
                    
                    # Try to read generated chart
                    try:
                        chart_content = sbx.files.read("chat_chart.png")
                        if isinstance(chart_content, (bytes, bytearray)):
                            chart_bytes = bytes(chart_content)
                        elif isinstance(chart_content, str):
                            # Try base64 decode
                            chart_bytes = base64.b64decode(chart_content)
                    except Exception:
                        # Try reading via code execution
                        read_code = """
import base64
import os
if os.path.exists('chat_chart.png'):
    with open('chat_chart.png', 'rb') as f:
        print(base64.b64encode(f.read()).decode('ascii'))
"""
                        read_result = sbx.run_code(read_code)
                        if read_result and read_result.logs and read_result.logs.stdout:
                            b64_str = "".join(read_result.logs.stdout).strip()
                            if b64_str:
                                chart_bytes = base64.b64decode(b64_str)
                except Exception as e:
                    code_output = f"Code execution error: {str(e)}"
        
        return assistant_response, chart_bytes, code_output
        
    except Exception as e:
        return f"Error processing your question: {str(e)}", None, None


def render_chat_interface(data: dict, dataset_url: Optional[str], uploaded_file, sbx):
    """Render the chat interface in sidebar or main area."""
    
    # Initialize session
    initialize_chat_session()
    
    # Store data context
    st.session_state.chat_context = data
    st.session_state.chat_sandbox = sbx
    
    # Build context summary
    data_context = get_data_context_summary(data)
    
    st.markdown("---")
    st.subheader("💬 Chat with Your Data")
    st.caption("Ask follow-up questions, request visualizations, or explore patterns")
    
    # Example questions
    with st.expander("💡 Example Questions", expanded=False):
        example_questions = [
            "Which region performed best last quarter?",
            "Show me a pie chart of revenue by category",
            "What's causing the decline in Region B?",
            "Create a comparison chart of top 5 vs bottom 5 segments",
            "Explain the anomaly detected on Dec 15",
            "What actions should I prioritize this week?",
            "Show monthly trend instead of weekly",
            "Which product category has the highest growth rate?"
        ]
        for q in example_questions:
            if st.button(f"📝 {q}", key=f"example_{hash(q)}", use_container_width=True):
                st.session_state.pending_question = q
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Display visualization if present
                if message.get("visualization"):
                    try:
                        st.image(message["visualization"], caption="Generated Visualization", use_container_width=True)
                    except Exception:
                        pass
    
    # Chat input
    user_input = st.chat_input("Ask a question about your data...")
    
    # Check for pending question from examples
    if hasattr(st.session_state, 'pending_question') and st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    
    if user_input:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response, chart_bytes, code_output = process_chat_query(
                    user_input, data_context, dataset_url, uploaded_file, sbx
                )
                
                st.markdown(response)
                
                # Store message with visualization if present
                msg_entry = {"role": "assistant", "content": response}
                
                if chart_bytes and len(chart_bytes) > 0:
                    try:
                        st.image(chart_bytes, caption="Generated Visualization", use_container_width=True)
                        msg_entry["visualization"] = chart_bytes
                        
                        # Add download button
                        st.download_button(
                            "📥 Download Chart",
                            data=chart_bytes,
                            file_name="chat_visualization.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.warning(f"Could not display chart: {e}")
                
                if code_output:
                    with st.expander("📋 Code Output"):
                        st.code(code_output)
                
                st.session_state.chat_messages.append(msg_entry)
    
    # Clear chat button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    with col2:
        if st.button("📤 Export Chat", use_container_width=True):
            chat_export = "\n\n".join([
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in st.session_state.chat_messages
            ])
            st.download_button(
                "Download",
                data=chat_export,
                file_name="chat_history.txt",
                mime="text/plain",
                key="download_chat"
            )


def prepare_sandbox_for_analysis(sbx) -> None:
    # Install required libs inside sandbox (idempotent and quiet)
    try:
        sbx.run_code(
            "pip install -q pandas numpy scikit-learn scipy matplotlib seaborn plotly openpyxl requests",
            language="bash",
        )
    except Exception:
        pass


def run_agent_in_sandbox(dataset_url: Optional[str], uploaded, prompt_type: str = "combined") -> Tuple[Optional[dict], Optional[str]]:
    load_env()
    client = OpenAI()

    # 1) Create/reuse sandbox and prep
    sbx = create_sandbox()  # reuse same name if running
    prepare_sandbox_for_analysis(sbx)

    # 2) Stage data (prefer URL)
    local_path = None
    if dataset_url and dataset_url.strip():
        data_instruction = f"Load the dataset from URL '{dataset_url.strip()}'. If it ends with .csv use pandas.read_csv; otherwise try pandas.read_excel(engine='openpyxl')."
    elif uploaded is not None:
        # Upload to sandbox as text for csv; excel best via URL
        fname = uploaded.name or "data.csv"
        if fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls"):
            try:
                sbx.files.write(fname, uploaded.getvalue())
            except Exception:
                return None, "Excel upload to sandbox may not be supported here. Prefer a URL/presigned link."
        else:
            try:
                content = uploaded.getvalue().decode("utf-8", errors="ignore")
                sbx.files.write(fname, content)
            except Exception:
                return None, "Failed to push CSV to sandbox. Try using a dataset URL."
        local_path = f"./{fname}"
        data_instruction = f"Load the dataset from local file path '{local_path}'. Use pandas.read_csv for .csv, pandas.read_excel(engine='openpyxl') for Excel."
    else:
        return None, "Provide a dataset URL or upload a file."

    # 3) Agent directive (load from prompts.json with prompt_type selection)
    prompts_dir = os.path.dirname(__file__)
    system = None
    query_template = None
    
    # Default fallback prompt
    default_system = (
        "You are an expert Data Scientist. Perform analysis in TWO PHASES.\n\n"
        "## PHASE 1: Basic Analysis\n"
        "1. Load data, parse dates, identify main numeric metric (largest sum)\n"
        "2. Aggregate by week/time period\n"
        "3. Calculate: total, WoW % change, trend slope\n"
        "4. Create chart.png (time series line) and group_top.png (bar chart)\n\n"
        "## PHASE 2: Pattern Detection\n"
        "1. For each segment (Region, Category, etc.), calculate weekly values\n"
        "2. Find anomalies: Z-score > 2 or WoW change > 30%\n"
        "3. Create heatmap.png if multiple groups exist\n\n"
        "## OUTPUT: Save ./insights.json with insights, recommendations, and meta fields.\n"
        "## CHARTS: Save as PNG with matplotlib Agg backend."
    )
    default_query = (
        "{DATASET_INSTRUCTION}\n"
        "Infer date/metric/group columns. Pick the numeric metric with the largest total. "
        "Write ./insights.json and save charts as PNG files."
    )
    
    try:
        with open(os.path.join(prompts_dir, "prompts.json"), "r", encoding="utf-8") as f:
            pconf = json.load(f)
            # New structure: prompts are organized by type
            if prompt_type and prompt_type in pconf:
                prompt_config = pconf[prompt_type]
                system = prompt_config.get("system")
                query_template = prompt_config.get("query_template")
            # Fallback to old structure for backwards compatibility
            elif "system" in pconf:
                system = pconf.get("system")
                query_template = pconf.get("query_template")
    except Exception:
        pass
    
    # Use defaults if not loaded
    system = system or default_system
    query_template = query_template or default_query
    
    # Build the final query
    query = query_template.replace("{DATASET_INSTRUCTION}", data_instruction)

    try:
        # 4) Drive the agent with limited toolset
        messages, usage = log(
            coding_agent,
            messages=[],
            query=query,
            client=client,
            system=system,
            tools_schemas=[execute_code_schema, execute_bash_schema],
            tools=sandbox_tools,
            max_steps=15,
            sbx=sbx,
        )

        # Build a lightweight run log from messages and capture inline images
        run_log_lines: List[str] = []
        images_from_metadata: List[bytes] = []
        for part in messages:
            if "type" in part:
                if part["type"] == "function_call":
                    name = part.get("name", "")
                    arguments = part.get("arguments", "")
                    run_log_lines.append(f"> CALL {name} args={arguments[:600]}")
                elif part["type"] == "function_call_output":
                    out = part.get("output", "")
                    run_log_lines.append(f"< RESULT {out[:1200]}")
            elif "role" in part:
                if part["role"] == "assistant" and part.get("content"):
                    run_log_lines.append(f"ASSISTANT: {str(part['content'])[:800]}")
                elif part["role"] == "user" and part.get("content"):
                    run_log_lines.append(f"USER: {str(part['content'])[:400]}")
            meta = part.get("_metadata") if isinstance(part, dict) else None
            if meta and isinstance(meta, dict):
                imgs = meta.get("images")
                if imgs and isinstance(imgs, list):
                    for img in imgs:
                        # Accept raw PNG/JPEG bytes or base64 strings
                        try:
                            if isinstance(img, (bytes, bytearray)):
                                if img[:8] == b"\x89PNG\r\n\x1a\n" or img[:3] == b"\xff\xd8\xff":
                                    images_from_metadata.append(bytes(img))
                                else:
                                    # Attempt base64 decode if it's base64 in bytes form
                                    decoded = base64.b64decode(img, validate=False)
                                    if decoded:
                                        images_from_metadata.append(decoded)
                            elif isinstance(img, str):
                                decoded = base64.b64decode(img, validate=False)
                                if decoded:
                                    images_from_metadata.append(decoded)
                        except Exception:
                            continue
        run_log_lines.append(f"tokens_used={usage}")
        run_log = "\n".join(run_log_lines)

        # 5) Read results (robust)
        def _read_text_file(path: str) -> Optional[str]:
            try:
                content = sbx.files.read(path)
                if isinstance(content, (bytes, bytearray)):
                    return content.decode("utf-8", errors="ignore")
                if isinstance(content, str):
                    return content
            except Exception:
                return None
            return None


        data: Dict = {}
        for p in ["insights.json", "/home/user/insights.json"]:
            txt = _read_text_file(p)
            if txt:
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict) and ("insights" in parsed or "recommendations" in parsed):
                        data = parsed
                        break
                except Exception:
                    pass

        # Fallback: parse last tool output's embedded results
        if not data:
            for part in reversed(messages):
                if isinstance(part, dict) and part.get("type") == "function_call_output":
                    try:
                        out = part.get("output", "{}")
                        parsed = json.loads(out)
                        results = parsed.get("results") if isinstance(parsed, dict) else None
                        if isinstance(results, list):
                            for r in results:
                                if isinstance(r, dict):
                                    js = r.get("json")
                                    if isinstance(js, dict) and ("insights" in js or "recommendations" in js):
                                        data = js
                                        break
                                    # sometimes JSON is in text
                                    txt = r.get("text")
                                    if isinstance(txt, str):
                                        try:
                                            js2 = json.loads(txt)
                                            if isinstance(js2, dict) and ("insights" in js2 or "recommendations" in js2):
                                                data = js2
                                                break
                                        except Exception:
                                            pass
                            if data:
                                break
                    except Exception:
                        continue

        # Helper to read binary file from sandbox as base64
        def _read_binary_file_via_code(filename: str) -> Optional[bytes]:
            """Read binary file by running code in sandbox to get base64."""
            try:
                code = f"""
import base64
import os
if os.path.exists({repr(filename)}):
    with open({repr(filename)}, 'rb') as f:
        print(base64.b64encode(f.read()).decode('ascii'))
else:
    print('')
"""
                exec_result = sbx.run_code(code)
                if exec_result and exec_result.logs and exec_result.logs.stdout:
                    b64_str = "".join(exec_result.logs.stdout).strip()
                    if b64_str:
                        return base64.b64decode(b64_str)
            except Exception:
                pass
            return None

        # Optional charts - use code execution to read binary files properly
        chart_bytes = None
        group_chart_bytes = None
        heatmap_bytes = None
        patterns_bytes = None
        
        # Try reading chart.png via code execution (avoids UTF-8 corruption)
        chart_bytes = _read_binary_file_via_code("chart.png")
        
        # Try reading group_top.png via code execution
        group_chart_bytes = _read_binary_file_via_code("group_top.png")
        
        # Try reading additional pattern analysis charts
        heatmap_bytes = _read_binary_file_via_code("heatmap.png")
        patterns_bytes = _read_binary_file_via_code("patterns.png")

        # Fallback to inline images returned via tool metadata
        if (chart_bytes is None or (isinstance(chart_bytes, (bytes, bytearray)) and len(chart_bytes) == 0)) and images_from_metadata:
            chart_bytes = images_from_metadata[0]
        if (group_chart_bytes is None or (isinstance(group_chart_bytes, (bytes, bytearray)) and len(group_chart_bytes) == 0)) and len(images_from_metadata) > 1:
            group_chart_bytes = images_from_metadata[1]

        # Persist any recovered images to sandbox so they are downloadable
        try:
            if isinstance(chart_bytes, (bytes, bytearray)) and len(chart_bytes) > 0:
                try:
                    sbx.files.write("chart.png", bytes(chart_bytes))
                except Exception:
                    pass
            if isinstance(group_chart_bytes, (bytes, bytearray)) and len(group_chart_bytes) > 0:
                try:
                    sbx.files.write("group_top.png", bytes(group_chart_bytes))
                except Exception:
                    pass
        except Exception:
            pass

        # Fallback to any PNGs found in the working directory
        if (chart_bytes is None or (isinstance(chart_bytes, (bytes, bytearray)) and len(chart_bytes) == 0)) or (
            group_chart_bytes is None or (isinstance(group_chart_bytes, (bytes, bytearray)) and len(group_chart_bytes) == 0)
        ):
            try:
                entries = sbx.files.list("")
                png_names = [e.name for e in entries if getattr(e, "name", "").lower().endswith(".png")]
                # Prefer known names first
                ordered = []
                for pref in ["chart.png", "group_top.png"]:
                    if pref in png_names:
                        ordered.append(pref)
                for n in png_names:
                    if n not in ordered:
                        ordered.append(n)
                if (chart_bytes is None or (isinstance(chart_bytes, (bytes, bytearray)) and len(chart_bytes) == 0)) and ordered:
                    try:
                        tmp = sbx.files.read(ordered[0])
                        chart_bytes = _to_bytes(tmp) or chart_bytes
                    except Exception:
                        pass
                if (group_chart_bytes is None or (isinstance(group_chart_bytes, (bytes, bytearray)) and len(group_chart_bytes) == 0)) and len(ordered) > 1:
                    try:
                        tmp = sbx.files.read(ordered[1])
                        group_chart_bytes = _to_bytes(tmp) or group_chart_bytes
                    except Exception:
                        pass
            except Exception:
                pass

        # Final fallback: try to generate charts directly inside sandbox from the dataset source
        if (
            (chart_bytes is None or (isinstance(chart_bytes, (bytes, bytearray)) and len(chart_bytes) == 0))
            or (group_chart_bytes is None or (isinstance(group_chart_bytes, (bytes, bytearray)) and len(group_chart_bytes) == 0))
        ):
            try:
                src = dataset_url or local_path
                if src:
                    gen_code = (
                        "import pandas as pd\n"
                        "import matplotlib; matplotlib.use('Agg')\n"
                        "import matplotlib.pyplot as plt\n"
                        "import seaborn as sns\n"
                        "import numpy as np\n"
                        f"src = {repr(src)}\n"
                        "df = None\n"
                        "try:\n"
                        "    if isinstance(src, str) and src.lower().endswith('.csv'):\n"
                        "        df = pd.read_csv(src)\n"
                        "    else:\n"
                        "        df = pd.read_excel(src, engine='openpyxl')\n"
                        "except Exception:\n"
                        "    try:\n"
                        "        df = pd.read_csv(src)\n"
                        "    except Exception:\n"
                        "        df = None\n"
                        "if df is not None and len(df.columns) > 0:\n"
                        "    cols = list(df.columns)\n"
                        "    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]\n"
                        "    # guess date column\n"
                        "    dt_cols = []\n"
                        "    for c in cols:\n"
                        "        try:\n"
                        "            if any(tok in c.lower() for tok in ['date','time','week','month','year','period']):\n"
                        "                parsed = pd.to_datetime(df[c], errors='coerce')\n"
                        "                if parsed.notna().mean() > 0.5:\n"
                        "                    df[c] = parsed\n"
                        "                    dt_cols.append(c)\n"
                        "        except Exception:\n"
                        "            pass\n"
                        "    date_col = dt_cols[0] if dt_cols else None\n"
                        "    metric_col = None\n"
                        "    if num_cols:\n"
                        "        metric_col = max(num_cols, key=lambda c: pd.to_numeric(df[c], errors='coerce').fillna(0).sum())\n"
                        "    cat_cols = [c for c in cols if df[c].dtype == 'object']\n"
                        "    group_col = cat_cols[0] if cat_cols else None\n"
                        "    if date_col and metric_col:\n"
                        "        ts = pd.DataFrame({'d': pd.to_datetime(df[date_col], errors='coerce'), 'm': pd.to_numeric(df[metric_col], errors='coerce')})\n"
                        "        ts = ts.dropna(subset=['d']).sort_values('d').set_index('d')\n"
                        "        if not ts.empty:\n"
                        "            ts_agg = ts.resample('W')['m'].sum().reset_index()\n"
                        "            plt.figure(figsize=(10,6))\n"
                        "            sns.lineplot(data=ts_agg, x='d', y='m')\n"
                        "            plt.title(f\"{metric_col} over time\")\n"
                        "            plt.tight_layout()\n"
                        "            plt.savefig('chart.png', dpi=150)\n"
                        "            plt.close()\n"
                        "    if group_col and metric_col:\n"
                        "        top = df.groupby(group_col)[metric_col].sum().sort_values(ascending=False).head(10).reset_index()\n"
                        "        if not top.empty:\n"
                        "            plt.figure(figsize=(10,6))\n"
                        "            sns.barplot(data=top, x=group_col, y=metric_col)\n"
                        "            plt.xticks(rotation=30, ha='right')\n"
                        "            plt.tight_layout()\n"
                        "            plt.savefig('group_top.png', dpi=150)\n"
                        "            plt.close()\n"
                    )
                    try:
                        sbx.run_code(gen_code, language="python")
                    except Exception:
                        pass

                    # attempt to read generated images
                    try:
                        tmp = sbx.files.read("chart.png")
                        if tmp is not None:
                            b = _to_bytes(tmp) or b""
                            if len(b) > 0:
                                chart_bytes = b if (chart_bytes is None or (isinstance(chart_bytes, (bytes, bytearray)) and len(chart_bytes) == 0)) else chart_bytes
                    except Exception:
                        pass
                    try:
                        tmp = sbx.files.read("group_top.png")
                        if tmp is not None:
                            b = _to_bytes(tmp) or b""
                            if len(b) > 0:
                                group_chart_bytes = b if (group_chart_bytes is None or (isinstance(group_chart_bytes, (bytes, bytearray)) and len(group_chart_bytes) == 0)) else group_chart_bytes
                    except Exception:
                        pass
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════
        # VISION LLM ANALYSIS: Analyze charts with GPT-4 Vision
        # ═══════════════════════════════════════════════════════════════
        vision_analysis = {}
        
        def analyze_chart_with_vision(image_bytes: bytes, chart_type: str, context: str = "") -> Optional[str]:
            """Use GPT-4 Vision to analyze a chart image and provide insights."""
            if not image_bytes or len(image_bytes) == 0:
                return None
            try:
                # Encode image to base64
                b64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                # Create vision prompt based on chart type
                prompts = {
                    "time_series": (
                        "You are a data analyst. Analyze this time series chart and provide:\n"
                        "1. **Overall Trend**: Is it increasing, decreasing, or stable?\n"
                        "2. **Key Patterns**: Any seasonality, cycles, or notable fluctuations?\n"
                        "3. **Anomalies**: Any unusual spikes or dips? When did they occur?\n"
                        "4. **Business Insight**: What does this mean for the business?\n"
                        "Be specific with values and dates if visible. Keep response concise (3-4 sentences per point)."
                    ),
                    "group_chart": (
                        "You are a data analyst. Analyze this bar/group comparison chart and provide:\n"
                        "1. **Top Performers**: Which segments are leading?\n"
                        "2. **Underperformers**: Which segments are lagging?\n"
                        "3. **Gaps**: What's the difference between best and worst?\n"
                        "4. **Recommendation**: What action should be taken?\n"
                        "Be specific with segment names and values if visible. Keep response concise."
                    ),
                    "heatmap": (
                        "You are a data analyst. Analyze this heatmap and provide:\n"
                        "1. **Hot Spots**: Which cells/areas show highest values?\n"
                        "2. **Cold Spots**: Which areas show lowest values?\n"
                        "3. **Patterns**: Any row/column patterns or clusters?\n"
                        "4. **Insight**: What cross-dimensional insight does this reveal?\n"
                        "Be specific with row/column names if visible."
                    ),
                    "patterns": (
                        "You are a data analyst. Analyze this pattern visualization and provide:\n"
                        "1. **Key Patterns**: What patterns are visible?\n"
                        "2. **Correlations**: Any relationships between variables?\n"
                        "3. **Outliers**: Any unusual data points?\n"
                        "4. **Actionable Insight**: What should the business do?\n"
                        "Be specific and actionable."
                    )
                }
                
                prompt = prompts.get(chart_type, prompts["time_series"])
                if context:
                    prompt += f"\n\nContext: {context}"
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Vision analysis unavailable: {str(e)}"
        
        # Analyze each chart if available
        context_str = f"Main metric: {data.get('overview', {}).get('main_metric', 'unknown')}" if data else ""
        
        if chart_bytes and len(chart_bytes) > 0:
            vision_analysis["time_series"] = analyze_chart_with_vision(chart_bytes, "time_series", context_str)
        
        if group_chart_bytes and len(group_chart_bytes) > 0:
            vision_analysis["group_chart"] = analyze_chart_with_vision(group_chart_bytes, "group_chart", context_str)
        
        if heatmap_bytes and len(heatmap_bytes) > 0:
            vision_analysis["heatmap"] = analyze_chart_with_vision(heatmap_bytes, "heatmap", context_str)
        
        if patterns_bytes and len(patterns_bytes) > 0:
            vision_analysis["patterns"] = analyze_chart_with_vision(patterns_bytes, "patterns", context_str)

        return {
            "data": data, 
            "chart_bytes": chart_bytes, 
            "group_chart_bytes": group_chart_bytes, 
            "heatmap_bytes": heatmap_bytes, 
            "patterns_bytes": patterns_bytes, 
            "vision_analysis": vision_analysis,
            "log": run_log, 
            "usage": usage
        }, None
    except Exception as e:
        return None, str(e)


def sidebar_controls(df: Optional[pd.DataFrame]) -> Tuple[Optional[str], Optional[str], Optional[str], str, bool, str, float, float]:
    st.sidebar.header("Settings")
    date_col = metric_col = group_col = None
    time_grain = st.sidebar.selectbox("Time grain", ["W (weekly)", "D (daily)", "M (monthly)"], index=0)
    time_grain = time_grain.split()[0]

    enable_anomaly = st.sidebar.checkbox("Enable anomaly detection", value=True)
    method = st.sidebar.selectbox("Anomaly method", ["Z-score", "IsolationForest"], index=0, disabled=not enable_anomaly)
    z_th = st.sidebar.slider("Z-score threshold", min_value=1.5, max_value=4.0, value=ANOMALY_Z_THRESHOLD, step=0.1, disabled=not enable_anomaly or method != "Z-score")
    contam = st.sidebar.slider("IsolationForest contamination", min_value=0.02, max_value=0.3, value=0.1, step=0.01, disabled=not enable_anomaly or method != "IsolationForest")

    if df is not None:
        schema = infer_schema(df)
        with st.sidebar.expander("Column selection", expanded=True):
            date_col = st.selectbox(
                "Date column (optional)", [None] + schema.datetime_columns, index=0 if schema.date_col is None else (1 + schema.datetime_columns.index(schema.date_col))
                if schema.date_col in schema.datetime_columns else 0
            )
            metric_col = st.selectbox(
                "Metric (numeric)",
                schema.numeric_columns if schema.numeric_columns else [None],
                index=(schema.numeric_columns.index(schema.metric_col) if schema.metric_col in schema.numeric_columns else 0) if schema.numeric_columns else 0,
            )
            group_col = st.selectbox(
                "Group by (categorical, optional)",
                [None] + schema.categorical_columns,
                index=0 if (schema.group_col is None or schema.group_col not in schema.categorical_columns) else (1 + schema.categorical_columns.index(schema.group_col)),
            )
    return date_col, metric_col, group_col, time_grain, enable_anomaly, method, float(z_th), float(contam)


def main() -> None:
    set_page()

    # Remote-only UI (always visible)
    st.header("🤖 Agent (Remote Sandbox)")
    st.caption("Run the analysis fully inside an E2B sandbox. Prefer a dataset URL/presigned URL so the sandbox downloads the file directly. The dataset never goes to the LLM.")

    remote_cols = st.columns([2, 1])
    with remote_cols[0]:
        dataset_url = st.text_input("Dataset URL (preferred)", placeholder="https://.../your-data.csv or .xlsx", key="agent_url_top")
    with remote_cols[1]:
        uploaded_remote = st.file_uploader("Or upload to sandbox", type=["csv", "xlsx", "xls"], key="agent_upload_top")

    with st.expander("System check (deployment readiness)"):
        st.write(f"OPENAI_API_KEY set: {'yes' if os.getenv('OPENAI_API_KEY') else 'no'}")
        st.write(f"E2B_API_KEY set: {'yes' if os.getenv('E2B_API_KEY') else 'no'}")
        st.write("Best practice: keep keys as environment variables in deployment (do not hardcode).")

    st.info("Best place to run E2B: from this server-side app (Streamlit process). Provide presigned dataset URLs so the sandbox downloads the file directly over HTTPS.")

    # Prompt type selector
    st.markdown("**Analysis Type**")
    prompt_options = {
        "combined": "🔄 Two-Phase Analysis (Basic + Deep Patterns)",
        "data_analysis": "📊 Basic Data Analysis (Trends, WoW, Anomalies)",
        "deep_insights": "🔬 Deep Pattern Analysis (Segment-level, Cross-dimensional)"
    }
    if "agent_prompt_type" not in st.session_state:
        st.session_state["agent_prompt_type"] = "combined"
    
    selected_prompt = st.selectbox(
        "Select analysis type",
        options=list(prompt_options.keys()),
        format_func=lambda x: prompt_options[x],
        index=list(prompt_options.keys()).index(st.session_state["agent_prompt_type"]),
        key="agent_prompt_selector_top",
        help="Choose the type of analysis to run. 'Two-Phase' is recommended for comprehensive insights."
    )
    st.session_state["agent_prompt_type"] = selected_prompt

    if "agent_auto_run" not in st.session_state:
        st.session_state["agent_auto_run"] = True
    if "agent_last_url" not in st.session_state:
        st.session_state["agent_last_url"] = ""
    if "agent_last_upload_name" not in st.session_state:
        st.session_state["agent_last_upload_name"] = ""
    
    # ═══════════════════════════════════════════════════════════════
    # RESULTS CACHE - Store results per analysis type
    # ═══════════════════════════════════════════════════════════════
    if "agent_results_cache" not in st.session_state:
        st.session_state["agent_results_cache"] = {}  # {prompt_type: {"result": ..., "error": ...}}
    
    # Legacy support - migrate old single result to cache
    if "agent_result" in st.session_state and st.session_state["agent_result"] is not None:
        old_type = st.session_state.get("agent_last_prompt_type", "combined")
        if old_type not in st.session_state["agent_results_cache"]:
            st.session_state["agent_results_cache"][old_type] = {
                "result": st.session_state["agent_result"],
                "error": st.session_state.get("agent_error")
            }
        st.session_state["agent_result"] = None
        st.session_state["agent_error"] = None
    
    if "agent_last_prompt_type" not in st.session_state:
        st.session_state["agent_last_prompt_type"] = "combined"

    auto_run = st.checkbox("Auto-run when input changes", value=st.session_state["agent_auto_run"], key="agent_autorun_top")
    st.session_state["agent_auto_run"] = auto_run

    changed_url = bool(dataset_url and dataset_url != st.session_state["agent_last_url"])
    changed_upload = bool(uploaded_remote and uploaded_remote.name != st.session_state["agent_last_upload_name"])
    
    # Get current prompt type
    current_prompt_type = st.session_state.get("agent_prompt_type", "combined")
    
    # Check if we have cached results for current analysis type
    cached_entry = st.session_state["agent_results_cache"].get(current_prompt_type)
    has_cached_result = cached_entry is not None and cached_entry.get("result") is not None

    def run_and_store():
        prompt_type = st.session_state.get("agent_prompt_type", "combined")
        with st.spinner(f"Running {prompt_options.get(prompt_type, 'analysis')}..."):
            result, err = run_agent_in_sandbox(dataset_url, uploaded_remote, prompt_type)
        
        # Store in cache by prompt type
        st.session_state["agent_results_cache"][prompt_type] = {
            "result": result,
            "error": err
        }
        st.session_state["agent_last_prompt_type"] = prompt_type
        st.session_state["agent_last_url"] = dataset_url or st.session_state["agent_last_url"]
        st.session_state["agent_last_upload_name"] = uploaded_remote.name if uploaded_remote else st.session_state["agent_last_upload_name"]

    # Clear cache if dataset changes
    if changed_url or changed_upload:
        st.session_state["agent_results_cache"] = {}
        if auto_run:
            run_and_store()

    # Show cached results indicator
    if has_cached_result:
        st.success(f"✅ Showing cached results for **{prompt_options.get(current_prompt_type, current_prompt_type)}**")
        col_run, col_clear = st.columns(2)
        with col_run:
            if st.button("🔄 Re-run Analysis", type="primary", key="agent_run_top"):
                run_and_store()
        with col_clear:
            if st.button("🗑️ Clear All Cache", key="agent_clear_cache"):
                st.session_state["agent_results_cache"] = {}
                st.rerun()
    else:
        if st.button("▶️ Run Agent in E2B Sandbox", type="primary", key="agent_run_top"):
            run_and_store()

    # Get result from cache for current prompt type
    cached_entry = st.session_state["agent_results_cache"].get(current_prompt_type, {})
    result = cached_entry.get("result")
    err = cached_entry.get("error")

    if err:
        st.error(err)
    elif result:
        data = result.get("data", {})
        insights = data.get("insights", [])
        recs = data.get("recommendations", [])

        # ═══════════════════════════════════════════════════════════════
        # BUSINESS SUMMARY (Executive Overview)
        # ═══════════════════════════════════════════════════════════════
        business_summary = data.get("business_summary", "")
        if business_summary:
            st.markdown("---")
            st.subheader("📋 Executive Business Summary")
            # Use native Streamlit container for reliable rendering
            with st.container():
                st.info(f"📊 **Executive Overview**\n\n{business_summary}")
            st.markdown("---")

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: DATA OVERVIEW
        # ═══════════════════════════════════════════════════════════════
        overview = data.get("overview", {})
        if overview:
            st.subheader("📊 Phase 1: Data Overview")
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_val = overview.get("total_value", 0)
                if isinstance(total_val, (int, float)) and total_val > 0:
                    st.metric("Total Revenue", f"${total_val:,.0f}" if total_val > 1000 else f"{total_val:,.2f}")
            with col2:
                wow = overview.get("latest_wow_change")
                if wow is not None:
                    st.metric("Latest WoW Change", f"{wow:+.2f}%", delta=f"{wow:+.2f}%")
            with col3:
                trend = overview.get("trend_direction", "")
                trend_icon = "📈" if trend == "increasing" else "📉" if trend == "decreasing" else "➡️"
                st.metric("Trend", f"{trend_icon} {trend.title()}" if trend else "N/A")
            with col4:
                slope = overview.get("trend_slope")
                if slope is not None:
                    st.metric("Weekly Change", f"{slope:+,.0f}/week" if abs(slope) > 100 else f"{slope:+,.2f}/week")
            
            # Date range info
            date_range = overview.get("date_range", "")
            main_metric = overview.get("main_metric", "")
            if date_range or main_metric:
                st.caption(f"📅 Period: {date_range} | 📏 Main Metric: {main_metric}")
            
            st.divider()

        # Optional agent narrative extracted from run log
        run_log_text = result.get("log", "") or ""
        if run_log_text:
            lines = [ln for ln in run_log_text.splitlines() if ln.startswith("ASSISTANT: ")]
            if lines:
                with st.expander("🤖 Agent Summary", expanded=False):
                    st.markdown(lines[-1].replace("ASSISTANT: ", "", 1))

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: DEEP PATTERN ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        detected_patterns = data.get("detected_patterns", [])
        segment_anomalies = data.get("segment_anomalies", [])
        
        if detected_patterns or segment_anomalies:
            st.subheader("🔬 Phase 2: Deep Pattern Analysis")
        
        if detected_patterns:
            st.markdown("#### 🔍 Detected Patterns")
            
            # Helper to extract pattern info (handles multiple dict formats and strings)
            def extract_pattern_info(p):
                if isinstance(p, dict):
                    # Try multiple possible keys for the pattern text
                    pattern_text = (
                        p.get("pattern") or 
                        p.get("description") or 
                        p.get("text") or 
                        p.get("message") or 
                        p.get("finding") or
                        p.get("insight") or
                        p.get("detail") or
                        None
                    )
                    
                    # If still no pattern text, try to build one from other fields
                    if not pattern_text:
                        # Try to construct from segment/metric/change info
                        segment = p.get("segment") or p.get("region") or p.get("category") or p.get("group") or ""
                        metric = p.get("metric") or p.get("value") or ""
                        change = p.get("metric_change") or p.get("change") or p.get("wow_change") or p.get("percent_change") or ""
                        period = p.get("period") or p.get("week") or p.get("date") or ""
                        channel = p.get("channel") or ""
                        
                        if segment or metric or change:
                            parts = []
                            if segment:
                                parts.append(f"{segment}")
                            if channel:
                                parts.append(f"({channel})")
                            if metric:
                                parts.append(f": {metric}")
                            if change:
                                parts.append(f" [{change}% WoW]" if "%" not in str(change) else f" [{change} WoW]")
                            if period:
                                parts.append(f" on {period}")
                            pattern_text = "".join(parts) if parts else None
                    
                    # Last resort: stringify the entire dict
                    if not pattern_text:
                        # Remove common metadata keys and show remaining content
                        display_dict = {k: v for k, v in p.items() if k not in ["severity", "priority", "type"]}
                        if display_dict:
                            pattern_text = ", ".join([f"{k}: {v}" for k, v in display_dict.items()])
                        else:
                            pattern_text = str(p)
                    
                    # Extract change percentage from various possible keys
                    change_val = (
                        p.get("metric_change") or 
                        p.get("change") or 
                        p.get("wow_change") or 
                        p.get("percent_change") or 
                        p.get("pct_change") or
                        ""
                    )
                    
                    return {
                        "severity": p.get("severity", "medium"),
                        "pattern": pattern_text,
                        "period": p.get("period") or p.get("week") or p.get("date") or "",
                        "change": change_val,
                        "rec": p.get("recommendation") or p.get("action") or p.get("suggestion") or ""
                    }
                else:
                    # Handle string format - try to extract % change for sorting
                    import re
                    change_match = re.search(r'([-+]?\d+\.?\d*)%', str(p))
                    change = change_match.group(0) if change_match else ""
                    return {
                        "severity": "high" if change_match and abs(float(change_match.group(1))) > 50 else "medium",
                        "pattern": str(p),
                        "period": "",
                        "change": change,
                        "rec": ""
                    }
            
            # Limit to top 10 patterns, sorted by severity and change magnitude
            def pattern_sort_key(p):
                info = extract_pattern_info(p)
                severity_rank = {"high": 0, "medium": 1, "low": 2}.get(info["severity"], 2)
                try:
                    change_val = abs(float(str(info["change"]).replace("%", "").replace("+", "")))
                except:
                    change_val = 0
                return (severity_rank, -change_val)
            
            sorted_patterns = sorted(detected_patterns, key=pattern_sort_key)
            top_patterns = sorted_patterns[:10]
            remaining_patterns = sorted_patterns[10:]
            
            # Display top patterns
            for pattern in top_patterns:
                info = extract_pattern_info(pattern)
                severity_icon = "🔴" if info["severity"] == "high" else "🟡" if info["severity"] == "medium" else "🟢"
                
                if info["rec"]:
                    st.markdown(f"""
**{severity_icon} {info["pattern"]}** {f'({info["period"]})' if info["period"] else ''}
- 📊 Change: `{info["change"]}`
- 💡 Action: _{info["rec"]}_
---
""")
                else:
                    st.markdown(f"**{severity_icon}** {info['pattern']}")
            
            # Show remaining patterns in expander if any
            if remaining_patterns:
                with st.expander(f"📋 Show {len(remaining_patterns)} more patterns...", expanded=False):
                    for pattern in remaining_patterns:
                        info = extract_pattern_info(pattern)
                        severity_icon = "🔴" if info["severity"] == "high" else "🟡" if info["severity"] == "medium" else "🟢"
                        st.markdown(f"**{severity_icon}** {info['pattern']}")
        
        # Display segment anomalies (limit to top 5)
        if segment_anomalies:
            st.markdown("#### ⚠️ Segment Anomalies")
            top_anomalies = segment_anomalies[:5]
            remaining_anomalies = segment_anomalies[5:]
            
            def extract_anomaly_info(anomaly):
                """Extract anomaly info from various dict formats or strings."""
                if isinstance(anomaly, dict):
                    # Try multiple possible keys for segment
                    segment = (
                        anomaly.get("segment") or 
                        anomaly.get("region") or 
                        anomaly.get("category") or 
                        anomaly.get("group") or 
                        anomaly.get("name") or
                        anomaly.get("dimension") or
                        "Unknown"
                    )
                    
                    # Try multiple possible keys for issue/description
                    issue = (
                        anomaly.get("issue") or 
                        anomaly.get("description") or 
                        anomaly.get("anomaly") or 
                        anomaly.get("finding") or
                        anomaly.get("pattern") or
                        anomaly.get("text") or
                        anomaly.get("message") or
                        ""
                    )
                    
                    # If still no issue text, try to build from metric info
                    if not issue:
                        change = anomaly.get("change") or anomaly.get("metric_change") or anomaly.get("wow_change") or ""
                        metric = anomaly.get("metric") or anomaly.get("value") or ""
                        z_score = anomaly.get("z_score") or anomaly.get("zscore") or ""
                        
                        parts = []
                        if metric:
                            parts.append(f"Metric: {metric}")
                        if change:
                            parts.append(f"Change: {change}%" if "%" not in str(change) else f"Change: {change}")
                        if z_score:
                            parts.append(f"Z-score: {z_score}")
                        issue = ", ".join(parts) if parts else "Anomaly detected"
                    
                    # Try multiple possible keys for gap/impact
                    gap = (
                        anomaly.get("gap") or 
                        anomaly.get("impact") or 
                        anomaly.get("severity") or 
                        anomaly.get("magnitude") or
                        anomaly.get("z_score") or
                        anomaly.get("change") or
                        ""
                    )
                    
                    rec = anomaly.get("recommendation") or anomaly.get("action") or anomaly.get("suggestion") or ""
                    
                    return segment, issue, gap, rec
                else:
                    # String format
                    return "Anomaly", str(anomaly), "", ""
            
            for anomaly in top_anomalies:
                segment, issue, gap, rec = extract_anomaly_info(anomaly)
                gap_text = f" ({gap})" if gap else ""
                st.warning(f"**{segment}**: {issue}{gap_text}")
                if rec:
                    st.caption(f"💡 Recommendation: {rec}")
            
            # Show remaining anomalies in expander if any
            if remaining_anomalies:
                with st.expander(f"📋 Show {len(remaining_anomalies)} more anomalies...", expanded=False):
                    for anomaly in remaining_anomalies:
                        segment, issue, gap, rec = extract_anomaly_info(anomaly)
                        gap_text = f" ({gap})" if gap else ""
                        st.caption(f"⚠️ **{segment}**: {issue}{gap_text}")

        # Standard insights
        st.subheader("📋 Key Insights")
        
        # Fallback: If insights is empty but we have detected_patterns or overview, generate insights
        display_insights = insights.copy() if insights else []
        
        if not display_insights:
            # Try to build insights from overview data
            overview = data.get("overview", {})
            if overview:
                main_metric = overview.get("main_metric", "metric")
                total_val = overview.get("total_value")
                wow = overview.get("latest_wow_change")
                trend = overview.get("trend_direction", "")
                date_range = overview.get("date_range", "")
                
                if total_val is not None:
                    display_insights.append(f"Total {main_metric}: ${total_val:,.0f}" if isinstance(total_val, (int, float)) else f"Total {main_metric}: {total_val}")
                if wow is not None:
                    direction = "increased" if wow >= 0 else "decreased"
                    display_insights.append(f"{main_metric.title()} {direction} {abs(wow):.2f}% week-over-week")
                if trend:
                    display_insights.append(f"Overall trend: {trend.title()}")
                if date_range:
                    display_insights.append(f"Analysis period: {date_range}")
            
            # Also include top patterns as insights if available
            patterns = data.get("detected_patterns", [])
            for p in patterns[:3]:  # Add top 3 patterns
                if isinstance(p, dict):
                    pattern_text = p.get("pattern", str(p))
                elif isinstance(p, str):
                    pattern_text = p
                else:
                    pattern_text = str(p)
                if pattern_text and pattern_text not in display_insights:
                    display_insights.append(pattern_text)
        
        if display_insights:
            for t in display_insights:
                st.info(f"{t}")
        else:
            st.info("No insights returned.")

        st.subheader("🎯 Recommendations")
        
        # Fallback: Build recommendations from patterns if empty
        display_recs = recs.copy() if recs else []
        
        if not display_recs:
            # Generate recommendations based on patterns and anomalies
            patterns = data.get("detected_patterns", [])
            anomalies = data.get("segment_anomalies", [])
            
            if patterns or anomalies:
                # Add generic recommendations based on data
                if any("decline" in str(p).lower() or "drop" in str(p).lower() or "decrease" in str(p).lower() for p in patterns):
                    display_recs.append("Investigate drivers for declining segments; review pricing, promotions, and inventory constraints.")
                if any("increase" in str(p).lower() or "growth" in str(p).lower() or "spike" in str(p).lower() for p in patterns):
                    display_recs.append("Double down on growth areas; expand stock, amplify campaigns, or replicate successful playbooks.")
                if anomalies:
                    display_recs.append("Review anomalous segments for root cause analysis and corrective actions.")
            
            if not display_recs:
                display_recs.append("Establish a weekly monitoring cadence with alerts for significant changes.")
        
        for r in display_recs:
            st.success(f"➡️ {r}")

        st.subheader("📊 Visualizations")
        chart_explanations = data.get("chart_explanations", {})
        
        # Primary charts row
        chart_cols = st.columns(2)
        
        # Get vision analysis from result
        vision_analysis = result.get("vision_analysis", {})
        
        if result.get("chart_bytes"):
            with chart_cols[0]:
                chart_blob = result["chart_bytes"]
                img = _to_streamlit_image(chart_blob)
                if img is not None:
                    st.image(img, caption="📈 Trend Over Time (with Anomalies)", use_container_width=True)
                    # Show AI Vision Analysis if available
                    if vision_analysis.get("time_series"):
                        with st.expander("🤖 AI Analysis of This Chart", expanded=True):
                            st.markdown(vision_analysis["time_series"])
                    elif chart_explanations.get("time_series"):
                        st.info(f"**What this shows:** {chart_explanations['time_series']}")
                    else:
                        st.info("**What this shows:** This line chart displays your main metric over time. Red markers highlight detected anomalies.")
        
        if result.get("group_chart_bytes"):
            with chart_cols[1]:
                group_chart_blob = result["group_chart_bytes"]
                img2 = _to_streamlit_image(group_chart_blob)
                if img2 is not None:
                    st.image(img2, caption="📊 Segment Performance", use_container_width=True)
                    # Show AI Vision Analysis if available
                    if vision_analysis.get("group_chart"):
                        with st.expander("🤖 AI Analysis of This Chart", expanded=True):
                            st.markdown(vision_analysis["group_chart"])
                    elif chart_explanations.get("group_chart"):
                        st.info(f"**What this shows:** {chart_explanations['group_chart']}")
                    else:
                        st.info("**What this shows:** This bar chart compares performance across segments. Taller bars = higher contribution.")
        
        # Additional pattern analysis charts
        has_additional_charts = result.get("heatmap_bytes") or result.get("patterns_bytes")
        if has_additional_charts:
            st.subheader("🔬 Deep Pattern Analysis")
            pattern_cols = st.columns(2)
            
            if result.get("heatmap_bytes"):
                with pattern_cols[0]:
                    heatmap_blob = result["heatmap_bytes"]
                    img3 = _to_streamlit_image(heatmap_blob)
                    if img3 is not None:
                        st.image(img3, caption="🗺️ Performance Heatmap", use_container_width=True)
                        # Show AI Vision Analysis if available
                        if vision_analysis.get("heatmap"):
                            with st.expander("🤖 AI Analysis of Heatmap", expanded=True):
                                st.markdown(vision_analysis["heatmap"])
                        elif chart_explanations.get("heatmap"):
                            st.info(f"**What this shows:** {chart_explanations['heatmap']}")
                        else:
                            st.info("**What this shows:** This heatmap reveals performance patterns across multiple dimensions.")
            
            if result.get("patterns_bytes"):
                with pattern_cols[1]:
                    patterns_blob = result["patterns_bytes"]
                    img4 = _to_streamlit_image(patterns_blob)
                    if img4 is not None:
                        st.image(img4, caption="🎯 Detected Patterns", use_container_width=True)
                        # Show AI Vision Analysis if available
                        if vision_analysis.get("patterns"):
                            with st.expander("🤖 AI Analysis of Patterns", expanded=True):
                                st.markdown(vision_analysis["patterns"])
                        elif chart_explanations.get("anomaly_chart"):
                            st.info(f"**What this shows:** {chart_explanations['anomaly_chart']}")
                        else:
                            st.info("**What this shows:** This visualization highlights specific patterns and anomalies detected in your data.")
        
        # Display actionable next steps
        next_actions = data.get("next_actions", [])
        if next_actions:
            st.subheader("🎯 Recommended Next Actions")
            st.markdown("*Prioritized actions based on data analysis:*")
            
            for action_item in next_actions:
                if isinstance(action_item, dict):
                    priority = action_item.get("priority", 3)
                    action = action_item.get("action", "")
                    impact = action_item.get("impact", "")
                    owner = action_item.get("owner", "")
                    
                    # Priority-based styling using Streamlit native components
                    if priority == 1:
                        priority_icon = "🔴"
                        priority_label = "URGENT"
                    elif priority == 2:
                        priority_icon = "🟡"
                        priority_label = "HIGH"
                    else:
                        priority_icon = "🟢"
                        priority_label = "MEDIUM"
                    
                    # Use native Streamlit components for reliable rendering
                    with st.container():
                        col_icon, col_content = st.columns([0.08, 0.92])
                        with col_icon:
                            st.markdown(f"### {priority_icon}")
                        with col_content:
                            owner_text = f" • 👤 {owner}" if owner else ""
                            st.markdown(f"**{priority_label}**{owner_text}")
                            st.markdown(f"{action}")
                            if impact:
                                st.success(f"📈 **Expected Impact:** {impact}")
                        st.divider()
                else:
                    # Fallback for string format
                    st.success(f"➡️ {action_item}")
        else:
            # Fallback: show recommendations as next actions if next_actions is empty
            if recs:
                st.subheader("🎯 Recommended Next Actions")
                for i, r in enumerate(recs, 1):
                    st.success(f"**{i}.** {r}")

        with st.expander("Artifacts (debug)"):
            cb = result.get("chart_bytes")
            gb = result.get("group_chart_bytes")

            def info_for(blob):
                info = {"present": blob is not None}
                if blob is not None:
                    try:
                        info["type"] = type(blob).__name__
                        b = _to_bytes(blob) or b""
                        info["len"] = len(b) if b is not None else None
                        info["head"] = b[:8].hex() if b else None
                    except Exception:
                        pass
                return info

            st.json({"chart_bytes": info_for(cb), "group_chart_bytes": info_for(gb)})

            cb_bytes = _to_bytes(cb) if cb is not None else None
            gb_bytes = _to_bytes(gb) if gb is not None else None

            if cb_bytes:
                st.download_button("Download chart.png", data=cb_bytes, file_name="chart.png", mime="image/png")
            if gb_bytes:
                st.download_button("Download group_top.png", data=gb_bytes, file_name="group_top.png", mime="image/png")

        with st.expander("Raw insights.json"):
            st.code(json.dumps(data, indent=2))

        with st.expander("Agent run log"):
            st.code(result.get("log", ""))
            usage = result.get("usage")
            if usage is not None:
                st.caption(f"Tokens used: {usage}")

        # ═══════════════════════════════════════════════════════════════
        # CHAT WITH YOUR DATA - Interactive Q&A and Visualization
        # ═══════════════════════════════════════════════════════════════
        # Get or create sandbox for chat
        try:
            chat_sbx = create_sandbox()
            prepare_sandbox_for_analysis(chat_sbx)
        except Exception:
            chat_sbx = None
        
        render_chat_interface(data, dataset_url, uploaded_remote, chat_sbx)

    # Return early to keep a single, remote-only UI
    return

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=False
    )

    if not uploaded_file:
        st.info("Upload a dataset to begin. Tip: Include a date column to unlock time-based insights.")
        st.markdown(
            "- Example metrics: revenue, sales, orders\n"
            "- Example dimensions: region, product, channel\n"
            "- Example time: date/week/month\n"
        )
        return

    try:
        df = safe_read(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    # Sanitize columns (trim and deduplicate) to avoid downstream errors
    df = sanitize_dataframe(df)

    st.subheader("Dataset Preview")
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]:,} columns")
    st.dataframe(df.head(MAX_PREVIEW_ROWS), width="stretch")

    date_col, metric_col, group_col, time_grain, enable_anomaly, method, z_th, contam = sidebar_controls(df)

    # Apply explicit selections into schema for analysis
    schema = infer_schema(df)
    schema.date_col = date_col or schema.date_col
    schema.metric_col = metric_col or schema.metric_col
    schema.group_col = group_col or schema.group_col

    # Run analysis (defensive against malformed inputs)
    try:
        insights, recs, debug = generate_insights_and_recs(
            df=df,
            schema=schema,
            time_grain=time_grain,
            enable_anomaly=enable_anomaly,
            anomaly_method=method,
            z_threshold=z_th,
            contamination=contam,
        )
    except Exception:
        st.warning("Analysis encountered a data formatting issue; showing previews only. Adjust column selections if needed.")
        insights, recs, debug = (
            [Insight("Basic summary only; analysis partially skipped due to data formatting.", "info")],
            ["Establish a weekly monitoring cadence with alerts for significant changes."],
            {},
        )

    # Visuals
    st.subheader("Key Visualizations")
    cols = st.columns(2)
    with cols[0]:
        if schema.date_col and schema.metric_col:
            try:
                ts_agg = aggregate_time_series(df, schema.date_col, schema.metric_col, time_grain)
                fig = px.line(
                    ts_agg,
                    x="period",
                    y=schema.metric_col,
                    title=f"{schema.metric_col} over time ({time_grain})",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")
            except Exception:
                st.info("Time series chart unavailable for this selection.")
    with cols[1]:
        if schema.group_col and schema.metric_col:
            try:
                top = (
                    df.groupby(schema.group_col)[schema.metric_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )
                if not top.empty:
                    fig = px.bar(
                        top,
                        x=schema.group_col,
                        y=schema.metric_col,
                        title=f"Top 10 {schema.group_col} by {schema.metric_col}",
                    )
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, width="stretch")
            except Exception:
                st.info("Group bar chart unavailable for this selection.")

    # Insights
    st.subheader("Insights")
    if not insights:
        st.info("No insights were generated. Try selecting a metric or enabling anomalies.")
    else:
        for ins in insights:
            if ins.severity == "positive":
                st.success(ins.text)
            elif ins.severity == "negative":
                st.warning(ins.text)
            elif ins.severity == "anomaly":
                st.error(ins.text)
            else:
                st.info(ins.text)

    # Recommendations
    st.subheader("Recommended Next Actions")
    for r in recs:
        st.write(f"- {r}")

    # Report download
    report_text = build_report_text(uploaded_file.name, schema, insights, recs)
    st.download_button(
        label="Download report",
        data=report_text.encode("utf-8"),
        file_name=REPORT_FILENAME,
        mime="text/plain",
    )

    with st.expander("Debug details"):
        st.json(
            {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "date_col": schema.date_col,
                "metric_col": schema.metric_col,
                "group_col": schema.group_col,
                "time_grain": time_grain,
                **debug,
            }
        )

    # ------------------------------
    # Agent (Remote Sandbox) runner
    # ------------------------------
    st.divider()
    st.header("🤖 Agent (Remote Sandbox)")
    st.caption("Run the analysis fully inside an E2B sandbox. Prefer providing a dataset URL/presigned URL so the sandbox downloads data directly. The dataset never goes to the LLM.")

    remote_cols = st.columns([2, 1])
    with remote_cols[0]:
        dataset_url = st.text_input("Dataset URL (preferred)", placeholder="https://.../your-data.csv or .xlsx")
    with remote_cols[1]:
        uploaded_remote = st.file_uploader(
            "Or upload to sandbox (CSV recommended)", type=["csv", "xlsx", "xls"], key="remote_upload"
        )

    with st.expander("System check (deployment readiness)"):
        st.write(f"OPENAI_API_KEY set: {'yes' if os.getenv('OPENAI_API_KEY') else 'no'}")
        st.write(f"E2B_API_KEY set: {'yes' if os.getenv('E2B_API_KEY') else 'no'}")
        st.write("Best practice: keep keys as environment variables in deployment (do not hardcode).")

    st.info("Best place to run E2B: from this server-side app (Streamlit process). Provide presigned dataset URLs so the sandbox downloads the file directly over HTTPS.")

    # Prompt type selector (second location)
    st.markdown("**Analysis Type**")
    prompt_options_2 = {
        "combined": "🔄 Two-Phase Analysis (Basic + Deep Patterns)",
        "data_analysis": "📊 Basic Data Analysis (Trends, WoW, Anomalies)",
        "deep_insights": "🔬 Deep Pattern Analysis (Segment-level, Cross-dimensional)"
    }
    if "agent_prompt_type" not in st.session_state:
        st.session_state["agent_prompt_type"] = "combined"
    
    selected_prompt_2 = st.selectbox(
        "Select analysis type",
        options=list(prompt_options_2.keys()),
        format_func=lambda x: prompt_options_2[x],
        index=list(prompt_options_2.keys()).index(st.session_state["agent_prompt_type"]),
        key="agent_prompt_selector_bottom",
        help="Choose the type of analysis to run. 'Two-Phase' is recommended for comprehensive insights."
    )
    st.session_state["agent_prompt_type"] = selected_prompt_2

    # Auto-run and manual run controls
    if "agent_auto_run" not in st.session_state:
        st.session_state["agent_auto_run"] = True
    if "agent_last_url" not in st.session_state:
        st.session_state["agent_last_url"] = ""
    if "agent_last_upload_name" not in st.session_state:
        st.session_state["agent_last_upload_name"] = ""
    if "agent_result" not in st.session_state:
        st.session_state["agent_result"] = None
    if "agent_error" not in st.session_state:
        st.session_state["agent_error"] = None

    use_above_upload = uploaded_file is not None and st.checkbox("Use dataset uploaded above", value=True)
    effective_upload = uploaded_file if use_above_upload and uploaded_file is not None else uploaded_remote

    auto_run = st.checkbox("Auto-run when input changes", value=st.session_state["agent_auto_run"])
    st.session_state["agent_auto_run"] = auto_run

    changed_url = bool(dataset_url and dataset_url != st.session_state["agent_last_url"])
    changed_upload = bool(effective_upload and effective_upload.name != st.session_state["agent_last_upload_name"])

    def run_and_store():
        prompt_type = st.session_state.get("agent_prompt_type", "combined")
        with st.spinner("Running agent in sandbox..."):
            result, err = run_agent_in_sandbox(dataset_url, effective_upload, prompt_type)
        st.session_state["agent_result"] = result
        st.session_state["agent_error"] = err
        st.session_state["agent_last_url"] = dataset_url or st.session_state["agent_last_url"]
        st.session_state["agent_last_upload_name"] = effective_upload.name if effective_upload else st.session_state["agent_last_upload_name"]

    if auto_run and (changed_url or changed_upload):
        run_and_store()

    if st.button("Run Agent in E2B Sandbox", type="primary"):
        run_and_store()

    result = st.session_state.get("agent_result")
    err = st.session_state.get("agent_error")

    if err:
        st.error(err)
    elif result:
        data = result.get("data", {})
        insights = data.get("insights", [])
        recs = data.get("recommendations", [])

        st.subheader("Agent Insights")
        if insights:
            for t in insights:
                st.info(f"{t}")
        else:
            st.info("No insights returned.")

        st.subheader("Agent Recommendations")
        if recs:
            for r in recs:
                st.write(f"- {r}")
        else:
            st.write("- Establish a weekly monitoring cadence with alerts.")

        # Get vision analysis
        vision_analysis_2 = result.get("vision_analysis", {})
        
        chart_cols = st.columns(2)
        if result.get("chart_bytes"):
            try:
                with chart_cols[0]:
                    st.image(result["chart_bytes"], caption="Time Series / Primary Chart", use_container_width=True)
                    if vision_analysis_2.get("time_series"):
                        with st.expander("🤖 AI Chart Analysis", expanded=True):
                            st.markdown(vision_analysis_2["time_series"])
            except Exception:
                pass
        if result.get("group_chart_bytes"):
            try:
                with chart_cols[1]:
                    st.image(result["group_chart_bytes"], caption="Top Groups", use_container_width=True)
                    if vision_analysis_2.get("group_chart"):
                        with st.expander("🤖 AI Chart Analysis", expanded=True):
                            st.markdown(vision_analysis_2["group_chart"])
            except Exception:
                pass

        with st.expander("Raw insights.json"):
            st.code(json.dumps(data, indent=2))

        with st.expander("Agent run log"):
            st.code(result.get("log", ""))
            usage = result.get("usage")
            if usage is not None:
                st.caption(f"Tokens used: {usage}")


if __name__ == "__main__":
    main()


