
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# Ichimoku Cloud - 3 Method Dashboard
# Excel input needs ONLY:
# Date, Open, High, Low, Close
#
# The Python program calculates:
# Ichimoku values -> Cloud Top/Bottom -> Signal
# -> Lots -> Avg Entry -> Target % -> Exit -> PnL
#
# You can upload ONE OHLC Excel file and switch between
# Method 1 / Method 2 / Method 3 using the dropdown.
# ============================================================

st.set_page_config(
    page_title="Ichimoku Cloud - 3 Method Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------- CSS --------------------
st.markdown("""
<style>
.dashboard-header {
    background: linear-gradient(135deg, #111827, #1f2937);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,.16);
}
.dashboard-title {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin: 0;
}
.dashboard-subtitle {
    color: #d1d5db;
    font-size: 14px;
    margin-top: 6px;
}
.panel {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 18px;
}
.kpi-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
    min-height: 105px;
}
.kpi-label {
    color: #6b7280;
    font-size: 13px;
}
.kpi-value {
    color: #111827;
    font-size: 25px;
    font-weight: 800;
    margin-top: 6px;
}
.rule-box {
    background: #eef2ff;
    border-left: 5px solid #4f46e5;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 10px 0 20px;
}
.status-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 12px 16px;
}
.section-title {
    font-size: 20px;
    font-weight: 800;
    margin: 18px 0 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">📊 Ichimoku Cloud — 3 Method Dashboard</div>
    <div class="dashboard-subtitle">
        Upload one OHLC Excel file and switch between Method 1, Method 2 and Method 3.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Helpers
# ============================================================

def normalize_name(x):
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def find_col(df, names):
    normalized = {normalize_name(c): c for c in df.columns}

    for name in names:
        key = normalize_name(name)
        if key in normalized:
            return normalized[key]

    # Partial match
    for c in df.columns:
        nc = normalize_name(c)
        for name in names:
            if normalize_name(name) in nc:
                return c

    return None


def load_ohlc(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)

    # Prefer Data sheet, otherwise first sheet
    sheet_name = None
    for s in xls.sheet_names:
        if normalize_name(s) == "data":
            sheet_name = s
            break

    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    date_col = find_col(df, ["Date", "Datetime", "Timestamp"])
    open_col = find_col(df, ["Open"])
    high_col = find_col(df, ["High"])
    low_col = find_col(df, ["Low"])
    close_col = find_col(df, ["Close"])

    missing = []
    for label, col in [
        ("Date", date_col),
        ("Open", open_col),
        ("High", high_col),
        ("Low", low_col),
        ("Close", close_col),
    ]:
        if col is None:
            missing.append(label)

    if missing:
        raise ValueError(
            "OHLC Excel file-la indha columns missing: "
            + ", ".join(missing)
        )

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce"),
        "Open": pd.to_numeric(df[open_col], errors="coerce"),
        "High": pd.to_numeric(df[high_col], errors="coerce"),
        "Low": pd.to_numeric(df[low_col], errors="coerce"),
        "Close": pd.to_numeric(df[close_col], errors="coerce"),
    })

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    return out


def calculate_ichimoku(df):
    d = df.copy()

    # Standard Ichimoku periods
    # Tenkan = 9
    # Kijun = 26
    # Senkou Span B = 52
    tenkan_period = 9
    kijun_period = 26
    span_b_period = 52

    d["Tenkan"] = (
        d["High"].rolling(tenkan_period).max()
        + d["Low"].rolling(tenkan_period).min()
    ) / 2

    d["Kijun"] = (
        d["High"].rolling(kijun_period).max()
        + d["Low"].rolling(kijun_period).min()
    ) / 2

    d["Span A"] = (d["Tenkan"] + d["Kijun"]) / 2

    d["Span B"] = (
        d["High"].rolling(span_b_period).max()
        + d["Low"].rolling(span_b_period).min()
    ) / 2

    d["Cloud Top"] = d[["Span A", "Span B"]].max(axis=1)
    d["Cloud Bottom"] = d[["Span A", "Span B"]].min(axis=1)

    return d


def calculate_signals(d):
    d = d.copy()

    def signal(row):
        close = row["Close"]
        top = row["Cloud Top"]
        bottom = row["Cloud Bottom"]

        if pd.isna(close) or pd.isna(top) or pd.isna(bottom):
            return ""

        if close > top and close > bottom:
            return "BUY"

        if close < top and close < bottom:
            return "SELL"

        return "INSIDE"

    d["Signal"] = d.apply(signal, axis=1)
    return d


# ============================================================
# Method calculations
# ============================================================

def calculate_method1(d, target_pct, max_lots, initial_lots):
    """
    METHOD 1
    ----------------
    BUY -> accumulate lots until Max Lots.
    EXIT only when:
      1. Lots >= Max Lots
      2. Target % >= user Target %
    Both must happen on the same row.
    SELL / INSIDE does not by itself exit.
    After EXIT, next BUY starts Initial Lots.
    """

    d = d.copy()

    lots = []
    avg_entry = []
    target_values = []
    exits = []
    pnl = []

    current_lots = 0
    current_avg = np.nan
    in_position = False

    for _, row in d.iterrows():
        sig = row["Signal"]
        close = row["Close"]

        row_lots = current_lots
        row_avg = current_avg
        row_target = np.nan
        row_exit = ""
        row_pnl = np.nan

        if sig == "BUY":

            if not in_position:
                current_lots = initial_lots
                current_avg = close
                in_position = True

            else:
                if current_lots < max_lots:
                    old_lots = current_lots
                    add_lots = min(initial_lots, max_lots - current_lots)

                    current_lots += add_lots

                    if current_lots > 0:
                        current_avg = (
                            current_avg * old_lots
                            + close * add_lots
                        ) / current_lots

            row_lots = current_lots
            row_avg = current_avg

        elif in_position:
            row_lots = current_lots
            row_avg = current_avg

        if in_position and current_lots > 0 and pd.notna(current_avg):
            row_target = (close - current_avg) / current_avg

        # Method 1 exit condition
        if (
            in_position
            and current_lots >= max_lots
            and pd.notna(row_target)
            and row_target >= target_pct
        ):
            row_exit = "EXIT"
            row_pnl = (close - current_avg) * current_lots

            # Reset AFTER recording this row
            current_lots = 0
            current_avg = np.nan
            in_position = False

        lots.append(row_lots)
        avg_entry.append(row_avg)
        target_values.append(row_target)
        exits.append(row_exit)
        pnl.append(row_pnl)

    d["Lots"] = lots
    d["Avg Entry"] = avg_entry
    d["Target %"] = target_values
    d["Exit"] = exits
    d["PnL"] = pnl

    return d


def calculate_method2(d, target_pct, max_lots, initial_lots):
    """
    METHOD 2
    ----------------
    BUY -> start at Initial Lots and accumulate up to Max Lots.
    EXIT as soon as positive Target % is reached.
    Max Lots does NOT need to be reached.
    SELL / INSIDE does not by itself exit.
    After EXIT, next BUY starts Initial Lots.
    """

    d = d.copy()

    lots = []
    avg_entry = []
    target_values = []
    exits = []
    pnl = []

    current_lots = 0
    current_avg = np.nan
    in_position = False

    for _, row in d.iterrows():
        sig = row["Signal"]
        close = row["Close"]

        row_lots = current_lots
        row_avg = current_avg
        row_target = np.nan
        row_exit = ""
        row_pnl = np.nan

        if sig == "BUY":

            if not in_position:
                current_lots = initial_lots
                current_avg = close
                in_position = True

            else:
                if current_lots < max_lots:
                    old_lots = current_lots
                    add_lots = min(initial_lots, max_lots - current_lots)

                    current_lots += add_lots

                    current_avg = (
                        current_avg * old_lots
                        + close * add_lots
                    ) / current_lots

            row_lots = current_lots
            row_avg = current_avg

        elif in_position:
            row_lots = current_lots
            row_avg = current_avg

        if in_position and current_lots > 0 and pd.notna(current_avg):
            row_target = (close - current_avg) / current_avg

        # Method 2 exit: target alone
        if (
            in_position
            and pd.notna(row_target)
            and row_target >= target_pct
        ):
            row_exit = "EXIT"
            row_pnl = (close - current_avg) * current_lots

            current_lots = 0
            current_avg = np.nan
            in_position = False

        lots.append(row_lots)
        avg_entry.append(row_avg)
        target_values.append(row_target)
        exits.append(row_exit)
        pnl.append(row_pnl)

    d["Lots"] = lots
    d["Avg Entry"] = avg_entry
    d["Target %"] = target_values
    d["Exit"] = exits
    d["PnL"] = pnl

    return d


def calculate_method3(d, target_pct, max_lots, initial_lots):
    """
    METHOD 3
    ----------------
    BUY -> accumulate lots up to Max Lots.
    EXIT only when ALL 3 are true on the same row:
      1. Lots >= Max Lots
      2. Target % >= target
      3. Close is below both Cloud Top and Cloud Bottom
    After EXIT, next BUY starts Initial Lots.
    """

    d = d.copy()

    lots = []
    avg_entry = []
    target_values = []
    exits = []
    pnl = []

    current_lots = 0
    current_avg = np.nan
    in_position = False

    for _, row in d.iterrows():
        sig = row["Signal"]
        close = row["Close"]
        cloud_top = row["Cloud Top"]
        cloud_bottom = row["Cloud Bottom"]

        row_lots = current_lots
        row_avg = current_avg
        row_target = np.nan
        row_exit = ""
        row_pnl = np.nan

        if sig == "BUY":

            if not in_position:
                current_lots = initial_lots
                current_avg = close
                in_position = True

            else:
                if current_lots < max_lots:
                    old_lots = current_lots
                    add_lots = min(initial_lots, max_lots - current_lots)

                    current_lots += add_lots

                    current_avg = (
                        current_avg * old_lots
                        + close * add_lots
                    ) / current_lots

            row_lots = current_lots
            row_avg = current_avg

        elif in_position:
            row_lots = current_lots
            row_avg = current_avg

        if in_position and current_lots > 0 and pd.notna(current_avg):
            row_target = (close - current_avg) / current_avg

        below_cloud = (
            pd.notna(cloud_top)
            and pd.notna(cloud_bottom)
            and close < cloud_top
            and close < cloud_bottom
        )

        # Method 3: ALL 3 conditions
        if (
            in_position
            and current_lots >= max_lots
            and pd.notna(row_target)
            and row_target >= target_pct
            and below_cloud
        ):
            row_exit = "EXIT"
            row_pnl = (close - current_avg) * current_lots

            current_lots = 0
            current_avg = np.nan
            in_position = False

        lots.append(row_lots)
        avg_entry.append(row_avg)
        target_values.append(row_target)
        exits.append(row_exit)
        pnl.append(row_pnl)

    d["Lots"] = lots
    d["Avg Entry"] = avg_entry
    d["Target %"] = target_values
    d["Exit"] = exits
    d["PnL"] = pnl

    return d


def run_method(d, method, target_pct, max_lots, initial_lots):
    if method == "Method 1":
        return calculate_method1(d, target_pct, max_lots, initial_lots)

    if method == "Method 2":
        return calculate_method2(d, target_pct, max_lots, initial_lots)

    return calculate_method3(d, target_pct, max_lots, initial_lots)


def method_rule(method):
    if method == "Method 1":
        return (
            "<b>Method 1:</b> EXIT only when "
            "<b>Max Lots + Target %</b> are both reached on the same row."
        )

    if method == "Method 2":
        return (
            "<b>Method 2:</b> EXIT when the positive "
            "<b>Target %</b> is reached. Max Lots does not need to be reached."
        )

    return (
        "<b>Method 3:</b> EXIT only when all 3 are true on the same row: "
        "<b>Max Lots + Target % + Close below Cloud</b>."
    )


# ============================================================
# Controls
# ============================================================

if "source_df" not in st.session_state:
    st.session_state.source_df = None

col1, col2, col3, col4, col5 = st.columns([2.2, 1.2, 1.0, 1.0, 1.1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload OHLC Excel",
        type=["xlsx", "xls"]
    )

with col2:
    selected_method = st.selectbox(
        "Select Method",
        ["Method 1", "Method 2", "Method 3"],
        index=0
    )

with col3:
    target_percent_input = st.number_input(
        "Target %",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1
    )

with col4:
    max_lots = st.number_input(
        "Max Lots",
        min_value=1,
        value=10,
        step=1
    )

with col5:
    initial_lots = st.number_input(
        "Initial Lots",
        min_value=1,
        value=1,
        step=1
    )

target_pct = target_percent_input / 100.0


# ============================================================
# Read Excel ONCE
# ============================================================

if uploaded_file is not None:
    try:
        st.session_state.source_df = load_ohlc(uploaded_file)
    except Exception as e:
        st.error(f"Excel read error: {e}")
        st.stop()

source_df = st.session_state.source_df

if source_df is None:
    st.info(
        "First OHLC Excel file upload. "
        "Excel Date, Open, High, Low, Close columns."
    )
    st.stop()


# ============================================================
# Calculate
# ============================================================

base = calculate_ichimoku(source_df)
base = calculate_signals(base)

result = run_method(
    base,
    selected_method,
    target_pct,
    int(max_lots),
    int(initial_lots)
)

# Percentage display column
result["%"] = result["Target %"]

exit_rows = result[result["Exit"] == "EXIT"].copy()

total_pnl = exit_rows["PnL"].sum() if not exit_rows.empty else 0.0
total_exits = len(exit_rows)
max_lot_used = int(result["Lots"].max()) if not result.empty else 0
total_rows = len(result)

win_count = int((exit_rows["PnL"] > 0).sum()) if not exit_rows.empty else 0
loss_count = int((exit_rows["PnL"] < 0).sum()) if not exit_rows.empty else 0

# Cumulative PnL
result["Cumulative PnL"] = result["PnL"].fillna(0).cumsum()


# ============================================================
# Status / rule
# ============================================================

st.markdown(
    f'<div class="status-box">✅ <b>{selected_method}</b> selected — '
    f'{total_rows:,} OHLC rows calculated.</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="rule-box">{method_rule(selected_method)}</div>',
    unsafe_allow_html=True
)


# ============================================================
# KPI cards
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Selected Method</div>'
        f'<div class="kpi-value">{selected_method}</div></div>',
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Total EXITs</div>'
        f'<div class="kpi-value">{total_exits}</div></div>',
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Total PnL</div>'
        f'<div class="kpi-value">{total_pnl:,.2f}</div></div>',
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Max Lots Used</div>'
        f'<div class="kpi-value">{max_lot_used}</div></div>',
        unsafe_allow_html=True
    )

with k5:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Winning EXITs</div>'
        f'<div class="kpi-value">{win_count}</div></div>',
        unsafe_allow_html=True
    )

with k6:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Losing EXITs</div>'
        f'<div class="kpi-value">{loss_count}</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# Charts
# ============================================================

st.markdown('<div class="section-title">📈 Performance</div>', unsafe_allow_html=True)

chart1, chart2 = st.columns(2)

with chart1:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result["Date"],
            y=result["Cumulative PnL"],
            mode="lines",
            name="Cumulative PnL"
        )
    )

    fig.update_layout(
        title=f"{selected_method} — Cumulative PnL",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

with chart2:
    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=result["Date"],
            y=result["Lots"],
            mode="lines",
            name="Lots"
        )
    )

    fig2.update_layout(
        title=f"{selected_method} — Lots Progression",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified"
    )

    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# EXIT table
# ============================================================

st.markdown('<div class="section-title">🚪 EXIT Trades</div>', unsafe_allow_html=True)

if exit_rows.empty:
    st.info("Current settings-la EXIT கிடைக்கவில்லை.")
else:
    exit_display = exit_rows[
        [
            "Date",
            "Close",
            "Signal",
            "Lots",
            "Avg Entry",
            "Target %",
            "Exit",
            "PnL",
        ]
    ].copy()

    exit_display["Target %"] = exit_display["Target %"] * 100

    st.dataframe(
        exit_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# Full calculated data
# ============================================================

st.markdown(
    '<div class="section-title">📋 Calculated Data</div>',
    unsafe_allow_html=True
)

display = result.copy()

display["Target %"] = display["Target %"] * 100
display["%"] = display["%"] * 100

# Round numeric columns
numeric_cols = display.select_dtypes(include=np.number).columns
display[numeric_cols] = display[numeric_cols].round(6)

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# Download
# ============================================================

csv_data = result.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Calculated CSV",
    data=csv_data,
    file_name=f"{selected_method.lower().replace(' ', '_')}_result.csv",
    mime="text/csv"
)

st.caption(
    "Input Excel-la OHLC data mattum pothum. "
    "Ichimoku, Signal, Lots, Avg Entry, Target %, Exit and PnL Python-la calculate aagum."
)
