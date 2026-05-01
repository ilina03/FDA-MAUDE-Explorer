"""
maude_app.py — FDA MAUDE Adverse Event Explorer
Companion to the FDA Device Recall Explorer (app.py).

Run with:  streamlit run maude_app.py
"""

import streamlit as st
import pandas as pd
from datetime import date

from maude_api import (
    fetch_events,
    get_date_range_default,
    EVENT_TYPE_VALUES,
    EVENT_TYPE_MAP,
)
from maude_charts import (
    chart_event_trend,
    chart_event_type_distribution,
    chart_top_devices,
    chart_reporter_breakdown,
    chart_report_source,
    chart_topic_bars,
    chart_topic_keywords,
)
from nlp import compute_topics

st.set_page_config(
    page_title="FDA MAUDE Explorer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles — same design system as recall explorer ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    background-color: #f9fafb;
    color: #111827;
    font-size: 14px;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] * { color: #111827 !important; }
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6b7280 !important;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
    background: #f9fafb;
}

.page-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.02em;
    margin: 0;
}
.page-meta {
    font-size: 0.78rem;
    color: #6b7280;
    margin: 0.2rem 0 1.25rem;
}
.page-meta a { color: #2563eb; text-decoration: none; }

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.1rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    line-height: 1.1;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9ca3af !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #e5e7eb !important;
    gap: 0;
    padding: 0;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #111827 !important;
    border-bottom: 2px solid #111827 !important;
    background: transparent !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px;
}

[data-testid="stTextInput"] input {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #111827;
}
[data-testid="stTextInput"] input:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
}

[data-testid="stButton"] > button,
[data-testid="stDownloadButton"] > button {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    color: #374151;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    border-radius: 6px;
    padding: 0.4rem 1rem;
    transition: all 0.12s;
}
[data-testid="stButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    background: #f3f4f6;
    border-color: #d1d5db;
    color: #111827;
}

.callout {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 0.65rem 1rem;
    font-size: 0.8rem;
    color: #1e40af;
    margin-bottom: 0.8rem;
    line-height: 1.55;
}
.callout code {
    font-size: 0.75rem;
    background: #dbeafe;
    padding: 1px 4px;
    border-radius: 3px;
}

/* Amber callout variant for NLP section */
.callout-amber {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 0.65rem 1rem;
    font-size: 0.8rem;
    color: #92400e;
    margin-bottom: 0.8rem;
    line-height: 1.55;
}

.section-label {
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6b7280;
    margin: 0.6rem 0 0.35rem;
}

.topic-pill {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #1e40af;
    margin: 2px;
}

hr { border: none; border-top: 1px solid #e5e7eb; margin: 0.75rem 0; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**Filters**")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Event type filter — replaces classification filter from recall explorer
    st.markdown('<p class="section-label">Event Type</p>', unsafe_allow_html=True)
    selected_event_types = []
    et_cols = st.columns(2)
    et_options = [("Injury", "Injury"), ("Malfunction", "Malfn."),
                  ("Death", "Death"), ("Other", "Other")]
    for i, (full, short) in enumerate(et_options):
        with et_cols[i % 2]:
            if st.checkbox(short, value=True, key=f"et_{full}"):
                selected_event_types.append(full)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Date Range</p>', unsafe_allow_html=True)
    default_start_str, default_end_str = get_date_range_default()
    date_start = st.date_input(
        "From", value=date.fromisoformat(default_start_str),
        min_value=date(1991, 1, 1), max_value=date.today(),
        label_visibility="collapsed",
    )
    date_end = st.date_input(
        "To", value=date.fromisoformat(default_end_str),
        min_value=date(1991, 1, 1), max_value=date.today(),
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Device Name</p>', unsafe_allow_html=True)
    device_filter = st.text_input(
        "device", placeholder="e.g. pacemaker, catheter, infusion pump",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Manufacturer</p>', unsafe_allow_html=True)
    manufacturer_filter = st.text_input(
        "mfr", placeholder="e.g. Medtronic, Abbott, Boston Scientific",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Max Records</p>', unsafe_allow_html=True)
    max_records = st.selectbox(
        "max", options=[100, 250, 500, 1000, 2000], index=2,
        label_visibility="collapsed",
        help="More records → richer topic modeling but slower load. Cached 1 hr.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Trend Granularity</p>', unsafe_allow_html=True)
    freq = "M" if st.radio(
        "freq", ["Monthly", "Quarterly"], horizontal=True,
        label_visibility="collapsed",
    ) == "Monthly" else "Q"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.7rem;color:#9ca3af;line-height:1.6;">'
        'Source: <a href="https://open.fda.gov/apis/device/event/" '
        'style="color:#2563eb;text-decoration:none;">openFDA /device/event</a><br>'
        'Medical devices only · Not for clinical use<br>'
        'Companion to the '
        '<a href="https://fda-device-recall.streamlit.app" '
        'style="color:#2563eb;text-decoration:none;">Device Recall Explorer</a></p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════════
if not selected_event_types:
    st.warning("Select at least one event type in the sidebar.")
    st.stop()

with st.spinner("Loading from openFDA…"):
    try:
        df = fetch_events(
            event_types=tuple(selected_event_types),
            start_date=date_start.strftime("%Y-%m-%d"),
            end_date=date_end.strftime("%Y-%m-%d"),
            device_name=device_filter.strip() or None,
            manufacturer=manufacturer_filter.strip() or None,
            max_records=max_records,
        )
    except RuntimeError as e:
        st.error(f"API Error: {e}")
        st.stop()


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="page-title">FDA MAUDE Adverse Event Explorer</p>'
    '<p class="page-meta">'
    'Data from <a href="https://open.fda.gov/apis/device/event/">openFDA /device/event</a> · '
    'Patient injury &amp; malfunction reports · For research use only</p>',
    unsafe_allow_html=True,
)

if df.empty:
    st.info("No records match the current filters. Try widening the date range or changing the device/manufacturer filter.")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════
m1, m2, m3, m4, m5, m6 = st.columns(6)

total      = len(df)
n_injury   = (df["event_type"] == "Injury").sum()
n_death    = (df["event_type"] == "Death").sum()
n_malfn    = (df["event_type"] == "Malfunction").sum()
n_devices  = df["generic_name"].replace("", pd.NA).dropna().nunique()
n_with_txt = df["has_narrative"].sum()

m1.metric("Total Reports",  f"{total:,}")
m2.metric("Injuries",       f"{n_injury:,}", help="Event type = Injury")
m3.metric("Deaths",         f"{n_death:,}",  help="Event type = Death")
m4.metric("Malfunctions",   f"{n_malfn:,}",  help="Event type = Malfunction")
m5.metric("Device Types",   f"{n_devices:,}", help="Unique generic device names")
m6.metric("Have Narrative", f"{n_with_txt:,}", help="Records with usable text for NLP")

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_trend, tab_devices, tab_topics, tab_reporters, tab_data = st.tabs([
    "Trend",
    "Devices & Manufacturers",
    "Failure Narratives",       # ← the new NLP tab
    "Reporters & Sources",
    "Raw Data",
])


# ── Trend ─────────────────────────────────────────────────────────
with tab_trend:
    col_left, col_right = st.columns([3, 1], gap="large")
    with col_left:
        st.plotly_chart(
            chart_event_trend(df, freq=freq),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col_right:
        st.plotly_chart(
            chart_event_type_distribution(df),
            use_container_width=True, config={"displayModeBar": False},
        )


# ── Devices & Manufacturers ───────────────────────────────────────
with tab_devices:
    view = st.radio(
        "View by", ["Device (Generic Name)", "Manufacturer"],
        horizontal=True, label_visibility="collapsed",
    )
    by_col = "generic_name" if "Device" in view else "manufacturer"
    top_n  = st.slider("Top N", 5, 30, 15, step=5)

    st.plotly_chart(
        chart_top_devices(df, by=by_col, top_n=top_n),
        use_container_width=True, config={"displayModeBar": False},
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Device Detail Table</p>', unsafe_allow_html=True)

    device_table = (
        df.groupby("generic_name")
        .agg(
            Reports     =("report_number",  "count"),
            Injuries    =("event_type",      lambda x: (x == "Injury").sum()),
            Deaths      =("event_type",      lambda x: (x == "Death").sum()),
            Malfunctions=("event_type",      lambda x: (x == "Malfunction").sum()),
            Manufacturers=("manufacturer",   lambda x: ", ".join(sorted(x.dropna().replace("", pd.NA).dropna().unique())[:3])),
            Earliest    =("date_received",   lambda x: x.dropna().min().strftime("%Y-%m-%d") if len(x.dropna()) > 0 else ""),
            Latest      =("date_received",   lambda x: x.dropna().max().strftime("%Y-%m-%d") if len(x.dropna()) > 0 else ""),
        )
        .sort_values("Reports", ascending=False)
        .reset_index()
        .rename(columns={"generic_name": "Device"})
    )
    st.dataframe(device_table, use_container_width=True, hide_index=True)


# ── Failure Narratives (NLP) ──────────────────────────────────────
with tab_topics:
    st.markdown(
        '<div class="callout-amber">'
        '⚗️ <b>Topic modeling</b> uses TF-IDF + NMF to cluster the '
        '<code>mdr_text</code> narrative field and surface common failure themes — '
        'e.g. "software alarm", "catheter fracture", "battery failure". '
        'Topics are discovered automatically; no fixed keyword list. '
        'Requires records with usable narrative text.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Controls
    ctrl_col1, ctrl_col2 = st.columns([1, 3])
    with ctrl_col1:
        n_topics = st.slider("Number of topics", min_value=3, max_value=10, value=6)

    with st.spinner("Running topic model…"):
        topic_result = compute_topics(_df=df, n_topics=n_topics)

    if topic_result is None:
        st.info(
            f"Not enough narrative text to model ({n_with_txt} records have text). "
            "Try increasing Max Records or widening the date range."
        )
    else:
        # Summary line
        st.markdown(
            f'<p class="section-label">'
            f'{topic_result.n_docs_modeled:,} records modeled · '
            f'{topic_result.n_docs_skipped:,} skipped (too short)'
            f'</p>',
            unsafe_allow_html=True,
        )

        # Two-column layout: bar chart left, keyword heatmap right
        nlp_col1, nlp_col2 = st.columns([1, 1], gap="large")
        with nlp_col1:
            st.plotly_chart(
                chart_topic_bars(topic_result),
                use_container_width=True, config={"displayModeBar": False},
            )
        with nlp_col2:
            st.plotly_chart(
                chart_topic_keywords(topic_result),
                use_container_width=True, config={"displayModeBar": False},
            )

        # ── Interactive topic drill-down ──────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Drill into a Topic</p>', unsafe_allow_html=True)

        topic_options = {
            f"Topic {i+1}: {t.label} ({t.doc_count:,} reports)": i
            for i, t in enumerate(topic_result.topics)
        }
        selected_label = st.selectbox(
            "Select topic", list(topic_options.keys()),
            label_visibility="collapsed",
        )
        selected_tid = topic_options[selected_label]
        selected_topic = topic_result.topics[selected_tid]

        # Keyword pills
        pills_html = " ".join(
            f'<span class="topic-pill">{kw}</span>'
            for kw in selected_topic.keywords
        )
        st.markdown(
            f'<div style="margin-bottom:0.8rem;">{pills_html}</div>',
            unsafe_allow_html=True,
        )

        # Representative narratives for the selected topic
        rep = topic_result.representative_texts(df, topic_id=selected_tid, n=5)
        if not rep.empty:
            rep["date_received"] = pd.to_datetime(rep["date_received"]).dt.strftime("%Y-%m-%d")
            rep["topic_score"]   = rep["topic_score"].map("{:.2f}".format)
            rep.columns = ["Report #", "Date", "Event Type", "Device", "Manufacturer", "Narrative", "Score"]
            st.dataframe(rep, use_container_width=True, hide_index=True)
        else:
            st.info("No representative records found for this topic.")


# ── Reporters & Sources ───────────────────────────────────────────
with tab_reporters:
    rep_col1, rep_col2 = st.columns([1, 1], gap="large")
    with rep_col1:
        st.plotly_chart(
            chart_reporter_breakdown(df, top_n=12),
            use_container_width=True, config={"displayModeBar": False},
        )
    with rep_col2:
        st.plotly_chart(
            chart_report_source(df),
            use_container_width=True, config={"displayModeBar": False},
        )


# ── Raw Data ──────────────────────────────────────────────────────
with tab_data:
    row = st.columns([3, 1])
    with row[0]:
        search = st.text_input(
            "Search", placeholder="Filter by any text across all columns…",
            label_visibility="collapsed",
        )

    display_df = df.copy()
    display_df["date_received"] = display_df["date_received"].dt.strftime("%Y-%m-%d")
    display_df["date_of_event"] = display_df["date_of_event"].dt.strftime("%Y-%m-%d").replace("NaT", "")

    if search:
        mask = display_df.apply(
            lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1,
        )
        display_df = display_df[mask]

    cols = [c for c in [
        "report_number", "date_received", "date_of_event", "event_type",
        "generic_name", "brand_name", "manufacturer", "model_number",
        "reporter_code", "report_source", "narrative",
    ] if c in display_df.columns]

    st.markdown(f'<p class="section-label">{len(display_df):,} records</p>',
                unsafe_allow_html=True)
    st.dataframe(display_df[cols], use_container_width=True, hide_index=True)

    with row[1]:
        st.download_button(
            label="Export CSV",
            data=display_df[cols].to_csv(index=False).encode(),
            file_name=f"fda_maude_{date.today().isoformat()}.csv",
            mime="text/csv",
        )
