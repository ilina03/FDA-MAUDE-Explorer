"""
maude_charts.py — Plotly chart builders for the MAUDE Adverse Event Explorer.

Companion to charts.py from the recall explorer — same palette, same _apply()
helper, same clinical white SaaS aesthetic. New charts specific to MAUDE:
  - chart_event_type_distribution()   (replaces chart_class_distribution)
  - chart_top_devices()               (replaces chart_top_manufacturers — device-first)
  - chart_reporter_breakdown()        (no recall equivalent)
  - chart_topic_bars()                (NLP-only, no recall equivalent)
  - chart_topic_scatter()             (NLP-only, no recall equivalent)
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ── Palette — identical to charts.py so both tools share a visual language ───
C = {
    "bg":      "#ffffff",
    "page":    "#f9fafb",
    "border":  "#e5e7eb",
    "ink":     "#111827",
    "muted":   "#6b7280",
    "subtle":  "#9ca3af",
    "surface": "#f3f4f6",
}

# Event type colours — intentionally different from recall classification colours
# so the two tools are visually distinct even if side-by-side.
EVENT_COLORS = {
    "Injury":     "#dc2626",   # red   — same severity intuition as Class I
    "Death":      "#7f1d1d",   # dark red
    "Malfunction": "#2563eb",  # blue  — device-level, not patient-level
    "Other":      "#9ca3af",   # gray
    "No Answer":  "#d1d5db",   # light gray
}

# Topic palette — 8 distinct colours for up to 8 NMF topics
TOPIC_COLORS = [
    "#2563eb", "#059669", "#d97706", "#7c3aed",
    "#db2777", "#0891b2", "#65a30d", "#9ca3af",
]

_FONT = "'Inter', -apple-system, sans-serif"

_BASE = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor =C["bg"],
    font=dict(family=_FONT, color=C["muted"], size=11),
    margin=dict(l=12, r=12, t=40, b=12),
    legend=dict(
        bgcolor=C["bg"],
        bordercolor=C["border"],
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor=C["border"], linecolor=C["border"],
        tickfont=dict(size=10, color=C["muted"]),
        title_font=dict(size=10, color=C["muted"]),
        showgrid=True, zeroline=False,
    ),
    yaxis=dict(
        gridcolor=C["border"], linecolor=C["border"],
        tickfont=dict(size=10, color=C["muted"]),
        title_font=dict(size=10, color=C["muted"]),
        showgrid=True, zeroline=False,
    ),
)


def _apply(fig: go.Figure, title="", height=340) -> go.Figure:
    fig.update_layout(
        **_BASE,
        height=height,
        title=dict(
            text=title,
            font=dict(size=11, color=C["muted"], family=_FONT),
            x=0, xanchor="left",
        ),
    )
    return fig


def _empty(message: str, height: int = 240) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=C["muted"], size=12, family=_FONT),
    )
    _apply(fig, height=height)
    return fig


# ── Chart 1: Trend over time ──────────────────────────────────────────────────
# Mirrors chart_trend_over_time() from charts.py — same structure, different
# grouping column (event_type instead of classification).

def chart_event_trend(df: pd.DataFrame, freq: str = "M") -> go.Figure:
    if df.empty or "date_received" not in df.columns:
        return _empty("No data available.")

    df2 = df.dropna(subset=["date_received"]).copy()
    if df2.empty:
        return _empty("No dated records to plot.")

    df2["period"] = df2["date_received"].dt.to_period(freq).dt.to_timestamp()
    grouped = (
        df2.groupby(["period", "event_type"])
        .size()
        .reset_index(name="count")
        .sort_values("period")
    )

    fig = go.Figure()
    for et in ["Injury", "Death", "Malfunction", "Other", "No Answer"]:
        if et not in grouped["event_type"].unique():
            continue
        sub   = grouped[grouped["event_type"] == et]
        color = EVENT_COLORS.get(et, C["muted"])
        fig.add_trace(go.Scatter(
            x=sub["period"], y=sub["count"],
            name=et, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            hovertemplate=f"<b>{et}</b><br>%{{x|%b %Y}}: %{{y}} reports<extra></extra>",
        ))

    label = "Monthly" if freq == "M" else "Quarterly"
    _apply(fig, title=f"Adverse Event Reports — {label}", height=320)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Reports")
    return fig


# ── Chart 2: Event type donut ─────────────────────────────────────────────────
# Replaces chart_class_distribution() — same donut shape, different data.

def chart_event_type_distribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty("No data.")

    counts = df["event_type"].value_counts().reset_index()
    counts.columns = ["et", "count"]

    fig = go.Figure(go.Pie(
        labels=counts["et"],
        values=counts["count"],
        hole=0.6,
        marker=dict(
            colors=[EVENT_COLORS.get(e, C["muted"]) for e in counts["et"]],
            line=dict(color=C["bg"], width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=10, family=_FONT),
        hovertemplate="%{label}<br>%{value:,} reports (%{percent})<extra></extra>",
    ))

    _apply(fig, title="By Event Type", height=290)
    fig.update_layout(showlegend=False, margin=dict(l=8, r=8, t=40, b=0))
    return fig


# ── Chart 3: Top devices ──────────────────────────────────────────────────────
# Replaces chart_top_manufacturers() — MAUDE is device-first so generic_name
# is the primary breakdown axis, with a manufacturer option.

def chart_top_devices(
    df: pd.DataFrame,
    by: str = "generic_name",
    top_n: int = 15,
) -> go.Figure:
    if df.empty:
        return _empty("No data available.")

    col_label = "Device" if by == "generic_name" else "Manufacturer"
    counts = (
        df[by].replace("", pd.NA).dropna()
        .value_counts().head(top_n).reset_index()
    )
    counts.columns = ["name", "count"]
    counts = counts.sort_values("count")

    fig = go.Figure(go.Bar(
        x=counts["count"], y=counts["name"],
        orientation="h",
        marker=dict(color=C["ink"], opacity=0.8, line_width=0),
        text=counts["count"],
        textposition="outside",
        textfont=dict(size=10, color=C["muted"]),
        hovertemplate=f"<b>%{{y}}</b><br>%{{x}} reports<extra></extra>",
    ))

    _apply(fig, title=f"Top {top_n} by {col_label}", height=max(320, top_n * 28))
    fig.update_xaxes(title_text="Report Count")
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


# ── Chart 4: Reporter occupation breakdown ────────────────────────────────────
# Unique to MAUDE — no recall equivalent. Shows who is filing the reports.

def chart_reporter_breakdown(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    if df.empty or "reporter_code" not in df.columns:
        return _empty("No reporter data available.")

    counts = (
        df["reporter_code"].replace("", pd.NA).dropna()
        .value_counts().head(top_n).reset_index()
    )
    counts.columns = ["reporter", "count"]
    counts = counts.sort_values("count")

    fig = go.Figure(go.Bar(
        x=counts["count"], y=counts["reporter"],
        orientation="h",
        marker=dict(color=C["ink"], opacity=0.75, line_width=0),
        text=counts["count"],
        textposition="outside",
        textfont=dict(size=10, color=C["muted"]),
        hovertemplate="<b>%{y}</b><br>%{x} reports<extra></extra>",
    ))

    _apply(fig, title="Reports by Reporter Occupation", height=max(260, top_n * 30))
    fig.update_xaxes(title_text="Report Count")
    return fig


# ── Chart 5: Report source breakdown ─────────────────────────────────────────

def chart_report_source(df: pd.DataFrame) -> go.Figure:
    if df.empty or "report_source" not in df.columns:
        return _empty("No source data.")

    counts = df["report_source"].value_counts().reset_index()
    counts.columns = ["src", "count"]

    palette = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#9ca3af"]

    fig = go.Figure(go.Pie(
        labels=counts["src"],
        values=counts["count"],
        hole=0.6,
        marker=dict(
            colors=palette[:len(counts)],
            line=dict(color=C["bg"], width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=10, family=_FONT),
        hovertemplate="%{label}<br>%{value:,} reports (%{percent})<extra></extra>",
    ))

    _apply(fig, title="By Report Source", height=290)
    fig.update_layout(showlegend=False, margin=dict(l=8, r=8, t=40, b=0))
    return fig


# ── Chart 6: NLP — topic bar chart ───────────────────────────────────────────
# Takes a TopicResult from nlp.compute_topics() and shows the topic distribution.

def chart_topic_bars(topic_result) -> go.Figure:
    """
    Horizontal bar chart of topic prevalence.
    topic_result : TopicResult from nlp.compute_topics()
    """
    if topic_result is None or not topic_result.topics:
        return _empty("Run topic modeling to see failure clusters.")

    topics = topic_result.topics
    labels = [f"Topic {i+1}: {t.label}" for i, t in enumerate(topics)]
    counts = [t.doc_count for t in topics]
    colors = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in range(len(topics))]

    fig = go.Figure(go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line_width=0),
        text=[f"{t.pct:.0%}" for t in topics],
        textposition="outside",
        textfont=dict(size=10, color=C["muted"]),
        hovertemplate="<b>%{y}</b><br>%{x} reports<extra></extra>",
    ))

    _apply(fig, title="Failure Narrative Clusters (NMF Topics)", height=max(280, len(topics) * 48))
    fig.update_xaxes(title_text="Reports Assigned")
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


# ── Chart 7: NLP — keyword heatmap per topic ─────────────────────────────────

def chart_topic_keywords(topic_result) -> go.Figure:
    """
    Heatmap of top keywords across all topics.
    Rows = topics, columns = keywords, cell = NMF H-matrix weight (normalized).
    """
    if topic_result is None or not topic_result.topics:
        return _empty("Run topic modeling to see keyword heatmap.")

    topics   = topic_result.topics
    # Gather the union of all top keywords across topics (up to 10 per topic)
    all_kw   = []
    seen     = set()
    for t in topics:
        for kw in t.keywords[:10]:
            if kw not in seen:
                all_kw.append(kw)
                seen.add(kw)
    all_kw = all_kw[:20]  # cap at 20 columns for readability

    topic_labels = [f"T{i+1}: {t.label}" for i, t in enumerate(topics)]

    # Build matrix from topic keyword membership (1 if keyword is in topic's top list, else 0)
    # This is a display heuristic — for exact weights we'd need the H matrix from NMF
    matrix = []
    for t in topics:
        kw_set = set(t.keywords)
        row = [1.0 if kw in kw_set else 0.0 for kw in all_kw]
        matrix.append(row)

    z = np.array(matrix)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=all_kw,
        y=topic_labels,
        colorscale=[
            [0,   C["surface"]],
            [0.5, "#bfdbfe"],
            [1,   "#2563eb"],
        ],
        showscale=False,
        hovertemplate="<b>%{x}</b> in <b>%{y}</b><extra></extra>",
    ))

    _apply(fig, title="Top Keywords by Topic", height=max(220, len(topics) * 44))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_layout(margin=dict(l=160, r=12, t=40, b=80))
    return fig
