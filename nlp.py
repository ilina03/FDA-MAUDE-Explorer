"""
nlp.py — Keyword clustering & topic modeling for MAUDE adverse event narratives

This is the feature that makes the MAUDE explorer meaningfully different from
the recall explorer. Instead of keyword counting against a fixed list (like
chart_recall_reasons() in charts.py), this module discovers failure themes
from the free text using TF-IDF + NMF topic modeling.

Why NMF over LDA or BERTopic?
  - NMF runs fast enough in-browser with sklearn (no GPU, no heavy downloads)
  - BERTopic needs sentence-transformers (~500 MB) which is impractical for
    a Streamlit Cloud deployment
  - LDA is slower and produces less interpretable topics on short medical texts
  - NMF gives coherent, human-readable topics in under 2 seconds on 500 docs

Usage:
    from nlp import compute_topics, TopicResult
    result = compute_topics(df)
    result.topics                         # list[TopicSummary]
    result.doc_topics                     # pd.Series, same index as df
    result.representative_texts(df, 0, n=3)  # top 3 examples for topic 0
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# ── Medical-domain stopwords ──────────────────────────────────────────────────
_MAUDE_STOPWORDS = {
    "patient", "device", "report", "reported", "reporting", "event",
    "adverse", "complaint", "mdr", "manufacturer", "facility", "user",
    "hospital", "received", "date", "number", "code", "information",
    "available", "unknown", "confirmed", "states", "united",
    "physician", "nurse", "healthcare", "medical", "clinical", "therapy",
    "treatment", "procedure", "use", "used", "using", "per", "also",
    "noted", "stated", "indicated", "following", "during", "after",
    "without", "product", "type", "performed", "placed", "time",
    "the", "a", "an", "is", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "for", "and", "but", "or", "at",
    "by", "from", "to", "in", "on", "of", "with", "as", "it",
    "its", "this", "that", "these", "those", "which", "who",
    "what", "when", "where", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "than", "then",
    "there", "they", "them", "their", "he", "she", "we", "our",
    "due", "also", "however", "one", "two", "three",
}


@dataclass
class TopicSummary:
    topic_id:  int
    label:     str
    keywords:  list
    doc_count: int
    pct:       float


@dataclass
class TopicResult:
    topics:          list
    doc_topics:      pd.Series
    doc_scores:      np.ndarray
    n_docs_modeled:  int
    n_docs_skipped:  int

    def representative_texts(self, df: pd.DataFrame, topic_id: int, n: int = 3) -> pd.DataFrame:
        mask = self.doc_topics == topic_id
        if not mask.any():
            return pd.DataFrame()
        topic_idx = topic_id
        scores_for_topic = self.doc_scores[mask.values, topic_idx]
        top_local = np.argsort(scores_for_topic)[::-1][:n]
        global_idx = df[mask].iloc[top_local].index
        out = df.loc[global_idx, [
            "report_number", "date_received", "event_type",
            "generic_name", "manufacturer", "narrative",
        ]].copy()
        out["topic_score"] = scores_for_topic[top_local]
        return out


@st.cache_data(ttl=3600, show_spinner=False)
def compute_topics(
    _df: pd.DataFrame,
    n_topics: int = 6,
    max_features: int = 2000,
    top_keywords: int = 8,
    min_doc_len: int = 40,
) -> "TopicResult | None":
    """
    Run TF-IDF + NMF topic modeling on df["narrative"].

    Parameters
    ----------
    _df          : DataFrame from fetch_events() — must have "narrative" column.
                   Leading underscore tells Streamlit not to hash it.
    n_topics     : Number of NMF topics. 5-8 works well for MAUDE.
    max_features : TF-IDF vocabulary cap.
    top_keywords : Terms to extract per topic for the readable label.
    min_doc_len  : Skip narratives shorter than this (empty/boilerplate).

    Returns
    -------
    TopicResult or None if there is not enough text to model.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
        from sklearn.preprocessing import normalize
    except ImportError:
        st.error("scikit-learn is required for topic modeling. Add `scikit-learn>=1.4.0` to requirements.txt.")
        return None

    if _df.empty or "narrative" not in _df.columns:
        return None

    has_text  = _df["narrative"].fillna("").str.len() >= min_doc_len
    df_valid  = _df[has_text].copy()
    n_skipped = int((~has_text).sum())

    if len(df_valid) < n_topics * 3:
        return None

    cleaned = [_clean_text(t) for t in df_valid["narrative"].tolist()]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(_MAUDE_STOPWORDS),
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{2,}\b",
        )
        tfidf_matrix = vectorizer.fit_transform(cleaned)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=400, init="nndsvda")
        W   = nmf.fit_transform(tfidf_matrix)
        H   = nmf.components_

    W_norm         = normalize(W, norm="l1", axis=1)
    doc_assignments = np.argmax(W_norm, axis=1)
    doc_topics      = pd.Series(doc_assignments, index=df_valid.index, dtype=int)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for tid in range(n_topics):
        top_idx  = np.argsort(H[tid])[::-1][:top_keywords]
        keywords = [feature_names[i] for i in top_idx]
        label    = " / ".join(keywords[:3])
        count    = int((doc_assignments == tid).sum())
        pct      = count / len(df_valid) if df_valid.shape[0] > 0 else 0.0
        topics.append(TopicSummary(topic_id=tid, label=label, keywords=keywords, doc_count=count, pct=pct))

    topics.sort(key=lambda t: t.doc_count, reverse=True)

    return TopicResult(
        topics         = topics,
        doc_topics     = doc_topics,
        doc_scores     = W_norm,
        n_docs_modeled = len(df_valid),
        n_docs_skipped = n_skipped,
    )


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\bpt\b",   "patient",           text)
    text = re.sub(r"\bdr\b",   "doctor",            text)
    text = re.sub(r"\bsw\b",   "software",          text)
    text = re.sub(r"\bhcp\b",  "healthcare provider", text)
    text = text.replace("-", " ")
    text = re.sub(r"\b\d+\b",   " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+",      " ", text).strip()
    return text
