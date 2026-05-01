"""
maude_api.py — openFDA Device Event (MAUDE) API client

Companion to api.py from the FDA Device Recall Explorer.
Targets /device/event instead of /device/recall.

KEY STRUCTURAL DIFFERENCES from /device/recall:
  - `device` is an array — a single report can name multiple devices.
    We take device[0] for simplicity (most reports cite one device).
  - `mdr_text` is an array of dicts with `text` and `text_type_code`.
    text_type_code values: "Description of Event or Problem" (code 2)
    is the patient narrative; "Manufacturer Narrative" (code B) is also
    useful. We concatenate all text blocks into one string for NLP.
  - Dates come from `date_received` (when FDA received the report)
    and `date_of_event` (when the incident actually occurred — often blank).
  - `event_type` is NOT reliably set server-side for filtering, so we
    filter client-side the same way classification is handled in the
    recall explorer.

Confirmed queryable fields on /device/event (Lucene syntax):
  date_received           "YYYYMMDD"
  device.generic_name     free text
  device.manufacturer_d_name  free text
  event_type              "IN" (injury) / "M" (malfunction) / "D" (death) /
                          "O" (other) / "* " (no answer)
  reporter_occupation_code  free text / coded

Fields parsed but NOT used as server-side search terms:
  mdr_text[*].text            narrative — fetched then processed in nlp.py
  mdr_text[*].text_type_code  to identify which block is the patient narrative
"""

import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

BASE_URL = "https://api.fda.gov/device/event.json"

# openFDA codes → human-readable labels
EVENT_TYPE_MAP = {
    "IN": "Injury",
    "M":  "Malfunction",
    "D":  "Death",
    "O":  "Other",
    "*":  "No Answer",
}

# text_type_code values we care about for NLP (others exist but are less useful)
NARRATIVE_CODES = {
    "1": "Description of Event or Problem",
    "2": "Description of Event or Problem",   # some records use "2" instead of "1"
    "B": "Manufacturer Narrative",
}

EVENT_TYPE_VALUES = list(EVENT_TYPE_MAP.values())  # for sidebar checkboxes

_PAGE_LIMIT = 100


# ── Query builder ─────────────────────────────────────────────────────────────

def _build_search_query(
    start_date: str | None = None,
    end_date: str | None = None,
    device_name: str | None = None,
    manufacturer: str | None = None,
) -> str:
    """
    Build an openFDA Lucene query string for /device/event.

    Date format for this endpoint is YYYYMMDD (no hyphens) — different
    from /device/recall which uses YYYY-MM-DD. This trips up many clients.

    event_type is NOT used here (filtered client-side; see fetch_events()).
    """
    parts = []

    if start_date and end_date:
        # Convert YYYY-MM-DD → YYYYMMDD
        s = start_date.replace("-", "")
        e = end_date.replace("-", "")
        parts.append(f"date_received:[{s} TO {e}]")
    elif start_date:
        s = start_date.replace("-", "")
        parts.append(f"date_received:[{s} TO *]")
    elif end_date:
        e = end_date.replace("-", "")
        parts.append(f"date_received:[* TO {e}]")

    if device_name and device_name.strip():
        safe = device_name.strip().replace('"', "")
        parts.append(f'device.generic_name:"{safe}"')

    if manufacturer and manufacturer.strip():
        safe = manufacturer.strip().replace('"', "")
        parts.append(f'device.manufacturer_d_name:"{safe}"')

    return " AND ".join(parts) if parts else ""


# ── Main fetch function ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_events(
    event_types: tuple[str, ...] | None = None,   # human labels e.g. ("Injury", "Malfunction")
    start_date: str | None = None,                # "YYYY-MM-DD"
    end_date: str | None = None,                  # "YYYY-MM-DD"
    device_name: str | None = None,
    manufacturer: str | None = None,
    max_records: int = 500,
) -> pd.DataFrame:
    """
    Fetch adverse event reports from openFDA /device/event.

    Paginates with skip/limit (same pattern as recall explorer's fetch_recalls).
    event_type is filtered client-side after fetch because server-side
    event_type queries are unreliable — many records have blank or
    non-standard codes.

    Parameters
    ----------
    event_types : tuple of str, optional
        Human-readable labels from EVENT_TYPE_MAP.values():
        "Injury", "Malfunction", "Death", "Other", "No Answer"
    start_date, end_date : str, "YYYY-MM-DD"
    device_name : str
        Searches device.generic_name (partial match supported)
    manufacturer : str
        Searches device.manufacturer_d_name
    max_records : int
        Hard cap on rows fetched (API allows max 1000 per skip window,
        but we stay well under that)

    Returns
    -------
    pd.DataFrame with columns defined in _parse_records()
    """
    query = _build_search_query(start_date, end_date, device_name, manufacturer)

    records: list[dict] = []
    skip = 0

    while len(records) < max_records:
        batch_size = min(_PAGE_LIMIT, max_records - len(records))
        params: dict = {"limit": batch_size, "skip": skip}
        if query:
            params["search"] = query

        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            if resp.status_code == 404:
                # 404 from openFDA means "no results for this query" — not a real error
                break
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"openFDA API error ({resp.status_code}): {e}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error: {e}") from e

        batch = data.get("results", [])
        if not batch:
            break

        records.extend(batch)

        total_available = data.get("meta", {}).get("results", {}).get("total", 0)
        fetched_so_far  = skip + len(batch)
        if fetched_so_far >= total_available or fetched_so_far >= max_records:
            break

        skip += len(batch)
        time.sleep(0.05)   # same throttle as recall explorer

    if not records:
        return pd.DataFrame()

    df = _parse_records(records)

    # ── Client-side event_type filter (mirrors recall explorer's classification filter) ──
    if event_types:
        et_set = set(event_types)
        if et_set != set(EVENT_TYPE_VALUES):
            df = df[df["event_type"].isin(et_set)]

    return df


# ── Record parser ─────────────────────────────────────────────────────────────

def _parse_records(records: list[dict]) -> pd.DataFrame:
    """
    Flatten the nested MAUDE JSON structure into a tidy DataFrame.

    MAUDE is significantly messier than /device/recall:
      - device[] is an array — we take device[0] for the primary device
      - mdr_text[] is an array of text blocks — we concatenate all narratives
      - date fields are YYYYMMDD strings (no hyphens, unlike recall endpoint)
      - Many fields can be None, "", or missing entirely
    """
    rows = []

    for r in records:
        # ── Dates ────────────────────────────────────────────────────────────
        # Format is YYYYMMDD — pandas handles this with format="%Y%m%d"
        date_received = pd.to_datetime(
            r.get("date_received", ""), format="%Y%m%d", errors="coerce"
        )
        date_of_event = pd.to_datetime(
            r.get("date_of_event", ""), format="%Y%m%d", errors="coerce"
        )

        # ── Device block — array, take first element ──────────────────────
        # The array can contain 0, 1, or many devices. Most reports have 1.
        devices = r.get("device") or []
        dev = devices[0] if devices else {}

        generic_name   = (dev.get("generic_name")   or "").strip()
        brand_name     = (dev.get("brand_name")      or "").strip()
        manufacturer   = (dev.get("manufacturer_d_name") or "").strip()
        model_number   = (dev.get("model_number")    or "").strip()
        device_age     = (dev.get("device_age_text") or "").strip()
        catalog_number = (dev.get("catalog_number")  or "").strip()

        # ── Narrative text — array of blocks ──────────────────────────────
        # Each block has: { text_type_code, text }
        # We join ALL blocks into one string for NLP — the topic model will
        # surface meaningful signal regardless of which code is which.
        mdr_text_blocks = r.get("mdr_text") or []
        narrative_parts = []
        for block in mdr_text_blocks:
            if isinstance(block, dict):
                txt = (block.get("text") or "").strip()
                if txt:
                    narrative_parts.append(txt)
        narrative = " ".join(narrative_parts)

        # ── Event type — map raw code to label ────────────────────────────
        raw_et    = (r.get("event_type") or "").strip()
        # API returns full words ("Injury", "Malfunction") not codes ("IN", "M")
        if raw_et in EVENT_TYPE_MAP.values():
            event_type = raw_et
        else:
            event_type = EVENT_TYPE_MAP.get(raw_et, "Other")

        # ── Reporter ──────────────────────────────────────────────────────
        reporter_code = (r.get("reporter_occupation_code") or "").strip()

        # ── Source type (hospital / home / etc.) ─────────────────────────
        # report_source_code: "H"=hospital, "U"=user facility, "D"=distributor
        # "P"=importer, "M"=manufacturer, "V"=voluntary
        source_code = (r.get("report_source_code") or "").strip()
        SOURCE_MAP = {
            "H": "Hospital",
            "U": "User Facility",
            "D": "Distributor",
            "P": "Importer",
            "M": "Manufacturer",
            "V": "Voluntary",
        }
        report_source = SOURCE_MAP.get(source_code, source_code or "Unknown")

        rows.append({
            "report_number":   r.get("report_number", ""),
            "date_received":   date_received,
            "date_of_event":   date_of_event,
            "event_type":      event_type,
            "event_type_raw":  raw_et,
            "generic_name":    generic_name,
            "brand_name":      brand_name,
            "manufacturer":    manufacturer,
            "model_number":    model_number,
            "device_age":      device_age,
            "catalog_number":  catalog_number,
            "narrative":       narrative,
            "reporter_code":   reporter_code,
            "report_source":   report_source,
            "num_devices":     len(devices),        # flag multi-device reports
        })

    df = pd.DataFrame(rows)

    # ── Derived columns ───────────────────────────────────────────────────────
    df["year"] = df["date_received"].dt.year
    df["year_month"] = (
        df["date_received"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    # Flag records that have usable narrative text for NLP
    df["has_narrative"] = df["narrative"].str.len() > 30

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_date_range_default() -> tuple[str, str]:
    """Default to last 3 years (MAUDE has millions of records — start narrower)."""
    end   = datetime.today()
    start = end - timedelta(days=365 * 3)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
