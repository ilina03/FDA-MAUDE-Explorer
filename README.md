# FDA MAUDE Explorer

[**Live App**](https://fda-maude-explorer.streamlit.app) · [**GitHub Pages**](https://ilina03.github.io/FDA-MAUDE-Explorer)

A Streamlit dashboard for exploring FDA MAUDE (Manufacturer and User Facility
Device Experience) adverse event data pulled directly from the openFDA API.
Filter by event type, device name, and manufacturer — no account or API key needed.

Covers millions of records from 1991 to present, continuously updated by the FDA.

Companion to the [FDA Device Recall Explorer](https://github.com/ilina03/fda-device-recall-explorer).

---

## What it does

MAUDE captures what happens in the field — patient injuries, device
malfunctions, and deaths reported by hospitals, manufacturers, and healthcare
providers. Where the recall explorer covers regulatory enforcement actions,
this tool covers the complaints and adverse events that precede or accompany them.

The FDA categorizes each report by event type:

- **Injury** — patient was harmed during or after device use
- **Malfunction** — device failed to perform as intended (no patient injury confirmed)
- **Death** — patient death associated with device use
- **Other** — reports that don't fit the above categories

The dashboard lets you slice that data across several views:

- **Trend chart** — report volume over time, monthly or quarterly, by event type
- **Device breakdown** — top devices and manufacturers with a sortable detail table
- **Failure Narratives** — NLP topic modeling on the free-text `mdr_text` field,
  surfacing common failure themes like "software alarm", "catheter fracture", and
  "battery failure" without a fixed keyword list
- **Reporters & Sources** — who is filing the reports (nurses, physicians, manufacturers)
  and from where (hospitals, user facilities, voluntary submissions)
- **Raw data table** — full record view with CSV export

---

## The NLP feature

The recall explorer uses a fixed keyword list to analyze failure text. This tool
goes further: it runs **TF-IDF + NMF topic modeling** on the `mdr_text` narrative
field to automatically discover clusters of failure narratives in the data.

You can control the number of topics (3–10), drill into any topic to read
representative reports, and see which keywords are driving each cluster. Topics
shift when you change filters — a pacemaker query surfaces different failure
themes than a surgical robot query.

This runs entirely with scikit-learn — no model downloads, no GPU required,
deployable on Streamlit Community Cloud.

---

## One technical note

The MAUDE `/device/event` endpoint has a messier structure than `/device/recall`.
Two fields that catch many clients off guard:

`device` and `mdr_text` are **arrays**, not flat fields — a single report can
name multiple devices, and the narrative text is split across multiple typed
blocks with inconsistent `text_type_code` values across manufacturers. The app
concatenates all text blocks per report before passing them to the topic model.

Dates are also formatted as `YYYYMMDD` (no hyphens), unlike the recall endpoint
which uses `YYYY-MM-DD`. The API query strings must match the format of the field.

---

## Run locally

```bash
git clone https://github.com/ilina03/FDA-MAUDE-Explorer.git
cd fda-maude-explorer
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run maude_app.py
```

**Dependencies:** Python 3.10+, Streamlit, pandas, Plotly, requests, scikit-learn, numpy

---

Data from [openFDA /device/event](https://open.fda.gov/apis/device/event/) ·
For research use only, not for clinical decision-making · MIT License
