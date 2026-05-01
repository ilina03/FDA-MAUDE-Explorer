# FDA MAUDE Explorer

[**Live App**](https://fda-maude-explorer.streamlit.app) · [**GitHub Pages**](https://ilina03.github.io/fda-maude-explorer)

A Streamlit dashboard for exploring FDA MAUDE (Manufacturer and User Facility
Device Experience) adverse event data pulled directly from the openFDA API.
Filter by event type, device name, and manufacturer — no account or API key needed.

Covers millions of records from 1991 to present, continuously updated by the FDA.

Companion to the [FDA Device Recall Explorer](https://github.com/ilina03/fda-device-recall-explorer).

\---

## What it does

MAUDE captures what happens in the field — patient injuries, device
malfunctions, and deaths reported by hospitals, manufacturers, and healthcare
providers. Where the recall explorer covers regulatory enforcement actions,
this tool covers the complaints and adverse events that precede or accompany them.

The FDA categorizes each report by event type:

* **Injury** — patient was harmed during or after device use
* **Malfunction** — device failed to perform as intended (no patient injury confirmed)
* **Death** — patient death associated with device use
* **Other** — reports that don't fit the above categories

The dashboard lets you slice that data across several views:

* **Trend chart** — report volume over time, monthly or quarterly, by event type
* **Device breakdown** — top devices and manufacturers with a sortable detail table
* **Failure Narratives** — NLP topic modeling on the free-text `mdr\_text` field,
surfacing common failure themes like "software alarm", "catheter fracture", and
"battery failure" without a fixed keyword list
* **Reporters \& Sources** — who is filing the reports (nurses, physicians, manufacturers)
and from where (hospitals, user facilities, voluntary submissions)
* **Raw data table** — full record view with CSV export

\---

## Run locally

```bash
git clone https://github.com/ilina03/fda-maude-explorer.git
cd fda-maude-explorer
python -m venv .venv \&\& source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run maude\_app.py
```

**Dependencies:** Python 3.10+, Streamlit, pandas, Plotly, requests, scikit-learn, numpy

\---

Data from [openFDA /device/event](https://open.fda.gov/apis/device/event/) ·
For research use only, not for clinical decision-making · MIT License

