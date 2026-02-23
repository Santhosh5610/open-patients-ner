# Medical NER Pipeline — Open-Patients Dataset

> Named Entity Recognition, normalization, and medical code mapping on 1,000 clinical records from the [NCBI Open-Patients](https://huggingface.co/datasets/ncbi/Open-Patients) dataset.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/open-patients-ner.git
cd open-patients-ner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard + chatbot
cd app
export ANTHROPIC_API_KEY="sk-ant-..."   # required for chatbot only
streamlit run app.py
```

> **No API key?** The dashboard charts and code table still work — only the NER Assistant chatbot requires the key.

**Python**: 3.10+  
**Key libraries**: `transformers`, `torch`, `pandas`, `streamlit`, `plotly`, `anthropic`

---

## Repository Structure

```
open-patients-ner/
│
├── README.md                        ← You are here
├── APPROACH.md                      ← Methodology, decisions, limitations
├── requirements.txt                 ← All Python dependencies
├── .gitignore
│
├── notebooks/
│   ├── 01_data_profiling.ipynb      ← Dataset inspection & sampling strategy
│   ├── 02_preprocessing.ipynb       ← Text chunking (450 tokens, 50 overlap)
│   ├── 03_ner_extraction.ipynb      ← Entity extraction with d4data/biomedical-ner-all
│   └── 04_code_mapping.ipynb        ← Medical code assignment (ICD-10, RxNorm, SNOMED-CT)
│
├── app/
│   ├── app.py                       ← Streamlit dashboard + NER Assistant chatbot
│   ├── chatbot.py                   ← Tool-calling chat engine (Anthropic Claude Haiku)
│   ├── data.py                      ← Shared data loading & caching
│   ├── top10_coded.csv              ← Top 10 per category with medical codes (40 entities)
│   ├── top10_entities_v2.csv        ← Top 10 per category with source breakdown
│   └── entity_frequencies_v2.csv    ← All 6,544 unique entity frequencies
│
├── data/
│   ├── open_patients_1000_clean.csv         ← Cleaned input records
│   ├── open_patients_1000_preprocessed.csv  ← 1,000 sampled records
│   ├── open_patients_1000_profiled.csv      ← Profiled with length stats
│   ├── open_patients_chunks.csv             ← 1,667 chunks after splitting
│   ├── raw_entities.csv                     ← 21,361 raw NER extractions
│   └── filtered_entities_v2.csv             ← 15,934 after cleaning & dedup
│
└── screenshots/
    ├── dashboard_full.png
    └── dashboard_codes.png
```

---

## Pipeline Overview

| Step | Notebook | Input → Output |
|------|----------|----------------|
| **1. Data Profiling** | `01_data_profiling.ipynb` | 180,142 records → 1,000 stratified sample |
| **2. Preprocessing** | `02_preprocessing.ipynb` | 1,000 records → 1,667 chunks (450-token windows) |
| **3. NER Extraction** | `03_ner_extraction.ipynb` | 1,667 chunks → 21,361 raw → 15,934 filtered entities |
| **4. Code Mapping** | `04_code_mapping.ipynb` | Top 40 entities → ICD-10-CM / RxNorm / ATC / SNOMED-CT |

Each notebook is self-contained and produces its output CSV, which feeds the next step.

---

## Dashboard

The Streamlit dashboard (`app/app.py`) includes:

- **Metric cards** — 10 entities per category with total mention counts
- **Bar charts** — Top 10 entities for Conditions, Symptoms, Medications, Procedures
- **Code lookup table** — All 40 coded entities with ICD-10-CM / RxNorm / SNOMED-CT codes
- **NER Assistant chatbot** — Tool-calling chatbot that queries the extracted data deterministically

```bash
cd app
streamlit run app.py
```

The chatbot uses Anthropic tool calling (Claude Haiku 4.5). All numerical answers come from DataFrame lookups — the LLM never invents numbers.

---

## Key Results

| Category | #1 Entity | Record Count | Code |
|----------|-----------|:---:|------|
| **Condition** | infection | 27 | B99 (ICD-10-CM) |
| **Symptom** | pain | 309 | R52 (ICD-10-CM) |
| **Medication** | antibiotic | 58 | J01 (ATC) |
| **Procedure** | CT | 333 | 77477005 (SNOMED-CT) |

Full results in `app/top10_coded.csv` and detailed methodology in [APPROACH.md](APPROACH.md).

---

## Known Limitations

1. **Single NER model** — no ensemble; vulnerable to systematic errors
2. **No negation detection** — "no fever" counted as positive mention
3. **No relation extraction** — cannot link conditions to treatments
4. **PMC skew** — 729/1,000 records are published case reports, not population-representative
5. **Parent-level codes** — Generic terms map to unspecified ICD-10 parent codes
6. **Top 10 only coded** — Remaining 6,504 entities unverified

See [APPROACH.md](APPROACH.md) for full discussion.

---

## License

Assessment deliverable. Open-Patients dataset provided by NCBI under its original license.
