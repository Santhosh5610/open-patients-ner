# Approach — Medical NER on Open-Patients

## Problem

Extract medical entities from 1,000 clinical records, normalize them, compute frequencies, and map the top results to standard medical codes (ICD-10, RxNorm, SNOMED-CT). Deliver a dashboard and a chat interface over the results.

---

## 1. Data Understanding

**Dataset**: NCBI Open-Patients — 180,142 clinical patient descriptions from 4 sources.

**Sampling**: Pure proportional sampling would give ~0 TREC records (only 215 exist in 180K). I used hybrid stratified sampling: include all 215 TREC records, then proportionally sample PMC and USMLE to fill the remaining 785 slots. Final split: 729 PMC, 125 TREC-CT, 90 TREC-CDS, 56 USMLE.

**Key finding**: PMC records average 424 words (max 1,910) vs ~110 words for the other sources. This drives the chunking requirement — 42% of records exceed the 512-token model limit.

**Decision**: Chunk at 450 tokens with 50-token overlap. This produced 1,667 chunks from 1,000 records.

---

## 2. NER Extraction

**Model**: `d4data/biomedical-ner-all` (HuggingFace) — a BioBERT-based token classifier trained on multiple biomedical NER corpora. Selected because it natively recognizes the exact entity types needed: `Disease_disorder`, `Sign_symptom`, `Medication`, `Therapeutic_procedure`, `Diagnostic_procedure`.

**What I extracted** (4 categories):

| Model Label | My Category |
|---|---|
| Disease_disorder | Condition |
| Sign_symptom | Symptom |
| Medication | Medication |
| Therapeutic_procedure + Diagnostic_procedure | Procedure |

**Filtering applied during extraction**:
- Confidence threshold ≥ 0.60 (removes low-quality spans)
- Full-word boundary check (rejects partial token matches)
- Minimum length filtering (≤3 chars rejected unless in a valid medical abbreviation whitelist: CT, MRI, HIV, etc.)
- Subword artifact removal (any span containing `##` dropped)

**Raw output**: 21,361 entity mentions from 1,667 chunks.

---

## 3. Post-Processing & Normalization

This is where most of the quality comes from. Five sequential steps:

1. **Confidence filter** (≥ 0.60): 50,265 → 20,515
2. **Surface cleaning**: lowercase, strip punctuation, collapse whitespace, remove subword artifacts → 19,948
3. **Normalization**: canonical forms via a hand-curated variant map (`ct scan` → `ct`, `x-ray` / `xray` → `x-ray`, `blood pressure measurement` → `blood pressure`). Also merged plural→singular, removed trailing noise words → 19,524
4. **Within-chunk dedup**: same entity appearing multiple times in one chunk counted once → 17,037
5. **Within-record dedup**: same entity across chunks of one record counted once → 15,934 final entity–record pairs

**Final unique entities**: 6,544 across 993 records (7 records had zero qualifying entities).

**What I deliberately dropped**: `Age`, `Sex`, `Biological_structure`, `Lab_value`, `Clinical_event`, `Nonbiological_location`, and other model labels not in the 4 target categories. These are noise for this task.

---

## 4. Medical Code Mapping

For the **top 10 entities per category** (40 total), I mapped to standard coding systems:

| Category | System | Why |
|---|---|---|
| Conditions | ICD-10-CM | US standard for diagnosis coding |
| Symptoms | ICD-10-CM | Symptom codes live in Chapter R (R00–R99) |
| Medications (specific drugs) | RxNorm | NLM standard for drug ingredients (tty=IN) |
| Medications (drug classes) | ATC | "antibiotic" and "steroid" are classes, not ingredients — no single RxNorm CUI exists |
| Procedures | SNOMED-CT | Clinical terminology; ICD-10-PCS deliberately avoided (inpatient billing, not NLP coding) |

**Lookup approach**: Hybrid — NLM RxNorm API for medications, `simple_icd_10_cm` library validated against CDC April 2025 release for conditions/symptoms, SNOMED International browser for procedures.

**Why not fully automated**: The first automated pass produced clinically wrong codes. Examples: "pain" → N50.819 (Scrotal pain), "fever" → A78 (Q fever), "mass" → E68 (Sequelae of hyperalimentation). These are valid codes, but the wrong *kind* of specificity — the NLP extracted generic terms, so codes should be generic parent codes. I did 3 coding iterations, and the final 40/40 were manually verified.

**Parent-level coding decision**: When NLP extracts "infection" from text, we don't know *what* infection. Assigning B99 ("Other and unspecified infectious diseases") is honest. Assigning B34.9 ("Viral infection, unspecified") would be guessing.

---

## 5. Dashboard

Built with Streamlit + Plotly. Design decisions:

- **Metric cards**: Entity count and total mentions per category at a glance
- **Horizontal bar charts**: Top 10 per category, sorted by record count
- **Code lookup table**: Filterable by category, shows entity → code → description → system
- **No unnecessary UI complexity**: Dark theme for readability, no animations, no tabs within tabs

---

## 6. Chat Interface

Architecture: **Anthropic tool calling** (Claude Haiku 4.5)

- 5 tools: `lookup_entity`, `get_top_entities`, `compare_categories`, `get_dataset_summary`, `get_global_top_entities`
- All tools are deterministic DataFrame lookups — the LLM decides *which* tool to call, Python returns the *numbers*
- System prompt contains methodology and limitation context so the model can answer process questions without hallucinating
- The LLM never invents counts or codes

**Why tool calling over RAG**: The dataset is small (40 coded entities, 6,544 in frequency table). Function calling over structured DataFrames is simpler, faster, and eliminates retrieval noise. RAG would be overengineering.

---

## 7. What I Would Do Differently With More Time

- **Ensemble NER**: Run SciSpaCy + BioBERT + a clinical model, take majority vote
- **Negation detection**: Use NegEx or a negation-aware model to distinguish "no fever" from "fever"
- **Relation extraction**: Link conditions → medications → procedures within the same record
- **Full coding**: Extend medical code mapping beyond top 10 per category
- **USMLE question stripping**: Remove trailing diagnostic questions before NER (known noise source)

---

## 8. Reproducibility

Every intermediate CSV is committed. The notebooks are numbered and sequential — running `01` through `04` regenerates all outputs. The dashboard reads only from CSVs, so it works without rerunning the NER pipeline.
