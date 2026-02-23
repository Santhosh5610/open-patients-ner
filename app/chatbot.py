"""
chatbot.py - NER Assistant: tool definitions, dispatch, system prompt, and chat turn runner.

The model handles intent; Python functions handle the actual data lookups.
That split is intentional, anything touching numbers goes through a DataFrame
so the model can't hallucinate stats. Static knowledge (methodology, limitations)
lives directly in the system prompt since it doesn't change.
"""

import os
import json
import re

import anthropic
import pandas as pd
import streamlit as st

from data import CATEGORIES, load_all

# ── Load once at module import,  no need to reload on every chat turn 
coded_df, entity_df, freq_df = load_all()


# HELPERS

def _parse_sources(src_str) -> str:
    if pd.isna(src_str):
        return "N/A"
    pairs = re.findall(r"'([^']+)':\s*(?:np\.int64\()?(\d+)\)?", str(src_str))
    return ", ".join(f"{k}: {v}" for k, v in pairs)


def _get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            key = ""
    return key



# TOOL FUNCTIONS
# Each of these is a plain DataFrame lookup — the model calls them
# by name, we run the query, hand the JSON result back.

def lookup_entity(entity_name: str) -> str:
    name = entity_name.strip().lower()

    coded_match = coded_df[coded_df["entity"] == name]
    if len(coded_match) > 0:
        r = coded_match.iloc[0]
        ent_row = entity_df[entity_df["entity"] == name]
        src = _parse_sources(ent_row.iloc[0]["sources"]) if len(ent_row) > 0 else "N/A"
        return json.dumps({
            "found": True, "in_top_10": True,
            "entity": r["entity"].title(), "category": r["category"],
            "rank": int(r["rank"]), "record_count": int(r["record_count"]),
            "avg_confidence": round(float(r["avg_confidence"]), 3),
            "code": str(r["code"]), "code_description": r["code_desc"],
            "code_system": r["code_system"], "lookup_method": r["lookup_method"],
            "sources": src,
        })

    freq_match = freq_df[freq_df["entity"] == name]
    if len(freq_match) > 0:
        r = freq_match.iloc[0]
        cat_data = (freq_df[freq_df["category"] == r["category"]]
                    .sort_values("record_count", ascending=False)
                    .reset_index(drop=True))
        rank_idx = cat_data[cat_data["entity"] == name].index
        return json.dumps({
            "found": True, "in_top_10": False,
            "entity": r["entity"].title(), "category": r["category"],
            "rank": int(rank_idx[0]) + 1 if len(rank_idx) > 0 else None,
            "total_in_category": len(cat_data),
            "record_count": int(r["record_count"]),
            "avg_confidence": round(float(r["avg_confidence"]), 3),
            "sources": _parse_sources(r.get("sources")),
            "note": "Not in top 10 — no medical code mapping available.",
        })

    return json.dumps({
        "found": False, "entity": entity_name,
        "note": (
            "Not found in any of the 6,544 extracted entities. "
            "It either didn't appear in the 1,000 records, was below "
            "the 0.60 confidence threshold, or was removed during normalization."
        ),
    })


def get_top_entities(category: str) -> str:
    cat = category.strip().title()
    if cat not in CATEGORIES:
        return json.dumps({"error": f"Unknown category '{category}'. Valid: {CATEGORIES}"})

    rows = coded_df[coded_df["category"] == cat].sort_values("rank")
    entities = []
    for _, r in rows.iterrows():
        ent_row = entity_df[entity_df["entity"] == r["entity"]]
        src = _parse_sources(ent_row.iloc[0]["sources"]) if len(ent_row) > 0 else "N/A"
        entities.append({
            "rank": int(r["rank"]), "entity": r["entity"].title(),
            "record_count": int(r["record_count"]),
            "code": str(r["code"]), "code_system": r["code_system"],
            "code_description": r["code_desc"], "sources": src,
        })

    total_unique = len(freq_df[freq_df["category"] == cat])
    return json.dumps({
        "category": cat, "top_10": entities,
        "total_mentions_top10": sum(e["record_count"] for e in entities),
        "total_unique_entities": total_unique,
    })


def compare_categories() -> str:
    results = []
    for cat in CATEGORIES:
        results.append({
            "category": cat,
            "top10_mentions": int(coded_df[coded_df["category"] == cat]["record_count"].sum()),
            "unique_entities_total": len(freq_df[freq_df["category"] == cat]),
        })
    return json.dumps({
        "categories": results,
        "grand_total_mentions": sum(r["top10_mentions"] for r in results),
        "grand_total_unique": len(freq_df),
    })


def get_dataset_summary() -> str:
    return json.dumps({
        "source": "Open-Patients (NCBI/HuggingFace)",
        "total_population": 180142, "sample_size": 1000,
        "sampling_method": "Hybrid stratified: all TREC records + proportional PMC/USMLE",
        "sources": {"PMC": 729, "TREC-CT": 125, "TREC-CDS": 90, "USMLE": 56},
        "records_with_entities": 993,
        "ner_model": "d4data/biomedical-ner-all",
        "pipeline": {
            "raw_extractions": 50265, "after_confidence_0.6": 20515,
            "after_cleaning": 19948, "after_normalization": 19524,
            "after_within_chunk_dedup": 17037, "after_within_record_dedup": 15934,
            "unique_entities": 6544, "coded_top_10": 40,
        },
    })


def get_global_top_entities(n: int = 5) -> str:
    """Top N entities by record count, ranked across all four categories combined."""
    n = max(1, min(int(n), 20))  # keep n sane — anything outside 1-20 is probably a mistake
    top = (freq_df.sort_values("record_count", ascending=False)
                  .head(n)
                  .reset_index(drop=True))
    results = []
    for _, row in top.iterrows():
        entry = {
            "rank": int(row.name) + 1,
            "entity": row["entity"].title(),
            "category": row["category"],
            "record_count": int(row["record_count"]),
        }
        coded_match = coded_df[coded_df["entity"] == row["entity"]]
        if len(coded_match) > 0:
            r = coded_match.iloc[0]
            entry["code"] = str(r["code"])
            entry["code_system"] = r["code_system"]
            entry["code_description"] = r["code_desc"]
        results.append(entry)
    return json.dumps({
        "note": "Ranked across ALL 6,544 entities from all 4 categories.",
        "top_entities": results,
    })


# TOOL REGISTRY
# These are the tool definitions we pass to the Anthropic API.
# The model reads the descriptions and picks which one to call.

TOOLS = [
    {
        "name": "lookup_entity",
        "description": (
            "Look up a specific medical entity by name across all 6,544 extracted entities. "
            "Returns count, category, rank, confidence, and medical code if in top 10."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Medical entity to look up, e.g. 'diabetes', 'pain', 'heparin'",
                }
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "get_top_entities",
        "description": (
            "Get the top 10 entities for ONE specific category (Condition, Symptom, "
            "Medication, or Procedure). Use get_global_top_entities instead when the "
            "question asks about the most/least frequent entity across ALL categories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": CATEGORIES,
                }
            },
            "required": ["category"],
        },
    },
    {
        "name": "compare_categories",
        "description": "Compare entity mention counts and unique entity totals across all 4 categories.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_dataset_summary",
        "description": "Return dataset statistics and NER pipeline processing numbers.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_global_top_entities",
        "description": (
            "Return the top N most-mentioned entities ranked across ALL 4 categories combined. "
            "Use this for questions like: 'what is the most/least mentioned entity overall?', "
            "'what are the top 5 entities?', 'which entity appears most frequently?' "
            "Do NOT use get_top_entities for these — it only searches within one category."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "How many top entities to return. Defaults to 5.",
                    "default": 5,
                }
            },
            "required": [],
        },
    },
]

_DISPATCH = {
    "lookup_entity":          lambda a: lookup_entity(a["entity_name"]),
    "get_top_entities":       lambda a: get_top_entities(a["category"]),
    "compare_categories":     lambda a: compare_categories(),
    "get_dataset_summary":    lambda a: get_dataset_summary(),
    "get_global_top_entities": lambda a: get_global_top_entities(a.get("n", 5)),
}

# ── System prompt — defines behavior, tool selection rules, and all static knowledge ──
SYSTEM_PROMPT = """You are a data assistant for a medical NER project on the Open-Patients dataset.
Answer questions about entities extracted from 1,000 clinical records.

RULES:
- For ANY data question (counts, codes, rankings), ALWAYS call a tool. Never invent numbers.
- For methodology/limitations questions, answer from the knowledge below.
- Be concise — this is a sidebar chat. 2–4 sentences unless a list is genuinely needed.
- Include medical codes and record counts when relevant.
- Never give clinical advice.

TOOL SELECTION RULES (follow exactly):
- "most/least mentioned", "highest/lowest frequency", "top entity overall", "most common overall"
  → ALWAYS use get_global_top_entities. Never use get_top_entities for cross-category questions.
- "top conditions / symptoms / medications / procedures" (one category named)
  → use get_top_entities with that category.
- "compare categories" or questions about which category dominates
  → use compare_categories.
- Specific entity name mentioned (e.g. "diabetes", "heparin", "CT")
  → use lookup_entity.
- Dataset size, pipeline steps, record counts, sources
  → use get_dataset_summary.

METHODOLOGY:
- 1,000 records sampled (hybrid stratified: all rare TREC + proportional PMC/USMLE)
- NER: d4data/biomedical-ner-all → Disease_disorder→Condition, Sign_symptom→Symptom, Medication, Procedure
- Post-processing: confidence≥0.60, normalization, variant merge, 2-level dedup
- Pipeline: 50,265 raw → 15,934 final pairs → 6,544 unique entities
- Top 10 per category manually coded: ICD-10-CM (Conditions/Symptoms), RxNorm (drugs), ATC (drug classes), SNOMED-CT (Procedures)

CODING RATIONALE:
- ICD-10-CM parent codes used for generic NLP terms (e.g. 'pain', 'infection') — specific sub-codes require clinical context the NLP doesn't have
- RxNorm for specific drug ingredients; ATC for drug classes like 'antibiotic'
- SNOMED-CT for procedures — ICD-10-PCS deliberately avoided (inpatient billing, not NLP coding)
- 3 coding iterations: NLM API pass 1 produced wrong codes ('pain'→Scrotal pain, 'fever'→Q fever); final 40/40 manually verified

LIMITATIONS:
- Single NER model, no ensemble
- No negation detection ("no fever" counted as positive)
- No relation extraction (can't link diabetes→insulin)
- 729/1000 records from PMC — skews toward unusual published cases
- Generic terms use parent-level ICD codes, not billable specifics"""


# CHAT TURN RUNNER

def run_turn(user_text: str, api_history: list) -> tuple[str, list[str], list]:
    """
    Run one full turn: send the message, loop through any tool calls,
    and return the final reply.

    Args:
        user_text:   The user's message.
        api_history: Everything said so far, in Anthropic API format.

    Returns:
        (reply_text, tools_used, updated_api_history)
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "⚠️ No API key found. Set `ANTHROPIC_API_KEY` as an env var "
            "or in `.streamlit/secrets.toml`.",
            [],
            api_history,
        )

    client = anthropic.Anthropic(api_key=api_key)
    history = api_history + [{"role": "user", "content": user_text}]
    tools_used: list[str] = []

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=history,
    )

    # Keep going until the model stops asking for tools
    while response.stop_reason == "tool_use":
        tool_results = []
        assistant_content = response.content

        for block in assistant_content:
            if block.type == "tool_use":
                tools_used.append(block.name)
                func = _DISPATCH.get(block.name)
                # Run the function if we know it, otherwise return a clear error
                result = func(block.input) if func else json.dumps({"error": "Unknown tool"})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        history.append({"role": "assistant", "content": assistant_content})
        history.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=history,
        )

    # Pull all text blocks out of the response and stitch them together
    reply = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    if not reply:
        reply = "I wasn't able to generate a response. Please try again."

    history.append({"role": "assistant", "content": response.content})

    return reply, list(dict.fromkeys(tools_used)), history  # dict.fromkeys deduplicates while preserving order