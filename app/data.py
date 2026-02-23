"""
data.py — Data loading and caching for the Medical NER Dashboard.
All pd.read_csv calls live here. Gracefully handles optional CSVs.
"""

import pandas as pd
import streamlit as st

CATEGORIES = ["Condition", "Symptom", "Medication", "Procedure"]

COLORS = {
    "Condition":  "#6366f1",
    "Symptom":    "#f59e0b",
    "Medication": "#10b981",
    "Procedure":  "#3b82f6",
}


@st.cache_data
def load_coded() -> pd.DataFrame:
    """Top-10-per-category coded entities. Always required."""
    df = pd.read_csv("top10_coded.csv")
    df["entity"] = df["entity"].str.strip()
    return df


@st.cache_data
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (coded_df, entity_df, freq_df).
    entity_df and freq_df are optional — falls back to stubs if files are missing.
    """
    coded_df = pd.read_csv("top10_coded.csv")
    coded_df["entity"] = coded_df["entity"].str.strip().str.lower()

    try:
        entity_df = pd.read_csv("top10_entities_v2.csv")
        entity_df["entity"] = entity_df["entity"].str.strip().str.lower()
    except FileNotFoundError:
        entity_df = coded_df[["entity", "category"]].copy()
        entity_df["sources"] = None

    try:
        freq_df = pd.read_csv("entity_frequencies_v2.csv")
        freq_df["entity"] = freq_df["entity"].str.strip().str.lower()
    except FileNotFoundError:
        freq_df = coded_df[["entity", "category", "record_count", "avg_confidence"]].copy()
        freq_df["sources"] = None

    return coded_df, entity_df, freq_df
