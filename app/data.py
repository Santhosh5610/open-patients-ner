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
    """Loads the top-10-per-category coded entities. This one's required — the dashboard won't work without it."""
    df = pd.read_csv("top10_coded.csv")
    df["entity"] = df["entity"].str.strip()
    return df


@st.cache_data
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (coded_df, entity_df, freq_df).

    coded_df is always required. entity_df and freq_df are optional —
    if their CSVs aren't present, we stub them out from coded_df so
    the chatbot still has something reasonable to work with.
    """
    coded_df = pd.read_csv("top10_coded.csv")
    coded_df["entity"] = coded_df["entity"].str.strip().str.lower()

    try:
        entity_df = pd.read_csv("top10_entities_v2.csv")
        entity_df["entity"] = entity_df["entity"].str.strip().str.lower()
    except FileNotFoundError:
        # File's missing — build a minimal stub so nothing downstream breaks
        entity_df = coded_df[["entity", "category"]].copy()
        entity_df["sources"] = None

    try:
        freq_df = pd.read_csv("entity_frequencies_v2.csv")
        freq_df["entity"] = freq_df["entity"].str.strip().str.lower()
    except FileNotFoundError:
        # Same deal — fall back to coded_df columns rather than crashing
        freq_df = coded_df[["entity", "category", "record_count", "avg_confidence"]].copy()
        freq_df["sources"] = None

    return coded_df, entity_df, freq_df