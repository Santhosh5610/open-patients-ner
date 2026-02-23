import re
import streamlit as st
import plotly.graph_objects as go

from data import CATEGORIES, COLORS, load_coded
from chatbot import run_turn

# ── Page config 
st.set_page_config(
    page_title="Medical NER Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ── Session state — initialize on first load so nothing explodes later ────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "api_history" not in st.session_state: st.session_state.api_history = []

# GLOBAL CSS — dark theme, DM Sans/Mono, strip Streamlit chrome
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color-scheme: dark; }
.stApp, .main, [data-testid="stAppViewContainer"] { background: #0c0e18 !important; }

/* Hide sidebar and all the default Streamlit chrome we don't want */
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

/* Pull default padding so the columns actually reach the edges */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Left column — scrollable dashboard ── */
.dash-col {
    height: 100vh;
    overflow-y: auto;
    padding: 32px 36px 60px 36px;
    scrollbar-width: thin;
    scrollbar-color: #1c1f2e transparent;
}
.dash-col::-webkit-scrollbar { width: 4px; }
.dash-col::-webkit-scrollbar-thumb { background: #1c1f2e; border-radius: 4px; }

/* ── Right column — chat panel, fixed height, no overflow ── */
.chat-col {
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #090b13;
    border-left: 1px solid #171a28;
    overflow: hidden;
}

/* Chat panel header with title + live indicator */
.chat-head {
    padding: 18px 20px 14px;
    border-bottom: 1px solid #171a28;
    flex-shrink: 0;
}
.chat-title {
    font-size: 11px; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #f1f5f9;
    display: flex; align-items: center; gap: 8px;
}
.live-dot {
    width: 6px; height: 6px; background: #10b981; border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
    animation: blink 2.4s ease-in-out infinite; flex-shrink: 0;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
.chat-sub {
    font-size: 10px; color: #252840; font-family: 'DM Mono', monospace;
    margin-top: 4px; letter-spacing: .04em;
}

/* Quick-access suggestion chips above the chat */
.chips {
    padding: 10px 16px;
    border-bottom: 1px solid #171a28;
    display: flex; flex-wrap: wrap; gap: 6px;
    flex-shrink: 0;
}
.chip {
    font-size: 10px; font-family: 'DM Mono', monospace;
    background: #0f1120; border: 1px solid #1d2035;
    border-radius: 100px; padding: 4px 11px; color: #3d4468;
    cursor: pointer; transition: border-color .15s, color .15s;
    white-space: nowrap;
}
.chip:hover { border-color: #6366f1; color: #a5b4fc; }

/* Steamlit buttons default to white — override to match our dark chip style */
[data-testid="stButton"] button {
    background: #0f1120 !important;
    border: 1px solid #1d2035 !important;
    color: #4b5270 !important;
    border-radius: 8px !important;
    font-size: 11px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 400 !important;
    padding: 4px 8px !important;
    transition: border-color .15s, color .15s !important;
    box-shadow: none !important;
}
[data-testid="stButton"] button:hover {
    background: #13152a !important;
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
    box-shadow: none !important;
}
[data-testid="stButton"] button:focus {
    box-shadow: none !important;
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
}

/* ── Message bubbles — rolling our own HTML here because st.chat_message
   doesn't give enough styling control for the design ── */
.msg-row {
    display: flex;
    width: 100%;
    margin: 4px 0;
    box-sizing: border-box;
}
.msg-user {
    justify-content: flex-end;
    padding-right: 4px;
}
.msg-assistant {
    justify-content: flex-start;
    padding-left: 2px;
}
.msg-bubble-user {
    background: #151728;
    border: 1px solid #252840;
    border-radius: 14px 14px 4px 14px;
    padding: 9px 13px;
    max-width: 86%;
    color: #e2e8f0;
    font-size: 13px;
    line-height: 1.6;
    word-break: break-word;
}

/* Style the markdown output that assistant messages render into */
[data-testid="stMarkdownContainer"] p {
    font-size: 12.5px !important;
    line-height: 1.7 !important;
    color: #9ca8c8 !important;
    margin: 2px 0 6px !important;
}
[data-testid="stMarkdownContainer"] li {
    font-size: 12.5px !important;
    line-height: 1.7 !important;
    color: #9ca8c8 !important;
}
[data-testid="stMarkdownContainer"] strong {
    color: #dde3f5 !important;
    font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] code {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    background: #0f1120 !important;
    border: 1px solid #1d2035 !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    color: #a5b4fc !important;
}

/* Just in case any Streamlit avatars slip through, hide them */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

/* Strip the border and shadow Streamlit wraps around the message container */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    outline: none !important;
}

/* ── Chat input — a lot of overrides needed to get a clean single-border look ── */

/* Kill the outer wrapper border — we only want the textarea border */
[data-testid="stChatInput"] {
    background: #090b13 !important;
    border: none !important;
    border-top: 1px solid #171a28 !important;
    box-shadow: none !important;
    padding: 10px 14px 14px !important;
}

/* Flatten all the nested div borders Streamlit injects inside the input */
[data-testid="stChatInput"] div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] label {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* The actual textarea — one clean border, no doubling */
[data-testid="stChatInputTextArea"] {
    background: #0d0f1e !important;
    border: 1px solid #2a2d45 !important;
    border-radius: 14px !important;
    color: #c4cce8 !important;
    font-size: 12.5px !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 11px 14px !important;
    transition: border-color .2s !important;
    min-height: 44px !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: #6366f1 !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: #3a3f5c !important; }

/* Submit button — slate blue-grey, rounded */
[data-testid="stChatInputSubmitButton"] {
    background: #2e3350 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    min-width: 36px !important;
    width: 36px !important;
    height: 36px !important;
    transition: background .15s !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    background: #3a4060 !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: none !important; }
[data-testid="stChatInputSubmitButton"] svg path,
[data-testid="stChatInputSubmitButton"] svg polyline,
[data-testid="stChatInputSubmitButton"] svg line {
    stroke: #a8b4d0 !important;
    fill: none !important;
}

/* Small tag that shows which tool(s) the assistant called */
.tool-tag {
    font-size: 9px; font-family: 'DM Mono', monospace; color: #252840;
    display: flex; align-items: center; gap: 5px;
    padding: 0 2px 4px 2px; letter-spacing: .04em;
}
.tool-pip { width: 4px; height: 4px; background: #6366f1; border-radius: 50%; display: inline-block; }

/* Placeholder text when no messages exist yet */
.empty-hint {
    color: #1d2035; font-size: 11px; font-family: 'DM Mono', monospace;
    line-height: 1.9; text-align: center; padding: 40px 20px 0;
}

/* ── Dashboard cards and section headers ── */
.metric-card {
    background: #0f1120; border: 1px solid #171a28;
    border-radius: 14px; padding: 22px 20px; text-align: center;
}
.metric-label {
    font-size: 10px; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #252840; margin-bottom: 10px;
}
.metric-value { font-size: 38px; font-weight: 600; line-height: 1; margin-bottom: 6px; }
.metric-sub { font-size: 11px; color: #252840; font-family: 'DM Mono', monospace; }
.section-rule {
    font-size: 10px; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #1d2035;
    margin: 36px 0 16px; padding-bottom: 10px; border-bottom: 1px solid #171a28;
}
</style>
""", unsafe_allow_html=True)


# LAYOUT — load data, define suggestion chips, split into columns

df = load_coded()

SUGGESTIONS = [
    "Top conditions", "Diabetes ICD code",
    "Compare categories", "Dataset stats",
    "Top medications", "Limitations?",
]

dash_col, chat_col = st.columns([3.2, 1], gap="small")


# RIGHT — CHAT PANEL (header → chips → messages → input)

with chat_col:
    # Panel header — title and live indicator dot
    st.markdown("""
    <div class="chat-head">
        <div class="chat-title"><span class="live-dot"></span>NER Assistant</div>
        <div class="chat-sub">tool-calling · 6,544 entities · 40 coded</div>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion chips — each one is a small button that pre-fills the input
    st.markdown('<div class="chips">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="small")
    for i, label in enumerate(SUGGESTIONS):
        col = c1 if i % 2 == 0 else c2
        with col:
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                st.session_state["pending"] = label
    st.markdown('</div>', unsafe_allow_html=True)

    # Message history — fixed-height scrollable container, bubbles are raw HTML
    with st.container(height=480, border=False):
        if not st.session_state.messages:
            st.markdown(
                '<div class="empty-hint">Ask about any entity,<br>code, or methodology.</div>',
                unsafe_allow_html=True,
            )
        else:
            for m in st.session_state.messages:
                if m["role"] == "user":
                    st.markdown(
                        f'<div class="msg-row msg-user">'
                        f'<div class="msg-bubble msg-bubble-user">{m["content"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    tool_html = ""
                    if m.get("tool"):
                        tool_html = (
                            f'<div class="tool-tag">'
                            f'<span class="tool-pip"></span>{m["tool"]}'
                            f'</div>'
                        )
                    st.markdown(
                        f'<div class="msg-row msg-assistant">{tool_html}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(m["content"])

    # Chat input pinned to the bottom
    user_input = st.chat_input("Ask about entities, codes, methodology…")

    # If a chip was clicked, it stashes the label in session state — pull it out here
    if "pending" in st.session_state:
        user_input = st.session_state.pop("pending")


# PROCESS CHAT TURN — run the model, update history, rerun to render

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with chat_col:
        with st.spinner(""):
            reply, tools_used, updated = run_turn(user_input, st.session_state.api_history)

    st.session_state.api_history = updated
    tool_label = ("↳ " + " · ".join(tools_used)) if tools_used else None
    st.session_state.messages.append({
        "role": "assistant", "content": reply, "tool": tool_label,
    })
    st.rerun()


# LEFT — DASHBOARD (header → metrics → charts → table → footer)

with dash_col:
    # Dashboard page header with dataset subtitle
    st.markdown("""
    <div style="padding: 28px 0 20px;">
        <div style="font-size:10px;font-weight:700;letter-spacing:.16em;
                    text-transform:uppercase;color:#1d2035;margin-bottom:10px;">
            Open-Patients · 1,000 Clinical Records
        </div>
        <h1 style="font-size:28px;font-weight:600;color:#f1f5f9;margin:0;line-height:1.2;">
            Medical Named Entity Dashboard
        </h1>
        <p style="color:#252840;margin:8px 0 0;font-size:12px;font-family:'DM Mono',monospace;">
            Top 10 entities per category · ICD-10-CM / RxNorm / SNOMED-CT
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Four summary cards — one per entity category
    m1, m2, m3, m4 = st.columns(4, gap="small")
    for col, cat in zip([m1, m2, m3, m4], CATEGORIES):
        total = int(df[df["category"] == cat]["record_count"].sum())
        color = COLORS[cat]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{cat}s</div>
                <div class="metric-value" style="color:{color};">10</div>
                <div class="metric-sub">{total:,} mentions</div>
            </div>
            """, unsafe_allow_html=True)

    # Horizontal bar charts — two per row, one per category
    st.markdown('<div class="section-rule">Top 10 by Category — Record Count</div>',
                unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="medium")
    for idx, cat in enumerate(CATEGORIES):
        cat_df = df[df["category"] == cat].sort_values("record_count", ascending=True)
        c = COLORS[cat]
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)

        fig = go.Figure(go.Bar(
            x=cat_df["record_count"], y=cat_df["entity"], orientation="h",
            marker=dict(
                color=cat_df["record_count"],
                colorscale=[[0, f"rgba({r},{g},{b},0.1)"], [1, c]],
                line=dict(width=0),
            ),
            text=cat_df["record_count"], textposition="outside",
            textfont=dict(size=10, color="#252840", family="DM Mono"),
            hovertemplate="<b>%{y}</b><br>Records: %{x}<extra></extra>",
        ))
        fig.update_layout(
            height=300, margin=dict(l=10, r=50, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=cat, font=dict(size=12, color="#3d4468", family="DM Sans"),
                       x=0, pad=dict(b=6)),
            xaxis=dict(showgrid=True, gridcolor="#171a28", zeroline=False, showline=False,
                       tickfont=dict(size=9, color="#252840", family="DM Mono")),
            yaxis=dict(showgrid=False, zeroline=False, showline=False,
                       tickfont=dict(size=11, color="#6b7496", family="DM Sans")),
            hoverlabel=dict(bgcolor="#0f1120", bordercolor="#1d2035",
                            font=dict(size=12, color="#f1f5f9", family="DM Sans")),
        )
        with (ch1 if idx % 2 == 0 else ch2):
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Filterable lookup table for all 40 coded entities
    st.markdown('<div class="section-rule">Medical Code Lookup — Top 40 Entities</div>',
                unsafe_allow_html=True)

    fc, _, ic = st.columns([2, 4, 2])
    with fc:
        selected = st.selectbox("Filter", ["All"] + CATEGORIES, label_visibility="collapsed")
    with ic:
        st.markdown('<p style="text-align:right;color:#1d2035;font-size:10px;margin-top:10px;'
                    'font-family:\'DM Mono\',monospace;">ICD-10-CM · RxNorm · SNOMED-CT · ATC</p>',
                    unsafe_allow_html=True)

    display_df = (df if selected == "All" else df[df["category"] == selected]).sort_values(["category","rank"])
    table_df = display_df[["category","rank","entity","record_count","code","code_desc","code_system"]].rename(
        columns={"category":"Category","rank":"Rank","entity":"Entity","record_count":"Records",
                 "code":"Code","code_desc":"Description","code_system":"System"})

    st.dataframe(table_df, use_container_width=True, hide_index=True, height=430,
        column_config={
            "Category": st.column_config.TextColumn(width="small"),
            "Rank":     st.column_config.NumberColumn(width="small", format="%d"),
            "Entity":   st.column_config.TextColumn(width="medium"),
            "Records":  st.column_config.ProgressColumn(width="small", format="%d",
                            min_value=0, max_value=int(display_df["record_count"].max())),
            "Code":        st.column_config.TextColumn(width="small"),
            "Description": st.column_config.TextColumn(width="large"),
            "System":      st.column_config.TextColumn(width="small"),
        })

    # Footer — dataset attribution and coding standard versions
    st.markdown("""
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid #171a28;
                display:flex;justify-content:space-between;align-items:center;">
        <span style="color:#1d2035;font-size:10px;font-family:'DM Mono',monospace;">
            d4data/biomedical-ner-all · 993/1,000 records · 15,934 entity pairs
        </span>
        <span style="color:#1d2035;font-size:10px;font-family:'DM Mono',monospace;">
            ICD-10-CM validated · CDC April 2025 · SNOMED-CT verified
        </span>
    </div>
    """, unsafe_allow_html=True)