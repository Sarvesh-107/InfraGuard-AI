"""LLM-powered assistant grounded in PAIMANA forecast cards."""

import bootstrap  # noqa: F401

import streamlit as st

import assistant
from components.layout import setup_page
from components.styles import glass_divider
from data_loader import load_forecast_cards

setup_page(
    "assistant",
    title="Project Intelligence Assistant",
    subtitle="Answers use only dashboard numbers (never invented projects or figures)",
    page_title="Assistant | PAIMANA AI",
)

cards, _, asof_label = load_forecast_cards()

st.caption(
    "Cloud model (Groq / GPT-OSS 120B). If the API key is missing or the service is "
    "unavailable, you get a quick summary instead."
)
glass_divider()

examples = [
    "Which Railways projects are Critical risk?",
    "Summarize risk in Bihar",
    "Show high risk NHAI projects in Rajasthan",
    "Which airport projects are most delayed?",
]

if "assistant_prefill" in st.session_state:
    default_q = st.session_state.pop("assistant_prefill")
else:
    default_q = ""

st.session_state.setdefault("assistant_history", [])
st.session_state.setdefault("assistant_last_pick", "(write your own)")

pick = st.radio("Example questions", ["(write your own)"] + examples, horizontal=True)
if pick != "(write your own)" and pick != st.session_state.assistant_last_pick:
    # a fresh example topic -- follow-up context from the previous topic no longer applies
    st.session_state.assistant_history = []
st.session_state.assistant_last_pick = pick
if pick != "(write your own)" and not default_q:
    default_q = pick

with st.form("ask"):
    q = st.text_input("Your question", value=default_q)
    asked = st.form_submit_button("Ask")

if asked and q.strip():
    gen, meta = assistant.answer_stream(q, cards, asof_label, history=st.session_state.assistant_history)
    with st.container(border=True):
        st.markdown(f"**Q:** {q}")
        with st.spinner("Finding the matching projects…"):
            text = st.write_stream(gen)
        st.caption(
            f"Source: {meta['source']} · filter: {meta['filter']} · matched projects: {meta['matched']}"
        )
    st.session_state.ai_result = (q, {"answer": text, **meta})
    st.session_state.assistant_history.append({"question": q, "filter": meta["filter"]})
    st.session_state.assistant_history = st.session_state.assistant_history[-assistant.HISTORY_CAP:]
elif "ai_result" in st.session_state:
    q_, r = st.session_state.ai_result
    with st.container(border=True):
        st.markdown(f"**Q:** {q_}")
        st.write(r["answer"])
        st.caption(
            f"Source: {r['source']} · filter: {r['filter']} · matched projects: {r['matched']}"
        )

st.warning(
    f"Limits. Data stops at {asof_label}. End dates for projects more than a year "
    "from finishing are rough estimates and are labelled as such."
)
