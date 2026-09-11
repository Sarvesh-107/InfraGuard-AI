"""Project intelligence assistant -- adapted from the old dashboard's
src/llm/assistant.py to the real forecast cards.

Same design as before: pick the matching projects with plain filters, hand the
LLM a small table of numbers WE computed, and let it only narrate them. It never
invents a project, cost or date. Falls back to a templated answer when the
Groq API call fails (missing key, rate limit, network error), so the dashboard
still works without it.

Needs a Groq API key (free tier): https://console.groq.com/keys
Set it as GROQ_API_KEY in .streamlit/secrets.toml locally (already gitignored)
and in the app's "Secrets" settings on Streamlit Community Cloud when deployed.
Never commit the key.
"""
import re

import pandas as pd
import streamlit as st
from groq import Groq, APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

GROQ_MODEL = "openai/gpt-oss-120b"   # swap for "openai/gpt-oss-20b" if free-tier rate limits bite during a demo
MAX_TOKENS = 280        # uncapped answers ran long and blew past the "under 80 words" rule anyway
# short-term memory: how many past (question, filter) turns the model gets to resolve
# follow-ups like "and in Bihar?". Only question + filter/sort are kept, never the
# narrated answer, to keep this cheap
HISTORY_CAP = 3
# a "how many"/"list all" question is asked to enumerate up to LIST_ROW_CAP rows instead of
# naming 3, which needs more room than MAX_TOKENS gives an ordinary answer -- tunable, same
# as LIST_ROW_CAP below. 700 measured as too tight for 11-15 rows (cut off mid-list); 1200
# covers 15 rows with room to spare
LIST_MAX_TOKENS = 1200

# everyday words -> sector names as they appear in the reports
SECTOR_ALIASES = {
    "road": "ROAD TRANSPORT AND HIGHWAYS", "highway": "ROAD TRANSPORT AND HIGHWAYS",
    "rail": "RAILWAYS", "airport": "CIVIL AVIATION", "aviation": "CIVIL AVIATION",
    "water": "WATER RESOURCES", "urban": "URBAN DEVELOPMENT", "metro": "URBAN DEVELOPMENT",
    "telecom": "TELECOMMUNICATIONS", "oil": "PETROLEUM", "gas": "PETROLEUM",
    "port": "SHIPPING AND PORTS", "shipping": "SHIPPING AND PORTS",
    "health": "HEALTH AND FAMILY WELFARE", "hospital": "HEALTH AND FAMILY WELFARE",
    "education": "DEPARTMENT OF HIGHER EDUCATION",
}


def _mentions(question, values):
    # whole-word match: plain substring matching let agency codes like "ER"
    # match inside ordinary words such as "where"
    q = question.lower()
    return [v for v in values if re.search(rf"\b{re.escape(str(v).lower())}\b", q)]


# question intent -> (sort column, ascending, label). Checked in order; first match wins.
_SORT_RULES = [
    (re.compile(r"\bdelayed?\b|\bslip(?:s|ped|ping|page)?\b|\bbehind schedule\b|\blate\b", re.I),
     "slip_p50_mo", False, "most delayed"),
    (re.compile(r"\bcost overrun\b|\bover budget\b|\bescalation\b|\bexpensive\b", re.I),
     "cost_variance_pct", False, "highest cost overrun"),
    (re.compile(r"\bstalled\b|\bno progress\b|\bbehind on progress\b|\bleast progress\b", re.I),
     "physical_progress_pct", True, "least progress"),
]
_DEFAULT_SORT = ("risk_score", False, "highest risk")
# label shown in context_block()'s header, keyed by sort column
_SORT_HEADERS = {
    "slip_p50_mo": "MOST DELAYED",
    "cost_variance_pct": "HIGHEST COST OVERRUN",
    "physical_progress_pct": "LEAST PROGRESS",
    "risk_score": "RISKIEST",
}


def _pick_sort(question):
    """-> (sort column, ascending, plain-English label) matching question intent."""
    for pattern, col, ascending, label in _SORT_RULES:
        if pattern.search(question):
            return col, ascending, label
    return _DEFAULT_SORT


def retrieve(question, df, apply_tier_filter=True):
    """-> (matching projects, plain-English description of the filter and sort,
    sort column, sort ascending). apply_tier_filter=False skips the risk-tier narrowing
    (used to retry a query that over-filtered to zero rows -- see _retrieve_with_fallback)."""
    sub, applied = df, []
    sectors = set(_mentions(question, df.sector.dropna().unique()))
    sectors |= {s for w, s in SECTOR_ALIASES.items() if re.search(rf"\b{w}\b", question, re.I)}
    if sectors:
        sub = sub[sub.sector.isin(sectors)]
        applied.append(f"sector: {', '.join(sorted(sectors))}")
    states = _mentions(question, df.state.dropna().unique())
    if states:
        sub = sub[sub.state.isin(states)]
        applied.append(f"state: {', '.join(states)}")
    agencies = [a for a in _mentions(question, df.agency.dropna().unique()) if len(str(a)) > 2]
    if agencies:
        sub = sub[sub.agency.isin(agencies)]
        applied.append(f"agency: {', '.join(agencies)}")
    if apply_tier_filter:
        if re.search(r"\bcritical\b", question, re.I):
            sub = sub[sub.risk_tier == "Critical"]
            applied.append("tier: Critical")
        elif re.search(r"\bhigh[ -]?risk\b|\brisky\b|\bat risk\b", question, re.I):
            sub = sub[sub.risk_tier.isin(["High", "Critical"])]
            applied.append("tier: High or Critical")
    sort_col, sort_ascending, sort_label = _pick_sort(question)
    filter_desc = "; ".join(applied) or "whole portfolio"
    return sub, f"{filter_desc}; sort: {sort_label} first", sort_col, sort_ascending


def _retrieve_with_fallback(question, df):
    """-> (sub, applied, sort_col, sort_ascending, retry_note). Same as retrieve(), but if
    the filters over-narrow to zero rows and a risk-tier filter was one of them (the most
    likely single cause of over-filtering to zero, e.g. a niche sector/state combined with
    "critical"), retries once with the risk-tier filter dropped. retry_note is "" unless
    that retry is what produced the results, in which case it explains the swap."""
    sub, applied, sort_col, sort_ascending = retrieve(question, df)
    if sub.empty and "tier: " in applied:
        original_applied = applied
        retry_sub, retry_applied, retry_sort_col, retry_sort_ascending = retrieve(
            question, df, apply_tier_filter=False)
        if not retry_sub.empty:
            note = (f"_No exact match for ({original_applied}); showing "
                     f"({retry_applied}) instead:_\n\n")
            return retry_sub, retry_applied, retry_sort_col, retry_sort_ascending, note
    return sub, applied, sort_col, sort_ascending, ""


# row cap for an ordinary question: with 15 rows the model ignored "don't list" and read
# them all out, so a normal answer only sees the top few
DEFAULT_ROW_CAP = 5
# row cap for a "how many" / "list all" question -- tunable, not a final answer: raise it
# further if the model still truncates enumerations, or lower it if it blows past
# LIST_MAX_TOKENS before finishing
LIST_ROW_CAP = 15


def context_block(sub, df, applied, asof, sort_col="risk_score", ascending=False,
                   row_cap=DEFAULT_ROW_CAP):
    cols = ["project_name", "sector", "state", "agency", "cost_original_cr",
            "cost_final_expected_cr", "cost_variance_cr", "cost_variance_pct",
            "date_original", "date_projected_p50", "slip_p50_mo",
            "physical_progress_pct", "risk_score", "risk_tier", "date_confidence", "why"]
    top = sub.sort_values(sort_col, ascending=ascending).head(row_cap)[cols].copy()
    for c in ("date_original", "date_projected_p50"):   # "Mar 2026", not "2026-03-01"
        if pd.api.types.is_datetime64_any_dtype(top[c]):
            top[c] = top[c].dt.strftime("%b %Y")
    # Counts and totals are deliberately NOT given to the model: llama3 read
    # "matched_projects: 28" and answered "4 projects". header() states them instead.
    header_label = _SORT_HEADERS.get(sort_col, sort_col.upper())
    return (f"DATA AS OF: {asof}. FILTER: {applied}.\n\n"
            f"THE {len(top)} {header_label} MATCHING PROJECTS (the only source for project facts):\n"
            + top.to_string(index=False))


SYSTEM_PROMPT = (
    "You are the PAIMANA project intelligence assistant for infrastructure monitoring "
    "officers at MoSPI. Answer ONLY from the CONTEXT: never invent project names, costs, "
    "dates or numbers that are not in it. Name projects by project_name. Quote predicted "
    "final cost in Rs crore and predicted end dates. Predicted end dates marked "
    "'extrapolation' are rough estimates -- say so when you quote one. If the context "
    "cannot answer the question, say so."
)
# Only added when the question asks: in the system prompt it made the model
# volunteer a land-acquisition disclaimer on unrelated questions.
NOT_IN_DATA = re.compile(r"\b(?:land|litigation|court|contractor|clearance|forest)\b", re.I)
NOT_IN_DATA_RULE = ("Start with one sentence saying the monthly reports do not contain "
                    "land-acquisition, litigation, clearance or contractor information. ")
# "how many X" / "list all Y" / "which projects..." / "all of them" -- questions that are
# structurally asking for a count or an enumeration, not a 3-item highlight reel. Gets the
# larger LIST_ROW_CAP context and LIST_ANSWER_RULES below instead of the normal 5/3 defaults.
LIST_QUESTION = re.compile(r"\bhow many\b|\blist all\b|\bwhich projects\b|\ball of them\b", re.I)
# Format rules go AFTER the context, right before the answer: placed at the top,
# llama3 forgot them by the time it had read the table.
ANSWER_RULES = (
    "Reply in under 80 words of plain language. Name the 3 most urgent projects from the "
    "table, each with its predicted end date and predicted final cost, and finish with one "
    "short practical recommendation. Do not give counts or totals -- they are shown "
    "separately. Do not list other projects."
)
# Used instead of ANSWER_RULES for a LIST_QUESTION: the "only 3, don't list the rest" rule
# is exactly what must NOT apply here, since the question is asking to enumerate.
LIST_ANSWER_RULES = (
    "List every matching project shown in the table above -- not just the top 3 -- each "
    "with its predicted end date and predicted final cost. Reply in plain language, under "
    "200 words. Do not give counts or totals -- they are shown separately."
)


def header(sub, applied):
    """Counts and totals computed here, never by the model, so they are always right."""
    if sub.empty:
        filters_only = re.sub(r"; sort: .*$", "", applied)   # sort isn't a filter -- omit it here
        return (f"**No projects match.** Applied filter(s): {filters_only}. Risk tier "
                "combined with a specific sector/state/agency is the most common cause "
                "of zero results -- try another sector, state, agency or risk tier.")
    esc = sub.cost_variance_cr.sum()
    return (f"**{len(sub):,} projects match** ({applied}): "
            f"{(sub.risk_tier == 'Critical').sum()} Critical, {(sub.risk_tier == 'High').sum()} High "
            f"· predicted cost escalation {'+' if esc >= 0 else '−'}₹{abs(esc):,.0f} cr "
            f"· typical slippage {sub.slip_p50_mo.median():.0f} months.\n\n")


def _fallback(sub, sort_col="risk_score", ascending=False):
    """Templated answer for when the cloud call fails or is unavailable. Uses the same
    sort_col/ascending context_block() used for the AI path, so e.g. a delay question
    still describes the most-delayed rows here, not whichever has the highest risk_score."""
    top = sub.sort_values(sort_col, ascending=ascending).head(3)
    parts = [
        f"'{t.project_name}' ({t.sector}, {t.state}): risk {t.risk_score:.0f}/100, "
        f"predicted final cost Rs {t.cost_final_expected_cr:,.0f} cr vs Rs "
        f"{t.cost_original_cr:,.0f} cr approved, predicted end "
        f"{pd.to_datetime(t.date_projected_p50):%b %Y} ({t.slip_p50_mo:.0f} months late). "
        f"Why: {t.why}."
        for t in top.itertuples()
    ]
    return " ".join(parts) + " Recommendation: review the Critical projects in this set first."


def _history_block(history):
    """-> short 'PREVIOUS QUESTIONS' text so the model can resolve a follow-up like
    "and in Bihar?" or "what about the second one?" against the prior turn, or "" if
    there's no history. Only question + filter/sort go in -- never the narrated answer --
    and it's capped at HISTORY_CAP turns regardless of how much the caller passes in."""
    if not history:
        return ""
    recent = list(history)[-HISTORY_CAP:]
    lines = "\n".join(f"- {h['question']} -> {h['filter']}" for h in recent)
    return (
        "PREVIOUS QUESTIONS THIS SESSION (only to resolve references like 'and in "
        f"Bihar?' or 'the second one' -- answer the CURRENT question, not these):\n{lines}\n\n"
    )


def answer_stream(question, df, asof, history=None):
    """-> (generator of answer text, meta). The generator streams the model's words as
    they are written; meta["source"] is final once the generator is exhausted. `history`
    is an optional list of {"question", "filter"} dicts from earlier turns this session."""
    history_block = _history_block(history)
    sub, applied, sort_col, sort_ascending, retry_note = _retrieve_with_fallback(question, df)
    is_list_question = bool(LIST_QUESTION.search(question))
    row_cap = LIST_ROW_CAP if is_list_question else DEFAULT_ROW_CAP
    max_tokens = LIST_MAX_TOKENS if is_list_question else MAX_TOKENS
    ctx = context_block(sub, df, applied, asof, sort_col, sort_ascending, row_cap=row_cap)
    # no model call happens below when sub is empty -- don't claim one did
    meta = {"source": "no match" if sub.empty else "AI model (Groq, cloud)",
            "filter": applied, "matched": len(sub)}
    answer_rules = LIST_ANSWER_RULES if is_list_question else ANSWER_RULES
    rules = (NOT_IN_DATA_RULE if NOT_IN_DATA.search(question) else "") + answer_rules

    def gen():
        yield retry_note + header(sub, applied)
        if sub.empty:
            return
        got = False
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            meta["source"] = "quick summary (GROQ_API_KEY not set)"
            yield _fallback(sub, sort_col, sort_ascending)
            return
        try:
            client = Groq(api_key=api_key)
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=max_tokens,
                stream=True,
                # gpt-oss is a reasoning model: without this it can burn the whole
                # MAX_TOKENS budget on hidden "analysis" tokens and never reach the answer
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f"{history_block}CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\n{rules}"},
                ],
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    got = True
                    yield delta
                if chunk.choices[0].finish_reason == "length":   # hit MAX_TOKENS mid-sentence
                    yield " …"
        except AuthenticationError:
            if got:
                yield "\n\n_(answer cut short -- the AI model stopped responding)_"
                return
            meta["source"] = "quick summary (invalid Groq API key)"
            yield _fallback(sub, sort_col, sort_ascending)
            return
        except RateLimitError:
            if got:
                yield "\n\n_(answer cut short -- the AI model stopped responding)_"
                return
            meta["source"] = "quick summary (Groq rate limit/quota hit)"
            yield _fallback(sub, sort_col, sort_ascending)
            return
        except (APIConnectionError, APITimeoutError):
            if got:
                yield "\n\n_(answer cut short -- the AI model stopped responding)_"
                return
            meta["source"] = "quick summary (network/timeout error)"
            yield _fallback(sub, sort_col, sort_ascending)
            return
        if not got:
            meta["source"] = "quick summary (AI model returned no output)"
            yield _fallback(sub, sort_col, sort_ascending)
    return gen(), meta


def answer(question, df, asof, history=None):
    g, meta = answer_stream(question, df, asof, history=history)
    return {"answer": "".join(g), **meta}
