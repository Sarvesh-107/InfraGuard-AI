"""Project intelligence assistant -- adapted from the old dashboard's
src/llm/assistant.py to the real forecast cards.

Same design as before: pick the matching projects with plain filters, hand the
LLM a small table of numbers WE computed, and let it only narrate them. It never
invents a project, cost or date. Falls back to a templated answer when the
local Ollama server is not running, so the dashboard works without it.

Needs `ollama serve` with a model pulled, e.g. `ollama pull llama3`.
"""
import json
import re

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
TIMEOUT_S = 90            # max wait for the NEXT chunk; a cold model load takes 30-45 s
MAX_TOKENS = 280        # on this CPU the model writes ~5 words/s: uncapped answers took 80 s+

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


def retrieve(question, df):
    """-> (matching projects, plain-English description of the filter)."""
    sub, applied = df, []
    sectors = set(_mentions(question, df.sector.dropna().unique()))
    sectors |= {s for w, s in SECTOR_ALIASES.items() if re.search(rf"\b{w}", question, re.I)}
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
    if re.search(r"\bcritical\b", question, re.I):
        sub = sub[sub.risk_tier == "Critical"]
        applied.append("tier: Critical")
    elif re.search(r"\bhigh[ -]?risk\b|\brisky\b|\bat risk\b", question, re.I):
        sub = sub[sub.risk_tier.isin(["High", "Critical"])]
        applied.append("tier: High or Critical")
    return sub, ("; ".join(applied) or "whole portfolio")


def context_block(sub, df, applied, asof):
    cols = ["project_name", "sector", "state", "agency", "cost_original_cr",
            "cost_final_expected_cr", "cost_variance_cr", "cost_variance_pct",
            "date_original", "date_projected_p50", "slip_p50_mo",
            "physical_progress_pct", "risk_score", "risk_tier", "date_confidence", "why"]
    # only the 5 riskiest: with 15 rows the model ignored "don't list" and read them all out
    top = sub.sort_values("risk_score", ascending=False).head(5)[cols].copy()
    for c in ("date_original", "date_projected_p50"):   # "Mar 2026", not "2026-03-01"
        if pd.api.types.is_datetime64_any_dtype(top[c]):
            top[c] = top[c].dt.strftime("%b %Y")
    # Counts and totals are deliberately NOT given to the model: llama3 read
    # "matched_projects: 28" and answered "4 projects". header() states them instead.
    return (f"DATA AS OF: {asof}. FILTER: {applied}.\n\n"
            "THE 5 RISKIEST MATCHING PROJECTS (the only source for project facts):\n"
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
NOT_IN_DATA = re.compile(r"land|litigation|court|contractor|clearance|forest", re.I)
NOT_IN_DATA_RULE = ("Start with one sentence saying the monthly reports do not contain "
                    "land-acquisition, litigation, clearance or contractor information. ")
# Format rules go AFTER the context, right before the answer: placed at the top,
# llama3 forgot them by the time it had read the table.
ANSWER_RULES = (
    "Reply in under 80 words of plain language. Name the 3 most urgent projects from the "
    "table, each with its predicted end date and predicted final cost, and finish with one "
    "short practical recommendation. Do not give counts or totals -- they are shown "
    "separately. Do not list other projects."
)


def header(sub, applied):
    """Counts and totals computed here, never by the model, so they are always right."""
    if sub.empty:
        return f"**No projects match** ({applied}). Try another sector, state or risk tier."
    esc = sub.cost_variance_cr.sum()
    return (f"**{len(sub):,} projects match** ({applied}): "
            f"{(sub.risk_tier == 'Critical').sum()} Critical, {(sub.risk_tier == 'High').sum()} High "
            f"· predicted cost escalation {'+' if esc >= 0 else '−'}₹{abs(esc):,.0f} cr "
            f"· typical slippage {sub.slip_p50_mo.median():.0f} months.\n\n")


def _fallback(sub):
    t = sub.sort_values("risk_score", ascending=False).iloc[0]
    return (
        f"Highest risk: '{t.project_name}' ({t.sector}, {t.state}), risk "
        f"{t.risk_score:.0f}/100, predicted final cost Rs {t.cost_final_expected_cr:,.0f} cr "
        f"vs Rs {t.cost_original_cr:,.0f} cr approved, predicted end "
        f"{pd.to_datetime(t.date_projected_p50):%b %Y} ({t.slip_p50_mo:.0f} months late). "
        f"Why: {t.why}. Recommendation: review the Critical projects in this set first."
    )


def answer_stream(question, df, asof):
    """-> (generator of answer text, meta). The generator streams the model's words as
    they are written; meta["source"] is final once the generator is exhausted."""
    sub, applied = retrieve(question, df)
    ctx = context_block(sub, df, applied, asof)
    meta = {"source": "AI model (llama3, local)", "filter": applied, "matched": len(sub)}
    rules = (NOT_IN_DATA_RULE if NOT_IN_DATA.search(question) else "") + ANSWER_RULES

    def gen():
        yield header(sub, applied)
        if sub.empty:
            return
        got = False
        try:
            # connect timeout 5 s -> instant fallback when Ollama isn't running
            with requests.post(OLLAMA_URL, stream=True, timeout=(5, TIMEOUT_S), json={
                    "model": OLLAMA_MODEL, "stream": True, "keep_alive": "30m",
                    "options": {"num_predict": MAX_TOKENS},
                    "prompt": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {question}\n\n{rules}\n\nANSWER:"}) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    msg = json.loads(line) if line else {}
                    if msg.get("response"):
                        got = True
                        yield msg["response"]
                    if msg.get("done_reason") == "length":   # hit MAX_TOKENS mid-sentence
                        yield " …"
        except (requests.RequestException, ValueError):
            if got:          # failed mid-answer: say so rather than bolt on a summary
                yield "\n\n_(answer cut short -- the AI model stopped responding)_"
                return
        if not got:
            meta["source"] = "quick summary (AI model not running)"
            yield _fallback(sub)
    return gen(), meta


def answer(question, df, asof):
    g, meta = answer_stream(question, df, asof)
    return {"answer": "".join(g), **meta}
