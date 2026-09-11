"""Checks for assistant.py's question -> sort-column mapping, list/count-question
handling (row cap, answer rules), short-term history formatting, the empty-result
risk-tier retry fallback, and the templated _fallback() answer's row selection.

    python test_assistant.py
"""
import pandas as pd

import assistant as A

# minimal columns retrieve() touches: sector/state/agency/risk_tier for filtering
_DF = pd.DataFrame({
    "sector": ["RAILWAYS", "CIVIL AVIATION"],
    "state": ["BIHAR", "ASSAM"],
    "agency": ["NHAI", "AAI"],
    "risk_tier": ["Critical", "High"],
})


def test_default_sort_is_risk_score():
    _, applied, col, ascending = A.retrieve("Summarize risk in Bihar", _DF)
    assert col == "risk_score" and ascending is False
    assert "sort: highest risk first" in applied
    print("ok  no sort keyword -> risk_score descending")


def test_delay_intent_sorts_by_slip():
    _, applied, col, ascending = A.retrieve("Which airport projects are most delayed?", _DF)
    assert col == "slip_p50_mo" and ascending is False
    assert "sort: most delayed first" in applied
    print("ok  'delayed' -> slip_p50_mo descending")


def test_cost_intent_sorts_by_cost_variance_pct():
    _, applied, col, ascending = A.retrieve("Which Railways projects have the worst cost overrun?", _DF)
    assert col == "cost_variance_pct" and ascending is False
    assert "sort: highest cost overrun first" in applied
    print("ok  'cost overrun' -> cost_variance_pct descending")


def test_progress_intent_sorts_by_progress_ascending():
    _, applied, col, ascending = A.retrieve("Which NHAI projects are stalled?", _DF)
    assert col == "physical_progress_pct" and ascending is True
    assert "sort: least progress first" in applied
    print("ok  'stalled' -> physical_progress_pct ascending")


def test_context_block_sorts_by_chosen_column():
    df = pd.DataFrame({
        "project_name": ["PROJ_LOWSLIP", "PROJ_HIGHSLIP", "PROJ_MIDSLIP"],
        "sector": ["RAILWAYS"] * 3, "state": ["BIHAR"] * 3, "agency": ["NHAI"] * 3,
        "cost_original_cr": [10, 10, 10], "cost_final_expected_cr": [12, 15, 11],
        "cost_variance_cr": [2, 5, 1], "cost_variance_pct": [20, 50, 10],
        "date_original": pd.to_datetime(["2025-01-01"] * 3),
        "date_projected_p50": pd.to_datetime(["2025-06-01"] * 3),
        "slip_p50_mo": [1, 9, 3], "physical_progress_pct": [80, 20, 50],
        "risk_score": [30, 90, 60], "risk_tier": ["Low", "Critical", "High"],
        "date_confidence": ["ok"] * 3, "why": ["-"] * 3,
    })
    ctx = A.context_block(df, df, "whole portfolio", "Jun 2025",
                           sort_col="slip_p50_mo", ascending=False)
    # PROJ_HIGHSLIP has the highest slip_p50_mo (9): it must lead the data rows
    assert ctx.index("PROJ_HIGHSLIP") < ctx.index("PROJ_MIDSLIP") < ctx.index("PROJ_LOWSLIP")
    assert "MOST DELAYED" in ctx  # header reflects the chosen sort column
    print("ok  context_block() sorts rows by the given column")


def test_list_question_detects_count_and_enumeration_intents():
    assert A.LIST_QUESTION.search("How many Railways projects are Critical?")
    assert A.LIST_QUESTION.search("List all NHAI projects in Rajasthan")
    assert A.LIST_QUESTION.search("Which projects are delayed in Bihar")
    assert A.LIST_QUESTION.search("Show me all of them")
    assert not A.LIST_QUESTION.search("Summarize risk in Bihar")
    print("ok  list/count question detection")


def test_context_block_row_cap_expands_for_list_questions():
    n = 8   # > DEFAULT_ROW_CAP (5), < LIST_ROW_CAP (15)
    df = pd.DataFrame({
        "project_name": [f"PROJ_{i}" for i in range(n)],
        "sector": ["RAILWAYS"] * n, "state": ["BIHAR"] * n, "agency": ["NHAI"] * n,
        "cost_original_cr": [10] * n, "cost_final_expected_cr": [12] * n,
        "cost_variance_cr": [2] * n, "cost_variance_pct": [20] * n,
        "date_original": pd.to_datetime(["2025-01-01"] * n),
        "date_projected_p50": pd.to_datetime(["2025-06-01"] * n),
        "slip_p50_mo": [1] * n, "physical_progress_pct": [50] * n,
        "risk_score": list(range(n)), "risk_tier": ["Low"] * n,   # PROJ_0 = lowest risk_score
        "date_confidence": ["ok"] * n, "why": ["-"] * n,
    })
    default_ctx = A.context_block(df, df, "whole portfolio", "Jun 2025")
    list_ctx = A.context_block(df, df, "whole portfolio", "Jun 2025", row_cap=A.LIST_ROW_CAP)
    assert f"THE {A.DEFAULT_ROW_CAP} " in default_ctx
    assert f"THE {n} " in list_ctx   # n < LIST_ROW_CAP, so the header reports all n rows
    # descending risk_score sort drops the lowest-scored rows first under the tight cap
    assert "PROJ_0" not in default_ctx
    assert "PROJ_0" in list_ctx
    print("ok  list/count question gets the larger row cap")


def test_fallback_retry_drops_tier_filter_when_it_zeros_the_result():
    df = pd.DataFrame({
        "sector": ["RAILWAYS", "RAILWAYS"],
        "state": ["BIHAR", "BIHAR"],
        "agency": ["NHAI", "NHAI"],
        "risk_tier": ["High", "Low"],   # neither is Critical
    })
    sub, applied, sort_col, sort_ascending, note = A._retrieve_with_fallback(
        "Which Railways projects are Critical risk?", df)
    assert not sub.empty and len(sub) == 2
    assert "tier:" not in applied   # tier filter was dropped on retry
    assert note and "No exact match for" in note
    print("ok  retry drops the risk-tier filter and explains the swap")


def test_fallback_retry_gives_up_when_still_empty():
    df = pd.DataFrame({
        "sector": ["RAILWAYS"], "state": ["BIHAR"], "agency": ["NHAI"], "risk_tier": ["High"],
    })
    # "airport" -> CIVIL AVIATION, which isn't in df at all, so dropping tier can't help
    sub, applied, sort_col, sort_ascending, note = A._retrieve_with_fallback(
        "Which airport projects are Critical risk?", df)
    assert sub.empty
    assert note == ""
    print("ok  retry gives up and returns the original empty result when still empty")


def test_fallback_follows_given_sort_not_always_risk_score():
    df = pd.DataFrame({
        "project_name": ["PROJ_HIGHRISK", "PROJ_HIGHSLIP", "PROJ_MID"],
        "sector": ["RAILWAYS"] * 3, "state": ["BIHAR"] * 3, "agency": ["NHAI"] * 3,
        "cost_original_cr": [10, 10, 10], "cost_final_expected_cr": [12, 15, 11],
        "cost_variance_cr": [2, 5, 1], "cost_variance_pct": [20, 50, 10],
        "date_original": pd.to_datetime(["2025-01-01"] * 3),
        "date_projected_p50": pd.to_datetime(["2025-06-01"] * 3),
        "slip_p50_mo": [1, 9, 3], "physical_progress_pct": [80, 20, 50],
        # risk_score and slip_p50_mo deliberately disagree on which row ranks first
        "risk_score": [90, 30, 60], "risk_tier": ["Critical", "Low", "High"],
        "date_confidence": ["ok"] * 3, "why": ["-"] * 3,
    })
    default_text = A._fallback(df)   # default: risk_score descending
    delay_text = A._fallback(df, sort_col="slip_p50_mo", ascending=False)
    assert default_text.index("PROJ_HIGHRISK") < default_text.index("PROJ_HIGHSLIP")
    assert delay_text.index("PROJ_HIGHSLIP") < delay_text.index("PROJ_HIGHRISK")
    print("ok  _fallback() row selection follows the given sort, not always risk_score")


def test_header_no_match_names_filters_without_sort_suffix():
    msg = A.header(pd.DataFrame(), "sector: CIVIL AVIATION; tier: Critical; sort: highest risk first")
    assert "sector: CIVIL AVIATION; tier: Critical" in msg
    assert "sort:" not in msg
    assert "No projects match" in msg
    print("ok  no-match header names filters, omits the irrelevant sort suffix")


def test_answer_stream_meta_source_reflects_no_model_call_on_empty_match():
    # full column set: context_block() runs (and needs these) even though sub ends up empty
    df = pd.DataFrame({
        "project_name": ["X"], "sector": ["RAILWAYS"], "state": ["BIHAR"], "agency": ["NHAI"],
        "cost_original_cr": [10], "cost_final_expected_cr": [12], "cost_variance_cr": [2],
        "cost_variance_pct": [20], "date_original": pd.to_datetime(["2025-01-01"]),
        "date_projected_p50": pd.to_datetime(["2025-06-01"]), "slip_p50_mo": [1],
        "physical_progress_pct": [50], "risk_score": [50], "risk_tier": ["High"],
        "date_confidence": ["ok"], "why": ["-"],
    })
    _, meta = A.answer_stream("Which airport projects are Critical risk?", df, "Jun 2025")
    assert meta["matched"] == 0
    assert meta["source"] == "no match"   # no Groq call happens for a genuinely empty result
    print("ok  meta['source'] doesn't claim a model call that never happened")


def test_history_block_empty_for_no_history():
    assert A._history_block(None) == ""
    assert A._history_block([]) == ""
    print("ok  no history -> empty block")


def test_history_block_includes_question_and_filter_not_answer():
    history = [{"question": "Summarize risk in Bihar",
                "filter": "state: BIHAR; sort: highest risk first"}]
    block = A._history_block(history)
    assert "Summarize risk in Bihar" in block
    assert "state: BIHAR; sort: highest risk first" in block
    print("ok  history block carries question + filter")


def test_history_block_caps_at_history_cap():
    history = [{"question": f"question {i}", "filter": f"filter {i}"} for i in range(6)]
    block = A._history_block(history)
    # only the most recent HISTORY_CAP turns should appear
    assert "question 5" in block and "question 3" in block
    assert "question 2" not in block and "question 0" not in block
    assert block.count("question ") == A.HISTORY_CAP
    print("ok  history block caps at HISTORY_CAP turns")


if __name__ == "__main__":
    test_default_sort_is_risk_score()
    test_delay_intent_sorts_by_slip()
    test_cost_intent_sorts_by_cost_variance_pct()
    test_progress_intent_sorts_by_progress_ascending()
    test_context_block_sorts_by_chosen_column()
    test_list_question_detects_count_and_enumeration_intents()
    test_context_block_row_cap_expands_for_list_questions()
    test_fallback_retry_drops_tier_filter_when_it_zeros_the_result()
    test_fallback_retry_gives_up_when_still_empty()
    test_fallback_follows_given_sort_not_always_risk_score()
    test_header_no_match_names_filters_without_sort_suffix()
    test_answer_stream_meta_source_reflects_no_model_call_on_empty_match()
    test_history_block_empty_for_no_history()
    test_history_block_includes_question_and_filter_not_answer()
    test_history_block_caps_at_history_cap()
    print("\nall checks passed")
