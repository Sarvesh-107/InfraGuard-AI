# CURRENT HEADLINE NUMBERS (2026-09-11, after data-reading fixes)

> Sections further down were written before four data-reading bugs were fixed
> and quote slightly older figures. **Use this box.**

| Output | Accuracy (tested on unseen projects / projects that finished) |
|---|---|
| Risk score | Critical 81% slipped · High 62% · Medium 39% · Low 9% · top 50 → 92% (small sample, swings) |
| Risk model | XGBoost PR-AUC 0.654 grouped · 0.669 temporal · agency-forecast baseline 0.439 |
| Final cost | **81%** within ±1%, 85% within ±10% (278 finished projects) |
| End date | within 6 mo **61%**, 12 mo 77%, typical miss **5 mo** vs agency's own date 8 mo (82% too optimistic); worst case holds 94% |
| Live roster (Jun-2025) | 131 Critical · 288 High · ₹9.11 lakh cr expected cost in High+Critical |
| Portfolio | approved (first-reported original) ₹25.05 lakh cr → predicted final ₹30.23 lakh cr (**+20.7%**). Reconciles with the report's own TOTAL row: original ₹26.86 / anticipated ₹29.75 lakh cr — the gap is the 132 rewritten "original" costs |

**Data-reading bugs fixed (each now has a test in `test_v2.py`):**
1. May/Jun-2025 write revised dates as `Jun-2023` — unreadable, so all 536 live
   rescheduled projects looked "never revised".
2. Dec-24/Jan-25/Mar-25 put stray spaces inside dates (`2 / 2 0 2 6`) — ~880
   original dates silently dropped.
3. Nov-2024 names the column `Progress (%)` — a whole month of progress missing.
4. A missing progress change was counted as "no progress" — every project looked
   stalled in its first month and across the Nov-2024 gap.
5. From Dec-2024 approval dates are written `3-2019` — lost for 6 of 10 months,
   so project age / share of planned time used were guessed from averages.
6. The report OVERWRITES the "original" cost of 132 projects mid-year (mostly
   Apr-2025) with a newer sanction. First-reported original = approved; the
   rewrite is booked as a revision. Predicted final cost is never below a raised
   sanction (`forecast.expected_final_cost`). 27 projects no longer look under budget.
7. 2025 exports write 0% progress as "-". Read as 0% unless the project already
   had progress; a sudden 0% after >20% is a reporting glitch -> "not reported".

**Dashboard:** `streamlit run app.py` (from `Prototype/`). Layout follows the old
`src/dashboard/app.py`: 6 KPIs, Early Warning (charts + 15-column watchlist + detail
card with schedule chart), Prediction Models (accuracy, rules vs statistical vs ML,
risk drivers), Benchmarking, and the AI assistant (`assistant.py`, local Ollama
llama3, streamed; counts/totals computed in code, never by the model). Reads only
`reports/` and `models/`, never retrains. Pipeline order to refresh it:
`panel_v2 → train_v2 → score_v2 → hazard → forecast → backtest`, then `test_v2`.
The end-date accuracy on the dashboard comes from `backtest.py`, which uses
`forecast.project_dates` — the same rule the dashboard displays.

---

# Escalation-Risk Model — v2 (10-month panel)

**Question:** *which ongoing projects will declare a cost or schedule escalation
in the next ~3 months?* Ranked over the live roster, each with reasons.

> Supersedes v1 (`src/panel.py` / `train.py` / `score.py`), which was built on the
> 4-month `sih1` silver extract. The two datasets use **different project-code
> schemes** (`N24001287` vs `618087`) with **zero overlap**, so they cannot be
> joined; v2 replaces v1 rather than extending it.

## Data — `data/newdata/`
Monthly PAIMANA exports. `Table_3` = completed during month, `Table_4` = added,
`Table_6/7` = ongoing roster (Sept-2024 ships as `Table_6`, not `Table_7`).

- **10 clean monthly rosters**, Jul-2024 → Jun-2025; 17,096 rows, 1,974 projects,
  **1,414 projects observed in all 10 months**.
- 329 completion records (317 unique), 278 of which have prior roster history.
- Two new columns that v1 never had: **`Cost Anticipated`** and **`Date of
  Commissioning Anticipated`** — the agency's *own live forecast*, which moves
  before a formal revision is declared.

### Excluded, and why
| Period | Reason |
|---|---|
| 2024-06, 2024-08 ongoing | **Structurally corrupt**, not merely misaligned: cells hold pipe-joined fragments (`234.27 \| (N.A.) \| {234.27}`), project names split across fields, some rows interleave two projects. No parse recovers them without inventing numbers. |
| 2025-02 | Not present in the export set. |
| 2024-11, 2025-04, 2025-05 as *origins* | Their `t+3` snapshot does not exist, so they would contribute only completions — pure negatives with no escalating counterparts, biasing the base rate down. |

### Parsing semantics (defined once, in `newdata.py`)
- `N.A.` in a revised column means **not revised**, never zero → `effective_cost`
  / `effective_doc` fall back to the original.
- Dates arrive as `9/2018`, `11/2019` **and** Excel-mangled `Dec-24`.
- Every export ends with a **`TOTAL` footer row** (~₹27 lakh cr, no project code).
  Kept as a project it dominates every cost feature — dropped by code pattern.
- Sector labels are **truncated at varying widths**: `ROAD TRANSPORT AND` and
  `HIGHWAYS` are both `ROAD TRANSPORT AND HIGHWAYS`. An explicit alias map folds
  25 raw labels into **20 real sectors**. (A prefix rule was tried first and was
  wrong — it absorbed `CIVIL AVIATION` into a run-together value.)

## Panel
Training unit = **(project, as-of month t)**; features use only snapshots ≤ t,
labels read the snapshot at t+3. **10,273 rows over 6 origin months, 1,906
projects, 28 features.** Base rate **27.2%**.

`y_adverse` = effective **or anticipated** cost rises, or effective **or
anticipated** commissioning date slips >30 days, within 3 months. Projects that
leave the roster by **commissioning** are labelled negative — finishing is a
success, not an escalation.

## Results
**Protocol A — GroupKFold by `project_code`** (mandatory: each project
contributes 6 rows; letting two straddle a split leaks the answer).

| Model | PR-AUC | ROC-AUC | Brier | P@100 | Lift |
|---|---|---|---|---|---|
| baseline: always "no" | 0.272 | 0.500 | 0.272 | 0.45 | 1.65 |
| baseline: already overdue | 0.335 | 0.620 | 0.424 | 0.33 | 1.21 |
| baseline: already revised | 0.305 | 0.566 | 0.381 | 0.34 | 1.25 |
| baseline: current slip persists | 0.295 | 0.545 | — | 0.41 | 1.51 |
| baseline: **agency's own forecast** | 0.332 | 0.608 | — | 0.39 | 1.43 |
| Logistic regression (statistical) | 0.518 | 0.764 | 0.197 | 0.63 | 2.31 |
| Random forest (ML) | 0.630 | 0.832 | 0.166 | **0.86** | 3.16 |
| **XGBoost (ML)** | **0.649** | **0.841** | **0.138** | 0.85 | 3.12 |

**Protocol B — temporal holdout**, train Jul–Dec 2024 → test Jan/Mar 2025 (the
deployment condition):

| Model | PR-AUC | ROC-AUC | P@100 |
|---|---|---|---|
| baseline: **agency's own forecast** | 0.435 | 0.637 | 0.60 |
| Logistic regression | 0.534 | 0.723 | 0.65 |
| **Random forest** | **0.631** | 0.809 | **0.77** |
| XGBoost | 0.625 | 0.814 | 0.68 |

**The claim worth making:** the model beats the **agency's own live forecast** —
the strongest baseline in the data — by **+0.20 PR-AUC** under temporal holdout.
It is not just re-reading what agencies already admit.

**PS part (b), measured:** ML over statistical is **+0.131 PR-AUC** (0.649 vs
0.518) grouped, +0.097 temporal.

### What the 10-month history bought
| Feature | Gain |
|---|---|
| `f_n_antic_doc_changes` — times the agency has already moved its own forecast | **6.5%** |
| `f_mo_remaining` | 4.6% |
| `f_n_doc_changes` | 4.0% |

**21.7%** of total model importance comes from velocity/behaviour features that
require the panel and **did not exist in v1**. The top feature is a *behavioural*
one: an agency that has revised its forecast twice will revise it again.

### Where it works — and where it does not
| Elapsed share of planned duration | n | Base rate | PR-AUC |
|---|---|---|---|
| <50% | 541 | 0.04 | 0.108 |
| 50–100% | 1,558 | 0.14 | 0.453 |
| 100–150% | 1,244 | 0.36 | 0.624 |
| >150% | 1,818 | 0.30 | 0.662 |

**Honest limitation:** weakest on young projects, where warning is worth most
(though 0.108 against a 0.04 base rate is still a 2.7× lift). Late projects are
easy — being overdue mechanically forces a new date.

## Integrity (`python src/test_v2.py`)
- **Shuffle control passes**: permuted labels → PR-AUC 0.274 vs base rate 0.272.
- **As-of guarantee is tested, not asserted**: features are recomputed with
  history truncated at t and must match the full-history values exactly across
  all 28 features and 1,747 projects.
- Corrupt exports, TOTAL rows and truncated sectors each have a regression test.
- Encoders/imputers fit **inside** CV folds. Nothing synthetic.

## Deliberately not built
| Skipped | Why | Add when |
|---|---|---|
| Realised-overrun models (cost & final delay) | **attempted and rejected — see §Outcome models below**; both produce excellent-looking scores that are artifacts | the panel reaches ≥24 months, or ongoing projects are used as censored observations |
| Survival / Cox with censoring | 6 origin months is thin for hazard estimation | ≥12 origin months |
| Sequence models (GRU/TCN) | 1,906 projects | low thousands of completions |
| Sample weighting by `elapsed_share` | unvalidated; would trade the strong late signal for the weak early one | early-horizon performance becomes the priority |
| SHAP | `shap` not installed; fixed reason codes cover the need | explanations get contested |
| Entity resolution to the `sih1` era | zero code overlap; needs fuzzy name matching | a code crosswalk appears |

## Output
`reports/watchlist_2025-06.csv` — 1,595 live projects, scored 0–100, tiered,
each with plain-language reasons. 138 Critical + 373 High = **₹10.76 lakh cr**.

## Run
```
python src/panel_v2.py && python src/train_v2.py && python src/score_v2.py
python src/test_v2.py
```


---

# Outcome models (PS part (a)) — attempted, and rejected

`src/outcomes.py` builds the realised-overrun models the problem statement asks
for, and then demonstrates why **neither can be honestly validated on a 10-month
window**. It is kept as a runnable proof, not as a shipped model.

### 1. Final delay — the score is an artifact
The label decomposes into an input feature plus a window-truncated remainder:

```
y_delay  =  f_months_past_doc   +   months_to_go
            (already a feature)     (max 11 — capped by the panel, not the projects)
```

| Model | MAE (months) | R² |
|---|---|---|
| baseline: predict 0 | 40.0 | −0.954 |
| baseline: global median | 26.7 | −0.002 |
| **TRIVIAL: delay so far + median remainder** | **7.3** | 0.549 |
| tuned random forest | 6.1 | 0.761 |

The tuned model buys **1.2 months over arithmetic**. Its apparent skill is that it
only ever has to guess a number between 1 and 11. A project that will really
commission in 2030 needs `months_to_go = 60` — a value the model has never seen
and cannot output, so in deployment it would under-predict systematically.
**Verdict: do not ship.**

### 2. Cost overrun vs original — near-tautological
| | P(final cost > original) |
|---|---|
| cost **already revised** at t | 0.477 |
| **not yet revised** at t | 0.031 |

208 of 238 positive rows come from already-revised projects. A model here mostly
re-reads its own input feature `f_is_cost_revised`. **Verdict: do not ship.**

### 3. Further cost rise beyond what is declared — honest but too thin
Removing the tautology (target = final cost above the cost *declared at t*)
equalises the conditionals (0.064 vs 0.045) but leaves **16 positive projects out
of 278**. **Verdict: too thin to model.**

### 4. What the completions *do* support — no model required
> **249 of 278 completed projects (90%) commissioned later than the date they
> were still declaring while in flight.** Median delay at commissioning:
> **34 months**.

That is a defensible, model-free headline. The forward-looking model that *does*
validate on this window is the 3-month escalation model above.

**Why this section exists:** the naive versions of both models score well
(R² 0.76, PR-AUC 0.84) and would have survived a casual review. Reporting them as
findings would have repeated the earlier circular-result mistake in this project's
history. The check that caught it — decomposing the label and comparing against a
*trivial arithmetic* baseline rather than a weak one — is worth running on any
future target here.


---

# Projected completion date — how it is built (`hazard.py` + `forecast.py`)

A plain regression on "months until completion" is impossible here: only
projects that finished *inside* the 10-month window have an observed remaining
life, and it is **11 months at most**. Such a model can never output "finishes in
4 years" — it has never seen one.

### The hazard model
Instead we ask a question the data can answer: **did this project finish this
month?** ~1.8% did. Crucially, a project still running is not a missing outcome —
it is evidence it did *not* finish that month, which is how the ~1,900 unfinished
projects become training data instead of being discarded.

| Prediction | Validation |
|---|---|
| finishes within **3 months** | **ROC-AUC 0.835** |
| finishes within **6 months** | ROC-AUC 0.798 |
| ranking (finished next period) | ROC-AUC 0.873 |

Top 10% band finishes at **10.2%** per period vs **0.15%** for the bottom half —
a **68× separation**. This is a genuinely checkable prediction.

**What the hazard model must NOT be used for:** converting to an absolute date by
assuming today's completion chance holds forever. That gives a median remaining
life of **196 months (16 years)** with 45% at the cap — not credible, because a
project at 40% today will be at 80% in three years and its hazard will rise.
Calibration is fine (predicted 1.90% vs actual 1.79%), so this is a flaw in the
constant-hazard assumption, not a bug.

### How the date is actually assembled
Head-to-head backtest of remaining-months error (247 finished projects):

| Method | MAE (months) | Within 6 mo |
|---|---|---|
| **Agency's own date** | **6.5** | **72%** |
| Cohort benchmark | 15.2 | 49% |
| Pace-based | 22.8 | 68% |
| Hazard, constant-h | 65.2 | 24% |

**This table contains a trap and must not be read naively.** It can only be
evaluated on projects that finished *within 11 months* — exactly the population
where the agency's own date is accurate. The same agency date is **missed 85% of
the time** across all projects. Any method predicting "soon" wins this benchmark
by construction.

So the date is assembled to use each component only where it is valid:
- **hazard says the end is near (P(finish ≤12mo) ≥ 50%)** → use the agency's date
  (validated, ~6.5 months average error). 176 projects.
- **otherwise** → the later of the agency's date and the cohort benchmark, and the
  card is stamped `extrapolation - unverifiable beyond 1 year`. 1,419 projects.
- **P80** = benchmark, widened so the band always carries the observed
  median-to-pessimistic spread (median band width **28 months**).

Three invariants are enforced and tested: no date in the past, P80 never before
P50, no zero-width band.

### The honest bottom line
**A multi-year completion date cannot be validated with 10 months of data.** What
*is* validated is the near-term completion probability (ROC-AUC 0.835) and, for
the 176 projects genuinely close to finishing, the date itself. Everything beyond
that is labelled as extrapolation on the card rather than presented as a forecast.
