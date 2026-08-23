# Findings

What was actually tested this research cycle, and what came out of it —
including the negative results.

## 1. The original DAG/pruning design had real correctness bugs

Modeling conversation history as a DAG and deleting nodes judged irrelevant
(the project's original design) had three concrete failure modes:

- Deleting a node with weakly-linked children silently orphaned them —
  no dangling-reference check, no re-parenting, no warning.
- The structural "safety gate" against deletion was based on lexical
  keyword-overlap edge weights, which are forced to zero whenever the
  *parent* node doesn't hit a keyword — meaning any keyword-sparse message
  (a pivot, a transition) was always structurally deletable regardless of
  how important its children were.
- Branching nodes scored themselves independently of their children by
  design, so a low-scoring pivot message could never inherit the importance
  of an important sub-thread beneath it.

This is why the project moved from "prune" to "flag, never delete."

## 2. A continuously-updated anchor creates a circular dependency

If "what's relevant" is judged against a running summary, and the summary is
built from what's judged relevant, updating either one requires re-litigating
the other. Fix: freeze the anchor once, computed non-recursively, rather than
continuously re-derived.

## 3. One-shot "compress to a single goal" anchors are too abstract to be useful

Tested against a real 170-message conversation (a resume/interview-prep
session, not a crafted fixture). The original single-shot `ANCHOR_PROMPT`
("state it as a single concrete goal") produced:

> "The person is trying to get a job at De Shaw."

That's accurate, but too broad — almost anything career-adjacent can be
argued as "relevant" to it, so it has little power to distinguish real drift
from on-topic detail.

## 4. Two-stage extraction (summarize, then extract required parts) measurably improves anchor specificity

Replacing the one-shot compression with `summarize()` → `extract_required()`,
tested against the same real conversation at two window sizes:

- **k=3** (142s, two chained real Ollama calls): *"...evaluating the user's
  qualifications and experience in areas such as C++, Python, and operating
  system fundamentals."*
- **k=6** (173s): *"...including a RISC-V processor simulator, a mobile app,
  and an LLM-based context management system."* — it named the actual three
  projects being discussed, without ever being told their names.

Increasing `k` produced a demonstrably more grounded anchor, not just a
subjectively "nicer" one. This is a real, measured result, not an assumption.

## 5. Real cost: two-stage extraction is slow on a local 8B model

k=3 took 142s, k=6 took 173s (`llama3.1:8b` via local Ollama) — roughly 5-6x
slower than the original single-shot extraction (26-36s), because it's two
chained calls *and* each call processes more text as `k` grows. Not yet
tested against a hosted model (`AnthropicClient`), which would likely be
faster since it isn't competing for the same local CPU/GPU — but that
comparison was intentionally not run (would require an API key, and this
project's aim shifted to research over production tooling before that
tradeoff mattered).

## 6. LLM disambiguation's "reason" field is unreliable in practice

`classify_relevance` against real `llama3.1:8b` returned valid verdicts
(`RELEVANT`/`OFF_TOPIC`) both times tested, but the accompanying reason came
back empty once and as a bare repeat of the verdict ("OFF_TOPIC") the other
time — despite an explicit `VERDICT: reason` format instruction. The verdict
itself is usable; the explanation, which was the actual point of using an LLM
here instead of a bare threshold, currently isn't reliable. Not yet fixed —
candidates are a stricter structured-output format (e.g. JSON) or a few-shot
example in the prompt.

## 7. The bag-of-words mock embedder has a real length-bias confound on natural data

On the crafted 8-15 message fixtures, the mock (after fixing two earlier
bugs — random-noise vectors with no content signal, and punctuation breaking
token matching) produced sensible-looking band separation. On the real
170-message conversation, a clear pattern emerged instead: long messages
(mostly assistant turns) consistently scored higher regardless of content,
short messages (mostly user turns) scored consistently lower — a length
artifact of bag-of-words overlap, not a real topic signal. This wasn't
visible until testing against real, naturally-varied-length data; the
crafted fixtures didn't have enough length variance to expose it. Real
sentence-transformer embeddings would very likely not share this specific
confound, but that hasn't been verified — `Embedder()` has never been run
this session (see open items).

## 8. Real implementation bug: anchor extraction didn't filter by role

`AnchorExtractor` originally accepted plain strings with no role
information, so an assistant's own message could become an anchor candidate
if it was long enough to clear the noise filter — confirmed on the real
conversation, where the anchor's second candidate was initially an assistant
status message ("Read /areas/axiomchat.md... updated..."), not anything the
user said. Fixed by requiring role-tagged messages and filtering candidates
to `role == "user"` only.

## 9. The real `Embedder()` fails in this environment — a real, identified network problem

Starting `api.py` for real (it constructs `Embedder()` unconditionally at
startup) failed with `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
failed: unable to get local issuer certificate` while downloading
`all-MiniLM-L6-v2` from Hugging Face. This is a certificate-trust problem in
this specific environment (the kind typically caused by a proxy or antivirus
doing SSL inspection without its CA installed), not a bug in the code, and
not necessarily present on a real deployment host. It also exposed a real
design gap: `api.py` had no way to substitute `MockEmbedder`, unlike the LLM
client's existing provider switch — fixed by adding `get_embedder()`
(`AXIOM_EMBEDDER=mock|real`) to `embedder.py`, mirroring `get_llm_client()`.

## 10. Real bug: `Store`'s SQLite connection broke under FastAPI's threading model

With the embedder made swappable, `api.py` started successfully for the
first time — but `POST /sessions` then failed with `sqlite3.ProgrammingError:
SQLite objects created in a thread can only be used in that same thread.`
`Store`'s connection is created once at startup in the main thread; FastAPI
runs synchronous route handlers in a worker thread pool, a different thread
per request. This wasn't a rare edge case — it would have failed on every
single real request. Fixed with `check_same_thread=False` plus an explicit
`threading.Lock` around each connection use (lifting the check alone permits
cross-thread access but doesn't make concurrent access safe, only possible).

**After both fixes, `api.py` was verified end-to-end for the first time**:
server starts, `POST /sessions` creates a session with a real extracted
anchor, `POST /sessions/{id}/messages` correctly banded a repeated on-topic
message as `relevant` (cosine 1.0) and an unrelated one as `unrelated`
(cosine 0.12) — using `MockEmbedder`/`MockLLMClient` throughout, so this
confirms the request-handling and wiring logic, not real-model accuracy.

**Update:** the root cause of #9 was identified precisely — Avast Antivirus's
HTTPS-scanning root certificate doesn't mark its Basic Constraints extension
as critical, which Windows' own lenient validator accepts but Python's
stricter OpenSSL-based validator correctly rejects. The proper fix
(`truststore`/`pip-system-certs`, which delegate to the OS trust store)
needs a PyPI install, also blocked in this environment. With Avast's HTTPS
scanning temporarily disabled, `Embedder()` downloaded and cached
`all-MiniLM-L6-v2` successfully — it now works in this environment for all
future runs (model is cached locally; `HF_HUB_OFFLINE=1` avoids the library's
extra network probes for optional adapter configs even when re-scanning is
back on).

## 11. Single-anchor cosine thresholds are miscalibrated for real embedding behavior

With the real embedder finally working, it was run against the real
170-message conversation with a single, good, distilled anchor (from #4).
Result: **147/170 (86%) `unrelated`, zero `relevant`.** Concrete miss: *"ok,
so I can keep the parallel computing right"* — unambiguously on-topic —
scored 0.220 and landed `unrelated`. Across the run, genuinely on-topic
messages mostly scored 0.15-0.45; genuinely unrelated content mostly scored
-0.05 to 0.15 — real signal, but compressed into a lower, narrower band than
intuition suggests. The fixed thresholds (`low=0.3`, `high=0.6`) were always
placeholder defaults, never tuned — `0.3` sits in the middle of where
genuinely relevant content actually lands. This is a bigger problem than the
mock's length-bias artifact (#7): a single blended anchor point structurally
cannot represent a session that covers several real, distinct facets (resume
bullets, CUDA skill, LeetCode prep, GitHub concerns) — anything not close to
the *average* of those facets scores low, even when it's squarely on-topic
for one specific facet.

## 12. Multi-facet anchors (real embedder + real Ollama-identified facets) substantially fix this

Redesigned the anchor as a **set** of facets instead of one point:
`summarize()` → `extract_facets()` (a short list of distinct topics, not one
blended sentence), embedded separately, with each message scored by its
*best* match across all facets rather than distance to one averaged anchor.
Tested against the same real conversation, real embedder, real
Ollama-identified facets (`k=6`, `llama3.1:8b`):

Real facets extracted (genuinely distinct, not vague): building an
LLM-based context management system, developing a RISC-V processor
simulator, creating a mobile app, CUDA/parallel computing skills, applying to
De Shaw, COA/OS/DSA knowledge, algorithmic challenges. (One parsing bug: the
model's introductory sentence — "Here are the distinct topics..." — was
picked up as a spurious extra facet; needs a filter, not yet fixed.)

Result: **`unrelated` dropped from 86% to 67.6% (115/170), `relevant` went
from 0 to 5.** Concretely, *"ok, so I can keep the parallel computing
right"* — the exact miss from #11 — now scores 0.618 and correctly lands
`relevant`, matched against the real "improving parallel computing skills"
facet. Not a cherry-pick: this is the precise failure case the redesign was
built to fix, and it fixed it.

Honest nuance on the remaining 67.6%: much of it (GitHub-commit-verification
anxiety, resume-delimiter formatting, interview scheduling) may not be
misclassification at all — these are legitimate additional facets of the
conversation that simply hadn't emerged yet within the first `k=6` messages
the anchor was built from. Open question for later: should new facets be
allowed to accumulate as a session evolves, without violating the
non-recursive/no-continuous-re-derivation principle from #2? Not resolved.

**Real cost:** anchor extraction took 260s; the full scoring pass took 784s,
because 50 messages landed in the `ambiguous` band (more escalations, which
is partly *why* accuracy improved) and each disambiguation call's prompt now
lists all 11 facets, making every call slower too. Multi-facet is a real
accuracy win, but it multiplies both the escalation rate and the per-call
cost — not yet measured against a hosted model, which would likely help with
latency specifically.

## 13. Facet count/quality: three prompt iterations, only one real fix

Attempted to reduce facet redundancy (e.g. "CUDA programming" and "parallel
computing" listed separately) by capping the count. Three variants tested
against the same real conversation, real Ollama:

- **Capped at 6**: lost specificity — generalized real projects ("RISC-V
  processor simulator") into vague categories ("Course projects"). A
  regression, not an optimization.
- **Range 8-12**: the model ignored the range entirely and returned 19,
  padding the list with individual tool/language names lifted verbatim from
  the resume's skills table ("VS Code", "Google Colab", "SQL programming") —
  a local 8B model doesn't reliably hold multiple simultaneous constraints
  (be specific + merge duplicates + stay in a numeric range), consistent
  with #6's finding about unreliable format-following.
- **Root-caused the 19-facet overshoot**: both `SUMMARY_PROMPT` and
  `FACETS_PROMPT` explicitly instruct "preserve/keep concrete
  names/technologies" — applied to source text that's a literal skills-table
  enumeration, the model faithfully listed every item, which is correct
  instruction-following, not malfunction.

**The fix that actually worked**: not a better number, but
`Datahub.strip_structured_lists()` — a cheap heuristic (comma/pipe-delimited,
mostly-short segments, no sentence-ending punctuation) that drops
inventory-style lines before the text ever reaches `summarize()`, applied in
`AnchorExtractor.extract()`. Re-tested: **12 facets, zero tool-name
pollution**, and every facet is a genuine project/skill/topic. Also faster
(194.5s vs. 227-261s) since the input text is shorter. One remaining
imperfection: the CUDA/parallel-computing merge still doesn't fire
consistently run-to-run on identical input — further evidence of real
non-determinism in this model's instruction-following, not something a
prompt tweak reliably fixes.

## What's still unverified

- `AnthropicClient` has never executed — no API key was configured, and
  getting one was explicitly deferred once the project's framing shifted to
  research.
- `messages_fts` (FTS5) is populated on every write but never queried by any
  scoring logic — a real, working feature with zero current consumers.
- Whether new facets should accumulate over a long session, rather than
  being fixed forever at the first `k` messages (see #12).
- The full 170-message scoring pass has not yet been re-run against the
  final facet config from #13 — #12's accuracy numbers (67.6% unrelated, 5
  relevant) are from an earlier, tool-name-polluted facet list. A re-run is
  in progress.
