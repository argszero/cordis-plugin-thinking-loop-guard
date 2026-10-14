# @argszero/cordis-plugin-thinking-loop-guard

Thinking-loop guard for the DeepSeek Harness (`dsh`). Detects an agent that
degrades into a **thinking loop** and reacts to break it before it burns tokens
until a human manually aborts the turn.

A "thinking loop" is one of three shapes, all of which a long-context model under
high reasoning effort can settle into:

1. **reasoning-only calls** — a step that emits only `reasoning-delta` chunks, with
   zero `text-delta` and zero `tool-call-delta`;
2. **restated-material calls** — a step that repeats most of the previous step's
   reasoning material (the same stalled conclusion re-derived in new words) while
   also emitting some text, so it *looks* like progress;
3. **self-repeating reasoning** — a single step whose own text is a low-entropy
   repetition ("好。执行。好。执行。").

Two further shapes never reach a call boundary at all — a call that bleeds for
minutes without ending, in the visible output or in its thinking — so they are cut
**from inside the stream** instead of judged after it; see [single-call
repetition](#single-call-repetition-maxrepeatedtext), [cycled
repetition](#cycled-repetition-maxrepeatedcyclechars) and [reasoning-channel
repetition](#reasoning-channel-repetition-maxrepeatedreasoningcyclechars).

The mid-stream rules measure a **shape** — a periodicity in the text. A sixth
shape is degenerate without being periodic at all: a call that keeps producing
sentences but has no new material left (twenty distinct sentences cycled, or the
same twenty shuffled), which no shape rule can see. That is measured as
**density** instead — see [reasoning that has run out of
material](#reasoning-that-has-run-out-of-material-minreasoningnoveltyratio).

## The gap it closes

The in-tree `guard/` family watches two failure modes, both **tool-call-centric**:

| Guard | Hook | Catches |
|---|---|---|
| `guard/timeout-policy` | `tools/execute` | a tool call that exceeds a declared `timeoutMs` |
| `guard/repeat-tool-reminder` | `tools/post-execute` | the model repeating the **same tool call** chain |

Neither fires when the model emits **no tool call at all** — the exact shape of a
thinking loop. When a model (observed with
`deepseek-v4.1-flash-expires-on-0910` under `max`/`high` reasoning effort and a
long context) keeps emitting low-entropy reasoning ("好。执行。好。", repeated
"Let me / Wait / Actually / Hmm"), the agent-loop `turn()` `while (true)` never
sets `turnEnds` (`StepEndReason` only carries `completed` / `max-tokens`), so the
turn never ends until the user aborts it. See
[deepseek-ai/deepseek-harness discussion #5976](https://github.com/deepseek-ai/deepseek-harness/discussions/5976).

## How it works

This plugin is a **community-side fix** that needs no harness patch. It observes
the public `llm/stream` waterfall (present on **dsh 0.1.2-rc.1 and 0.1.5-alpha.1**)
and tallies the `StreamChunk` composition of each loop-built model call. The live
`Agent` is reached from the request's `sessionId` via `ctx.agents.get(...)`,
so the guard works on the widely-installed 0.1.2-rc.1 as well as current master
(no dependency on the 0.1.5-alpha.1-only `agent/assistant-stream` seam).

- a chunk of type `text-delta` or `tool-call-delta` **resets** the counter — this
  call produced output, so it is not a thinking loop;
- a call that is **only** `reasoning-delta`, and at least `minReasoningChars`
  long, counts toward `maxThinkingSteps`.

When the threshold is crossed (or the reasoning text shows an unambiguous
low-entropy repetition, i.e. `repeatRatio`), it escalates:

1. `escalate: 'warn'` → `agent.inject(...)` a notice into the next pre-step;
2. `escalate: 'steer'` (default) → `agent.steer(...)` a "stop deliberating and act"
   steering message into the next step boundary;
3. `escalate: 'cancel'` → `agent.cancel(cause)` hard-aborts the active turn.

`agent.steer()`, `agent.inject()`, and `agent.cancel()` are all public methods on
the live `Agent`. A listener can react but cannot veto/rewrite an in-flight step;
steer and cancel are sufficient to break the loop.

`agent.cancel()` takes the **closed** `AgentCancelCause` union (`user` / `parent`
/ `disposed` / `hook`) and this plugin hands it the `hook` member it owns —
`{ kind: 'hook', reason: <cancelCause> }` — so the persisted turn ending reads
`{ kind: 'aborted', reason: { kind: 'hook', reason: 'thinking-loop' } }`. The
configured `cancelCause` string is the member's `reason`, not a bare cause, and
the config surface is unchanged by that. See [Notes](#notes) for why the object
matters.

## Single-call repetition (`maxRepeatedText`)

Everything above is judged **after a call ends** — which is structurally too late
for one reported failure:

> [discussion #2848](https://github.com/deepseek-ai/deepseek-harness/discussions/2848)
> — with a very large context, one model call repeated a single sentence for
> ~10 minutes: ~2825 identical text chunks, ~420,000 characters, until the user
> aborted it manually. No `finish` chunk, no stop reason, no `assistant/message`.

There is no call boundary to react at, so `maxRepeatedText` (default `60`) counts
consecutive **identical trimmed `text-delta` payloads** and, on the Nth, ends the
stream from inside the wrapper:

- the pending upstream deltas are abandoned (they are never flushed into the log);
- a terminal `finish` chunk with a `REPETITIVE_OUTPUT` failure is yielded, which
  fails the step through the loop's normal error path and reaches the
  `agent/request-error` waterfall, where a retry policy can act on the code;
- a steering correction is queued first, so the resumed turn is told what
  happened instead of silently continuing.

Two facts make this expressible from a plugin, both verified against the sources
and re-checked by running the wrapper's own output through the shipped
`@deepseek-ai/dsh-llm/invariant`:

- the agent loop consumes the `llm/stream` iterable itself
  (`for await (const chunk of stream) live.push(chunk)`), so a listener's chunk
  reaches the same assembler an adapter's does;
- the invariant **requires** a terminal finish chunk and explicitly permits an
  `error`/`aborted` finish with blocks still open. A quiet `stop` finish is *not*
  an option — the breaker fires while the call's text block is open, and the
  invariant rejects that with `finished with N open block(s)`.

Anchoring matters: only the **trailing** run is counted, so an early legitimate
burst (a heading, a repeated log prefix) followed by real work does not trip it,
while a call that is stuck right now does.

There is deliberately **no model fallback and no automatic retry** by this
plugin: silently re-billing a degenerate model would be worse than the loop.

## Cycled repetition (`maxRepeatedCycleChars`)

Counting identical *deltas* only covers the period-1 case. The next report of the
same failure mode is a **cycle**, and it slips straight through:

> [discussion #7043](https://github.com/deepseek-ai/deepseek-harness/discussions/7043)
> — a long session (≈150 turns, heavy tool use, Windows, `deepseek-flash`) where
> the assistant starts emitting the same few meaningless short lines for tens of
> lines — `好。 / 发。 / 好。 / 好。`, and a mixed-language variant with
> `Emitting.` in front — **where a tool call should have been**. The user sees
> "it stopped again mid-task"; the model never stopped generating, it just
> generated nothing else. Frequency rises with session length (0 in the first
> third, most turns in the last third), and while degenerated it also produced a
> malformed edit that deleted a function body.

Measured against the previous version, that shape **never fired the breaker** —
not under any of four chunkings (one delta per line, per cycle, per four
characters, per character) and not within 40 repeats: the longest run of
consecutive identical deltas is **2**, against a threshold of 60. A delta-level
rule cannot see a cycle.

`maxRepeatedCycleChars` (default `64`) adds the missing rule at the character
level: when the tail of the call's visible output is an **exact repetition** of
one period no longer than that, spanning at least `minRepeatedCycleChars`
(default `256`), the same terminal `REPETITIVE_OUTPUT` finish cuts the stream. On
the reported shapes it fires after ~256–270 characters instead of never.

Why characters and why exactness:

- **Characters**, not lines: the reported period is 12 and 26 characters, and a
  provider that streams the same bleed with no line break at all is caught
  identically. Nothing depends on where the newlines land.
- **Exactness**, not a low-entropy ratio: measured on the same texts, a coverage
  measure reads **0.92–0.99** for the two reported shapes at a 512-character
  tail, but **0.72** for a 40-row markdown table, **0.76** for a 60-line log
  listing, **0.82** for generated CSS rows and **0.84** for a JSON dump.
  Separating "degenerate" from "legitimately repetitive" by 0.06 is not a margin
  worth truncating a user's call over. Verbatim periodicity is **0** for every
  one of those samples.
- A period must contain **at least two distinct characters**, so a long `=====` or
  `─────` rule line in a generated document is never read as a loop.

Both rules share one breaker and report which one fired in the log line
(`rule: identical-chunks` / `rule: repeating-cycle`). `maxRepeatedCycleChars: 0`
disables the cycle rule alone; `maxRepeatedText: 0` disables the chunk rule
alone.

## Reasoning-channel repetition (`maxRepeatedReasoningCycleChars`)

Both rules above watch the call's **visible output**, and this plugin's per-call
detector is documented as deliberately blind to reasoning, which revisits itself
legitimately. The next report is the failure that blindness costs:

> [discussion #9524](https://github.com/deepseek-ai/deepseek-harness/discussions/9524)
> — `deepseek-vision` channel, flash-tier model, thinking in Chinese: the
> thinking block degenerates into self-replication of the **shortest standalone
> sentence** — `好。好。好。`, or `Go. Go. Go.` at an action point — measured at a
> **longest repeated fragment run of 7298**, ~99% of one block being a single
> repeated character, the reply swallowed and the session unusable. The same
> environment, same language, with a pro-tier model: no degradation at all.

Nothing in v0.1.x could act on it, and that is structural rather than a missed
threshold:

- the visible-output breaker never receives a `reasoning-delta` at all;
- the per-call detector *does* read reasoning (`repeatRatio`), but only **after**
  the call has ended — and then needs `maxThinkingSteps` consecutive stalled
  calls before it reacts.

That is inverted for a block that is 99% one character: the worst call of the
session is the call nothing can interrupt, and by the time the post-call rule
reads it the call has already spent its whole output budget.

`maxRepeatedReasoningCycleChars` (default `64`) runs the **same periodic-tail
scan** on the reasoning channel, in a sibling detector that ends the stream with
the same terminal `REPETITIVE_OUTPUT` finish and the same pre-queued correction.
It differs from the visible rule in exactly **one** dimension:

- **A one-character period is admitted when the repeated character is content**
  (a letter or a digit). The visible rule refuses period 1 to spare `=====` and
  `─────` divider lines — and that exemption is exactly wrong here, because the
  reported mechanism collapses onto the *shortest* unit: excluding period 1 by
  inheritance would leave the decisive case (a single character repeated
  thousands of times) uncatchable while the rule *appeared* to cover the
  channel. A run of punctuation or whitespace is still read as layout, so a
  divider in a thinking block is not truncated; the mirror-image miss is stated
  rather than papered over — a bleed of a *purely* punctuation character (a
  one-character period of `。`) is not caught, and the reported mechanism always
  produces a unit containing a word.

Every other threshold is deliberately shared (`minRepeatedReasoningCycleChars`
defaults to the same `256`, and the period ceiling to the same `64`), so the
difference between the two channels is one concept a reader can check rather
than a second table of tuned constants. The two channels are separately
configurable and separately disableable.

Why the span minimum is `256`: anchored on the reporter's own measurements. The
reported bleeds are `Go. ` × 238 (≈950 characters) and a single-character run of
LRS 7298, while the same reporter's clean sessions measure LRS 1–2 — so 256
characters of an exact period fires inside every reported bleed and sits far
above the healthy readings. Measured on the guard itself, the reported shapes are
cut after ~256 characters, where the severe one had already emitted thousands of
the same unit.

What it is not: **a guard, not a fix.** It ends the degenerate call and steers
the resumed turn; it does not make a model stop being prone to collapse. That a
pro-tier model does not degrade in the same environment says the propensity is
model- and tier-side, which is why this plugin deliberately offers no fallback
model and no automatic retry.

## Reasoning that has run out of material (`minReasoningNoveltyRatio`)

Every rule above measures a **shape**: is the tail periodic, is the call silent,
is this step materially the previous one. The next report is a call that defeats
all of them at once:

> [discussion #9683](https://github.com/deepseek-ai/deepseek-harness/discussions/9683)
> — a single thinking block of **114,298 characters containing only twenty
> distinct sentences**, laid out two ways in the report: cycled in a fixed order,
> and in an order with no period at all. Measured over 503 comparable blocks, 19
> behaved this way.

Shape rules are structurally blind to it, which is why this is a second
**statistic** and not a looser threshold:

- the cycle rule's period ceiling is 64 characters; the reported cycle's period
  is **489**, so no tail is ever periodic and the rule reports 0 at every span;
- in the irregular case there is no period to find in the first place;
- the per-call rules ask whether a *step* is stalled, and a call like this emits
  text, so it looks like progress from the outside.

What it does not have is anything new to say. `minReasoningNoveltyRatio`
(default `0.05`) measures exactly that: the share of one call's **sentence
units** that are distinct, counted over the whole call rather than its tail.

- A **sentence unit** is the text between two sentence terminators (`.` `!` `?`
  `。` `！` `？` `…` or a newline), trimmed. Units shorter than
  `minReasoningSentenceChars` (default `8`) are dropped.
- There is **no periodicity requirement**. Order is irrelevant, which is the
  point: the cycled and the irregular layouts of #9683's block measure the same.
- **Whole units, never substrings.** This is load-bearing and was measured on
  #5976's own text: the substring `好。` occurs **4** times there while the text
  contains **3** units, because `完好。` ("intact") contains `好。` inside an
  ordinary word. A substring count reports repetitions no reader can see.
- `minReasoningSentences` (default `64`) is a floor on how many units a call must
  produce before its share is judged at all — a three-sentence call cannot
  demonstrate that it is out of material, and a short burst is normal.

### How the three numbers were picked

They are measured, not chosen (`measure-5..7.mjs` in the delivery notes):

| | measured |
|---|---|
| reported shapes | `0.0001` (one repeated sentence), `0.0043` (the twenty-sentence cycle, either order) |
| real text — 60 recorded reasoning blocks + 3,983 long assistant messages | never below **0.885** among the 96 that reach the unit floor |
| the plugin's own generated-document controls (table, CSS, JSON dump, log listing, code stubs, bullet list) | never below **0.35** |

Two orders of magnitude separate the reported shapes from real text, and `0.05`
sits in that gap — about the geometric mean of its two nearest neighbours, so the
exact value inside the gap is not load-bearing. The **unit floor** is what the
last row pins: a JSON dump repeats its own `"kind": "item",` line forty times, and
only the distinct `"id": N,` line on every record keeps the measure high — at a
floor of **10** those id lines stop counting and the same dump measures `0.025`,
i.e. it fires falsely. At 8 it measures `0.35`. It is also the smallest floor at
which a seven-character Chinese clause plus its `。` is still countable, so it is
pinned from both sides.

### What it still cannot see

Stated rather than papered over. #5976's **fifth** reported form is short
two-character fragments (`好。` / `发。` / `写。`) interleaved among ordinary
sentences — 142 of them in a 2,173-character step, against a same-session
baseline of 0–3 per thousand. Its statistic is fragment *density*, not repetition
*share*: the fragments are short enough that the unit floor drops them, and the
ordinary sentences around them keep the share high. Catching it needs the
reporter's own per-call reasoning to calibrate against, which has been asked for
and not yet received; a threshold picked without it would be a guess.

**Scope:** reasoning channel only, deliberately. The visible channel legitimately
carries generated documents whose lines repeat (the JSON dump above is exactly
that), so its repetition rules stay the two exact-period ones.

## Install

Load it as an `@deepseek-ai/cordis` plugin in your `dsh` profile, or mount the
bundle patch:

```yaml
# cordis.patch.yml (already packaged in this plugin)
- insert:
    - id: thinking-loop-guard
      name: '@argszero/cordis-plugin-thinking-loop-guard'
```

## Configuration

```ts
interface Config {
  /** Consecutive stalled calls before reacting. Default 3. */
  maxThinkingSteps?: number
  /** Minimum reasoning text in one call before it is judged at all. Default 2048 chars. */
  minReasoningChars?: number
  /** Intra-call repeated-gram coverage at which a call is flagged. Language-agnostic (handles CJK, no whitespace). Default 0.5. */
  repeatRatio?: number
  /** Cross-call similarity: how much of the previous call's distinct reasoning must reappear before this call counts as a repetition. 0 disables. Default 0.8. */
  similarityThreshold?: number
  /** Action on a threshold crossing: 'warn' | 'steer' (default) | 'cancel'. */
  escalate?: 'warn' | 'steer' | 'cancel'
  /** How many times one agent may be reacted to before the guard stops re-firing. Default 4. */
  maxFires?: number
  /** Cancel cause *reason* when escalate is 'cancel'. Default 'thinking-loop'. */
  cancelCause?: string
  /** Consecutive identical visible-output chunks that end the stream mid-call. 0 disables. Default 60. */
  maxRepeatedText?: number
  /** Longest repeating period of the visible output, in characters, that ends the stream mid-call. 0 disables. Default 64. */
  maxRepeatedCycleChars?: number
  /** Shortest tail span that must be an exact repetition of one period before the cycle rule fires. Default 256. */
  minRepeatedCycleChars?: number
  /** Longest repeating period of the call's REASONING, in characters, that ends the stream mid-call. 0 disables. Default 64. */
  maxRepeatedReasoningCycleChars?: number
  /** Shortest reasoning tail span that must be an exact repetition of one period before the reasoning rule fires. Default 256. */
  minRepeatedReasoningCycleChars?: number
  /** Shortest sentence unit of the REASONING, in characters, that counts toward the density measure. Default 8. */
  minReasoningSentenceChars?: number
  /** Sentence units a call must produce before its density is judged at all. Default 64. */
  minReasoningSentences?: number
  /** Lowest share of a call's reasoning sentence units that must be distinct before the call is judged degenerate. 0 disables. Default 0.05. */
  minReasoningNoveltyRatio?: number
  /** Error code on a mid-stream break. Default 'REPETITIVE_OUTPUT'. */
  breakCode?: string
  /** Steer the agent after a mid-stream break so the resumed turn is corrected. Default true. */
  breakCorrection?: boolean
}
```

## Notes

- Per-`Agent` state is kept in a `WeakMap`, so a disposed agent is collected and
  its counters dropped.
- **A reaction does not latch.** A single steer often does not break a strong loop,
  so after firing the run counter resets and the guard re-fires after another
  `maxThinkingSteps` stalled calls, up to `maxFires` times. Set `maxFires: 1` to
  restore the old one-shot behaviour.
- The cross-call signal is **containment of the smaller gram set**, not Jaccard
  similarity. A stuck model usually restates the previous step's material and
  appends another sentence; Jaccard would dilute that with the new material and
  read the step as progress, while containment reports the repetition directly.
- A call shorter than `minReasoningChars` is neither counted nor treated as
  progress: a brief "ok, continuing" must not clear a run built from substantive
  steps.
- Both content checks are cheap O(n) heuristics over fixed-length grams
  (language-agnostic, so CJK with no whitespace works) and run only once a call is
  already long, so cost is bounded.
- **The cancel cause must be a union member.** `AgentCancelCause` is closed
  (`packages/core/session/src/types.ts`), and the loop reads the abort signal back
  with `abortedCancelCause()` (`packages/core/agent-loop/src/agent.ts`), whose last
  branch is `default: return assertNever(cause)`. Handing it a bare string throws
  **inside the turn's `catch`**, so the `turnEnds` assignment never runs and the
  `finally` writes `turn/end` with `reason: null`. Nothing on the write path
  checks that field, so the null reaches the log — and one null reason is enough to
  make the client's turn rendering throw and blank **every** session in the list
  (discussions #8832, #9156). This plugin emits the `hook` member it owns;
  `test/cancel-cause.spec.mjs` pins the emitted object, the persisted ending, and
  the bare-string control that still reproduces `{"reason":null}`.

## Analyzing a session offline (`tools/analyze-session.mjs`)

When the guard fires — or when it *should* have fired and did not — the next
question is what the model actually did. This tool replays a **session jsonl**
through the **same `LoopDetector` the plugin runs**, so its verdict is the
installed guard's verdict, not a second opinion from a re-implementation:

```sh
node node_modules/@argszero/cordis-plugin-thinking-loop-guard/tools/analyze-session.mjs \
  ~/.dsh/sessions/<session>.jsonl
```

```text
  #   turn/step   reasonChars  textChars  chunks  verdict             fired  repeatedRun  cycleSpan  reasonCycle  novelty  break
   1  0/0                  64          0       0  reasoning-only                       0          0            0    1.000
   2  0/1                  64          3       1  repeated-material   repeated-material            1          0            0    1.000
   3  0/2                  82          2       1  progress                             1          0            0    0.667

stalled steps: 2/3  (over the 3 step(s) this tool could read — see coverage)  |  reactions: 1  |  steps that emitted text: 2
intra-call repetition: 0 call(s) would be cut mid-stream (0 by identical chunks, maxRepeatedText = 60; 0 by a repeating cycle in the visible output, ...; 0 by a repeating cycle in the reasoning, ...; 0 by exhausted material in the reasoning, ...)
```

The `repeatedRun` / `cycleSpan` / `reasonCycle` / `novelty` / `break` columns
describe the intra-call rules: [identical deltas](#single-call-repetition-maxrepeatedtext),
[cycled visible output](#cycled-repetition-maxrepeatedcyclechars), [cycled
reasoning](#reasoning-channel-repetition-maxrepeatedreasoningcyclechars) and
[exhausted reasoning](#reasoning-that-has-run-out-of-material-minreasoningnoveltyratio).
`novelty` is the share of the call's sentence units that were new material, read
at the moment the density rule would have fired and otherwise over the whole
call, so a `1.000` on a call that says the same three sentences says "every unit
was distinct" rather than "not measured" — the third call above repeats one of
its three sentences and reads `0.667`. Every
rule the plugin ships is answered here in the same release, because a rule the
plugin has and the tool does not would make the tool quietly wrong about the very
sessions it is run on. Replayed against a session shaped like discussion #2848 —
one call streaming 2825 identical text chunks — the tool reports `repeatedRun
2825 / BREAK` **while the cross-call verdict stays `progress`**, which is the
point: that failure is invisible to every per-call rule.

- Override the detector knobs to match your profile: `--similarity 0.6`,
  `--threshold 2`, `--min-chars 512`, `--max-fires 4`,
  `--max-repeated-text 60`, `--max-repeated-cycle 64`,
  `--max-repeated-reasoning-cycle 64`, `--min-reasoning-sentence-chars 8`,
  `--min-reasoning-sentences 64`, `--min-reasoning-novelty 0.05`. `--json`
  emits the raw per-step records.
- It reads **both** durable attempt formats: `assistant/chunk` (session format v1,
  dsh ≤ 0.1.2-rc.1) and `assistant/attempt` (session format v2/v3, dsh ≥ 0.1.5).
  Those two names do not overlap, so a reader can only be format-specific — this
  tool handles both. Inside a compacted attempt it reads `text-chunks`,
  `reasoning-chunks` and `tool-call-chunks` (the tool-call group keys its deltas
  `args` on current master and `texts` on 0.1.5/0.1.6; both are read).
- Unparseable lines and unknown record types are skipped rather than guessed at,
  so a future format addition degrades to "fewer steps observed", never a wrong
  verdict — **and every skip is reported**, because a zero you cannot distinguish
  from "nothing was readable" is not a verdict:

  ```text
  stalled steps: 0/161  (over the 161 step(s) this tool could read — see coverage)
  …
  coverage: 41207 line(s) — 38902 parsed (assistant/chunk 38899, assistant/attempt 3), 2305 skipped
    skipped by type: user/message 12, tool/result 40, reasoning-chunks 2200, (no type) 53
    ⚠ reasoning-chunks (2200) look like assistant stream content and are NOT in the verdict —
      it covers only the 161 step(s) above. If your file flattens stream records to top-level
      lines, the tool cannot attribute them to a (turn, step); it reports them here instead of guessing.
  ```

  `--json` carries the same numbers under `coverage`, with
  `verdictScopeIncomplete` as the machine-readable flag: **read `stalled` only
  together with it.** An empty step list is reported the same way — it means the
  two readable event types are absent, not that the session was healthy.
- Your session file never leaves your machine; the tool only reads it.

## Version history

| Version | Change |
|---|---|
| 0.3.0 | Adds the **density rule** (`minReasoningNoveltyRatio`, default 0.05; `minReasoningSentenceChars`, default 8; `minReasoningSentences`, default 64) for discussion [#9683](https://github.com/deepseek-ai/deepseek-harness/discussions/9683): a call that keeps producing sentences but has no new material left is now cut mid-call, which no shape rule can do. The reported block is 114,298 characters containing **only twenty distinct sentences**, cycled in one layout and irregular in another — the cycle rule's period ceiling is 64 and the reported period is 489, so no tail is ever periodic and it reports 0 at every span; the irregular layout has no period at all. The measure is the share of a call's **sentence units** (text between sentence terminators `.` `!` `?` `。` `！` `？` `…` or a newline, trimmed) that are distinct, counted over the whole call rather than its tail — a different statistic over the same characters, not a looser threshold. Whole units, never substrings: on #5976's own text the substring `好。` occurs 4 times while there are 3 units, because `完好。` contains it inside an ordinary word. Calibrated on the report's own shapes (0.0001 and 0.0043), 4,043 real texts (never below 0.885 once the unit floor is reached) and this plugin's six generated-document controls (never below 0.35) — `0.05` is roughly the geometric mean of its two nearest neighbours, and the unit floor of 8 is pinned from both sides (at 10 the JSON dump control fires falsely; at 8 a 7-character Chinese clause is still countable). The fifth shape of #5976 — two-character fragments interleaved among ordinary sentences — is **not** reached, because its statistic is fragment density rather than repetition share, and the README says so with the numbers. The analyzer gains the matching `reasoningNovelty` column, `--min-reasoning-novelty` and the summary line in the same release, for the same reason 0.2.0 added `reasoningCycleSpan`. `test/reasoning-novelty.spec.mjs` covers the primitive, the substring trap, both reported layouts, the six negative controls, the independent off switch, and the `apply()` cut through the shipped `llm/invariant`. |
| 0.2.0 | Adds the **reasoning-channel breaker** (`maxRepeatedReasoningCycleChars`, default 64; `minRepeatedReasoningCycleChars`, default 256) for discussion [#9524](https://github.com/deepseek-ai/deepseek-harness/discussions/9524): a call whose **thinking** collapses onto the shortest standalone sentence (`好。好。好。`, `Go. Go. Go.`) is now cut mid-call, which no shipped rule could do — the visible-output breaker never receives a `reasoning-delta`, and the per-call detector only reads reasoning after the call has ended and then needs three consecutive stalled calls. Same algorithm and same span/period thresholds as the visible cycle rule, in a sibling detector, with one deliberate difference: a **one-character period is admitted when the repeated character is content**, because the reported mechanism collapses onto the *shortest* unit and the output rule's divider exemption would have left the decisive single-character case uncatchable. The miss that symmetry buys (a purely punctuation bleed) is documented rather than papered over. The analyzer gains the matching `reasoningCycleSpan` column in the same release, because a rule the plugin ships and the tool does not answer for would make the tool quietly wrong about the sessions it is run on. `test/reasoning-breaker.spec.mjs` covers the primitive, the period-1 decision, the false-positive controls, the `apply()` cut through the shipped `llm/invariant` (with a **reasoning** block still open), and the independent off switches. |
| 0.1.12 | **Fixes a `turn/end` corruption this plugin was producing.** `escalate: 'cancel'` handed `agent.cancel()` a bare string cast to `AgentCancelCause`, a union core treats as closed (`user` / `parent` / `disposed` / `hook`). The loop reads the abort signal back with `abortedCancelCause()`, whose last branch is `default: return assertNever(cause)` — so the out-of-union cause threw **inside the turn's `catch`**, `turnEnds` stayed `null`, and the `finally` appended `{ turn, reason: null }`. The write path does not check that field, so the null reached the log; one such record makes the client's turn rendering throw and blanks the entire session list (discussions [#8832](https://github.com/deepseek-ai/deepseek-harness/discussions/8832), [#9156](https://github.com/deepseek-ai/deepseek-harness/discussions/9156)). The guard now emits the `hook` member it owns — `{ kind: 'hook', reason: <cancelCause> }` — the same shape core writes for `deepseek-account/signed-out`; the persisted ending is `{ kind: 'aborted', reason: { kind: 'hook', reason: 'thinking-loop' } }`. The config surface is unchanged (`cancelCause` is still the string, now read as the member's `reason`). `test/cancel-cause.spec.mjs` pins the emitted object, the round-trip, and the bare-string control that still reproduces `{"reason":null}`; `scripts/witness-cancel-cause.mjs` proves the boundary across published lines (`0.1.6-alpha.2` has no `abortedCancelCause`; `0.1.7-rc.2` and `0.2.0-rc.2` do, and both now end the turn with a *readable* reason). |
| 0.1.11 | **Documentation only — no code change.** The README now documents the **v4 message-source trap** that issue #2 hit, because it is ecosystem-wide rather than specific to this plugin: a shared `{ kind: 'plugin' }` entry was still in `MessageSourceMap` through the 0.1.6 line, so a plugin built against an older line compiles clean and is refused *at write time* on a v4 session; and the two delivery paths (`steer` → `user/message`, `inject` → `agent/inbox/spliced`) are refused by different validators, so a test that covers one proves nothing about the other. Shipped as its own version rather than folded into 0.1.10 because 0.1.10 was already published and npm renders the package page from the published tarball. |
| 0.1.10 | Fixes issue #2: the injected notices now carry a **producer-owned `source.kind`**. On session format v4 both injection paths — `agent.steer(...)`, which lands as `user/message`, and `agent.inject(...)`, which lands as `agent/inbox/spliced` — stamped `kind: 'plugin'`, a source kind v4 retired. The encoder refuses that *per row* and therefore refuses the whole turn (`format v4 message requires a producer-owned source kind`), so the guard whose job is to end a loop ended the turn instead, on the first notice it ever emitted. Both paths now stamp `kind: 'plugin:thinking-loop-guard'` — the same shape the v3→v4 migration rewrites historical rows into. `test/source-kind.spec.mjs` pins both paths (reverting the kind fails it). Also in this release: the peer range admits `0.1.7-alpha.1`, which had been excluded for an artifact of how the line was measured rather than anything about the line itself, and `npm run test:probe-lines` now re-measures every claimed line with **all** of that line's dsh peers pinned — pinning only the two packages this source imports let npm derive the other peers from newer builds, which is what made an earlier probe report three good lines as broken. |
| 0.1.9 | The analyzer now **discloses what it did not read**. Issue #1's follow-up: a reporter ran it on a flattened dump, got `stalled 0/161`, and had to establish by hand that the reasoning behind those turns sits in lines the tool never parses — a silent skip and an absent event printed the same zero. Every run now ends with a coverage block (lines parsed, skipped-by-type inventory, records dropped inside parsed events) plus a warning when a skipped type's name says it carries assistant stream content, `--json` carries `verdictScopeIncomplete`, and the `stalled` line states that it covers only the steps read. Also reads the `args` key that current master uses for `tool-call-chunks` (`texts` on 0.1.5/0.1.6): missing it dropped every tool-argument delta and flipped `hasOutput` on a call whose only output was a tool call. |
| 0.1.8 | Adds the **cycle rule** (`maxRepeatedCycleChars`, default 64; `minRepeatedCycleChars`, default 256) for discussion [#7043](https://github.com/deepseek-ai/deepseek-harness/discussions/7043): a call bleeding a *cycle* of short lines no longer needs 60 byte-identical deltas to be cut. Measured, the previous version never fired on that shape under any chunking (longest identical run 2). Exact verbatim periodicity was chosen over a low-entropy ratio because the ratio only separates the loop from legitimately repetitive output by ~0.06. The breaker now logs which rule fired. |
| 0.1.7 | Admits the **0.1.3-alpha.2 line and the whole 0.1.6 line**. The shipped range had gone stale: it refused the newest dsh release, so `npm install` failed with `ERESOLVE` for a plugin whose suite passes there. Every admitted line is now one the suite has been run against with all dsh peers pinned to it. `test/peer-range.spec.mjs` computes the admitted set with `semver` instead of pattern-matching the range string. |
| 0.1.6 | Adds the **mid-stream breaker** (`maxRepeatedText`, default 60) for discussion [#2848](https://github.com/deepseek-ai/deepseek-harness/discussions/2848): one call repeating the same `text-delta` is now cut *inside* the call with a terminal `REPETITIVE_OUTPUT` finish, because the per-call detectors above cannot reach a call that never ends. The analyzer reports `repeatedRun`/`break`. |
| 0.1.5 | Ships `tools/analyze-session.mjs`, an offline session analyzer that replays a session jsonl through the same detector (both durable attempt formats). Exposes `./tools/*` in `exports`. |
| 0.1.0 | First release. Listened on `agent/assistant-stream` — **broken on 0.1.2-rc.1** (that event is 0.1.5-alpha.1-only). Issue #1. |
| 0.1.1 | Switched to the `llm/stream` waterfall (present on both lines). |
| 0.1.2 | Declared `inject = ['agents']` (without it `apply()` throws `cannot get property "agents" without inject` and every session fails to run). Fixed the base-less peer range. |
| 0.1.3 | Fixed the peer range for the 0.1.5 line (`>=0.1.2-rc.1 <0.2.0` alone admits only `0.1.2-rc.1`). |
| 0.1.4 | Fixed the **detector**: added cross-call restated-material detection, stopped resetting on incidental text output, and made reactions re-fire instead of latching. Issue #1's re-test on 0.1.2-rc.1 showed the loop recurring with 0.1.3 loaded and firing. |

## Compatibility

- **Every published dsh line from 0.1.2-rc.1 through 0.2.0-rc.2** — fifteen lines —
  works: each has had the full suite run against it with **every** dsh package
  pinned to that single line (79 tests, `tsc` clean). That matrix is not a table
  kept by hand: `npm run test:probe-lines` installs each line into its own tree
  and runs it, and `test/peer-lines.mjs` is the record both it and the range test
  read.
- **Every older prerelease gets a loud `ERESOLVE`** rather than a silent runtime
  failure on a generation the guard was never exercised against. One published
  line is in that group deliberately — `0.2.1-alpha.1`, where `dsh-llm` no longer
  ships the stream-grammar invariant module our protocol-legality arm runs
  against, so the suite would be testing less there without saying so. It is
  measured, not skipped: the probe asserts that line still fails for that reason.
- The guard does **not** depend on the `agent/assistant-stream` event that
  arrived in 0.1.5-alpha.1. It wraps the older `llm/stream` waterfall, which
  every admitted line has — that is what makes the wide support possible.
- The guard needs `@deepseek-ai/dsh-agent` / `@deepseek-ai/dsh-llm`:

  ```
  >=0.1.2-rc.1 <0.1.3 || >=0.1.3-alpha.2 <0.1.4 || >=0.1.5-alpha.1 <0.2.0 || >=0.1.6-alpha.1 <0.2.0 || >=0.1.7-alpha.1 <0.2.0 || >=0.2.0-rc.1 <0.2.1
  ```

### Why the peer range looks like that

Every dsh release published today is a prerelease (`0.1.2-rc.1`,
`0.1.5-alpha.1`, `0.1.5-rc.1`, …), and **a comparator only admits prereleases that
share its own `major.minor.patch` tuple**. That produces two failure modes, and
you have to avoid both:

```jsonc
// Nothing at all: 0.1.2-rc.1 is LOWER than 0.1.2, and every other
// prerelease has a different tuple.
">=0.1.2"

// Only 0.1.2-rc.1: correct lower bound, but a 0.1.5-line user then gets
// ERESOLVE because the tuple of "^0.1.5-x" is not 0.1.2.
">=0.1.2-rc.1 <0.2.0"

// What we ship: one comparator per supported tuple line.
">=0.1.2-rc.1 <0.1.3 || >=0.1.3-alpha.2 <0.1.4 || >=0.1.5-alpha.1 <0.2.0 || >=0.1.6-alpha.1 <0.2.0 || >=0.1.7-alpha.1 <0.2.0 || >=0.2.0-rc.1 <0.2.1"
```

Several generations are in active use at once — as of this release `@deepseek-ai/dsh`
tags `latest` and `next` at `0.2.0-rc.2` and `alpha` at `0.2.1-alpha.1`, while the
individual packages carry staler `latest` tags of their own (`dsh-agent` still
says `0.1.0-rc.6`) — so one comparator per live tuple is needed. v0.1.1 shipped
the first form and could not be installed at all (`ETARGET`); v0.1.2 shipped the
second and rejected the `0.1.5` line (`ERESOLVE`); v0.1.6 shipped a union that had
simply gone stale — it refused the `0.1.3-alpha.2` line and the whole `0.1.6`
line, newest release included, for a plugin whose suite passes on both.

The range deliberately admits **only** the lines this plugin is tested on: the
fifteen listed in `test/peer-lines.mjs`, and nothing else among the published
versions. Everything older (`0.1.0-rc.*`, `0.1.1-rc.*`, `0.1.2-alpha.*`) and the
one line excluded on measurement (`0.2.1-alpha.1`) gets a loud `ERESOLVE` rather
than a silent runtime failure on a generation the guard has never been exercised
against. Add a comparator for your line only after running the suite against it —
`test/peer-range.spec.mjs` computes this set with `semver` and fails if the range
and the tested set drift apart in either direction, and
`npm run test:probe-lines` is what produces that set in the first place.

### Why `inject` is required

The plugin resolves the live Agent through `ctx.agents.get(sessionId)`. Cordis
throws `cannot get property "agents" without inject` for any service read the
fiber did not declare, so the module exports `inject = ['agents']`. This is a
**runtime** guard — TypeScript compiles `ctx.agents` fine whether or not it is
declared, which is how v0.1.1 shipped crashing on activation (#1). If you fork
this plugin and add another `ctx.<service>` read, declare it in `inject` too.

### Why the reaction messages declare their own source kind (session format v4)

Session format **v4** refuses a message whose source is the retired bare
`kind: 'plugin'`, and it refuses it at write time — the whole turn fails with
`format v4 message requires a producer-owned source kind`, and the notice that
was supposed to end the loop takes the turn with it. If you are writing a plugin
that injects or steers a message, that is the failure to design against, and it
has two traps worth knowing before you meet them:

- **Your build cannot see it.** `MessageSourceMap` is what the compiler checks
  `kind` against, and through the 0.1.6 line it still carried a shared
  `plugin: { kind: 'plugin', plugin: string }` entry. So the retired literal
  type-checked against the line you happened to build on, while the v4 writer on
  your user's line refused it. At runtime the check is a threshold rather than a
  whitelist — any non-empty kind other than `'plugin'` is admitted — but the
  compiler has no such threshold: `MessageSource` is the union of everything
  declared in `MessageSourceMap`, so an undeclared kind does not compile, and
  declaring your own entry is load-bearing rather than decoration.

  ```ts
  declare module '@deepseek-ai/dsh-llm' {
    interface MessageSourceMap {
      'plugin:your-plugin': { kind: 'plugin:your-plugin' } & ContextFormed
    }
  }
  ```

- **Both delivery paths are checked, by different validators.**
  `agent.steer(...)` lands as a `user/message` row (`assertV4MessageSources`) and
  `agent.inject(...)` lands inside `agent/inbox/spliced.inserted`
  (`assertV4SourceRowAdmission`). A test that exercises one path proves nothing
  about the other — this plugin's issue #2 was reported with both affected.

Use the kind the harness's own V3→V4 migration derives for a released plugin name
(`plugin:${plugin}`, `session-format-v3-to-v4/src/sources.ts`), so a message your
plugin writes now and one migrated out of an older log of it carry the *same*
attribution instead of drifting apart. v4 preserves unknown attribution, so a
`plugin` field can stay beside `kind` and readers that match on it keep working.

## License

MIT
