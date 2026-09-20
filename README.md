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
  /** Cancel cause when escalate is 'cancel'. Default 'thinking-loop'. */
  cancelCause?: string
  /** Consecutive identical visible-output chunks that end the stream mid-call. 0 disables. Default 60. */
  maxRepeatedText?: number
  /** Longest repeating period of the visible output, in characters, that ends the stream mid-call. 0 disables. Default 64. */
  maxRepeatedCycleChars?: number
  /** Shortest tail span that must be an exact repetition of one period before the cycle rule fires. Default 256. */
  minRepeatedCycleChars?: number
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
  #   turn/step   reasonChars  textChars  chunks  verdict             fired  repeatedRun  break
   1  0/0                  33          0       1  reasoning-only
   2  0/1                  33          3       1  repeated-material   repeated-material   1

stalled steps: 2/2  |  reactions: 1  |  steps that emitted text: 1
intra-call repetition: 0 call(s) would be cut mid-stream (maxRepeatedText = 60)
```

The `repeatedRun` / `break` columns describe the [single-call
case](#single-call-repetition-maxrepeatedtext). Replayed against a session shaped
like discussion #2848 — one call streaming 2825 identical text chunks — the tool
reports `repeatedRun 2825 / BREAK` **while the cross-call verdict stays
`progress`**, which is the point: that failure is invisible to every per-call
rule.

- Override the detector knobs to match your profile: `--similarity 0.6`,
  `--threshold 2`, `--min-chars 512`, `--max-fires 4`,
  `--max-repeated-text 60`. `--json` emits the raw per-step records.
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

- **Every published dsh line from 0.1.2-rc.1 through 0.1.6-alpha.2**: works. All
  eight lines below have had the full suite run against them with every dsh peer
  pinned to that single line (52 tests when that matrix was last run, at 0.1.7),
  and `tsc` is clean on all eight.
- The guard does **not** depend on the `agent/assistant-stream` event that
  arrived in 0.1.5-alpha.1. It wraps the older `llm/stream` waterfall, which
  every admitted line has — that is what makes the wide support possible.
- The guard needs `@deepseek-ai/dsh-agent` / `@deepseek-ai/dsh-llm`:

  ```
  >=0.1.2-rc.1 <0.1.3 || >=0.1.3-alpha.2 <0.1.4 || >=0.1.5-alpha.1 <0.2.0 || >=0.1.6-alpha.1 <0.2.0
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
">=0.1.2-rc.1 <0.1.3 || >=0.1.3-alpha.2 <0.1.4 || >=0.1.5-alpha.1 <0.2.0 || >=0.1.6-alpha.1 <0.2.0"
```

The npm `latest` tag for `@deepseek-ai/dsh` is `0.1.2-rc.1`, while the `next` and
`alpha` tags move through the `0.1.5` and `0.1.6` lines — several generations are
in active use at once, so one comparator per live tuple is needed. v0.1.1 shipped
the first form and could not be installed at all (`ETARGET`); v0.1.2 shipped the
second and rejected the `0.1.5` line (`ERESOLVE`); v0.1.6 shipped a union that had
simply gone stale — it refused the `0.1.3-alpha.2` line and the whole `0.1.6`
line, newest release included, for a plugin whose suite passes on both.

The range deliberately admits **only** the lines this plugin is tested on. Its
admitted set over every published dsh version is exactly `0.1.2-rc.1`,
`0.1.3-alpha.2`, `0.1.5-alpha.1`, `0.1.5-alpha.2`, `0.1.5-rc.1`, `0.1.5-rc.2`,
`0.1.6-alpha.1`, `0.1.6-alpha.2`; older prereleases (`0.1.0-rc.*`, `0.1.1-rc.*`,
`0.1.2-alpha.*`) get a loud `ERESOLVE` rather than a silent runtime failure on a
generation the guard has never been exercised against. Add a comparator for your
line only after running the suite against it — `test/peer-range.spec.mjs`
computes this set with `semver` and fails if the range and the tested set drift
apart in either direction.

### Why `inject` is required

The plugin resolves the live Agent through `ctx.agents.get(sessionId)`. Cordis
throws `cannot get property "agents" without inject` for any service read the
fiber did not declare, so the module exports `inject = ['agents']`. This is a
**runtime** guard — TypeScript compiles `ctx.agents` fine whether or not it is
declared, which is how v0.1.1 shipped crashing on activation (#1). If you fork
this plugin and add another `ctx.<service>` read, declare it in `inject` too.

## License

MIT
