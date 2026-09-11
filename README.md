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
  #   turn/step   reasonChars  textChars  verdict             fired
   1  0/0                  33          0  reasoning-only
   2  0/1                  33          3  repeated-material   repeated-material

stalled steps: 2/2  |  reactions: 1  |  steps that emitted text: 1
```

- Override the detector knobs to match your profile: `--similarity 0.6`,
  `--threshold 2`, `--min-chars 512`, `--max-fires 4`. `--json` emits the raw
  per-step records.
- It reads **both** durable attempt formats: `assistant/chunk` (session format v1,
  dsh ≤ 0.1.2-rc.1) and `assistant/attempt` (session format v2, dsh ≥ 0.1.5).
  Those two names do not overlap, so a reader can only be format-specific — this
  tool handles both.
- Unparseable lines and unknown record types are skipped rather than guessed at,
  so a future format addition degrades to "fewer steps observed", never a wrong
  verdict.
- Your session file never leaves your machine; the tool only reads it.

## Version history

| Version | Change |
|---|---|
| 0.1.5 | Ships `tools/analyze-session.mjs`, an offline session analyzer that replays a session jsonl through the same detector (both durable attempt formats). Exposes `./tools/*` in `exports`. |
| 0.1.0 | First release. Listened on `agent/assistant-stream` — **broken on 0.1.2-rc.1** (that event is 0.1.5-alpha.1-only). Issue #1. |
| 0.1.1 | Switched to the `llm/stream` waterfall (present on both lines). |
| 0.1.2 | Declared `inject = ['agents']` (without it `apply()` throws `cannot get property "agents" without inject` and every session fails to run). Fixed the base-less peer range. |
| 0.1.3 | Fixed the peer range for the 0.1.5 line (`>=0.1.2-rc.1 <0.2.0` alone admits only `0.1.2-rc.1`). |
| 0.1.4 | Fixed the **detector**: added cross-call restated-material detection, stopped resetting on incidental text output, and made reactions re-fire instead of latching. Issue #1's re-test on 0.1.2-rc.1 showed the loop recurring with 0.1.3 loaded and firing. |

## Compatibility

- **dsh 0.1.2-rc.1** (the widely-installed npm release): works — `llm/stream`,
  `StreamChunk`, `GenerateOptions.sessionId`, `ctx.agents.get`, and
  `agent.steer/cancel/inject` are all present.
- **dsh 0.1.5-alpha.1**: works — same `llm/stream` seam. The newer
  `agent/assistant-stream` event is NOT required; the guard does not depend on it.
- The guard needs `@deepseek-ai/dsh-agent` / `@deepseek-ai/dsh-llm`:

  ```
  >=0.1.2-rc.1 <0.2.0 || >=0.1.5-alpha.1 <0.2.0
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
">=0.1.2-rc.1 <0.2.0 || >=0.1.5-alpha.1 <0.2.0"
```

The npm `latest` tag for `@deepseek-ai/dsh` is `0.1.2-rc.1`, while the `next` and
`alpha` tags point at the `0.1.5` line — both are in active use, so both
comparators are needed. v0.1.1 shipped the first form and could not be installed
at all (`ETARGET`); v0.1.2 shipped the second and rejected the `0.1.5` line
(`ERESOLVE`).

### Why `inject` is required

The plugin resolves the live Agent through `ctx.agents.get(sessionId)`. Cordis
throws `cannot get property "agents" without inject` for any service read the
fiber did not declare, so the module exports `inject = ['agents']`. This is a
**runtime** guard — TypeScript compiles `ctx.agents` fine whether or not it is
declared, which is how v0.1.1 shipped crashing on activation (#1). If you fork
this plugin and add another `ctx.<service>` read, declare it in `inject` too.

## License

MIT
