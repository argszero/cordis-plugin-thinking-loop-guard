# @argszero/cordis-plugin-thinking-loop-guard

Thinking-loop guard for the DeepSeek Harness (`dsh`). Detects an agent that
degrades into a **pure-thinking loop** — a step that emits only `reasoning-delta`
chunks, with **zero `text-delta` and zero `tool-call-delta`** — and reacts to break
the loop before it burns tokens until a human manually aborts the turn.

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

This plugin is a **community-side fix** that needs no harness patch. It subscribes
to the public `agent/assistant-stream` event (scoped emit) and tallies the
`StreamChunk` composition of each attempt:

- a chunk of type `text-delta` or `tool-call-delta` **resets** the counter — this
  step produced output, so it is not a thinking loop;
- a step that is **only** `reasoning-delta`, and at least `minReasoningChars`
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
  /** Consecutive reasoning-only steps (each ≥ minReasoningChars, no output) before reacting. Default 3. */
  maxThinkingSteps?: number
  /** Minimum reasoning text in one step before it counts. Default 2048 chars. */
  minReasoningChars?: number
  /** Longest repeated token / total tokens at which the step is flagged. Default 0.5. */
  repeatRatio?: number
  /** Action on the threshold: 'warn' | 'steer' (default) | 'cancel'. */
  escalate?: 'warn' | 'steer' | 'cancel'
  /** Cancel cause when escalate is 'cancel'. Default 'thinking-loop'. */
  cancelCause?: string
}
```

## Notes

- The detector is deliberately **conservative**: a step must be long
  (`minReasoningChars`) and reasoning-only; a short thinking burst or a step with
  any real output is ignored.
- Per-`Agent` state is kept in a `WeakMap`, so a disposed agent is collected and
  its counters dropped.
- The low-entropy check is a cheap heuristic (longest repeated whitespace token
  over total tokens); it runs only once a step is already long, so cost is bound.

## License

MIT
