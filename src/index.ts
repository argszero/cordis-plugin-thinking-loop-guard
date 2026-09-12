/**
 * Thinking-loop guard for the dsh harness.
 *
 * Closes a guard-layer gap the in-tree `guard/` family does not cover.
 * `guard/timeout-policy` is a per-tool `tools/execute` deadline and
 * `guard/repeat-tool-reminder` is a same-tool-call chain detector — only a tool
 * *call* arms either. When a model degrades into a pure-thinking loop
 * (`deepseek-v4.1-flash-expires-on-0910` under `max`/`high` reasoning effort and
 * a long context), it emits only `reasoning-delta` chunks — zero `text-delta`,
 * zero `tool-call-delta` — so neither fires, `turn()` never sets `turnEnds`, and
 * the agent-loop `while (true)` never breaks until the user manually aborts.
 *
 * This plugin observes the public `llm/stream` waterfall (present on dsh
 * **0.1.2-rc.1 and 0.1.5-alpha.1**), tallies the `StreamChunk` composition of
 * each loop-built model call, and escalates through a configured reaction.
 *
 * Why `llm/stream` and not `agent/assistant-stream`: the latter (scoped emit
 * with `start`/`chunk`/`end` frames) only exists from dsh 0.1.5-alpha.1; the
 * widely-installed 0.1.2-rc.1 has neither it nor `AssistantStreamFrame`, so a
 * plugin pinned to that seam silently no-ops on the common release. `llm/stream`
 * is a Cordis waterfall around every streaming model call in both versions and
 * carries the same `StreamChunk` delta types, and its `GenerateOptions` carries
 * `sessionId`, from which the live Agent is reachable via `ctx.agents.get(...)`.
 *
 * ## Detection (three shapes, because one does not cover the loop)
 *
 * Issue #1's re-test on dsh 0.1.2-rc.1 showed the v0.1.2 detector did not fire on
 * the reproduction: the loop recurred with the plugin loaded and working. The
 * v0.1.2 detector had two blind spots that together miss the reported shape:
 *
 *  1. It required **consecutive** reasoning-only calls and reset the counter on
 *     *any* text output. A loop whose steps each emit some (boilerplate) text
 *     after their reasoning never accumulates, because every step looks like
 *     progress.
 *  2. Its only content signal was `repeatRatio` — verbatim n-gram repetition
 *     *within one call*. A model that re-derives the same stalled conclusion with
 *     different wording every step scores near zero on that measure, so it also
 *     slipped through.
 *
 * This version keeps the intra-call measure and adds a cross-call one: the
 * **containment** of the previous call's distinct n-grams in this call's. A step
 * is "stalled" when it either produced no output at all (shape 1) *or* it repeats
 * most of the previous step's distinct reasoning material (shape 2 — the same
 * conclusion reworded, which is what a stuck model actually does). Anything else
 * is genuine progress and resets the run.
 *
 * ## Mid-stream repetition (issue #2848, v0.1.6)
 *
 * The three shapes above are judged **after** a model call ends, which is
 * structurally too late for the failure in issue #2848: a ~10-minute, 420,000
 * character bleed where the model repeated one sentence across ~2825 text chunks
 * — all inside a **single** call. No per-call detector can help, because there is
 * no call boundary to react at; the harness must intervene mid-stream.
 *
 * `maxRepeatedText` therefore adds an intra-call breaker: count consecutive
 * identical normalized `text-delta` payloads and, on the Nth, **end the stream
 * from inside the `llm/stream` wrapper** by yielding a terminal `finish` chunk
 * and returning. Two facts make that expressible and safe, both verified on
 * 0.1.5-alpha.1 and re-checked against the 0.1.2-rc.1 typings:
 *
 *  - The agent loop consumes the waterfall's iterable (`for await (const chunk
 *    of stream) live.push(chunk)`), so a listener's own chunk reaches the same
 *    assembler as an adapter's.
 *  - `packages/llm/llm/src/invariant.ts` requires a terminal finish chunk and
 *    explicitly permits an `error`/`aborted` finish with blocks still open — so
 *    a mid-call cut is a sanctioned protocol outcome, not a violation. A stream
 *    that simply ends is an invariant failure, which is why the breaker emits a
 *    finish rather than merely returning.
 *
 * The break is always an `error` finish carrying `REPETITIVE_OUTPUT`: it fails
 * the step through the loop's normal error path (and the `agent/request-error`
 * waterfall, where a retry policy may act on the code). A quiet `stop` is not
 * offered, because the breaker fires with the call's text block still open and
 * the invariant admits only `error`/`aborted` finishes in that state — measured,
 * not assumed. `breakCorrection` steers the agent so the resumed turn is told
 * what happened. Deliberately **no model fallback and no automatic retry**: a
 * degenerate model must not be silently re-billed.
 *
 * ## Reaction
 *
 * On a threshold crossing it applies `escalate`. Unlike v0.1.2 it does **not**
 * latch after one reaction: a single steer often does not break a strong loop
 * (which is exactly what issue #1's re-test observed), so the run counter resets
 * and re-fires after another `maxThinkingSteps`, up to `maxFires` times.
 *
 * Mechanism notes (verified against packages/core/agent-loop/src/agent.ts,
 * packages/llm/llm/src/index.ts, and packages/core/agent/src/runtime-types.ts on
 * dsh 0.1.5-alpha.1; the `llm/stream` / `StreamChunk` / `ctx.agents.get` trio is
 * present unchanged on 0.1.2-rc.1):
 *  - `llm/stream` is a waterfall; a listener wraps `next()` and sees each chunk.
 *  - `chunk: StreamChunk` distinguishes `reasoning-delta` / `text-delta` /
 *    `tool-call-delta` (packages/llm/llm/src/types.ts).
 *  - Loop-built requests carry `markAgentLoopRequest`; `isAgentLoopRequest`
 *    filters out arbitrary non-agent streaming (tool streams, etc.).
 *  - `options.sessionId` → `ctx.agents.get(sessionId)` yields the live `Agent`.
 *  - `agent.steer(message)`, `agent.inject(message)`, and `agent.cancel(cause)`
 *    are public methods; a listener can react but not veto an in-flight step.
 *
 * @module @argszero/cordis-plugin-thinking-loop-guard
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent, AgentCancelCause } from '@deepseek-ai/dsh-agent'
import { createUserMessage, isAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import type { LlmFailure, StreamChunk } from '@deepseek-ai/dsh-llm'

type UserMessage = ReturnType<typeof createUserMessage>

/** The agent's public methods the guard reacts with. */
type GuardableAgent = Pick<Agent, 'inject' | 'steer' | 'cancel'>

/** Plugin configuration. */
export interface Config {
  /**
   * Consecutive stalled model calls before a reaction. A call is stalled when it
   * produced no text/tool output, or when it repeated most of the previous
   * call's distinct reasoning material. Default `3`.
   */
  maxThinkingSteps?: number
  /**
   * Minimum reasoning text within one call before that call is judged at all; a
   * short burst is normal. Default `2048` chars.
   */
  minReasoningChars?: number
  /**
   * Intra-call low-entropy ratio: coverage of one call's reasoning text by
   * repeated fixed-length grams. Language-agnostic (handles CJK with no
   * whitespace): a degenerate "好。执行。" loop is near 1.0; coherent exploration
   * is low. Only consulted once `minReasoningChars` is met. Default `0.5`.
   */
  repeatRatio?: number
  /**
   * Cross-call similarity: how much of the previous call's distinct reasoning
   * material must reappear in this call before the call counts as a repetition
   * of stalled thinking rather than progress. `1.0` requires an exact rerun of
   * the same material; `0` disables this signal. Default `0.8`.
   */
  similarityThreshold?: number
  /**
   * Reaction to fire on a threshold crossing. `warn` injects a notice, `steer`
   * sends a steering message, `cancel` hard-aborts the turn. Default `steer`.
   */
  escalate?: 'warn' | 'steer' | 'cancel'
  /**
   * How many times one agent may be reacted to before the guard stops
   * re-firing. A single intervention often does not break a strong loop, so the
   * default allows several. Default `4`.
   */
  maxFires?: number
  /** Cancel cause used when `escalate` is `cancel`. Default `'thinking-loop'`. */
  cancelCause?: string
  /**
   * Consecutive identical normalized visible-output chunks that end the stream
   * mid-call. `0` disables the breaker. This is the only guard here that fires
   * *inside* a model call, so it is the only one that can stop a single-call
   * repetition bleed (issue #2848). Default `60`.
   */
  maxRepeatedText?: number
  /**
   * Error code carried by the breaker's terminal failure. Default
   * `'REPETITIVE_OUTPUT'`.
   *
   * There is deliberately no way to choose a quiet `stop` finish instead: the
   * breaker fires while the call's text block is still open, and the `llm/stream`
   * invariant rejects a `stop` finish with open blocks (only `error`/`aborted`
   * may leave them open). Verified by running the wrapper's own output through
   * `@deepseek-ai/dsh-llm/invariant` — see `test/text-breaker.spec.mjs`.
   */
  breakCode?: string
  /**
   * Steer the agent after a mid-stream break so the resumed turn is told what
   * happened instead of silently continuing. Default `true`.
   */
  breakCorrection?: boolean
}

/** Resolved config: every field carries its validated default. */
type ResolvedConfig = Required<Config>

export const Config: z<Config> = z.object({
  maxThinkingSteps: z.number().min(2).default(3),
  minReasoningChars: z.number().min(256).default(2048),
  repeatRatio: z.number().min(0).max(1).default(0.5),
  similarityThreshold: z.number().min(0).max(1).default(0.8),
  escalate: z.union(['warn', 'steer', 'cancel']).default('steer'),
  maxFires: z.number().min(1).default(4),
  cancelCause: z.string().default('thinking-loop'),
  maxRepeatedText: z.number().step(1).min(0).default(60),
  breakCode: z.string().default('REPETITIVE_OUTPUT'),
  breakCorrection: z.boolean().default(true),
})

export const name = 'thinking-loop-guard'

/**
 * Cordis services this plugin resolves off the Context.
 *
 * Declaring `agents` is REQUIRED, not documentation: Cordis's context proxy
 * throws `cannot get property "agents" without inject` for any service read
 * that the fiber did not declare (vendor/cordis/src/reflect.ts, the
 * `waterfall('internal/get', …)` guard). The plugin reaches the live Agent via
 * `ctx.agents.get(options.sessionId)`, so without this line `apply()` throws on
 * activation and every session in that deployment fails to run.
 *
 * TypeScript cannot catch this: `ctx.agents` type-checks as soon as
 * `@deepseek-ai/dsh-agent` is in the type graph, because the guard is purely
 * runtime. That is exactly how v0.1.1 shipped broken (issue #1).
 */
export const inject = ['agents']

const PLUGIN_SOURCE = { kind: 'plugin', plugin: 'thinking-loop-guard' } as const

function message(text: string, form: 'notice'): UserMessage {
  return createUserMessage({
    content: [{ type: 'text', text }],
    source: { ...PLUGIN_SOURCE, form, summary: 'thinking-loop-guard' },
  })
}

/* -------------------------------------------------------------------------- */
/* Detection primitives (exported for tests)                                  */
/* -------------------------------------------------------------------------- */

/** Fixed window used for both the intra-call and cross-call measures. */
export const GRAM_SIZE = 4

/**
 * The distinct fixed-length grams of one reasoning text.
 *
 * Fixed-length windows are the language-agnostic choice: CJK loops
 * ("好。执行。") have no whitespace, so a whitespace-token histogram collapses to
 * one token and can never distinguish repetition.
 *
 * @param text - the reasoning text of one call.
 * @param size - the window length in characters.
 * @returns the set of distinct windows (empty for text shorter than `size`).
 */
export function grams(text: string, size = GRAM_SIZE): Set<string> {
  const out = new Set<string>()
  for (let i = 0; i + size <= text.length; i++) out.add(text.slice(i, i + size))
  return out
}

/**
 * Low-entropy ratio: coverage of one text by repeated fixed-length grams.
 *
 * A tight repetition ("好。执行。" x N) scores near 1.0 while coherent reasoning
 * (which rarely repeats a window verbatim) scores near 0.
 *
 * @param reasoning - the reasoning text of one call.
 * @returns the fraction of windows that had already appeared in the same text.
 */
export function repeatRatio(reasoning: string): number {
  if (reasoning.length < 16) return 0
  const k = Math.min(GRAM_SIZE, Math.max(2, Math.floor(reasoning.length / 16)))
  let repeated = 0
  let total = 0
  const seen = new Set<string>()
  for (let i = 0; i + k <= reasoning.length; i++) {
    const gram = reasoning.slice(i, i + k)
    total++
    if (seen.has(gram)) repeated++
    else seen.add(gram)
  }
  return total === 0 ? 0 : repeated / total
}

/**
 * Containment of one gram set in another: the fraction of the SMALLER set's
 * grams present in the larger.
 *
 * Containment rather than Jaccard on purpose. A stuck model typically restates
 * the previous step's material and appends another sentence, so the new set is a
 * near-superset of the old one; Jaccard would dilute that with the new material
 * and let the step look like progress, while containment reports the repetition
 * directly.
 *
 * @param a - one call's distinct grams.
 * @param b - another call's distinct grams.
 * @returns `0` when either set is empty, otherwise the containment in `[0, 1]`.
 */
export function containment(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 || b.size === 0) return 0
  const [small, large] = a.size <= b.size ? [a, b] : [b, a]
  let shared = 0
  for (const gram of small) if (large.has(gram)) shared++
  return shared / small.size
}

/** What one completed model call looked like to the guard. */
export interface StepObservation {
  /** Whether the call emitted any `text-delta` or `tool-call-delta`. */
  readonly hasOutput: boolean
  /** The call's total reasoning text. */
  readonly reasoning: string
}

/** Why one call was counted as stalled (`undefined` = it was progress). */
export type StallReason = 'reasoning-only' | 'repeated-material' | 'low-entropy'

/**
 * The per-agent detector: accumulates stalled calls and decides when to react.
 *
 * Kept free of Cordis types so the decision rule is unit-testable without a
 * harness: the plugin only feeds it observations and applies the reaction.
 */
export class LoopDetector {
  private stalled = 0
  private fires = 0
  private previous: Set<string> | undefined
  private lastReason: StallReason | undefined

  /**
   * @param config - the resolved plugin configuration.
   */
  constructor(private readonly config: ResolvedConfig) {}

  /** The stalled-run length that would trigger the next reaction. */
  get threshold(): number {
    return this.config.maxThinkingSteps
  }

  /** How many reactions have fired for this agent so far. */
  get fired(): number {
    return this.fires
  }

  /**
   * Observe one completed model call.
   *
   * @param step - the call's output shape and reasoning text.
   * @returns the reason it counted as stalled, or `undefined` when it was progress.
   */
  observe(step: StepObservation): StallReason | undefined {
    // A short burst is normal and carries too little material to judge. It is
    // neither counted nor treated as progress — it must not clear a run built
    // from substantive steps.
    if (step.reasoning.length < this.config.minReasoningChars) return undefined

    const current = grams(step.reasoning)
    const repeated = this.previous !== undefined
      && this.config.similarityThreshold > 0
      && containment(this.previous, current) >= this.config.similarityThreshold
    this.previous = current

    const reason: StallReason | undefined = !step.hasOutput
      ? 'reasoning-only'
      : repeated
        ? 'repeated-material'
        : repeatRatio(step.reasoning) >= this.config.repeatRatio
          ? 'low-entropy'
          : undefined

    if (reason === undefined) {
      // Real progress: the run is over.
      this.stalled = 0
      this.lastReason = undefined
      return undefined
    }

    this.stalled += 1
    this.lastReason = reason
    return reason
  }

  /**
   * Whether a reaction is due now, consuming one fire when it is.
   *
   * A single intervention often does not break a strong loop, so a fire resets
   * the run counter (another full run of stalled calls is needed) and is capped
   * by `maxFires` so the guard cannot intervene forever.
   *
   * @returns the reason to report in the reaction, or `undefined` when no reaction is due.
   */
  takeFire(): StallReason | undefined {
    if (this.stalled < this.config.maxThinkingSteps) return undefined
    if (this.fires >= this.config.maxFires) return undefined
    this.fires += 1
    this.stalled = 0
    const reason = this.lastReason ?? 'reasoning-only'
    this.lastReason = undefined
    return reason
  }

  /** Forget every observation (used when a loop is judged broken by real output). */
  reset(): void {
    this.stalled = 0
    this.previous = undefined
    this.lastReason = undefined
  }
}

/* -------------------------------------------------------------------------- */
/* Mid-stream repetition breaker (issue #2848)                                */
/* -------------------------------------------------------------------------- */

/**
 * Count how many of a list of visible-output chunks are a trailing run of
 * identical payloads (compared after trimming).
 *
 * Anchored at the tail on purpose: the breaker wants the *current* run, so
 * interleaved reasoning deltas (which are not passed here) and any earlier
 * unrelated text cannot mask it. Whitespace-only chunks normalize to `''` and
 * therefore count as repetitions of each other, which is the intended reading —
 * a stream emitting nothing but blank lines for a minute is equally stuck.
 *
 * @param texts - the call's `text-delta` payloads, in stream order.
 * @returns the length of the trailing identical run (`0` for an empty list).
 */
export function countRepeatedText(texts: readonly string[]): number {
  const last = texts.at(-1)
  if (last === undefined) return 0
  const normalized = last.trim()
  let run = 0
  for (let i = texts.length - 1; i >= 0; i--) {
    if (texts[i]!.trim() !== normalized) break
    run++
  }
  return run
}

/**
 * The intra-call breaker: watches one call's visible output and reports when the
 * model has repeated itself enough to be considered degenerate.
 *
 * Kept separate from {@link LoopDetector} because the two answer different
 * questions on different clocks — this one must decide *while the stream is
 * still open* (a call that bleeds for ten minutes never reaches the other), and
 * it is deliberately blind to reasoning text, which legitimately revisits itself.
 * Free of Cordis types so the decision rule is unit-testable without a harness.
 */
export class TextRepetitionDetector {
  private texts: string[] = []
  private broken = false

  /**
   * @param config - the resolved plugin configuration.
   */
  constructor(private readonly config: ResolvedConfig) {}

  /** Whether the breaker has already fired for this call. */
  get tripped(): boolean {
    return this.broken
  }

  /** How many characters the call had emitted when the breaker tripped. */
  private chars = 0

  /** Visible-output characters observed so far in this call. */
  get emittedChars(): number {
    return this.chars
  }

  /**
   * Observe one `text-delta` payload.
   *
   * @param text - the delta's text.
   * @returns `true` exactly once, on the delta that completes the run.
   */
  push(text: string): boolean {
    if (this.broken) return false
    this.chars += text.length
    if (this.config.maxRepeatedText <= 0) return false
    this.texts.push(text)
    if (this.texts.length < this.config.maxRepeatedText) return false
    const run = countRepeatedText(this.texts)
    if (run < this.config.maxRepeatedText) return false
    this.broken = true
    return true
  }
}

/* -------------------------------------------------------------------------- */
/* Plugin                                                                     */
/* -------------------------------------------------------------------------- */

/** Build the reaction message for one escalation step. */
function reactionMessage(escalate: ResolvedConfig['escalate'], reason: StallReason): string {
  const detail = reason === 'reasoning-only'
    ? 'several long reasoning-only calls with no output'
    : reason === 'repeated-material'
      ? 'several calls that restate the same reasoning without adding anything'
      : 'a self-repeating reasoning loop'
  switch (escalate) {
    case 'warn':
      return `The agent has produced ${detail}. If this continues it will be interrupted.`
    case 'steer':
      return `You are repeating the same reasoning without acting (${detail}). Stop deliberating and `
        + 'either call a tool or produce a concise answer now.'
    case 'cancel':
      return `Aborting: ${detail}.`
  }
}

/**
 * Install the listener. Per-`Agent` state is keyed in a `WeakMap` so a disposed
 * agent is collected; detectors are scoped to one agent lifecycle.
 */
export function apply(ctx: Context, config: ResolvedConfig): void {
  const detectors = new WeakMap<object, LoopDetector>()

  function detectorFor(agent: Agent): LoopDetector {
    let detector = detectors.get(agent)
    if (detector === undefined) {
      detector = new LoopDetector(config)
      detectors.set(agent, detector)
    }
    return detector
  }

  function react(agent: GuardableAgent, reason: StallReason): void {
    switch (config.escalate) {
      case 'warn':
        agent.inject(message(reactionMessage('warn', reason), 'notice'))
        return
      case 'steer':
        agent.steer(message(reactionMessage('steer', reason), 'notice'))
        return
      case 'cancel':
        agent.cancel(config.cancelCause as unknown as AgentCancelCause)
        return
    }
  }

  /**
   * End one in-flight model call from inside the stream wrapper.
   *
   * The synthesized `finish` is the part that must not be skipped: the agent
   * loop feeds every chunk this wrapper yields into the same assembler an
   * adapter feeds, and `packages/llm/llm/src/invariant.ts` requires a terminal
   * finish chunk while explicitly allowing `error`/`aborted` to leave blocks
   * open. Returning without one would violate that grammar.
   *
   * Order matters — steer first, then return the terminal chunk. The user
   * message must be durably enqueued *before* the `error` finish throws out of
   * the step, so the resumed turn that the loop's `turn/end` finally block
   * schedules (`if (!this.inbox.hasPending) return false`) claims it. The error
   * path reaches `turn/end` through `throwError` without passing a turn-stopping
   * idle check, so this ordering is what makes the correction survive.
   *
   * @param agent - the live agent that owns the call.
   * @param chars - visible-output characters emitted before the break.
   * @returns the terminal chunk to yield as the call's last.
   */
  function breakStream(agent: GuardableAgent, chars: number): StreamChunk {
    const detail = `the model repeated the same visible output for ${chars} characters in one call`
    ctx.logger.warn(`thinking-loop-guard: breaking a repetitive stream (${chars} chars, one call)`)
    if (config.breakCorrection) {
      agent.steer(message(
        `Your output had repeated itself for ${chars} characters without progressing, so the response was `
        + 'cut off mid-stream. Do not resume the repetition: state the conclusion once, briefly, and then '
        + 'either call a tool or finish the answer.',
        'notice',
      ))
    }
    const failure: LlmFailure = { message: `Repetitive output aborted: ${detail}.`, code: config.breakCode }
    return { type: 'finish', reason: { kind: 'error', failure } }
  }

  // Observe every streaming model call through the `llm/stream` waterfall. This
  // is present on both dsh 0.1.2-rc.1 and 0.1.5-alpha.1 and carries the SAME
  // StreamChunk delta types, so no version branching is needed.
  ctx.on('llm/stream', (options, next): AsyncIterable<StreamChunk> => {
    // Only agent-loop-built calls carry the loop identity; skip arbitrary
    // non-agent streaming (tool streams, assistant replay, etc.).
    if (!isAgentLoopRequest(options)) return next()
    if (options.sessionId === undefined) return next()
    const agent = ctx.agents.get(options.sessionId)
    if (agent === undefined) return next()
    const detector = detectorFor(agent)

    return (async function* wrapped(): AsyncIterable<StreamChunk> {
      let reasoning = ''
      let hasOutput = false
      const textBreaker = new TextRepetitionDetector(config)
      for await (const chunk of next()) {
        if (chunk.type === 'reasoning-delta') reasoning += chunk.text
        else if (chunk.type === 'text-delta' || chunk.type === 'tool-call-delta') hasOutput = true
        yield chunk
        // Visible-output repetition is the one failure the post-call judgement
        // above cannot reach (issue #2848: ~2825 repeats inside ONE call), so it
        // is checked per chunk and ends the stream while it is still open. The
        // returning branch skips the post-call observation: the call did not
        // finish, and its breaker already reacted.
        if (chunk.type === 'text-delta' && textBreaker.push(chunk.text)) {
          // `return` here ends the generator — this is not the C# `yield break`.
          yield breakStream(agent, textBreaker.emittedChars)
          return
        }
      }
      // Judge the call only once its stream has ended: a call that turns out to
      // produce output after its reasoning is progress, not a thinking loop.
      const reason = detector.observe({ hasOutput, reasoning })
      if (reason !== undefined) ctx.logger.debug(`thinking-loop-guard: stalled call (${reason})`)
      const fire = detector.takeFire()
      if (fire !== undefined) react(agent, fire)
    })()
  })
}
