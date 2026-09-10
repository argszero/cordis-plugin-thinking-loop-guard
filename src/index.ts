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
import type { StreamChunk } from '@deepseek-ai/dsh-llm'

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
      for await (const chunk of next()) {
        if (chunk.type === 'reasoning-delta') reasoning += chunk.text
        else if (chunk.type === 'text-delta' || chunk.type === 'tool-call-delta') hasOutput = true
        yield chunk
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
