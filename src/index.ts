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
 * ## Cycled repetition (discussion #7043, v0.1.8)
 *
 * `maxRepeatedText` compares *deltas*, so it only covers the period-1 case: the
 * same payload emitted over and over. The #7043 report is a **cycle** — a few
 * short lines rotating ("好。 / 发。 / 好。 / 好。", and a mixed-language variant
 * with "Emitting." in front) for tens of lines, and the same signature in a long
 * session where the model bleeds instead of calling the tool it obviously means
 * to call. Measured against the shipped v0.1.7 detector, that shape **never
 * fires under any of four chunkings** (one delta per line, per cycle, per four
 * characters, per character) within 40 repeats: the longest run of identical
 * consecutive deltas is 3, against a threshold of 60.
 *
 * `maxRepeatedCycleChars` / `minRepeatedCycleChars` add the missing rule at the
 * character level: when the tail of the call's visible output is an exact
 * repetition of one period no longer than `maxRepeatedCycleChars` and spanning
 * at least `minRepeatedCycleChars`, the stream is cut by the same terminal
 * `error` finish. Character-level because the shape does not depend on line
 * breaks — a provider that streams the same bleed with no newline at all is
 * caught identically — and because the reported period is short (12 and 26
 * characters for the two reported shapes).
 *
 * Why exactness and not a low-entropy ratio, when `repeatRatio` was already in
 * the file: measured on the same texts, a *coverage* measure reads 0.92-0.99 for
 * the two reported shapes at a 512-character tail but 0.72 for a 40-row
 * markdown table, 0.76 for a log listing, 0.82 for generated CSS rows and 0.84
 * for a JSON dump. Separating "degenerate" from "legitimately repetitive" by
 * 0.06 is not a margin worth truncating a user's call over; verbatim
 * periodicity is 0 for every one of those samples.
 *
 * ## Reasoning-channel repetition (discussion #9524, v0.2.0)
 *
 * Both rules above watch the call's **visible output**, and the plugin's own
 * detector is documented as deliberately blind to reasoning, which revisits
 * itself legitimately. #9524 is the failure that blindness costs: under the
 * `deepseek-vision` channel with a flash-tier model thinking in Chinese, the
 * thinking block degenerates into self-replication of the shortest standalone
 * sentence — `好。好。好。`, `Go. Go. Go.` — measured at a longest repeated
 * fragment run of 7298, the reply swallowed and the session unusable; the same
 * environment with a pro-tier model does not degrade at all.
 *
 * Nothing in v0.1.x could act on it. The visible-output breaker never receives a
 * reasoning delta; the post-call judgement above reads reasoning but only once
 * the call has finished spending its whole budget, and then needs three
 * consecutive stalled calls. An inverted shape: the worst call of the session is
 * the one nothing can interrupt. `maxRepeatedReasoningCycleChars` /
 * `minRepeatedReasoningCycleChars` therefore run the same periodic-tail scan on
 * the reasoning channel, in a sibling detector, with one deliberate difference
 * (see {@link trailingReasoningCycle}): a **one-character period is admitted
 * when the repeated character is content**. Reasoning collapses onto a word, not
 * onto a divider, and the output rule's divider exemption would have left the
 * decisive case — a single character repeated thousands of times — uncatchable.
 * Every other threshold is shared, so the two channels differ in one concept.
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
 *  - `agent.cancel` takes the **closed** `AgentCancelCause` union. Anything else
 *    (a bare string, a custom `kind`) makes the loop's `abortedCancelCause`
 *    `assertNever` throw inside the turn's `catch`, which leaves `turnEnds` null
 *    and writes `turn/end` with `reason: null` — a record that blanks the whole
 *    client session list (#8832, #9156). See `cancelCauseFor`.
 *
 * @module @argszero/cordis-plugin-thinking-loop-guard
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent, AgentCancelCause } from '@deepseek-ai/dsh-agent'
import { createUserMessage, isAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import type { ContextFormed, LlmFailure, StreamChunk } from '@deepseek-ai/dsh-llm'

/**
 * This producer's own message source, declared where the harness expects it.
 *
 * `kind` is the producer identity, not a category: session format V4 refuses the
 * retired bare `'plugin'` wrapper (`format v4 message requires a producer-owned
 * source kind`, `session-format-v3-to-v4/src/message-sources.ts`), and
 * `MessageSourceMap` has carried no shared catch-all entry since the 0.1.7 line.
 * The value used here is the one the harness's own V3→V4 migration derives for a
 * released plugin name (`plugin:${plugin}`, `session-format-v3-to-v4/src/sources.ts`),
 * so a message this version writes natively and one migrated out of an older log
 * of this plugin carry the *same* attribution instead of drifting apart.
 *
 * v4 preserves unknown attribution, so `plugin` stays alongside `kind`: readers
 * that match on `source.plugin` keep working.
 */
declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    'plugin:thinking-loop-guard': { readonly kind: 'plugin:thinking-loop-guard'; readonly plugin: 'thinking-loop-guard' } & ContextFormed
  }
}

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
  /**
   * Cause **reason** used when `escalate` is `cancel`. Default `'thinking-loop'`.
   *
   * It is the `reason` of the closed union's `hook` member, not a bare cause:
   * the emitted value is `{ kind: 'hook', reason: <this string> }` and the
   * persisted turn ending is
   * `{ kind: 'aborted', reason: { kind: 'hook', reason: <this string> } }`.
   * A bare string would be outside `AgentCancelCause`, and the loop turns that
   * into `turn/end` with `reason: null`, which blanks the client's session list
   * (#8832, #9156). See {@link cancelCauseFor}.
   */
  cancelCause?: string
  /**
   * Consecutive identical normalized visible-output chunks that end the stream
   * mid-call. `0` disables the breaker. This is the only guard here that fires
   * *inside* a model call, so it is the only one that can stop a single-call
   * repetition bleed (issue #2848). Default `60`.
   *
   * Covers the period-1 case only (a byte-identical delta repeated); a model
   * bleeding a *cycle* of a few short lines resets this counter on every phase
   * change. See `maxRepeatedCycleChars`.
   */
  maxRepeatedText?: number
  /**
   * Longest repeating period, in visible-output characters, that
   * {@link trailingCycle} will recognize at the tail of a call's output.
   * `0` disables the cycle rule. Default `64`.
   *
   * Discussion #7043 reported a model that bleeds a cycle of a few short lines
   * ("好。 / 发。 / 好。 / 好。", and a mixed-language variant) for tens of
   * lines; measured on v0.1.7, that shape never fires `maxRepeatedText` under any
   * chunking, because no two consecutive deltas are identical.
   */
  maxRepeatedCycleChars?: number
  /**
   * Shortest tail span that must be an exact repetition of a single period before
   * the cycle rule ends the stream. Default `256`.
   *
   * Exact repetition rather than a low-entropy ratio on purpose: measured against
   * legitimately repetitive long output — a 40-row markdown table (0.72 at a
   * 512-char tail), generated CSS rows (0.82), a JSON dump (0.84), a log listing
   * (0.76) — a coverage threshold only separates the loop (0.92-0.99) by ~0.06,
   * too thin for a rule that truncates a call. Verbatim periodicity is 0 for all
   * of those samples and non-zero only for the reported shapes.
   */
  minRepeatedCycleChars?: number
  /**
   * Longest repeating period, in **reasoning** characters, that
   * {@link trailingReasoningCycle} will recognize at the tail of a call's
   * thinking. `0` disables the reasoning rule. Default `64`.
   *
   * Same rule and same two thresholds as the visible-output cycle above — the
   * channels differ in exactly one dimension, the **one-character period** the
   * divider guard excludes there (see {@link trailingReasoningCycle}). Keeping
   * every other number shared is deliberate: the diff between the two channels is
   * then one concept a reader can check, not a second table of tuned constants.
   */
  maxRepeatedReasoningCycleChars?: number
  /**
   * Shortest tail span that must be an exact repetition of one period before the
   * reasoning rule ends the stream. Default `256`, the same span the visible
   * cycle rule uses.
   *
   * Anchored on discussion #9524's own measurements rather than on taste. The
   * reported bleeds are `Go. ` × 238 (≈950 characters of a 4-character unit) and,
   * in the severe case, a single-character run of LRS 7298; the same reporter's
   * clean sessions measure LRS 1–2. 256 characters of an exact period therefore
   * fires inside every reported bleed and sits far above the healthy
   * measurements — the same margin the visible-output rule was given, which is
   * why the two defaults are equal.
   */
  minRepeatedReasoningCycleChars?: number
  /**
   * Shortest sentence unit, in characters, that counts toward the reasoning
   * channel's **density** measure. Default `8`.
   *
   * A unit is the text between two sentence terminators (`.` `!` `?` `。` `！`
   * `？` `…` or a newline), trimmed. The floor is what keeps generated documents
   * clean: a JSON dump repeats its own `"kind": "item",` line forty times, and
   * only the distinct `"id": N,` line on every record keeps the measure high —
   * at a floor of 10 those id lines stop counting and the same dump fires
   * falsely (measured 0.025 against the default threshold). At 8 it measures
   * 0.35. The floor is also the smallest one at which a seven-character Chinese
   * clause plus its full stop is still countable, so it is pinned from both
   * sides rather than picked.
   */
  minReasoningSentenceChars?: number
  /**
   * How many sentence units a call must produce before its novelty is judged.
   * Default `64`.
   *
   * A ratio over a handful of units says nothing: a three-sentence call cannot
   * demonstrate that it is out of material. The value is set where the measure
   * was actually calibrated — of the real texts that reach 64 units, the lowest
   * novelty is 0.885 — and it is low enough that a total collapse is cut at
   * roughly a thousand characters rather than after a whole output budget. One
   * unit is at least {@link Config.minReasoningSentenceChars} characters, so the
   * floor is also a character floor: 64 units cannot span fewer than 512.
   */
  minReasoningSentences?: number
  /**
   * Intra-call **density**: the lowest share of a call's sentence units that must
   * be new material before the call is judged degenerate. `0` disables the rule.
   * Default `0.05`.
   *
   * The complement of the shape rules rather than a replacement. A call that
   * cycles twenty sentences has a *tail* that is never periodic — the period is
   * 489 characters, past the 64 cap — and, in the irregular case, no period at
   * all; but it also has almost nothing new to say, which is what this measures.
   * Measured on those shapes: 0.0001 for one repeated sentence, 0.0043 for the
   * twenty-sentence cycle in either order. Measured on real text (60 recorded
   * reasoning blocks, 3,983 long assistant messages): never below 0.885 once a
   * call reaches the unit floor. The two sets are two orders of magnitude apart,
   * and 0.05 sits in that gap — about the geometric mean of its two nearest
   * neighbours (0.0043 and 0.885), so the exact value inside the gap is not
   * load-bearing.
   */
  minReasoningNoveltyRatio?: number
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
  maxRepeatedCycleChars: z.number().step(1).min(0).default(64),
  minRepeatedCycleChars: z.number().step(1).min(2).default(256),
  maxRepeatedReasoningCycleChars: z.number().step(1).min(0).default(64),
  minRepeatedReasoningCycleChars: z.number().step(1).min(2).default(256),
  minReasoningSentenceChars: z.number().step(1).min(2).default(8),
  minReasoningSentences: z.number().step(1).min(2).default(64),
  minReasoningNoveltyRatio: z.number().min(0).max(1).default(0.05),
  breakCode: z.string().default('REPETITIVE_OUTPUT'),
  breakCorrection: z.boolean().default(true),
})

export const name = 'thinking-loop-guard'

/**
 * The cancellation cause this guard hands to `agent.cancel()`.
 *
 * `AgentCancelCause` is a **closed** union — `user` / `parent` / `disposed` /
 * `hook` (`packages/core/session/src/types.ts`) — and `hook` is the member a
 * plugin owns. The object matters, not the string: the loop reads the cause back
 * with `abortedCancelCause(signal)` (`packages/core/agent-loop/src/agent.ts`),
 * whose last branch is `default: return assertNever(cause)`. Handing it anything
 * whose `kind` is outside the union throws **inside the turn's `catch`**, so the
 * `turnEnds = { kind: 'aborted', reason: cause }` assignment never runs, the
 * `let turnEnds = null` initializer survives, and the `finally` writes
 * `turn/end` with `reason: null`. `Session.append` does not check that field
 * (`validateSessionEventData` covers surface messages, `request/header` and
 * `tool/result` only), so the null reaches the log — and a single null reason is
 * enough to make the client's turn rendering throw and blank **every** session
 * in the list (discussions #8832, #9156). Core writes the same shape itself:
 * `{ kind: 'hook', reason: 'deepseek-account/signed-out' }`.
 *
 * The configured string is the cause's `reason`, not its `kind`, so the config
 * surface is unchanged: `cancelCause: 'x'` reads back as
 * `{ kind: 'aborted', reason: { kind: 'hook', reason: 'x' } }`.
 *
 * @param config - the resolved configuration; only `cancelCause` is read.
 * @returns a cause the loop's closed-union switch returns instead of throwing.
 */
export function cancelCauseFor(config: Pick<ResolvedConfig, 'cancelCause'>): AgentCancelCause {
  return { kind: 'hook', reason: config.cancelCause }
}

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

const PLUGIN_SOURCE = { kind: 'plugin:thinking-loop-guard', plugin: 'thinking-loop-guard' } as const

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
 * How many newly emitted visible characters must accumulate before the cycle
 * rule re-scans the tail. The scan is bounded by `minRepeatedCycleChars`, so a
 * fixed stride keeps the whole rule linear in emitted characters.
 */
const CYCLE_CHECK_STRIDE = 32

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
 * Whether one repeated character is content rather than layout.
 *
 * A letter or a digit repeated for hundreds of characters is a model stuck on
 * one token; the characters a divider is drawn with (`=`, `-`, `─`, whitespace,
 * punctuation) are not. This is the whole of the reasoning channel's deviation
 * from the visible-output rule — see {@link trailingReasoningCycle}.
 */
const CONTENT_CHAR = /[\p{L}\p{N}]/u

/**
 * Which rules the mid-stream breaker ships, as reported in the log line.
 *
 * JSDoc'd rather than left inline because the analyzer and the tests switch on
 * the same four strings.
 *
 * Three of them measure a **shape** — a periodicity in the text. The fourth
 * measures **density**: not "is the tail periodic" but "did this call say
 * anything it had not already said". The two are different statistics over the
 * same text, and a call can be degenerate under either one alone (see
 * {@link SentenceUnitCounter} for why the shape rules cannot see density).
 */
export type BreakRule = 'identical-chunks' | 'repeating-cycle' | 'repeating-reasoning' | 'reasoning-novelty'

/** The rules that watch the reasoning channel, in one place for `breakStream`. */
const REASONING_RULES: readonly BreakRule[] = ['repeating-reasoning', 'reasoning-novelty']

/**
 * Sentence terminators. What separates one unit from the next, kept with the
 * unit it ends.
 *
 * A newline is included deliberately: in code-heavy thinking the line, not the
 * full stop, is the sentence, and a rule that only split on `.!?` would read a
 * pasted listing as one enormous sentence.
 */
const SENTENCE_TERMINATOR = /[.!?。！？…\n]/

/**
 * The tail span that is an exact repetition of one period `admits` accepts, or
 * `0` when the tail is not periodic.
 *
 * The scan stops as soon as the minimum is met, so the returned span is
 * `max(minSpan, 2 * period)` rather than the longest periodic tail that exists:
 * both callers only distinguish zero from non-zero, and stopping early keeps the
 * rule bounded on a call that bleeds for hundreds of thousands of characters.
 * Cost per check is `O(maxPeriod * minSpan)`.
 *
 * @param text - the accumulated text of one call's channel.
 * @param maxPeriod - the longest period to consider, in characters.
 * @param minSpan - the shortest qualifying tail span, in characters.
 * @param admits - whether a candidate period (as its template text) may qualify.
 * @returns the qualifying span in characters (`>= minSpan`), or `0`.
 */
function periodicTail(
  text: string,
  maxPeriod: number,
  minSpan: number,
  admits: (template: string) => boolean,
): number {
  if (maxPeriod < 1) return 0
  for (let period = 1; period <= maxPeriod; period++) {
    const template = text.slice(text.length - period)
    if (!admits(template)) continue
    const need = Math.max(minSpan, 2 * period) - period
    const limit = Math.min(text.length - period, need)
    let matched = 0
    while (matched < limit && text[text.length - period - 1 - matched] === text[text.length - 1 - matched]) matched++
    if (matched >= need) return matched + period
  }
  return 0
}

/**
 * The tail span that is an exact repetition of one short period, or `0` when the
 * tail is not periodic (discussion #7043).
 *
 * This generalizes {@link countRepeatedText} from "the same delta twice" to
 * "the same *cycled* output for hundreds of characters", and it is
 * newline-agnostic: the period is measured in characters, so a bleed that never
 * emits a line break is caught identically to a one-line-per-delta bleed.
 *
 * Two guards keep it off legitimate output:
 *
 *  - **Exactness.** The whole span must be a verbatim repetition; a near-repeat
 *    scores `0`. Measured: a 40-row markdown table, generated CSS rows, a JSON
 *    dump and a 60-line log listing are all period-free while scoring
 *    0.72-0.84 on a *coverage* measure of the same text — a ratio rule would
 *    have had to separate the loop from those by ~0.06.
 *  - **A period of at least two distinct characters.** A run of `=` or `-` is a
 *    legitimate divider, and a one-character period would flag every long rule
 *    line in a generated document.
 *
 * @param text - the accumulated visible output of one call.
 * @param maxPeriod - the longest period to consider, in characters.
 * @param minSpan - the shortest qualifying tail span, in characters.
 * @returns the qualifying span in characters (`>= minSpan`), or `0`.
 */
export function trailingCycle(text: string, maxPeriod: number, minSpan: number): number {
  return periodicTail(text, maxPeriod, minSpan, template => new Set(template).size >= 2)
}

/**
 * The tail span that is an exact repetition of one short period **in the
 * reasoning channel**, or `0` (discussion #9524).
 *
 * Same algorithm and same two thresholds as {@link trailingCycle}; one gate
 * differs, and that gate is the whole reason this is a second function.
 *
 * The output rule refuses a one-character period, because a run of `=` or `─`
 * is a legitimate rule line in a generated document. That exemption is exactly
 * wrong for reasoning: #9524's mechanism is the model collapsing onto the
 * **shortest standalone unit** and emitting nothing else, so the reported period
 * is 1–2 characters — and the decisive case is a *single* character repeated
 * thousands of times. Excluding period 1 by inheritance would leave the worst
 * measured failure uncatchable while the rule appeared to cover the channel.
 *
 * So period 1 is admitted here **when the repeated character is content** — a
 * letter or a digit ({@link CONTENT_CHAR}). A stuck model repeats a unit that
 * means something; 256 characters of punctuation or whitespace is layout, and
 * this rule does not get to truncate a call over a divider. The known miss is
 * the mirror image and is stated rather than papered over: a bleed of a purely
 * punctuation character (a one-character period of `。`) is not caught. The
 * reported mechanism requires the unit to be a *sentence*, so it always contains
 * a word character — and any period of two or more characters, `好。` included,
 * is caught by the shared rule with no exemption at all.
 *
 * @param text - the call's accumulated reasoning text.
 * @param maxPeriod - the longest period to consider, in characters.
 * @param minSpan - the shortest qualifying tail span, in characters.
 * @returns the qualifying span in characters (`>= minSpan`), or `0`.
 */
export function trailingReasoningCycle(text: string, maxPeriod: number, minSpan: number): number {
  return periodicTail(text, maxPeriod, minSpan, template => template.length === 1
    ? CONTENT_CHAR.test(template)
    : new Set(template).size >= 2)
}

/**
 * Intra-call **density**: the share of one call's sentence units that the model
 * had not already produced (discussions #5976, #9683).
 *
 * Every other rule in this file measures a **shape** — a periodicity in the
 * text. Density is a different statistic over the same characters, and the
 * reports that motivate this class are exactly the shapes a shape rule cannot
 * see: #9683's thinking collapsed onto twenty sentences cycled in a fixed order
 * (a period of 489 characters, far past {@link trailingCycle}'s cap of 64, so no
 * tail is ever periodic) and onto the same twenty in an **irregular** order (no
 * period exists at all); #5976's is the same hole from the other side, short
 * fragments interleaved among ordinary sentences so the tail never settles.
 *
 * The measure is `distinct / total` over whole sentence units — never a
 * substring count. The distinction is load-bearing and was measured: in #5976's
 * own text the substring `好。` occurs 4 times while the text contains 3 units,
 * because `完好。` ("intact") contains `好。` inside an ordinary word. A substring
 * rule reports repetitions no reader can see; whole units cannot.
 *
 * Calibrated on three sets, all measured (`tmp/c647/measure-5..7.mjs`):
 *
 *  - **Reported shapes** (#9683's four, plus a Chinese twin built from the same
 *    vocabulary): `0.0001` for a single repeated sentence, `0.0043` for the
 *    twenty-sentence cycle in either order. All fire.
 *  - **Real text**: 60 recorded reasoning blocks and 3,983 long assistant
 *    messages from this project's own session history. Of the 96 texts that
 *    reach {@link Config.minReasoningSentences} units at the default unit floor,
 *    the lowest novelty is **0.885** — 17x above the default threshold.
 *  - **Generated-document negatives** (the plugin's own list: a markdown table,
 *    generated CSS, a JSON dump, a log listing, repeated code stubs, a bullet
 *    list): all stay at or above **0.35**.
 *
 * The unit floor exists for the last set. A JSON dump repeats its own
 * `"kind": "item",` line 40 times; what keeps that from reading as a loop is
 * that every record contributes a distinct id, and the floor is what stops the
 * short structural lines from being counted at all. At 8 characters the dump
 * measures 0.35; at 10 the floor swallows the ids' shorter neighbours and the
 * same dump measures 0.025, i.e. it FALSELY fires. 8 is also the floor at which
 * a 7-character Chinese clause plus its `。` is still a countable unit, so it is
 * pinned from both sides rather than chosen.
 */
export class SentenceUnitCounter {
  private readonly seen = new Set<string>()
  private pending = ''
  private units = 0

  /**
   * @param minChars - shortest trimmed unit that counts toward the tally. Units
   * shorter than this are dropped, not merged into the next one.
   */
  constructor(private readonly minChars: number) {}

  /** How many sentence units the text has produced. */
  get total(): number {
    return this.units
  }

  /** How many of those units are distinct. */
  get distinct(): number {
    return this.seen.size
  }

  /**
   * Share of the call's units that are new material: `distinct / total`, or `1`
   * while no unit has finished (an empty text is not evidence of a loop).
   */
  get novelty(): number {
    return this.units === 0 ? 1 : this.seen.size / this.units
  }

  /**
   * Observe one delta of the channel.
   *
   * Incremental by construction: only the fragment after the last terminator is
   * carried between calls, so a delta may split a sentence anywhere — including
   * mid-character-run — without changing the tally.
   *
   * @param text - the delta's text.
   * @returns the novelty after this delta (see {@link novelty}).
   */
  push(text: string): number {
    this.pending += text
    let start = 0
    for (let i = 0; i < this.pending.length; i++) {
      if (!SENTENCE_TERMINATOR.test(this.pending[i]!)) continue
      const unit = this.pending.slice(start, i + 1).trim()
      start = i + 1
      if (unit.length < this.minChars) continue
      this.units++
      this.seen.add(unit)
    }
    this.pending = this.pending.slice(start)
    return this.novelty
  }
}

/**
 * The intra-call breaker: watches one call's visible output and reports when the
 * model has repeated itself enough to be considered degenerate.
 *
 * Kept separate from {@link LoopDetector} because the two answer different
 * questions on different clocks — this one must decide *while the stream is
 * still open* (a call that bleeds for ten minutes never reaches the other) — and
 * it watches **visible output only**, which is the channel its two rules were
 * calibrated on. Reasoning gets its own sibling, {@link ReasoningRepetitionDetector},
 * since the exemptions that keep this class off legitimate output are wrong for
 * thinking (see {@link trailingReasoningCycle}). Free of Cordis types so the
 * decision rule is unit-testable without a harness.
 */
export class TextRepetitionDetector {
  private texts: string[] = []
  private broken = false
  private visible = ''
  private lastCycleCheck = 0
  private reason: BreakRule | undefined
  /**
   * @param config - the resolved plugin configuration.
   */
  constructor(private readonly config: ResolvedConfig) {}

  /** Whether the breaker has already fired for this call. */
  get tripped(): boolean {
    return this.broken
  }

  /** Which rule fired, for the log line and for tests. */
  get trippedBy(): BreakRule | undefined {
    return this.reason
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
    if (this.config.maxRepeatedText <= 0 && this.config.maxRepeatedCycleChars <= 0) return false
    this.texts.push(text)
    if (this.config.maxRepeatedText > 0
      && this.texts.length >= this.config.maxRepeatedText
      && countRepeatedText(this.texts) >= this.config.maxRepeatedText) {
      this.broken = true
      this.reason = 'identical-chunks'
      return true
    }
    if (this.config.maxRepeatedCycleChars <= 0) return false
    this.visible += text
    // The scan grows with the qualifying span, so run it on a fixed stride
    // rather than per delta: a bleed is hundreds of characters long and 32
    // characters of latency cost nothing next to the cost of not firing.
    if (this.visible.length - this.lastCycleCheck < CYCLE_CHECK_STRIDE) return false
    this.lastCycleCheck = this.visible.length
    const span = trailingCycle(this.visible, this.config.maxRepeatedCycleChars, this.config.minRepeatedCycleChars)
    if (span === 0) return false
    this.broken = true
    this.reason = 'repeating-cycle'
    return true
  }
}

/**
 * The intra-call breaker for the **reasoning** channel: watches one call's
 * thinking and reports when the model has collapsed onto a short repeated unit.
 *
 * A sibling of {@link TextRepetitionDetector} rather than a mode of it, because
 * the two watch different text under different exemptions (see
 * {@link trailingReasoningCycle}) even though they share one periodic-tail scan.
 * Free of Cordis types so the decision rule is unit-testable without a harness.
 *
 * Discussion #9524 is why it exists. Under the `deepseek-vision` channel with a
 * flash-tier model thinking in Chinese, a reasoning block degenerates into
 * self-replication of the shortest standalone sentence — `好。好。好。`, `Go. Go.
 * Go.` — measured at 57–62% of block-tail fragments being ≤2 characters and a
 * longest repeated fragment run of 7298, with the reply swallowed and the
 * session unusable. Every rule that already existed is structurally unable to
 * act on it: the visible-output breaker never receives a reasoning delta, and
 * this plugin's own post-call judgement reads reasoning only *after* the call has
 * spent its whole output budget, and then needs three consecutive stalled calls.
 * The worst call of a session was the one nothing could interrupt.
 *
 * It carries two rules, and they are different statistics over the same text:
 * the **shape** rule ({@link trailingReasoningCycle}) ends a call whose tail has
 * collapsed onto a short cycle, and the **density** rule
 * ({@link SentenceUnitCounter}) ends one that has run out of things to say even
 * though no period exists. Discussion #9683's thinking does the second without
 * the first — twenty sentences cycled at a period of 489 characters, and the same
 * twenty in an irregular order — and #5976's does it with short fragments
 * interleaved among ordinary sentences. Only one of the two can fire per call;
 * the shape rule is checked first because it is the more specific finding and,
 * on the shapes both reach, fires far earlier.
 */
export class ReasoningRepetitionDetector {
  private broken = false
  private text = ''
  private lastCheck = 0
  private chars = 0
  private reason: BreakRule | undefined
  private readonly units: SentenceUnitCounter
  /**
   * The density arm's three knobs, normalized once.
   *
   * `ResolvedConfig` makes them required, so a TypeScript caller cannot omit
   * them; the `??` is for the untyped callers that do exist — the tests and the
   * offline analyzer both build a plain object. It resolves toward **off**, so an
   * omitted key can never make a guard fire.
   */
  private readonly noveltyRatio: number
  private readonly minSentences: number

  /**
   * @param config - the resolved plugin configuration.
   */
  constructor(private readonly config: ResolvedConfig) {
    this.noveltyRatio = config.minReasoningNoveltyRatio ?? 0
    this.minSentences = config.minReasoningSentences ?? 2
    this.units = new SentenceUnitCounter(config.minReasoningSentenceChars ?? 8)
  }

  /** Whether the breaker has already fired for this call. */
  get tripped(): boolean {
    return this.broken
  }

  /** Which rule fired, for the log line and for tests. */
  get trippedBy(): BreakRule | undefined {
    return this.reason
  }

  /** Reasoning characters observed so far in this call. */
  get emittedChars(): number {
    return this.chars
  }

  /**
   * Novelty of the reasoning observed so far, or `1` while the density rule is
   * off or has too few units to judge. Exposed for the tests and the analyzer,
   * which report the number the decision was made from.
   */
  get novelty(): number {
    return this.units.novelty
  }

  /**
   * Observe one `reasoning-delta` payload.
   *
   * @param text - the delta's text.
   * @returns `true` exactly once, on the delta that completes the span.
   */
  push(text: string): boolean {
    if (this.broken) return false
    this.chars += text.length
    if (this.config.maxRepeatedReasoningCycleChars <= 0 && this.noveltyRatio <= 0) return false
    this.text += text
    // The density tally is updated on EVERY delta, not on the stride: dropping a
    // delta would drop the units inside it. Only the judgement is strided.
    if (this.noveltyRatio > 0) this.units.push(text)
    // Same stride reasoning as the visible rule: the scan is bounded by the
    // qualifying span, so a fixed stride keeps the whole rule linear in emitted
    // characters even on a bleed that runs for hundreds of thousands of them.
    if (this.text.length - this.lastCheck < CYCLE_CHECK_STRIDE) return false
    this.lastCheck = this.text.length
    // Shape first: it is the more specific finding, and on the shapes both rules
    // reach (one sentence repeated forever) it fires far earlier, so which rule a
    // call is attributed to stays stable.
    if (this.config.maxRepeatedReasoningCycleChars > 0) {
      const span = trailingReasoningCycle(
        this.text,
        this.config.maxRepeatedReasoningCycleChars,
        this.config.minRepeatedReasoningCycleChars,
      )
      if (span > 0) {
        this.broken = true
        this.reason = 'repeating-reasoning'
        return true
      }
    }
    if (this.noveltyRatio <= 0) return false
    if (this.units.total < this.minSentences) return false
    if (this.units.novelty > this.noveltyRatio) return false
    this.broken = true
    this.reason = 'reasoning-novelty'
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
        agent.cancel(cancelCauseFor(config))
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
   * The same holds one layer up for the **reasoning** rule: `interruptedBlocks()`
   * (`packages/llm/llm/src/assembler.ts`) enumerates "closed and open text/reasoning
   * blocks with non-whitespace content", so an `error` finish with a reasoning
   * block still open is the sanctioned outcome here too, and the block that was
   * emitted is kept as the attempt's prefix. The invariant test in
   * `test/reasoning-breaker.spec.mjs` runs the reasoning cut through the shipped
   * validator rather than trusting that reading.
   *
   * Order matters — steer first, then return the terminal chunk. The user
   * message must be durably enqueued *before* the `error` finish throws out of
   * the step, so the resumed turn that the loop's `turn/end` finally block
   * schedules (`if (!this.inbox.hasPending) return false`) claims it. The error
   * path reaches `turn/end` through `throwError` without passing a turn-stopping
   * idle check, so this ordering is what makes the correction survive.
   *
   * @param agent - the live agent that owns the call.
   * @param chars - characters of the offending channel emitted before the break.
   * @param rule - which of the three rules fired.
   * @returns the terminal chunk to yield as the call's last.
   */
  function breakStream(agent: GuardableAgent, chars: number, rule: BreakRule): StreamChunk {
    const channel = REASONING_RULES.includes(rule) ? 'reasoning' : 'visible output'
    const detail = `the model repeated the same ${channel} for ${chars} characters in one call`
    ctx.logger.warn(`thinking-loop-guard: breaking a repetitive stream (${chars} chars, one call, rule: ${rule})`)
    if (config.breakCorrection) {
      agent.steer(message(
        `Your ${channel} had repeated itself for ${chars} characters without progressing, so the response was `
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
      const reasoningBreaker = new ReasoningRepetitionDetector(config)
      for await (const chunk of next()) {
        if (chunk.type === 'reasoning-delta') reasoning += chunk.text
        else if (chunk.type === 'text-delta' || chunk.type === 'tool-call-delta') hasOutput = true
        yield chunk
        // Repetition inside ONE call is the failure the post-call judgement above
        // cannot reach (issue #2848: ~2825 repeats in a call that never ended;
        // discussion #9524: a reasoning block collapsed to one short unit), so it
        // is checked per chunk and ends the stream while it is still open. The
        // returning branch skips the post-call observation: the call did not
        // finish, and its breaker already reacted.
        if (chunk.type === 'text-delta' && textBreaker.push(chunk.text)) {
          // `return` here ends the generator — this is not the C# `yield break`.
          yield breakStream(agent, textBreaker.emittedChars, textBreaker.trippedBy ?? 'identical-chunks')
          return
        }
        if (chunk.type === 'reasoning-delta' && reasoningBreaker.push(chunk.text)) {
          yield breakStream(agent, reasoningBreaker.emittedChars, reasoningBreaker.trippedBy ?? 'repeating-reasoning')
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
