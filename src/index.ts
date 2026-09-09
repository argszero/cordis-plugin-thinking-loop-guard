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
 * This plugin is the community-side fix. It observes the public `llm/stream`
 * waterfall (present on dsh **0.1.2-rc.1 and 0.1.5-alpha.1**), tallies the
 * `StreamChunk` composition of each loop-built model call, and detects a call
 * that is reasoning-only AND shows a low-entropy repetition. It then escalates
 * through configured reactions: a `warn` inject, then a `steer`, then a
 * `cancel`.
 *
 * Why `llm/stream` and not `agent/assistant-stream`: the latter (scoped emit
 * with `start`/`chunk`/`end` frames) only exists from dsh 0.1.5-alpha.1; the
 * widely-installed 0.1.2-rc.1 has neither it nor `AssistantStreamFrame`, so a
 * plugin pinned to that seam silently no-ops on the common release. `llm/stream`
 * is a Cordis waterfall around every streaming model call in both versions and
 * carries the same `StreamChunk` delta types, and its `GenerateOptions` carries
 * `sessionId`, from which the live Agent is reachable via `ctx.agents.get(...)`.
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
   * Consecutive reasoning-only model calls (each at least `minReasoningChars`
   * long, no text/tool output) before a reaction. Default `3`.
   */
  maxThinkingSteps?: number
  /**
   * Minimum reasoning text within one call before it counts as a thinking step
   * at all; a short burst is normal. Default `2048` chars.
   */
  minReasoningChars?: number
  /**
   * Approximate low-entropy repeat detection: the longest repeated substring's
   * coverage of the step's reasoning text. Language-agnostic (handles CJK with
   * no whitespace): a degenerate "好。执行。" loop is near 1.0; coherent
   * exploration is low. Only consulted once `minReasoningChars` is met.
   * Default `0.5`.
   */
  repeatRatio?: number
  /**
   * Reaction to fire when the threshold is crossed. `warn` injects a notice,
   * `steer` sends a steering message, `cancel` hard-aborts the turn. Default
   * `steer`.
   */
  escalate?: 'warn' | 'steer' | 'cancel'
  /** Cancel cause used when `escalate` is `cancel`. Default `'thinking-loop'`. */
  cancelCause?: string
}

/** Resolved config: every field carries its validated default. */
type ResolvedConfig = Required<Config>

export const Config: z<Config> = z.object({
  maxThinkingSteps: z.number().min(2).default(3),
  minReasoningChars: z.number().min(256).default(2048),
  repeatRatio: z.number().min(0).max(1).default(0.5),
  escalate: z.union(['warn', 'steer', 'cancel']).default('steer'),
  cancelCause: z.string().default('thinking-loop'),
})

export const name = 'thinking-loop-guard'

const PLUGIN_SOURCE = { kind: 'plugin', plugin: 'thinking-loop-guard' } as const

function message(text: string, form: 'notice'): UserMessage {
  return createUserMessage({
    content: [{ type: 'text', text }],
    source: { ...PLUGIN_SOURCE, form, summary: 'thinking-loop-guard' },
  })
}

/**
 * Low-entropy ratio: coverage by repeated fixed-length grams.
 *
 * The degenerate loops #5976 describes are CJK ("好。执行。好。执行。"), which have
 * NO whitespace, so a whitespace-token histogram collapses to one token and can
 * never distinguish repetition. This detector is language-agnostic: it slides a
 * fixed-length window and reports the fraction of windows that have appeared
 * before, so a tight repetition ("好。执行。" x N) scores near 1.0 while coherent
 * reasoning (which rarely repeats a 4-char window verbatim) scores near 0.
 *
 * O(n) with a Set — cheap enough even for very long reasoning text (the guard
 * inspects steps well past `minReasoningChars`).
 */
function repeatRatio(reasoning: string): number {
  if (reasoning.length < 16) return 0
  const k = Math.min(4, Math.max(2, Math.floor(reasoning.length / 16)))
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

interface CallState {
  reasoning: string
  hasOutput: boolean
}

interface AgentState {
  call: CallState | null
  thinkingSteps: number
  reacted: boolean
}

/**
 * Install the listener. Per-`Agent` state is keyed in a `WeakMap` so a disposed
 * agent is collected; counters are scoped to one agent lifecycle.
 */
export function apply(ctx: Context, config: ResolvedConfig): void {
  const maxThinkingSteps = config.maxThinkingSteps
  const minReasoningChars = config.minReasoningChars
  const repeatRatioThreshold = config.repeatRatio
  const escalate = config.escalate
  const cancelCause = config.cancelCause

  const states = new WeakMap<object, AgentState>()

  function stateFor(agent: Agent): AgentState {
    let state = states.get(agent)
    if (state === undefined) {
      state = { call: null, thinkingSteps: 0, reacted: false }
      states.set(agent, state)
    }
    return state
  }

  function reset(state: AgentState): void {
    state.thinkingSteps = 0
    state.reacted = false
  }

  function react(state: AgentState, agent: Agent): void {
    if (state.reacted) return
    state.reacted = true
    switch (escalate) {
      case 'warn':
        agent.inject(message(
          'The agent has produced several long reasoning-only calls with no output. '
          + 'If this continues it will be interrupted.', 'notice'))
        return
      case 'steer':
        agent.steer(message(
          'You are repeating the same reasoning without acting. Stop deliberating and '
          + 'either call a tool or produce a concise answer now.', 'notice'))
        return
      case 'cancel':
        agent.cancel(cancelCause as unknown as AgentCancelCause)
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
    const state = stateFor(agent)

    // Reset the per-call candidate at the boundary; if the call yields any
    // text or tool-call delta it is NOT a thinking loop.
    state.call = { reasoning: '', hasOutput: false }

    return (async function* wrapped(): AsyncIterable<StreamChunk> {
      let reasoning = ''
      let hasOutput = false
      let finished = false
      try {
        for await (const chunk of next()) {
          if (chunk.type === 'reasoning-delta') {
            reasoning += chunk.text
          } else if (chunk.type === 'text-delta' || chunk.type === 'tool-call-delta') {
            hasOutput = true
          }
          yield chunk
        }
      } finally {
        if (!finished && !hasOutput) {
          const call = state.call
          if (call !== null) {
            call.reasoning = reasoning
            call.hasOutput = hasOutput
          }
          state.call = null
          finished = true
          if (hasOutput) {
            // Any text or tool output means this is NOT a thinking loop.
            reset(state)
          } else if (reasoning.length >= minReasoningChars) {
            // A long reasoning-only call: count it toward the threshold.
            state.thinkingSteps += 1
            const ratio = repeatRatio(reasoning)
            if (state.thinkingSteps >= maxThinkingSteps || ratio >= repeatRatioThreshold) {
              react(state, agent)
            }
          }
        }
      }
    })()
  })
}
