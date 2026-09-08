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
 * This plugin is the community-side fix. It listens to the public
 * `agent/assistant-stream` event, tallies the `StreamChunk` composition per
 * `(agent, turn, step)`, and detects a step that is reasoning-only AND shows a
 * low-entropy repetition. It then escalates through configured reactions:
 * a `warn` inject, then a `steer`, then a `cancel`.
 *
 * Mechanism notes (verified against packages/core/agent-loop/src/agent.ts and
 * packages/core/agent/src/runtime-types.ts on dsh 0.1.5-alpha.1):
 *  - `agent/assistant-stream` is scoped emit; `frame` is one
 *    `AssistantStreamFrame` (`start` | `chunk` | `end`).
 *  - `frame.chunk: StreamChunk` distinguishes `reasoning-delta` / `text-delta` /
 *    `tool-call-delta` (packages/llm/llm/src/types.ts).
 *  - `payload.agent.steer(message)`, `payload.agent.inject(message)`, and
 *    `payload.agent.cancel(cause)` are public methods.
 *  - A listener can react but cannot veto/rewrite an in-flight step; steer and
 *    cancel are sufficient to break the loop.
 *
 * @module @argszero/cordis-plugin-thinking-loop-guard
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent, AgentCancelCause, AssistantStreamFrame } from '@deepseek-ai/dsh-agent'
import { createUserMessage } from '@deepseek-ai/dsh-llm'

type UserMessage = ReturnType<typeof createUserMessage>

/** The agent's public methods the guard reacts with. */
type GuardableAgent = Pick<Agent, 'inject' | 'steer' | 'cancel'>

/** Plugin configuration. */
export interface Config {
  /**
   * Consecutive reasoning-only steps (each at least `minReasoningChars` long,
   * no text/tool output) before a reaction. Default `3`.
   */
  maxThinkingSteps?: number
  /**
   * Minimum reasoning text within one step before it counts as a thinking step
   * at all; a short burst is normal. Default `2048` chars.
   */
  minReasoningChars?: number
  /**
   * Approximate low-entropy repeat detection: the longest single repeated
   * whitespace-delimited token, as a fraction of total tokens in the step's
   * reasoning text. A degenerate "好。执行。" loop is near 1.0; coherent
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

/** Low-entropy ratio: longest repeated whitespace token / total tokens. */
function repeatRatio(reasoning: string): number {
  if (reasoning.length < 16) return 0
  const tokens = reasoning.split(/\s+/u).filter((t) => t.length > 0)
  if (tokens.length < 8) return 0
  const counts = new Map<string, number>()
  for (const tok of tokens) {
    const key = tok.toLowerCase()
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  let best = 1
  for (const [, count] of counts) if (count > best) best = count
  return best / tokens.length
}

interface Attempt {
  reasoning: string
  hasOutput: boolean
}

interface AgentState {
  attempt: Attempt | null
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
      state = { attempt: null, thinkingSteps: 0, reacted: false }
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
          'The agent has produced several long reasoning-only steps with no output. '
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

  ctx.on('agent/assistant-stream', (payload: { agent: Agent; frame: AssistantStreamFrame }) => {
    const { agent, frame } = payload
    const state = stateFor(agent)

    if (frame.type === 'start') {
      state.attempt = { reasoning: '', hasOutput: false }
      return
    }
    if (frame.type === 'chunk') {
      if (state.attempt === null) return
      const chunk = frame.chunk
      if (chunk.type === 'reasoning-delta') {
        state.attempt.reasoning += chunk.text
        return
      }
      if (chunk.type === 'text-delta' || chunk.type === 'tool-call-delta') {
        state.attempt.hasOutput = true
      }
      return
    }
    if (frame.type === 'end') {
      if (state.attempt === null) return
      const attempt = state.attempt
      state.attempt = null

      if (attempt.hasOutput) {
        // Any text or tool output means this is NOT a thinking loop.
        reset(state)
        return
      }
      if (attempt.reasoning.length < minReasoningChars) {
        // Too short to be a degenerate run; do not count it.
        return
      }
      // A long reasoning-only step: count it toward the threshold.
      state.thinkingSteps += 1
      const ratio = repeatRatio(attempt.reasoning)
      if (state.thinkingSteps >= maxThinkingSteps || ratio >= repeatRatioThreshold) {
        react(state, agent)
      }
    }
  })
}
