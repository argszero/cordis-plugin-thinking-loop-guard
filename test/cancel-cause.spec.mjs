/**
 * Cancel-cause regression: the guard must never manufacture a null `turn/end`
 * reason.
 *
 * Discussions #8832 (jianger114) and #9156 (pedestrian757) report the same
 * failure from two directions: one `turn/end` whose `reason` is `null` makes the
 * client's turn rendering throw, and the slot error boundary then blanks the
 * **whole conversation pane — every session, including brand-new ones**. The
 * record is durably on disk and survives restarts. The reporter of #9156 traced
 * one producer to a plugin calling `agent.cancel()` with a `kind` outside the
 * closed `AgentCancelCause` union, and both producers turned out to be **this
 * project's own packages** (`thinking-loop-guard`, `turn-budget-guard`), which
 * passed a bare string.
 *
 * The chain, quoted from dsh `origin/master`:
 *
 *   // packages/core/session/src/types.ts — the closed union
 *   export type AgentCancelCause =
 *     | { readonly kind: 'user' } | { readonly kind: 'parent' }
 *     | { readonly kind: 'hook'; readonly reason: string }
 *     | { readonly kind: 'disposed' }
 *
 *   // packages/core/agent-loop/src/agent.ts — the reader, whose last branch throws
 *   function abortedCancelCause(signal: AbortSignal): AgentCancelCause | undefined {
 *     if (!signal.aborted) return undefined
 *     const cause = signal.reason as AgentCancelCause
 *     switch (cause.kind) {
 *       case 'user': case 'parent': case 'disposed': return { kind: cause.kind }
 *       case 'hook': return { kind: 'hook', reason: cause.reason }
 *       /* v8 ignore next -- cancel accepts the closed AgentCancelCause union *\/
 *       default: return assertNever(cause)
 *     }
 *   }
 *
 *   // same file, the turn loop: the throw happens INSIDE the catch, so the
 *   // assignment never runs and `let turnEnds = null` survives into the finally
 *   } catch (error: unknown) {
 *     const cause = abortedCancelCause(signal)
 *     if (cause !== undefined) { turnEnds = { kind: 'aborted', reason: cause }; throw error }
 *     ...
 *   } finally {
 *     this.session.append('turn/end', { turn, reason: turnEnds! })
 *   }
 *
 * `Session.append` does not check the field (`validateSessionEventData` inspects
 * surface messages, `request/header` and `tool/result` only), so the null is
 * admitted and written.
 *
 * What this file cannot do is call the real `abortedCancelCause`: its package
 * (`@deepseek-ai/dsh-agent-loop`) is not a dependency of this plugin. The two
 * functions below therefore mirror the quoted code — including its control flow,
 * so the control arm can reproduce the reported `{"reason":null}` — and the
 * real-artifact replay is `scripts/witness-cancel-cause.mjs`, which extracts the
 * shipped function out of the *published* `lib/index.js` and runs it. This is the
 * same split `source-kind.spec.mjs` uses, and for the same reason: what is
 * pinned here is the part that can regress, namely the cause this plugin emits.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { markAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import { cancelCauseFor } from '../lib/index.js'
import * as plugin from '../lib/index.js'

const CONFIG = {
  maxThinkingSteps: 3,
  minReasoningChars: 64,
  repeatRatio: 0.5,
  similarityThreshold: 0.8,
  escalate: 'cancel',
  maxFires: 4,
  cancelCause: 'thinking-loop',
  maxRepeatedText: 0,
  breakCode: 'REPETITIVE_OUTPUT',
  breakCorrection: true,
  maxRepeatedCycleChars: 0,
  minRepeatedCycleChars: 512,
}

/** Upstream `assertNever`'s effect, verbatim: a runtime value outside the type always throws. */
function assertNever(value) {
  throw new Error(`unreachable variant: ${JSON.stringify(value)}`)
}

/**
 * The upstream reader, mirrored: `abortedCancelCause(signal)` plus the turn
 * loop's `catch`/`finally` that turn its result into the persisted record.
 *
 * @param cause - what `agent.cancel()` handed the signal.
 * @returns the `turn/end` payload the loop would append.
 */
function persistedTurnEnd(cause) {
  const signal = { aborted: true, reason: cause }
  let turnEnds = null
  try {
    // Stands in for any error leaving the turn body once the signal aborted.
    throw new Error('turn aborted')
  } catch (error) {
    const read = (() => {
      if (!signal.aborted) return undefined
      switch (signal.reason.kind) {
        case 'user':
        case 'parent':
        case 'disposed':
          return { kind: signal.reason.kind }
        case 'hook':
          return { kind: 'hook', reason: signal.reason.reason }
        default:
          return assertNever(signal.reason)
      }
    })()
    if (read !== undefined) {
      turnEnds = { kind: 'aborted', reason: read }
      throw error
    }
    turnEnds = { kind: 'error', error: { message: String(error), code: 'UNKNOWN' } }
  } finally {
    // The loop appends `{ turn, reason: turnEnds }` here, with a non-null
    // assertion on a variable that the `catch` throw above left null.
    return { turn: 1, reason: turnEnds }
  }
}

/** A minimal Cordis-shaped context capturing the `llm/stream` listener. */
function fakeContext(agent) {
  const listeners = []
  return {
    on(name, listener) {
      if (name === 'llm/stream') listeners.push(listener)
    },
    logger: { warn() {}, debug() {} },
    agents: { get: () => agent },
    fire(options, next) {
      assert.equal(listeners.length, 1, 'apply() must register exactly one llm/stream listener')
      return listeners[0](options, next)
    },
  }
}

/** A guard-able agent stand-in recording the exact object handed to `cancel`. */
function fakeAgent() {
  const cancels = []
  return {
    cancels,
    steer: () => {},
    inject: () => {},
    cancel: cause => { cancels.push(cause) },
  }
}

/** Deterministic high-entropy reasoning text, as in `detection.spec.mjs`. */
function reasoning(seed, length = 400) {
  let hash = 0
  for (const ch of seed) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0
  let out = ''
  while (out.length < length) {
    hash = (hash * 1103515245 + 12345) >>> 0
    out += String.fromCharCode(97 + (hash % 26))
  }
  return out
}

/** One reasoning-only call: zero text, zero tool deltas — shape 1 of the loop. */
async function* reasoningOnly(text) {
  yield { type: 'reasoning-delta', index: 0, text }
  yield { type: 'finish', reason: { kind: 'stop' } }
}

async function drain(iterable) {
  const out = []
  for await (const chunk of iterable) out.push(chunk)
  return out
}

const OPTIONS = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })

/** Drive three stalled calls so the detector escalates through `react()`. */
async function escalate(ctx) {
  for (let call = 0; call < 3; call++) {
    await drain(ctx.fire(OPTIONS, () => reasoningOnly(reasoning('the same stalled subject'))))
  }
}

test('escalate:cancel hands the loop a member of the closed union, not the raw string', async () => {
  const agent = fakeAgent()
  const ctx = fakeContext(agent)
  plugin.apply(ctx, CONFIG)
  await escalate(ctx)

  assert.equal(agent.cancels.length, 1, 'the third stalled call must cancel exactly once')
  const cause = agent.cancels[0]
  assert.equal(typeof cause, 'object', `a bare string is outside AgentCancelCause: ${JSON.stringify(cause)}`)
  assert.deepEqual(cause, { kind: 'hook', reason: 'thinking-loop' })
})

test('the emitted cause round-trips instead of throwing in the loop', async () => {
  const agent = fakeAgent()
  const ctx = fakeContext(agent)
  plugin.apply(ctx, CONFIG)
  await escalate(ctx)

  const end = persistedTurnEnd(agent.cancels[0])
  assert.notEqual(end.reason, null, 'a null reason is the defect this guard must not produce')
  assert.deepEqual(end.reason, { kind: 'aborted', reason: { kind: 'hook', reason: 'thinking-loop' } })
  // The reads the client performs on that record — each one used to be the
  // `reading 'kind'` of a null dereference (`turnProcessAlwaysOpen`,
  // `conversation-nodes/turn-error.ts`).
  assert.equal(end.reason.kind, 'aborted')
  assert.equal(end.reason.reason.kind, 'hook')
  assert.equal(end.reason.reason.reason, 'thinking-loop')
})

test('control: the reported producer reproduces {"reason":null}', () => {
  // Without this arm the two tests above are only as strong as the mirror: the
  // mirror has to be able to fail. A bare string is exactly what v0.1.11 sent
  // (`agent.cancel(config.cancelCause as unknown as AgentCancelCause)`).
  const end = persistedTurnEnd('thinking-loop')
  assert.equal(end.reason, null, 'the bare string must still produce the reported null')
  assert.throws(() => end.reason.kind, TypeError, 'and the client dereference throws on it')
})

test('cancelCauseFor keeps the configured string as the reason', () => {
  assert.deepEqual(cancelCauseFor({ cancelCause: 'custom-cause' }), { kind: 'hook', reason: 'custom-cause' })
  // A non-string is still a string by the time it is a reason: the config schema
  // is the only place that validates it, so the union member is stable whatever
  // the caller put in.
  assert.equal(typeof cancelCauseFor({ cancelCause: '' }).reason, 'string')
})

test('the schema ships the documented default and still accepts the key', () => {
  const resolved = plugin.Config({})
  assert.equal(resolved.cancelCause, 'thinking-loop')
  assert.equal(plugin.Config({ cancelCause: 'x' }).cancelCause, 'x')
})
