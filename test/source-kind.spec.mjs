/**
 * Source-kind regression for issue #2.
 *
 * #2 (shark6s, dsh Desktop 0.2.0-rc.2) reported that the guard works but the
 * turn it reacts to dies: every reaction message this plugin writes used the
 * retired bare `{ kind: 'plugin' }` wrapper, and session format **v4** refuses
 * it — `format v4 message requires a producer-owned source kind`.
 *
 * Both delivery paths are caught by that refusal, and they are caught by
 * different validators:
 *
 *  - `agent.steer(message)` lands as a `user/message` event, checked by
 *    `assertV4MessageSources` before the Session adopts it;
 *  - `agent.inject(message)` lands inside `agent/inbox/spliced.inserted`,
 *    checked by `assertV4SourceRowAdmission`.
 *
 * So this file drives the shipped `apply()` and asserts on the messages the two
 * reactions actually build — a test of a helper would not see the failure, and
 * neither would a test that only exercised one of the two paths.
 *
 * The rule below is quoted from the harness
 * (`packages/session/session-format-v3-to-v4/src/message-sources.ts`):
 *
 *   if (!isSessionFormatJsonObject(value) || typeof value['kind'] !== 'string'
 *       || value['kind'].length === 0 || value['kind'] === 'plugin')
 *     throw new SessionFormatError('format v4 message requires a producer-owned source kind')
 *
 * It is a *threshold*, not a whitelist: any non-empty string other than the
 * retired literal is admitted. There is no `'plugin'` entry in
 * `MessageSourceMap` from the 0.1.7 line on — each producer declares its own
 * kind, which is why the declaration in `src/index.ts` is load-bearing rather
 * than cosmetic: drop it and `tsc` refuses the literal as not assignable to
 * `MessageSource`.
 *
 * What this file cannot do is call the validator itself; the package exporting
 * it is not a dependency of this plugin. Both row shapes were replayed through
 * the real `assertV4RowAdmission` on a `0.2.0-rc.2` install when this fix was
 * made — old kind and new, on `user/message` and on
 * `agent/inbox/spliced.inserted`: the old fails on both rows, the new passes on
 * both. That run is quoted in the issue #2 reply; what is pinned *here* is the
 * part that can regress, namely the source these two reactions write.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { markAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import * as plugin from '../lib/index.js'

const CONFIG = {
  maxThinkingSteps: 3,
  minReasoningChars: 64,
  repeatRatio: 0.5,
  similarityThreshold: 0.8,
  escalate: 'warn',
  maxFires: 4,
  cancelCause: 'thinking-loop',
  maxRepeatedText: 4,
  breakCode: 'REPETITIVE_OUTPUT',
  breakCorrection: true,
  maxRepeatedCycleChars: 0,
  minRepeatedCycleChars: 512,
}

/** The v4 admission rule, verbatim in effect (see the file header). */
function admitsSource(message) {
  const source = message?.source
  return typeof source === 'object' && source !== null
    && typeof source.kind === 'string' && source.kind.length > 0 && source.kind !== 'plugin'
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

/** A guard-able agent stand-in recording the reactions it receives. */
function fakeAgent() {
  const steered = []
  const injected = []
  return {
    steered,
    injected,
    steer: m => steered.push(m),
    inject: m => injected.push(m),
    cancel: () => {},
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

test('the inject path (agent/inbox/spliced) carries a producer-owned source kind', async () => {
  const agent = fakeAgent()
  const ctx = fakeContext(agent)
  plugin.apply(ctx, CONFIG)

  // Three stalled calls: the detector escalates on the third (maxThinkingSteps),
  // and `escalate: 'warn'` reacts through `agent.inject`.
  for (let call = 0; call < 3; call++) {
    await drain(ctx.fire(OPTIONS, () => reasoningOnly(reasoning('the same stalled subject'))))
  }

  assert.equal(agent.injected.length, 1, 'the third stalled call must produce exactly one injection')
  const message = agent.injected[0]
  assert.ok(admitsSource(message), 'v4 refuses this source: ' + JSON.stringify(message.source))
  assert.equal(message.source.kind, 'plugin:thinking-loop-guard')
  assert.equal(message.source.plugin, 'thinking-loop-guard', 'unknown attribution is preserved by v4')
  assert.equal(message.source.form, 'notice')
})

test('the steer path (user/message) carries a producer-owned source kind', async () => {
  const agent = fakeAgent()
  const ctx = fakeContext(agent)
  plugin.apply(ctx, { ...CONFIG, maxRepeatedText: 4 })

  // Four identical deltas trip the mid-stream breaker, which steers its
  // correction before yielding the terminal finish.
  async function* repetitive() {
    for (let i = 0; i < 8; i++) yield { type: 'text-delta', index: 0, text: 'tick' }
  }
  const out = await drain(ctx.fire(OPTIONS, () => repetitive()))

  assert.equal(out.at(-1).reason.kind, 'error', 'the breaker must end the call itself')
  assert.equal(agent.steered.length, 1, 'the break correction must be steered')
  const message = agent.steered[0]
  assert.ok(admitsSource(message), 'v4 refuses this source: ' + JSON.stringify(message.source))
  assert.equal(message.source.kind, 'plugin:thinking-loop-guard')
  assert.equal(message.source.plugin, 'thinking-loop-guard')
})

test('control: the quoted rule rejects the retired shape', () => {
  // Without this arm the two tests above could pass on a build that writes the
  // retired shape, if the rule were written loosely enough to admit everything.
  // These are the four clauses of the upstream predicate, each exercised.
  assert.equal(admitsSource({ source: { kind: 'plugin', plugin: 'x' } }), false, "'plugin' is retired")
  assert.equal(admitsSource({ source: { kind: '' } }), false, 'an empty kind is not an identity')
  assert.equal(admitsSource({ source: { kind: 42 } }), false, 'a non-string kind is not an identity')
  assert.equal(admitsSource({ source: undefined }), false, 'a message with no source is refused')
  assert.equal(admitsSource({ source: { kind: 'plugin:thinking-loop-guard' } }), true,
    'and a producer-owned kind is admitted')
})
