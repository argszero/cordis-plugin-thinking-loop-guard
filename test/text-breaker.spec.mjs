/**
 * Mid-stream repetition breaker — the issue #2848 shape.
 *
 * #2848 is a single model call that repeated one sentence for ~420,000
 * characters over ten minutes (~2825 text chunks). Every detector that judges a
 * call *after* it ends is structurally too late for that, so v0.1.6 ends the
 * stream from inside the `llm/stream` wrapper.
 *
 * The load-bearing assertions are the ones that exercise `apply()` rather than
 * the pure helper: a breaker that counts correctly but never emits its terminal
 * `finish` would still leave the call running, and that is exactly the failure a
 * helper-only test cannot see. `test/…` runs on the built `lib/index.js`, so
 * these also pin the shipped artifact.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { markAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import * as plugin from '../lib/index.js'
import { TextRepetitionDetector, countRepeatedText } from '../lib/index.js'

const CONFIG = {
  maxThinkingSteps: 3,
  minReasoningChars: 2048,
  repeatRatio: 0.5,
  similarityThreshold: 0.8,
  escalate: 'steer',
  maxFires: 4,
  cancelCause: 'thinking-loop',
  maxRepeatedText: 4,
  breakCode: 'REPETITIVE_OUTPUT',
  breakCorrection: true,
}

/* -------------------------------------------------------------------------- */
/* the primitives                                                             */
/* -------------------------------------------------------------------------- */

test('countRepeatedText measures the trailing identical run after trimming', () => {
  assert.equal(countRepeatedText([]), 0)
  assert.equal(countRepeatedText(['a', 'b', 'c']), 1)
  assert.equal(countRepeatedText(['a', 'tick', 'tick', 'tick']), 3)
  assert.equal(countRepeatedText(['same ', ' same', 'same']), 3, 'whitespace-only differences are not differences')
})

test('countRepeatedText anchors at the tail, not at any run in the call', () => {
  // An early burst of repeats must not trip a breaker that is watching the
  // stream's CURRENT behaviour — the model resumed working after it.
  assert.equal(countRepeatedText(['x', 'x', 'x', 'progress continues here']), 1)
})

test('countRepeatedText counts blank-only output as repetition', () => {
  assert.equal(countRepeatedText(['\n', '  ', '\n\n']), 3)
})

test('the detector trips exactly once, on the delta that completes the run', () => {
  const d = new TextRepetitionDetector({ ...CONFIG, maxRepeatedText: 3 })
  assert.equal(d.push('tick'), false)
  assert.equal(d.push('tick'), false)
  assert.equal(d.push('tick'), true, 'the third identical delta completes the run')
  assert.equal(d.push('tick'), false, 'a tripped breaker must not report again')
  assert.equal(d.tripped, true)
})

test('the detector ignores repeated bursts shorter than the threshold', () => {
  const d = new TextRepetitionDetector({ ...CONFIG, maxRepeatedText: 10 })
  for (let i = 0; i < 25; i++) assert.equal(d.push(i % 2 === 0 ? 'left' : 'right'), false)
  assert.equal(d.tripped, false)
})

test('the detector tracks emitted characters for the break report', () => {
  const d = new TextRepetitionDetector({ ...CONFIG, maxRepeatedText: 3 })
  d.push('abc')
  d.push('de')
  assert.equal(d.emittedChars, 5)
})

test('`maxRepeatedText: 0` disables the breaker', () => {
  const d = new TextRepetitionDetector({ ...CONFIG, maxRepeatedText: 0 })
  for (let i = 0; i < 500; i++) assert.equal(d.push('tick'), false)
  assert.equal(d.tripped, false)
})

/* -------------------------------------------------------------------------- */
/* through `apply()` — the assertions that can actually fail                  */
/* -------------------------------------------------------------------------- */

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
  let cancelled = 0
  return {
    steered,
    injected,
    get cancelled() { return cancelled },
    steer: m => steered.push(m),
    inject: m => injected.push(m),
    cancel: () => { cancelled++ },
  }
}

/**
 * A provider stream that emits `count` identical visible-output deltas.
 *
 * Only `text-delta` chunks are produced — the reported loop emitted text and
 * nothing else, so reasoning deltas and a provider finish would only make the
 * fixture less faithful.
 *
 * @param count - how many identical deltas to emit.
 * @param text - the repeated payload.
 * @param terminal - the chunk the adapter would have finished with.
 */
async function* textStream(count, text = 'The `register` API matches. ', terminal = null) {
  yield { type: 'block-start', index: 0, blockType: 'text' }
  for (let i = 0; i < count; i++) yield { type: 'text-delta', index: 0, text }
  if (terminal !== null) yield terminal
}

/** Run one stream through the installed listener and collect every chunk. */
async function collect(agent, count, config = CONFIG, text) {
  const ctx = fakeContext(agent)
  plugin.apply(ctx, { ...CONFIG, ...config })
  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  const out = []
  for await (const chunk of ctx.fire(options, () => textStream(count, text))) out.push(chunk)
  return out
}

test('a repetitive single call is cut short with a terminal error finish', async () => {
  const agent = fakeAgent()
  const out = await collect(agent, 50)
  const deltas = out.filter(c => c.type === 'text-delta')
  const finish = out.at(-1)

  assert.equal(deltas.length, 4, 'the stream must stop at the threshold, not run to the model max_tokens')
  assert.equal(finish.type, 'finish', 'the last chunk must be terminal — a bare return violates the llm invariant')
  assert.deepEqual(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, 'REPETITIVE_OUTPUT')
  assert.ok(finish.reason.failure.message.length > 0)
})

test('the terminal finish is the last chunk: nothing follows it', async () => {
  const out = await collect(fakeAgent(), 50)
  const at = out.findIndex(c => c.type === 'finish')
  assert.ok(at > 0, 'a finish must be emitted')
  assert.equal(at, out.length - 1, 'the invariant rejects any chunk after terminal finish')
})

test('the pending deltas after the break are dropped, not flushed', async () => {
  // The fixture's adapter would keep going to 50; the breaker must abandon the
  // upstream iteration rather than drain it into the durable log.
  const out = await collect(fakeAgent(), 5000)
  assert.equal(out.filter(c => c.type === 'text-delta').length, 4)
})

test('the break steers the agent with the observed size of the repetition', async () => {
  const agent = fakeAgent()
  await collect(agent, 50)
  assert.equal(agent.steered.length, 1, 'the correction must be queued before the error finish')
  const text = agent.steered[0].content.map(b => b.text).join('')
  assert.match(text, /repeated itself/)
  assert.match(text, /\d+ characters/)
})

test('`breakCorrection: false` still breaks but says nothing', async () => {
  const agent = fakeAgent()
  const out = await collect(agent, 50, { breakCorrection: false })
  assert.equal(out.at(-1).reason.kind, 'error')
  assert.equal(agent.steered.length, 0)
})

/**
 * A Cordis-shaped context whose `llm/stream` listeners form the real chain.
 *
 * `prepend` listeners are called first (the shipped invariant registers that
 * way), so the array order here IS the wrapper nesting the llm service builds.
 */
function chainContext(agent) {
  const listeners = []
  const ctx = {
    on(name, listener, options) {
      if (name !== 'llm/stream') return
      if (options?.prepend) listeners.unshift(listener)
      else listeners.push(listener)
    },
    logger: { warn() {}, debug() {} },
    agents: { get: () => agent },
    get: () => undefined,
  }
  return { ctx, listeners }
}

/** Install the shipped dsh-llm invariant, returning the failures it records. */
async function installInvariant(context) {
  const { apply } = await import('@deepseek-ai/dsh-llm/invariant')
  const failures = []
  const fail = m => failures.push(m)
  const inner = new Proxy(context, { get: (t, k) => (k === 'on' ? t.on.bind(t) : t[k]) })
  await apply(
    { ...inner, invariants: { register: (_name, installer) => installer(inner, fail) } },
    fail,
  )
  return failures
}

/** Drive one raw stream through the whole installed chain. */
async function runChain(listeners, options, source) {
  const out = []
  const step = i => (i >= listeners.length
    ? source()
    : listeners[i](options, () => step(i + 1)))
  for await (const chunk of step(0)) out.push(chunk)
  return out
}

test('the break finish is protocol-legal: the shipped llm invariant accepts it', async () => {
  // The breaker fires with the call's text block still OPEN, and the invariant
  // only tolerates that for an `error`/`aborted` reason. This runs the wrapper's
  // own output through the real validator instead of trusting a reading of
  // `packages/llm/llm/src/invariant.ts` — which is precisely what an earlier
  // `stop` finish got wrong ("finished with 1 open block(s)").
  const { ctx, listeners } = chainContext(fakeAgent())
  const failures = await installInvariant(ctx)
  plugin.apply(ctx, CONFIG)
  assert.equal(listeners.length, 2, 'the chain must be invariant -> plugin')

  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  const seen = await runChain(listeners, options, () => textStream(50))

  assert.deepEqual(failures, [], 'the synthesized finish must not violate the stream grammar')
  assert.equal(seen.at(-1).type, 'finish')
  assert.equal(seen.at(-1).reason.kind, 'error')
})

test('an UNBROKEN repetitive call would have failed only at the provider ceiling', async () => {
  // Control for the test above: the same fixture WITHOUT the plugin ends at the
  // adapter's own terminal chunk. It proves the chain harness itself is sound,
  // so the two tests differ only by the breaker.
  const raw = await (async () => {
    const out = []
    for await (const c of textStream(50, undefined, { type: 'finish', reason: { kind: 'stop' } })) out.push(c)
    return out
  })()
  assert.equal(raw.filter(c => c.type === 'text-delta').length, 50, 'without the breaker all 50 deltas flow')
  assert.deepEqual(raw.at(-1).reason, { kind: 'stop' })
})

test('a call inside the threshold is left completely alone', async () => {
  const agent = fakeAgent()
  // 3 identical deltas against a threshold of 4 — a legitimately repeated short
  // line (a header row, a log prefix) must not be treated as degeneration.
  const out = await collect(agent, 3, { maxRepeatedText: 4 })
  assert.equal(out.filter(c => c.type === 'text-delta').length, 3)
  assert.equal(out.at(-1).type, 'text-delta', 'no finish may be synthesized for a healthy call')
  assert.equal(agent.steered.length, 0)
})

test('varying visible output never trips the breaker', async () => {
  const agent = fakeAgent()
  const ctx = fakeContext(agent)
  plugin.apply(ctx, CONFIG)
  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  async function* varying() {
    for (let i = 0; i < 200; i++) yield { type: 'text-delta', index: 0, text: `chunk number ${i} ` }
  }
  const out = []
  for await (const chunk of ctx.fire(options, () => varying())) out.push(chunk)
  assert.equal(out.filter(c => c.type === 'text-delta').length, 200)
  assert.equal(agent.steered.length, 0)
})

/* -------------------------------------------------------------------------- */
/* configuration                                                              */
/* -------------------------------------------------------------------------- */

test('the schema accepts the breaker keys and applies their defaults', () => {
  // Guards a real failure mode: schemastery reserves some key names, and a
  // collision fails at plugin *load* rather than in a type check.
  const resolved = plugin.Config({})
  assert.equal(resolved.maxRepeatedText, 60)
  assert.equal(resolved.breakCode, 'REPETITIVE_OUTPUT')
  assert.equal(resolved.breakCorrection, true)
  assert.equal(plugin.Config({ maxRepeatedText: 0 }).maxRepeatedText, 0, '0 must survive as the documented off switch')
})

test('the schema rejects a negative repetition threshold', () => {
  assert.throws(() => plugin.Config({ maxRepeatedText: -1 }))
})
