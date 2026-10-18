/**
 * Reasoning-channel repetition breaker — the discussion #9524 shape.
 *
 * #9524 is a single model call whose **thinking** collapsed onto the shortest
 * standalone sentence and stayed there: `好。好。好。` under Chinese, `Go. Go.
 * Go.` at an action point, measured at a longest repeated fragment run of 7298,
 * with the reply swallowed and the session unusable. It is the one shape no
 * shipped rule could reach:
 *
 *  - the visible-output breaker (`maxRepeatedText`, `maxRepeatedCycleChars`)
 *    never receives a `reasoning-delta`;
 *  - this plugin's post-call judgement does read reasoning, but only after the
 *    call has ended — and then needs `maxThinkingSteps` consecutive stalled calls.
 *
 * So the worst call of a session was the call nothing could interrupt. These
 * assertions cover the three things that can go wrong in the fix: the period-1
 * decision (the reason this is not the visible rule pointed at another string),
 * the false-positive control (reasoning revisits itself by nature), and whether
 * the wrapper actually emits a terminal chunk the shipped invariant accepts.
 *
 * `test/*.spec.mjs` runs on the built `lib/index.js`, so these pin the shipped
 * artifact rather than the source.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { markAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import * as plugin from '../lib/index.js'
import { ReasoningRepetitionDetector, trailingCycle, trailingReasoningCycle } from '../lib/index.js'
import { grammarSource, installGrammar } from './grammar.mjs'

/** Which grammar this dsh line can offer; read once, named in the arm below. */
const REGIME = await grammarSource()

/** The visible-output rules are off here: every assertion below is about reasoning. */
const CONFIG = {
  maxThinkingSteps: 3,
  minReasoningChars: 2048,
  repeatRatio: 0.5,
  similarityThreshold: 0.8,
  escalate: 'steer',
  maxFires: 4,
  cancelCause: 'thinking-loop',
  maxRepeatedText: 0,
  maxRepeatedCycleChars: 0,
  minRepeatedCycleChars: 256,
  maxRepeatedReasoningCycleChars: 64,
  minRepeatedReasoningCycleChars: 256,
  // The density arm runs at its shipped default here too: these fixtures are all
  // short-unit bleeds (period 1-4 characters), which the unit floor keeps out of
  // the density tally entirely, so the arm must stay silent and leave the
  // attribution to the shape rule. `reasoning-novelty.spec.mjs` owns the
  // positives for it.
  minReasoningSentenceChars: 8,
  minReasoningSentences: 64,
  minReasoningNoveltyRatio: 0.05,
  breakCode: 'REPETITIVE_OUTPUT',
  breakCorrection: true,
}

/* -------------------------------------------------------------------------- */
/* the primitive, and the one decision it makes differently                   */
/* -------------------------------------------------------------------------- */

test('trailingReasoningCycle finds the reported shapes at their own periods', () => {
  // A-type: the shortest standalone Chinese sentence, period 2.
  assert.equal(trailingReasoningCycle('好。'.repeat(400), 64, 256), 256)
  // B-type: the action token that leaks into Chinese thinking, period 4.
  assert.equal(trailingReasoningCycle('Go. '.repeat(400), 64, 256), 256)
  // Below the span minimum: a short repeat is not a bleed.
  assert.equal(trailingReasoningCycle('好。'.repeat(60), 64, 256), 0)
  // Ordinary variable reasoning is not periodic at all.
  assert.equal(trailingReasoningCycle('I should read the file first, then check whether the test passes.', 64, 256), 0)
})

test('the reasoning rule admits a one-character period that the output rule refuses', () => {
  // This is the whole reason the channel has its own function: #9524's decisive
  // case is a SINGLE character repeated thousands of times, and the visible rule
  // excludes period 1 to spare `=====` divider lines.
  const bleed = '好'.repeat(400)
  assert.equal(trailingCycle(bleed, 64, 256), 0, 'the output rule must keep refusing period 1')
  assert.equal(trailingReasoningCycle(bleed, 64, 256), 256, 'reasoning must catch it')
  assert.equal(trailingReasoningCycle('x'.repeat(400), 64, 256), 256, 'a Latin token bleeds the same way')
  assert.equal(trailingReasoningCycle('7'.repeat(400), 64, 256), 256)
})

test('a one-character period is admitted only for a character that carries content', () => {
  // The cost of that admission, stated as a test rather than as a comment: a
  // divider is still a divider in reasoning, and a long run of layout
  // punctuation or whitespace is not a stuck model.
  for (const ch of ['=', '-', '─', '\n', ' ', '.', '。', '*', '_']) {
    assert.equal(trailingReasoningCycle(ch.repeat(400), 64, 256), 0, `${JSON.stringify(ch)} must not be a bleed on its own`)
  }
  // …and the known miss that symmetry buys, pinned so it is a decision and not a
  // surprise: a bleed of a pure punctuation character is not caught.
  assert.equal(trailingReasoningCycle('。'.repeat(4000), 64, 256), 0, 'documented miss: a one-character period of punctuation')
})

test('a multi-character period is admitted on the shared rule, punctuation included', () => {
  // `好。` is period 2 with two distinct characters, so it never needed the
  // exemption — the two channels agree wherever the unit is longer than one
  // character.
  const circle = '。！'.repeat(400)
  assert.equal(trailingCycle(circle, 64, 256), 256)
  assert.equal(trailingReasoningCycle(circle, 64, 256), 256)
})

test('the reasoning rule is exact: a near-repeat of a long period is not a bleed', () => {
  // A model re-deriving the same conclusion with different wording is the shape
  // the post-call detector exists for; it must not be truncated mid-call. These
  // are the reasoning-channel analogues of the output samples the cycle rule was
  // calibrated against — visually repetitive, and not periodic.
  const nearRepeats = {
    'restated conclusion': Array.from({ length: 200 }, (_, i) => `第 ${i} 步：我需要先读文件，再判断是否需要修改。\n`).join(''),
    'checked items': Array.from({ length: 200 }, (_, i) => `- item-${i} checked against the same rule\n`).join(''),
    'bulleted plan': Array.from({ length: 120 }, (_, i) => `${i + 1}. run the same command and compare the output\n`).join(''),
  }
  for (const [label, text] of Object.entries(nearRepeats)) {
    assert.equal(trailingReasoningCycle(text, 64, 256), 0, `${label} must not be flagged`)
  }
  // A bleed interrupted by one foreign character stops being periodic — the rule
  // reads the tail that is live now, which is what a guard can honestly see.
  assert.equal(trailingReasoningCycle(`${'好。'.repeat(200)}x`, 64, 256), 0)
})

test('`maxPeriod` below 1 disables the primitive', () => {
  assert.equal(trailingReasoningCycle('好。'.repeat(400), 0, 256), 0)
  assert.equal(trailingCycle('好。'.repeat(400), 0, 256), 0)
})

/* -------------------------------------------------------------------------- */
/* the detector                                                               */
/* -------------------------------------------------------------------------- */

test('the reasoning detector trips exactly once, on the delta that completes the span', () => {
  const d = new ReasoningRepetitionDetector(CONFIG)
  let fired = 0
  for (let i = 0; i < 400; i++) if (d.push('好。')) fired++
  assert.equal(fired, 1, 'exactly one delta completes the detection')
  assert.equal(d.tripped, true)
  assert.equal(d.trippedBy, 'repeating-reasoning')
  assert.ok(d.emittedChars >= 256, `must fire at the span minimum, fired at ${d.emittedChars}`)
})

test('the detector fires inside the walk, not at the end of the call', () => {
  const d = new ReasoningRepetitionDetector(CONFIG)
  for (let i = 0; i < 4000; i++) if (d.push('好。')) break
  assert.ok(d.emittedChars <= 290, `the reported bleed ran to LRS 7298; this must cut near the minimum, cut at ${d.emittedChars}`)
})

test('a short repetition is left alone', () => {
  const d = new ReasoningRepetitionDetector(CONFIG)
  for (let i = 0; i < 60; i++) assert.equal(d.push('好。'), false)
  assert.equal(d.tripped, false, '240 characters is below the span minimum')
})

test('`maxRepeatedReasoningCycleChars: 0` disables the reasoning rule', () => {
  const d = new ReasoningRepetitionDetector({ ...CONFIG, maxRepeatedReasoningCycleChars: 0 })
  for (let i = 0; i < 5000; i++) assert.equal(d.push('好。'), false)
  assert.equal(d.tripped, false)
})

test('the scan runs on a fixed stride, not once per delta', () => {
  // The stride is what keeps the rule affordable: each scan costs
  // `O(maxPeriod * minSpan)` and a bleed can run for hundreds of thousands of
  // characters. It has exactly one behavioural signature, and this is it — with a
  // span minimum below the stride the rule fires at the first 32-character
  // window instead of at the minimum, which is the most latency the cadence can
  // ever add.
  const tight = new ReasoningRepetitionDetector({ ...CONFIG, minRepeatedReasoningCycleChars: 2 })
  let fired = 0
  for (let i = 0; i < 100; i++) if (tight.push('好。')) fired++
  assert.equal(fired, 1)
  assert.equal(tight.emittedChars, 32, 'the first scan window, not the 2-character minimum')
})

test('the detector counts characters, so the provider chunking does not matter', () => {
  // A provider may deliver the same bleed one character at a time; the rule is
  // character-level, so it must fire at the same point.
  const perChar = new ReasoningRepetitionDetector(CONFIG)
  let fired = 0
  for (let i = 0; i < 4000; i++) if (perChar.push(i % 2 === 0 ? '好' : '。')) fired++
  assert.equal(fired, 1)
  assert.ok(perChar.emittedChars <= 290, `fired at ${perChar.emittedChars}`)
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
 * A provider stream that bleeds reasoning and then moves on.
 *
 * Faithful to #9524 in the detail that matters: the bleed sits **before** the
 * tool call the model obviously means to make, so the fixture can assert that
 * the pending tool call never reaches the durable log.
 *
 * @param count - how many reasoning deltas to emit.
 * @param unit - the repeated reasoning unit.
 */
async function* reasoningStream(count, unit = '好。', trailingText = true) {
  yield { type: 'block-start', index: 0, blockType: 'reasoning' }
  for (let i = 0; i < count; i++) yield { type: 'reasoning-delta', index: 0, text: unit }
  yield { type: 'reasoning-end', index: 0 }
  if (!trailingText) return
  yield { type: 'block-start', index: 1, blockType: 'text' }
  yield { type: 'text-delta', index: 1, text: 'Here is the answer.' }
  yield { type: 'finish', reason: { kind: 'stop' } }
}

/** Run one stream through the installed listener and collect every chunk. */
async function collect(agent, count, config = CONFIG, unit, trailingText = true) {
  const ctx = fakeContext(agent)
  plugin.apply(ctx, { ...CONFIG, ...config })
  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  const out = []
  for await (const chunk of ctx.fire(options, () => reasoningStream(count, unit, trailingText))) out.push(chunk)
  return out
}

test('a self-replicating reasoning call is cut short with a terminal error finish', async () => {
  const agent = fakeAgent()
  const out = await collect(agent, 4000)
  const finish = out.at(-1)

  assert.ok(
    out.filter(c => c.type === 'reasoning-delta').length < 4000,
    'the stream must be cut at the span minimum, not drained to the model ceiling',
  )
  assert.equal(finish.type, 'finish', 'the last chunk must be terminal — a bare return violates the llm invariant')
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, 'REPETITIVE_OUTPUT')
  assert.match(finish.reason.failure.message, /reasoning/)
})

test('the tool call the model meant to make is dropped, not flushed', async () => {
  // The bleed sits before the action, and the provider would have finished with
  // `stop` plus a text block. Neither may reach the consumer.
  const out = await collect(fakeAgent(), 4000)
  assert.equal(out.filter(c => c.type === 'text-delta').length, 0, 'the trailing output must not be flushed')
  assert.equal(out.filter(c => c.type === 'finish').length, 1)
  assert.equal(out.at(-1).reason.kind, 'error', 'the adapter stop the provider sent must never arrive')
})

test('the terminal finish is the last chunk: nothing follows it', async () => {
  const out = await collect(fakeAgent(), 4000)
  const at = out.findIndex(c => c.type === 'finish')
  assert.ok(at > 0, 'a finish must be emitted')
  assert.equal(at, out.length - 1, 'the invariant rejects any chunk after terminal finish')
})

test('the break steers the agent, naming the reasoning channel and the size', async () => {
  const agent = fakeAgent()
  await collect(agent, 4000)
  assert.equal(agent.steered.length, 1, 'the correction must be queued before the error finish')
  const text = agent.steered[0].content.map(b => b.text).join('')
  assert.match(text, /reasoning/, 'the correction must name the channel it interrupted')
  assert.match(text, /\d+ characters/)
})

test('the English action-token variant trips the same way', async () => {
  const agent = fakeAgent()
  const out = await collect(agent, 4000, CONFIG, 'Go. ')
  assert.equal(out.at(-1).reason.kind, 'error')
  assert.equal(agent.steered.length, 1)
})

test('a reasoning call inside the span is left completely alone', async () => {
  const agent = fakeAgent()
  const out = await collect(agent, 60)
  assert.equal(out.at(-1).type, 'finish')
  assert.deepEqual(out.at(-1).reason, { kind: 'stop' }, 'a healthy call keeps the provider’s own finish')
  assert.equal(agent.steered.length, 0)
})

test('reasoning that revisits itself without becoming periodic is not cut', async () => {
  // The false-positive control that matters for a channel whose whole job is
  // revisiting: a long thinking text that restates itself in different words,
  // and one that legitimately contains a divider and a table.
  const agent = fakeAgent()
  const ctx = fakeContext(agent)
  plugin.apply(ctx, CONFIG)
  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  const table = '| plan | status |\n| --- | --- |\n'
    + Array.from({ length: 40 }, (_, i) => `| step-${i} | pending |\n`).join('')
  async function* rich() {
    yield { type: 'block-start', index: 0, blockType: 'reasoning' }
    for (let i = 0; i < 40; i++) {
      yield { type: 'reasoning-delta', index: 0, text: `第 ${i} 步：我需要先读文件，再判断是否需要修改。`.replace('第 0 步', `第 ${i} 步`) }
    }
    yield { type: 'reasoning-delta', index: 0, text: table }
    yield { type: 'reasoning-delta', index: 0, text: '='.repeat(400) }
    yield { type: 'reasoning-end', index: 0 }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
  const out = []
  for await (const chunk of ctx.fire(options, () => rich())) out.push(chunk)
  assert.deepEqual(out.at(-1).reason, { kind: 'stop' }, 'honest reasoning must keep its own terminal chunk')
  assert.equal(agent.steered.length, 0)
})

test('the visible-output rules are unaffected by the reasoning rule being on', async () => {
  // Regression control for the refactor: `trailingCycle` now shares its scan with
  // the reasoning variant, so the output behaviour is re-pinned here.
  const agent = fakeAgent()
  const ctx = fakeContext(agent)
  const config = { ...CONFIG, maxRepeatedText: 4, maxRepeatedCycleChars: 64 }
  plugin.apply(ctx, config)
  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  async function* repeating() {
    yield { type: 'block-start', index: 0, blockType: 'text' }
    for (let i = 0; i < 50; i++) yield { type: 'text-delta', index: 0, text: 'tick' }
  }
  const out = []
  for await (const chunk of ctx.fire(options, () => repeating())) out.push(chunk)
  assert.equal(out.at(-1).reason.kind, 'error')
  assert.equal(out.at(-1).reason.failure.code, 'REPETITIVE_OUTPUT')
  assert.match(out.at(-1).reason.failure.message, /visible output/)
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
  const failures = []
  await installGrammar(context, m => failures.push(m))
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

test(`the reasoning cut is protocol-legal with a reasoning block still open (${REGIME} grammar)`, async () => {
  // The output breaker fires with a text block open; the reasoning breaker fires
  // with a REASONING block open, which is a different position in the grammar.
  // `interruptedBlocks()` says it is allowed; this runs the wrapper's actual
  // output through the grammar instead of trusting that reading — and
  // the fixture deliberately never emits `reasoning-end`, exactly like a call
  // whose stream never finished.
  const { ctx, listeners } = chainContext(fakeAgent())
  const failures = await installInvariant(ctx)
  plugin.apply(ctx, CONFIG)
  assert.equal(listeners.length, 2, 'the chain must be grammar -> plugin')

  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  const seen = await runChain(listeners, options, () => reasoningStream(4000, '好。', false))

  assert.deepEqual(failures, [], 'the synthesized finish must not violate the stream grammar')
  assert.equal(seen.at(-1).type, 'finish')
  assert.equal(seen.at(-1).reason.kind, 'error')
  assert.ok(seen.some(c => c.type === 'reasoning-delta'), 'the fixture must actually have emitted the reasoning block')
})

test('an UNBROKEN reasoning bleed would have run to the adapter’s own finish', async () => {
  // Control for the test above: the same raw stream WITHOUT the plugin reaches
  // the adapter's terminal chunk, so the chain harness itself is sound and the
  // two tests differ only by the breaker.
  const raw = await (async () => {
    const out = []
    for await (const c of reasoningStream(4000)) out.push(c)
    return out
  })()
  assert.equal(raw.filter(c => c.type === 'reasoning-delta').length, 4000, 'without the breaker all 4000 deltas flow')
  assert.deepEqual(raw.at(-1).reason, { kind: 'stop' })
})

/* -------------------------------------------------------------------------- */
/* configuration                                                              */
/* -------------------------------------------------------------------------- */

test('the schema ships the reasoning rule on, with its documented defaults', () => {
  const resolved = plugin.Config({})
  assert.equal(resolved.maxRepeatedReasoningCycleChars, 64)
  assert.equal(resolved.minRepeatedReasoningCycleChars, 256)
  assert.equal(plugin.Config({ maxRepeatedReasoningCycleChars: 0 }).maxRepeatedReasoningCycleChars, 0, '0 must survive as the off switch')
  assert.throws(() => plugin.Config({ minRepeatedReasoningCycleChars: 1 }))
  assert.throws(() => plugin.Config({ maxRepeatedReasoningCycleChars: -1 }))
})

test('the reasoning rule and the visible rules switch off independently', async () => {
  // Each of the three rules has to be disableable without touching the others:
  // an operator who trusts their own guard on one channel must be able to keep
  // this plugin off that channel exactly.
  const reasoningOff = await collect(fakeAgent(), 4000, { maxRepeatedReasoningCycleChars: 0 })
  assert.deepEqual(reasoningOff.at(-1).reason, { kind: 'stop' }, 'reasoning rule off => the bleed is not cut')

  const agent = fakeAgent()
  const out = await collect(agent, 4000, { maxRepeatedReasoningCycleChars: 0, maxRepeatedCycleChars: 64 })
  assert.equal(out.at(-1).type, 'finish')
  assert.equal(agent.steered.length, 0)
})
