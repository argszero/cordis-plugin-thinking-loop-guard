/**
 * Reasoning-channel **density** rule — the discussion #9683 shape, and the half
 * of #5976 that the shape rules cannot reach.
 *
 * Every other rule in this plugin measures a *shape*: is the tail periodic, is
 * the reasoning-only, is this call materially the previous one. #9683's own
 * numbers describe a call that defeats all of them at once — a block of 114,298
 * characters containing only twenty distinct sentences, laid out two ways in the
 * report: cycled in a fixed order (a period of 489 characters, so no tail is
 * ever periodic under the 64-character cap) and in an irregular order (no period
 * exists at all). The call was neither silent nor repetitive-looking; it had
 * simply run out of things to say. Measuring *that* is a different statistic
 * over the same characters.
 *
 * These assertions pin the four things that can go wrong:
 *
 *  - the positive decision, on the report's own shapes;
 *  - the false-positive controls — the generated documents this plugin already
 *    refuses to cut, plus real long-form prose;
 *  - the substring trap that a "count the repeats" implementation falls into
 *    (`完好。` contains `好。` inside an ordinary word);
 *  - that the wrapper actually emits a terminal chunk the shipped invariant
 *    accepts, for THIS rule and not only for its sibling.
 *
 * `test/*.spec.mjs` runs on the built `lib/index.js`, so these pin the shipped
 * artifact rather than the source.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { markAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import * as plugin from '../lib/index.js'
import { ReasoningRepetitionDetector, SentenceUnitCounter } from '../lib/index.js'

/** Both reasoning rules on, at their shipped defaults; the visible rules off. */
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
  // The shape rule is deliberately OFF in most tests below: every assertion here
  // is about the density rule, and the report's shapes trip both. Where the shape
  // rule must stay on, the test says so.
  maxRepeatedReasoningCycleChars: 0,
  minRepeatedReasoningCycleChars: 256,
  minReasoningSentenceChars: 8,
  minReasoningSentences: 64,
  minReasoningNoveltyRatio: 0.05,
  breakCode: 'REPETITIVE_OUTPUT',
  breakCorrection: true,
}

/* -------------------------------------------------------------------------- */
/* the report's shapes                                                        */
/* -------------------------------------------------------------------------- */

/** The twenty sentences #9683 says one block was built from, plus the toy one. */
const TWENTY = [
  'Let me write the file.', 'Let me run the command.', 'I will write the file now.',
  'Let me check the output.', 'Let me run it again.', 'I need to write this down.',
  'Let me apply the patch.', 'Let me verify the result.', 'Let me run the tests.',
  'I will write the next one.', 'Let me inspect the file.', 'Let me run the command now.',
  'Let me update the file.', 'Let me check the logs.', 'I will run the command.',
  'Let me write the answer.', 'Let me confirm the change.', 'Let me run the build.',
  'Let me write it out.', 'Let me finish the step.',
]

/** The reported block size, built by repeating the vocabulary in cycle order. */
const CYCLED = (() => {
  const cycle = TWENTY.map(s => `${s} `).join('')
  let text = ''
  while (text.length < 114298) text += cycle
  return text.slice(0, 114298)
})()

/** The same vocabulary in an order with no period at all. */
const IRREGULAR = (() => {
  let text = ''
  let k = 0
  while (text.length < 114298) {
    text += `${TWENTY[(k * 7 + Math.floor(k / 20)) % 20]} `
    k++
  }
  return text.slice(0, 114298)
})()

/**
 * Feed one call's reasoning through the shipped detector and report what it did.
 *
 * @param text - the call's reasoning.
 * @param config - a config override.
 * @param delta - how the provider chunks the text.
 */
function feed(text, config = CONFIG, delta = 16) {
  const detector = new ReasoningRepetitionDetector({ ...CONFIG, ...config })
  for (let i = 0; i < text.length; i += delta) {
    const piece = text.slice(i, i + delta)
    if (detector.push(piece)) {
      return { fired: true, at: i + piece.length, by: detector.trippedBy, novelty: detector.novelty }
    }
  }
  return { fired: false, at: -1, by: detector.trippedBy, novelty: detector.novelty }
}

test('the primitive measures whole units, and the substring trap does not exist', () => {
  // This is the assertion that separates a density measure from a "count the
  // repeats" one. #5976's own text contains the unit `好。` three times, and the
  // SUBSTRING `好。` four times, because `完好。` ("intact") contains it inside an
  // ordinary word. A measure that counted substrings would report a repetition no
  // reader can see.
  const counter = new SentenceUnitCounter(2)
  counter.push('好。发。好。')
  assert.equal(counter.total, 3, 'three units')
  assert.equal(counter.distinct, 2, 'two distinct units')
  counter.push('**紧急**：确认文件完好。')
  assert.equal(counter.total, 4, '`完好。` is ONE unit, not `完好` plus a `好。`')
  assert.equal(counter.distinct, 3)
})

test('a unit may be split across deltas without changing the tally', () => {
  const whole = new SentenceUnitCounter(8)
  whole.push('Let me run the command now. ')
  const split = new SentenceUnitCounter(8)
  split.push('Let me run the ')
  split.push('command now. ')
  assert.equal(split.total, whole.total)
  assert.equal(split.distinct, whole.distinct)
})

test('units shorter than the floor are dropped, not merged into the next one', () => {
  // The floor is what keeps a JSON dump's own structural lines from reading as a
  // loop; merging them forward would defeat it.
  const counter = new SentenceUnitCounter(8)
  counter.push('a. b. c. This one is long enough.')
  assert.equal(counter.total, 1)
})

test('a newline ends a unit, so a pasted listing is not one enormous sentence', () => {
  const counter = new SentenceUnitCounter(8)
  counter.push('first line here\nsecond line here\nthird line here\n')
  assert.equal(counter.total, 3)
})

test('the reported shapes fire, cycled and irregular alike', () => {
  for (const [label, text] of [['cycled', CYCLED], ['irregular', IRREGULAR]]) {
    const result = feed(text)
    assert.equal(result.fired, true, `${label} must fire`)
    assert.equal(result.by, 'reasoning-novelty')
    assert.ok(result.at < 20000, `${label} fired at ${result.at} — it must not wait for the whole block`)
  }
})

test('a single repeated sentence fires the density rule too', () => {
  // With the shape rule off, "Let me run. " forever is still visible to the
  // density measure — the two rules are complements, not alternatives.
  const result = feed('Let me run. '.repeat(7993))
  assert.equal(result.fired, true)
  assert.equal(result.by, 'reasoning-novelty')
})

test('the shape rule wins the attribution when both could fire', () => {
  // Deterministic attribution, and the reason the order is fixed: on a shape both
  // rules can see, the more specific finding must be the one reported.
  const result = feed('Let me run. '.repeat(7993), {
    maxRepeatedReasoningCycleChars: 64,
    minRepeatedReasoningCycleChars: 256,
  })
  assert.equal(result.fired, true)
  assert.equal(result.by, 'repeating-reasoning')
})

test('provider chunking does not change the decision', () => {
  for (const delta of [1, 8, 64, 512]) {
    const result = feed(CYCLED, CONFIG, delta)
    assert.equal(result.fired, true, `delta ${delta} must fire`)
    assert.equal(result.by, 'reasoning-novelty')
  }
})

/* -------------------------------------------------------------------------- */
/* the false-positive controls                                                */
/* -------------------------------------------------------------------------- */

test('the generated documents this plugin already refuses to cut stay clean', () => {
  // The same six samples `text-breaker.spec.mjs` uses. A JSON dump is the one
  // that constrains the unit floor: it repeats its own `"kind": "item",` line
  // forty times, and only the distinct `"id": N,` line per record keeps the
  // measure high.
  const samples = {
    'markdown table': '| id | name | status |\n| --- | --- | --- |\n'
      + Array.from({ length: 40 }, (_, i) => `| ${i} | item-${i} | ${i % 2 ? 'ok' : 'pending'} |\n`).join(''),
    'generated css': Array.from({ length: 40 }, (_, i) => `.row-${i} { display: flex; align-items: center; gap: 8px; }\n`).join(''),
    'json dump': JSON.stringify(Array.from({ length: 40 }, (_, i) => ({ id: i, kind: 'item', enabled: true })), null, 2),
    'log listing': Array.from({ length: 60 }, (_, i) => `2026-09-18T16:0${i % 10}:00 INFO processing record ${i} done\n`).join(''),
    'code stubs': Array.from({ length: 40 }, (_, i) => `export function handler${i}(ctx: Context): void {\n  ctx.logger.info('handler ${i}')\n}\n`).join(''),
    'bullet list': Array.from({ length: 40 }, (_, i) => `- step ${i + 1}: run the same command and check the output\n`).join(''),
  }
  for (const [label, text] of Object.entries(samples)) {
    const result = feed(text, CONFIG, 16)
    assert.equal(result.fired, false, `${label} must not be flagged (novelty ${result.novelty.toFixed(3)})`)
  }
})

test('the unit floor is what keeps the JSON dump clean, and 10 would break it', () => {
  // Stated as a test rather than as a comment, because the floor is the one
  // number in this rule that is not free: the documented rationale is that at a
  // floor of 10 the dump's record ids stop counting and the dump fires.
  const dump = JSON.stringify(Array.from({ length: 40 }, (_, i) => ({ id: i, kind: 'item', enabled: true })), null, 2)
  assert.equal(feed(dump, { minReasoningSentenceChars: 8 }, 16).fired, false)
  assert.equal(feed(dump, { minReasoningSentenceChars: 10 }, 16).fired, true, 'the documented failure mode of a higher floor')
})

test('long-form prose that revisits itself in different words is not cut', () => {
  // Real reasoning restates itself constantly. What it does not do is run out of
  // sentences, and that is the only thing this rule looks at.
  const text = Array.from({ length: 120 }, (_, i) =>
    `Step ${i}: I need to read the file first, then decide whether the change is needed at all. `).join('')
  const result = feed(text, CONFIG, 64)
  assert.equal(result.fired, false, `novelty ${result.novelty.toFixed(3)}`)
})

test('too few units is not evidence: the rule needs the unit floor', () => {
  // A three-sentence call cannot demonstrate that it is out of material, however
  // repetitive it is, and a short burst is normal.
  const result = feed('Let me run. '.repeat(5), CONFIG, 4)
  assert.equal(result.fired, false)
  const nearly = new SentenceUnitCounter(8)
  nearly.push('Let me run. '.repeat(63))
  assert.equal(nearly.total, 63, 'one unit short of the floor')
  assert.ok(nearly.novelty <= 0.05, 'and it would have fired if the floor were lower')
})

/* -------------------------------------------------------------------------- */
/* configuration                                                              */
/* -------------------------------------------------------------------------- */

test('`minReasoningNoveltyRatio: 0` disables the density rule', () => {
  const result = feed(CYCLED, { minReasoningNoveltyRatio: 0 })
  assert.equal(result.fired, false, 'the rule is off')
  assert.equal(result.at, -1)
})

test('the density rule and the shape rule switch off independently', () => {
  const cycled = feed(CYCLED, {
    minReasoningNoveltyRatio: 0,
    maxRepeatedReasoningCycleChars: 64,
    minRepeatedReasoningCycleChars: 256,
  })
  assert.equal(cycled.fired, false, 'neither rule may cut a cycle past the period cap')

  const short = feed('好。'.repeat(4000), {
    minReasoningNoveltyRatio: 0,
    maxRepeatedReasoningCycleChars: 64,
    minRepeatedReasoningCycleChars: 256,
  })
  assert.equal(short.fired, true, 'the shape rule still catches the short-unit bleed')
  assert.equal(short.by, 'repeating-reasoning')
})

test('the schema ships the density rule on, with its documented defaults', () => {
  const resolved = plugin.Config({})
  assert.equal(resolved.minReasoningSentenceChars, 8)
  assert.equal(resolved.minReasoningSentences, 64)
  assert.equal(resolved.minReasoningNoveltyRatio, 0.05)
  assert.equal(plugin.Config({ minReasoningNoveltyRatio: 0 }).minReasoningNoveltyRatio, 0, '0 must survive as the off switch')
})

test('the schema rejects thresholds that would make the rule fire on everything', () => {
  assert.throws(() => plugin.Config({ minReasoningSentenceChars: 1 }))
  assert.throws(() => plugin.Config({ minReasoningSentences: 1 }))
  assert.throws(() => plugin.Config({ minReasoningNoveltyRatio: -1 }))
  assert.throws(() => plugin.Config({ minReasoningNoveltyRatio: 2 }))
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
  return { steered, steer: m => steered.push(m), inject: () => {}, cancel: () => {} }
}

/** A provider stream whose thinking bleeds and never ends on its own. */
async function* bleedingStream(text, delta) {
  yield { type: 'block-start', index: 0, blockType: 'reasoning' }
  for (let i = 0; i < text.length; i += delta) {
    yield { type: 'reasoning-delta', index: 0, text: text.slice(i, i + delta) }
  }
  yield { type: 'reasoning-end', index: 0 }
  yield { type: 'block-start', index: 1, blockType: 'text' }
  yield { type: 'text-delta', index: 1, text: 'Here is the answer.' }
  yield { type: 'finish', reason: { kind: 'stop' } }
}

test('a call that runs out of material is cut, naming the reasoning channel', async () => {
  const agent = fakeAgent()
  const ctx = fakeContext(agent)
  plugin.apply(ctx, CONFIG)
  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  const out = []
  for await (const chunk of ctx.fire(options, () => bleedingStream(CYCLED, 64))) out.push(chunk)

  const finish = out.at(-1)
  assert.equal(finish.type, 'finish', 'the last chunk must be terminal')
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, 'REPETITIVE_OUTPUT')
  assert.match(finish.reason.failure.message, /reasoning/)
  assert.equal(agent.steered.length, 1, 'the correction must be queued before the error finish')
  assert.match(agent.steered[0].content.map(b => b.text).join(''), /reasoning/)
})

test('the trailing answer is dropped, not flushed', async () => {
  const ctx = fakeContext(fakeAgent())
  plugin.apply(ctx, CONFIG)
  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  const out = []
  for await (const chunk of ctx.fire(options, () => bleedingStream(CYCLED, 64))) out.push(chunk)
  assert.equal(out.filter(c => c.type === 'text-delta').length, 0)
  assert.equal(out.filter(c => c.type === 'finish').length, 1)
})

test('the density cut is protocol-legal with a reasoning block still open', async () => {
  // The sibling rule's test covers the same wrapper; this one runs the DENSITY
  // path through the shipped validator, because a rule is only shipped if the
  // branch it exits through is legal.
  const { apply } = await import('@deepseek-ai/dsh-llm/invariant')
  const listeners = []
  const ctx = {
    on(name, listener, options) {
      if (name !== 'llm/stream') return
      if (options?.prepend) listeners.unshift(listener)
      else listeners.push(listener)
    },
    logger: { warn() {}, debug() {} },
    agents: { get: () => fakeAgent() },
    get: () => undefined,
  }
  const failures = []
  const inner = new Proxy(ctx, { get: (t, k) => (k === 'on' ? t.on.bind(t) : t[k]) })
  await apply({ ...inner, invariants: { register: (_n, installer) => installer(inner, m => failures.push(m)) } }, m => failures.push(m))
  plugin.apply(ctx, CONFIG)
  assert.equal(listeners.length, 2, 'the chain must be invariant -> plugin')

  const options = markAgentLoopRequest({ sessionId: 's1', provider: 'p', model: 'm', messages: [] })
  async function* source() {
    yield { type: 'block-start', index: 0, blockType: 'reasoning' }
    // Long enough for the ratio to fall under the threshold: the twenty-sentence
    // cycle needs ~400 units, because until then the twenty distinct sentences
    // still outnumber the repeats by too much.
    for (let i = 0; i < 24000; i += 64) yield { type: 'reasoning-delta', index: 0, text: CYCLED.slice(i, i + 64) }
  }
  const step = i => (i >= listeners.length ? source() : listeners[i](options, () => step(i + 1)))
  const seen = []
  for await (const chunk of step(0)) seen.push(chunk)

  assert.deepEqual(failures, [], 'the synthesized finish must not violate the stream grammar')
  assert.equal(seen.at(-1).type, 'finish')
  assert.equal(seen.at(-1).reason.kind, 'error')
  assert.ok(seen.some(c => c.type === 'reasoning-delta'), 'the fixture must actually have emitted a reasoning block')
})
