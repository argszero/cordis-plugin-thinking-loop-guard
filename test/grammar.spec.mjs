/**
 * The transcription in `./grammar.mjs` is only worth the assertions that use it
 * if it says what the shipped validator said. This file is that check: one
 * corpus, both implementations, same verdicts and same messages.
 *
 * It runs on the lines that still publish `@deepseek-ai/dsh-llm/invariant` — the
 * dev pin does, every claimed line up to `0.2.0-rc.2` does — and it reports
 * itself **skipped**, with the reason, on the lines where upstream removed the
 * module (`f028f25667`, first shipped in `0.2.1-alpha.1`). A skip is not a pass:
 * `scripts/probe-lines.mjs` prints the count of skipped arms beside the pass
 * count for exactly this reason, and the arm's own name says which regime it ran.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { FROZEN, SHIPPED, grammarSource, validateStream } from './grammar.mjs'

const REGIME = await grammarSource()

/** A stream that satisfies every rule: one closed text block, usage, finish. */
const WELL_FORMED = [
  { type: 'block-start', index: 0, blockType: 'text' },
  { type: 'text-delta', index: 0, text: 'hello' },
  { type: 'block-end', index: 0, block: { type: 'text', text: 'hello' } },
  { type: 'usage', usage: { in: 1, out: 1 } },
  { type: 'finish', reason: { kind: 'stop' } },
]

/** One stream per rule, each one violation away from {@link WELL_FORMED}. */
const CORPUS = [
  { name: 'well-formed', chunks: WELL_FORMED },
  { name: 'chunk after terminal finish', chunks: [...WELL_FORMED, { type: 'text-delta', index: 0, text: 'late' }] },
  { name: 'repeated block-start index', chunks: [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'finish', reason: { kind: 'stop' } },
  ] },
  { name: 'delta with no open block', chunks: [
    { type: 'text-delta', index: 0, text: 'hello' },
    { type: 'finish', reason: { kind: 'stop' } },
  ] },
  { name: 'delta against the wrong block type', chunks: [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'text-delta', index: 0, text: 'hello' },
    { type: 'finish', reason: { kind: 'error', failure: { code: 'X', message: 'x' } } },
  ] },
  { name: 'block-end closing a different type', chunks: [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'r' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ] },
  { name: 'block-end with no open block', chunks: [
    { type: 'block-end', index: 3, block: { type: 'text', text: 't' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ] },
  { name: 'negative block index', chunks: [
    { type: 'block-start', index: -1, blockType: 'text' },
    { type: 'finish', reason: { kind: 'stop' } },
  ] },
  { name: 'usage twice', chunks: [
    { type: 'usage', usage: { in: 1, out: 1 } },
    { type: 'usage', usage: { in: 1, out: 1 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ] },
  { name: 'stop finish with an open block', chunks: [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'partial' },
    { type: 'finish', reason: { kind: 'stop' } },
  ] },
  { name: 'error finish with an open block (legal)', chunks: [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'partial' },
    { type: 'finish', reason: { kind: 'error', failure: { code: 'X', message: 'x' } } },
  ] },
  { name: 'aborted finish with an open block (legal)', chunks: [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'thinking' },
    { type: 'finish', reason: { kind: 'aborted' } },
  ] },
  { name: 'stream ending without a finish', chunks: [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'done' } },
  ] },
  { name: 'empty stream', chunks: [] },
]

/** Run one chunk list through a validator and return the messages it recorded. */
async function verdicts(install) {
  const out = []
  for (const entry of CORPUS) {
    const failures = []
    const source = async function* () { for (const chunk of entry.chunks) yield chunk }
    const run = await install(m => failures.push(m), source)
    for await (const _chunk of run) { /* consume: the validator reports as it reads */ }
    out.push({ name: entry.name, failures })
  }
  return out
}

/** The shipped validator, installed through the runtime it asks for. */
async function shipped(fail, source) {
  const { apply } = await import('@deepseek-ai/dsh-llm/invariant')
  let run
  const ctx = {
    on(name, listener) {
      if (name === 'llm/stream') run = listener
    },
    get: () => undefined,
  }
  await apply({ ...ctx, invariants: { register: (_name, installer) => installer(ctx, fail) } }, fail)
  return run({}, source)
}

/** The transcription, called directly — same input, same shape. */
function frozen(fail, source) {
  return validateStream(source(), fail)
}

test('the transcription and the shipped validator agree on every rule', async (t) => {
  if (REGIME === FROZEN) {
    t.skip('this dsh line publishes no @deepseek-ai/dsh-llm/invariant to compare against')
    return
  }
  const fromShipped = await verdicts(shipped)
  const fromFrozen = await verdicts(frozen)

  assert.equal(fromFrozen.length, CORPUS.length)
  for (let i = 0; i < CORPUS.length; i++) {
    assert.deepEqual(
      fromFrozen[i].failures,
      fromShipped[i].failures,
      `"${CORPUS[i].name}": the transcription and the shipped validator must record the same messages`,
    )
  }
  // A corpus where nothing fails would make the comparison above vacuous.
  const violating = fromShipped.filter(entry => entry.failures.length > 0).length
  assert.ok(violating >= 8, `the corpus must exercise the rules, only ${violating} of ${CORPUS.length} entries failed`)
})
