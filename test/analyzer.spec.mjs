/**
 * Fixture tests for the offline session analyzer (tools/analyze-session.mjs).
 *
 * The analyzer exists so issue #1's reporter can answer two questions from a
 * session file instead of re-running the model. That is only trustworthy if it
 * parses BOTH durable attempt formats and reproduces the detector's own verdict,
 * so each format gets a fixture here:
 *
 *   - v1 `assistant/chunk` (dsh <= 0.1.2-rc.1) — one event per chunk;
 *   - v2 `assistant/attempt` (dsh >= 0.1.5) — compacted `stream` records.
 *
 * Run through `node --test` so CI-less consumers cannot silently break the tool.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ANALYZER = fileURLToPath(new URL('../tools/analyze-session.mjs', import.meta.url))
const TOOL = fileURLToPath(new URL('../lib/index.js', import.meta.url))

/** Long enough to clear the analyzer's `--min-chars 8` test override. */
const STALLED = 'We must reconsider whether the retry budget is the true root cause of the failure here.'
const FRESH = 'The socket closed after the peer wrote a partial frame, so the reader saw an incomplete message.'

/**
 * Scratch directories created by this file, removed on process exit.
 *
 * `mkdtempSync` alone leaks one directory per call, and this suite calls it once
 * per test — a full-suite run left 164 of them behind in the OS tempdir before
 * this was added. `process.on('exit')` (not `after()`) is deliberate: it still
 * runs when a test fails or the runner bails, which is exactly when the leak
 * would otherwise accumulate unnoticed.
 */
const scratchDirs = []
process.on('exit', () => {
  for (const dir of scratchDirs) {
    try { rmSync(dir, { recursive: true, force: true }) } catch { /* best effort */ }
  }
})

/** Create a tracked scratch directory. Every temp dir MUST come from here. */
function scratchDir() {
  const dir = mkdtempSync(join(tmpdir(), 'tlg-analyzer-'))
  scratchDirs.push(dir)
  return dir
}

function writeSession(lines) {
  const dir = scratchDir()
  const file = join(dir, 'session.jsonl')
  writeFileSync(file, lines.map(line => JSON.stringify(line)).join('\n') + '\n')
  return file
}

function run(file, extra = []) {
  const out = execFileSync(process.execPath, [ANALYZER, file, '--json', '--min-chars', '8', '--threshold', '2', ...extra], { encoding: 'utf8' })
  return JSON.parse(out)
}

function runText(file, extra = []) {
  return execFileSync(process.execPath, [ANALYZER, file, '--min-chars', '8', '--threshold', '2', ...extra], { encoding: 'utf8' })
}

test('analyzer reads v1 assistant/chunk events and groups them by (turn, step)', () => {
  const chunk = (turn, step, data) => ({ type: 'assistant/chunk', seq: 0, time: 0, data: { turn, step, chunk: data } })
  const file = writeSession([
    // step 1: reasoning only (no text) -> reasoning-only
    chunk(0, 0, { type: 'reasoning-delta', index: 0, text: STALLED }),
    chunk(0, 0, { type: 'finish', reason: 'stop' }),
    // step 2: same reasoning again, but this time with text -> repeated-material
    chunk(0, 1, { type: 'reasoning-delta', index: 0, text: STALLED }),
    chunk(0, 1, { type: 'text-delta', index: 1, text: 'still thinking' }),
    chunk(0, 1, { type: 'finish', reason: 'stop' }),
    { type: 'user/message', seq: 0, time: 0, data: { message: { id: 'm1', role: 'user', content: [] }, turn: 0, step: 2 } },
  ])
  const { steps } = run(file)
  assert.equal(steps.length, 2, 'two model calls')
  assert.equal(steps[0].verdict, 'reasoning-only')
  assert.equal(steps[0].hasOutput, false)
  assert.equal(steps[1].hasOutput, true)
  assert.equal(steps[1].verdict, 'repeated-material')
  // threshold 2: the second stalled step fires
  assert.equal(steps[1].fired, 'repeated-material')
})

test('analyzer reads v2 assistant/attempt events with compacted stream records', () => {
  const attempt = (turn, step, stream) => ({ type: 'assistant/attempt', seq: 0, time: 0, data: { turn, step, stream } })
  const file = writeSession([
    attempt(0, 0, [{ type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] }]),
    attempt(0, 1, [
      { type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] },
      { type: 'text-chunks', time0: 1, texts: [[1, 'still thinking']] },
    ]),
    attempt(0, 2, [
      { type: 'reasoning-chunks', time0: 0, texts: [[0, FRESH]] },
      { type: 'chunk', time: 1, chunk: { type: 'tool-call-delta', index: 0, id: 'c1', argumentsDelta: '{}' } },
    ]),
  ])
  const { steps } = run(file)
  assert.equal(steps.length, 3)
  assert.equal(steps[0].verdict, 'reasoning-only')
  assert.equal(steps[1].verdict, 'repeated-material')
  // Fresh material with real output is progress and ends the run.
  assert.equal(steps[2].verdict, 'progress')
  assert.equal(steps[2].textChars, '{}'.length)
})

test('analyzer survives unrelated and malformed lines without losing the steps', () => {
  const dir = scratchDir()
  const file = join(dir, 'session.jsonl')
  writeFileSync(file, [
    '{"type":"turn/start","seq":0,"time":0,"data":{"turn":0}}',
    'not json at all',
    JSON.stringify({ type: 'assistant/attempt', seq: 1, time: 1, data: { turn: 0, step: 0, stream: [{ type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] }] } }),
    '',
    JSON.stringify({ type: 'turn/end', seq: 2, time: 2, data: { turn: 0 } }),
  ].join('\n') + '\n')
  const { steps } = run(file)
  assert.equal(steps.length, 1)
  assert.equal(steps[0].verdict, 'reasoning-only')
})

test('an unknown stream record type is skipped rather than guessed at', () => {
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'future-record-kind', texts: [[0, STALLED]] },
      { type: 'reasoning-chunks', time0: 1, texts: [[1, STALLED]] },
    ] } },
    { type: 'assistant/attempt', seq: 1, time: 1, data: { turn: 0, step: 1, stream: [
      { type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] },
      { type: 'text-chunks', time0: 1, texts: [[1, 'text']] },
    ] } },
  ])
  const { steps } = run(file)
  assert.equal(steps[0].verdict, 'reasoning-only')
  assert.equal(steps[1].verdict, 'repeated-material')
})

test('analyzer reports no steps for a session with no assistant events', () => {
  const file = writeSession([{ type: 'turn/start', seq: 0, time: 0, data: { turn: 0 } }])
  const { steps } = run(file)
  assert.deepEqual(steps, [])
})

test('analyzer verdicts agree with the shipped LoopDetector for the same input', async () => {
  const { LoopDetector } = await import(TOOL)
  const config = { maxThinkingSteps: 2, minReasoningChars: 8, repeatRatio: 0.5, similarityThreshold: 0.8, maxFires: 4 }
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [{ type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] }] } },
    { type: 'assistant/attempt', seq: 1, time: 1, data: { turn: 0, step: 1, stream: [
      { type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] },
      { type: 'text-chunks', time0: 1, texts: [[1, 'x']] },
    ] } },
  ])
  const analyzed = run(file)
  const detector = new LoopDetector(config)
  const direct = [
    detector.observe({ hasOutput: false, reasoning: STALLED }),
    detector.observe({ hasOutput: true, reasoning: STALLED }),
  ]
  assert.deepEqual(analyzed.steps.map(s => s.verdict), direct.map(r => r ?? 'progress'))
})

/* -------------------------------------------------------------------------- */
/* the issue #2848 shape: one call that repeats itself forever                */
/* -------------------------------------------------------------------------- */

/**
 * The reported session had ONE model call streaming ~2825 identical text
 * chunks (~420,000 characters). It never finished, so no per-call verdict can
 * describe it — the only useful question is whether the intra-call breaker would
 * have stopped it, which is what `repeatedRun`/`wouldBreak` report.
 */
test('analyzer flags a single call that repeats one sentence to the end', () => {
  const sentence = 'The `register` API matches. '
  const texts = Array.from({ length: 300 }, (_, i) => [i, sentence])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 20, stream: [
      { type: 'text-chunks', time0: 0, texts },
      { type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] },
    ] } },
  ])
  const { steps } = run(file, ['--max-repeated-text', '60'])
  assert.equal(steps.length, 1)
  assert.equal(steps[0].repeatedRun, 300, 'the whole call is one identical-chunk run')
  assert.equal(steps[0].wouldBreak, true)
  assert.equal(steps[0].textChunks, 300)
})

test('analyzer counts only the TRAILING run, so an early repeat is not a break', () => {
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'text-chunks', time0: 0, texts: [[0, 'header'], [1, 'header'], [2, 'header'], [3, 'then real work continued here']] },
    ] } },
  ])
  const { steps } = run(file, ['--max-repeated-text', '3'])
  assert.equal(steps[0].repeatedRun, 1)
  assert.equal(steps[0].wouldBreak, false)
})

test('analyzer leaves a healthy multi-chunk call alone', () => {
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'text-chunks', time0: 0, texts: [[0, 'chunk one'], [1, 'chunk two'], [2, 'chunk three']] },
    ] } },
  ])
  const { steps } = run(file)
  assert.equal(steps[0].repeatedRun, 1)
  assert.equal(steps[0].wouldBreak, false)
})

/* -------------------------------------------------------------------------- */
/* the discussion #7043 shape: a call cycling a few lines instead of working   */
/* -------------------------------------------------------------------------- */

/**
 * The tool answers "would the shipped breaker have cut this call?" — so it must
 * answer it for BOTH rules the plugin ships. This fixture is the reported shape
 * exactly: `好。 / 发。 / 好。 / 好。` cycled for tens of lines, where a tool call
 * belonged. Its identical-delta run is 2, which is why the chunk rule alone
 * reported "nothing to see" on the very session the reporter would analyze.
 */
test('analyzer flags a call cycling a few short lines, which the chunk rule misses', () => {
  const cycle = '好。\n发。\n好。\n好。\n'
  const texts = cycle.repeat(40).split(/(?<=\n)/).filter(s => s !== '').map((text, i) => [i, text])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 4, step: 9, stream: [
      { type: 'text-chunks', time0: 0, texts },
    ] } },
  ])
  const { steps } = run(file)
  assert.equal(steps.length, 1)
  assert.equal(steps[0].repeatedRun, 2, 'the chunk rule cannot see a cycle')
  assert.ok(steps[0].cycleSpan >= 256, `the cycle rule must see it, got ${steps[0].cycleSpan}`)
  assert.equal(steps[0].wouldBreak, true)
  assert.equal(steps[0].wouldBreakBy, 'repeating-cycle')
})

test('analyzer reads the same shape from the v1 assistant/chunk format', () => {
  const cycle = '好。\n发。\n好。\n好。\n'
  const lines = cycle.repeat(40).split(/(?<=\n)/).filter(s => s !== '')
    .map((text, i) => ({ type: 'assistant/chunk', seq: i, time: 0, data: { turn: 0, step: 0, chunk: { type: 'text-delta', index: 0, text } } }))
  const { steps } = run(writeSession(lines))
  assert.equal(steps[0].wouldBreakBy, 'repeating-cycle')
})

test('analyzer does not flag a legitimately repetitive but healthy call', () => {
  // The false-positive control at the tool level too: generated CSS rows look
  // periodic to a coverage measure and are not periodic.
  const texts = Array.from({ length: 40 }, (_, i) => [i, `.row-${i} { display: flex; align-items: center; gap: 8px; }\n`])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [{ type: 'text-chunks', time0: 0, texts }] } },
  ])
  const { steps } = run(file)
  assert.equal(steps[0].cycleSpan, 0)
  assert.equal(steps[0].wouldBreak, false)
})

/* -------------------------------------------------------------------------- */
/* the discussion #9524 shape: a call whose THINKING collapses onto one unit   */
/* -------------------------------------------------------------------------- */

/**
 * The tool's contract is "the same decision rule the plugin runs" — so a rule
 * the plugin ships and this tool does not answer for would make the tool quietly
 * wrong about the sessions it is run on. Discussion #9524's reporters have the
 * session files; these fixtures are the shape they would run it against.
 */
test('analyzer flags a call whose reasoning collapses onto one short sentence', () => {
  // 200 deltas of the 2-character unit = 400 characters: past the 256 span.
  const texts = Array.from({ length: 200 }, (_, i) => [i, '好。'])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 7, step: 3, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const { steps } = run(file)
  assert.equal(steps.length, 1, 'the reasoning deltas must be parsed, not skipped')
  assert.ok(steps[0].reasoningCycleSpan >= 256, `the reasoning rule must see it, got ${steps[0].reasoningCycleSpan}`)
  assert.equal(steps[0].wouldBreak, true)
  assert.equal(steps[0].wouldBreakBy, 'repeating-reasoning')
})

test('analyzer flags a single-character reasoning bleed the visible rule cannot', () => {
  // Period 1, which is the whole reason the reasoning rule carries its own
  // function: the visible rule is silent on this by design.
  const texts = Array.from({ length: 400 }, (_, i) => [i, '好'])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const { steps } = run(file)
  assert.ok(steps[0].reasoningCycleSpan >= 256, `got ${steps[0].reasoningCycleSpan}`)
  assert.equal(steps[0].wouldBreakBy, 'repeating-reasoning')
})

test('analyzer reads the reasoning shape from the v1 assistant/chunk format too', () => {
  const lines = Array.from({ length: 400 }, (_, i) => ({
    type: 'assistant/chunk', seq: i, time: 0,
    data: { turn: 0, step: 0, chunk: { type: 'reasoning-delta', index: 0, text: 'Go. ' } },
  }))
  const { steps } = run(writeSession(lines))
  assert.ok(steps[0].reasoningCycleSpan >= 256, `got ${steps[0].reasoningCycleSpan}`)
  assert.equal(steps[0].wouldBreakBy, 'repeating-reasoning')
  assert.equal(steps[0].textChars, 0, 'the reasoning channel must not be read as visible output')
})

test('analyzer does not flag reasoning that restates itself without becoming periodic', () => {
  const texts = Array.from({ length: 200 }, (_, i) => [i, `第 ${i} 步：我需要先读文件，再判断是否需要修改。\n`])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const { steps } = run(file)
  assert.equal(steps[0].reasoningCycleSpan, 0)
  assert.equal(steps[0].wouldBreak, false)
})

test('the reasoning rule can be switched off for the run, independently', () => {
  // 200 deltas of the 2-character unit = 400 characters: past the 256 span.
  const texts = Array.from({ length: 200 }, (_, i) => [i, '好。'])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const { steps } = run(file, ['--max-repeated-reasoning-cycle', '0'])
  assert.equal(steps[0].reasoningCycleSpan, 0)
  assert.equal(steps[0].wouldBreak, false)
  // …and the span minimum is a real knob, not a second off switch.
  const { steps: tight } = run(file, ['--min-repeated-reasoning-cycle', '512'])
  assert.equal(tight[0].reasoningCycleSpan, 0)
})

test('analyzer flags a call that cycles its sentences past the period cap', () => {
  // Discussion #9683's block: twenty sentences, period 489 characters. The shape
  // rule cannot see it at all (its period cap is 64, so no tail is ever
  // periodic); the density rule can. Both numbers are asserted, because the
  // point of shipping a second statistic is that the first one misses.
  const sentences = [
    'Let me write the file.', 'Let me run the command.', 'I will write the file now.',
    'Let me check the output.', 'Let me run it again.', 'I need to write this down.',
    'Let me apply the patch.', 'Let me verify the result.', 'Let me run the tests.',
    'I will write the next one.', 'Let me inspect the file.', 'Let me run the command now.',
    'Let me update the file.', 'Let me check the logs.', 'I will run the command.',
    'Let me write the answer.', 'Let me confirm the change.', 'Let me run the build.',
    'Let me write it out.', 'Let me finish the step.',
  ]
  const texts = Array.from({ length: 700 }, (_, i) => [i, `${sentences[i % 20]} `])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 4, step: 1, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const { steps } = run(file)
  assert.equal(steps[0].reasoningCycleSpan, 0, 'no tail is periodic — this is the shape the cycle rule misses')
  assert.equal(steps[0].wouldBreak, true)
  assert.equal(steps[0].wouldBreakBy, 'reasoning-novelty')
  assert.ok(steps[0].reasoningNovelty <= 0.05, `got ${steps[0].reasoningNovelty}`)
})

test('analyzer reports the density it measured even when the rule does not fire', () => {
  const texts = Array.from({ length: 700 }, (_, i) => [i, `第 ${i} 步：我需要先读文件，再判断是否需要修改。\n`])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const { steps } = run(file)
  assert.equal(steps[0].wouldBreak, false)
  assert.equal(steps[0].reasoningNovelty, 1, 'every unit is new material')
})

test('the density rule can be switched off for the run, independently', () => {
  const sentences = ['Let me write the file.', 'Let me run the command.', 'Let me check the output.']
  const texts = Array.from({ length: 700 }, (_, i) => [i, `${sentences[i % 3]} `])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const { steps } = run(file, ['--min-reasoning-novelty', '0'])
  assert.equal(steps[0].wouldBreak, false, 'the density rule is off')
  // …and the off switch is the only way it stops answering: the same file with
  // the rule on is a break.
  assert.equal(run(file).steps[0].wouldBreakBy, 'reasoning-novelty')
})

test('the text report names the density column and the rule it counts', () => {
  const sentences = ['Let me write the file.', 'Let me run the command.', 'Let me check the output.']
  const texts = Array.from({ length: 700 }, (_, i) => [i, `${sentences[i % 3]} `])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const out = runText(file)
  assert.match(out, /novelty/, 'the column must be in the table header')
  assert.match(out, /1 by exhausted material in the reasoning/)
  assert.match(out, /BREAK\(reasoning-novelty\)/)
})

test('the text report names the reasoning column and the rule it counts', () => {
  const texts = Array.from({ length: 200 }, (_, i) => [i, '好。'])
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'reasoning-chunks', time0: 0, texts },
    ] } },
  ])
  const out = runText(file)
  assert.match(out, /reasonCycle/, 'the column must be in the table header')
  assert.match(out, /1 by a repeating cycle in the reasoning/)
  assert.match(out, /BREAK\(repeating-reasoning\)/)
})

/* -------------------------------------------------------------------------- */
/* issue #1, 2026-09-20: a silent zero is not a verdict                       */
/* -------------------------------------------------------------------------- */

/**
 * The reporter ran this tool on a flattened dump, got `stalled 0/161`, and had
 * to establish BY HAND that the reasoning behind those turns sits in lines the
 * tool never reads. The tool's own docstring said unknown input degrades to
 * "fewer steps observed" — but nothing in the output said how many fewer, so a
 * zero from an unreadable file and a zero from a healthy file printed the same.
 * These tests pin the disclosure: the counts, the scope note, and the warning.
 */
test('coverage names the skipped event types instead of silently answering zero', () => {
  const file = writeSession([
    { type: 'turn/start', seq: 0, time: 0, data: { turn: 0 } },
    { type: 'assistant/attempt', seq: 1, time: 1, data: { turn: 0, step: 0, stream: [{ type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] }] } },
    { type: 'user/message', seq: 2, time: 2, data: { turn: 0, step: 1 } },
    // A flattened dump: stream records hoisted to top-level lines, which carry
    // no (turn, step) and so cannot be attributed to a call.
    { type: 'reasoning-chunks', time0: 3, index: 0, dt: [0], texts: ['some reasoning'] },
    { type: 'reasoning-chunks', time0: 4, index: 0, dt: [0], texts: ['more reasoning'] },
  ])
  const { coverage, steps } = run(file)
  assert.equal(steps.length, 1, 'only the attempt is a readable step')
  assert.equal(coverage.lines, 6, 'five records plus the trailing newline')
  assert.equal(coverage.parsed, 1)
  assert.deepEqual(coverage.parsedByEventType, { 'assistant/attempt': 1 })
  assert.equal(coverage.skipped, 4)
  assert.equal(coverage.skippedByType['reasoning-chunks'], 2)
  assert.equal(coverage.skippedByType['turn/start'], 1)
  assert.equal(coverage.skippedByType['user/message'], 1)
  assert.equal(coverage.verdictScopeIncomplete, true, 'stream-shaped lines were skipped')
})

test('coverage warns in the text output and scopes the stalled count', () => {
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [{ type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] }] } },
    { type: 'reasoning-chunks', time0: 1, index: 0, dt: [0], texts: ['unreadable reasoning'] },
  ])
  const out = runText(file)
  assert.match(out, /coverage: \d+ line\(s\) — 1 parsed \(assistant\/attempt 1\), 1 skipped/)
  assert.match(out, /skipped by type: reasoning-chunks 1/)
  assert.match(out, /⚠ reasoning-chunks \(1\) look like assistant stream content and are NOT in the verdict/)
  assert.match(out, /over the 1 step\(s\) this tool could read/)
})

test('coverage stays quiet when the file is fully readable', () => {
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [{ type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] }] } },
  ])
  const { coverage } = run(file)
  assert.equal(coverage.skipped, 0)
  assert.equal(coverage.unparsable, 0)
  assert.equal(coverage.verdictScopeIncomplete, false)
  const out = runText(file)
  assert.match(out, /coverage: \d+ line\(s\) — 1 parsed \(assistant\/attempt 1\), 0 skipped/)
  assert.ok(!out.includes('⚠'), 'no warning without stream-shaped skips')
})

test('benign skipped types do not raise the stream-content warning', () => {
  // Skipping conversation bookkeeping is normal; only names that claim to carry
  // assistant stream content make the verdict's scope incomplete.
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [{ type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] }] } },
    { type: 'user/message', seq: 1, time: 1, data: { turn: 0, step: 1 } },
    { type: 'tool/result', seq: 2, time: 2, data: { turn: 0, step: 1 } },
  ])
  const { coverage } = run(file)
  assert.equal(coverage.skipped, 2)
  assert.equal(coverage.verdictScopeIncomplete, false)
  assert.ok(!runText(file).includes('⚠'))
})

test('coverage counts records dropped INSIDE a parsed event, not just whole lines', () => {
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'future-record-kind', texts: [[0, STALLED]] },
      { type: 'reasoning-chunks', time0: 1, texts: [[1, STALLED]] },
    ] } },
  ])
  const { coverage } = run(file)
  assert.equal(coverage.skipped, 0, 'the line itself was parsed')
  assert.equal(coverage.unknownRecordTypes['future-record-kind'], 1)
  const out = runText(file)
  assert.match(out, /unrecognised inside parsed events: stream record future-record-kind ×1/)
})

test('analyzer reads the master tool-call-chunks shape, which keys args not texts', () => {
  // 0.1.5/0.1.6 wrote `texts` for tool-call groups; current master writes `args`.
  // Reading only `texts` dropped every tool-argument delta, which flipped
  // `hasOutput` on a call whose only output was a tool call.
  const args = ['{"path":"src/index.ts"', ',"line":42}']
  const file = writeSession([
    { type: 'assistant/attempt', seq: 0, time: 0, data: { turn: 0, step: 0, stream: [
      { type: 'reasoning-chunks', time0: 0, texts: [[0, STALLED]] },
      { type: 'tool-call-chunks', time0: 1, index: 0, id: 'c1', dt: [0, 1], args },
    ] } },
  ])
  const { steps, coverage } = run(file)
  assert.equal(steps[0].hasOutput, true, 'a tool call is output')
  assert.equal(steps[0].textChars, args.join('').length)
  assert.equal(coverage.unknownRecordTypes['tool-call-chunks'], undefined, 'recognised, not dropped')
})

test('an unreadable file says so instead of presenting an empty verdict', () => {
  const file = writeSession([
    { type: 'reasoning-chunks', time0: 0, index: 0, dt: [0], texts: ['flat dump'] },
    { type: 'text-chunks', time0: 1, index: 0, dt: [0], texts: ['flat dump'] },
  ])
  const { steps, coverage } = run(file)
  assert.deepEqual(steps, [])
  assert.equal(coverage.verdictScopeIncomplete, true)
  const out = runText(file)
  assert.match(out, /no assistant steps found/)
  assert.match(out, /check the skipped-type inventory above/)
  assert.match(out, /skipped by type: reasoning-chunks 1, text-chunks 1/)
})
