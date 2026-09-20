#!/usr/bin/env node
/**
 * Offline replay of the thinking-loop guard over a real dsh session file.
 *
 * Issue #1 (jilian-dsh) needs two answers that only their machine has:
 * "did the recurring calls emit text?" and "how similar were the reasoning
 * texts?". Both are answerable from a session jsonl without re-running the
 * model — and the answer is trustworthy only if it is produced by the SAME
 * decision rule the plugin runs, so this script reuses `LoopDetector` from the
 * plugin rather than re-implementing the heuristic (a re-implementation would
 * be a second source of truth, and the whole point is to report what the
 * installed guard would have decided).
 *
 * Usage:
 *   node tools/analyze-session.mjs <session.jsonl> [--similarity 0.8] [--threshold 3]
 *                                [--min-chars 2048] [--max-fires 4] [--json]
 *
 * Reads both durable attempt formats:
 *   - `assistant/chunk` (session format v1: dsh <= 0.1.2-rc.1) — one event per
 *     stream chunk, grouped by (turn, step);
 *   - `assistant/attempt` (session format v2: dsh >= 0.1.5) — one event per
 *     attempt carrying the whole compacted `stream` record array.
 *
 * A "step" is one model call, i.e. one (turn, step) group. For each step it
 * reports whether text/tool output was emitted and what the reasoning was, then
 * feeds the same `StepObservation` the plugin feeds at runtime.
 *
 * It also reports the **intra-call** shape (issue #2848): the trailing run of
 * identical visible-output chunks inside one call, which is the only measure
 * that describes a call repeating itself for minutes without ever ending. That
 * shape is invisible to every per-call verdict on purpose — the call never
 * completes — so it gets its own column. Two columns after it answer "would the
 * shipped breaker have cut this call?" for both rules the plugin ships:
 * `repeatedRun` (identical deltas, `--max-repeated-text`) and `cycleSpan` (the
 * exact verbatim period at the tail, `--max-repeated-cycle`).
 *
 * The cycle rule exists because of discussion #7043: a call bleeding `好。 / 发。 /
 * 好。 / 好。` for tens of lines has a run of identical deltas of 2, so the chunk
 * rule never fires — and the reporter's session file is exactly what this tool
 * is pointed at to answer whether the shipped plugin would have cut it.
 *
 * Every run ends with a **coverage block**: how many lines were parsed, which
 * event types were skipped and how many each, which records were dropped inside
 * parsed events, and a warning when a skipped type's name says it carries
 * assistant stream content. The verdict is scoped to what was read, and the
 * block is what makes that scope visible — `stalled 0/161` on a file whose
 * reasoning sits in lines this tool does not read is not evidence of health, and
 * the tool now says so itself. `--json` carries the same numbers under
 * `coverage`, with `verdictScopeIncomplete` as the machine-readable flag.
 */
import { readFileSync } from 'node:fs'
import { LoopDetector, countRepeatedText, trailingCycle } from '../lib/index.js'

const DEFAULT_CONFIG = {
  maxThinkingSteps: 3,
  minReasoningChars: 2048,
  repeatRatio: 0.5,
  similarityThreshold: 0.8,
  maxFires: 4,
  maxRepeatedText: 60,
  maxRepeatedCycleChars: 64,
  minRepeatedCycleChars: 256,
}

function parseArgs(argv) {
  const config = { ...DEFAULT_CONFIG }
  let file
  let asJson = false
  const flags = { '--similarity': 'similarityThreshold', '--threshold': 'maxThinkingSteps', '--min-chars': 'minReasoningChars', '--max-fires': 'maxFires', '--max-repeated-text': 'maxRepeatedText', '--max-repeated-cycle': 'maxRepeatedCycleChars', '--min-repeated-cycle': 'minRepeatedCycleChars' }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--json') { asJson = true; continue }
    if (flags[arg] !== undefined) {
      const value = Number(argv[++i])
      if (!Number.isFinite(value)) throw new Error(`${arg} expects a number`)
      config[flags[arg]] = value
      continue
    }
    if (arg.startsWith('--')) throw new Error(`unknown flag ${arg}`)
    file = arg
  }
  if (file === undefined) throw new Error('usage: node tools/analyze-session.mjs <session.jsonl> [--json]')
  return { file, config, asJson }
}

/**
 * Per-run parse accounting.
 *
 * Issue #1 (2026-09-20 follow-up) is why this exists. The tool skips what it does
 * not understand — the right call — but it used to do so *silently*, and then
 * print `stalled 0/N` with the same confidence as on a fully-parsed file. A
 * reporter read `stalled 0/161` on a dump whose reasoning lived in lines this
 * tool never looks at, and had to work out by hand that the zero was empty. A
 * zero you cannot distinguish from "nothing was readable" is not a verdict, so
 * every dropped line and record is counted and disclosed below.
 */
function newCoverage() {
  return {
    lines: 0,
    blank: 0,
    unparsable: 0,
    parsed: 0,
    eventTypes: new Map(),
    skippedTypes: new Map(),
    typelessSkipped: 0,
    unknownChunkTypes: new Map(),
    unknownRecordTypes: new Map(),
  }
}

function bump(map, key) {
  map.set(key, (map.get(key) ?? 0) + 1)
}

function skippedTotal(coverage) {
  let total = coverage.typelessSkipped
  for (const count of coverage.skippedTypes.values()) total += count
  return total
}

/**
 * Types whose *name* says they carry assistant stream content. Such a line being
 * skipped means the verdict below is scoped to less than the file contains, so
 * it is worth a warning rather than a line in an inventory nobody reads. Names
 * like `user/message` or `turn/end` are skipped by design and stay quiet.
 */
function looksLikeStreamContent(type) {
  return /chunk|delta|reasoning|stream|attempt/i.test(type)
}

/** True when a chunk is model-authored output (as opposed to reasoning). */
function isOutputChunk(type) {
  return type === 'text-delta' || type === 'tool-call-delta'
}

/** Pull the reasoning/text/tool deltas out of one raw chunk. */
function readChunk(chunk, coverage) {
  if (chunk === null || typeof chunk !== 'object') return undefined
  const type = chunk.type
  if (typeof type !== 'string') return undefined
  if (type === 'reasoning-delta') return { type, text: String(chunk.text ?? '') }
  if (type === 'text-delta') return { type, text: String(chunk.text ?? '') }
  if (type === 'tool-call-delta') return { type, text: String(chunk.argumentsDelta ?? '') }
  // Unknown chunk kinds carry no text this tool can use, but they still use up a
  // position in the call, so they are counted instead of vanishing.
  bump(coverage.unknownChunkTypes, type)
  return { type, text: '' }
}

/**
 * Flatten one v1 `assistant/chunk` event into chunk-like deltas.
 *
 * The durable event carries `data.chunk` with the plugin-facing `StreamChunk`
 * shape, so the mapping is direct.
 */
function deltasFromChunkEvent(event, coverage) {
  const chunk = event?.data?.chunk
  const delta = readChunk(chunk, coverage)
  return delta === undefined ? [] : [delta]
}

/**
 * Flatten one v2 `assistant/attempt` event's compacted stream into deltas.
 *
 * The durable stream is a compacted record array, not raw chunks:
 *   - `{ type: 'chunk', time, chunk }` — one chunk, unchanged;
 *   - `{ type: 'text-chunks' | 'reasoning-chunks', texts: string[] }` and
 *     `{ type: 'tool-call-chunks', args: string[] }` — run-length-compacted
 *     delta groups. The tool-call group is the one that changed key: 0.1.5/0.1.6
 *     wrote `texts` for it, current master writes `args`, so both are read.
 *     Missing the newer key dropped every tool-argument delta, which flipped
 *     `hasOutput` on a call whose only output was a tool call.
 * Unknown record types are ignored rather than guessed at, so a future format
 * addition degrades to "fewer steps observed" instead of a wrong verdict — and
 * `coverage.unknownRecordTypes` says how many were ignored.
 */
function deltasFromAttemptEvent(event, coverage) {
  const stream = event?.data?.stream
  if (!Array.isArray(stream)) return []
  const out = []
  for (const record of stream) {
    if (record === null || typeof record !== 'object') continue
    if (record.type === 'chunk') {
      const delta = readChunk(record.chunk, coverage)
      if (delta !== undefined) out.push(delta)
      continue
    }
    const isToolCallGroup = record.type === 'tool-call-chunks'
    const texts = Array.isArray(record.texts)
      ? record.texts
      : (isToolCallGroup && Array.isArray(record.args) ? record.args : undefined)
    if (texts === undefined) {
      bump(coverage.unknownRecordTypes, typeof record.type === 'string' ? record.type : '(no type)')
      continue
    }
    const type = record.type === 'text-chunks'
      ? 'text-delta'
      : record.type === 'reasoning-chunks'
        ? 'reasoning-delta'
        : isToolCallGroup
          ? 'tool-call-delta'
          : undefined
    if (type === undefined) {
      bump(coverage.unknownRecordTypes, typeof record.type === 'string' ? record.type : '(no type)')
      continue
    }
    for (const entry of texts) {
      const text = Array.isArray(entry) ? entry[1] : entry
      out.push({ type, text: String(text ?? '') })
    }
  }
  return out
}

/** Group the file's events into per-model-call steps, in file order. */
function readSteps(lines, coverage) {
  const steps = []
  const byCoordinate = new Map()
  for (const line of lines) {
    if (line.trim().length === 0) { coverage.blank += 1; continue }
    let event
    try { event = JSON.parse(line) } catch { coverage.unparsable += 1; continue }
    const type = event?.type
    if (type !== 'assistant/chunk' && type !== 'assistant/attempt') {
      // Skipped by design — but disclosed, because "skipped" and "absent" look
      // identical from the summary line alone.
      if (typeof type === 'string') bump(coverage.skippedTypes, type)
      else coverage.typelessSkipped += 1
      continue
    }
    coverage.parsed += 1
    bump(coverage.eventTypes, type)
    // v1 chunk events carry (turn, step) in data; v2 attempts are already one step.
    const turn = event?.data?.turn
    const step = event?.data?.step
    const key = `${turn}/${step}`
    let group = byCoordinate.get(key)
    if (group === undefined) {
      group = { turn, step, reasoning: '', text: '', texts: [], hasOutput: false }
      byCoordinate.set(key, group)
      steps.push(group)
    }
    const deltas = type === 'assistant/chunk' ? deltasFromChunkEvent(event, coverage) : deltasFromAttemptEvent(event, coverage)
    for (const delta of deltas) {
      if (delta.type === 'reasoning-delta') group.reasoning += delta.text
      else if (isOutputChunk(delta.type)) {
        group.hasOutput = true
        group.text += delta.text
        // Only text deltas participate in the intra-call breaker; tool-argument
        // deltas are chunked by the provider's own tokenizer.
        if (delta.type === 'text-delta') group.texts.push(delta.text)
      }
    }
  }
  return steps
}

/** Render the coverage block: what was read, what was skipped, and the warning. */
function renderCoverage(coverage, stepCount) {
  const lines = []
  const parsedBy = [...coverage.eventTypes].map(([type, count]) => `${type} ${count}`).join(', ')
  const skipped = skippedTotal(coverage)
  lines.push(`coverage: ${coverage.lines} line(s) — ${coverage.parsed} parsed`
    + `${parsedBy === '' ? '' : ` (${parsedBy})`}, ${skipped} skipped`
    + `${coverage.unparsable === 0 ? '' : `, ${coverage.unparsable} unparsable`}`
    + `${coverage.blank === 0 ? '' : `, ${coverage.blank} blank`}`)
  if (skipped > 0) {
    const inventory = [...coverage.skippedTypes].map(([type, count]) => `${type} ${count}`)
    if (coverage.typelessSkipped > 0) inventory.push(`(no type) ${coverage.typelessSkipped}`)
    lines.push(`  skipped by type: ${inventory.join(', ')}`)
  }
  const unrecognised = [
    ...[...coverage.unknownChunkTypes].map(([type, count]) => `chunk ${type} ×${count}`),
    ...[...coverage.unknownRecordTypes].map(([type, count]) => `stream record ${type} ×${count}`),
  ]
  if (unrecognised.length > 0) lines.push(`  unrecognised inside parsed events: ${unrecognised.join(', ')}`)
  const suspicious = [...coverage.skippedTypes]
    .filter(([type]) => looksLikeStreamContent(type))
    .map(([type, count]) => `${type} (${count})`)
  if (suspicious.length > 0) {
    lines.push(`  ⚠ ${suspicious.join(', ')} look like assistant stream content and are NOT in the verdict —`
      + ` it covers only the ${stepCount} step(s) below. If your file flattens stream records to`
      + ` top-level lines, the tool cannot attribute them to a (turn, step); it reports them here instead of guessing.`)
  }
  return lines
}

const EMPTY_VERDICT_NOTE = [
  '(no assistant steps found — this tool reads v1 `assistant/chunk` and v2/v3 `assistant/attempt` only.)',
  '  An empty verdict here means those two event types are absent, not that the session was healthy:',
  '  check the skipped-type inventory above, which names what the file does contain.',
]

const { file, config, asJson } = parseArgs(process.argv.slice(2))
const lines = readFileSync(file, 'utf8').split('\n')
const coverage = newCoverage()
coverage.lines = lines.length
const steps = readSteps(lines, coverage)
const coverageReport = renderCoverage(coverage, steps.length)

const detector = new LoopDetector({ pollMs: 0, graceMs: 0, escalate: 'steer', ...config })
const report = []
for (const [index, step] of steps.entries()) {
  const reason = detector.observe({ hasOutput: step.hasOutput, reasoning: step.reasoning })
  const fire = detector.takeFire()
  const repeatRun = countRepeatedText(step.texts)
  // Both rules run on the SAME input the plugin's breaker sees: text deltas
  // only, in stream order. This tool exists to answer "would the shipped breaker
  // have cut this call?", so a rule the plugin has but the tool does not would
  // make the tool quietly wrong.
  const cycleSpan = trailingCycle(step.texts.join(''), config.maxRepeatedCycleChars, config.minRepeatedCycleChars)
  const byChunks = config.maxRepeatedText > 0 && repeatRun >= config.maxRepeatedText
  const byCycle = cycleSpan > 0
  const wouldBreakBy = byChunks ? 'identical-chunks' : (byCycle ? 'repeating-cycle' : null)
  report.push({
    step: index + 1,
    turn: step.turn,
    call: step.step,
    reasoningChars: step.reasoning.length,
    textChars: step.text.length,
    textChunks: step.texts.length,
    hasOutput: step.hasOutput,
    verdict: reason ?? 'progress',
    fired: fire ?? null,
    // The intra-call shape (issue #2848). `repeatedRun` is the trailing run of
    // identical visible-output chunks; `wouldBreak` answers whether the shipped
    // breaker would have ended this call mid-stream.
    repeatedRun: repeatRun,
    cycleSpan,
    wouldBreak: wouldBreakBy !== null,
    wouldBreakBy,
  })
}
const broken = report.filter(r => r.wouldBreak).length

if (asJson) {
  process.stdout.write(`${JSON.stringify({
    file,
    config,
    coverage: {
      lines: coverage.lines,
      parsed: coverage.parsed,
      skipped: skippedTotal(coverage),
      unparsable: coverage.unparsable,
      blank: coverage.blank,
      parsedByEventType: Object.fromEntries(coverage.eventTypes),
      skippedByType: Object.fromEntries(coverage.skippedTypes),
      typelessSkipped: coverage.typelessSkipped,
      unknownChunkTypes: Object.fromEntries(coverage.unknownChunkTypes),
      unknownRecordTypes: Object.fromEntries(coverage.unknownRecordTypes),
      // True when the file contains skipped lines whose names say they carry
      // assistant stream content — i.e. the step list below is scoped to less
      // than the file contains. Read `stalled` only together with this flag.
      verdictScopeIncomplete: [...coverage.skippedTypes.keys()].some(looksLikeStreamContent),
    },
    steps: report,
  }, null, 2)}\n`)
} else {
  console.log(`session: ${file}`)
  console.log(`config:  ${JSON.stringify(config)}`)
  console.log(`steps:   ${steps.length} model call(s)`)
  console.log('')
  console.log('  #   turn/step   reasonChars  textChars  chunks  verdict             fired  repeatedRun  cycleSpan  break')
  for (const row of report) {
    console.log(
      `  ${String(row.step).padStart(2)}  ${String(row.turn)}/${String(row.call)}`.padEnd(20)
      + `${String(row.reasoningChars).padStart(9)}  ${String(row.textChars).padStart(9)}  `
      + `${String(row.textChunks).padStart(6)}  ${row.verdict.padEnd(18)}  ${String(row.fired ?? '').padEnd(5)}  `
      + `${String(row.repeatedRun).padStart(11)}  ${String(row.cycleSpan).padStart(9)}  `
      + `${row.wouldBreak ? `BREAK(${row.wouldBreakBy})` : ''}`,
    )
  }
  const stalls = report.filter(r => r.verdict !== 'progress').length
  const fires = report.filter(r => r.fired !== null).length
  const withText = report.filter(r => r.hasOutput).length
  console.log('')
  console.log(`stalled steps: ${stalls}/${report.length}`
    + `  (over the ${report.length} step(s) this tool could read — see coverage)`
    + `  |  reactions: ${fires}  |  steps that emitted text: ${withText}`)
  const byChunks = report.filter(r => r.wouldBreakBy === 'identical-chunks').length
  const byCycle = report.filter(r => r.wouldBreakBy === 'repeating-cycle').length
  console.log(`intra-call repetition: ${broken} call(s) would be cut mid-stream `
    + `(${byChunks} by identical chunks, maxRepeatedText = ${config.maxRepeatedText}; `
    + `${byCycle} by a repeating cycle, maxRepeatedCycleChars = ${config.maxRepeatedCycleChars}, `
    + `minRepeatedCycleChars = ${config.minRepeatedCycleChars})`)
  if (byCycle === 0 && byChunks === 0 && report.some(r => r.repeatedRun > 1)) {
    const worst = Math.max(...report.map(r => r.repeatedRun))
    console.log(`  (the longest identical-chunk run seen was ${worst}; lower --max-repeated-text to cut such calls)`)
  }
  console.log('')
  for (const line of coverageReport) console.log(line)
  if (report.length === 0) {
    for (const line of EMPTY_VERDICT_NOTE) console.log(line)
  }
}
