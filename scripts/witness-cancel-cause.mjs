/**
 * Run the **published** agent-loop's own cancel-cause reader against this
 * plugin's cause, on a real release line.
 *
 * `test/cancel-cause.spec.mjs` pins the object this plugin emits; it cannot call
 * the reader itself, because the package that owns it
 * (`@deepseek-ai/dsh-agent-loop`) is not a dependency here. This script closes
 * that gap by extracting `abortedCancelCause` **out of the published
 * `lib/index.js`** — not re-typing it — and executing it, so a drift in the real
 * function is measured rather than assumed.
 *
 * Why it matters: that function's last branch is `default: return
 * assertNever(cause)`, and it is called from inside the turn loop's `catch`. A
 * cause whose `kind` is outside the closed union makes it throw *there*, so the
 * `turnEnds` assignment never runs and the `finally` appends `turn/end` with
 * `reason: null` — the record that blanks the client's session list on every
 * session (#8832, #9156). One measured `null` is worth more than any argument
 * about it.
 *
 * Usage:
 *   node scripts/witness-cancel-cause.mjs                      # the affected lines
 *   node scripts/witness-cancel-cause.mjs --line=0.2.0-rc.2     # one line
 *
 * Network is required (each line is `npm pack`ed into a scratch directory). The
 * package is not installed anywhere, only unpacked, so a witness run cannot
 * change what the suite resolves.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { cancelCauseFor } from '../lib/index.js'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))

/** Lines to measure when none is named: the affected range plus its predecessor. */
const DEFAULT_LINES = ['0.1.6-alpha.2', '0.1.7-rc.2', '0.2.0-rc.2', '0.2.1-alpha.2']

const ANCHOR = 'function abortedCancelCause('

/**
 * The published build of one package, unpacked.
 *
 * `npm pack` rather than `npm install`: nothing here should be resolvable by the
 * suite afterwards, and a witness that mutates the tree it measures is how a
 * "green" run stops meaning anything.
 *
 * @param name - the package to unpack.
 * @param line - the published version.
 * @param scratch - a scratch directory the caller removes.
 * @returns the unpacked package directory.
 */
function unpack(name, line, scratch) {
  const dir = join(scratch, `${name.split('/').pop()}-${line}`)
  execFileSync('mkdir', ['-p', dir])
  execFileSync('npm', ['pack', `${name}@${line}`, '--silent'], { cwd: dir, stdio: 'ignore' })
  const tarball = readdirSync(dir).find(entry => entry.endsWith('.tgz'))
  if (tarball === undefined) throw new Error(`${name}@${line}: npm pack produced no tarball`)
  execFileSync('tar', ['xzf', join(dir, tarball), '-C', dir])
  return join(dir, 'package')
}

/**
 * `abortedCancelCause` as the line actually shipped it.
 *
 * @param line - the published version to unpack.
 * @param scratch - a scratch directory the caller removes.
 * @returns the function source and the line's own `assertNever`, or `undefined`
 *   when the build has no such anchor.
 */
async function shippedReader(line, scratch) {
  const pkg = unpack('@deepseek-ai/dsh-agent-loop', line, scratch)
  const bundle = join(pkg, 'lib', 'index.js')
  if (!existsSync(bundle)) throw new Error(`${line}: the artifact has no lib/index.js`)
  const source = readFileSync(bundle, 'utf8')
  const start = source.indexOf(ANCHOR)
  // An absent anchor is a finding, not a pass: builds before the 0.1.7 line wrote
  // the abort reason verbatim (`reason: signal.reason`) and had no reader at all.
  if (start === -1) return undefined
  // The reader calls `assertNever`, which the line re-exports from its own
  // `dsh-util-values`; importing that copy keeps the witness executed against the
  // measured line rather than against this repo's resolution of the name.
  const utils = unpack('@deepseek-ai/dsh-util-values', line, scratch)
  const { assertNever } = await import(pathToFileURL(join(utils, 'lib', 'index.js')).href)
  if (typeof assertNever !== 'function') throw new Error(`${line}: dsh-util-values exports no assertNever`)
  let depth = 0
  for (let i = source.indexOf('{', start); i < source.length; i++) {
    if (source[i] === '{') depth++
    else if (source[i] === '}') {
      depth--
      if (depth === 0) return { source: source.slice(start, i + 1), assertNever }
    }
  }
  throw new Error(`${line}: unbalanced braces after ${ANCHOR}`)
}

/**
 * What the loop would persist for one cause, using the line's own reader.
 *
 * @param read - the extracted `abortedCancelCause`.
 * @param cause - what `agent.cancel()` handed the signal.
 * @returns `{ reason }` — the value the `finally` appends, or the throw's message.
 */
function persistedReason(read, cause) {
  let turnEnds = null
  try {
    const readBack = read({ aborted: true, reason: cause })
    if (readBack !== undefined) turnEnds = { kind: 'aborted', reason: readBack }
    else turnEnds = { kind: 'error', error: { message: 'unexpected', code: 'UNKNOWN' } }
  } catch (error) {
    // The real loop rethrows from the catch and appends in the finally; the
    // append is what this witness is about, so both outcomes are reported.
    return { threw: error.message, reason: turnEnds }
  }
  return { reason: turnEnds }
}

const scratch = mkdtempSync(join(tmpdir(), 'witness-cancel-cause-'))
const named = process.argv.find(arg => arg.startsWith('--line='))?.slice('--line='.length)
const lines = named === undefined ? DEFAULT_LINES : [named]
let failures = 0

for (const line of lines) {
  const shipped = await shippedReader(line, scratch)
  if (shipped === undefined) {
    console.log(`ABSENT  ${line.padEnd(16)} no abortedCancelCause in this build — the abort reason was written verbatim, so a bad cause cannot become null here`)
    continue
  }
  const read = new Function('assertNever', `return (${shipped.source})`)(shipped.assertNever)
  const ours = cancelCauseFor({ cancelCause: 'thinking-loop' })

  const bare = persistedReason(read, 'thinking-loop')
  const fixed = persistedReason(read, ours)
  const user = persistedReason(read, { kind: 'user' })

  const bareIsNull = bare.reason === null && typeof bare.threw === 'string'
  const fixedIsValid = fixed.reason?.kind === 'aborted'
    && fixed.reason.reason?.kind === 'hook'
    && fixed.reason.reason.reason === 'thinking-loop'
  const userIsValid = user.reason?.kind === 'aborted' && user.reason.reason?.kind === 'user'

  if (!bareIsNull) failures++
  if (!fixedIsValid) failures++
  if (!userIsValid) failures++

  console.log(`${bareIsNull && fixedIsValid && userIsValid ? 'PASS  ' : 'FAIL  '}${line.padEnd(16)}`
    + ` bare string -> reason ${bare.reason === null ? 'null' : JSON.stringify(bare.reason)}`
    + ` (${bare.threw ?? 'no throw'}) |`
    + ` this plugin -> ${fixedIsValid ? JSON.stringify(fixed.reason) : `UNEXPECTED ${JSON.stringify(fixed)}`} |`
    + ` {kind:'user'} -> ${userIsValid ? 'ok' : `UNEXPECTED ${JSON.stringify(user)}`}`)
}

rmSync(scratch, { recursive: true, force: true })
console.log(`\n${failures === 0
  ? 'every measured line behaved as expected: the bare string still yields a null reason, this plugin\'s cause does not'
  : `${failures} arm(s) differ — re-read the line before trusting this witness`}`)
process.exit(failures === 0 ? 0 : 1)
