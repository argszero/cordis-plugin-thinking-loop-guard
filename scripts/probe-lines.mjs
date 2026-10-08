/**
 * Re-measure the dsh lines this plugin's peer range claims.
 *
 * The peer range is a *claim about other people's releases*, and the only honest
 * way to hold it is to install each claimed line into its own tree and run the
 * real suite there. Doing that by hand is how v0.1.6 shipped a range that
 * excluded the newest line for two releases — and how v0.1.9 shipped one that
 * ended at `<0.2.0`, admitting `0.2.0-rc.N` only by an accident of prerelease
 * ordering while refusing the stable `0.2.0` outright.
 *
 * Each line gets its own copy of this package inside the OS temp directory, with
 * `node_modules` and `package-lock.json` removed first: a leftover tree is what
 * makes a probe silently measure the previous line. The resolved versions are
 * read back and asserted equal to the requested line before anything is run —
 * npm will happily install a *different* build and report success. What makes
 * that assertion meaningful is pinning the **whole line**, not just the two
 * packages this source imports: `dsh-agent` names its siblings as peers, and a
 * probe that pins only `dsh-agent`/`dsh-llm` leaves npm free to resolve the rest
 * to the newest build inside *their* ranges. Measured 2026-10-08 — that mixture
 * is unrunnable: on the `0.1.5-alpha.1` line npm derived
 * `dsh-system-prompt@0.1.5-rc.3`, whose own peer is `dsh-llm@^0.1.5-rc.3`, and
 * refused the install with `ERESOLVE` against the pinned `dsh-llm@0.1.5-alpha.1`.
 * The line was never broken; the probe was measuring three lines at once.
 *
 * So the peer set is read **off the line's own `dsh-agent`** (it grows: 0.1.5
 * adds `dsh-util-values`, 0.1.7 adds `dsh-workspace`), everything in it is pinned
 * to the line, and the install runs with `--legacy-peer-deps`. That flag is about
 * this probe's tree, not a consumer's: it stops npm from auto-installing *unlisted*
 * transitive peers at their newest build — the same mixing, one level deeper
 * (measured: a whole-line pin without the flag still dies on
 * `dsh-session-persistence@0.1.5-rc.3`, a peer of `dsh-workspace`, which is three
 * packages below anything this plugin imports). With the flag the tree is exactly
 * the line, and `tsc` plus the suite is what judges it. Whether a *consumer* can
 * install this plugin beside their dsh is a different question, and one
 * `test/peer-range.spec.mjs` answers by computing admission from the range.
 *
 * Lines in {@link EXCLUDED} are measured too, and must still fail the way their
 * recorded reason says. Skipping them would make "this line is broken" and "this
 * line was never tried" the same green.
 *
 * Usage: `npm run test:probe-lines` (network + a few minutes), or
 * `node scripts/probe-lines.mjs --line=<version>` to measure one line that is not
 * in the record yet — that measurement is how a line earns a place in either list.
 */
import { execFileSync } from 'node:child_process'
import { cpSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { EXCLUDED, PEER_DEPS, SUPPORTED } from '../test/peer-lines.mjs'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const scratch = mkdtempSync(join(tmpdir(), 'thinking-loop-guard-lines-'))

function run(command, args, cwd) {
  try {
    return { ok: true, out: execFileSync(command, args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) }
  } catch (error) {
    return { ok: false, out: `${error.stdout ?? ''}${error.stderr ?? ''}` }
  }
}

/** A line's `package.json`, as published. */
function manifest(name, line) {
  const viewed = run('npm', ['view', `${name}@${line}`, '--json'], ROOT)
  if (!viewed.ok) return undefined
  try { return JSON.parse(viewed.out) } catch { return undefined }
}

/** The cordis build a line's own `dsh-agent` asks for. */
function cordisSpecFor(line) {
  return manifest('@deepseek-ai/dsh-agent', line)?.peerDependencies?.['@deepseek-ai/cordis'] ?? '^4.0.2'
}

/**
 * Every dsh package a line's own `dsh-agent` names as a peer, plus the two this
 * source imports — the set that has to be pinned together for the tree to be the
 * line rather than a mixture of it.
 */
function linePeers(line) {
  const declared = Object.keys(manifest('@deepseek-ai/dsh-agent', line)?.peerDependencies ?? {})
    .filter((name) => name.startsWith('@deepseek-ai/dsh-'))
  return [...new Set([...declared, ...PEER_DEPS])].sort()
}

/** Install one line into its own copy and report what actually derived. */
function measure(line) {
  const dir = join(scratch, line)
  cpSync(ROOT, dir, {
    recursive: true,
    filter: (source) => !/(?:^|\/)(?:node_modules|\.git|lib)(?:\/|$)/.test(source.slice(ROOT.length)),
  })
  rmSync(join(dir, 'node_modules'), { recursive: true, force: true })
  rmSync(join(dir, 'package-lock.json'), { force: true })

  const peers = linePeers(line)
  const spec = [`@deepseek-ai/cordis@${cordisSpecFor(line)}`, ...peers.map(dep => `${dep}@${line}`)]
  const install = run('npm', ['install', '--no-save', '--no-audit', '--no-fund', '--legacy-peer-deps', ...spec], dir)
  if (!install.ok) return { line, stage: 'install', detail: install.out.trim().split('\n').slice(-3).join(' | ') }

  const resolved = peers.map((dep) => {
    try { return JSON.parse(readFileSync(join(dir, 'node_modules', dep, 'package.json'), 'utf8')).version } catch { return 'absent' }
  })
  const wrong = resolved.filter(version => version !== line)
  if (wrong.length > 0) {
    return { line, stage: 'resolved', detail: `asked for ${line}, got ${wrong.join(' / ')}` }
  }

  // `tsc` may not exist in a bare tree; the package declares it, so the install
  // above provides it. Build first — the suite imports `lib/`.
  const build = run('npx', ['tsc'], dir)
  if (!build.ok) return { line, stage: 'tsc', detail: build.out.trim().split('\n').slice(0, 2).join(' | ') }

  const suite = run('node', ['--test', 'test/source-kind.spec.mjs', 'test/detection.spec.mjs', 'test/text-breaker.spec.mjs', 'test/analyzer.spec.mjs', 'test/inject.spec.mjs', 'test/inject-mechanism.spec.mjs', 'test/peer-range.spec.mjs'], dir)
  const summary = /^ℹ pass (\d+)$/m.exec(suite.out)
  const failed = /^ℹ fail (\d+)$/m.exec(suite.out)
  if (!suite.ok || summary === null || Number(failed?.[1] ?? 1) !== 0) {
    return { line, stage: 'suite', detail: suite.out.trim().split('\n').filter(l => l.startsWith('✖') || l.startsWith('Error')).slice(0, 2).join(' | ') || 'no pass/fail summary' }
  }
  return { line, stage: 'pass', detail: `${summary[1]} tests pass` }
}

const only = process.argv.find(arg => arg.startsWith('--line='))?.slice('--line='.length)
if (only) {
  const result = measure(only)
  console.log(`${result.stage === 'pass' ? 'PASS' : 'FAIL'}  ${only.padEnd(16)} ${result.stage !== 'pass' ? `[${result.stage}] ` : ''}${result.detail}`)
  rmSync(scratch, { recursive: true, force: true })
  process.exit(result.stage === 'pass' ? 0 : 1)
}

let failures = 0
console.log(`probing ${SUPPORTED.length} claimed line(s) + ${EXCLUDED.length} excluded line(s) in ${scratch}\n`)

for (const line of SUPPORTED) {
  const result = measure(line)
  if (result.stage !== 'pass') failures++
  console.log(`${result.stage === 'pass' ? 'PASS' : 'FAIL'}  ${line.padEnd(16)} ${result.stage !== 'pass' ? `[${result.stage}] ` : ''}${result.detail}`)
}

if (EXCLUDED.length > 0) console.log('')
for (const { line, expect, why } of EXCLUDED) {
  const result = measure(line)
  const stage = expect === 'tsc-fails' ? 'tsc' : 'suite'
  const asExpected = result.stage === stage
  if (!asExpected) failures++
  console.log(`${asExpected ? 'OK  ' : 'FAIL'}  ${line.padEnd(16)} expected ${expect.padEnd(12)} got ${result.stage.padEnd(12)} ${asExpected ? why : `the exclusion's reason changed, re-measure it: ${result.detail}`}`)
}

rmSync(scratch, { recursive: true, force: true })
console.log(`\n${failures === 0 ? 'all lines behaved as recorded' : `${failures} line(s) differ from the record`}`)
process.exit(failures === 0 ? 0 : 1)
