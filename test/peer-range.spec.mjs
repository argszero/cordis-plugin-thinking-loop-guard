import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { satisfies } from 'semver'
import { PUBLISHED, SUPPORTED, PEER_DEPS } from './peer-lines.mjs'

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'))

// The measured line lists live in `./peer-lines.mjs` so that this assertion and
// `npm run test:probe-lines` read the same record: the probe is the instrument
// that produced it, this file is the guard that keeps the shipped range honest
// against it.

/**
 * Guard the peer range by *computing* admission, not by pattern-matching it.
 *
 * v0.1.6 shipped a test that merely asserted the range string contained
 * `0.1.2-rc.N` and `0.1.5-(alpha|beta|rc).N`. That form cannot tell a correct
 * range from an incorrect one — it fails only on a *different-looking* string —
 * and so it certified a range that was already wrong: the shipped
 * `>=0.1.2-rc.1 <0.2.0 || >=0.1.5-alpha.1 <0.2.0` admits only the 0.1.2-rc and
 * 0.1.5 tuples, because a semver comparator admits a prerelease only when some
 * comparator in the same group shares that prerelease's major.minor.patch
 * tuple. The 0.1.3-alpha.2 line and the whole 0.1.6 line were excluded, so a
 * user on the newest shipped dsh line got
 *
 *   npm error ERESOLVE unable to resolve dependency tree
 *   npm error peer @deepseek-ai/dsh-agent@">=0.1.2-rc.1 <0.2.0 || ..." from
 *   npm error   @argszero/cordis-plugin-thinking-loop-guard@0.1.6
 *
 * for a plugin whose suite passes on that line. The dsh packages are *peers*, so
 * the range cannot be papered over with `--legacy-peer-deps`: the install simply
 * fails.
 *
 * Asserting the admitted set *exactly* makes both failure directions loud: a
 * range that quietly drops a supported line fails the equality below, and a
 * range that quietly admits an untested line fails it too.
 */
for (const dep of PEER_DEPS) {
  test(`peer range for ${dep} admits exactly the tested dsh lines`, () => {
    const range = pkg.peerDependencies[dep]
    assert.ok(range, 'the dsh peer dependency must be declared')

    const admitted = PUBLISHED.filter((v) => satisfies(v, range))
    assert.deepEqual(
      admitted,
      SUPPORTED,
      `the range "${range}" admits [${admitted.join(', ')}] but the suite has only been ` +
        `run against [${SUPPORTED.join(', ')}]`,
    )

    const refused = PUBLISHED.filter((v) => !satisfies(v, range))
    assert.deepEqual(
      refused,
      PUBLISHED.filter((v) => !SUPPORTED.includes(v)),
      'older, untested prereleases must keep getting a loud ERESOLVE',
    )
  })
}

test('a bare >=0.1.2 comparator would match no published prerelease', () => {
  // Not a semver evaluator: this pins the *reason* the union exists, so a later
  // "simplification" to `>=0.1.2` fails here with the explanation attached.
  for (const version of PUBLISHED) {
    assert.equal(satisfies(version, '>=0.1.2'), false, `>=0.1.2 unexpectedly admits ${version}`)
  }
  assert.equal(satisfies('0.1.6-alpha.2', pkg.peerDependencies['@deepseek-ai/dsh-llm']), true)
})

test('the newest shipped dsh line is admitted (issue #1 regression)', () => {
  // Issue #1: the whole 0.1.6 line was refused, so a user on the then-newest
  // dsh could not install the plugin at all. The same defect recurred twice
  // after that without anything here seeing it, because the range was only ever
  // exercised by re-reading the string: the union ended at `<0.2.0`, which
  // admits `0.2.0-rc.N` only by an accident of prerelease ordering and refuses
  // the stable `0.2.0` outright, and no comparator was ever added for the 0.1.7
  // line. So 0.1.10 measured the published versions instead — the fifteen lines
  // that pass are admitted, and the line that does not is excluded on
  // measurement rather than left in by accident.
  for (const line of ['0.1.6-alpha.2', '0.1.7-rc.2', '0.2.0-rc.2']) {
    for (const dep of PEER_DEPS) {
      assert.equal(
        satisfies(line, pkg.peerDependencies[dep]),
        true,
        `${dep} must admit ${line}`,
      )
    }
  }
})

test('the upper bound is a minor line, not a release (issue #3)', () => {
  // Issue #3 (jilian-dsh, 2026-10-18): the union ended at `<0.2.1`. Under npm's
  // reading that refuses `0.2.1-rc.N` outright — a prerelease is admitted only by
  // a comparator sharing its tuple and carrying a prerelease — and, more to the
  // point, the day `0.2.1` final ships every 0.2.1 profile would fail the peer
  // check and `loadProfileDirectory()` would push the whole bundle into
  // `skippedBundles`: the guard would stop loading and say nothing.
  //
  // So the newest measured line is bounded by the *minor* line, not by the next
  // release. `0.3.0-0` rather than `0.3.0` because a bare `<0.3.0` would, under
  // the `includePrerelease` reading dsh's own compatibility gate uses
  // (`app-boot/src/plugin-compatibility.ts`, which the runtime applies to these
  // peers), admit `0.3.0-alpha.N` — a next-minor prerelease nobody has run.
  for (const dep of PEER_DEPS) {
    const range = pkg.peerDependencies[dep]
    for (const future of ['0.2.1', '0.2.2', '0.2.9']) {
      assert.equal(satisfies(future, range), true, `${dep} must admit ${future}`)
    }
    for (const beyond of ['0.3.0-0', '0.3.0-alpha.1', '0.3.0', '0.3.1']) {
      assert.equal(satisfies(beyond, range), false, `${dep} must not admit ${beyond}`)
    }
  }
  // The alternative the issue also offered — one `engines.dsh` range — was
  // declined deliberately: `unmetRequirements()` returns as soon as an own
  // `engines.dsh` exists (`plugin-compatibility.ts:57-61`) and ignores peers
  // entirely, so a single blanket range would silently widen the claim over
  // lines this suite has never run. The union is the record of what was run.
  assert.equal(Object.hasOwn(pkg, 'engines'), false, 'engines.dsh alone decides and would widen the support claim')
})

test('the range admits the same set under the reading the bundle gate uses', () => {
  // Two consumers read this one string and they do not read it alike:
  // `unmetRequirements()` calls `satisfies(..., { includePrerelease: true })`
  // (`packages/boot/app-boot/src/plugin-compatibility.ts:52`), while npm's
  // resolver uses the default reading — which refuses a prerelease unless some
  // comparator in the same `major.minor.patch` tuple carries a tag of its own.
  // The assertion above answers "can npm install it"; this one answers "will
  // dsh load it", and a range that passes only one of them is a range that is
  // broken for half its users.
  //
  // Measured on the form this release first considered — widening the last
  // segment to `>=0.2.0-rc.1 <0.3.0-0` — 15 of these 22 lines are admitted by
  // default and all 22 under the flag, because `0.2.1-alpha.1` finds no
  // comparator of its own tuple. That range would have been right for the gate
  // and an `ERESOLVE` for the installer.
  for (const dep of PEER_DEPS) {
    const range = pkg.peerDependencies[dep]
    assert.deepEqual(
      PUBLISHED.filter((v) => satisfies(v, range, { includePrerelease: true })),
      SUPPORTED,
      `the range "${range}" admits a different set with includePrerelease than without`,
    )
  }
})

test('the dev pins stay on a line the range admits', () => {
  // A dev pin outside the peer range would mean the suite ran against a line the
  // published package refuses — the exact mismatch this file exists to prevent.
  for (const dep of PEER_DEPS) {
    const pin = pkg.devDependencies[dep]
    assert.ok(SUPPORTED.includes(pin), `devDependency ${dep}@"${pin}" is not in the tested set`)
  }
})
