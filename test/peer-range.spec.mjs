import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'))

/**
 * Guard the peer range against the two ways it has already been wrong.
 * See README "Why the peer range looks like that".
 *
 *   ">=0.1.2"             matches NOTHING -> ETARGET on install
 *   ">=0.1.2-rc.1 <0.2.0" matches only 0.1.2-rc.1 -> 0.1.5-line users get ERESOLVE
 */
for (const dep of ['@deepseek-ai/dsh-agent', '@deepseek-ai/dsh-llm']) {
  test(`peer range for ${dep} admits both supported dsh prerelease lines`, () => {
    const range = pkg.peerDependencies[dep]
    assert.ok(range, 'the dsh peer dependency must be declared')
    assert.ok(
      !/^(>=\^~)?\s*\d+\.\d+\.\d+\s*$/.test(range.trim()),
      `a bare non-prerelease comparator ("${range}") matches no published dsh version`,
    )
    assert.match(range, /0\.1\.2-rc\.\d+/, 'expected a 0.1.2-rc.N lower bound')
    assert.match(range, /0\.1\.5-(alpha|beta|rc)\.\d+/,
      'the range must also admit the 0.1.5 prerelease line (next/alpha tags)')
  })
}
