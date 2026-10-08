/**
 * The dsh lines this plugin's peer range is measured against.
 *
 * One list, two readers: `test/peer-range.spec.mjs` asserts that the shipped
 * range admits exactly {@link SUPPORTED}, and `scripts/probe-lines.mjs` is the
 * instrument that measures it — install each line into its own tree, then run
 * `npx tsc && node --test` there. Splitting the instrument from the assertion is
 * what keeps the claim reproducible: re-running the probe after a line moves is
 * the only thing that may add a name here.
 */

/**
 * Every `@deepseek-ai/dsh-agent` / `@deepseek-ai/dsh-llm` version published as
 * of 2026-10-08, oldest first; `npm view @deepseek-ai/dsh-agent versions`
 * refreshes it.
 *
 * Deliberately frozen: it is a record of what the range was checked against, not
 * a live query. A version published later is not covered by this test — that is
 * what `npm run test:probe-lines` is for.
 */
export const PUBLISHED = [
  '0.0.1-rc.1', '0.0.1-rc.2', '0.0.1-rc.3', '0.0.1-rc.5',
  '0.1.0-rc.2', '0.1.0-rc.3', '0.1.0-rc.6', '0.1.0-rc.7', '0.1.0-rc.8',
  '0.1.1-rc.1', '0.1.1-rc.2',
  '0.1.2-alpha.2', '0.1.2-alpha.3', '0.1.2-alpha.4', '0.1.2-alpha.5',
  '0.1.2-rc.1',
  '0.1.3-alpha.2',
  '0.1.5-alpha.1', '0.1.5-alpha.2', '0.1.5-rc.1', '0.1.5-rc.2', '0.1.5-rc.3',
  '0.1.6-alpha.1', '0.1.6-alpha.2',
  '0.1.7-alpha.1', '0.1.7-alpha.2', '0.1.7-rc.1', '0.1.7-rc.2',
  '0.2.0-rc.1', '0.2.0-rc.2',
  '0.2.1-alpha.1',
]

/**
 * The versions this plugin is expected to install against. Each one has had the
 * full suite run against it with **every** dsh package pinned to that single
 * line — the peers named by that line's own `dsh-agent`, not just the two this
 * source imports (`scripts/probe-lines.mjs` holds the measurement) — and
 * `npx tsc` is clean on all fifteen, where the suite is 74/74. The seam the guard wraps
 * (`llm/stream`) plus the three symbols it imports (`createUserMessage`,
 * `isAgentLoopRequest`, `markAgentLoopRequest`) predate all of them, so nothing
 * here is claimed on faith.
 */
export const SUPPORTED = [
  '0.1.2-rc.1',
  '0.1.3-alpha.2',
  '0.1.5-alpha.1', '0.1.5-alpha.2', '0.1.5-rc.1', '0.1.5-rc.2', '0.1.5-rc.3',
  '0.1.6-alpha.1', '0.1.6-alpha.2',
  '0.1.7-alpha.1', '0.1.7-alpha.2', '0.1.7-rc.1', '0.1.7-rc.2',
  '0.2.0-rc.1', '0.2.0-rc.2',
]

/**
 * Published lines that must stay **out** of the range, each with the reason the
 * probe records. The probe does not merely skip these: it runs them and asserts
 * they still fail the way this table says, so "should fail" and "was not
 * measured" stay distinguishable. A reason that stops reproducing is a probe
 * failure, not a silent edit.
 *
 *  - **0.2.1-alpha.1** does not ship the stream-grammar validator at all: the
 *    `@deepseek-ai/dsh-llm/invariant` subpath and its `lib/invariant.js` are both
 *    gone (the installed `lib/` is `index.js`, `lib/types/**` and the two typert
 *    entries), so the one arm that runs our synthesized `finish` through the real
 *    validator cannot run there. That arm is the difference between reading the
 *    grammar and testing against it — a line where it cannot run is a line this
 *    suite would silently test less on, which is worse than one that fails loudly.
 *
 * Two lines were excluded in earlier releases for reasons that did not survive
 * being re-measured, and both are worth keeping in mind before adding a third:
 * `0.1.7-alpha.1` was excluded because it "resolved with a nested second copy of
 * `@deepseek-ai/dsh-llm`" — that copy was a probe pinning only two packages and
 * letting npm derive the other peers to the newest build in range, not a property
 * of the line; pinned whole, it compiles and passes, and is admitted from 0.1.10
 * on. The same artifact is what made `0.1.5-alpha.1`/`0.1.6-alpha.1` uninstallable
 * in the first version of the probe. An exclusion is a claim about *upstream*, so
 * it has to be measured the way a consumer would get the line — all of it.
 */
export const EXCLUDED = [
  { line: '0.2.1-alpha.1', expect: 'suite-fails', why: 'dsh-llm no longer ships the stream-grammar invariant module the protocol-legal arm runs' },
]

/** The dsh packages whose peer range is being measured. */
export const PEER_DEPS = ['@deepseek-ai/dsh-agent', '@deepseek-ai/dsh-llm']
