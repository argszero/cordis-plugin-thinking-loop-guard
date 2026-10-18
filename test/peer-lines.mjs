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
 * of 2026-10-18, oldest first; `npm view @deepseek-ai/dsh-agent versions`
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
  '0.2.1-alpha.1', '0.2.1-alpha.2', '0.2.1-alpha.3', '0.2.1-alpha.4', '0.2.1-alpha.5',
  '0.2.1-rc.1', '0.2.1-rc.2',
]

/**
 * The versions this plugin is expected to install against. Each one has had the
 * full suite run against it with **every** dsh package pinned to that single
 * line — the peers named by that line's own `dsh-agent`, not just the two this
 * source imports (`scripts/probe-lines.mjs` holds the measurement) — with
 * `npx tsc` clean. The seam the guard wraps (`llm/stream`) plus the three symbols
 * it imports (`createUserMessage`, `isAgentLoopRequest`, `markAgentLoopRequest`)
 * predate all of them, so nothing here is claimed on faith.
 *
 * The `0.2.1` lines are measured with the grammar arm in its `frozen` regime —
 * upstream deleted the module those lines would have been asked
 * (`f028f25667`) — and the probe prints that line's skipped-arm count beside its
 * pass count for exactly that reason. The dev pin stays on `0.2.0-rc.2`, the last
 * line that ships the module, so a plain `npm test` still asks the real validator
 * and `test/grammar.spec.mjs` keeps the transcription anchored.
 */
export const SUPPORTED = [
  '0.1.2-rc.1',
  '0.1.3-alpha.2',
  '0.1.5-alpha.1', '0.1.5-alpha.2', '0.1.5-rc.1', '0.1.5-rc.2', '0.1.5-rc.3',
  '0.1.6-alpha.1', '0.1.6-alpha.2',
  '0.1.7-alpha.1', '0.1.7-alpha.2', '0.1.7-rc.1', '0.1.7-rc.2',
  '0.2.0-rc.1', '0.2.0-rc.2',
  '0.2.1-alpha.1', '0.2.1-alpha.2', '0.2.1-alpha.3', '0.2.1-alpha.4', '0.2.1-alpha.5',
  '0.2.1-rc.1', '0.2.1-rc.2',
]

/**
 * Published lines that must stay **out** of the range, each with the reason the
 * probe records. The probe does not merely skip these: it runs them and asserts
 * they still fail the way this table says, so "should fail" and "was not
 * measured" stay distinguishable. A reason that stops reproducing is a probe
 * failure, not a silent edit.
 *
 * **Empty since 0.3.2, and that is the substantive change of this release.** The
 * one entry here was `0.2.1-alpha.1`, excluded because `f028f25667` ("remove
 * runtime invariant plugins", 2026-09-30) deleted `@deepseek-ai/dsh-invariants`
 * and every `<pkg>/invariant` subpath, so the arm that ran our synthesized
 * `finish` through the real validator could not run on that line at all.
 *
 * Excluding it was the wrong instrument for the right observation. A line this
 * suite cannot fully test is still a line a *user* runs, and the exclusion did
 * not protect them — the bundle simply stopped loading, silently, the day
 * `0.2.1` shipped (`skippedBundles`, discussion #10077 and its follow-up on the
 * peer bound). The grammar arm now asks the shipped module where the line still
 * publishes it and falls back to the transcription in `test/grammar.mjs` where
 * upstream removed it, naming the regime in the arm's title; `test/grammar.spec.mjs`
 * holds the transcription to the module on every line that still has one, and
 * reports itself **skipped** where none exists. So "cannot ask upstream" is
 * disclosed per line by the probe's skip count rather than converted into a
 * refusal that costs the user the guard.
 *
 * Two lines were excluded in earlier releases for reasons that did not survive
 * being re-measured, and both are worth keeping in mind before adding another:
 * `0.1.7-alpha.1` was excluded because it "resolved with a nested second copy of
 * `@deepseek-ai/dsh-llm`" — that copy was a probe pinning only two packages and
 * letting npm derive the other peers to the newest build in range, not a property
 * of the line; pinned whole, it compiles and passes, and is admitted from 0.1.10
 * on. The same artifact is what made `0.1.5-alpha.1`/`0.1.6-alpha.1` uninstallable
 * in the first version of the probe. An exclusion is a claim about *upstream*, so
 * it has to be measured the way a consumer would get the line — all of it.
 */
export const EXCLUDED = []

/** The dsh packages whose peer range is being measured. */
export const PEER_DEPS = ['@deepseek-ai/dsh-agent', '@deepseek-ai/dsh-llm']
