/**
 * The stream grammar a wrapper's output has to satisfy — asked of whichever
 * source the installed dsh line still publishes.
 *
 * Upstream **deleted the runtime invariant runtime** in `f028f25667` ("refactor:
 * remove runtime invariant plugins", 2026-09-30): `@deepseek-ai/dsh-invariants`,
 * `ctx.invariants`, every package's `./invariant` subpath and the five
 * `sdk-minimal` rows are gone from `0.2.1-alpha.1` on, replaced by an upgrade
 * guide (`docs/upgrade-guide/v0.2.0-rc.2/remove-runtime-invariants/`) telling
 * plugins to report such failures through their own channel. Nothing shipped
 * checks this grammar in its place — the module is not deprecated, it is
 * removed — so on the 0.2.1 line there is no arbiter left to ask.
 *
 * The pre-0.3.2 answer was to declare that line untestable and refuse it in the
 * peer range. That answer was wrong for a reason the range cannot express: a
 * line this suite cannot fully test is still a line a *user* runs, and refusing
 * it does not protect them — it removes the guard from their profile. So the
 * grammar is asked where it exists and **transcribed** where it does not, with
 * the regime named in every arm's title so a green under {@link FROZEN} is never
 * read as the shipped validator agreeing.
 *
 * `test/grammar.spec.mjs` is what keeps the transcription honest: on a tree that
 * still publishes the module it feeds the same corpus to both implementations
 * and requires the same verdict *and* the same message, and it fails loudly if
 * the two ever drift.
 */

/** The shipped `<pkg>/invariant` module answered. */
export const SHIPPED = 'shipped'
/** The shipped module is gone from this dsh line; {@link validateStream} answered. */
export const FROZEN = 'frozen'

/** Module-resolution failures that mean "this line does not publish the module". */
const ABSENT = new Set(['ERR_PACKAGE_PATH_NOT_EXPORTED', 'ERR_MODULE_NOT_FOUND'])

const PACKAGE = '@deepseek-ai/dsh-llm/invariant'

/**
 * Read which grammar this tree can offer.
 *
 * A resolution error is a fact about the line, not a test failure — but only the
 * two codes that mean *the subpath is not there* are read that way. Anything else
 * (a broken install, a throwing module body) is re-thrown: a suite that turned a
 * broken dependency into the frozen path would report the transcription's opinion
 * while its one source of truth was unreachable.
 */
export async function grammarSource() {
  try {
    await import(PACKAGE)
    return SHIPPED
  } catch (error) {
    if (ABSENT.has(error?.code)) return FROZEN
    throw error
  }
}

/** Require one chunk index to be a non-negative safe integer. */
function validateIndex(index, fail) {
  if (!Number.isSafeInteger(index) || index < 0) {
    fail(`LLM stream block index must be a non-negative safe integer, got ${index}`)
  }
}

/** Require a delta to address an open block of its matching type. */
function validateDelta(open, index, expected, fail) {
  validateIndex(index, fail)
  const actual = open.get(index)
  if (actual !== expected) {
    fail(`${expected} delta at index ${index} requires an open ${expected} block, got ${String(actual)}`)
  }
}

/**
 * Enforce the stream grammar as chunks are consumed.
 *
 * A transcription of `packages/llm/llm/src/invariant.ts` as it stood in
 * `dsh-v0.2.0-rc.2` (the last release that shipped it), message for message, so
 * that a failure reads the same whichever implementation produced it. It lives
 * here only as the fallback: wherever the module exists it is the module that
 * runs, and `test/grammar.spec.mjs` compares the two.
 */
export async function* validateStream(source, fail) {
  const open = new Map()
  let usageSeen = false
  let finished = false
  for await (const chunk of source) {
    if (finished) fail(`LLM stream emitted ${chunk.type} after terminal finish`)
    switch (chunk.type) {
      case 'block-start':
        validateIndex(chunk.index, fail)
        if (open.has(chunk.index)) fail(`LLM stream repeated block-start index ${chunk.index}`)
        open.set(chunk.index, chunk.blockType)
        break
      case 'text-delta':
        validateDelta(open, chunk.index, 'text', fail)
        break
      case 'reasoning-delta':
        validateDelta(open, chunk.index, 'reasoning', fail)
        break
      case 'tool-call-delta':
        validateDelta(open, chunk.index, 'tool-call', fail)
        break
      case 'block-end': {
        validateIndex(chunk.index, fail)
        const blockType = open.get(chunk.index)
        if (blockType === undefined) fail(`LLM stream block-end index ${chunk.index} has no open block`)
        if (chunk.block.type !== blockType) {
          fail(`LLM stream block-end index ${chunk.index} closes ${chunk.block.type}, expected ${blockType}`)
        }
        open.delete(chunk.index)
        break
      }
      case 'usage':
        if (usageSeen) fail('LLM stream emitted usage more than once')
        usageSeen = true
        break
      case 'finish':
        if (open.size > 0 && chunk.reason.kind !== 'error' && chunk.reason.kind !== 'aborted') {
          fail(`LLM stream finished with ${open.size} open block(s)`)
        }
        finished = true
        break
    }
    yield chunk
  }
  if (!finished) fail('LLM stream ended without a terminal finish chunk')
}

/**
 * Install the grammar on a `llm/stream` waterfall, recording every violation
 * through `fail`, and report which implementation answered.
 *
 * Registering `prepend` is what puts this listener *outside* the plugin under
 * test: `llm/stream` listeners wrap one another, and the outermost one sees what
 * a consumer sees. This plugin registers without `prepend`, so the grammar is
 * installed first and wraps the wrapper — the assertion is about the plugin's own
 * output, not about the stream it was handed.
 *
 * @param context - the `llm/stream` registry the arms build (real `ctx.on`).
 * @param fail - receives one message per violation.
 * @returns {@link SHIPPED} or {@link FROZEN} — the implementation that ran.
 */
export async function installGrammar(context, fail) {
  const source = await grammarSource()
  if (source === SHIPPED) {
    // The shipped companion is a plugin: it asks `ctx.invariants` for a register
    // callback and installs its validation through it. Both shapes are stubbed
    // here because the runtime that provided them is exactly what is missing.
    const { apply } = await import(PACKAGE)
    const inner = new Proxy(context, { get: (target, key) => (key === 'on' ? target.on.bind(target) : target[key]) })
    await apply({ ...inner, invariants: { register: (_name, installer) => installer(inner, fail) } }, fail)
    return SHIPPED
  }
  context.on('llm/stream', (_options, next) => validateStream(next(), fail), { prepend: true })
  return FROZEN
}
