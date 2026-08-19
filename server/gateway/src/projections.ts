/**
 * Outbound response projections (SEC-45).
 *
 * `GET /api/v1/urls/{session_id}` used to return `r.metadata` verbatim, and the
 * ingest writer stores the whole captured page body under `metadata["content"]`
 * -- so the list endpoint dumped the full text of every captured page to an
 * anonymous caller. Python now projects. The facade projects AGAIN.
 *
 * That duplication is deliberate. This is a route the frozen suite classifies
 * as intentionally public, which means the projection is the only thing between
 * an unauthenticated caller and the corpus. Trusting the upstream to have kept
 * projecting is the same bet this repo has lost three times ("documented
 * invariant, therefore implemented invariant"), and the cost of not making it
 * is one whitelist.
 *
 * Deny-by-default in both directions: unknown top-level keys and unknown
 * metadata keys are dropped, so a future capture field carrying sensitive data
 * cannot leak by accident.
 */

/** Mirrors the top-level shape of `routes.py::get_urls`. */
const URL_RECORD_FIELDS = ['original', 'normalized', 'status', 'metadata'] as const;

/**
 * Mirrors `routes.py::URL_LIST_METADATA_FIELDS`, and — since SECSUITE 1.7.0 —
 * the frozen SEC-45 probe as well. All three name the same six keys.
 *
 * They did not always. The probe asserted a subset of FIVE while the service
 * emitted six, so the first capture redirected off the origin its credentials
 * belong to would have failed SEC-45 against the Python stack. It never fired
 * because no probe capture triggers a scope drop. Resolved 2026-08-19 by
 * widening the probe: `credential_scope_drop` records that credentials were
 * DROPPED rather than sent onward (hostnames, booleans and a hop count — no
 * credentials, no paths, no queries), and withholding it is what would let a
 * logged-out capture read as an authenticated one.
 *
 * A seventh key requires changing all three together. That is the point.
 */
const URL_METADATA_FIELDS = [
  'title',
  'status_code',
  'auth_type',
  'auth_used',
  'capture_id',
  'credential_scope_drop',
] as const;

function pick(source: Record<string, unknown>, keys: readonly string[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    if (Object.hasOwn(source, key)) out[key] = source[key];
  }
  return out;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function projectUrlRecord(record: unknown): unknown {
  if (!isRecord(record)) return record;
  const projected = pick(record, URL_RECORD_FIELDS);
  const metadata = projected['metadata'];
  projected['metadata'] = isRecord(metadata) ? pick(metadata, URL_METADATA_FIELDS) : {};
  return projected;
}

/**
 * The projector registry. Keys are the `Projection` literals declared in the
 * route table, so an unimplemented projector is a type error at assembly time
 * rather than a route that quietly forwards everything the upstream sent.
 */
export const PROJECTORS = {
  url_listing: (payload: unknown): unknown =>
    Array.isArray(payload) ? payload.map(projectUrlRecord) : payload,
} as const;
