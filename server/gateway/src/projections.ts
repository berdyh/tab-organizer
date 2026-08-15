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
 * Mirrors `routes.py::URL_LIST_METADATA_FIELDS`.
 *
 * NOTE: this list has seven entries; the frozen SEC-45 probe asserts
 * `set(metadata) <= {title, status_code, auth_type, auth_used, capture_id}` --
 * five. The two disagree about `credential_scope_drop` and `auth_type` is in
 * both. The disagreement is invisible today only because no probe capture
 * triggers a credential scope drop. Matching the SERVICE here rather than the
 * probe, because dropping `credential_scope_drop` would hide that a redirect
 * carried the fetch off-origin and the credentials were dropped -- which is the
 * signal that keeps a logged-out capture from reading as an authenticated one.
 * Flagged for a ledger decision; see the gateway MODULE.md card.
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
