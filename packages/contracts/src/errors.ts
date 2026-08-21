/**
 * The `{code, cause, fix}` error contract.
 *
 * `docs/ARCHITECTURE_PLAN.md` ("Test intentions") requires every API and
 * `tabctl` error to carry all three fields. The reason is the repo's own
 * history: a degraded stack that says only "provider unavailable" sends the
 * reader to grep, while one that names the fix ends the incident. `code` is a
 * stable machine identifier, `cause` states what was observed, `fix` states
 * the single action that resolves it.
 *
 * Nothing here may carry a secret. These bodies are returned to unauthenticated
 * callers on public routes, and the repo has already shipped one no-secrets
 * test that passed while leaking a credentialed `OLLAMA_HOST` -- so the rule is
 * "no credentials of any shape", not "no named keys".
 */

export interface ErrorBody {
  readonly code: string;
  readonly cause: string;
  readonly fix: string;
}

export interface ErrorResponse {
  readonly error: ErrorBody;
  /**
   * FastAPI renders errors as `{"detail": ...}` and the frozen suite reads
   * `response.text` in places. Keeping `detail` alongside `error` means the
   * facade stays wire-compatible with the Python stack it fronts during the
   * dual-stack window, without weakening the richer contract.
   */
  readonly detail: string;
}

export function errorResponse(body: ErrorBody): ErrorResponse {
  return { error: body, detail: `${body.code}: ${body.cause} Fix: ${body.fix}` };
}

/** A route was reached without the bearer token its scope requires. */
export const ERR_UNAUTHENTICATED = 'unauthenticated';
/** The scope's token is not configured, so the door is closed, not open. */
export const ERR_TOKEN_UNCONFIGURED = 'token_unconfigured';
/** The upstream Python data plane could not be reached or failed to answer. */
export const ERR_UPSTREAM_UNAVAILABLE = 'upstream_unavailable';
