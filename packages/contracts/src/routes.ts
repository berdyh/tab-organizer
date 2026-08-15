/**
 * The gateway's route table -- ONE declaration per route, from which both the
 * published OpenAPI document and the runtime auth enforcement are derived.
 *
 * Why one table instead of two mechanisms:
 *
 * SEC-44 exists because `GET /api/v1/auth/pending` and
 * `POST /api/v1/auth/credentials` shipped with no auth dependency while still
 * attaching the privileged Browser Engine token server-side -- a confused
 * deputy that let any unauthenticated caller read the pending-auth queue and
 * plant credentials for an arbitrary domain. In FastAPI a route's presence in
 * the OpenAPI document and its `Depends(...)` guard are two independent facts,
 * so they can disagree silently; the probe had to be invented to notice.
 *
 * Here they cannot disagree, because they are the same fact read twice. And a
 * route declared `scope: 'public'` structurally requires a `publicReason`, so
 * the reviewed-allowlist discipline the probe enforces at runtime is also a
 * compile error. The probe stays -- it is the black-box proof -- but it should
 * now only ever fail if this table is wrong, never if someone forgot a guard.
 *
 * NOT EXPOSED: every `/api/v1/platform/*` route. Plan decision 41 deletes the
 * B2B platform subsystem by omission at cutover; not porting it to the facade
 * is that decision taking effect, not an oversight.
 */

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

/**
 * Which bearer scope a route demands. These are the receive-side scopes of the
 * Python stack, mirrored exactly: each service accepts its own scope and no
 * other. `public` is the accepted local-single-user boundary -- safe only
 * because published ports are loopback-only AND the CORS allowlist holds
 * (SEC-43), which is why those two invariants are load-bearing here.
 */
export type AuthScope = 'public' | 'agent' | 'callback';

/** Who answers: the facade itself, or the Python data plane behind it. */
export type Upstream = 'local' | 'backend';

/**
 * A named response transform applied on the way out. Keyed by literal so the
 * gateway's registry is exhaustiveness-checked -- an unimplemented projector is
 * a type error, not a route that quietly returns everything upstream sent.
 */
export type Projection = 'url_listing';

interface RouteBase {
  readonly method: HttpMethod;
  readonly path: string;
  readonly summary: string;
  readonly upstream: Upstream;
  readonly projection?: Projection;
}

export type RouteSpec =
  | (RouteBase & { readonly scope: 'agent' | 'callback' })
  | (RouteBase & {
      readonly scope: 'public';
      /** Why this route is safe to reach without a token. Required by the type. */
      readonly publicReason: string;
    });

export const ROUTES = [
  // -- Liveness and identity. Must never require a secret (SEC-23), and must
  //    never carry a key, token, or URL userinfo in the payload.
  {
    method: 'GET',
    path: '/',
    summary: 'Service identity banner',
    upstream: 'local',
    scope: 'public',
    publicReason: 'service identity banner, no user data',
  },
  {
    method: 'GET',
    path: '/health',
    summary: 'Gateway liveness',
    upstream: 'local',
    scope: 'public',
    publicReason: 'liveness probe; scripts/cli.py polls it',
  },
  {
    method: 'GET',
    path: '/api/v1/health',
    summary: 'Aggregated stack liveness',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'aggregated liveness probe for the local UI',
  },

  // -- Session and URL CRUD the local UI drives without a token. Loopback-only
  //    by deployment; this is the accepted local-single-user boundary.
  {
    method: 'POST',
    path: '/api/v1/sessions',
    summary: 'Create a session',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI creates sessions without a token',
  },
  {
    method: 'GET',
    path: '/api/v1/sessions',
    summary: 'List sessions',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI session picker',
  },
  {
    method: 'GET',
    path: '/api/v1/sessions/{session_id}',
    summary: 'Session stats',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI session stats',
  },
  {
    method: 'DELETE',
    path: '/api/v1/sessions/{session_id}',
    summary: 'Delete a session',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI session delete',
  },
  {
    method: 'POST',
    path: '/api/v1/urls',
    summary: 'Add URLs to a session',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI URL input',
  },
  {
    method: 'GET',
    path: '/api/v1/urls/{session_id}',
    summary: 'List URL records for a session',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI URL list (projection: SEC-45)',
    // SEC-45: the ingest writer stores the whole captured page body under
    // metadata["content"], so an unfiltered listing dumps the full text of
    // every captured page. Python already projects; the facade projects again
    // rather than trusting it, because this is the endpoint an unauthenticated
    // caller reaches.
    projection: 'url_listing',
  },
  {
    method: 'POST',
    path: '/api/v1/scrape',
    summary: 'Start a scrape batch',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI starts a scrape batch',
  },
  {
    method: 'GET',
    path: '/api/v1/scrape/status/{session_id}',
    summary: 'Poll scrape batch status',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI polls batch status',
  },
  {
    method: 'POST',
    path: '/api/v1/cluster',
    summary: 'Trigger clustering',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI clustering trigger',
  },
  {
    method: 'GET',
    path: '/api/v1/clusters/{session_id}',
    summary: 'Read clusters for a session',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI cluster view',
  },
  {
    method: 'POST',
    path: '/api/v1/export',
    summary: 'Render an export',
    upstream: 'backend',
    scope: 'public',
    publicReason: 'local UI export download',
  },

  // -- Agent-facing surface. BACKEND_AGENT_API_TOKEN and nothing else.
  {
    method: 'POST',
    path: '/api/v1/tabs/import',
    summary: 'Import open browser tabs over CDP',
    upstream: 'backend',
    scope: 'agent',
  },
  {
    method: 'GET',
    path: '/api/v1/tabs/import/{job_id}',
    summary: 'Poll a tab-import job',
    upstream: 'backend',
    scope: 'agent',
  },
  {
    method: 'POST',
    path: '/api/v1/tabs/open',
    summary: 'Open tabs in the attached browser',
    upstream: 'backend',
    scope: 'agent',
  },
  {
    method: 'POST',
    path: '/api/v1/search',
    summary: 'Hybrid keyword + semantic search',
    upstream: 'backend',
    scope: 'agent',
  },
  {
    method: 'POST',
    path: '/api/v1/chat',
    summary: 'RAG chat over the corpus',
    upstream: 'backend',
    scope: 'agent',
  },
  {
    method: 'GET',
    path: '/api/v1/auth/pending',
    summary: 'Read the pending-auth queue',
    upstream: 'backend',
    scope: 'agent',
  },
  {
    method: 'POST',
    path: '/api/v1/auth/credentials',
    summary: 'Submit credentials for a pending auth request',
    upstream: 'backend',
    scope: 'agent',
  },

  // -- Writer callbacks. BACKEND_CALLBACK_TOKEN and nothing else.
  {
    method: 'POST',
    path: '/api/v1/callback/scrape-complete',
    summary: 'Legacy scrape-completion callback',
    upstream: 'backend',
    scope: 'callback',
  },
  {
    method: 'POST',
    path: '/api/v1/ingest/v1',
    summary: 'Versioned idempotent capture ingest',
    upstream: 'backend',
    scope: 'callback',
  },
] as const satisfies readonly RouteSpec[];

/** Convert an OpenAPI path template to Fastify's parameter syntax. */
export function toFastifyPath(openApiPath: string): string {
  return openApiPath.replace(/\{([^}]+)\}/g, ':$1');
}
