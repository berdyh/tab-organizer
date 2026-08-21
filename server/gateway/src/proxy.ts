/**
 * Forwarding to the Python data plane.
 *
 * The facade is an EDGE facade (plan decision 23): it owns policy -- origin,
 * scope, projection, request identity -- and Python keeps owning data until the
 * wk10 atomic cutover. That split exists because `sessions/manager.py` loads
 * SQLite into process-local dicts and persists by DELETE-all-then-reinsert, so
 * a second concurrent writer silently erases the first. One writer, always.
 *
 * The single most important property in this file: **the facade never attaches
 * a privileged token on behalf of a caller who did not present one.** The
 * caller's own Authorization header is forwarded verbatim on scoped routes and
 * omitted entirely on public ones. Minting a server-side token here would
 * rebuild the exact confused deputy SEC-44 was invented to catch -- the two
 * credential-proxy routes that carried no auth dependency while attaching the
 * privileged Browser Engine token themselves.
 */

import {
  ERR_UPSTREAM_UNAVAILABLE,
  errorResponse,
  type RouteSpec,
} from '@tab-organizer/contracts';
import type { GatewayConfig } from './config.ts';

export interface UpstreamRequest {
  readonly method: string;
  /** Path plus query string, exactly as received. */
  readonly url: string;
  readonly authorization: string | undefined;
  readonly contentType: string | undefined;
  readonly accept: string | undefined;
  readonly requestId: string;
  readonly body: Buffer | undefined;
}

export interface UpstreamResponse {
  readonly status: number;
  readonly contentType: string;
  readonly body: Buffer;
}

const DEFAULT_TIMEOUT_MS = 120_000;

export async function forward(
  route: RouteSpec,
  request: UpstreamRequest,
  config: GatewayConfig,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<UpstreamResponse | { readonly failure: ReturnType<typeof errorResponse> }> {
  const headers: Record<string, string> = {
    'X-Request-ID': request.requestId,
  };
  if (request.contentType) headers['content-type'] = request.contentType;
  if (request.accept) headers['accept'] = request.accept;

  // Public routes forward NO credential. Scoped routes forward the caller's
  // own, already validated above, so the upstream re-validates it independently.
  if (route.scope !== 'public' && request.authorization) {
    headers['authorization'] = request.authorization;
  }

  // Cookies are never forwarded. The facade has no cookie-based auth, and a
  // browser that reached a public route would otherwise have its cookies
  // replayed against the data plane.

  const hasBody = request.body !== undefined && request.body.length > 0;

  try {
    const response = await fetch(`${config.backendUrl}${request.url}`, {
      method: request.method,
      headers,
      ...(hasBody ? { body: new Uint8Array(request.body as Buffer) } : {}),
      signal: AbortSignal.timeout(timeoutMs),
      redirect: 'manual',
    });
    return {
      status: response.status,
      contentType: response.headers.get('content-type') ?? 'application/json',
      body: Buffer.from(await response.arrayBuffer()),
    };
  } catch (error) {
    return {
      failure: errorResponse({
        code: ERR_UPSTREAM_UNAVAILABLE,
        // The upstream base URL is deliberately NOT interpolated here: it may
        // carry userinfo (`http://user:pw@host`) under a misconfiguration, and
        // this body reaches unauthenticated callers on public routes.
        cause: `The backend data plane did not answer within ${timeoutMs}ms or refused the connection (${
          error instanceof Error ? error.name : 'unknown error'
        }).`,
        fix: 'Check that backend-core is running and BACKEND_CORE_URL points at it.',
      }),
    };
  }
}
