/**
 * Bearer-scope enforcement, fail-closed.
 *
 * Mirrors `services/backend-core/app/api/routes.py::_require_backend_agent_auth`
 * and `_require_backend_callback_auth` exactly, including the two-step failure:
 * an UNCONFIGURED scope is a 401 (the door is closed, not unguarded) and a
 * WRONG credential is a 401. SEC-21 freezes the first of those, which is the
 * one that is easy to get backwards -- a service that treats "no token
 * configured" as "no token required" reads as working right up until it is the
 * only thing between an agent and the credential store.
 */

import { createHash, timingSafeEqual } from 'node:crypto';
import {
  ERR_TOKEN_UNCONFIGURED,
  ERR_UNAUTHENTICATED,
  errorResponse,
  type AuthScope,
  type ErrorResponse,
} from '@tab-organizer/contracts';
import type { GatewayConfig } from './config.ts';

/**
 * Constant-time string comparison.
 *
 * `timingSafeEqual` throws on length mismatch and would leak the expected
 * token's length if fed raw bytes, so both sides are hashed to a fixed 32 bytes
 * first. This is the standard construction, and it is why the comparison can be
 * made before checking anything about the candidate's shape.
 */
function constantTimeEquals(a: string, b: string): boolean {
  const left = createHash('sha256').update(a, 'utf8').digest();
  const right = createHash('sha256').update(b, 'utf8').digest();
  return timingSafeEqual(left, right);
}

/** Extract the credential from an `Authorization: Bearer <token>` header. */
function bearerToken(authorization: string | undefined): string | null {
  const [scheme, ...rest] = (authorization ?? '').split(' ');
  if ((scheme ?? '').toLowerCase() !== 'bearer') return null;
  return rest.join(' ').trim();
}

export type AuthOutcome =
  | { readonly ok: true }
  | { readonly ok: false; readonly body: ErrorResponse };

const SCOPE_ENV: Record<Exclude<AuthScope, 'public'>, string> = {
  agent: 'BACKEND_AGENT_API_TOKEN',
  callback: 'BACKEND_CALLBACK_TOKEN',
};

export function authorize(
  scope: AuthScope,
  authorization: string | undefined,
  config: GatewayConfig,
): AuthOutcome {
  if (scope === 'public') return { ok: true };

  const expected = scope === 'agent' ? config.agentToken : config.callbackToken;
  const envVar = SCOPE_ENV[scope];

  if (!expected) {
    return {
      ok: false,
      body: errorResponse({
        code: ERR_TOKEN_UNCONFIGURED,
        cause: `The ${scope} scope has no token configured, so this route is closed.`,
        fix: `Set ${envVar}, or run ./scripts/cli.py start to generate the four service tokens.`,
      }),
    };
  }

  const presented = bearerToken(authorization);
  if (presented === null || !constantTimeEquals(presented, expected)) {
    return {
      ok: false,
      body: errorResponse({
        code: ERR_UNAUTHENTICATED,
        cause: `This route requires a valid ${scope}-scope bearer token.`,
        fix: `Send Authorization: Bearer $${envVar}.`,
      }),
    };
  }

  return { ok: true };
}
