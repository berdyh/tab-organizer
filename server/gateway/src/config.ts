/**
 * Runtime configuration, resolved once at boot.
 *
 * Two invariants from `CLAUDE.md` are enforced here rather than documented:
 *
 * 1. **Loopback by default.** Every published port in `docker-compose.yml`
 *    binds `127.0.0.1` because these services are thinly authenticated and hold
 *    the user's captured page content. The gateway inherits that: `GATEWAY_HOST`
 *    defaults to loopback, and exposing it to the LAN is an explicit act.
 * 2. **No provider or token is ever selected implicitly.** An unset token is
 *    unset -- it never falls back to another scope's value and never invents
 *    one. Unconfigured means the door is CLOSED (401), which is what SEC-21
 *    freezes.
 */

export const CORS_ORIGINS_ENV = 'CORS_ALLOWED_ORIGINS';

/**
 * The Streamlit Web UI on the two loopback spellings a browser may use. Both
 * are distinct origins to a browser, so both are listed. Mirrors
 * `services/cors.py::DEFAULT_ALLOWED_ORIGINS` -- SEC-43 reads the same default
 * out of its fixture, so the two must not drift.
 */
export const DEFAULT_ALLOWED_ORIGINS = [
  'http://localhost:8089',
  'http://127.0.0.1:8089',
] as const;

export interface GatewayConfig {
  readonly host: string;
  readonly port: number;
  readonly backendUrl: string;
  /** `'*'` only when an operator typed it; warned about at boot, never default. */
  readonly allowedOrigins: readonly string[] | '*';
  readonly agentToken: string;
  readonly callbackToken: string;
  readonly version: string;
}

function env(name: string): string {
  return (process.env[name] ?? '').trim();
}

/** Mirrors `services/cors.py::allowed_origins()`, including the wildcard warning. */
export function resolveAllowedOrigins(): readonly string[] | '*' {
  const configured = env(CORS_ORIGINS_ENV)
    .split(',')
    .map((origin) => origin.trim().replace(/\/+$/, ''))
    .filter((origin) => origin.length > 0);

  if (configured.length === 0) return DEFAULT_ALLOWED_ORIGINS;

  if (configured.includes('*')) {
    // Dropping credentials blunts this but does not close it: the exfiltration
    // chain reads unauthenticated endpoints and never needed credentials. Say
    // so at boot rather than failing closed -- an operator who typed "*" keeps
    // it, but not silently.
    console.warn(
      JSON.stringify({
        event: 'cors.wildcard_configured',
        level: 'WARNING',
        env_var: CORS_ORIGINS_ENV,
        detail:
          "any website the user visits can read this service's responses, " +
          'including captured page text',
      }),
    );
    return '*';
  }
  return configured;
}

export function loadConfig(): GatewayConfig {
  return {
    host: env('GATEWAY_HOST') || '127.0.0.1',
    port: Number(env('GATEWAY_PORT') || '8081'),
    backendUrl: (env('BACKEND_CORE_URL') || 'http://localhost:8080').replace(/\/+$/, ''),
    allowedOrigins: resolveAllowedOrigins(),

    // Scope tokens. `agent` has no fallback at all. `callback` mirrors the
    // Python receive-side resolution in `routes.py::_backend_callback_token`,
    // which reads BACKEND_CALLBACK_TOKEN or AI_ENGINE_API_TOKEN -- SEC-21's
    // `backend_callback` case unsets BOTH, so the facade must read both or it
    // would answer 401 where Python answers 200 and diverge from the frozen
    // contract in the permissive direction on the next reconfiguration.
    agentToken: env('BACKEND_AGENT_API_TOKEN'),
    callbackToken: env('BACKEND_CALLBACK_TOKEN') || env('AI_ENGINE_API_TOKEN'),

    version: env('GATEWAY_VERSION') || '0.1.0',
  };
}
