/**
 * Fail-closed scope enforcement.
 *
 * The first case is the one that matters and the one that is easy to get
 * backwards: with NO token configured, the door must be CLOSED. SEC-21 freezes
 * that as black-box behaviour but auto-skips in attached mode, because proving
 * it requires deleting configuration from the service's environment -- which
 * only a harness that owns that environment can do, and `SEC_BOOT_*_CMD` is
 * deferred to ~wk8 (decision 42). Until boot mode lands, THIS test is the only
 * executable check that the TS facade fails closed. It is a deployment-shape
 * assertion living in a tooling unit test, which is exactly where the plan's
 * 2026-07-24 corollary says such assertions belong.
 */

import { describe, expect, it } from 'vitest';
import { authorize } from './auth.ts';
import type { GatewayConfig } from './config.ts';

const AGENT = 'agent-token-value';
const CALLBACK = 'callback-token-value';

function config(overrides: Partial<GatewayConfig> = {}): GatewayConfig {
  return {
    host: '127.0.0.1',
    port: 8081,
    backendUrl: 'http://127.0.0.1:8080',
    allowedOrigins: ['http://localhost:8089'],
    agentToken: AGENT,
    callbackToken: CALLBACK,
    version: 'test',
    ...overrides,
  };
}

describe('authorize', () => {
  it('lets public routes through with no credential', () => {
    expect(authorize('public', undefined, config()).ok).toBe(true);
  });

  it('CLOSES the door when the scope has no token configured', () => {
    for (const scope of ['agent', 'callback'] as const) {
      const outcome = authorize(scope, `Bearer ${AGENT}`, config({ agentToken: '', callbackToken: '' }));
      expect(outcome.ok, scope).toBe(false);
      if (!outcome.ok) expect(outcome.body.error.code).toBe('token_unconfigured');
    }
  });

  it('rejects a missing, malformed, or wrong credential', () => {
    for (const header of [undefined, '', 'Bearer', 'Basic ' + AGENT, 'Bearer wrong', AGENT]) {
      const outcome = authorize('agent', header, config());
      expect(outcome.ok, JSON.stringify(header)).toBe(false);
    }
  });

  it('does not accept another scope’s token', () => {
    // The four service tokens must never be equal -- one shared value makes the
    // agent credential also open the credential and CDP control plane. Each
    // door accepts exactly its own scope.
    expect(authorize('agent', `Bearer ${CALLBACK}`, config()).ok).toBe(false);
    expect(authorize('callback', `Bearer ${AGENT}`, config()).ok).toBe(false);
  });

  it('accepts the right token for the right scope, case-insensitively on the scheme', () => {
    expect(authorize('agent', `Bearer ${AGENT}`, config()).ok).toBe(true);
    expect(authorize('agent', `bearer ${AGENT}`, config()).ok).toBe(true);
    expect(authorize('callback', `Bearer ${CALLBACK}`, config()).ok).toBe(true);
  });

  it('returns {code, cause, fix} and never echoes a credential', () => {
    const outcome = authorize('agent', `Bearer ${CALLBACK}`, config());
    expect(outcome.ok).toBe(false);
    if (outcome.ok) return;

    expect(outcome.body.error).toEqual({
      code: expect.any(String),
      cause: expect.any(String),
      fix: expect.any(String),
    });

    // A 401 body that quotes what was presented, or what was expected, hands an
    // attacker an oracle and puts a secret in logs. Checked against BOTH tokens
    // because the repo's first no-secrets test passed while leaking a
    // credential it had not thought to name.
    const rendered = JSON.stringify(outcome.body);
    expect(rendered).not.toContain(AGENT);
    expect(rendered).not.toContain(CALLBACK);
  });
});
