/**
 * Guards on the route table itself.
 *
 * These are the properties the frozen suite's SEC-44 checks from the outside,
 * asserted here from the inside so they fail in `pnpm test` without needing a
 * running stack, a Python data plane, or Docker. The black-box probe stays
 * authoritative -- this is the fast feedback loop, not a replacement.
 */

import { describe, expect, it } from 'vitest';
import { ROUTES, toFastifyPath } from './routes.ts';
import { buildOpenApiDocument } from './openapi.ts';

describe('route table', () => {
  it('gives every publicly reachable route a written reason', () => {
    for (const route of ROUTES) {
      if (route.scope === 'public') {
        expect(route.publicReason.trim().length, `${route.method} ${route.path}`).toBeGreaterThan(0);
      }
    }
  });

  it('declares no /platform/ route (plan decision 41: delete by omission)', () => {
    expect(ROUTES.filter((route) => route.path.includes('/platform/'))).toEqual([]);
  });

  it('has no duplicate method+path pair', () => {
    const seen = new Set<string>();
    for (const route of ROUTES) {
      const key = `${route.method} ${route.path}`;
      expect(seen.has(key), `duplicate ${key}`).toBe(false);
      seen.add(key);
    }
  });

  it('converts OpenAPI path templates to Fastify parameters', () => {
    expect(toFastifyPath('/api/v1/urls/{session_id}')).toBe('/api/v1/urls/:session_id');
    expect(toFastifyPath('/api/v1/sessions')).toBe('/api/v1/sessions');
  });
});

describe('published OpenAPI document', () => {
  const document = buildOpenApiDocument('test');

  it('describes exactly the routes the table declares — no more, no fewer', () => {
    const published = new Set<string>();
    for (const [path, operations] of Object.entries(document.paths)) {
      for (const method of Object.keys(operations)) {
        published.add(`${method.toUpperCase()} ${path}`);
      }
    }
    const declared = new Set(ROUTES.map((route) => `${route.method} ${route.path}`));
    expect([...published].sort()).toEqual([...declared].sort());
  });

  it('marks scoped routes as requiring a credential and public ones as not', () => {
    for (const route of ROUTES) {
      const operation = document.paths[route.path]?.[route.method.toLowerCase()] as {
        security: unknown[];
      };
      if (route.scope === 'public') {
        expect(operation.security, `${route.method} ${route.path}`).toEqual([]);
      } else {
        expect(operation.security.length, `${route.method} ${route.path}`).toBe(1);
      }
    }
  });

  it('does not list itself', () => {
    // SEC-44 enumerates the route table FROM this document. If the document
    // were a member of its own paths it would have to be either authenticated
    // (unreadable to the probe that must read it) or added to an allowlist
    // reserved for reviewed user-facing exposure. FastAPI excludes it for the
    // same reason.
    expect(Object.keys(document.paths)).not.toContain('/openapi.json');
  });
});
