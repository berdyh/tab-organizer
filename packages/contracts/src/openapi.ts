/**
 * Derive the published OpenAPI document from `ROUTES`.
 *
 * SEC-44 enumerates the route table from this document rather than from a
 * hand-maintained list, so that a newly added route is unclassified by
 * construction and fails until someone protects it or documents it. That
 * property only holds if the document is generated from the same declaration
 * the guard reads -- writing it by hand would restore exactly the drift the
 * probe exists to catch.
 */

import { ROUTES, type AuthScope, type RouteSpec } from './routes.ts';

export interface OpenApiDocument {
  readonly openapi: string;
  readonly info: { readonly title: string; readonly version: string };
  readonly components: unknown;
  readonly paths: Record<string, Record<string, unknown>>;
}

const SECURITY_SCHEME_BY_SCOPE: Record<Exclude<AuthScope, 'public'>, string> = {
  agent: 'backendAgentToken',
  callback: 'backendCallbackToken',
};

function pathParameters(path: string): readonly unknown[] {
  return [...path.matchAll(/\{([^}]+)\}/g)].map((match) => ({
    name: match[1],
    in: 'path',
    required: true,
    schema: { type: 'string' },
  }));
}

function operation(route: RouteSpec): Record<string, unknown> {
  const parameters = pathParameters(route.path);
  return {
    summary: route.summary,
    operationId: `${route.method.toLowerCase()}_${route.path.replace(/[^a-zA-Z0-9]+/g, '_')}`,
    ...(parameters.length > 0 ? { parameters } : {}),
    // A public route publishes an EMPTY security list, which in OpenAPI means
    // "no credential required" -- an explicit statement, not an omission.
    security:
      route.scope === 'public' ? [] : [{ [SECURITY_SCHEME_BY_SCOPE[route.scope]]: [] }],
    responses: {
      '200': { description: 'Success' },
      ...(route.scope === 'public'
        ? {}
        : { '401': { description: 'Missing or invalid bearer token' } }),
    },
  };
}

export function buildOpenApiDocument(version: string): OpenApiDocument {
  const paths: Record<string, Record<string, unknown>> = {};
  for (const route of ROUTES) {
    const entry = (paths[route.path] ??= {});
    entry[route.method.toLowerCase()] = operation(route);
  }
  return {
    openapi: '3.1.0',
    info: { title: 'tab-organizer gateway', version },
    components: {
      securitySchemes: {
        backendAgentToken: {
          type: 'http',
          scheme: 'bearer',
          description: 'BACKEND_AGENT_API_TOKEN — local agent-facing surface.',
        },
        backendCallbackToken: {
          type: 'http',
          scheme: 'bearer',
          description: 'BACKEND_CALLBACK_TOKEN — capture writer callbacks.',
        },
      },
    },
    paths,
  };
}
