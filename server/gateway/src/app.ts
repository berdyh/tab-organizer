/**
 * Gateway assembly: turn the declarative route table into a running server.
 *
 * Every route on the wire comes from `ROUTES`, and its guard comes from the
 * same entry that put it in the OpenAPI document. There is exactly one
 * deliberate exception, `/openapi.json` itself, and it is excluded from its own
 * `paths` for the same reason FastAPI excludes it: the document is a
 * description of the API, not a member of it. If it listed itself, SEC-44 would
 * demand it be either authenticated (making the route table unreadable to the
 * probe that must read it) or allowlisted (putting a meta-route on a list
 * reserved for reviewed user-facing exposure).
 */

import cors from '@fastify/cors';
import { randomUUID } from 'node:crypto';
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from 'fastify';
import {
  ROUTES,
  buildOpenApiDocument,
  toFastifyPath,
  type RouteSpec,
} from '@tab-organizer/contracts';
import { authorize } from './auth.ts';
import type { GatewayConfig } from './config.ts';
import { PROJECTORS } from './projections.ts';
import { forward } from './proxy.ts';

const REQUEST_ID_HEADER = 'x-request-id';
const MAX_REQUEST_ID_LEN = 128;
const REQUEST_ID_DISALLOWED = /[^A-Za-z0-9._-]/g;

/**
 * Constrain an inbound id so it cannot inject headers or log lines. Mirrors
 * `services/observability.py::_sanitize_request_id`, including the fallback to
 * a fresh id when nothing survives sanitisation.
 */
function sanitizeRequestId(raw: string | undefined): string {
  const cleaned = (raw ?? '').trim().replace(REQUEST_ID_DISALLOWED, '').slice(0, MAX_REQUEST_ID_LEN);
  return cleaned || randomUUID().replaceAll('-', '');
}

function requestIdOf(request: FastifyRequest): string {
  return (request as FastifyRequest & { gatewayRequestId?: string }).gatewayRequestId ?? '';
}

function headerValue(request: FastifyRequest, name: string): string | undefined {
  const value = request.headers[name];
  return Array.isArray(value) ? value[0] : value;
}

async function handleProxied(
  route: RouteSpec,
  config: GatewayConfig,
  request: FastifyRequest,
  reply: FastifyReply,
): Promise<void> {
  const outcome = authorize(route.scope, headerValue(request, 'authorization'), config);
  if (!outcome.ok) {
    await reply.code(401).send(outcome.body);
    return;
  }

  const result = await forward(
    route,
    {
      method: request.method,
      url: request.url,
      authorization: headerValue(request, 'authorization'),
      contentType: headerValue(request, 'content-type'),
      accept: headerValue(request, 'accept'),
      requestId: requestIdOf(request),
      body: request.body as Buffer | undefined,
    },
    config,
  );

  if ('failure' in result) {
    await reply.code(502).send(result.failure);
    return;
  }

  // Response headers are rebuilt, not copied. The upstream's CORS headers must
  // never reach the browser -- this facade owns origin policy -- and neither
  // must Set-Cookie.
  reply.header('content-type', result.contentType);

  if (route.projection && result.contentType.includes('application/json')) {
    const project = PROJECTORS[route.projection];
    try {
      await reply.code(result.status).send(project(JSON.parse(result.body.toString('utf8'))));
      return;
    } catch {
      // A projected route that returns unparseable JSON is not passed through
      // unprojected -- that would be the projection failing open on exactly the
      // route whose whole purpose is to withhold page content.
      await reply.code(502).send({
        error: {
          code: 'projection_failed',
          cause: 'The upstream returned a body this route could not project.',
          fix: 'Check backend-core logs for this request id; the response shape changed.',
        },
        detail: 'projection_failed',
      });
      return;
    }
  }

  await reply.code(result.status).send(result.body);
}

export async function buildApp(config: GatewayConfig): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });

  // Forward request bodies byte-for-byte. The facade does not interpret what it
  // relays; parsing and re-serialising would silently normalise payloads the
  // upstream validates.
  //
  // `removeAllContentTypeParsers` is load-bearing, not tidiness: Fastify ships
  // a built-in `application/json` parser that takes precedence over a `'*'`
  // registration, so without this line every JSON request body arrives as a
  // parsed object, fails the `Buffer` check below, and is forwarded as NO body
  // at all. The upstream then answers "Field required" for a field the caller
  // did send -- a silent request-mangling proxy, which is worse than one that
  // refuses.
  app.removeAllContentTypeParsers();
  app.addContentTypeParser('*', { parseAs: 'buffer' }, (_request, body, done) => {
    done(null, body);
  });

  // Registered BEFORE CORS so a preflight still carries a request id. Starlette
  // needed the reverse add order for the same reason (see main.py:78-82); the
  // hazard is identical, the spelling is not.
  app.addHook('onRequest', async (request, reply) => {
    const id = sanitizeRequestId(headerValue(request, REQUEST_ID_HEADER));
    (request as FastifyRequest & { gatewayRequestId?: string }).gatewayRequestId = id;
    reply.header('X-Request-ID', id);
  });

  await app.register(cors, {
    origin: config.allowedOrigins === '*' ? '*' : [...config.allowedOrigins],
    // Never. This is half of the pair that made every unauthenticated read a
    // cross-site exfiltration primitive; the other half was the wildcard.
    credentials: false,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  });

  const openApiDocument = buildOpenApiDocument(config.version);
  app.get('/openapi.json', async (_request, reply) => reply.send(openApiDocument));

  for (const route of ROUTES) {
    const path = toFastifyPath(route.path);

    if (route.upstream === 'local') {
      if (route.path === '/') {
        app.get(path, async () => ({
          service: 'tab-organizer-gateway',
          version: config.version,
          status: 'running',
        }));
      } else {
        // No secret, no token, no URL userinfo: this body is unauthenticated by
        // design (SEC-23), and the repo's first no-secrets test passed while
        // leaking a credentialed OLLAMA_HOST because it looked for named keys
        // rather than credentials in general.
        app.get(path, async () => ({
          status: 'healthy',
          service: 'tab-organizer-gateway',
          version: config.version,
        }));
      }
      continue;
    }

    app.route({
      method: route.method,
      url: path,
      handler: (request, reply) => handleProxied(route, config, request, reply),
    });
  }

  return app;
}
