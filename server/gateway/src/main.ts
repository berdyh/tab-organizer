/**
 * Gateway entrypoint.
 *
 * Runs under Node's native type stripping, so there is no build step between
 * this file and a listening server: `node server/gateway/src/main.ts`. That is
 * what lets the planned `SEC_BOOT_BACKEND_CMD` boot mode (decision 42, ~wk8)
 * be a one-line command against this implementation rather than a pipeline.
 */

import { buildApp } from './app.ts';
import { loadConfig } from './config.ts';

const config = loadConfig();
const app = await buildApp(config);

await app.listen({ host: config.host, port: config.port });

console.log(
  JSON.stringify({
    event: 'gateway.listening',
    level: 'INFO',
    service: 'tab-organizer-gateway',
    host: config.host,
    port: config.port,
    version: config.version,
    // Origin policy is announced at boot for the same reason the Python stack
    // announces its active provider: a policy nobody can see is a policy nobody
    // notices regressing.
    allowed_origins: config.allowedOrigins === '*' ? '*' : [...config.allowedOrigins],
    // Which scopes hold a token -- never the values.
    scopes_configured: {
      agent: Boolean(config.agentToken),
      callback: Boolean(config.callbackToken),
    },
  }),
);

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    void app.close().then(() => process.exit(0));
  });
}
