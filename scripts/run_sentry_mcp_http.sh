#!/bin/sh
set -eu

UPSTREAM_PORT="${UPSTREAM_PORT:-3001}"

STDIO_CMD="npx -y @sentry/mcp-server@latest"

if [ -n "${SENTRY_HOST:-}" ]; then
  STDIO_CMD="${STDIO_CMD} --host=${SENTRY_HOST}"
fi

if [ -n "${MCP_DISABLE_SKILLS:-}" ]; then
  STDIO_CMD="${STDIO_CMD} --disable-skills=${MCP_DISABLE_SKILLS}"
fi

npx -y supergateway \
  --stdio "${STDIO_CMD}" \
  --outputTransport streamableHttp \
  --port "${UPSTREAM_PORT}" &

exec node /app/scripts/sentry_mcp_auth_proxy.mjs
