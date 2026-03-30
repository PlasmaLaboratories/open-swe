import http from "node:http";

const port = Number(process.env.PORT || "3000");
const upstreamPort = Number(process.env.UPSTREAM_PORT || "3001");
const authToken = process.env.AUTH_TOKEN || "";

function unauthorized(res) {
  res.writeHead(401, { "content-type": "text/plain; charset=utf-8" });
  res.end("Unauthorized");
}

const server = http.createServer((req, res) => {
  if (authToken) {
    const authHeader = req.headers.authorization;
    if (authHeader !== `Bearer ${authToken}`) {
      unauthorized(res);
      return;
    }
  }

  const upstream = http.request(
    {
      hostname: "127.0.0.1",
      port: upstreamPort,
      method: req.method,
      path: req.url,
      headers: req.headers,
    },
    (upstreamRes) => {
      res.writeHead(upstreamRes.statusCode || 502, upstreamRes.headers);
      upstreamRes.pipe(res);
    },
  );

  upstream.on("error", (error) => {
    res.writeHead(502, { "content-type": "text/plain; charset=utf-8" });
    res.end(`Upstream error: ${error.message}`);
  });

  req.pipe(upstream);
});

server.listen(port, "0.0.0.0", () => {
  process.stdout.write(`Sentry MCP auth proxy listening on ${port}\n`);
});
