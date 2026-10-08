"""A stand-in for the preview proxy, whose one job is to drop `Authorization`.

The hosted preview has been answering requests from a signed-in browser as if nobody had
signed in: every request whose credential travelled in a body arrived intact, and every
request carrying a token in the `Authorization` header reached the server with no credential
at all. This proxy reproduces that, so the fix can be tested against it rather than reasoned
about. It forwards everything else untouched, on the theory that a proxy consumes the one
header it has a reason to consume and leaves the rest alone.
"""
from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ("127.0.0.1", 8000)
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "upgrade"}


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def forward(self, method):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else None
        # The whole point: `Authorization` does not go on.
        headers = {k: v for k, v in self.headers.items() if k.lower() != "authorization"}
        headers["Host"] = "%s:%d" % UPSTREAM
        connection = http.client.HTTPConnection(*UPSTREAM, timeout=30)
        try:
            connection.request(method, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(payload)
        finally:
            connection.close()

    do_GET = lambda self: self.forward("GET")
    do_POST = lambda self: self.forward("POST")
    do_PUT = lambda self: self.forward("PUT")
    do_DELETE = lambda self: self.forward("DELETE")
    do_OPTIONS = lambda self: self.forward("OPTIONS")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8010), Proxy).serve_forever()
