from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from inference import run_episode


class ArenaHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, code: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "not_found"}, code=404)

    def do_POST(self):
        if self.path.startswith("/reset"):
            score = run_episode(task_name="static_opponent", agent_name="trained", episode_idx=1, emit_logs=False)
            self._send_json({"task": "static_opponent", "agent": "trained", "score": score})
            return
        self._send_json({"error": "not_found"}, code=404)


def run_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    server = HTTPServer((host, port), ArenaHandler)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
