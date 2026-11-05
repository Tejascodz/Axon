"""A stand-in OpenAI-compatible LLM for CI: answers from the first source it is shown, cites it,
and streams the reply word by word like vLLM or SGLang. Lets CI run /ask end to end (retrieval,
guardrails, streaming, citation checks, metrics) without a GPU. Not for anything else."""
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SOURCE = re.compile(r'<source id="(c\d+)"[^>]*>\n(.*?)\n</source>', re.S)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/health", "/metrics"):
            body = b"" if self.path == "/health" else b"# TYPE sglang:num_queue_reqs gauge\nsglang:num_queue_reqs 0\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")   # Prometheus 3 requires one
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self.send_error(404)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = req["messages"][-1]["content"]
        found = SOURCE.findall(user)
        if found:
            sid, text = found[0]
            sentence = re.split(r"(?<=[.!?])\s", text.strip())[0]
            reply = f"Answer: {' '.join(sentence.split()[:6])}\nEvidence: {sentence} [{sid}]"
        else:
            reply = "Answer: The sources do not contain the answer.\nEvidence: No sources were given."
        words = reply.split(" ")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i, w in enumerate(words):
            piece = w if i == 0 else " " + w
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
        usage = {"prompt_tokens": len(user.split()), "completion_tokens": len(words)}
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()  # inside the CI network only  # nosec B104
