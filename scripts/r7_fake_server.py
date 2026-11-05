"""Smoke tests only: the real Axon server (nyaya.server) with the test suite's stand-in encoder and a
lexical stand-in claim checker, so scripts/round7.py --smoke can drive it on a CPU."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "scripts")]

import uvicorn  # noqa: E402
from test_system import _FakeAxon  # noqa: E402

from nyaya import server  # noqa: E402

ax = _FakeAxon()
ax.revision = None                                        # as the smoke soak index was built
server._state["axon"] = ax
if os.environ.get("AXON_VERIFY") == "1":
    from round7 import make_verifier
    v = make_verifier(True)
    v.threshold = float(os.environ.get("AXON_VERIFY_THRESHOLD", "0.5"))
    server._state["verifier"] = v

if __name__ == "__main__":
    args = sys.argv[1:]
    uvicorn.run(server.app, host=args[args.index("--host") + 1], port=int(args[args.index("--port") + 1]), log_level="warning")
