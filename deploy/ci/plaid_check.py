"""CI: PLAID (fast-plaid) builds and searches inside the image, on the installed torch build."""
import tempfile

import numpy as np
from pylate import indexes

rng = np.random.default_rng(0)
docs = [rng.standard_normal((int(rng.integers(20, 60)), 128)).astype("float32") for _ in range(600)]
docs = [d / np.linalg.norm(d, axis=1, keepdims=True) for d in docs]
with tempfile.TemporaryDirectory() as d:
    pl = indexes.PLAID(index_folder=d, index_name="t", override=True, show_progress=False)
    pl.add_documents(documents_ids=[f"d{i}" for i in range(600)], documents_embeddings=docs)
    top = [r["id"] for r in pl(docs[17][:8][None] + 0.01, k=5)[0]]
assert top[0] == "d17", top
print("plaid ok:", top[:3])
