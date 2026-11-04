"""Index lifecycle: build, incremental update, compaction.

An index folder holds:
  config.json     format, model, version (incremented by every change), counts, options
  chunks.jsonl    one record per chunk: id, source, part, text, tenant (optional), deleted (optional)
  vectors.pt      the VectorStore (flat token vectors, document boundaries, live mask)
  manifest.json   per source file: content hash and chunk ids, so updates re-encode only what changed
  plaid/          optional approximate-search index

`update` compares files by SHA-256: new files are chunked and encoded, removed files' chunks are
marked deleted (hidden from search at once), changed files are both. Nothing else is re-encoded.
`compact` drops deleted chunks for good without re-encoding anything. Every write goes to a
temporary file and is then renamed into place, config.json last, and an exclusive lock stops two
writers; readers (a running server) see either the old index or the new one, never a mix.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import time

from .store import VectorStore

FORMAT = 2
EXTS = (".txt", ".md", ".pdf")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def scan(src: str, tenants_from_dirs: bool = False):
    """Indexable files under src: [(relative path, absolute path, tenant)] and the files refused."""
    from .rag import check_file
    files, refused = [], []
    for root, dirs, names in os.walk(src):
        dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(root, d)))   # never leave src
        for f in sorted(names):
            if not f.lower().endswith(EXTS):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, src)
            why = check_file(p)
            if why:
                refused.append({"source": rel, "reason": why})
                continue
            tenant = (rel.split(os.sep)[0] if os.sep in rel else "_root") if tenants_from_dirs else None
            files.append((rel, p, tenant))
    return files, refused


def file_chunks(path: str, rel: str, tenant: str | None) -> list[dict]:
    from .rag import chunk, read_file
    out = []
    for i, c in enumerate(chunk(read_file(path))):
        rec = {"source": rel, "part": i, "text": c}
        if tenant is not None:
            rec["tenant"] = tenant
        out.append(rec)
    return out


@contextlib.contextmanager
def locked(out: str):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, ".lock"), "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another process is changing the index in {out}") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_lines(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def check_model(cfg: dict, axon):
    """Query and document vectors are only comparable from the same model: refuse a mismatch."""
    mine = (axon.model_name, getattr(axon, "revision", None))
    built = (cfg.get("model"), cfg.get("model_revision")) if "model_revision" in cfg else (cfg.get("model"), mine[1])
    if built != mine:
        raise RuntimeError(f"the index was built with {built[0]} (revision {built[1]}), this process has "
                           f"{mine[0]} (revision {mine[1]}): vectors would not be comparable")


def read_config(out: str) -> dict:
    return json.load(open(os.path.join(out, "config.json")))


def _write(out, cfg, chunks, store, manifest):
    store.save(os.path.join(out, "vectors.pt"))
    _atomic_lines(os.path.join(out, "chunks.jsonl"), chunks)
    _atomic_json(os.path.join(out, "manifest.json"), manifest)
    cfg.update(format=FORMAT, n=len(chunks), n_live=store.n_live, multi_tenant=any("tenant" in c for c in chunks),
               updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _atomic_json(os.path.join(out, "config.json"), cfg)           # last: readers key off this file


def _plaid(out, override):
    from pylate import indexes
    return indexes.PLAID(index_folder=out, index_name="plaid", override=override, show_progress=False)


def _plaid_build(out, chunks, store):
    live = [i for i in range(len(chunks)) if bool(store.live[i])]
    pl = _plaid(out, override=True)
    pl.add_documents(documents_ids=[chunks[i]["id"] for i in live],
                     documents_embeddings=[store.doc(i).float().numpy() for i in live])


# --------------------------------------------------------------------------- build

def build(src_or_chunks, out: str, axon, plaid: bool = False, tenants_from_dirs: bool = False, log=None) -> dict:
    """Index a folder, or a list of {"source", "text"[, "tenant"]} records, from scratch."""
    t0 = time.time()
    manifest, refused = {}, []
    if isinstance(src_or_chunks, str):
        files, refused = scan(src_or_chunks, tenants_from_dirs)
        chunks = []
        for rel, p, tenant in files:
            try:
                recs = file_chunks(p, rel, tenant)
            except Exception as e:                                    # a broken file must not stop the build
                refused.append({"source": rel, "reason": f"unreadable: {type(e).__name__}: {str(e)[:80]}"})
                continue
            manifest[rel] = {"sha256": sha256(p), "tenant": tenant, "chunks": []}
            chunks += recs
    else:
        chunks = [dict(c) for c in src_or_chunks]
    for n, c in enumerate(chunks):
        c["id"] = f"c{n}"
        if c["source"] in manifest:
            manifest[c["source"]]["chunks"].append(c["id"])
    with locked(out):
        store = axon.encode_store([c["text"] for c in chunks], log=log)
        cfg = {"model": axon.model_name, "model_revision": getattr(axon, "revision", None), "version": 1, "plaid": False, "skipped": refused,
               "tenants_from_dirs": tenants_from_dirs, "source": src_or_chunks if isinstance(src_or_chunks, str) else None}
        if plaid:
            try:
                _plaid_build(out, chunks, store)
                cfg["plaid"] = True
            except Exception as e:                                    # approximate search is optional
                cfg["plaid_error"] = repr(e)[:300]
        _write(out, cfg, chunks, store, manifest)
    return {"chunks": len(chunks), "refused": refused, "seconds": round(time.time() - t0, 2),
            "vector_mb": round(store.nbytes() / 2 ** 20, 1), "plaid": cfg["plaid"]}


# --------------------------------------------------------------------------- update

def update(src: str, out: str, axon, log=None) -> dict:
    """Bring the index in line with the folder: encode new and changed files only."""
    t0 = time.time()
    with locked(out):
        cfg = read_config(out)
        if cfg.get("format") != FORMAT:
            raise RuntimeError("this index was built by an older version: rebuild it once with `build`")
        check_model(cfg, axon)
        chunks = [json.loads(line) for line in open(os.path.join(out, "chunks.jsonl"))]
        manifest = json.load(open(os.path.join(out, "manifest.json")))
        store = VectorStore.load(os.path.join(out, "vectors.pt"))
        pos = {c["id"]: n for n, c in enumerate(chunks)}
        files, refused = scan(src, cfg.get("tenants_from_dirs", False))
        on_disk = {rel: (p, tenant) for rel, p, tenant in files}
        hashes = {rel: sha256(p) for rel, (p, _) in on_disk.items()}
        removed = sorted(set(manifest) - set(on_disk))
        changed = sorted(r for r in on_disk if r in manifest and manifest[r]["sha256"] != hashes[r])
        added = sorted(set(on_disk) - set(manifest))
        gone = [cid for r in removed + changed for cid in manifest[r]["chunks"]]
        for cid in gone:
            chunks[pos[cid]]["deleted"] = True
        store.delete(pos[cid] for cid in gone)
        for r in removed:
            del manifest[r]
        new = []
        for r in changed + added:
            p, tenant = on_disk[r]
            try:
                recs = file_chunks(p, r, tenant)
            except Exception as e:
                refused.append({"source": r, "reason": f"unreadable: {type(e).__name__}: {str(e)[:80]}"})
                manifest.pop(r, None)
                continue
            for c in recs:
                c["id"] = f"c{len(chunks) + len(new)}"
                new.append(c)
            manifest[r] = {"sha256": hashes[r], "tenant": tenant, "chunks": [c["id"] for c in recs]}
        if new:
            store.append(axon.encode_store([c["text"] for c in new], log=log))
            chunks += new
        if cfg.get("plaid") and (gone or new):
            try:
                pl = _plaid(out, override=False)
                if gone:
                    pl.remove_documents(documents_ids=gone)
                if new:
                    first = len(chunks) - len(new)
                    pl.add_documents(documents_ids=[c["id"] for c in new],
                                     documents_embeddings=[store.doc(first + k).float().numpy() for k in range(len(new))])
            except Exception as e:                                     # fall back to exact search, never wrong
                cfg["plaid"], cfg["plaid_error"] = False, f"update failed, exact search in use: {repr(e)[:200]}"
        cfg["version"] = cfg.get("version", 1) + (1 if (gone or new) else 0)
        cfg["skipped"] = refused
        _write(out, cfg, chunks, store, manifest)
    return {"added": len(added), "changed": len(changed), "removed": len(removed),
            "unchanged": len(on_disk) - len(added) - len(changed), "chunks_added": len(new),
            "chunks_deleted": len(gone), "live_chunks": store.n_live, "dead_chunks": len(store) - store.n_live,
            "version": cfg["version"], "seconds": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- compact

def compact(out: str) -> dict:
    """Drop deleted chunks for good and renumber; no re-encoding."""
    t0 = time.time()
    with locked(out):
        cfg = read_config(out)
        chunks = [json.loads(line) for line in open(os.path.join(out, "chunks.jsonl"))]
        manifest = json.load(open(os.path.join(out, "manifest.json")))
        store = VectorStore.load(os.path.join(out, "vectors.pt"))
        before = len(chunks)
        new_store, keep = store.compacted()
        remap = {}
        kept = []
        for n, old in enumerate(keep):
            c = dict(chunks[old])
            c.pop("deleted", None)
            remap[c["id"]] = f"c{n}"
            c["id"] = f"c{n}"
            kept.append(c)
        for m in manifest.values():
            m["chunks"] = [remap[cid] for cid in m["chunks"] if cid in remap]
        if cfg.get("plaid") or cfg.get("plaid_error"):
            try:
                _plaid_build(out, kept, new_store)
                cfg["plaid"] = True
                cfg.pop("plaid_error", None)
            except Exception as e:
                cfg["plaid"], cfg["plaid_error"] = False, repr(e)[:300]
        cfg["version"] = cfg.get("version", 1) + 1
        _write(out, cfg, kept, new_store, manifest)
    return {"chunks_before": before, "chunks_after": len(kept), "version": cfg["version"],
            "seconds": round(time.time() - t0, 2)}
