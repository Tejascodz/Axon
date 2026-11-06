"""Index lifecycle: build, incremental update, compaction, consistent loading.

An index folder (format 3) holds one committed version at a time, named by config.json:

  config.json              the commit point: format, model, version, counts, options, and `files`,
                           the exact files of this version
  chunks.v<N>.jsonl        one record per chunk: id, source, part, text, tenant (optional), deleted (optional)
  manifest.v<N>.json       per source file: content hash and chunk ids, so updates re-encode only what changed
  vectors.v<N>.meta        document boundaries and the live mask of the vector store
  vectors.<M>.f16          the token vectors (raw fp16), shared by versions until a build or compaction
  plaid.v<M>/              optional approximate-search index; plaid.v<M>.version names the index
                           version it matches

A writer takes an exclusive lock, writes the new version's files under new names (vectors are
appended to the shared data file; readers only ever read the prefix their version knows), and
commits by renaming a new config.json into place. A crash at any point leaves the previous version
intact and loadable; the next writer clears what the crashed one left. Readers read config.json
first and then exactly the files it names, so they never mix two versions. Files of the version
before the current one are kept, so a reader that started just before a commit can finish; older
ones are deleted.

`update` compares files by SHA-256: new files are chunked and encoded, removed files' chunks are
marked deleted (hidden from search at once), changed files are both. Nothing else is re-encoded.
`compact` drops deleted chunks for good without re-encoding anything. Chunk ids never change (new
chunks get the next unused id), so compaction keeps the PLAID index: deleted chunks already left it
at update time. Only a PLAID index that no longer matches is rebuilt.
Indexes written by earlier releases (format 2: chunks.jsonl, manifest.json, vectors.pt) load as
they are and are migrated by the next update or compaction.
"""
from __future__ import annotations

import contextlib
import fcntl
import glob
import hashlib
import json
import os
import shutil
import time

from .store import VectorStore

FORMAT = 3
EXTS = (".txt", ".md", ".pdf")
PLAID_BATCH = 50_000                    # documents handed to PLAID at once (bounds the float32 copy)
_LEGACY = {"chunks": "chunks.jsonl", "manifest": "manifest.json", "vectors": "vectors.pt"}


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
    with open(os.path.join(out, ".lock"), "w", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another process is changing the index in {out}") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _fsync_dir(path):
    """Make a rename durable: the directory entry itself must reach the disk."""
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path) or ".")


def _atomic_lines(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_config(out: str) -> dict:
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def check_model(cfg: dict, axon):
    """Query and document vectors are only comparable from the same model: refuse a mismatch."""
    mine = (axon.model_name, getattr(axon, "revision", None))
    built = (cfg.get("model"), cfg.get("model_revision")) if "model_revision" in cfg else (cfg.get("model"), mine[1])
    if built != mine:
        raise RuntimeError(f"the index was built with {built[0]} (revision {built[1]}), this process has "
                           f"{mine[0]} (revision {mine[1]}): vectors would not be comparable")


# --------------------------------------------------------------------------- reading

def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _inside(out: str, name: str) -> str:
    """A file named by config.json, which must be a plain file name inside the index folder."""
    if not isinstance(name, str) or not name or os.path.basename(name) != name or name in (".", ".."):
        raise ValueError(f"index in {out} names an invalid file: {name!r}")
    return os.path.join(out, name)


def load(out: str, cfg: dict | None = None):
    """(config, chunks, store, manifest) of the committed version, all from the files it names."""
    cfg = cfg or read_config(out)
    files = cfg.get("files")
    if files is None:                                                 # format 2
        chunks = _read_lines(os.path.join(out, _LEGACY["chunks"]))
        mp = os.path.join(out, _LEGACY["manifest"])
        manifest = _read_json(mp) if os.path.exists(mp) else {}
        store = VectorStore.load(os.path.join(out, _LEGACY["vectors"]))
    else:
        chunks = _read_lines(_inside(out, files["chunks"]))
        manifest = _read_json(_inside(out, files["manifest"]))
        store = VectorStore.open(_inside(out, files["data"]), _inside(out, files["meta"]))
    if len(store) != len(chunks):
        raise ValueError(f"index in {out} is inconsistent: {len(chunks)} chunks, {len(store)} vector sets")
    return cfg, chunks, store, manifest


def plaid_dir(cfg: dict) -> str:
    name = cfg.get("plaid_dir") or "plaid"                            # format 2 used one fixed name
    if os.path.basename(name) != name or name in (".", ".."):
        raise ValueError(f"invalid PLAID folder name in config.json: {name!r}")
    return name


def plaid_current(out: str, cfg: dict) -> bool:
    """The PLAID index matches this version (an interrupted update can leave it ahead or behind)."""
    if not cfg.get("plaid"):
        return False
    name = plaid_dir(cfg)
    if cfg.get("files") is None:                                      # format 2 kept no marker
        return os.path.isdir(os.path.join(out, name))
    try:
        with open(_inside(out, f"{name}.version"), encoding="utf-8") as f:
            return int(f.read().strip()) == cfg["version"]
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- writing

def _commit(out, cfg, chunks, store, manifest, data_name, previous_plaid=None):
    """Write this version's files under new names, then config.json (the commit), then clear what
    neither this version nor the previous one uses."""
    v = cfg["version"]
    files = {"chunks": f"chunks.v{v}.jsonl", "manifest": f"manifest.v{v}.json", "meta": f"vectors.v{v}.meta",
             "data": data_name}
    _atomic_lines(os.path.join(out, files["chunks"]), chunks)
    _atomic_json(os.path.join(out, files["manifest"]), manifest)
    store.write_meta(os.path.join(out, files["meta"]))
    previous = cfg.get("files") or dict(_LEGACY)
    cfg.update(format=FORMAT, files=files, previous_files=previous, n=len(chunks), n_live=store.n_live,
               multi_tenant=any("tenant" in c for c in chunks),
               updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _atomic_json(os.path.join(out, "config.json"), cfg)               # the commit point
    keep = set(files.values()) | set(previous.values()) | {"config.json", ".lock"}
    for d in {cfg.get("plaid_dir"), previous_plaid} - {None}:
        keep |= {d, f"{d}.version"}
    _collect(out, keep)


def _collect(out, keep):
    pats = ("chunks.v*.jsonl", "manifest.v*.json", "vectors.v*.meta", "vectors.*.f16", "plaid.v*", "*.tmp",
            "plaid", *_LEGACY.values())
    for pat in pats:
        for p in glob.glob(os.path.join(out, pat)):
            if os.path.basename(p) in keep:
                continue
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(p)


def _plaid(out, name, override):
    from pylate import indexes
    return indexes.PLAID(index_folder=out, index_name=name, override=override, show_progress=False)


def _plaid_add(pl, chunks, store, positions, shuffle=False, log=None):
    """Hand documents to PLAID in batches: it wants float32 arrays, so all at once would need a
    float32 copy of every vector in memory. PLAID learns its centroids from the first batch, so for
    a new index the order is shuffled (fixed seed): the first batch is a sample of the whole
    collection, not of whichever files sort first."""
    if shuffle:
        import random
        positions = list(positions)
        random.Random(0).shuffle(positions)
    for i in range(0, len(positions), PLAID_BATCH):
        part = positions[i:i + PLAID_BATCH]
        t = time.time()
        pl.add_documents(documents_ids=[chunks[j]["id"] for j in part],
                         documents_embeddings=[store.doc(j).float().numpy() for j in part])
        if log:
            log(f"PLAID: {min(i + PLAID_BATCH, len(positions))}/{len(positions)} documents ({time.time() - t:.0f} s for this batch)")


def _mark_plaid(out, name, version):
    tmp = os.path.join(out, f"{name}.version.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(version))
    os.replace(tmp, os.path.join(out, f"{name}.version"))


def _plaid_build(out, cfg, chunks, store, log=None):
    """A fresh PLAID index under a new name (the one a running server uses is left alone)."""
    name = f"plaid.v{cfg['version']}"
    try:
        _plaid_add(_plaid(out, name, override=True), chunks, store,
                   [i for i in range(len(chunks)) if bool(store.live[i])], shuffle=True, log=log)
        _mark_plaid(out, name, cfg["version"])
        cfg.update(plaid=True, plaid_dir=name)
        cfg.pop("plaid_error", None)
    except Exception as e:                                            # approximate search is optional
        cfg.update(plaid=False, plaid_error=repr(e)[:300])
        cfg.pop("plaid_dir", None)
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)


# --------------------------------------------------------------------------- build

def build(src_or_chunks, out: str, axon, plaid: bool = False, tenants_from_dirs: bool = False, log=None) -> dict:
    """Index a folder, or a list of {"source", "text"[, "tenant"]} records, from scratch. Replaces
    any index already in `out` with one commit; a running server switches over on its next request."""
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
        old = read_config(out) if os.path.exists(os.path.join(out, "config.json")) else {}
        version = old.get("version", 0) + 1
        data_name = f"vectors.{version}.f16"
        store = axon.encode_store([c["text"] for c in chunks], log=log, path=os.path.join(out, data_name))
        cfg = {"model": axon.model_name, "model_revision": getattr(axon, "revision", None), "version": version,
               "plaid": False, "skipped": refused, "tenants_from_dirs": tenants_from_dirs, "next_id": len(chunks),
               "source": src_or_chunks if isinstance(src_or_chunks, str) else None}
        if old.get("files"):
            cfg["files"] = old["files"]                               # becomes previous_files at commit
        if plaid:
            _plaid_build(out, cfg, chunks, store, log=log)
        previous_plaid = plaid_dir(old) if old.get("plaid") else None
        _commit(out, cfg, chunks, store, manifest, data_name, previous_plaid)
    return {"chunks": len(chunks), "refused": refused, "seconds": round(time.time() - t0, 2),
            "vector_mb": round(store.nbytes() / 2 ** 20, 1), "plaid": cfg["plaid"], "version": version}


# --------------------------------------------------------------------------- update

def update(src: str, out: str, axon, log=None) -> dict:
    """Bring the index in line with the folder: encode new and changed files only."""
    t0 = time.time()
    with locked(out):
        cfg, chunks, store, manifest = load(out)
        check_model(cfg, axon)
        pos = {c["id"]: n for n, c in enumerate(chunks)}
        files, refused = scan(src, cfg.get("tenants_from_dirs", False))
        on_disk = {rel: (p, tenant) for rel, p, tenant in files}
        hashes = {rel: sha256(p) for rel, (p, _) in on_disk.items()}
        removed = sorted(set(manifest) - set(on_disk))
        changed = sorted(r for r in on_disk if r in manifest and manifest[r]["sha256"] != hashes[r])
        added = sorted(set(on_disk) - set(manifest))
        gone = [cid for r in removed + changed for cid in manifest[r]["chunks"]]
        report = {"added": len(added), "changed": len(changed), "removed": len(removed),
                  "unchanged": len(on_disk) - len(added) - len(changed)}
        for r in removed:
            del manifest[r]
        new = []
        next_id = cfg.get("next_id", len(chunks))
        for r in changed + added:
            p, tenant = on_disk[r]
            try:
                recs = file_chunks(p, r, tenant)
            except Exception as e:
                refused.append({"source": r, "reason": f"unreadable: {type(e).__name__}: {str(e)[:80]}"})
                manifest.pop(r, None)                                  # its old chunks are deleted above
                continue
            for c in recs:
                c["id"] = f"c{next_id}"
                next_id += 1
                new.append(c)
            manifest[r] = {"sha256": hashes[r], "tenant": tenant, "chunks": [c["id"] for c in recs]}
        if not gone and not new:                                        # nothing changed: nothing to commit
            return {**report, "chunks_added": 0, "chunks_deleted": 0, "live_chunks": store.n_live,
                    "dead_chunks": len(store) - store.n_live, "version": cfg["version"],
                    "seconds": round(time.time() - t0, 2)}
        plaid_ok = plaid_current(out, cfg)
        if cfg.get("plaid"):
            cfg["plaid_dir"] = plaid_dir(cfg)
        cfg["version"] += 1
        data_name = (cfg.get("files") or {}).get("data")
        if data_name is None:                                           # format 2: move the vectors into a data file
            data_name = f"vectors.{cfg['version']}.f16"
            store.write_data(os.path.join(out, data_name))
        for cid in gone:
            chunks[pos[cid]]["deleted"] = True
        store.delete(pos[cid] for cid in gone)
        if new:
            store.append_to(axon.encode_store([c["text"] for c in new], log=log))
            chunks += new
        if cfg.get("plaid"):
            if not plaid_ok:
                cfg.update(plaid=False, plaid_error="PLAID did not match the index (an update was interrupted); "
                                                    "exact search in use until `compact` rebuilds it")
            else:
                try:
                    pl = _plaid(out, cfg["plaid_dir"], override=False)
                    if gone:
                        pl.remove_documents(documents_ids=gone)
                    if new:
                        _plaid_add(pl, chunks, store, list(range(len(chunks) - len(new), len(chunks))))
                    _mark_plaid(out, cfg["plaid_dir"], cfg["version"])
                except Exception as e:                                 # fall back to exact search, never wrong
                    cfg.update(plaid=False, plaid_error=f"update failed, exact search in use: {repr(e)[:200]}")
        cfg["skipped"], cfg["next_id"] = refused, next_id
        _commit(out, cfg, chunks, store, manifest, data_name)
    return {**report, "chunks_added": len(new), "chunks_deleted": len(gone), "live_chunks": store.n_live,
            "dead_chunks": len(store) - store.n_live, "version": cfg["version"], "seconds": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- compact

def compact(out: str) -> dict:
    """Drop deleted chunks for good; no re-encoding. Chunk ids are kept, so a PLAID index that
    matches the index is kept as it is (its deleted chunks were removed at update time); one that
    does not match (or a failed one) is rebuilt."""
    t0 = time.time()
    with locked(out):
        cfg, chunks, store, manifest = load(out)
        before = len(chunks)
        plaid_ok = plaid_current(out, cfg)
        previous_plaid = plaid_dir(cfg) if cfg.get("plaid") else None
        cfg.setdefault("next_id", 1 + max((int(c["id"][1:]) for c in chunks if c["id"][1:].isdigit()), default=-1))
        cfg["version"] += 1
        data_name = f"vectors.{cfg['version']}.f16"
        new_store, keep = store.compacted(path=os.path.join(out, data_name))
        kept = []
        for old in keep:
            c = dict(chunks[old])
            c.pop("deleted", None)
            kept.append(c)
        live_ids = {c["id"] for c in kept}
        for m in manifest.values():
            m["chunks"] = [cid for cid in m["chunks"] if cid in live_ids]
        plaid = "none"
        if cfg.get("plaid") and plaid_ok:
            cfg["plaid_dir"] = plaid_dir(cfg)
            _mark_plaid(out, cfg["plaid_dir"], cfg["version"])
            plaid = "kept"
        elif cfg.get("plaid") or cfg.get("plaid_error"):
            _plaid_build(out, cfg, kept, new_store)
            plaid = "rebuilt" if cfg.get("plaid") else "failed"
        _commit(out, cfg, kept, new_store, manifest, data_name, previous_plaid)
    return {"chunks_before": before, "chunks_after": len(kept), "version": cfg["version"], "plaid": plaid,
            "seconds": round(time.time() - t0, 2)}
