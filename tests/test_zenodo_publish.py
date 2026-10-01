"""scripts/zenodo_publish.py against a fake, in-process Zenodo.

No test here touches the network: every request goes to a loopback
http.server that implements just enough of the deposit and records APIs
(create, newversion, metadata PUT, file DELETE, bucket PUT, publish, public
record GET) to drive the script end to end. The token is a throwaway string
in a 0600 file under tmp_path.

The safety properties are what these tests are mostly for:
  * the token never appears in stdout, stderr, an exception or a URL, even
    when the server echoes it back in an error body;
  * --dry-run makes no request at all (and does not read the token);
  * nothing is published without a TTY and the typed title.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "zenodo_publish.py"
)
TOKEN = "FAKE-tok3n-7f1d0c2b9e8a-do-not-leak"


def _load():
    spec = importlib.util.spec_from_file_location("_zenodo_publish", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_zenodo_publish"] = module
    spec.loader.exec_module(module)
    return module


zp = _load()


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


# --- the fake Zenodo ----------------------------------------------------------


class FakeZenodo:
    def __init__(self):
        self.deps: dict[int, dict] = {}
        self.buckets: dict[str, dict[str, bytes]] = {}
        self.log: list[tuple[str, str, str | None]] = []
        self.next_id = 1000
        self.next_fid = 1
        self.fail_5xx = 0  # fail this many requests with 503 first
        self.echo_auth_on_403 = False
        self.corrupt_listing: set[str] = set()  # report wrong md5 for these
        self.published: list[int] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    # state helpers
    def new_dep(self, metadata, files=None, published=False):
        dep_id = self.next_id
        self.next_id += 1
        bucket = f"b{dep_id}"
        self.buckets[bucket] = dict(files or {})
        self.deps[dep_id] = {
            "id": dep_id,
            "metadata": dict(metadata),
            "bucket": bucket,
            "fids": {name: self._fid() for name in self.buckets[bucket]},
            "published": published,
        }
        return dep_id

    def _fid(self):
        self.next_fid += 1
        return f"f{self.next_fid}"

    def dep_json(self, dep_id):
        d = self.deps[dep_id]
        files = []
        for name, data in sorted(self.buckets[d["bucket"]].items()):
            md5 = _md5(data)
            if name in self.corrupt_listing:
                md5 = "0" * 32
            files.append(
                {
                    "id": d["fids"][name],
                    "filename": name,
                    "filesize": len(data),
                    "checksum": md5,
                }
            )
        return {
            "id": dep_id,
            "metadata": d["metadata"],
            "submitted": d["published"],
            "state": "done" if d["published"] else "unsubmitted",
            "files": files,
            "links": {
                "bucket": f"{self.base}/files/{d['bucket']}",
                "html": f"https://zenodo.org/uploads/{dep_id}",
            },
        }

    def record_json(self, dep_id):
        d = self.deps[dep_id]
        return {
            "id": dep_id,
            "metadata": d["metadata"],
            "files": [
                {"key": n, "size": len(b), "checksum": "md5:" + _md5(b)}
                for n, b in sorted(self.buckets[d["bucket"]].items())
            ],
        }

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload=None):
                body = b"" if payload is None else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def _dispatch(self, method):
                auth = self.headers.get("Authorization")
                fake.log.append((method, self.path, auth))
                body = self._body()
                if fake.fail_5xx:
                    fake.fail_5xx -= 1
                    return self._send(503, {"message": "busy"})
                parts = self.path.split("?")[0].strip("/").split("/")
                if parts[:2] == ["api", "records"] and method == "GET":
                    dep_id = int(parts[2])
                    if not fake.deps[dep_id]["published"]:
                        return self._send(404, {"message": "not found"})
                    return self._send(200, fake.record_json(dep_id))
                if auth != f"Bearer {TOKEN}":
                    msg = "bad token"
                    if fake.echo_auth_on_403:
                        msg = f"bad token {auth}"
                    return self._send(403, {"message": msg})
                if parts[0] == "files" and method == "PUT":
                    bucket, name = parts[1], urllib.request.unquote(parts[2])
                    fake.buckets[bucket][name] = body
                    dep_id = int(bucket[1:])
                    fake.deps[dep_id]["fids"][name] = fake._fid()
                    return self._send(
                        201,
                        {
                            "key": name,
                            "size": len(body),
                            "checksum": "md5:" + _md5(body),
                        },
                    )
                if parts[:3] != ["api", "deposit", "depositions"]:
                    return self._send(404, {"message": "no route"})
                if len(parts) == 3 and method == "POST":
                    meta = json.loads(body)["metadata"]
                    return self._send(201, fake.dep_json(fake.new_dep(meta)))
                dep_id = int(parts[3])
                if dep_id not in fake.deps:
                    return self._send(404, {"message": "no deposition"})
                if len(parts) == 4 and method == "GET":
                    return self._send(200, fake.dep_json(dep_id))
                if len(parts) == 4 and method == "PUT":
                    fake.deps[dep_id]["metadata"] = json.loads(body)[
                        "metadata"
                    ]
                    return self._send(200, fake.dep_json(dep_id))
                if parts[4:] == ["actions", "newversion"] and method == "POST":
                    src = fake.deps[dep_id]
                    new = fake.new_dep(
                        src["metadata"], fake.buckets[src["bucket"]]
                    )
                    return self._send(
                        201,
                        {
                            "id": dep_id,
                            "links": {
                                "latest_draft": (
                                    f"{fake.base}/api/deposit/depositions/{new}"
                                )
                            },
                        },
                    )
                if parts[4:] == ["actions", "publish"] and method == "POST":
                    fake.deps[dep_id]["published"] = True
                    fake.published.append(dep_id)
                    return self._send(202, fake.dep_json(dep_id))
                if parts[4] == "files" and method == "DELETE":
                    d = fake.deps[dep_id]
                    name = next(
                        n for n, f in d["fids"].items() if f == parts[5]
                    )
                    del fake.buckets[d["bucket"]][name]
                    del d["fids"][name]
                    return self._send(204)
                return self._send(404, {"message": "no route"})

            def do_GET(self):
                self._dispatch("GET")

            def do_POST(self):
                self._dispatch("POST")

            def do_PUT(self):
                self._dispatch("PUT")

            def do_DELETE(self):
                self._dispatch("DELETE")

        return H


@pytest.fixture
def fake(monkeypatch):
    server = FakeZenodo()
    monkeypatch.setenv(zp.BASE_URL_ENV, server.base)
    monkeypatch.setattr(zp, "BACKOFF", 0.0)
    monkeypatch.setattr(zp, "_SECRETS", [])
    yield server
    server.close()


@pytest.fixture
def token_file(tmp_path):
    p = tmp_path / "token"
    p.write_text(TOKEN + "\n")
    p.chmod(0o600)
    return p


@pytest.fixture
def bundle(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "grid.parquet").write_bytes(b"x" * 300_000)
    (d / "README.txt").write_bytes(b"new readme\n")
    return d


METADATA = {
    "title": "EXOZIPPy test grid",
    "creators": [{"name": "Eastman, Jason"}],
    "description": "test",
    "license": "cc-by-4.0",
    "upload_type": "dataset",
}


def _run(capsys, *argv):
    rc = zp.main(list(argv))
    out, err = capsys.readouterr()
    assert TOKEN not in out and TOKEN not in err
    return rc, out, err


def _assert_auth_only_in_headers(fake):
    for _method, path, _auth in fake.log:
        assert TOKEN not in path


# --- tests --------------------------------------------------------------------


def test_new_version_replaces_and_verifies(fake, token_file, bundle, capsys):
    """
    Given a published record carrying an old README.txt and an unrelated file,
    When new-version uploads a bundle holding README.txt and grid.parquet,
    Then the old README is replaced, the unrelated file is kept, the draft
    holds the bundle plus a MANIFEST.txt, and nothing is published.
    """
    # Arrange
    rec = fake.new_dep(
        METADATA,
        {"README.txt": b"old readme\n", "keep.dat": b"k"},
        published=True,
    )
    # Act
    rc, out, err = _run(
        capsys,
        "new-version",
        "--record",
        str(rec),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
    )
    # Assert
    assert rc == 0, err
    draft = max(fake.deps)
    files = fake.buckets[fake.deps[draft]["bucket"]]
    assert files["README.txt"] == b"new readme\n"
    assert files["grid.parquet"] == b"x" * 300_000
    assert files["keep.dat"] == b"k"
    manifest = (bundle / "MANIFEST.txt").read_text()
    assert files["MANIFEST.txt"] == manifest.encode()
    assert f"grid.parquet 300000 {_md5(b'x' * 300_000)}" in manifest
    assert "DRAFT READY (not published)" in out
    assert f"https://zenodo.org/uploads/{draft}" in out
    assert (
        f'"url": "https://zenodo.org/records/{draft}/files/grid.parquet"'
        in out
    )
    assert fake.published == []
    assert all(a == f"Bearer {TOKEN}" for m, p, a in fake.log)
    _assert_auth_only_in_headers(fake)


def test_new_version_prune_removes_unlisted(fake, token_file, bundle, capsys):
    """
    Given a record carrying a file the bundle does not have,
    When new-version runs with --prune,
    Then the draft holds exactly the bundle.
    """
    rec = fake.new_dep(METADATA, {"keep.dat": b"k"}, published=True)
    rc, _, err = _run(
        capsys,
        "new-version",
        "--record",
        str(rec),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
        "--prune",
    )
    assert rc == 0, err
    files = fake.buckets[fake.deps[max(fake.deps)]["bucket"]]
    assert sorted(files) == ["MANIFEST.txt", "README.txt", "grid.parquet"]


def test_new_record_sets_metadata_and_community(
    fake, token_file, bundle, tmp_path, capsys
):
    """
    Given a YAML metadata file without communities,
    When new-record runs,
    Then the draft carries the metadata and the exozippy community.
    """
    import yaml

    meta = tmp_path / "meta.yaml"
    meta.write_text(yaml.safe_dump(METADATA))
    rc, out, err = _run(
        capsys,
        "new-record",
        "--bundle",
        str(bundle),
        "--metadata",
        str(meta),
        "--token-file",
        str(token_file),
    )
    assert rc == 0, err
    dep = fake.deps[max(fake.deps)]
    assert dep["metadata"]["title"] == METADATA["title"]
    assert dep["metadata"]["communities"] == [{"identifier": "exozippy"}]
    assert sorted(fake.buckets[dep["bucket"]]) == [
        "MANIFEST.txt",
        "README.txt",
        "grid.parquet",
    ]
    assert fake.published == []


def test_new_record_requires_metadata(
    fake, token_file, bundle, tmp_path, capsys
):
    """Missing required metadata fails before any request is made."""
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"title": "t"}))
    rc, _, err = _run(
        capsys,
        "new-record",
        "--bundle",
        str(bundle),
        "--metadata",
        str(meta),
        "--token-file",
        str(token_file),
    )
    assert rc == 1
    assert "missing required metadata" in err and "creators" in err
    assert fake.log == []


def test_verify_mismatch_fails_loudly(fake, token_file, bundle, capsys):
    """
    Given a server whose listing reports a wrong md5 for grid.parquet,
    When the upload is verified,
    Then the run fails, names the file, says not to publish, and does not
    print the DRAFT READY banner.
    """
    rec = fake.new_dep(METADATA, {}, published=True)
    fake.corrupt_listing.add("grid.parquet")
    rc, out, err = _run(
        capsys,
        "new-version",
        "--record",
        str(rec),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
    )
    assert rc == 1
    assert (
        "FAILED" in err and "grid.parquet" in err and "do not publish" in err
    )
    assert "DRAFT READY" not in out


def test_resume_skips_already_uploaded(fake, token_file, bundle, capsys):
    """
    Given a draft that already holds grid.parquet with the right md5,
    When `upload --draft` resumes,
    Then grid.parquet is not re-uploaded and the rest is.
    """
    data = (bundle / "grid.parquet").read_bytes()
    draft = fake.new_dep(METADATA, {"grid.parquet": data})
    rc, out, err = _run(
        capsys,
        "upload",
        "--draft",
        str(draft),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
    )
    assert rc == 0, err
    puts = [
        p for m, p, _ in fake.log if m == "PUT" and p.startswith("/files/")
    ]
    assert not any(p.endswith("/grid.parquet") for p in puts)
    assert any(p.endswith("/README.txt") for p in puts)
    assert "skip    grid.parquet" in out


def test_upload_retries_5xx(fake, token_file, bundle, capsys):
    """Two 503s are retried through; the run still succeeds."""
    draft = fake.new_dep(METADATA)
    fake.fail_5xx = 2
    rc, _, err = _run(
        capsys,
        "upload",
        "--draft",
        str(draft),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
    )
    assert rc == 0, err
    assert "attempt 1/" in err


def test_token_never_leaks_from_error_bodies(fake, token_file, bundle, capsys):
    """
    Given a server that echoes the Authorization header in a 403 body,
    When the script fails on it,
    Then the printed error is redacted and so is the exception text.
    """
    fake.echo_auth_on_403 = True
    bad = token_file.parent / "badtoken"
    bad.write_text("WRONG-" + TOKEN)
    bad.chmod(0o600)
    draft = fake.new_dep(METADATA)
    # Wrong token -> server echoes "Bearer WRONG-<TOKEN>" back.
    rc, _, err = _run(
        capsys,
        "upload",
        "--draft",
        str(draft),
        "--bundle",
        str(bundle),
        "--token-file",
        str(bad),
    )
    assert rc == 1
    assert "<redacted>" in err and "HTTP 403" in err
    # And at the exception level, too.
    client = zp.Client(fake.base, zp.read_token(token_file))
    zp._SECRETS.append("WRONG-" + TOKEN)
    client._token = "WRONG-" + TOKEN
    with pytest.raises(zp.ZenodoError) as exc:
        client.request("GET", client.url(f"/api/deposit/depositions/{draft}"))
    assert TOKEN not in str(exc.value)
    _assert_auth_only_in_headers(fake)


def test_unexpected_exception_is_redacted(
    fake, token_file, bundle, capsys, monkeypatch
):
    """An unexpected crash prints a traceback with the token redacted."""
    draft = fake.new_dep(METADATA)

    def boom(*a, **k):
        raise KeyError(f"surprise {TOKEN}")

    monkeypatch.setattr(zp, "sync_files", boom)
    rc, _, err = _run(
        capsys,
        "upload",
        "--draft",
        str(draft),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
    )
    assert rc == 1
    assert "KeyError" in err and "<redacted>" in err


def test_dry_run_makes_no_requests(
    fake, bundle, tmp_path, capsys, monkeypatch
):
    """
    Given --dry-run and a token file that does not exist,
    When each subcommand runs,
    Then no request reaches the server, urllib is never asked to open
    anything, no MANIFEST.txt is written, and the token is not needed.
    """

    def no_http(*a, **k):
        raise AssertionError("dry run attempted HTTP")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", no_http)
    monkeypatch.setattr(urllib.request, "urlopen", no_http)
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps(METADATA))
    missing = str(tmp_path / "no-such-token")
    for argv in (
        ["new-version", "--record", "1", "--bundle", str(bundle)],
        ["new-record", "--bundle", str(bundle), "--metadata", str(meta)],
        ["upload", "--draft", "1", "--bundle", str(bundle)],
        ["pin", "--record", "1"],
    ):
        rc, out, err = _run(
            capsys, *argv, "--dry-run", "--token-file", missing
        )
        assert rc == 0, err
        assert "dry-run" in out
    assert fake.log == []
    assert not (bundle / "MANIFEST.txt").exists()


def test_publish_refused_without_tty(fake, token_file, bundle, capsys):
    """--publish on a non-TTY verifies the draft, then refuses to publish."""
    draft = fake.new_dep(METADATA)
    rc, out, err = _run(
        capsys,
        "upload",
        "--draft",
        str(draft),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
        "--publish",
    )
    assert rc == 1
    assert "refusing" in err
    assert fake.published == []
    assert not any(p.endswith("/publish") for _, p, _ in fake.log)


def test_publish_refused_on_wrong_title(
    fake, token_file, bundle, capsys, monkeypatch
):
    """A TTY but a mistyped title does not publish."""
    draft = fake.new_dep(METADATA)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "EXOZIPPy test grd")
    rc, _, err = _run(
        capsys,
        "upload",
        "--draft",
        str(draft),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
        "--publish",
    )
    assert rc == 1 and "NOT published" in err
    assert fake.published == []


def test_publish_with_typed_title(
    fake, token_file, bundle, capsys, monkeypatch
):
    """Only the exact typed title at a TTY publishes (to the fake)."""
    draft = fake.new_dep(METADATA)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: METADATA["title"])
    rc, _, err = _run(
        capsys,
        "upload",
        "--draft",
        str(draft),
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
        "--publish",
    )
    assert rc == 0, err
    assert fake.published == [draft]


def test_pin_reads_published_record_without_token(fake, bundle, capsys):
    """
    Given a published record,
    When pin runs with no token file at all,
    Then it prints a registry entry with each file's url, size and md5,
    excluding MANIFEST.txt, and sends no Authorization header.
    """
    data = (bundle / "grid.parquet").read_bytes()
    rec = fake.new_dep(
        METADATA, {"grid.parquet": data, "MANIFEST.txt": b"m"}, published=True
    )
    rc, out, err = _run(
        capsys, "pin", "--record", str(rec), "--token-file", "/nonexistent"
    )
    assert rc == 0, err
    assert (
        f'"url": "https://zenodo.org/records/{rec}/files/grid.parquet"' in out
    )
    assert f'"size": {len(data)},' in out
    assert f'"md5": "{_md5(data)}",' in out
    assert "MANIFEST.txt" not in out.split("{", 1)[1]
    assert all(a is None for _, _, a in fake.log)


def test_pin_rejects_draft(fake, capsys):
    """pin on an unpublished draft fails (the public API 404s)."""
    draft = fake.new_dep(METADATA)
    rc, _, err = _run(capsys, "pin", "--record", str(draft))
    assert rc == 1 and "HTTP 404" in err


def test_token_file_must_be_private(tmp_path):
    """A group/world readable token file is refused, naming chmod."""
    p = tmp_path / "token"
    p.write_text(TOKEN)
    p.chmod(0o644)
    with pytest.raises(zp.ZenodoError) as exc:
        zp.read_token(p)
    assert "chmod 600" in str(exc.value)
    assert TOKEN not in str(exc.value)


def test_stale_manifest_is_refused(fake, token_file, bundle, capsys):
    """A MANIFEST.txt that disagrees with the files fails before any HTTP."""
    (bundle / "MANIFEST.txt").write_text("grid.parquet 1 abc\n")
    rc, _, err = _run(
        capsys,
        "upload",
        "--draft",
        "1",
        "--bundle",
        str(bundle),
        "--token-file",
        str(token_file),
    )
    assert rc == 1 and "does not match" in err
    assert fake.log == []


def test_base_url_override_refuses_foreign_host(monkeypatch):
    """The test hook cannot redirect the token to a non-Zenodo host."""
    monkeypatch.setenv(zp.BASE_URL_ENV, "https://evil.example.com")
    with pytest.raises(zp.ZenodoError):
        zp.base_url(False)
