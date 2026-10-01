#!/usr/bin/env python3
"""Upload a data bundle to Zenodo as a DRAFT, verify it, and stop.

This is the upload side of ``src/exozippy/utilities/zenodo.py`` (which only
downloads). The workflow it serves is documented in ``docs/zenodo.md``:

    build bundle -> new-version / new-record draft -> a human reviews the
    draft on zenodo.org and clicks Publish -> ``pin`` -> PR

Subcommands
-----------
new-version --record ID --bundle DIR
    Create a new-version draft of the published record ID, delete the
    carried-over files the bundle replaces (or every file not in the bundle,
    with --prune), upload the bundle, re-read the draft and verify every
    file's size and md5.
new-record --bundle DIR --metadata FILE
    Create a fresh draft with the given metadata (YAML or JSON), upload,
    verify.
upload --draft ID --bundle DIR
    Resume into an existing, unpublished draft (after an interrupted run).
    Files already on the draft with a matching md5 are skipped.
pin --record ID
    Read a PUBLISHED record's public API (no token) and print the asset
    registry entry -- filename, url, size, md5 -- to paste into the module
    that downloads it.

Safety rules, deliberate and not to be relaxed
----------------------------------------------
* NOTHING IS PUBLISHED BY DEFAULT. Publishing mints a permanent DOI. The
  script prints the draft's web URL and exits; a human reviews and publishes.
  ``--publish`` exists, but refuses unless stdin is a TTY and the operator
  types the draft's exact title.
* The token is read from a FILE (``--token-file``, else
  ``$ZENODO_TOKEN_FILE``, else ``~/.config/zenodo/token``;
  ``~/.config/zenodo/sandbox_token`` under ``--sandbox``). Never from the
  command line (visible in ``ps``), never from an env var VALUE. The file
  must not be group/world readable. The token travels only in an
  Authorization header that urllib does NOT forward across redirects, and
  every message this script prints or raises is passed through a redactor.
* ``--dry-run`` prints the plan and makes no HTTP request at all (it does
  not even read the token).

Stdlib only (urllib): curl is unreliable on the development box and the
repository does not depend on requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PRODUCTION = "https://zenodo.org"
SANDBOX = "https://sandbox.zenodo.org"
DEFAULT_TOKEN_FILE = "~/.config/zenodo/token"
DEFAULT_SANDBOX_TOKEN_FILE = "~/.config/zenodo/sandbox_token"
TOKEN_FILE_ENV = "ZENODO_TOKEN_FILE"
# Test hook: point the client at a local fake Zenodo. Only loopback http and
# https *.zenodo.org hosts are accepted (see _check_base_url), so a typo or a
# hostile environment can never send the token anywhere else.
BASE_URL_ENV = "EXOZIPPY_ZENODO_BASE_URL"

MANIFEST_NAME = "MANIFEST.txt"
DEFAULT_COMMUNITY = "exozippy"
REQUIRED_METADATA = (
    "title",
    "creators",
    "description",
    "license",
    "upload_type",
)

# Transport retries: Zenodo 5xx-es under load (see utilities/zenodo.py).
ATTEMPTS = 5
BACKOFF = 2.0  # seconds; doubles each attempt
TIMEOUT = 120.0  # seconds of socket silence, per read/write, not per transfer
CHUNK = 1 << 20


class ZenodoError(RuntimeError):
    """A failure with an actionable, already-redacted message."""


# --- secrets ----------------------------------------------------------------

_SECRETS: list[str] = []


def redact(text: object) -> str:
    """str(text) with every registered secret replaced."""
    s = str(text)
    for secret in _SECRETS:
        if secret:
            s = s.replace(secret, "<redacted>")
    return s


def read_token(path: Path) -> str:
    """Read the token from `path`; refuse a missing, empty or exposed file.

    The token value is registered with the redactor before it is returned,
    and no message below ever includes it.
    """
    path = path.expanduser()
    if not path.is_file():
        raise ZenodoError(
            f"Token file {path} does not exist. Create a personal access "
            f"token at https://zenodo.org/account/settings/applications/ "
            f"(scopes deposit:write and deposit:actions), save it there and "
            f"`chmod 600 {path}`."
        )
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ZenodoError(
            f"Token file {path} is readable by group/others (mode "
            f"{stat.S_IMODE(mode):o}). Run `chmod 600 {path}` and retry."
        )
    token = path.read_text().strip()
    if not token:
        raise ZenodoError(f"Token file {path} is empty.")
    if any(c.isspace() for c in token):
        raise ZenodoError(
            f"Token file {path} must contain the token alone on one line."
        )
    _SECRETS.append(token)
    return token


def token_path(args) -> Path:
    if args.token_file:
        return Path(args.token_file).expanduser()
    if os.environ.get(TOKEN_FILE_ENV):
        return Path(os.environ[TOKEN_FILE_ENV]).expanduser()
    default = (
        DEFAULT_SANDBOX_TOKEN_FILE if args.sandbox else DEFAULT_TOKEN_FILE
    )
    return Path(default).expanduser()


def _check_base_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    loopback = parsed.scheme == "http" and host in ("127.0.0.1", "localhost")
    zenodo = parsed.scheme == "https" and (
        host == "zenodo.org" or host.endswith(".zenodo.org")
    )
    if not (loopback or zenodo):
        raise ZenodoError(
            f"{BASE_URL_ENV}={url!r} is not a zenodo.org https URL or a "
            f"loopback http URL; refusing to send credentials there."
        )
    return url.rstrip("/")


def base_url(sandbox: bool) -> str:
    override = os.environ.get(BASE_URL_ENV)
    if override:
        return _check_base_url(override)
    return SANDBOX if sandbox else PRODUCTION


# --- local bundle -------------------------------------------------------------


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def scan_bundle(bundle: Path) -> tuple[dict[str, dict], list[str]]:
    """{filename: {"path", "size", "md5"}} for the bundle, plus ignored names.

    Zenodo records are flat, so a subdirectory is refused rather than
    silently skipped or flattened. Dotfiles (.DS_Store, editor droppings) are
    ignored and reported.
    """
    if not bundle.is_dir():
        raise ZenodoError(f"Bundle {bundle} is not a directory.")
    files: dict[str, dict] = {}
    ignored: list[str] = []
    for p in sorted(bundle.iterdir()):
        if p.name.startswith("."):
            ignored.append(p.name)
            continue
        if p.is_dir():
            raise ZenodoError(
                f"Bundle {bundle} contains the subdirectory {p.name}/. Zenodo "
                f"records are flat: tar/zip it, or move its files up."
            )
        if not p.is_file():
            raise ZenodoError(f"Bundle entry {p} is not a regular file.")
        files[p.name] = {
            "path": p,
            "size": p.stat().st_size,
            "md5": md5_of(p),
        }
    data = [n for n in files if n != MANIFEST_NAME]
    if not data:
        raise ZenodoError(f"Bundle {bundle} has no files to upload.")
    return files, ignored


def manifest_text(files: dict[str, dict]) -> str:
    lines = [
        f"{name} {meta['size']} {meta['md5']}"
        for name, meta in sorted(files.items())
        if name != MANIFEST_NAME
    ]
    return "\n".join(lines) + "\n"


def ensure_manifest(bundle: Path, files: dict, dry_run: bool) -> dict:
    """Write MANIFEST.txt (name size md5 per line) if absent; check it if not.

    A present manifest that disagrees with the files is refused: it would be
    published as a description of bytes that are not in the record.
    """
    expected = manifest_text(files)
    mpath = bundle / MANIFEST_NAME
    if MANIFEST_NAME in files:
        if mpath.read_text() != expected:
            raise ZenodoError(
                f"{mpath} does not match the bundle's files (sizes/md5s or "
                f"file list differ). Delete it and rerun to regenerate it."
            )
        return files
    if dry_run:
        print(f"[dry-run] would write {mpath}")
        files = dict(files)
        data = expected.encode()
        files[MANIFEST_NAME] = {
            "path": mpath,
            "size": len(data),
            "md5": hashlib.md5(data).hexdigest(),
        }
        return files
    mpath.write_text(expected)
    print(f"Wrote {mpath}")
    files = dict(files)
    files[MANIFEST_NAME] = {
        "path": mpath,
        "size": mpath.stat().st_size,
        "md5": md5_of(mpath),
    }
    return files


def load_metadata(path: Path) -> dict:
    """Read deposition metadata from YAML or JSON and normalize it.

    ``communities`` may be given as plain identifiers (``[exozippy]``); they
    are translated to Zenodo's ``[{"identifier": ...}]`` here, and the
    exozippy community is added when no communities are given.
    """
    if not path.is_file():
        raise ZenodoError(f"Metadata file {path} does not exist.")
    text = path.read_text()
    if path.suffix.lower() == ".json":
        meta = json.loads(text)
    else:
        import yaml

        meta = yaml.safe_load(text)
    if not isinstance(meta, dict):
        raise ZenodoError(f"Metadata file {path} must hold a mapping.")
    if "metadata" in meta and len(meta) == 1:
        meta = meta["metadata"]
    return meta


def normalize_metadata(meta: dict, *, require_all: bool, source: str) -> dict:
    meta = dict(meta)
    if require_all:
        missing = [k for k in REQUIRED_METADATA if not meta.get(k)]
        if missing:
            raise ZenodoError(
                f"{source} is missing required metadata: {', '.join(missing)}."
            )
        meta.setdefault("communities", [DEFAULT_COMMUNITY])
    if "creators" in meta:
        creators = meta["creators"]
        if not isinstance(creators, list) or not all(
            isinstance(c, dict) and c.get("name") for c in creators
        ):
            raise ZenodoError(
                f"{source}: creators must be a list of mappings with a "
                f"'name' ('Family, Given'), e.g. [{{name: 'Eastman, Jason'}}]."
            )
    if "communities" in meta:
        comms = []
        for c in meta["communities"]:
            if isinstance(c, str):
                comms.append({"identifier": c})
            elif isinstance(c, dict) and c.get("identifier"):
                comms.append({"identifier": c["identifier"]})
            else:
                raise ZenodoError(
                    f"{source}: community entry {c!r} is neither a name nor "
                    f"a mapping with an 'identifier'."
                )
        meta["communities"] = comms
    return meta


# --- HTTP ---------------------------------------------------------------------


class Client:
    """Minimal Zenodo deposit API client over urllib.

    The Authorization header is added with ``add_unredirected_header`` so a
    redirect (to a CDN, an object store, anywhere) never carries the token.
    """

    def __init__(self, base: str, token: str | None):
        self.base = base
        self._token = token
        self.opener = urllib.request.build_opener()

    def url(self, path: str) -> str:
        return self.base + path

    def request(
        self,
        method: str,
        url: str,
        *,
        body: dict | None = None,
        upload: Path | None = None,
        ok: tuple[int, ...] = (200, 201, 202, 204),
        retry: bool = True,
    ):
        """Send one request; return the decoded JSON body (or None).

        Retried with backoff on 5xx and transport errors when `retry` (a
        non-idempotent create is sent once). 4xx raises at once with
        Zenodo's own message, which is usually the actionable part.
        """
        attempts = ATTEMPTS if retry else 1
        last: BaseException | None = None
        for attempt in range(1, attempts + 1):
            handle = None
            try:
                headers = {"Accept": "application/json"}
                data = None
                if body is not None:
                    data = json.dumps(body).encode()
                    headers["Content-Type"] = "application/json"
                elif upload is not None:
                    # Stream from disk: urllib/http.client reads a file
                    # object in blocks when Content-Length is given.
                    handle = open(upload, "rb")
                    data = handle
                    headers["Content-Type"] = "application/octet-stream"
                    headers["Content-Length"] = str(upload.stat().st_size)
                req = urllib.request.Request(
                    url, data=data, method=method, headers=headers
                )
                if self._token is not None:
                    req.add_unredirected_header(
                        "Authorization", f"Bearer {self._token}"
                    )
                with self.opener.open(req, timeout=TIMEOUT) as resp:
                    status = resp.status
                    raw = resp.read()
                if status not in ok:
                    raise ZenodoError(
                        f"{method} {url} returned HTTP {status}, expected "
                        f"one of {ok}."
                    )
                return json.loads(raw) if raw.strip() else None
            except urllib.error.HTTPError as e:
                detail = redact(_error_detail(e))
                if e.code < 500:
                    raise ZenodoError(
                        f"{method} {url} failed: HTTP {e.code} {e.reason}. "
                        f"Zenodo said: {detail}{_hint(e.code)}"
                    ) from None
                last = ZenodoError(f"HTTP {e.code} {e.reason}: {detail}")
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last = ZenodoError(redact(e))
            finally:
                if handle is not None:
                    handle.close()
            if attempt < attempts:
                delay = BACKOFF * 2 ** (attempt - 1)
                print(
                    f"  {method} {url} failed (attempt {attempt}/{attempts}): "
                    f"{redact(last)}; retrying in {delay:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise ZenodoError(
            f"{method} {url} failed after {attempts} attempt(s): "
            f"{redact(last)}"
        ) from None


def _error_detail(e: urllib.error.HTTPError) -> str:
    try:
        raw = e.read().decode(errors="replace")
    except OSError:
        return "(no body)"
    try:
        payload = json.loads(raw)
    except ValueError:
        return raw[:500]
    msg = payload.get("message", "") if isinstance(payload, dict) else ""
    errs = payload.get("errors") if isinstance(payload, dict) else None
    return f"{msg} {json.dumps(errs)}" if errs else (msg or raw[:500])


def _hint(code: int) -> str:
    if code == 401:
        return " (the token is invalid or expired)."
    if code == 403:
        return (
            " (the token lacks the deposit:write/deposit:actions scope, or "
            "you do not own this record)."
        )
    if code == 404:
        return " (no such record/draft -- check the id and --sandbox)."
    return ""


# --- deposition helpers -------------------------------------------------------


def _strip_md5(checksum: str) -> str:
    return checksum[4:] if checksum.startswith("md5:") else checksum


def draft_files(dep: dict) -> dict[str, dict]:
    """{filename: {"id", "size", "md5"}} from a deposit-API deposition."""
    out = {}
    for f in dep.get("files") or []:
        out[f["filename"]] = {
            "id": f["id"],
            "size": int(f["filesize"]),
            "md5": _strip_md5(f["checksum"]),
        }
    return out


def get_draft(client: Client, dep_id) -> dict:
    dep = client.request(
        "GET", client.url(f"/api/deposit/depositions/{dep_id}")
    )
    if dep.get("submitted") and dep.get("state") == "done":
        raise ZenodoError(
            f"Deposition {dep_id} is already published; it cannot take files. "
            f"Use `new-version --record {dep_id}`."
        )
    return dep


def web_url(dep: dict) -> str:
    return dep["links"]["html"]


def update_metadata(client: Client, dep: dict, changes: dict) -> dict:
    meta = dict(dep.get("metadata") or {})
    # The server owns these; echoing a carried-over value back breaks a new
    # version ("DOI already exists").
    meta.pop("doi", None)
    meta.pop("prereserve_doi", None)
    meta.update(changes)
    return client.request(
        "PUT",
        client.url(f"/api/deposit/depositions/{dep['id']}"),
        body={"metadata": meta},
    )


def sync_files(client: Client, dep: dict, files: dict, prune: bool) -> None:
    """Make the draft hold the bundle; skip files already there intact."""
    remote = draft_files(dep)
    bucket = dep["links"]["bucket"]
    for name, r in sorted(remote.items()):
        local = files.get(name)
        if local is not None and (local["size"], local["md5"]) == (
            r["size"],
            r["md5"],
        ):
            continue
        if local is None and not prune:
            print(f"  keep    {name} (carried over; not in the bundle)")
            continue
        why = "replaced by the bundle" if local is not None else "--prune"
        print(f"  delete  {name} ({why})")
        client.request(
            "DELETE",
            client.url(
                f"/api/deposit/depositions/{dep['id']}/files/{r['id']}"
            ),
        )
    for name, local in sorted(files.items()):
        r = remote.get(name)
        if r is not None and (r["size"], r["md5"]) == (
            local["size"],
            local["md5"],
        ):
            print(f"  skip    {name} (already on the draft, md5 matches)")
            continue
        print(f"  upload  {name} ({local['size']} bytes)")
        resp = client.request(
            "PUT",
            f"{bucket}/{urllib.parse.quote(name)}",
            upload=local["path"],
        )
        got = _strip_md5(resp["checksum"])
        if got != local["md5"] or int(resp["size"]) != local["size"]:
            raise ZenodoError(
                f"Upload of {name} came back as {resp['size']} bytes md5 "
                f"{got}; local is {local['size']} bytes md5 {local['md5']}. "
                f"Rerun `upload --draft {dep['id']}` to retry."
            )


def verify(client: Client, dep_id, files: dict) -> dict:
    """Re-read the draft; every bundle file must be there, size AND md5."""
    dep = get_draft(client, dep_id)
    remote = draft_files(dep)
    problems = []
    for name, local in sorted(files.items()):
        r = remote.get(name)
        if r is None:
            problems.append(f"{name}: missing from the draft")
        elif r["size"] != local["size"]:
            problems.append(
                f"{name}: draft has {r['size']} bytes, local {local['size']}"
            )
        elif r["md5"] != local["md5"]:
            problems.append(
                f"{name}: draft md5 {r['md5']}, local {local['md5']}"
            )
    if problems:
        raise ZenodoError(
            f"Verification of draft {dep_id} FAILED -- do not publish it:\n  "
            + "\n  ".join(problems)
            + f"\nRerun `upload --draft {dep_id} --bundle ...` to repair it."
        )
    extra = sorted(set(remote) - set(files))
    print(f"Verified {len(files)} file(s) on draft {dep_id} (size + md5).")
    if extra:
        print(f"  also on the draft (carried over): {', '.join(extra)}")
    return dep


def registry_entry(record_id, files: dict[str, dict], web: str) -> str:
    """The asset-table entry, in utilities/zenodo.fetch_assets's shape."""
    lines = [
        "# size and md5 come from the Zenodo record's own API",
        f"# ({web}/api/records/{record_id}).",
        "{",
    ]
    for name, meta in sorted(files.items()):
        if name == MANIFEST_NAME:
            continue
        lines += [
            f'    "{name}": {{',
            f'        "url": "{web}/records/{record_id}/files/{name}",',
            f'        "size": {meta["size"]},',
            f'        "md5": "{meta["md5"]}",',
            "    },",
        ]
    lines.append("}")
    return "\n".join(lines)


def maybe_publish(client: Client, dep: dict, args) -> None:
    if not args.publish:
        return
    title = (dep.get("metadata") or {}).get("title", "")
    if not sys.stdin.isatty():
        raise ZenodoError(
            "--publish needs an interactive terminal to confirm; refusing. "
            "Publish from the draft's web page instead."
        )
    print(
        "\nPublishing mints a PERMANENT DOI and cannot be undone.\n"
        f"Type the record title exactly to publish:\n  {title}"
    )
    typed = input("> ").strip()
    if not title or typed != title:
        raise ZenodoError("Title did not match; NOT published.")
    client.request(
        "POST",
        client.url(f"/api/deposit/depositions/{dep['id']}/actions/publish"),
        retry=False,
    )
    print(f"Published record {dep['id']}. Now run `pin --record {dep['id']}`.")


def finish(client: Client, dep: dict, files: dict, args) -> None:
    dep = verify(client, dep["id"], files)
    print(
        f"\nDRAFT READY (not published): {web_url(dep)}\n"
        "Review it there and click Publish. Then run\n"
        f"  python scripts/zenodo_publish.py pin --record {dep['id']}\n"
        "to print the registry entry from the published record. Provisional "
        "entry (valid once published):\n"
    )
    print(
        registry_entry(
            dep["id"], files, PRODUCTION if not args.sandbox else SANDBOX
        )
    )
    maybe_publish(client, dep, args)


# --- subcommands --------------------------------------------------------------


def _plan(args, files, ignored, action: str) -> None:
    print(f"Plan ({'sandbox' if args.sandbox else 'zenodo.org'}): {action}")
    for name, meta in sorted(files.items()):
        print(f"  {name}  {meta['size']} bytes  md5 {meta['md5']}")
    if ignored:
        print(f"  ignored dotfiles: {', '.join(ignored)}")


def _prepare_bundle(args):
    bundle = Path(args.bundle).expanduser()
    files, ignored = scan_bundle(bundle)
    files = ensure_manifest(bundle, files, args.dry_run)
    return files, ignored


def _client(args, need_token: bool = True) -> Client:
    base = base_url(args.sandbox)
    token = read_token(token_path(args)) if need_token else None
    return Client(base, token)


def cmd_new_version(args) -> None:
    files, ignored = _prepare_bundle(args)
    changes = (
        normalize_metadata(
            load_metadata(Path(args.metadata)),
            require_all=False,
            source=args.metadata,
        )
        if args.metadata
        else {}
    )
    _plan(
        args,
        files,
        ignored,
        f"new-version draft of record {args.record}; delete carried-over "
        f"files {'not matching the bundle' if args.prune else 'the bundle replaces'}"
        + (f"; update metadata keys {sorted(changes)}" if changes else ""),
    )
    if args.dry_run:
        print("[dry-run] no HTTP requests made.")
        return
    client = _client(args)
    resp = client.request(
        "POST",
        client.url(
            f"/api/deposit/depositions/{args.record}/actions/newversion"
        ),
    )
    draft_url = resp["links"]["latest_draft"]
    dep = client.request("GET", draft_url)
    print(f"New-version draft {dep['id']} created from record {args.record}.")
    if changes:
        dep = update_metadata(client, dep, changes)
    sync_files(client, dep, files, args.prune)
    finish(client, dep, files, args)


def cmd_new_record(args) -> None:
    files, ignored = _prepare_bundle(args)
    meta = normalize_metadata(
        load_metadata(Path(args.metadata)),
        require_all=True,
        source=args.metadata,
    )
    _plan(args, files, ignored, f"new draft record {meta['title']!r}")
    if args.dry_run:
        print("[dry-run] no HTTP requests made.")
        return
    client = _client(args)
    # Sent once: retrying a create on a 5xx can leave two drafts.
    dep = client.request(
        "POST",
        client.url("/api/deposit/depositions"),
        body={"metadata": meta},
        retry=False,
    )
    print(f"Draft {dep['id']} created.")
    sync_files(client, dep, files, prune=False)
    finish(client, dep, files, args)


def cmd_upload(args) -> None:
    files, ignored = _prepare_bundle(args)
    _plan(args, files, ignored, f"resume upload into draft {args.draft}")
    if args.dry_run:
        print("[dry-run] no HTTP requests made.")
        return
    client = _client(args)
    dep = get_draft(client, args.draft)
    sync_files(client, dep, files, args.prune)
    finish(client, dep, files, args)


def published_files(record: dict) -> dict[str, dict]:
    out = {}
    for f in record.get("files") or []:
        out[f["key"]] = {
            "size": int(f["size"]),
            "md5": _strip_md5(f["checksum"]),
        }
    if not out:
        raise ZenodoError(f"Record {record.get('id')} lists no files.")
    return out


def cmd_pin(args) -> None:
    web = base_url(args.sandbox)
    url = f"{web}/api/records/{args.record}"
    if args.dry_run:
        print(f"[dry-run] would GET {url} (no token) and print the entry.")
        return
    record = Client(web, None).request("GET", url)
    files = published_files(record)
    if args.bundle:
        local, _ = scan_bundle(Path(args.bundle).expanduser())
        bad = [
            n
            for n, m in local.items()
            if n not in files
            or (files[n]["size"], files[n]["md5"]) != (m["size"], m["md5"])
        ]
        if bad:
            raise ZenodoError(
                f"Published record {args.record} does not match the local "
                f"bundle for: {', '.join(sorted(bad))}."
            )
        print(f"Local bundle matches published record {args.record}.")
    title = (record.get("metadata") or {}).get("title", "")
    print(f"Record {args.record}: {title}\n")
    print(
        registry_entry(
            args.record, files, PRODUCTION if not args.sandbox else SANDBOX
        )
    )


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zenodo_publish.py",
        description=(
            "Upload a data bundle to a Zenodo DRAFT, verify it, and stop. "
            "Nothing is published unless --publish is given AND the title is "
            "typed at a terminal. See docs/zenodo.md."
        ),
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--sandbox",
        action="store_true",
        help="use sandbox.zenodo.org (token ~/.config/zenodo/sandbox_token)",
    )
    common.add_argument(
        "--token-file",
        help=(
            f"file holding the token (default ${TOKEN_FILE_ENV}, else "
            f"{DEFAULT_TOKEN_FILE}); the token itself is never an argument"
        ),
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan; make no HTTP request and write nothing",
    )
    upload = argparse.ArgumentParser(add_help=False)
    upload.add_argument("--bundle", required=True, help="directory to upload")
    upload.add_argument(
        "--prune",
        action="store_true",
        help="also delete draft files that are not in the bundle",
    )
    upload.add_argument(
        "--publish",
        action="store_true",
        help="after verifying, publish -- only after typing the title at a TTY",
    )

    sub = p.add_subparsers(dest="command", required=True)
    nv = sub.add_parser(
        "new-version",
        parents=[common, upload],
        help="new-version draft of a published record",
    )
    nv.add_argument("--record", required=True, help="published record id")
    nv.add_argument(
        "--metadata",
        help="YAML/JSON of metadata keys to change (e.g. version)",
    )
    nv.set_defaults(func=cmd_new_version)

    nr = sub.add_parser(
        "new-record",
        parents=[common, upload],
        help="fresh draft record",
    )
    nr.add_argument(
        "--metadata",
        required=True,
        help="YAML/JSON: title, creators, description, license, upload_type",
    )
    nr.set_defaults(func=cmd_new_record)

    up = sub.add_parser(
        "upload",
        parents=[common, upload],
        help="resume uploading into an existing draft",
    )
    up.add_argument("--draft", required=True, help="unpublished draft id")
    up.set_defaults(func=cmd_upload)

    pin = sub.add_parser(
        "pin",
        parents=[common],
        help="print the registry entry of a PUBLISHED record (no token)",
    )
    pin.add_argument("--record", required=True, help="published record id")
    pin.add_argument(
        "--bundle",
        help="optionally check the record against this directory",
    )
    pin.set_defaults(func=cmd_pin)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ZenodoError as e:
        print(f"ERROR: {redact(e)}", file=sys.stderr)
        return 1
    except Exception:
        # Anything unexpected (a changed API shape, a disk error): show the
        # traceback, but through the redactor, then fail.
        print(redact(traceback.format_exc()), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted. Any draft created is still there, unpublished; "
            "resume with `upload --draft <id>`.",
            file=sys.stderr,
        )
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
