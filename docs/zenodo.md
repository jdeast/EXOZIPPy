# Publishing data to Zenodo

Large data products (the NextGen spectra grid, the MIST EEP grids, ...) live
on Zenodo, in the [exozippy community](https://zenodo.org/communities/exozippy),
and are fetched on first use by `src/exozippy/utilities/zenodo.py`, which
pins every file by size and md5. `scripts/zenodo_publish.py` is the upload
side. It creates a **draft**, uploads, verifies, and **stops**: publishing
mints a permanent DOI, so a human always reviews the draft and clicks Publish
on zenodo.org.

## One-time setup

Create a personal access token at
<https://zenodo.org/account/settings/applications/> with scopes
`deposit:write` and `deposit:actions`, and save it alone on one line in
`~/.config/zenodo/token` with `chmod 600` (the script refuses a file that is
group/world readable). Override the path with `--token-file` or
`$ZENODO_TOKEN_FILE`. The token is never accepted on the command line, and
every message the script prints is passed through a redactor.
`--sandbox` targets sandbox.zenodo.org with `~/.config/zenodo/sandbox_token`.

## The workflow

1. **Build the bundle**: a flat directory holding exactly the files of the
   record (Zenodo records have no subdirectories; dotfiles are ignored). The
   script writes `MANIFEST.txt` (`name size md5` per line) into it if absent,
   and refuses a stale one.
2. **Create the draft and upload**:

   ```bash
   # a new version of an existing record (replaces same-named files, keeps
   # the others; --prune makes the draft hold exactly the bundle)
   python scripts/zenodo_publish.py new-version --record 21893308 --bundle grid/
   # a brand-new record; metadata needs title, creators, description,
   # license, upload_type (communities defaults to [exozippy])
   python scripts/zenodo_publish.py new-record --bundle grid/ --metadata meta.yaml
   ```

   `new-version --metadata changes.yaml` merges keys (e.g. `version`,
   `publication_date`) into the carried-over metadata. Add `--dry-run` first
   to see the plan with no HTTP at all. Uploads stream from disk and retry on
   5xx/connection errors; if a run is interrupted, resume with
   `upload --draft <id> --bundle grid/`, which skips files already on the
   draft with a matching md5. After uploading, the draft is re-read and every
   file's size and md5 checked against the local copy; a mismatch fails
   loudly and the banner says not to publish.
3. **Review and publish** the draft at the URL the script prints. (`--publish`
   exists but requires typing the exact title at a terminal; prefer the web
   page, where the metadata can be read before the DOI is minted.)
4. **Pin**: `python scripts/zenodo_publish.py pin --record <new id>
   [--bundle grid/]` reads the PUBLISHED record's public API (no token) and
   prints the asset entry (`{filename: {url, size, md5}}`, the shape
   `fetch_assets` takes). With `--bundle` it also checks the record against
   the local files.
5. **PR**: paste the entry into the asset table that downloads it
   (`_EEP_GRID_ASSETS` in `models/MIST/eep_grid.py`, `_MODEL_DATA` in
   `components/sed/make_bc.py`; for `_BC_TABLE_FILES` in
   `models/NextGen/bc_tables.py`, which builds its urls from
   `ZENODO_RECORD`, update that id and copy only the sizes and md5s) and
   open a PR. A new record id means
   every pinned url in that table changes; a new version of the same record
   gets a NEW id, so the old pins keep working until the PR lands.

Tests: `tests/test_zenodo_publish.py` drives every subcommand against a fake
in-process Zenodo; nothing in the suite talks to zenodo.org.
