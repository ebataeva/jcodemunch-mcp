"""Framework-declared entry points, read from the index (#561, #562).

⚠⚠ **The authority already existed and had NO readers.** ``detect_framework``
runs at index time and ``profile_to_meta`` persists the profile's
``entry_point_patterns`` into ``context_metadata`` -- for Next.js that is
exactly ``src/app/**/route.ts``, ``page.tsx``, ``layout.tsx`` and
``middleware.ts``. A tree-wide search found the key written in one place and
read in none. Every consumer that needed to know "is this file a root?"
reproduced its own answer instead, and every one of those answers was Python:
``find_dead_code._ENTRY_POINT_FILENAMES`` is ``main.py`` / ``app.py`` /
``__main__.py`` and eleven siblings, with no JS entry in it at all.

⚠ So this module adds no knowledge. It is the read half of a write that was
already happening, which is the standing lesson in its usual costume: **ask the
authority instead of reproducing its logic.** A framework this does not cover
is fixed in ``framework_profiles.py``, once, and every consumer here inherits
it.

⚠⚠ **``matches()`` returning False is NOT "this is an ordinary module".** No
detected profile means no declaration was available, and a caller that reads
that as a negative finding is asserting something nobody measured. Callers
wanting the difference read ``profile_name`` -- ``None`` there means unknown.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import posixpath
import re
import shlex
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntryPointSpec:
    """The entry-point declaration an index carries, if any."""

    profile_name: Optional[str]
    patterns: tuple[str, ...]

    @property
    def declared(self) -> bool:
        """True when a framework profile actually named some roots.

        ⚠ A detected profile with an empty pattern list is still ``False``
        here: it declared nothing, so it can exclude nothing.
        """
        return bool(self.patterns)

    def matches(self, file_path: str) -> bool:
        """True when ``file_path`` is a root the framework declares."""
        if not self.patterns:
            return False
        norm = file_path.replace("\\", "/").lstrip("./")
        base = norm.rsplit("/", 1)[-1]
        for pat in self.patterns:
            if pat.endswith("/"):
                # Directory prefix (`cmd/`, `internal/`). fnmatch never
                # matches these -- `fnmatch("cmd/main.go", "cmd/")` is False --
                # so a prefix test is the only reading under which the Gin
                # profile declares anything at all.
                if norm == pat.rstrip("/") or norm.startswith(pat):
                    return True
                continue
            if fnmatch.fnmatch(norm, pat):
                return True
            if "/" not in pat and norm == base and fnmatch.fnmatch(base, pat):
                # A bare filename declares the ROOT-LEVEL file, not every file
                # of that name anywhere in the tree: `main.py` must not make
                # `src/vendor/main.py` a root. Profiles that mean the nested
                # form spell it out (`src/middleware.ts` sits beside
                # `middleware.ts` in the Next profile for exactly this reason).
                return True
        return False


_EMPTY = EntryPointSpec(profile_name=None, patterns=())

# ⚠⚠ A pattern that matches every source file DECLARES NOTHING, and consuming
# it is far worse than the defect this module fixes. The Flask and FastAPI
# profiles shipped `"*.py"` in their entry-point lists for their whole lives,
# harmless only because nothing read the field (the NestJS profile has a
# comment saying so). Under fnmatch `*` crosses `/`, so a naive reader would
# have declared every Python file in a Flask repo a live root -- turning the
# dead-code tool into one that reports nothing, on a whole ecosystem, silently.
#
# ⚠ The catch-alls are removed at the source too. This guard stays because a
# profile is a list of literals anyone can extend, and the failure is invisible
# from the edit: adding `*.ts` to a profile looks like widening coverage and is
# actually switching a subsystem off.
_CATCH_ALL_PATTERNS = frozenset({"*", "**", "*.*", "**/*"})


def _is_catch_all(pattern: str) -> bool:
    """True for a pattern that cannot distinguish a root from an ordinary file."""
    pat = pattern.strip()
    if pat in _CATCH_ALL_PATTERNS:
        return True
    # `*.py`, `*.ts`, `**/*.tsx`: a bare extension glob over the WHOLE tree.
    # ⚠ Directory scope is what saves a pattern here: `routes/*.php` names one
    # directory and is a perfectly good declaration, while `**/*.php` names
    # every PHP file there is. Only an unscoped (or `**`-scoped) extension
    # glob is a catch-all.
    head, _, stem = pat.rpartition("/")
    if head not in ("", "**"):
        return False
    return stem.startswith("*.") and "*" not in stem[2:] and stem[2:].isalnum()


def entry_point_spec(index) -> EntryPointSpec:
    """Read the framework profile an index was built with.

    Returns ``_EMPTY`` when the index predates profile persistence, was built
    for a framework we do not profile, or carries a malformed block -- all of
    which are "we do not know", never "there are no entry points".
    """
    meta = getattr(index, "context_metadata", None) or {}
    block = meta.get("framework_profile")
    if not isinstance(block, dict):
        return _EMPTY
    raw = block.get("entry_point_patterns")
    if not isinstance(raw, (list, tuple)):
        return _EMPTY
    kept: list[str] = []
    for p in raw:
        if not isinstance(p, str) or not p:
            continue
        if _is_catch_all(p):
            logger.debug(
                "entry_point_spec: ignoring catch-all pattern %r from profile %r",
                p, block.get("name"),
            )
            continue
        kept.append(p)
    patterns = tuple(kept)
    name = block.get("name")
    return EntryPointSpec(
        profile_name=name if isinstance(name, str) and name else None,
        patterns=patterns,
    )


# ---------------------------------------------------------------------------
# package.json: the files a manifest DECLARES as roots
# ---------------------------------------------------------------------------
#
# One reader (LEDGER L-102). It lived twice, in `find_dead_code` and in
# `get_dead_code_v2`, byte for byte, and neither read `scripts`: a server
# started by `"start": "node server.js"` has no importer by construction and
# was published dead at confidence 1.0.

_JS_ENTRY_SUFFIXES = (
    "", ".js", ".ts", ".mjs", ".cjs", ".mts", ".cts", ".jsx", ".tsx",
    "/index.js", "/index.ts", "/index.mjs", "/index.cjs",
)

# Programs that EXECUTE the file they are given. ⚠⚠ A linter, a formatter, a
# test runner and a bundler all take file arguments too, and a file named to
# one of those is declared nothing: adding `eslint` here turns every linted
# file into a live root and suppresses real findings (#569's direction).
_SCRIPT_RUNNERS = frozenset({
    "node", "nodejs", "nodemon", "ts-node", "ts-node-dev", "ts-node-esm", "tsx",
    "bun", "deno", "babel-node", "pm2-runtime", "electron",
})
# Tokens that stand in front of the program without being it.
_SCRIPT_WRAPPERS = frozenset({
    "npx", "pnpx", "bunx", "cross-env", "cross-env-shell", "env", "dotenv",
    "yarn", "pnpm", "exec", "dlx", "--",
})
# Flags whose VALUE is a module the runner loads before the entry file.
_PRELOAD_FLAGS = frozenset({"-r", "--require", "--import", "--loader", "--experimental-loader"})
# Words a runner takes before the file: `bun run x.ts`, `tsx watch x.ts`, `deno run x.ts`.
_RUNNER_SUBCOMMANDS = frozenset({"run", "watch", "start", "exec", "x"})
_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|&\n]")


def _resolve_script_path(pkg_dir: str, token: str, source_files: frozenset) -> Optional[str]:
    """The indexed file a script argument names, relative to its package, or None."""
    joined = posixpath.normpath(posixpath.join(pkg_dir, token.replace("\\", "/"))) if pkg_dir else (
        posixpath.normpath(token.replace("\\", "/"))
    )
    if joined.startswith("..") or joined.startswith("/"):
        return None
    if joined == ".":
        joined = ""
    for suffix in _JS_ENTRY_SUFFIXES:
        trial = (joined + suffix).lstrip("/")
        if trial in source_files:
            return trial
    return None


def _script_entries(command: str, script_names: frozenset, pkg_dir: str, source_files: frozenset) -> set[str]:
    """Files one `scripts` command runs: per runner invocation, its preloads and its entry file.

    The entry is the first plain argument, when it resolves to an indexed
    file. If it does not resolve (`node dist/build.js input.js`, with `dist/`
    not indexed) the command declares nothing here: the arguments after the
    entry belong to the program. An argument that directly follows a flag
    may be that flag's value (`nodemon --watch src server.js`), so it is
    taken only when there is no plain argument at all (`ts-node-dev --respawn
    src/index.ts`).
    """
    found: set[str] = set()
    for segment in _SEGMENT_SPLIT.split(command):
        try:
            tokens = shlex.split(segment.replace("\\", "/"))
        except ValueError:
            tokens = segment.split()
        i = 0
        while i < len(tokens) and (_ENV_ASSIGNMENT.match(tokens[i]) or tokens[i] in _SCRIPT_WRAPPERS):
            i += 1
        if i >= len(tokens) or tokens[i].rsplit("/", 1)[-1] not in _SCRIPT_RUNNERS:
            continue
        sure: Optional[str] = None
        maybe: Optional[str] = None
        after_flag = False
        plain_seen = False
        args = tokens[i + 1:]
        j = 0
        while j < len(args):
            arg = args[j]
            j += 1
            if arg.startswith("-"):
                flag, _, value = arg.partition("=")
                if flag in _PRELOAD_FLAGS:
                    if not value and j < len(args):
                        value = args[j]
                        j += 1
                    hit = _resolve_script_path(pkg_dir, value, source_files) if value else None
                    if hit:
                        found.add(hit)
                    after_flag = False
                else:
                    after_flag = not value
                continue
            was_after_flag, after_flag = after_flag, False
            if arg in _RUNNER_SUBCOMMANDS:
                continue
            hit = _resolve_script_path(pkg_dir, arg, source_files)
            if was_after_flag:
                maybe = maybe or hit
                continue
            if arg in script_names and "/" not in arg and "." not in arg:
                hit = None  # `bun run build`: a script's name, not a path
            sure = hit
            plain_seen = True
            break
        entry = sure if plain_seen else maybe
        if entry:
            found.add(entry)
    return found


def package_json_entries(index, store, owner: str, repo_name: str) -> set[str]:
    """Source files a ``package.json`` declares as roots.

    ``main`` / ``module`` / ``browser`` / ``exports`` / ``bin`` name the file a
    consumer loads; ``scripts`` names the file a runner executes
    (:func:`_script_entries`). JS equivalent of the Python ``app.py`` /
    ``main.py`` filename rule, read from the manifest instead of guessed.
    """
    entries: set[str] = set()
    source_files = frozenset(index.source_files)
    for f in index.source_files:
        fn = f.replace("\\", "/").rsplit("/", 1)[-1]
        if fn != "package.json":
            continue
        content = store.get_file_content(owner, repo_name, f)
        if not content:
            continue
        try:
            pkg = json.loads(content)
        except (ValueError, TypeError):
            continue
        if not isinstance(pkg, dict):
            continue
        candidates: list[str] = []
        for key in ("main", "module", "browser"):
            v = pkg.get(key)
            if isinstance(v, str):
                candidates.append(v)
        # `exports` can be a string, a dict of subpaths, or a conditional dict.
        exports = pkg.get("exports")
        if isinstance(exports, str):
            candidates.append(exports)
        elif isinstance(exports, dict):
            def _walk_exports(node):
                if isinstance(node, str):
                    candidates.append(node)
                elif isinstance(node, dict):
                    for v in node.values():
                        _walk_exports(v)
            _walk_exports(exports)
        # `bin` can be a string or a {name: path} dict.
        bins = pkg.get("bin")
        if isinstance(bins, str):
            candidates.append(bins)
        elif isinstance(bins, dict):
            candidates.extend(v for v in bins.values() if isinstance(v, str))

        pkg_dir = f.replace("\\", "/").rsplit("/", 1)[0] if "/" in f else ""
        for cand in candidates:
            cand = cand.lstrip("./").replace("\\", "/")
            joined = f"{pkg_dir}/{cand}" if pkg_dir else cand
            joined = joined.lstrip("/")
            if joined in source_files:
                entries.add(joined)
                continue
            for ext in _JS_ENTRY_SUFFIXES:
                trial = joined + ext
                if trial in source_files:
                    entries.add(trial)
                    break

        scripts = pkg.get("scripts")
        if isinstance(scripts, dict):
            names = frozenset(k for k in scripts if isinstance(k, str))
            for command in scripts.values():
                if isinstance(command, str):
                    entries |= _script_entries(command, names, pkg_dir, source_files)
    return entries
