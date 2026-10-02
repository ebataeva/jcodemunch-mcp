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
_PRELOAD_FLAGS = frozenset({"-r", "--require", "--import", "--loader", "--experimental-loader", "--preload"})
# Flags that take a value in the NEXT token. ⚠⚠ The value is not the entry:
# `nodemon --watch src server.js` runs `server.js`, and reading `src` as the
# entry rooted `src/index.ts`. A flag missing from this set has its value read
# as the entry, which roots a file only if that value resolves to one; add the
# flag here when that is seen.
_VALUE_FLAGS = frozenset({
    # node
    "-C", "--conditions", "--env-file", "--watch-path", "--inspect-port", "--title",
    "--input-type", "--max-old-space-size", "--stack-size", "--diagnostic-dir",
    # nodemon, ts-node-dev
    "-w", "--watch", "-e", "--ext", "-i", "--ignore", "--config", "-d", "--delay",
    "-s", "--signal", "--cwd", "--ignore-watch", "--debounce", "--interval",
    # ts-node, tsx
    "-P", "--project", "--compiler", "-O", "--compiler-options", "-I", "--dir",
    "--scope-dir", "--tsconfig", "--exclude", "--include",
    # deno, bun
    "-c", "--import-map", "--lock", "--cert", "--location", "--seed", "--v8-flags",
    "-d", "--define", "-l", "--tsconfig-override", "--port", "--env",
})
# With one of these the runner executes no file argument: inline code, a syntax
# check, the test runner, or (`nodemon --exec`) another command entirely.
_NO_ENTRY_FLAGS = frozenset({"--eval", "--print", "-p", "--check", "--test", "-x", "--exec", "--version", "--help", "-h", "-v"})
# node reads -e as --eval and -c as --check; nodemon and deno read them as value flags.
_NODE_FAMILY = frozenset({"node", "nodejs", "ts-node", "ts-node-esm", "babel-node", "tsx"})
_NODE_NO_ENTRY_FLAGS = frozenset({"-e", "-c"})
# Words that come before the file and mean "execute it".
_EXEC_SUBCOMMANDS = {"bun": frozenset({"run"}), "deno": frozenset({"run"}), "tsx": frozenset({"watch"})}
# Runners that take a subcommand: any OTHER bare word in that position is a
# subcommand that executes no file argument (`deno lint`, `bun build x.ts`,
# `bun test`). Only a path-shaped first argument is an entry there.
_SUBCOMMAND_RUNNERS = frozenset({"bun", "deno"})
# `bun run build` runs the SCRIPT named build before any file of that name.
_SCRIPT_NAME_RUNNERS = frozenset({"bun"})
_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_OPERATORS = frozenset({"&&", "||", ";", "|", "&", "|&", ";;"})


def _resolve_script_path(pkg_dir: str, token: str, source_files: frozenset) -> Optional[str]:
    """The indexed file a script argument names, relative to its package, or None.

    A directory argument (`node src`, `node .`) loads that directory's
    `package.json` `main` when it has one, which the field reader already
    handles, so a directory holding a `package.json` resolves to nothing here.
    Otherwise it loads `index.*`.
    """
    token = token.replace("\\", "/")
    joined = posixpath.normpath(posixpath.join(pkg_dir, token)) if pkg_dir else posixpath.normpath(token)
    if joined.startswith("..") or joined.startswith("/"):
        return None
    if joined == ".":
        joined = ""
    if joined:
        for suffix in _JS_ENTRY_SUFFIXES:
            if not suffix.startswith("/") and joined + suffix in source_files:
                return joined + suffix
    if (f"{joined}/package.json" if joined else "package.json") in source_files:
        return None
    for suffix in _JS_ENTRY_SUFFIXES:
        if suffix.startswith("/"):
            trial = (joined + suffix).lstrip("/")
            if trial in source_files:
                return trial
    return None


def _script_segments(command: str) -> list[list[str]]:
    """The command split into simple commands, with quoting read BEFORE the operators.

    An unbalanced quote returns nothing: a command that cannot be read declares nothing.
    """
    lexer = shlex.shlex(command.replace("\\", "/"), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _OPERATORS or (token and set(token) <= set("();<>|&")):
            segments.append([])
        else:
            segments[-1].append(token)
    return [s for s in segments if s]


def _script_entries(
    command: str, script_names: frozenset, pkg_dir: str, source_files: frozenset, _depth: int = 0
) -> set[str]:
    """Files one `scripts` command runs: per runner invocation, its preloads and its entry file.

    The entry is the first plain argument, when it resolves to an indexed
    file. If it does not resolve (`node dist/build.js input.js`, with `dist/`
    not indexed) the command declares nothing: the arguments after the entry
    belong to the program. A flag in `_VALUE_FLAGS` consumes the next token.

    ⚠⚠ Every doubt resolves to "declares nothing". A missed root leaves a live
    file reported, which the reader can see; a wrong root removes a dead file
    from the report, which nobody can (#569).
    """
    found: set[str] = set()
    for tokens in _script_segments(command):
        i = 0
        while i < len(tokens) and (
            _ENV_ASSIGNMENT.match(tokens[i]) or tokens[i] in _SCRIPT_WRAPPERS or tokens[i].startswith("-")
        ):
            i += 1
        if i >= len(tokens):
            continue
        runner = tokens[i].rsplit("/", 1)[-1]
        if runner not in _SCRIPT_RUNNERS:
            continue
        args = tokens[i + 1:]
        no_entry = _NO_ENTRY_FLAGS | (_NODE_NO_ENTRY_FLAGS if runner in _NODE_FAMILY else frozenset())
        value_flags = _VALUE_FLAGS - no_entry
        preloads: set[str] = set()
        entry: Optional[str] = None
        declares = True
        subcommand_open = runner in _SUBCOMMAND_RUNNERS or runner in _EXEC_SUBCOMMANDS
        j = 0
        while j < len(args):
            arg = args[j]
            j += 1
            if arg.startswith("-") and arg != "-":
                flag, eq, value = arg.partition("=")
                if flag in ("-x", "--exec") and runner in ("nodemon", "ts-node-dev"):
                    # The command nodemon runs instead of a file argument.
                    if not eq and j < len(args):
                        value = args[j]
                    if value and _depth < 2:
                        found |= _script_entries(value, script_names, pkg_dir, source_files, _depth + 1)
                    declares = False
                    break
                if flag in no_entry:
                    declares = False
                    break
                if flag in _PRELOAD_FLAGS:
                    if not eq and j < len(args):
                        value = args[j]
                        j += 1
                    hit = _resolve_script_path(pkg_dir, value, source_files) if value else None
                    if hit:
                        preloads.add(hit)
                elif flag in value_flags and not eq:
                    j += 1
                continue
            bare = "/" not in arg and "." not in arg
            if subcommand_open:
                subcommand_open = False
                if arg in _EXEC_SUBCOMMANDS.get(runner, ()):
                    continue
                if bare and runner in _SUBCOMMAND_RUNNERS:
                    declares = False  # `deno lint`, `bun build x.ts`, `bun test`
                    break
            if bare and runner in _SCRIPT_NAME_RUNNERS and arg in script_names:
                declares = False  # `bun run build`: the script, not a file
                break
            entry = _resolve_script_path(pkg_dir, arg, source_files)
            break
        if declares:
            found |= preloads
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
