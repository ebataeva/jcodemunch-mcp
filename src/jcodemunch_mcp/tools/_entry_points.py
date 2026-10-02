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
# `get_dead_code_v2`, with the same logic, and neither read `scripts`: a server
# started by `"start": "node server.js"` has no importer by construction and
# was published dead at confidence 1.0.

_JS_ENTRY_SUFFIXES = (
    "", ".js", ".ts", ".mjs", ".cjs", ".mts", ".cts", ".jsx", ".tsx",
    "/index.js", "/index.ts", "/index.mjs", "/index.cjs",
)

# Programs that EXECUTE the file they are given, each with ITS OWN flags. ⚠⚠ A
# linter, a formatter, a test runner and a bundler all take file arguments too,
# and a file named to one of those is declared nothing: adding `eslint` here
# turns every linted file into a live root and suppresses real findings (#569's
# direction).
#
# ⚠⚠ The flag tables are PER RUNNER because one spelling means different
# things: `--watch` takes a value for nodemon and none for node, bun and deno;
# `-e` is `--ext` for nodemon and `--eval` for node; `-r` is `--require` for
# node and `--reload` for deno. One shared table made `node --watch server.js
# worker.js` consume the entry and root `worker.js`.
#   value     flags that take their value in the NEXT token
#   preload   value flags whose value is a module loaded before the entry
#   no_entry  with one of these, no file argument is executed
#   exec_sub  words that come before the file and mean "execute it"
#   exec      flags whose value is a COMMAND the runner runs (nodemon --exec)
_NODE_VALUE = frozenset({
    "-C", "--conditions", "--env-file", "--watch-path", "--inspect-port", "--title",
    "--input-type", "--diagnostic-dir",
    "--experimental-default-type",
})
_NODE_PRELOAD = frozenset({"-r", "--require", "--import", "--loader", "--experimental-loader"})
_NODE_NO_ENTRY = frozenset({
    "-e", "--eval", "-p", "--print", "-c", "--check", "--test", "--run",
    "-i", "--interactive", "-v", "--version", "-h", "--help",
})
_TS_NODE_VALUE = frozenset({
    "-P", "--project", "-C", "--compiler", "-O", "--compiler-options", "--compilerOptions",
    "-I", "--ignore", "--dir", "--scope-dir", "--scopeDir", "-D", "--ignore-diagnostics", "--cwd",
    "--transpiler",
})
_TS_NODE_NO_ENTRY = frozenset({"-e", "--eval", "-p", "--print", "-i", "--interactive", "-v", "--version", "-h", "--help"})
_HELP = frozenset({"-v", "--version", "-h", "--help"})
_NONE: frozenset = frozenset()


def _spec(value=_NONE, preload=_NONE, no_entry=_HELP, exec_sub=_NONE, exec=_NONE, subcommands=False, script_names=False):
    return {
        "value": value, "preload": preload, "no_entry": no_entry, "exec_sub": exec_sub,
        "exec": exec, "subcommands": subcommands, "script_names": script_names,
    }


_NODE_SPEC = _spec(_NODE_VALUE, _NODE_PRELOAD, _NODE_NO_ENTRY, exec_sub=frozenset({"inspect"}))
_TS_NODE_SPEC = _spec(_TS_NODE_VALUE, frozenset({"-r", "--require"}), _TS_NODE_NO_ENTRY)
_SCRIPT_RUNNERS = {
    "node": _NODE_SPEC,
    "nodejs": _NODE_SPEC,
    "electron": _NODE_SPEC,
    "tsx": _spec(
        _NODE_VALUE | {"--tsconfig", "--ignore", "--include", "--exclude"},
        _NODE_PRELOAD, _NODE_NO_ENTRY, exec_sub=frozenset({"watch"}),
    ),
    "ts-node": _TS_NODE_SPEC,
    "ts-node-esm": _TS_NODE_SPEC,
    "ts-node-dev": _spec(
        _TS_NODE_VALUE | {"--watch", "--ignore-watch", "--debounce", "--interval"},
        frozenset({"-r", "--require"}), _TS_NODE_NO_ENTRY,
    ),
    "nodemon": _spec(
        frozenset({"-w", "--watch", "-e", "--ext", "-i", "--ignore", "--config", "-d", "--delay", "-s", "--signal", "--cwd"}),
        frozenset({"-r", "--require"}), exec=frozenset({"-x", "--exec"}),
    ),
    "babel-node": _spec(
        frozenset({"--presets", "--plugins", "--extensions", "-x", "--config-file", "--ignore", "--only", "--env-name", "--root-mode"}),
        frozenset({"-r", "--require"}), frozenset({"-e", "--eval", "-p", "--print"}) | _HELP,
    ),
    "bun": _spec(
        frozenset({"-c", "--config", "--cwd", "--env-file", "-d", "--define", "-l", "--loader", "--tsconfig-override", "--port", "--conditions"}),
        frozenset({"-r", "--preload", "--require", "--import"}),
        frozenset({"-e", "--eval", "-p", "--print", "--revision"}) | _HELP,
        exec_sub=frozenset({"run"}), subcommands=True, script_names=True,
    ),
    "deno": _spec(
        frozenset({"-c", "--config", "--import-map", "--lock", "--cert", "--location", "--seed", "--v8-flags", "-L", "--log-level"}),
        no_entry=_HELP | {"-V"}, exec_sub=frozenset({"run"}), subcommands=True,
    ),
    "pm2-runtime": _spec(exec_sub=frozenset({"start"})),
}
# Tokens that stand in front of the program without being it.
_SCRIPT_WRAPPERS = frozenset({
    "npx", "pnpx", "bunx", "cross-env", "cross-env-shell", "env", "dotenv",
    "yarn", "pnpm", "exec", "dlx", "--",
})
# A changed working directory: the file argument is no longer relative to the
# manifest. `cd client && node build.js` rooted the ROOT `build.js` and left the
# real entry reported. After one of these, the rest of the command declares
# nothing; a runner carrying one of the flags declares nothing.
_CHDIR_COMMANDS = frozenset({"cd", "pushd", "chdir"})
_CWD_FLAGS = frozenset({"--cwd", "--dir", "--prefix", "-C"})
_RUNNABLE_SUFFIXES = (".js", ".ts", ".mjs", ".cjs", ".mts", ".cts", ".jsx", ".tsx")
_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_OPERATORS = frozenset({"&&", "||", ";", "|", "&", "|&", ";;"})


def _resolve_script_path(
    pkg_dir: str, token: str, source_files: frozenset, own_main: bool = True
) -> Optional[str]:
    """The indexed file a script argument names, relative to its package, or None.

    A directory argument (`node src`, `node .`) loads that directory's
    `package.json` `main` when it has one, which the field reader already
    handles. So a directory holding a `package.json` resolves to nothing here,
    except the script's own package when its manifest names no `main`
    (``own_main`` False): node loads `index.*` there.
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
        if own_main or joined != pkg_dir:
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
    command: str,
    script_names: frozenset,
    pkg_dir: str,
    source_files: frozenset,
    own_main: bool = True,
    _depth: int = 0,
) -> set[str]:
    """Files one `scripts` command runs: per runner invocation, its preloads and its entry file.

    The entry is the first plain argument, when it resolves to an indexed
    file. If it does not resolve (`node dist/build.js input.js`, with `dist/`
    not indexed) the command declares nothing: the arguments after the entry
    belong to the program.

    ⚠⚠ Every doubt resolves to "declares nothing". A missed root leaves a live
    file reported, which the reader can see; a wrong root removes a dead file
    from the report, which nobody can (#569). Two rules exist only for that:
    a value flag never consumes another flag, and a value flag whose value is
    itself a runnable indexed file makes the segment undecidable (the table may
    be wrong about that flag, and then the "value" was the entry).
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
        if tokens[i] in _CHDIR_COMMANDS:
            break  # everything after runs somewhere else
        if any(t.partition("=")[0] in _CWD_FLAGS for t in tokens[:i]):
            continue  # `yarn --cwd sub node x.js`
        runner = tokens[i].rsplit("/", 1)[-1]
        spec = _SCRIPT_RUNNERS.get(runner)
        if spec is None:
            continue
        args = tokens[i + 1:]
        preloads: set[str] = set()
        entry_token: Optional[str] = None
        exec_command: Optional[str] = None
        declares = True
        subcommand_open = bool(spec["subcommands"] or spec["exec_sub"])
        j = 0
        while j < len(args) and declares:
            arg = args[j]
            j += 1
            if arg.startswith("-") and arg != "-":
                flag, eq, value = arg.partition("=")
                takes_value = flag in spec["value"] or flag in spec["preload"] or flag in spec["exec"]
                if flag in spec["no_entry"] or (flag in _CWD_FLAGS and flag not in spec["value"] - {"--cwd", "--dir"}):
                    declares = False
                elif takes_value:
                    if not eq:
                        if j >= len(args) or args[j].startswith("-"):
                            continue  # no value to read; never consume another flag
                        value = args[j]
                        j += 1
                    if flag in spec["exec"]:
                        exec_command = value
                    elif flag in spec["preload"]:
                        # `-r esm`, `--import tsx`: node resolves a bare name from
                        # node_modules, never `./esm`. Only a relative path is ours.
                        if value.startswith(("./", "../")):
                            hit = _resolve_script_path(pkg_dir, value, source_files, own_main)
                            if hit:
                                preloads.add(hit)
                    elif not eq and value.endswith(_RUNNABLE_SUFFIXES) and _resolve_script_path(
                        pkg_dir, value, source_files, own_main
                    ):
                        declares = False  # is that a value, or the entry behind a flag we misread?
                continue
            bare = "/" not in arg and "." not in arg
            if subcommand_open:
                subcommand_open = False
                if arg in spec["exec_sub"]:
                    continue
                if bare and spec["subcommands"]:
                    declares = False  # `deno lint`, `bun build x.ts`, `bun test`
                    continue
            if bare and spec["script_names"] and arg in script_names:
                declares = False  # `bun run build`: the script, not a file
                continue
            entry_token = arg
            break
        if not declares:
            continue
        if exec_command is not None:
            # nodemon appends its script argument to the command it was told to run.
            if _depth < 2:
                run = exec_command if entry_token is None else f"{exec_command} {shlex.quote(entry_token)}"
                found |= _script_entries(run, script_names, pkg_dir, source_files, own_main, _depth + 1)
            continue
        found |= preloads
        if entry_token is not None:
            hit = _resolve_script_path(pkg_dir, entry_token, source_files, own_main)
            if hit:
                found.add(hit)
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
            own_main = isinstance(pkg.get("main"), str) and bool(pkg.get("main"))  # node reads `main` only
            for command in scripts.values():
                if isinstance(command, str):
                    entries |= _script_entries(command, names, pkg_dir, source_files, own_main)
    return entries
