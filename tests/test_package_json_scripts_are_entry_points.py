"""A file a `package.json` script runs is an entry point (LEDGER L-102).

`_package_json_entries` read `main`, `module`, `browser`, `exports` and `bin`,
and not `scripts`. A server started by `"start": "node server.js"` has no
importer by construction, so `find_dead_code` reported it dead at confidence
1.0, everything only it imports as `all_importers_dead`, and the deletion
investigator then read a name it imports as "imported only by server.js, which
is itself unreachable".

The reader also existed twice, once in `find_dead_code` and once in
`get_dead_code_v2`, with the same logic. It is one function in
`tools/_entry_points.py` now, so a field read by one is read by both.

The other direction is the half that matters (#569: a fix for a false positive
can install a false negative). A wrong root removes a dead file from the
report and from the delete preflight with no symptom. Four review rounds each
found a spelling a denylist had not named (a flag's value, a subcommand word,
`cd client &&`, `pnpm --filter=web exec`), so the rule is an ALLOWLIST: a
command roots a file only when every token in front of it is known, and the
file is path-shaped. Most cases below assert that something roots NOTHING.

The command rule is tested on `_script_entries` directly and the reader on
manifest text with a fake store (one index build per case cost the suite
seconds it does not have); the tools are tested end to end on a few cases.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from jcodemunch_mcp.investigator import REFUTED, investigate_deletion_safety
from jcodemunch_mcp.tools import _entry_points, find_dead_code as fdc, get_dead_code_v2 as v2
from jcodemunch_mcp.tools.index_folder import index_folder

SRC = Path(__file__).resolve().parent.parent / "src" / "jcodemunch_mcp" / "tools"

# ── The command rule ─────────────────────────────────────────────────────────

# Every name a wrong rule could reach for is a real file here: the directory a
# flag watches, the word a subcommand uses, the package a preload names, the
# same file one directory down.
FILES = frozenset({
    "package.json", "server.js", "server.ts", "legacy.js", "preload.js", "config.js",
    "lib/shapes.js", "lib/server.js", "lib/index.js",
    "src/index.ts", "src/server.ts", "src/app.ts", "extras/index.ts",
    "lint.ts", "build.js", "test.js", "task.ts", "start.js", "inspect.js", "watch.js", "run.js",
    "esm.js", "tsx.ts", "register.js", "dotenv/config.js", "prisma/index.ts", "web/server.js",
})


def _roots(command: str, pkg_dir: str = "", files: frozenset = FILES, own_main: bool = True) -> set[str]:
    return _entry_points._script_entries(command, pkg_dir, files, own_main)


@pytest.mark.parametrize(
    "command",
    [
        "node server.js",
        "node ./server.js",
        "node .\\server.js",
        'node "server.js"',
        "NODE_ENV=production node server.js",
        "cross-env NODE_ENV=production PORT=1 node server.js",
        "npx nodemon server.js",
        "npx -y nodemon server.js",
        "./node_modules/.bin/nodemon server.js",
        "nodemon --watch lib server.js",
        "nodemon --watch lib --ext js,json --delay 2 server.js",
        "nodemon --watch src server.js",
        "nodemon -e ts server.js",
        "nodemon -V server.js",
        "nodemon -w lib --verbose server.js",
        "nodemon --ignore lib server.js",
        'nodemon --exec "node server.js"',
        "nodemon --watch lib -x 'node server.js'",
        "nodemon --exec babel-node server.js",
        "nodemon --exec node server.js",
        "nodemon --watch lib --exec node server.js",
        "node -r dotenv/config server.js",
        "node -r esm server.js",
        "node --import tsx server.js",
        "node --require register server.js",
        "node --inspect server.js",
        "node --inspect=0.0.0.0:9229 server.js",
        "node --trace-warnings --enable-source-maps server.js",
        "node --max-old-space-size=4096 server.js",
        "node --stack-size=2000 server.js",
        "node --env-file=.env server.js",
        "node --watch server.js",
        "node --watch server.js legacy.js",
        "node -C development server.js",
        "node inspect server.js",
        "npm run build && node server.js",
        "tsc -p . ; node server.js",
        "node server.js --port 3000",
        "node server.js legacy.js",
        "node server.js && cd lib && node legacy.js",
        "pm2-runtime server.js",
        "pm2-runtime start server.js",
        "bun --watch ./server.js",
        "bun --preload esm server.js",
    ],
)
def test_the_file_a_runner_executes_is_the_only_root(command):
    assert _roots(command) == {"server.js"}, command


@pytest.mark.parametrize(
    "command,entry",
    [
        ("ts-node src/server.ts", "src/server.ts"),
        ("ts-node --project tsconfig.json src/server.ts", "src/server.ts"),
        ("ts-node --files -T src/server.ts", "src/server.ts"),
        ("tsx src/server.ts", "src/server.ts"),
        ("tsx watch src/server.ts", "src/server.ts"),
        ("npx tsx src/server.ts", "src/server.ts"),
        ("bun run src/server.ts", "src/server.ts"),
        ("bun src/server.ts", "src/server.ts"),
        ("bun run --watch src/server.ts", "src/server.ts"),
        ("bun run build.js", "build.js"),
        ("bun ./build.js", "build.js"),
        ("deno run --allow-net src/server.ts", "src/server.ts"),
        ("deno run -A --unstable-kv src/server.ts", "src/server.ts"),
        ("deno run --config deno.json src/server.ts", "src/server.ts"),
        ("deno run --watch src/server.ts", "src/server.ts"),
        ("deno run -r src/server.ts src/app.ts", "src/server.ts"),  # deno's -r is --reload
        ("ts-node-dev --respawn src/server.ts", "src/server.ts"),
        ("ts-node-dev --respawn --ignore-watch extras src/server.ts", "src/server.ts"),
        ("node --loader ts-node/esm src/server.ts", "src/server.ts"),
        ('nodemon --watch src --exec "ts-node src/server.ts"', "src/server.ts"),
        ('nodemon -x "ts-node" src/server.ts', "src/server.ts"),
        ("ts-node ./src", "src/index.ts"),
        ("node ./lib", "lib/index.js"),
    ],
)
def test_other_runners_and_a_directory_argument(command, entry):
    assert _roots(command) == {entry}, command


@pytest.mark.parametrize(
    "command",
    [
        # a file named to a program that does not execute it
        "eslint server.js legacy.js",
        "prettier --write server.js",
        "jest server.js",
        "echo node server.js",
        "server.js",
        "npm run server.js",
        "node",
        "nodemon --exec eslint server.js",
        # the entry is not indexed: what follows it is the program's own argument
        "node dist/build.js server.js",
        "node dist/build.js --out x server.js",
        # a bare word is a subcommand or a script's name, never a path
        "node server",
        "node build",
        "node lib",
        "node watch server.js",
        "node inspect",
        "tsx inspect server.ts",
        "pm2-runtime inspect",
        "pm2-runtime start ecosystem.config.js",
        "deno lint",
        "deno fmt src/app.ts",
        "deno check src/app.ts",
        "deno test",
        "deno task build",
        "deno run start",
        "bun test",
        "bun build src/app.ts --outdir dist",
        "bun --watch build ./src/app.ts",
        "bun install",
        "bun run build",
        "bun run lint server.js",
        "bun run server",
        "bun run prisma generate",
        "bunx build",
        # a flag that executes no file argument: none of them is in a runner's known flags
        "node --check legacy.js",
        "node -c server.js legacy.js",
        "node --test legacy.js",
        "node -e \"require('./legacy.js')\"",
        "node -e server.js legacy.js",
        "node -p legacy.js",
        "node --run build",
        "node --run build ./server.js",
        "node --version",
        "deno -V server.js",
        # a known value flag: its value is not the entry
        "nodemon --ignore legacy.js",
        "nodemon --config legacy.js",
        "nodemon -w lib -e js",
        "nodemon -w src -e ts",
        "deno run --config legacy.js",
        "node --watch-path server.js legacy.js",
        "nodemon --ignore legacy.js server.js",
        "nodemon -w --verbose server.js",
        # an UNKNOWN flag: it may take a value or change the directory
        "node --redirect-warnings legacy.js server.js",
        "node --snapshot-blob legacy.js server.js",
        "node --openssl-config config.js server.js",
        "node --cpu-prof-dir lib server.js",
        "node --some-new-flag server.js",
        "nodemon --cwd lib server.js",
        "ts-node --cwd src server.ts",
        "ts-node --dir src server.ts",
        "bun --cwd lib run server.js",
        "bun --filter=web run server.js",
        # flags ts-node knows and ts-node-dev 2.0.0 does not: the command fails at start
        "ts-node-dev --esm server.ts",
        "ts-node-dev --respawn --inspect server.ts",
        "ts-node-dev --transpileOnly server.ts",
        "ts-node-dev --swc server.ts",
        "ts-node-dev -q server.ts",
        # the word after bun's bare config flag is the file bun runs, and it is not indexed
        "bun -c bunfig.toml ./server.js",
        "bun --config bunfig.toml ./server.js",
        "bun run --config bunfig.toml ./server.js",
        "deno run --v8-flags somevalue ./server.js",
        # a quoted assignment is a program name to a shell
        '"A=1" node server.js',
        # the segment before never returns, or renames the runner
        "exit 0; node server.js",
        "exec node other.js; node server.js",
        "alias node=echo; node server.js",
        "trap x EXIT; node server.js",
        # an UNKNOWN wrapper or wrapper flag: the command may run somewhere else
        "yarn node server.js",
        "yarn --cwd lib node server.js",
        "yarn --cwd=lib node server.js",
        "yarn workspace web node server.js",
        "pnpm exec node server.js",
        "pnpm --filter=web exec node server.js",
        "pnpm -F=web exec node server.js",
        "pnpm -r exec node server.js",
        "pnpm -C lib exec node server.js",
        "npm --prefix lib exec node server.js",
        "npx --workspace=web node server.js",
        "npx --workspaces node server.js",
        "npx -w web node server.js",
        "env --chdir=lib node server.js",
        "env -C lib node server.js",
        "dotenv -e .env -- node server.js",
        "sudo node server.js",
        "time node server.js",
        # a changed working directory
        "cd lib && node server.js",
        "pushd lib; node server.js",
        "chdir lib && node server.js",
        'nodemon --exec "cd lib && node" server.js',
        # shell grammar this reader does not model: the whole command declares nothing
        "(cd lib && node server.js)",
        "if [ -d lib ]; then cd lib; fi; node server.js",
        "{ cd lib; node server.js; }",
        "builtin cd lib && node server.js",
        "command cd lib; node server.js",
        "echo hi > node server.js",
        "node server.js > out.log",
        "node server.js | tee out.log",
        "node server.js &",
        "node server.js 2>&1",
        'sh -c "node server.js"',
        # quoting is read before the operators; an unreadable command declares nothing
        'echo "next: tsc; node legacy.js; done"',
        "node 'legacy.js",
        "node server.js 'x",
        # a directory that holds a package.json: its `main` decides, and the field reader reads that
        "node .",
        "electron .",
        # outside the package
        "node ../server.js",
        "node /server.js",
        "node /lib",
        "ts-node /src",
    ],
)
def test_every_doubt_declares_nothing(command):
    assert _roots(command) == set(), command


def test_a_preload_is_a_root_only_as_a_relative_path():
    assert _roots("node --require ./preload.js server.js") == {"preload.js", "server.js"}
    assert _roots("node --import=./preload.js server.js") == {"preload.js", "server.js"}
    assert _roots("node --watch -r ./preload.js server.js") == {"preload.js", "server.js"}
    assert _roots("bun --preload ./preload.js server.js") == {"preload.js", "server.js"}
    assert _roots("node -r ./esm server.js") == {"esm.js", "server.js"}
    assert _roots("node -r ../preload.js server.js", pkg_dir="lib") == {"preload.js", "lib/server.js"}
    assert _roots("node -r ./preload.js --check legacy.js") == set()  # a check executes nothing


def test_node_does_not_try_typescript_extensions_and_deno_tries_none():
    files = frozenset({"package.json", "worker.ts", "job.js"})
    assert _roots("node ./worker", files=files) == set()
    assert _roots("ts-node ./worker", files=files) == {"worker.ts"}
    assert _roots("node ./job", files=files) == {"job.js"}
    assert _roots("deno run ./job", files=files) == set()
    assert _roots("bun ./job", files=files) == set()


def test_a_nested_package_resolves_against_its_own_directory():
    files = frozenset({"server.js", "packages/api/server.js", "packages/api/package.json"})
    assert _roots("node server.js", pkg_dir="packages/api", files=files) == {"packages/api/server.js"}
    assert _roots("node ../../server.js", pkg_dir="packages/api", files=files) == {"server.js"}
    assert _roots("node ../../../server.js", pkg_dir="packages/api", files=files) == set()


def test_a_directory_argument_defers_to_that_directorys_manifest():
    """`node .` runs `main`. With `main` outside the index, a stale `index.js` is not the entry."""
    files = frozenset({"package.json", "index.js", "tools/package.json", "tools/index.js", "plain/index.js"})
    assert _roots("node .", files=files) == set()
    assert _roots("node ./tools", files=files) == set()
    assert _roots("node ./plain", files=files) == {"plain/index.js"}  # no manifest: node loads index.js
    assert _roots("node .", files=files, own_main=False) == {"index.js"}  # our manifest names no main
    assert _roots("node ./tools", files=files, own_main=False) == set()  # tools' own manifest decides

# ── Review round 5: what the allowlist trusts ────────────────────────────────


@pytest.mark.parametrize(
    "command",
    [
        "node --env-file .env server.js",  # node reads the next token as the value (measured on node 24)
        "node --unhandled-rejections strict server.js",
        "npx --yes nodemon server.js",
        "cross-env-shell NODE_ENV=production node server.js",
        "false || node server.js",
        "test -f x && node server.js",
        "node legacy.ts ; node server.js",  # an entry that is not indexed, then one that is
    ],
)
def test_more_forms_that_run_the_server(command):
    assert _roots(command) == {"server.js"}, command


@pytest.mark.parametrize(
    "command",
    [
        "node --env-file legacy.js server.js",  # a value that is itself a runnable file: doubt
        # a directory change the first word does not show
        "FOO=1 cd lib && node server.js",
        "CDPATH=. cd lib && node server.js",
        "\\cd lib && node server.js",
        "popd; node server.js",
        # a comment, and an operator that is only a quoted string
        "true # && node server.js",
        'echo "&&" node server.js',
        'echo ";" node server.js',
        "echo '|' node server.js",
        # shell words anywhere in the command
        "eval node server.js",
        "source env.sh; node server.js",
        "export A=1; node server.js",
        "set -e; node server.js",
        "A=1 eval node server.js",
        "for f in a b; do node server.js; done",
        "while true; do node server.js; done",
        "while true; do cd lib; done; node server.js",
        "for d in lib; do cd lib; done; node server.js",
        "node server.js < input.txt",
        # expansions and rewrites
        "node server\\ extra.js",
        "node ./$D/../server.js",
        "node lib/$D/../../server.js",
        "node ${ENTRY:-server.js}",
        "node src/*.ts",
        "node ~/server.js",
        "node %CD%/server.js",
        "node `which server.js`",
        "FOO=$HOME node server.js",
        # electron has no inspect word
        "electron inspect server.js",
    ],
)
def test_more_doubts_declare_nothing(command):
    assert _roots(command) == set(), command


def test_each_runner_appends_only_its_own_extensions():
    files = frozenset({"package.json", "worker.ts", "job.js"})
    for runner in ("node", "nodejs", "electron", "babel-node", "pm2-runtime"):
        assert _roots(f"{runner} ./worker", files=files) == set(), runner
        assert _roots(f"{runner} ./job", files=files) == {"job.js"}, runner
    # nodemon picks the extension from `-e`, and ESM resolution appends none
    for runner in ("nodemon", "ts-node-esm"):
        assert _roots(f"{runner} ./worker", files=files) == set(), runner
        assert _roots(f"{runner} ./job", files=files) == set(), runner
        assert _roots(f"{runner} ./job.js", files=files) == {"job.js"}, runner
    for runner in ("ts-node", "ts-node-dev", "tsx"):
        assert _roots(f"{runner} ./worker", files=files) == {"worker.ts"}, runner


# The flags that take their value in the NEXT token, written out here and not
# read from the table: moving one into a runner's `flags` set makes the value
# (a bare word below) end the command, and this fails. This is the table error
# that can root a wrong file, so every entry is pinned. `test_every_value_flag_is_written_out_for_every_runner`
# holds this list equal to the table.
_NODE_VALUES = ["-C", "--conditions", "--watch-path", "--inspect-port", "--title", "--input-type", "--env-file", "--unhandled-rejections"]
_TS_NODE_VALUES = ["-P", "--project", "-C", "--compiler", "-O", "--compiler-options", "--compilerOptions", "-I", "--ignore",
                   "--scope-dir", "--scopeDir", "-D", "--ignore-diagnostics", "--transpiler"]
VALUE_FLAGS = {
    "node": _NODE_VALUES,
    "nodejs": _NODE_VALUES,
    "electron": _NODE_VALUES,
    "tsx": _NODE_VALUES + ["--tsconfig"],
    "ts-node": _TS_NODE_VALUES,
    "ts-node-esm": _TS_NODE_VALUES,
    "ts-node-dev": ["-C", "--compiler", "-P", "--project", "-I", "--ignore", "-D", "--ignore-diagnostics", "-O",
                    "--compiler-options", "--scopeDir", "--transpiler", "--deps-level", "--compile-timeout",
                    "--ignore-watch", "--interval", "--debounce", "--watch", "--cache-directory"],
    "nodemon": ["-w", "--watch", "-e", "--ext", "-i", "--ignore", "--config", "-d", "--delay", "-s", "--signal",
                "-P", "--polling-interval"],
    "babel-node": ["--presets", "--plugins", "--extensions", "-x", "--config-file", "--ignore", "--only", "--env-name",
                   "--root-mode"],
    "bun": ["--env-file", "-d", "--define", "-l", "--loader", "--tsconfig-override", "--port",
            "--conditions"],
    "deno run": ["-c", "--config", "--import-map", "--lock", "--cert", "--location", "--seed", "-L",
                 "--log-level"],
    "pm2-runtime": [],
}


@pytest.mark.parametrize("runner,flag", [(r, f) for r, flags in VALUE_FLAGS.items() for f in flags])
def test_a_value_flag_consumes_its_value_for_its_runner(runner, flag):
    assert _roots(f"{runner} {flag} somevalue ./server.js") == {"server.js"}, (runner, flag)
    # and the value is never the entry: a runnable file there is a doubt, not a root
    assert _roots(f"{runner} {flag} ./legacy.js ./server.js") == set(), (runner, flag)

# ── Review round 6: the command text is an allowlist too ─────────────────────

# Every character a shell gives meaning to, and whitespace other than a space.
_UNREAD = "$`*?~%#()<>{}[]!^" + chr(10) + chr(9) + chr(13)


@pytest.mark.parametrize("char", list(_UNREAD), ids=[f"U+{ord(c):04X}" for c in _UNREAD])
def test_a_character_the_shell_interprets_makes_the_command_unread(char):
    """Anywhere in the command: before the entry, after it, or in another segment."""
    assert _roots("node server.js") == {"server.js"}
    assert _roots(f"node server.js ; echo x{char}y") == set(), repr(char)
    assert _roots(f"echo x{char}y ; node server.js") == set(), repr(char)
    assert _roots(f"node server.js --flag a{char}b") == set(), repr(char)


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("inside", ["&&", ";", "||", "|", "&"])
def test_an_operator_inside_quotes_is_not_an_operator(quote, inside):
    assert _roots(f"echo {quote}{inside}{quote} node server.js") == set(), (quote, inside)
    assert _roots(f"echo {quote}a {inside} node server.js{quote}") == set(), (quote, inside)


@pytest.mark.parametrize(
    "command",
    [
        # a backslash is a path separator inside a word and nothing else
        "\\cd lib && node server.js",
        "\\command cd lib && node server.js",
        "\\builtin cd lib && node server.js",
        # an escaped runner is still a runner to a shell; only the word-start rule refuses it
        "\\node server.js",
        "A=1 \\node server.js",
        "npx \\nodemon server.js",
        "true && \\node server.js",
        "true;\\node server.js",
        "node server\\ extra.js",
        'echo "a \\" && node server.js \\" b"',
        "node server.js \\; echo x",
        "node .\\\\server.js",
        # a shell word behind an assignment, and in a later segment
        "FOO=1 eval cd lib; node server.js",
        "FOO=1 command cd lib && node server.js",
        "FOO=1 builtin cd lib && node server.js",
        "FOO=1 source ./env.sh && node server.js",
        "true && eval cd lib && node server.js",
        "node server.js; set -e",
        # a newline separates commands in a shell and is whitespace to the lexer
        "node legacy.js\ncd lib && node server.js",
    ],
)
def test_shell_syntax_the_lexer_would_misread_is_not_read(command):
    assert _roots(command) == set(), command


def test_the_wrapper_order_npx_then_cross_env():
    assert _roots("npx cross-env A=1 node server.js") == {"server.js"}
    assert _roots("npx -y cross-env A=1 B=2 node server.js") == {"server.js"}
    assert _roots("A=1 npx nodemon server.js") == {"server.js"}
    assert _roots("cross-env A=1 npx nodemon server.js") == set()  # not a form the reader knows


def test_every_value_flag_is_written_out_for_every_runner():
    """`VALUE_FLAGS` below is the whole table, not a sample: a flag added to or
    dropped from a runner's `value` set fails here until this list says the same."""
    table = _entry_points._SCRIPT_RUNNERS
    assert {name.split()[0] for name in VALUE_FLAGS} == set(table), sorted(table)
    for name, flags in VALUE_FLAGS.items():
        assert set(flags) == set(table[name.split()[0]]["value"]), name
        assert len(flags) == len(set(flags)), name


# ── The reader, from manifest text (no index build) ──────────────────────────


class _Index:
    def __init__(self, files):
        self.source_files = list(files)


class _Store:
    def __init__(self, files):
        self._files = files

    def get_file_content(self, _owner, _name, path):
        return self._files.get(path)


def _entries(files: dict[str, str]) -> set[str]:
    return _entry_points.package_json_entries(_Index(files), _Store(files), "o", "r")


@pytest.mark.parametrize(
    "manifest,roots",
    [
        ({"scripts": {"start": "node server.js"}}, {"server.js"}),
        ({"scripts": {"start": "node .\\server.js"}}, {"server.js"}),
        ({"scripts": {"a": "eslint server.js", "b": "node lib/server.js"}}, {"lib/server.js"}),
        ({"scripts": {"start": "cd lib && node server.js"}}, set()),
        ({"scripts": "node server.js"}, set()),
        ({"scripts": ["node server.js"]}, set()),
        ({"scripts": {"start": {"cmd": "node server.js"}}}, set()),
        ({"scripts": {"start": ["node", "server.js"], "x": None}}, set()),
        ({"scripts": {"start": "node ."}}, {"index.js"}),
        ({"scripts": {"start": "node ."}, "module": "dist/x.mjs"}, {"index.js"}),  # node reads `main` only
        ({"scripts": {"start": "node ."}, "main": ""}, {"index.js"}),
        ({"scripts": {"start": "node ."}, "main": 5}, {"index.js"}),
        ({"scripts": {"start": "node ."}, "main": "dist/server.js"}, set()),
        ({"scripts": {"start": "node ."}, "main": "server.js"}, {"server.js"}),
    ],
)
def test_the_reader_passes_the_manifest_to_the_rule(manifest, roots):
    files = {"package.json": json.dumps(manifest), "server.js": "", "index.js": "", "lib/server.js": ""}
    assert _entries(files) == roots, manifest


def test_the_reader_resolves_a_nested_manifest_against_its_directory():
    files = {
        "package.json": json.dumps({"scripts": {}}),
        "packages/api/package.json": json.dumps({"scripts": {"start": "node server.js"}}),
        "packages/api/server.js": "",
        "server.js": "",
    }
    assert _entries(files) == {"packages/api/server.js"}


# ── The tools, end to end ────────────────────────────────────────────────────


def _index(root: Path, files: dict[str, str]) -> tuple[str, str]:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    storage = str(root / ".index")
    res = index_folder(str(root), use_ai_summaries=False, storage_path=storage)
    return res.get("repo", str(root)), storage


def _pkg(scripts, **extra) -> str:
    return json.dumps({"name": "fx", "version": "1.0.0", "scripts": scripts, **extra})


def _dead_files(root: Path, files: dict[str, str]) -> set[str]:
    repo, storage = _index(root, files)
    result = fdc.find_dead_code(repo, granularity="file", min_confidence=0.0, storage_path=storage)
    assert "error" not in result, result
    return {d["file"] for d in result["dead_files"]}


JS = {
    "server.js": "const { area } = require('./lib/shapes.js');\nfunction boot() { return area(); }\nboot();\n",
    "lib/shapes.js": "function area() { return 1; }\nmodule.exports = { area };\n",
    "legacy.js": "function old() { return 0; }\nmodule.exports = { old };\n",
}


def test_find_dead_code_keeps_the_server_and_what_it_imports(tmp_path):
    dead = _dead_files(tmp_path, {**JS, "package.json": _pkg({"start": "cross-env NODE_ENV=production node server.js"})})
    assert "server.js" not in dead and "lib/shapes.js" not in dead, dead
    assert "legacy.js" in dead, dead


@pytest.mark.parametrize(
    "scripts",
    [{}, {"lint": "eslint server.js legacy.js"}, {"x": "node --check server.js"}, {"x": "cd lib && node ../server.js"}],
)
def test_find_dead_code_still_reports_the_fixture_when_nothing_runs_it(tmp_path, scripts):
    """The fixture can fail: with no script that RUNS a file, every file in it is reported."""
    dead = _dead_files(tmp_path, {**JS, "package.json": _pkg(scripts)})
    assert {"server.js", "lib/shapes.js", "legacy.js"} <= dead, (scripts, dead)


def test_a_nested_package_end_to_end(tmp_path):
    files = {
        "package.json": _pkg({}),
        "packages/api/package.json": _pkg({"start": "node server.js"}),
        **{f"packages/api/{rel}": body for rel, body in JS.items()},
        "server.js": "function unrelated() { return 2; }\nmodule.exports = { unrelated };\n",
    }
    dead = _dead_files(tmp_path, files)
    assert "packages/api/server.js" not in dead and "packages/api/lib/shapes.js" not in dead, dead
    assert "server.js" in dead, dead  # the root file of the same name is not what the nested script runs


def test_the_fields_read_before_are_still_read(tmp_path):
    files = {**JS, "package.json": _pkg({}, main="server.js", bin={"fx": "legacy.js"})}
    dead = _dead_files(tmp_path, files)
    assert dead.isdisjoint({"server.js", "lib/shapes.js", "legacy.js"}), dead


def test_both_dead_code_tools_ask_the_one_reader():
    assert fdc._package_json_entries is _entry_points.package_json_entries
    assert v2._package_json_entries is _entry_points.package_json_entries
    for name in ("find_dead_code.py", "get_dead_code_v2.py"):
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        local = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and "package_json" in n.name]
        assert local == [], (name, local)


def test_get_dead_code_v2_does_not_call_the_server_unreachable(tmp_path):
    repo, storage = _index(tmp_path, {**JS, "package.json": _pkg({"start": "node server.js"})})
    result = v2.get_dead_code_v2(repo, min_confidence=0.0, storage_path=storage)
    assert "error" not in result, result
    flagged = {
        s["file"] for s in result["dead_symbols"] if "unreachable_file" in (s.get("signals") or s.get("reasons") or [])
    }
    assert "server.js" not in flagged and "lib/shapes.js" not in flagged, result["dead_symbols"]


def test_the_deletion_investigator_reads_the_server_as_a_live_importer(tmp_path):
    """At the destructive surface: a name only the script-run server imports is not deletable."""
    files = {
        "package.json": json.dumps({"name": "fx", "type": "module", "scripts": {"start": "node src/server.js"}}),
        "src/shapes.js": "export function area() { return 1; }\n",
        "src/server.js": "import { area } from './shapes.js';\nexport function boot() { return area(); }\nboot();\n",
    }
    repo, storage = _index(tmp_path, files)
    result = investigate_deletion_safety(repo, "src/shapes.js::area#function", storage_path=storage)
    assert "error" not in result, result
    obligation = next(o for o in result["obligations"] if o["obligation"] == "export_not_imported")
    assert obligation["status"] == REFUTED, obligation
    assert "unreachable" not in json.dumps(obligation), obligation


# -- Review round 8: each case below was run against the installed runner ------


@pytest.mark.parametrize(
    "command, expected",
    [
        # nodemon 3.1.14 reads its own options after the script (`--cwd`, `--exec`, `-e`)
        # and leaves node's there in the script's argv: any option after it declares nothing
        ("nodemon server.js --cwd lib", set()),
        ("nodemon server.js --port 3000", set()),
        ("nodemon server.js --watch lib", set()),
        ("nodemon server.js --exec babel-node", set()),
        ('nodemon legacy.js --exec "node server.js"', set()),
        ('nodemon server.js -x "node legacy.js"', set()),
        ("nodemon server.js -r ./legacy.js", set()),
        ("nodemon server.js --require ./legacy.js", set()),
        ("nodemon server.js -e ts", set()),
        # before the script they are read
        ("nodemon -r ./legacy.js server.js", {"legacy.js", "server.js"}),
        ("nodemon --exec babel-node server.js", {"server.js"}),
        ('nodemon --exec "node legacy.js" server.js', {"legacy.js"}),
        # nodemon takes the first argument that exists, and expands an extensionless one
        ("nodemon ./server ./legacy.js", set()),
        ("nodemon -e ts ./server", set()),
        ("nodemon server.js -- --cwd lib", {"server.js"}),
        ("nodemon server.js legacy.js", {"server.js"}),
        ("nodemon server.js production", {"server.js"}),
        # the `=` form is unambiguous where the runner reads it: the value is a value
        ("node --env-file=config.js server.js", {"server.js"}),
        ("node --env-file config.js server.js", set()),
        # and hands `--flag=value` to node: `--ignore=x` fails there, `--inspect=9231` works
        ("nodemon --inspect=9231 server.js", set()),
        ("nodemon --ignore=legacy.js server.js", set()),
        ("nodemon --exec=node server.js", set()),
        # node stops at the script: what follows is the script's
        ("node server.js --cwd lib", {"server.js"}),
        # tsx knows its watch filters only behind `watch`; alone they go to node and fail
        ("tsx --ignore extras src/server.ts", set()),
        ("tsx watch --ignore extras src/server.ts", set()),
        ("tsx watch src/server.ts", {"src/server.ts"}),
    ],
)
def test_options_after_the_script_and_the_equals_form_are_per_runner(command, expected):
    assert _roots(command) == expected, command


@pytest.mark.parametrize("runner", ["ts-node", "ts-node-dev", "tsx"])
def test_an_extensionless_path_that_names_two_files_declares_nothing(runner):
    """ts-node and tsx run `server.js` when `server.ts` sits beside it; ts-node-dev
    runs `server.ts`; `--prefer-ts-exts` flips ts-node. Two candidates are not decided."""
    both = frozenset({"package.json", "server.js", "server.ts", "src/index.js", "src/index.ts"})
    assert _roots(f"{runner} ./server", files=both) == set()
    assert _roots(f"{runner} ./src", files=both) == set()
    assert _roots(f"{runner} ./server.ts", files=both) == {"server.ts"}
    one = frozenset({"package.json", "server.ts", "src/index.ts"})
    assert _roots(f"{runner} ./server", files=one) == {"server.ts"}
    assert _roots(f"{runner} ./src", files=one) == {"src/index.ts"}


# Written out, and compared for equality: a loop over the source's own set cannot
# fail when a word leaves it.
SHELL_WORDS = [
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "{", "}", "!", "function", "builtin", "command", "eval", "source", ".", "set", "export",
    "exit", "exec", "return", "alias", "unalias", "trap", "unset", "shift", "break", "continue",
]


def test_every_shell_word_in_any_segment_makes_the_command_unread():
    assert _roots("node server.js") == {"server.js"}
    assert set(SHELL_WORDS) == set(_entry_points._SHELL_WORDS)
    for word in SHELL_WORDS:
        assert _roots(f"{word} x; node server.js") == set(), word
        assert _roots(f"node server.js && {word} x") == set(), word


@pytest.mark.parametrize("runner", ["node", "ts-node", "tsx"])
def test_a_json_file_beside_a_directory_of_the_same_stem_declares_nothing(runner):
    """`require.resolve('./x')` is `x.json` when both `x.json` and `x/index.js` exist (node 24, run)."""
    index = "x/index.js" if runner == "node" else "x/index.ts"
    files = frozenset({"package.json", "x.json", index})
    assert _roots(f"{runner} ./x", files=files) == set()
    assert _roots(f"{runner} ./x", files=files - {"x.json"}) != set()


# -- Review round 10: ESM resolution appends no extension (node 24, ts-node 10.9.2, run) --


@pytest.mark.parametrize(
    "command, expected",
    [
        # `node --import ./b x.js` fails with ERR_MODULE_NOT_FOUND: it runs nothing
        ("node --import ./preload server.js", set()),
        ("node --loader ./preload server.js", set()),
        ("node --experimental-loader ./preload server.js", set()),
        ("node --import ./preload.js server.js", {"preload.js", "server.js"}),
        # CommonJS `-r` appends `.js`
        ("node -r ./preload server.js", {"preload.js", "server.js"}),
        # tsx resolves the path itself
        ("tsx --import ./preload server.js", {"preload.js", "server.js"}),
        # `ts-node --esm ./a` fails the same way; the flag is unknown here
        ("ts-node --esm ./server", set()),
        ("ts-node --esm ./server.ts", set()),
    ],
)
def test_esm_resolution_appends_no_extension(command, expected):
    assert _roots(command) == expected, command


def test_a_json_file_shadows_only_what_the_runner_tries_after_it():
    """node tries `y.js` before `y.json` (run: `node ./y` ran `y.js`); ts-node tries `.json` before `.ts`."""
    files = frozenset({"package.json", "y.js", "y.json", "z.ts", "z.json"})
    assert _roots("node ./y", files=files) == {"y.js"}
    assert _roots("ts-node ./z", files=files) == set()
    assert _roots("ts-node ./z.ts", files=files) == {"z.ts"}


@pytest.mark.parametrize(
    "command",
    [
        # node 24, run: each fails at start with ERR_MODULE_NOT_FOUND
        "node --import ./q server.js",  # only q.mjs exists
        "node --import ./t server.js",  # only t.ts exists
        "tsx --loader ./preload server.js",
        # a preload that is not indexed is not decided either
        "node --import ./missing.js server.js",
    ],
)
def test_an_esm_preload_that_does_not_resolve_as_written_declares_nothing(command):
    files = FILES | {"q.mjs", "t.ts"}
    assert _roots(command, files=files) == set(), command
