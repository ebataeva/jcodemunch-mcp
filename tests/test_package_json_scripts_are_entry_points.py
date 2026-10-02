"""A file a `package.json` script runs is an entry point (LEDGER L-102).

`_package_json_entries` read `main`, `module`, `browser`, `exports` and `bin`,
and not `scripts`. A server started by `"start": "node server.js"` has no
importer by construction, so `find_dead_code` reported it dead at confidence
1.0, everything only it imports as `all_importers_dead`, and the deletion
investigator then read a name it imports as "imported only by server.js, which
is itself unreachable".

The reader also existed twice, once in `find_dead_code` and once in
`get_dead_code_v2`, byte for byte. It is one function in `tools/_entry_points.py`
now, so a field read by one is read by both.

The other direction is tested as firmly (#569: a fix for a false positive can
install a false negative). A script that NAMES a file without running it
(`eslint legacy.js`, `node build.js input.js`) declares nothing about it.
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


def _index(root: Path, files: dict[str, str]) -> tuple[str, str]:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    storage = str(root / ".index")
    res = index_folder(str(root), use_ai_summaries=False, storage_path=storage)
    return res.get("repo", str(root)), storage


def _pkg(scripts: dict[str, str], **extra) -> str:
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


@pytest.mark.parametrize(
    "command",
    [
        "node server.js",
        "node ./server.js",
        "node server",
        'node "server.js"',
        "NODE_ENV=production node server.js",
        "cross-env NODE_ENV=production PORT=1 node server.js",
        "npx nodemon server.js",
        "nodemon --watch lib server.js",
        "node -r dotenv/config server.js",
        "node --inspect=0.0.0.0:9229 server.js",
        "npm run build && node server.js",
        "tsc -p . ; node server.js",
        "node server.js --port 3000",
        "pm2-runtime server.js",
    ],
)
def test_a_file_a_script_runs_is_live_and_so_is_what_it_imports(tmp_path, command):
    dead = _dead_files(tmp_path, {**JS, "package.json": _pkg({"start": command})})
    assert "server.js" not in dead, (command, dead)
    assert "lib/shapes.js" not in dead, (command, dead)
    assert "legacy.js" in dead, (command, dead)


@pytest.mark.parametrize(
    "command",
    [
        "eslint server.js legacy.js",
        "prettier --write server.js",
        "echo node server.js",
        "node build.js server.js",
        "node dist/build.js --out x server.js",
        "bun run lint server.js",
        "jest server.js",
        "server.js",
        "npm run server.js",
        "node",
    ],
)
def test_a_script_that_names_a_file_without_running_it_declares_nothing(tmp_path, command):
    dead = _dead_files(tmp_path, {**JS, "package.json": _pkg({"lint": command})})
    assert {"server.js", "lib/shapes.js", "legacy.js"} <= dead, (command, dead)


def test_no_script_at_all_leaves_the_fixture_dead(tmp_path):
    """The fixture can fail: with no declaration every file in it is reported."""
    dead = _dead_files(tmp_path, {**JS, "package.json": _pkg({})})
    assert {"server.js", "lib/shapes.js", "legacy.js"} <= dead, dead


@pytest.mark.parametrize(
    "command,entry",
    [
        ("ts-node src/index.ts", "src/index.ts"),
        ("tsx watch src/index.ts", "src/index.ts"),
        ("npx tsx src/index.ts", "src/index.ts"),
        ("bun run src/index.ts", "src/index.ts"),
        ("deno run --allow-net src/index.ts", "src/index.ts"),
        ("ts-node-dev --respawn src/index.ts", "src/index.ts"),
        ("node --loader ts-node/esm src/index.ts", "src/index.ts"),
        ("node src", "src/index.ts"),
    ],
)
def test_typescript_runners_and_a_directory_argument(tmp_path, command, entry):
    files = {
        "package.json": _pkg({"dev": command}),
        "src/index.ts": "import { area } from './shapes';\nexport function boot(): number { return area(); }\n",
        "src/shapes.ts": "export function area(): number { return 1; }\n",
        "src/legacy.ts": "export function old(): number { return 0; }\n",
    }
    dead = _dead_files(tmp_path, files)
    assert entry not in dead and "src/shapes.ts" not in dead, (command, dead)
    assert "src/legacy.ts" in dead, (command, dead)


def test_a_preload_named_by_require_is_live_too(tmp_path):
    files = {**JS, "preload.js": "process.env.X = '1';\n", "package.json": _pkg({"start": "node --require ./preload.js server.js"})}
    dead = _dead_files(tmp_path, files)
    assert "preload.js" not in dead and "server.js" not in dead, dead
    assert "legacy.js" in dead, dead


def test_a_nested_package_resolves_against_its_own_directory(tmp_path):
    files = {
        "package.json": _pkg({}),
        "packages/api/package.json": _pkg({"start": "node server.js"}),
        **{f"packages/api/{rel}": body for rel, body in JS.items()},
        "server.js": "function unrelated() { return 2; }\nmodule.exports = { unrelated };\n",
    }
    dead = _dead_files(tmp_path, files)
    assert "packages/api/server.js" not in dead and "packages/api/lib/shapes.js" not in dead, dead
    assert "server.js" in dead, dead  # the root file of the same name is not what the nested script runs


def test_a_script_that_is_not_a_string_is_ignored(tmp_path):
    body = json.dumps({"name": "fx", "scripts": {"start": ["node", "server.js"], "x": None}})
    assert "server.js" in _dead_files(tmp_path, {**JS, "package.json": body})


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
