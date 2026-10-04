"""A repeat `index <subdir>` inside a git root is incremental (#961).

#504 fixed the repeat FULL-root walk. A subdir walk takes the merge path
instead (`_merge_with_existing` is set so files outside `walk_prefix` carry
over), and the incremental branch was gated on `_merge_with_existing is None`,
so every repeat subdir walk re-parsed the whole subdir.

⚠ A no-change path taken unconditionally would satisfy the first test and
index nothing; the second and fourth tests are the controls against that.
"""

import subprocess

import pytest

from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools.index_folder import index_folder


def _git(cwd, *args):
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path):
    work = tmp_path / "demo"
    work.mkdir()
    _git(work, "init", "-q")
    _git(work, "config", "user.email", "repro@example.invalid")
    _git(work, "config", "user.name", "repro")
    _git(work, "remote", "add", "origin", "https://github.com/acme/demo.git")
    (work / "main.py").write_text("def main():\n    return 1\n")
    (work / "pkg").mkdir()
    (work / "pkg" / "a.py").write_text("def alpha():\n    return 1\n")
    (work / "pkg" / "b.py").write_text("def beta():\n    return 2\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "A")

    def _index(target, store):
        return index_folder(
            str(target), use_ai_summaries=False, storage_path=str(store),
            identity_mode="git",
        )

    class Repo:
        path = work

        def __init__(self, store_dir):
            store_dir.mkdir()
            self.store_path = str(store_dir)

        def index(self):
            return _index(work, self.store_path)

        def index_subdir(self, rel):
            return _index(work / rel, self.store_path)

        def load(self, result):
            owner, name = result["repo"].split("/", 1)
            return IndexStore(base_path=self.store_path).load_index(owner, name)

    Repo.new = staticmethod(lambda name="store": Repo(tmp_path / name))
    return Repo


def _snapshot(index):
    return (
        dict(index.file_hashes),
        sorted(s["id"] if isinstance(s, dict) else s.id for s in index.symbols),
    )


class TestRepeatSubdirWalk:
    def test_a_repeat_subdir_walk_takes_the_no_change_path(self, repo):
        r = repo.new()
        first = r.index_subdir("pkg")
        assert first["success"] is True

        again = r.index_subdir("pkg")

        assert again["performed_incremental"] is True, (
            "a repeat subdir walk re-parsed the whole subdir"
        )
        assert again.get("message") == "No changes detected"

    def test_edits_inside_the_prefix_are_applied_and_carried_files_kept(self, repo):
        r = repo.new()
        r.index()
        r.index_subdir("pkg")

        (repo.path / "pkg" / "a.py").write_text("def alpha_v2():\n    return 9\n")
        (repo.path / "pkg" / "c.py").write_text("def gamma():\n    return 3\n")
        (repo.path / "pkg" / "b.py").unlink()

        result = r.index_subdir("pkg")

        assert result["performed_incremental"] is True
        assert (result["changed"], result["new"], result["deleted"]) == (1, 1, 1)

        index = r.load(result)
        names = {s["name"] if isinstance(s, dict) else s.name for s in index.symbols}
        assert {"alpha_v2", "gamma", "main"} <= names
        assert not {"alpha", "beta"} & names
        assert "main.py" in index.file_hashes, (
            "a file outside the walked prefix was pruned as deleted"
        )

    def test_a_subdir_walk_does_not_prune_outside_its_prefix(self, repo):
        """Pruning a file outside `walk_prefix` is the root walk's job; the
        subdir walk never saw it, so absence from its walk is not deletion."""
        r = repo.new()
        r.index()
        r.index_subdir("pkg")
        (repo.path / "main.py").unlink()

        result = r.index_subdir("pkg")

        assert result["performed_incremental"] is True
        assert result.get("deleted", 0) == 0
        assert "main.py" in r.load(result).file_hashes

    def test_the_incremental_result_matches_a_fresh_build(self, repo):
        inc = repo.new("store_inc")
        inc.index()
        inc.index_subdir("pkg")
        (repo.path / "pkg" / "a.py").write_text("def alpha_v2():\n    return 9\n")
        (repo.path / "pkg" / "c.py").write_text("def gamma():\n    return 3\n")
        (repo.path / "pkg" / "b.py").unlink()
        inc_result = inc.index_subdir("pkg")
        assert inc_result["performed_incremental"] is True  # precondition

        fresh = repo.new("store_fresh")
        fresh.index()
        fresh_result = fresh.index_subdir("pkg")

        assert _snapshot(inc.load(inc_result)) == _snapshot(fresh.load(fresh_result))

    def test_a_first_walk_of_a_new_prefix_still_merges(self, repo):
        """A prefix not yet in `source_roots` has no stored coverage to diff
        against, so it keeps the merge path and records the prefix."""
        r = repo.new()
        r.index()

        result = r.index_subdir("pkg")

        assert result["performed_incremental"] is not True
        index = r.load(result)
        assert "main.py" in index.file_hashes
        assert "pkg" in (index.source_roots or [])
