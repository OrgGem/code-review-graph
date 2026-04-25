"""Tests for CLI helpers."""

import argparse
import logging
from importlib.metadata import PackageNotFoundError

from code_review_graph import cli


def test_get_version_logs_and_falls_back_to_dev(monkeypatch, caplog):
    def _raise_package_not_found(_dist_name: str) -> str:
        raise PackageNotFoundError("code-review-graph")

    monkeypatch.setattr(cli, "pkg_version", _raise_package_not_found)

    with caplog.at_level(logging.DEBUG, logger="code_review_graph.cli"):
        version = cli._get_version()

    assert version == "dev"
    assert "Package metadata unavailable" in caplog.text


def test_instruction_files_for_platform_targets(tmp_path):
    targets = set(cli._instruction_files_to_modify(tmp_path, "antigravity"))
    assert "AGENTS.md (new)" in targets
    assert "GEMINI.md (new)" in targets

    targets = set(cli._instruction_files_to_modify(tmp_path, "kiro"))
    assert ".kiro/steering/code-review-graph.md (new)" in targets


def test_handle_init_installs_skills_and_instructions_for_three_agents(monkeypatch, tmp_path):
    from code_review_graph import incremental, skills

    monkeypatch.setattr(incremental, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(incremental, "ensure_repo_gitignore_excludes_crg", lambda _root: "unchanged")

    calls: dict[str, list] = {
        "install_platform_configs": [],
        "generate_skills": [],
        "inject_claude_md": [],
        "inject_platform_instructions": [],
    }

    monkeypatch.setattr(
        skills,
        "install_platform_configs",
        lambda repo_root, target, dry_run=False: (
            calls["install_platform_configs"].append((repo_root, target, dry_run)) or [target]
        ),
    )
    monkeypatch.setattr(
        skills,
        "generate_skills",
        lambda repo_root: (calls["generate_skills"].append(repo_root) or (repo_root / ".claude" / "skills")),
    )
    monkeypatch.setattr(
        skills,
        "inject_claude_md",
        lambda repo_root: calls["inject_claude_md"].append(repo_root),
    )
    monkeypatch.setattr(
        skills,
        "inject_platform_instructions",
        lambda repo_root, target="all": (
            calls["inject_platform_instructions"].append((repo_root, target)) or []
        ),
    )

    for platform in ("claude", "kiro", "antigravity"):
        args = argparse.Namespace(
            repo=str(tmp_path),
            dry_run=False,
            platform=platform,
            yes=True,
            no_instructions=False,
            no_skills=False,
            no_hooks=True,
        )
        cli._handle_init(args)

    seen_targets = [t for _, t, _ in calls["install_platform_configs"]]
    assert seen_targets == ["claude", "kiro", "antigravity"]
    assert len(calls["generate_skills"]) == 3
    assert [t for _, t in calls["inject_platform_instructions"]] == ["claude", "kiro", "antigravity"]
    assert len(calls["inject_claude_md"]) == 1
