"""
FAMOSE integration with the Cursor IDE CLI: headless ``agent`` with ``--model auto`` only.

Launches ``Cursor.exe`` + ``resources/app/out/cli.js`` when needed (Windows), with
``ELECTRON_RUN_AS_NODE=1`` so flags are handled by the Node CLI instead of Electron.

Standalone ``agent`` (``curl https://cursor.com/install | bash``) is Linux/macOS or WSL only;
see repo ``scripts/install_cursor_agent.sh`` / ``scripts/install_cursor_agent.ps1``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_TIMEOUT_SEC = 600.0
_ENV_CLI_OVERRIDE = "FAMOSE_CURSOR_CLI"


def resolve_workspace(raw: str | None, repo_root: Path) -> str:
    """Default workspace is the repository root; relative paths join to ``repo_root``."""
    if raw is None or not str(raw).strip():
        return str(repo_root)
    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve())
    return str((repo_root / p).resolve())


def _electron_pair_from_root(install_root: Path) -> list[str] | None:
    exe = install_root / "Cursor.exe"
    js = install_root / "resources" / "app" / "out" / "cli.js"
    if exe.is_file() and js.is_file():
        return [str(exe), str(js)]
    return None


def _electron_pair_from_app_bin_entry(entry: Path) -> list[str] | None:
    """
    ``resources/app/bin/cursor`` (no extension) or ``cursor.cmd`` — same layout as the IDE shim.
    ``shutil.which("cursor")`` on Windows often returns the extensionless file; without resolving
    to ``Cursor.exe`` + ``cli.js`` and ``ELECTRON_RUN_AS_NODE``, args are swallowed by Electron.
    """
    try:
        resolved = entry.resolve()
    except OSError:
        return None
    name = resolved.name.lower()
    if name not in ("cursor", "cursor.cmd"):
        return None
    try:
        install_root = resolved.parents[3]
    except IndexError:
        return None
    return _electron_pair_from_root(install_root)


def _scan_windows_programs_cursor() -> list[str] | None:
    la = (os.environ.get("LOCALAPPDATA") or "").strip()
    if not la:
        return None
    programs = Path(la) / "Programs"
    for name in ("cursor", "Cursor"):
        pair = _electron_pair_from_root(programs / name)
        if pair is not None:
            return pair
    return None


def _scan_macos_app_bundle() -> list[str] | None:
    app = Path("/Applications/Cursor.app")
    exe = app / "Contents" / "MacOS" / "Cursor"
    js = app / "Contents" / "Resources" / "app" / "out" / "cli.js"
    if exe.is_file() and js.is_file():
        return [str(exe), str(js)]
    return None


def resolve_launch_argv(*, config_cli: str) -> list[str]:
    """
    Build argv prefix before the ``agent`` subcommand: either a single PATH shim or
    ``[Cursor.exe, cli.js]`` for a direct Electron+Node launch on Windows.
    """
    raw = (os.environ.get(_ENV_CLI_OVERRIDE) or "").strip() or (
        config_cli.strip() or "cursor"
    )
    p = Path(raw)
    if p.is_file():
        if p.suffix.lower() == ".exe" and p.name.lower() == "cursor.exe":
            js = p.parent / "resources" / "app" / "out" / "cli.js"
            if js.is_file():
                return [str(p.resolve()), str(js)]
        pair = _electron_pair_from_app_bin_entry(p)
        if pair is not None:
            return pair
        if p.name == "cli.js":
            try:
                root = p.resolve().parents[3]
            except IndexError:
                root = None
            if root is not None:
                pair = _electron_pair_from_root(root)
                if pair is not None:
                    return pair
        return [str(p.resolve())]

    found = shutil.which(raw)
    if found:
        wp = Path(found)
        pair = _electron_pair_from_app_bin_entry(wp)
        if pair is not None:
            return pair
        return [found]

    if os.name == "nt":
        pair = _scan_windows_programs_cursor()
        if pair is not None:
            return pair
    if sys.platform == "darwin":
        pair = _scan_macos_app_bundle()
        if pair is not None:
            return pair

    return []


def _is_electron_cli_prefix(prefix: list[str]) -> bool:
    if len(prefix) != 2:
        return False
    js = str(prefix[1]).lower().replace("\\", "/")
    return js.endswith("/out/cli.js")


def _subprocess_env_for_prefix(prefix: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    if _is_electron_cli_prefix(prefix):
        env["ELECTRON_RUN_AS_NODE"] = "1"
        env["VSCODE_DEV"] = ""
    return env


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "").strip()
        text = str(m.get("content", "")).strip()
        if not text:
            continue
        if role == "system":
            parts.append(f"System instructions:\n{text}")
        elif role == "user":
            parts.append(f"User message:\n{text}")
        elif role == "assistant":
            parts.append(f"Prior assistant:\n{text}")
        else:
            parts.append(f"{role}:\n{text}")
    return "\n\n".join(parts).strip() or "."


def run_cursor_agent_auto_print(
    messages: list[dict[str, str]],
    *,
    workspace: str,
    config_cli: str,
    extra_args: list[str] | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> str:
    """
    Run ``cursor … agent --print --force --model auto --workspace <dir> <prompt>``.
    Model is fixed to ``auto`` (FAMOSE does not use ``llm_model`` for this provider).
    """
    prefix = resolve_launch_argv(config_cli=config_cli)
    if not prefix:
        raise RuntimeError(
            "Cursor CLI not found. In Cursor: Command Palette → "
            "\"Install 'cursor' command in PATH\" (restart the terminal), or set "
            f"{_ENV_CLI_OVERRIDE} to the path of Cursor.exe (Windows), e.g. "
            r"%LOCALAPPDATA%\Programs\cursor\Cursor.exe"
        )

    prompt = _messages_to_prompt(messages)
    argv: list[str] = [
        *prefix,
        "agent",
        "--print",
        "--force",
        "--model",
        "auto",
        "--workspace",
        workspace,
        *list(extra_args or []),
        prompt,
    ]
    env = _subprocess_env_for_prefix(prefix)

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Cursor CLI failed to start ({argv[0]!r}). "
            f"Install Cursor or set {_ENV_CLI_OVERRIDE} to Cursor.exe."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Cursor CLI timed out after {timeout_sec}s (famose llm_cursor_timeout_sec)."
        ) from e

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        tail = (err or out)[-4000:]
        raise RuntimeError(
            f"Cursor CLI exited with code {proc.returncode}. Last output:\n{tail}"
        )

    if not out:
        if err and "{" in err:
            out = err.strip()
        elif err:
            raise RuntimeError(
                "Cursor CLI returned empty stdout. If stderr mentions Electron/Chromium "
                "unknown options, the process was not in Node CLI mode (need Cursor.exe + "
                "cli.js and ELECTRON_RUN_AS_NODE=1 — e.g. avoid running the extensionless "
                "bin/cursor shim alone on Windows). Stderr:\n"
                + err[-4000:]
            )
        raise RuntimeError("Cursor CLI returned empty stdout and stderr.")

    return out
