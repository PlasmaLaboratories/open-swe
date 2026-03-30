from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _load_daytona_module(monkeypatch: pytest.MonkeyPatch):
    fake_daytona = types.ModuleType("daytona")

    class FakeCreateSandboxFromSnapshotParams:
        def __init__(self, snapshot: str) -> None:
            self.snapshot = snapshot

    class FakeDaytonaConfig:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    class FakeDaytona:
        def __init__(self, config: FakeDaytonaConfig) -> None:
            self.config = config

    fake_daytona.CreateSandboxFromSnapshotParams = FakeCreateSandboxFromSnapshotParams
    fake_daytona.Daytona = FakeDaytona
    fake_daytona.DaytonaConfig = FakeDaytonaConfig
    monkeypatch.setitem(sys.modules, "daytona", fake_daytona)

    fake_langchain_daytona = types.ModuleType("langchain_daytona")

    class FakeDaytonaSandbox:
        def __init__(self, sandbox) -> None:
            self.sandbox = sandbox

    fake_langchain_daytona.DaytonaSandbox = FakeDaytonaSandbox
    monkeypatch.setitem(sys.modules, "langchain_daytona", fake_langchain_daytona)

    module_path = Path(__file__).resolve().parents[1] / "agent" / "integrations" / "daytona.py"
    module = types.ModuleType("test_daytona_module")
    module.__file__ = str(module_path)
    source = module_path.read_text()
    code = compile(f"from __future__ import annotations\n{source}", str(module_path), "exec")
    exec(code, module.__dict__)
    return module


def test_get_daytona_sandbox_params_uses_default_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DAYTONA_SNAPSHOT", raising=False)
    daytona_module = _load_daytona_module(monkeypatch)

    params = daytona_module._get_daytona_sandbox_params()

    assert params.snapshot == daytona_module.DEFAULT_DAYTONA_SNAPSHOT


def test_get_daytona_sandbox_params_uses_env_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAYTONA_SNAPSHOT", "my-org/custom-sandbox:latest")
    daytona_module = _load_daytona_module(monkeypatch)

    params = daytona_module._get_daytona_sandbox_params()

    assert params.snapshot == "my-org/custom-sandbox:latest"
