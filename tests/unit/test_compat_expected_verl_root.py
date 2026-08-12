# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from pathlib import Path

import pytest

from verl_speco.integration import compat


def test_expected_verl_root_accepts_import_below_root(tmp_path, monkeypatch) -> None:
    expected = tmp_path / "verl-checkout"
    actual = expected / "verl" / "__init__.py"
    monkeypatch.setenv(compat.EXPECTED_VERL_ROOT_ENV, str(expected))
    monkeypatch.setattr(compat, "_imported_verl_path", lambda: actual.resolve())

    compat._check_expected_verl_root()


def test_expected_verl_root_rejects_different_checkout(tmp_path, monkeypatch) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "container" / "verl" / "__init__.py"
    monkeypatch.setenv(compat.EXPECTED_VERL_ROOT_ENV, str(expected))
    monkeypatch.setattr(compat, "_imported_verl_path", lambda: Path(actual).resolve())

    with pytest.raises(RuntimeError, match="different verl checkouts"):
        compat._check_expected_verl_root()


def test_imported_checkout_commit_uses_git_root_of_imported_module(
    tmp_path, monkeypatch
) -> None:
    checkout = tmp_path / "verl-checkout"
    imported = checkout / "verl" / "__init__.py"
    (checkout / ".git").mkdir(parents=True)
    imported.parent.mkdir(parents=True)
    imported.write_text("", encoding="utf-8")
    monkeypatch.setattr(compat, "_imported_verl_path", lambda: imported.resolve())
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"stdout": "abc123\n"})()

    monkeypatch.setattr(compat.subprocess, "run", fake_run)

    assert compat._read_imported_checkout_commit() == "abc123"
    assert calls[0][0] == ["git", "-C", str(checkout.resolve()), "rev-parse", "HEAD"]
