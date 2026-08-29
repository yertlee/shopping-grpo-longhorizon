"""smoke shell 的故障码传播测试：python 非 0 时整体必须非 0，日志仍被保存。

仅在存在 bash 的环境运行（远端 Linux / 本地 Git Bash）。
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SMOKE_SNIPPET = """
set -o pipefail
"$PYTHON_BIN" {exit_expr} 2>&1 | tee "$LOG_FILE"
status=${{PIPESTATUS[0]}}
test "$status" -eq 0
"""


def _find_bash():
    """优先真实 Git Bash；System32 的 WSL stub 不能执行本测试的脚本。"""
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        "/usr/bin/bash",
        "/bin/bash",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


BASH = _find_bash()


def _run_snippet(exit_code: int, log_file: Path):
    script = SMOKE_SNIPPET.format(exit_expr=f'-c "import sys; sys.exit({exit_code})"')
    env = dict(os.environ)
    env["PYTHON_BIN"] = Path(sys.executable).as_posix()
    env["LOG_FILE"] = log_file.as_posix()
    env["LC_ALL"] = "C"
    result = subprocess.run(
        [BASH, "-c", script], capture_output=True, env=env, timeout=60
    )
    result.stdout_text = (result.stdout or b"").decode("utf-8", errors="replace")
    result.stderr_text = (result.stderr or b"").decode("utf-8", errors="replace")
    return result


@unittest.skipIf(not BASH, "没有可用的 Git Bash")
class SmokeShellFailurePropagationTest(unittest.TestCase):
    def test_python_failure_propagates_and_log_is_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "smoke.log"
            result = _run_snippet(3, log_file)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(log_file.is_file(), "失败时日志仍必须被保存")

    def test_python_success_yields_zero_and_log_is_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "smoke.log"
            result = _run_snippet(0, log_file)
            self.assertEqual(result.returncode, 0, result.stderr_text)
            self.assertTrue(log_file.is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
