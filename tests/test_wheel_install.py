"""End-to-end test for the public CLI from a non-editable wheel install."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    version("build")
except PackageNotFoundError:
    _BUILD_FRONTEND_AVAILABLE = False
else:
    _BUILD_FRONTEND_AVAILABLE = True


@unittest.skipUnless(
    # A repository-level ``build/`` directory is a valid setuptools output
    # directory but not the PyPA ``build`` frontend.  Check package metadata so
    # it cannot make this optional test run and fail during collection.
    _BUILD_FRONTEND_AVAILABLE,
    "the wheel test requires the dev extra",
)
class WheelInstallTest(unittest.TestCase):
    def test_wheel_public_cli_runs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            dist = temporary / "dist"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    str(dist),
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=temporary,
            )
            wheel = next(dist.glob("shopping_grpo-*.whl"))
            environment = temporary / "venv"
            binary_directory = "Scripts" if os.name == "nt" else "bin"
            executable_suffix = ".exe" if os.name == "nt" else ""
            python = environment / binary_directory / f"python{executable_suffix}"
            cli = environment / binary_directory / f"shopping-grpo{executable_suffix}"
            uv = shutil.which("uv")
            if uv:
                subprocess.run(
                    [uv, "venv", str(environment)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [
                        uv,
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--no-deps",
                        str(wheel),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                venv.EnvBuilder(with_pip=True).create(environment)
                pip = environment / binary_directory / "pip"
                subprocess.run(
                    [str(pip), "install", "--no-deps", str(wheel)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            smoke = subprocess.run(
                [str(cli), "smoke", "--json"],
                cwd=temporary,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                len(json.loads(smoke.stdout)["checks"]),
                5,
            )

            trajectory = temporary / "non_blind_example.jsonl"
            trajectory.write_bytes(
                (root / "examples/trajectories.jsonl").read_bytes()
            )
            evaluated = subprocess.run(
                [str(cli), "evaluate", str(trajectory)],
                cwd=temporary,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(evaluated.stdout)["trajectory_count"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
