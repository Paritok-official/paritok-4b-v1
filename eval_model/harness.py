"""Run the official SWE-bench evaluation harness (Docker) on a predictions file
and read back how many instances resolved.

Returns (resolved, completed, total):
  * resolved  — patch applied AND all tests pass (the numerator).
  * completed — resolved + unresolved (patches that applied and were graded).
  * total     — every instance submitted (the resolve-rate denominator).

The metric is resolved / total.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys


# swebench imports the Unix-only `resource` module, so it can't run in Windows python.
# On Windows we drive it through WSL instead — the same "call local Docker from WSL"
# path you'd run by hand — resolving (once) a WSL python that can import swebench.
_WSL_PY: list = []          # [] = not probed; [None] = none found; [str] = the python


def _win_to_wsl_path(p: str) -> str:
    """C:\\a\\b  ->  /mnt/c/a/b  (absolute)."""
    p = os.path.abspath(p)
    drive, rest = os.path.splitdrive(p)
    return ("/mnt/" + drive[0].lower() + rest.replace("\\", "/")) if drive else p.replace("\\", "/")


def _wsl_python() -> str | None:
    """A WSL python that can `import swebench`, or None. Tries $PARITOK_EVAL_WSL_PYTHON,
    then a conventional venv, then python3."""
    if _WSL_PY:
        return _WSL_PY[0]
    override = os.environ.get("PARITOK_EVAL_WSL_PYTHON", "").strip()
    for py in ([override] if override else []) + ["~/swebench-env/bin/python", "python3"]:
        try:
            r = subprocess.run(["wsl", "bash", "-lc", f"{py} -c 'import swebench.harness.run_evaluation'"],
                               capture_output=True, timeout=120)
        except Exception:
            _WSL_PY.append(None)
            return None
        if r.returncode == 0:
            _WSL_PY.append(py)
            return py
    _WSL_PY.append(None)
    return None


def available() -> tuple[bool, str]:
    """Whether the SWE-bench harness can run from here. On Windows it runs via WSL (it
    imports the Unix-only `resource` module and can't run in Windows python). Returns
    (ok, reason); reason is user-facing when not ok."""
    if platform.system() == "Windows":
        if _wsl_python():
            return True, ""      # driven through WSL below
        return (False, "on Windows the harness runs through WSL, but no WSL python with swebench "
                       "was found. In WSL: `pip install swebench` (or point PARITOK_EVAL_WSL_PYTHON "
                       "at its python), with Docker Desktop's WSL integration enabled.")
    try:
        r = subprocess.run([sys.executable, "-c", "import swebench.harness.run_evaluation"],
                           capture_output=True, timeout=120)
    except Exception as e:  # noqa: BLE001 — any launch failure means "can't score here"
        return False, f"could not launch the swebench harness ({type(e).__name__})."
    if r.returncode == 0:
        return True, ""
    err = (r.stderr or b"").decode("utf-8", "replace").strip()
    if "No module named 'swebench'" in err:
        return False, "swebench is not installed here (pip install swebench)."
    return False, f"the swebench harness failed to import: {err.splitlines()[-1] if err else 'unknown'}"


def write_predictions(preds: dict[str, str], path: str, model_name: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for iid, patch in preds.items():
            f.write(json.dumps({
                "instance_id": iid,
                "model_name_or_path": model_name,
                "model_patch": patch or "",
            }, ensure_ascii=False) + "\n")


def run(preds: dict[str, str], instance_ids: list[str], *, run_id: str,
        workdir: str, max_workers: int = 6) -> tuple[int, int, int]:
    # Absolute so the paths still resolve when the harness runs with cwd=workdir.
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    model_name = f"paritok-{run_id}"
    pred_path = os.path.join(workdir, f"predictions_{run_id}.jsonl")
    write_predictions(preds, pred_path, model_name)

    if platform.system() == "Windows":
        # Route through WSL: swebench (in WSL) talks to Docker Desktop; it writes the
        # report into cwd, which via /mnt is the same Windows dir we read back below.
        py = _wsl_python()
        if not py:
            raise RuntimeError("no WSL python with swebench found (set PARITOK_EVAL_WSL_PYTHON)")
        inner = (
            f"cd '{_win_to_wsl_path(workdir)}' && {py} -m swebench.harness.run_evaluation "
            f"--dataset_name princeton-nlp/SWE-bench_Lite "
            f"--predictions_path '{_win_to_wsl_path(pred_path)}' "
            f"--instance_ids {' '.join(instance_ids)} "
            f"--max_workers {max_workers} --cache_level instance --clean False --run_id {run_id}"
        )
        subprocess.run(["wsl", "bash", "-lc", inner], check=True)
    else:
        subprocess.run(
            [sys.executable, "-m", "swebench.harness.run_evaluation",
             "--dataset_name", "princeton-nlp/SWE-bench_Lite",
             "--predictions_path", pred_path,
             "--instance_ids", *instance_ids,
             "--max_workers", str(max_workers),
             "--cache_level", "instance",
             "--clean", "False",
             "--run_id", run_id],
            cwd=workdir, check=True,
        )

    # The harness writes exactly `<model_name>.<run_id>.json` into cwd. Read that
    # precise path — a loose glob would also catch our own predictions dumps.
    report_path = os.path.join(workdir, f"{model_name}.{run_id}.json")
    if not os.path.exists(report_path):
        matches = [p for p in glob.glob(os.path.join(workdir, f"*{run_id}*.json"))
                   if not p.endswith("_preds.json")]
        if not matches:
            raise RuntimeError(f"no SWE-bench report produced for run_id={run_id}")
        report_path = matches[0]
    report = json.load(open(report_path, encoding="utf-8"))
    if "resolved_instances" not in report and "completed_instances" not in report:
        raise RuntimeError(f"unexpected report shape at {report_path}: {list(report)[:6]}")
    resolved = int(report.get("resolved_instances", len(report.get("resolved_ids", []))))
    completed = int(report.get("completed_instances", resolved))
    total = int(report.get("total_instances", 0)) or len(instance_ids)
    return resolved, completed, total
