import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/6_real_benchmark_panel.sh"


def test_panel_schedules_individual_models_and_limits_concurrency(tmp_path):
    worker = tmp_path / "worker"
    worker.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys, time
from pathlib import Path
args = sys.argv
def arg(key): return args[args.index(key) + 1]
assert arg('--max-runs') == '1'
assert arg('--epochs') == '58'
assert os.environ['OMP_NUM_THREADS'] == '4'
assert os.environ['MKL_NUM_THREADS'] == '4'
assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
start = time.monotonic()
time.sleep(0.3)
Path(arg('--save-jsonl')).write_text(json.dumps(dict(
    offset=int(arg('--run-offset')), start=start, end=time.monotonic())))
sys.exit(7 if os.environ.get('FAIL_OFFSET') == arg('--run-offset') else 0)
''')
    worker.chmod(0o755)
    env = dict(os.environ, PYTHON=str(worker), GPU_IDS="-1", CPU_THREADS="4",
               MAX_PARALLEL="3", BENCHMARKS="college_msg", SEED="0", MAX_RUNS="5",
               RUN_OFFSET="1", TOPOLOGY_EPOCHS="10", PHYSICAL_EPOCHS="2", CYCLES="3",
               EVAL_EVERY="58", RESULTS_DIR=str(tmp_path / "results"),
               LOG_DIR=str(tmp_path / "logs"))
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(p.read_text()) for p in (tmp_path / "results").glob("*.jsonl")]
    assert sorted(r["offset"] for r in rows) == [1, 2, 3, 4, 5]
    maximum = max(sum(r["start"] <= t["start"] < r["end"] for r in rows) for t in rows)
    assert 2 <= maximum <= 3
    assert len(list((tmp_path / "logs").glob("*.log"))) == 5
    # Existing per-model files skip cleanly, even when every job exits immediately.
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Skipping existing") == 5
    env.update(RESULTS_DIR=str(tmp_path / "failed_results"), FAIL_OFFSET="2")
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAILED:" in result.stderr
    assert len(list((tmp_path / "failed_results").glob("*.jsonl"))) == 5
