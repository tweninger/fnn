"""Create an isolated pinned DyGLib checkout and install the FNN bridge."""
import argparse
from pathlib import Path
import subprocess

COMMIT = "3aacc36b94b8d2d8293d70a74fdf6d39089b4163"


def install(target, source):
    target = Path(target).resolve()
    if target.exists():
        raise SystemExit(f"Refusing to overwrite {target}; choose a fresh --target")
    subprocess.run(["git", "clone", source, str(target)], check=True)
    subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=target, check=True)
    for filename in ("train_link_prediction.py", "evaluate_link_prediction.py",
                     "evaluate_models_utils.py", "utils/EarlyStopping.py", "utils/load_configs.py"):
        path = target / filename
        text = path.read_text()
        text = text.replace("['JODIE', 'DyRep', 'TGN']", "['JODIE', 'DyRep', 'TGN', 'FNN']")
        text = text.replace("from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts",
                            "from models.MemoryModel import compute_src_dst_node_time_shifts\nfrom experiments.dyglib.fnn import MemoryModel")
        if filename == "utils/load_configs.py":
            text = text.replace("choices=['JODIE',", "choices=['FNN', 'JODIE',")
            text = text.replace("choices=['wikipedia',", "choices=['college_msg', 'email_eu_core', 'sociopatterns', 'wikipedia',")
        if filename == "utils/EarlyStopping.py":
            # These are locally generated checkpoints, not untrusted downloads.
            text = text.replace("map_location=map_location)", "map_location=map_location, weights_only=False)")
        path.write_text(text)
    path = target / "utils/DataLoader.py"
    text = path.read_text().replace("random.sample(test_node_set,", "random.sample(sorted(test_node_set),")
    path.write_text(text)  # Python 3.11 no longer accepts sets in random.sample.
    (target / "FNN_PATCH.txt").write_text(f"Upstream {COMMIT}\nFNN bridge: experiments/dyglib/fnn.py in the parent project.\n")
    print(f"Installed {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="derived/dyglib")
    parser.add_argument("--source", default="https://github.com/yule-BUAA/DyGLib.git")
    args = parser.parse_args()
    install(args.target, args.source)
