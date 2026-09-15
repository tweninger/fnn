"""Create an isolated pinned DyGLib checkout and install the FNN bridge."""
import argparse
from pathlib import Path
import subprocess

COMMIT = "3aacc36b94b8d2d8293d70a74fdf6d39089b4163"


def update(target):
    """Add channel options to an existing bridge without replacing its data."""
    target = Path(target).resolve()
    marker = target / "FNN_PATCH.txt"
    if not marker.exists() or COMMIT not in marker.read_text():
        raise SystemExit("--update requires a pinned FNN bridge checkout")
    edits = {}
    for filename in ("utils/load_configs.py", "train_link_prediction.py", "evaluate_link_prediction.py"):
        path = target / filename
        text = path.read_text()
        if "fnn_state_dim" in text:
            continue
        if filename == "utils/load_configs.py":
            anchor = "    parser.add_argument('--batch_size'"
            if anchor not in text:
                raise SystemExit(f"Cannot locate CLI anchor in {path}")
            text = text.replace(anchor,
                "    parser.add_argument('--fnn_state_dim', type=int, default=1, help='FNN field channels (positive integer)')\n" + anchor)
        else:
            anchor = "dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)"
            name = "f'{args.model_name}_seed{args.seed}'"
            if text.count(anchor) != 1 or name not in text:
                raise SystemExit(f"Cannot locate FNN construction/name anchors in {path}")
            text = text.replace(anchor, "dst_node_std_time_shift=dst_node_std_time_shift, device=args.device, fnn_state_dim=args.fnn_state_dim)")
            text = text.replace(name, name + " + (f'_dim{args.fnn_state_dim}' if args.model_name == 'FNN' and args.fnn_state_dim != 1 else '')")
        edits[path] = text
    path = target / "evaluate_link_prediction.py"
    text = edits.get(path, path.read_text())
    anchor = "dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,"
    if anchor in text:
        text = text.replace(anchor,
            "dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, "
            "neighbor_sampler=(get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, "
            "time_scaling_factor=args.time_scaling_factor, seed=0) if args.model_name == 'FNN' else full_neighbor_sampler),")
    name = "f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'"
    text = text.replace(name, "f'{args.negative_sample_strategy}_negative_sampling_{args.load_model_name}'")
    text = text.replace(
        "            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'FNN']:\n                for node_id, node_raw_messages",
        "            if args.model_name == 'FNN':\n"
        "                pending = model[0].memory_bank.node_raw_messages\n"
        "                if pending is not None:\n"
        "                    model[0].memory_bank.node_raw_messages = tuple(x.to(args.device) for x in pending)\n"
        "            elif args.model_name in ['JODIE', 'DyRep', 'TGN']:\n                for node_id, node_raw_messages")
    edits[path] = text
    # Preflight all files before writing any changes; leave results/data intact.
    for path, text in edits.items():
        path.write_text(text)
    print(f"Updated channel options in {target}")


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
    update(target)
    print(f"Installed {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="derived/dyglib")
    parser.add_argument("--source", default="https://github.com/yule-BUAA/DyGLib.git")
    parser.add_argument("--update", action="store_true", help="Patch an existing FNN checkout in place; preserve data and results")
    args = parser.parse_args()
    if args.update:
        update(args.target)
    else:
        install(args.target, args.source)
