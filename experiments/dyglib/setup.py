"""Create an isolated pinned DyGLib checkout and install the FNN bridge."""
import argparse
from pathlib import Path
import subprocess

COMMIT = "3aacc36b94b8d2d8293d70a74fdf6d39089b4163"

RUNNER_PATCH = Path(__file__).with_name("runner.patch")


def apply_runner_patch(target):
    """Replay the reviewed, live-runner delta exactly once."""
    if not RUNNER_PATCH.is_file():
        raise SystemExit(f"Missing runner patch: {RUNNER_PATCH}")
    applicable = subprocess.run(
        ["git", "apply", "--check", str(RUNNER_PATCH)],
        cwd=target, capture_output=True, text=True,
    )
    if applicable.returncode == 0:
        subprocess.run(["git", "apply", str(RUNNER_PATCH)], cwd=target, check=True)
        return
    already_applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(RUNNER_PATCH)],
        cwd=target, capture_output=True, text=True,
    )
    if already_applied.returncode != 0:
        raise SystemExit(
            f"Runner patch does not match {target}; inspect local edits before updating.\n"
            f"{applicable.stderr}"
        )


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
    for filename in ("utils/load_configs.py", "train_link_prediction.py", "evaluate_link_prediction.py"):
        path = target / filename
        text = edits.get(path, path.read_text())
        if "fnn_spectral_rank" not in text:
            if filename == "utils/load_configs.py":
                anchor = "    parser.add_argument('--batch_size'"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate spectral CLI anchor in {path}")
                text = text.replace(anchor, "    parser.add_argument('--fnn_spectral_rank', type=int, default=0)\n" + anchor)
            else:
                anchor = "fnn_state_dim=args.fnn_state_dim)"
                name = "f'{args.model_name}_seed{args.seed}'"
                if anchor not in text or name not in text:
                    raise SystemExit(f"Cannot locate spectral construction anchors in {path}")
                text = text.replace(anchor, "fnn_state_dim=args.fnn_state_dim, fnn_spectral_rank=args.fnn_spectral_rank)")
                text = text.replace(name, name + " + (f'_spectral{args.fnn_spectral_rank}' if args.model_name == 'FNN' and args.fnn_spectral_rank > 0 else '')")
        text = "\n".join(line for line in text.split("\n")
                         if "parser.add_argument('--fnn_sparse_propagation'" not in line)
        text = text.replace(", fnn_sparse_propagation=args.fnn_sparse_propagation", "")
        text = text.replace(" + ('_sparseprop' if args.model_name == 'FNN' and args.fnn_sparse_propagation else '')", "")
        if "fnn_propagate" not in text:
            if filename == "utils/load_configs.py":
                anchor = "    parser.add_argument('--batch_size'"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate sparse CLI anchor in {path}")
                text = text.replace(anchor, "    parser.add_argument('--fnn_propagate', '-fnn_propagate', type=int, default=0, help='FNN input propagation hops; 0 disables')\n" + anchor)
            else:
                anchor = "fnn_spectral_rank=args.fnn_spectral_rank)"
                name = "f'{args.model_name}_seed{args.seed}'"
                if anchor not in text or name not in text:
                    raise SystemExit(f"Cannot locate sparse construction anchors in {path}")
                text = text.replace(anchor, "fnn_spectral_rank=args.fnn_spectral_rank, fnn_propagate=args.fnn_propagate)")
                text = text.replace(name, name + " + (f'_propagate{args.fnn_propagate}' if args.model_name == 'FNN' and args.fnn_propagate > 0 else '')")
        if "fnn_clock" not in text:
            if filename == "utils/load_configs.py":
                anchor = "    parser.add_argument('--batch_size'"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate clock CLI anchor in {path}")
                text = text.replace(
                    anchor,
                    "    parser.add_argument('--fnn_clock', choices=['event', 'event_exact', 'event_exact_unit', 'normalized', 'normalized_exact'], default='event', "
                    "help='FNN event clock, batch-min normalized clock, or exact timestamp-grouped normalized clock')\n"
                    "    parser.add_argument('--fnn_time_cap', type=float, default=10.0, "
                    "help='maximum normalized elapsed gap per FNN update')\n" + anchor)
            else:
                anchor = "fnn_propagate=args.fnn_propagate)"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate clock construction anchors in {path}")
                text = text.replace(anchor, "fnn_propagate=args.fnn_propagate, fnn_clock=args.fnn_clock, fnn_time_cap=args.fnn_time_cap)")
        text = text.replace("choices=['event', 'normalized'], default='event'",
                            "choices=['event', 'event_exact', 'event_exact_unit', 'normalized', 'normalized_exact'], default='event'")
        if "fnn_ablation" not in text:
            if filename == "utils/load_configs.py":
                anchor = "    parser.add_argument('--batch_size'"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate ablation CLI anchor in {path}")
                text = text.replace(
                    anchor,
                    "    parser.add_argument('--fnn_ablation', choices=['none', 'fixed_topology', 'fixed_physical'], "
                    "default='none', help='freeze the training-observed topology or physical coefficients')\n" + anchor)
            else:
                anchor = "fnn_time_cap=args.fnn_time_cap)"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate ablation construction anchor in {path}")
                text = text.replace(anchor, "fnn_time_cap=args.fnn_time_cap, fnn_ablation=args.fnn_ablation)")
        text = text.replace(
            "choices=['none', 'fixed_topology', 'fixed_physical']",
            "choices=['none', 'fixed_topology', 'fixed_gates', 'fixed_physical', 'fixed_gates_physical']",
        )
        text = text.replace(
            "choices=['none', 'fixed_topology', 'fixed_gates', 'fixed_physical']",
            "choices=['none', 'fixed_topology', 'fixed_gates', 'fixed_physical', 'fixed_gates_physical']",
        )
        text = text.replace(
            "args.fnn_ablation == 'fixed_physical'",
            "args.fnn_ablation in ['fixed_physical', 'fixed_gates_physical']",
        )
        text = text.replace(
            "args.fnn_ablation == 'fixed_gates'",
            "args.fnn_ablation in ['fixed_gates', 'fixed_gates_physical']",
        )
        if "fnn_fixed_gate_value" not in text:
            if filename == "utils/load_configs.py":
                anchor = "    parser.add_argument('--batch_size'"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate fixed-gate CLI anchor in {path}")
                text = text.replace(anchor,
                    "    parser.add_argument('--fnn_fixed_gate_value', type=float, default=0.5, "
                    "help='constant sigmoid gate for fixed_gates ablation (strictly between 0 and 1)')\n" + anchor)
            else:
                anchor = "fnn_ablation=args.fnn_ablation)"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate fixed-gate construction anchor in {path}")
                text = text.replace(anchor, "fnn_ablation=args.fnn_ablation, "
                                    "fnn_fixed_gate_value=args.fnn_fixed_gate_value)")
        if "fnn_gamma_init" not in text:
            if filename == "utils/load_configs.py":
                anchor = "    parser.add_argument('--batch_size'"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate physical-init CLI anchor in {path}")
                options = (
                    "    parser.add_argument('--fnn_gamma_init', type=float, default=0.15, help='positive initial FNN damping')\n"
                    "    parser.add_argument('--fnn_omega_init', type=float, default=0.8, help='positive initial FNN frequency')\n"
                    "    parser.add_argument('--fnn_input_scale_init', type=float, default=1.0, help='positive initial FNN input scale')\n"
                )
                text = text.replace(anchor, options + anchor)
            else:
                anchor = "fnn_fixed_gate_value=args.fnn_fixed_gate_value)"
                if anchor not in text:
                    raise SystemExit(f"Cannot locate physical-init construction anchor in {path}")
                text = text.replace(
                    anchor,
                    "fnn_fixed_gate_value=args.fnn_fixed_gate_value, fnn_gamma_init=args.fnn_gamma_init, "
                    "fnn_omega_init=args.fnn_omega_init, fnn_input_scale_init=args.fnn_input_scale_init)",
                )
        if filename == "utils/load_configs.py" and "fnn_physical_weight_decay" not in text:
            anchor = "    parser.add_argument('--batch_size'"
            if anchor not in text:
                raise SystemExit(f"Cannot locate physical-decay CLI anchor in {path}")
            text = text.replace(
                anchor,
                "    parser.add_argument('--fnn_physical_weight_decay', type=float, default=None, "
                "help='FNN physical-parameter decay; defaults to --weight_decay')\n" + anchor,
            )
        if filename == "utils/load_configs.py" and "start_seed" not in text:
            anchor = "    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')"
            if anchor not in text:
                raise SystemExit(f"Cannot locate run-count CLI anchor in {path}")
            text = text.replace(
                anchor,
                anchor + "\n    parser.add_argument('--start_seed', type=int, default=0, "
                "help='first random seed/run index')",
            )
        if filename == "utils/load_configs.py" and "run_tag" not in text:
            anchor = "    parser.add_argument('--start_seed', type=int, default=0, help='first random seed/run index')"
            if anchor not in text:
                raise SystemExit(f"Cannot locate start-seed CLI anchor in {path}")
            text = text.replace(
                anchor,
                anchor + "\n    parser.add_argument('--run_tag', type=str, default='', "
                "help='suffix for isolating run artifacts')",
            )
        if filename == "train_link_prediction.py" and "args.start_seed" not in text:
            anchor = "    for run in range(args.num_runs):"
            if anchor not in text:
                raise SystemExit(f"Cannot locate training run-loop anchor in {path}")
            text = text.replace(
                anchor,
                "    for run in range(args.start_seed, args.start_seed + args.num_runs):",
            )
        if filename != "utils/load_configs.py" and "_lr{args.learning_rate}" not in text:
            name = "f'{args.model_name}_seed{args.seed}'"
            if name not in text:
                raise SystemExit(f"Cannot locate hyperparameter naming anchor in {path}")
            text = text.replace(name, name + " + f'_lr{args.learning_rate}_wd{args.weight_decay}_bs{args.batch_size}'")
        if filename != "utils/load_configs.py" and "_ntime_cap" not in text:
            assignment = "args.save_model_name =" if filename == "train_link_prediction.py" else "args.load_model_name ="
            lines = text.split("\n")
            matches = [index for index, line in enumerate(lines) if assignment in line]
            if not matches:
                matches = [index for index, line in enumerate(lines)
                           if "f'{args.model_name}_seed{args.seed}'" in line]
            if len(matches) != 1:
                raise SystemExit(f"Cannot locate clock naming anchor in {path}")
            suffix = " + (f'_ntime_cap{args.fnn_time_cap:g}' if args.model_name == 'FNN' and args.fnn_clock == 'normalized' else '')"
            suffix += " + ('_etime_exact' if args.model_name == 'FNN' and args.fnn_clock == 'event_exact' else '')"
            suffix += " + ('_etime_exact_unit' if args.model_name == 'FNN' and args.fnn_clock == 'event_exact_unit' else '')"
            suffix += " + (f'_ntime_exact_cap{args.fnn_time_cap:g}' if args.model_name == 'FNN' and args.fnn_clock == 'normalized_exact' else '')"
            lines[matches[0]] += suffix
            text = "\n".join(lines)
        elif filename != "utils/load_configs.py" and "_ntime_exact_cap" not in text:
            assignment = "args.save_model_name =" if filename == "train_link_prediction.py" else "args.load_model_name ="
            lines = text.split("\n")
            matches = [index for index, line in enumerate(lines) if assignment in line]
            if len(matches) != 1:
                raise SystemExit(f"Cannot locate exact-clock naming anchor in {path}")
            lines[matches[0]] += " + ('_etime_exact' if args.model_name == 'FNN' and args.fnn_clock == 'event_exact' else '')"
            lines[matches[0]] += " + ('_etime_exact_unit' if args.model_name == 'FNN' and args.fnn_clock == 'event_exact_unit' else '')"
            lines[matches[0]] += " + (f'_ntime_exact_cap{args.fnn_time_cap:g}' if args.model_name == 'FNN' and args.fnn_clock == 'normalized_exact' else '')"
            text = "\n".join(lines)
        if filename != "utils/load_configs.py" and "'_physwd'" not in text:
            assignment = "args.save_model_name =" if filename == "train_link_prediction.py" else "args.load_model_name ="
            lines = text.split("\n")
            matches = [index for index, line in enumerate(lines) if assignment in line]
            if not matches:
                matches = [index for index, line in enumerate(lines)
                           if "f'{args.model_name}_seed{args.seed}'" in line]
            if len(matches) != 1:
                raise SystemExit(f"Cannot locate physical-decay naming anchor in {path}")
            lines[matches[0]] += " + ('_physwd' if args.model_name == 'FNN' and args.weight_decay > 0 else '')"
            text = "\n".join(lines)
        if filename != "utils/load_configs.py" and "'_fixtop'" not in text:
            assignment = "args.save_model_name =" if filename == "train_link_prediction.py" else "args.load_model_name ="
            lines = text.split("\n")
            matches = [index for index, line in enumerate(lines) if assignment in line]
            if not matches:
                matches = [index for index, line in enumerate(lines)
                           if "f'{args.model_name}_seed{args.seed}'" in line]
            if len(matches) != 1:
                raise SystemExit(f"Cannot locate ablation naming anchor in {path}")
            lines[matches[0]] += " + ('_fixtop' if args.model_name == 'FNN' and args.fnn_ablation == 'fixed_topology' else '') + ('_fixphys' if args.model_name == 'FNN' and args.fnn_ablation in ['fixed_physical', 'fixed_gates_physical'] else '')"
            text = "\n".join(lines)
        if filename != "utils/load_configs.py" and "_fixgate" not in text:
            assignment = "args.save_model_name =" if filename == "train_link_prediction.py" else "args.load_model_name ="
            lines = text.split("\n")
            matches = [index for index, line in enumerate(lines) if assignment in line]
            if not matches:
                matches = [index for index, line in enumerate(lines)
                           if "f'{args.model_name}_seed{args.seed}'" in line]
            if len(matches) != 1:
                raise SystemExit(f"Cannot locate fixed-gate naming anchor in {path}")
            lines[matches[0]] += " + (f'_fixgate{args.fnn_fixed_gate_value:g}' if args.model_name == 'FNN' and args.fnn_ablation in ['fixed_gates', 'fixed_gates_physical'] else '')"
            text = "\n".join(lines)
        if filename != "utils/load_configs.py" and "args.run_tag" not in text:
            assignment = "args.save_model_name =" if filename == "train_link_prediction.py" else "args.load_model_name ="
            lines = text.split("\n")
            matches = [index for index, line in enumerate(lines) if assignment in line]
            if not matches:
                matches = [index for index, line in enumerate(lines)
                           if "f'{args.model_name}_seed{args.seed}'" in line]
            if len(matches) != 1:
                raise SystemExit(f"Cannot locate run-tag naming anchor in {path}")
            lines[matches[0]] += " + (f'_{args.run_tag}' if args.run_tag else '')"
            text = "\n".join(lines)
        if filename == "train_link_prediction.py" and "FNN physical parameters:" not in text:
            anchor = "            # perform testing once after test_interval_epochs"
            if anchor not in text:
                raise SystemExit(f"Cannot locate periodic-test anchor in {path}")
            snapshot = (
                "            # Record the physical state at the same epochs used for periodic testing.\n"
                "            if args.model_name == 'FNN' and (epoch + 1) % args.test_interval_epochs == 0:\n"
                "                physical = model[0].field.physical_parameters()\n"
                "                snapshot = {\n"
                "                    name: physical[name].detach().cpu().tolist()\n"
                "                    for name in ('gamma', 'omega', 'input_force_scale')\n"
                "                    if name in physical\n"
                "                }\n"
                "                logger.info(f'FNN physical parameters: {json.dumps(snapshot, separators=(\",\", \":\"))}')\n\n"
            )
            text = text.replace(anchor, snapshot + anchor)
        if filename == "train_link_prediction.py" and "transformed_decay_parameters" not in text:
            optimizer_anchor = "        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)"
            if optimizer_anchor not in text:
                raise SystemExit(f"Cannot locate optimizer anchor in {path}")
            text = text.replace(
                optimizer_anchor,
                optimizer_anchor + "\n"
                "        if args.model_name == 'FNN':\n"
                "            transformed_decay_parameters = model[0].transformed_decay_parameters()\n"
                "            transformed_decay_ids = {id(parameter) for parameter in transformed_decay_parameters}\n"
                "            optimizer.param_groups[0]['params'] = [\n"
                "                parameter for parameter in optimizer.param_groups[0]['params']\n"
                "                if id(parameter) not in transformed_decay_ids\n"
                "            ]\n"
                "            if transformed_decay_parameters:\n"
                "                optimizer.add_param_group({'params': transformed_decay_parameters, 'weight_decay': 0.0})"
            )
            loss_anchor = "                loss = loss_func(input=predicts, target=labels)"
            backward_anchor = "                loss.backward()"
            if loss_anchor not in text or backward_anchor not in text:
                raise SystemExit(f"Cannot locate loss anchors in {path}")
            text = text.replace(
                loss_anchor,
                loss_anchor + "\n"
                "                optimization_loss = (loss + model[0].transformed_weight_decay(physical_weight_decay)\n"
                "                                     if args.model_name == 'FNN' else loss)",
                1,
            ).replace(backward_anchor, "                optimization_loss.backward()", 1)
        if filename == "train_link_prediction.py" and "physical_weight_decay = " not in text:
            optimizer_anchor = "        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)"
            if optimizer_anchor not in text:
                raise SystemExit(f"Cannot locate selective physical-decay optimizer anchor in {path}")
            definition = (
                "        physical_weight_decay = (args.weight_decay if args.fnn_physical_weight_decay is None\n"
                "                                 else args.fnn_physical_weight_decay)\n"
                "        if physical_weight_decay < 0:\n"
                "            raise ValueError('fnn_physical_weight_decay must be nonnegative')\n"
            )
            text = text.replace(optimizer_anchor, definition + optimizer_anchor, 1)
            text = text.replace("if args.model_name == 'FNN' and args.weight_decay:",
                                "if args.model_name == 'FNN':", 1)
            text = text.replace("model[0].transformed_weight_decay(args.weight_decay)",
                                "model[0].transformed_weight_decay(physical_weight_decay)", 1)
        edits[path] = text
    # Anchor checks above precede writes; the runner patch checks its own context.
    for path, text in edits.items():
        path.write_text(text)
    apply_runner_patch(target)
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
