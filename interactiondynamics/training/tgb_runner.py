"""Opt-in event-time FNN experiment using the official TGB evaluation API.

No binned rollout or force-regression metric is reported for unit impulses.
"""
from __future__ import annotations

import json
import math
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from interactiondynamics.core.events import EventBatch
from interactiondynamics.models.fnn import FieldNeuralNetwork


def timestamp_groups(times, mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        raise ValueError("Empty TGB split")
    if np.any(np.diff(times[ids]) < 0):
        raise ValueError("TGB events must be chronologically ordered")
    return np.split(ids, np.flatnonzero(np.diff(times[ids]) != 0) + 1)


def phase_for(epoch, top, physical):
    if epoch <= top:
        return 'topology'
    offset = (epoch - top - 1) % (3 * physical + top)
    return ('omega_raw', 'gamma_raw', 'input_force_scale_raw')[offset // physical] if offset < 3 * physical else 'topology'


def load_tgb(name, root):
    try:
        from tgb.linkproppred.dataset import LinkPropPredDataset
        from tgb.linkproppred.evaluate import Evaluator
        from tgb.utils.info import PROJ_DIR
    except ImportError as exc:
        raise RuntimeError("Install the optional TGB dependency: pip install '.[tgb]'") from exc
    # TGB 2.3 prefixes its package directory even to the supplied root.
    relative_root = os.path.relpath(Path(root).resolve(), PROJ_DIR)
    ds = LinkPropPredDataset(name=name, root=relative_root)
    ds.load_val_ns()
    ds.load_test_ns()
    return ds, Evaluator(name=name)


def resolve_device(requested):
    if requested not in {'auto', 'cpu', 'cuda'}:
        raise ValueError(f'Unsupported device: {requested}')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but CUDA is unavailable in this PyTorch environment')
    return torch.device('cuda' if requested == 'auto' and torch.cuda.is_available()
                        else 'cpu' if requested == 'auto' else requested)


def run_tgb(args, dataset=None, evaluator=None):
    device = resolve_device(getattr(args, 'device', 'auto'))
    if min(args.epochs, args.topology_epochs, args.physical_epochs, args.num_neg, args.threads) < 1:
        raise ValueError("Epoch counts, negative count and threads must be positive")
    if not math.isfinite(args.time_unit) or args.time_unit <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("time-unit and lr must be finite and positive")
    cycle = args.topology_epochs + 3 * args.physical_epochs
    if args.epochs <= args.topology_epochs or (args.epochs - args.topology_epochs) % cycle:
        raise ValueError(f"End on topology: epochs must be {args.topology_epochs} + R * {cycle}, R >= 1")
    path = Path(args.save_jsonl)
    checkpoint = path.with_suffix('.best.pt')
    if path.exists() or checkpoint.exists():
        raise FileExistsError("Choose a fresh --save-jsonl path; existing runs are preserved")
    if dataset is None:
        dataset, evaluator = load_tgb(args.dataset, args.root)
    from tgb.utils.info import DATA_VERSION_DICT
    data = dataset.full_data
    src = np.asarray(data['sources'], dtype=np.int64)
    dst = np.asarray(data['destinations'], dtype=np.int64)
    times = np.asarray(data['timestamps'], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Invalid or unsorted timestamps")
    masks = [np.asarray(m, dtype=bool) for m in [dataset.train_mask, dataset.val_mask, dataset.test_mask]]
    if not np.all(np.sum(masks, axis=0) == 1):
        raise ValueError("Splits must partition the event stream")
    groups = dict(zip(['train', 'val', 'test'], [timestamp_groups(times, m) for m in masks]))
    if not times[groups['train'][-1][-1]] < times[groups['val'][0][0]] or not times[groups['val'][-1][-1]] < times[groups['test'][0][0]]:
        raise ValueError("Split boundaries must not split simultaneous events")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    nodes = max(int(max(src.max(), dst.max())) + 1, int(getattr(dataset, 'num_nodes', 0)))
    model = FieldNeuralNetwork(num_nodes=nodes, force_dim=1, state_dim=1,
        gamma_init=.12, omega_init=.7, dt=1., topology_mode='observed_sparse',
        state_score=True, learn_gamma=True, learn_omega=True, learn_input_force_scale=True)
    # Candidate vocabulary uses training positives and training-sampled
    # alternatives only. Official held-out negatives never create parameters.
    destinations = np.unique(dst)
    neg = rng.choice(destinations, size=(masks[0].sum(), args.num_neg))
    model.set_sparse_topology_candidates(torch.from_numpy(np.concatenate([src[masks[0]], np.repeat(src[masks[0]], args.num_neg)])),
        torch.from_numpy(np.concatenate([dst[masks[0]], neg.ravel()])))
    # Move after sparse parameter creation and before optimizer construction.
    model = model.to(device)
    print(f'TGB device={device} | threads={args.threads}', flush=True)
    scalars = {n:p for n,p in model.named_parameters() if n.endswith('_raw')}
    other = [p for n,p in model.named_parameters() if n not in scalars]
    optimizers = {'topology': torch.optim.Adam(other, lr=args.lr)}
    optimizers.update({n:torch.optim.Adam([p], lr=args.lr) for n,p in scalars.items()})

    def events(ids):
        return EventBatch(src=torch.from_numpy(src[ids]), dst=torch.from_numpy(dst[ids]),
            t=torch.from_numpy(times[ids]), features=torch.ones((len(ids),1)),
            is_external=torch.zeros(len(ids),dtype=torch.bool)).to(device)

    last_validation = None

    def pass_stream(split, state, last_time, optimizer=None, score=True):
        total, count = 0., 0
        description = f'TGB {split}' if score else f'TGB {split} history replay'
        progress = tqdm(groups[split], desc=description, unit='timestamp', leave=False)
        for step, ids in enumerate(progress, start=1):
            time = float(times[ids[0]])
            state = model.advance_elapsed(state, 0. if last_time is None else (time-last_time)/args.time_unit)
            batch = events(ids)
            if optimizer is not None:
                negatives = rng.choice(destinations, size=(len(ids), args.num_neg))
                # Exclude simultaneous positive destinations for this source.
                for source in np.unique(src[ids]):
                    selected = src[ids] == source
                    allowed = np.setdiff1d(destinations, dst[ids][selected])
                    if not len(allowed):
                        raise ValueError('No negative destinations for source')
                    negatives[selected] = rng.choice(allowed, size=(selected.sum(), args.num_neg))
                pos_score = model.score(state, batch)
                query = EventBatch(src=batch.src.repeat_interleave(args.num_neg), dst=torch.from_numpy(negatives.ravel()).to(device))
                scores = torch.cat([pos_score[:,None], model.score(state, query).reshape(len(ids),-1)],dim=1)
                loss = torch.nn.functional.cross_entropy(scores, torch.zeros(len(ids),dtype=torch.long,device=device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite TGB training loss')
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                total += float(loss.detach()) * len(ids)
                count += len(ids)
                state.detach_()
            elif score:
                negatives = dataset.negative_sampler.query_batch(src[ids], dst[ids], times[ids], split_mode=split)
                for index, candidates in enumerate(negatives):
                    destinations_query = np.concatenate([[dst[ids[index]]], np.asarray(candidates,dtype=np.int64)])
                    query = EventBatch(src=torch.full((len(destinations_query),),int(src[ids[index]])),
                                       dst=torch.from_numpy(destinations_query)).to(device)
                    predictions = model.score(state, query).detach().cpu().numpy()
                    value = evaluator.eval({'y_pred_pos': predictions[:1], 'y_pred_neg': predictions[1:],
                                            'eval_metric': [dataset.eval_metric]})[dataset.eval_metric]
                    total += float(np.asarray(value).mean())
                    count += 1
            # All queries at this timestamp are scored before any revelation.
            state = model.observe_impulses(state, batch)
            last_time = time
            # Update text without forcing a terminal redraw on every event.
            if step == 1 or step % 100 == 0 or step == len(groups[split]):
                if optimizer is not None:
                    progress.set_postfix({
                        'loss': f'{total/max(count,1):.4f}',
                        f'last_val_{dataset.eval_metric}': (
                            'pending' if last_validation is None else f'{last_validation:.4f}'
                        ),
                    }, refresh=False)
                elif score:
                    progress.set_postfix({f'running_{dataset.eval_metric}': f'{total/max(count,1):.4f}'},
                                         refresh=False)
        return state, last_time, total/max(count,1)

    path.parent.mkdir(parents=True, exist_ok=True)
    best = -float('inf')
    best_epoch = None
    for epoch in range(1,args.epochs+1):
        phase = phase_for(epoch,args.topology_epochs,args.physical_epochs)
        for n,p in model.named_parameters():
            p.requires_grad_(n == phase if phase != 'topology' else n not in scalars)
        model.train()
        state = model.init_state(1,nodes,device)
        _,_,loss = pass_stream('train',state,None,optimizers[phase])
        model.eval()
        with torch.no_grad():
            # Rebuild history at the current weights, not mixed epoch weights.
            state = model.init_state(1,nodes,device)
            state,last,_ = pass_stream('train',state,None,score=False)
            _,_,val = pass_stream('val',state,last)
        if not math.isfinite(val):
            raise RuntimeError('Nonfinite validation metric')
        last_validation = val
        if val > best:
            best,best_epoch = val,epoch
            torch.save(model.state_dict(), checkpoint)
        row = {'epoch':epoch,'phase':phase,'train_loss':loss,'val':{dataset.eval_metric:val},
               'parameters':{k:float(v.detach()) for k,v in model.physical_parameters().items()},
               'protocol':'tgb-event-time-unit-impulse','config':vars(args), 'tgb_version':version('py-tgb'),
               'dataset_version':DATA_VERSION_DICT[args.dataset], 'selection_metric':dataset.eval_metric,
               'device':str(device)}
        with path.open('a') as handle: handle.write(json.dumps(row)+'\n')
        print(f'ep {epoch:03d} | {phase} | loss={loss:.4f} | val {dataset.eval_metric}={val:.4f}',flush=True)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    model.eval()
    with torch.no_grad():
        state = model.init_state(1,nodes,device)
        state,last,_ = pass_stream('train',state,None,score=False)
        state,last,_ = pass_stream('val',state,last,score=False)
        _,_,test = pass_stream('test',state,last)
    if not math.isfinite(test):
        raise RuntimeError('Nonfinite test metric')
    with path.open('a') as handle:
        handle.write(json.dumps({'selected_epoch':best_epoch,'val':{dataset.eval_metric:best},
                                 'test':{dataset.eval_metric:test},'seed':args.seed})+'\n')
    print(f'Best epoch {best_epoch} | test {dataset.eval_metric}={test:.4f}',flush=True)
