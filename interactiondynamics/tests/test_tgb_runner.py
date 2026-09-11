import argparse
import json

import numpy as np
import pytest
import torch

from interactiondynamics.models.fnn import FieldNeuralNetwork
from interactiondynamics.training.tgb_runner import run_tgb, timestamp_groups


def test_elapsed_fnn_semigroup_and_no_time_at_impulse():
    model = FieldNeuralNetwork(num_nodes=3, force_dim=1, state_dim=1,
                              gamma_init=.12, omega_init=.7, dt=1.)
    state = model.init_state(1,3,torch.device('cpu'))
    state.node.fill_(1.)
    a = model.advance_elapsed(model.advance_elapsed(state,.3),.7)
    b = model.advance_elapsed(state,1.)
    assert torch.allclose(a.node,b.node,atol=1e-6)
    assert torch.allclose(a.node_prev,b.node_prev,atol=1e-6)
    assert torch.equal(model.advance_elapsed(state,0).node,state.node)
    with pytest.raises(ValueError): model.advance_elapsed(state,-1)


def test_timestamp_groups_keep_simultaneous_events_together():
    assert [g.tolist() for g in timestamp_groups(np.array([0,0,1,2]),np.ones(4,dtype=bool))] == [[0,1],[2],[3]]


def test_tgb_tiny_end_to_end(tmp_path):
    pytest.importorskip('tgb')
    from tgb.linkproppred.evaluate import Evaluator
    class Sampler:
        calls = []
        def query_batch(self, src, dst, ts, split_mode):
            self.calls.append(split_mode)
            return [np.array([2 if d == 1 else 1]) for d in dst]
    sampler = Sampler()
    ds = argparse.Namespace(full_data={'sources':np.zeros(8,dtype=np.int64),
        'destinations':np.array([1,2,1,2,1,2,1,2]),'timestamps':np.arange(8)},
        train_mask=np.arange(8)<4,val_mask=(np.arange(8)>=4)&(np.arange(8)<6),
        test_mask=np.arange(8)>=6,negative_sampler=sampler,eval_metric='mrr')
    args = argparse.Namespace(epochs=5,topology_epochs=1,physical_epochs=1,num_neg=1,
        threads=1,time_unit=1.,lr=.003,seed=0,dataset='tgbl-wiki',root=str(tmp_path),
        save_jsonl=str(tmp_path/'run.jsonl'))
    run_tgb(args,ds,Evaluator(name='tgbl-wiki'))
    rows = [json.loads(line) for line in (tmp_path/'run.jsonl').read_text().splitlines()]
    assert [r['phase'] for r in rows[:-1]] == ['topology','omega_raw','gamma_raw','input_force_scale_raw','topology']
    assert 0 < rows[-1]['test']['mrr'] <= 1
    assert sampler.calls.count('test') == 2
    assert all(r['parameters']['dt'] == 1. for r in rows[:-1])
    with pytest.raises(FileExistsError): run_tgb(args,ds,Evaluator(name='tgbl-wiki'))


def test_tgb_cli():
    from interactiondynamics.train import parse_args
    args = parse_args(['tgb','--save-jsonl','/tmp/example.jsonl'])
    assert args.dataset == 'tgbl-wiki' and args.epochs == 9
