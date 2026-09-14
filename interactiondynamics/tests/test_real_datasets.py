import gzip

import numpy as np
import pandas as pd
import pytest
import torch

from interactiondynamics.data.download_benchmarks import SOURCES
from interactiondynamics.data.social import SocialConfig, SocialEventDataset
from interactiondynamics.data.traffic import TrafficConfig, TrafficDataset


@pytest.mark.parametrize('name', ['college_msg', 'email_eu_core', 'sociopatterns'])
def test_social_cli_preset(name):
    from interactiondynamics.train import parse_args
    from interactiondynamics.training.presets import build_suite
    from interactiondynamics.training.runner import apply_model_overrides
    args = parse_args(['quick', '--dataset', name, '--max-runs', '1'])
    suite = build_suite('quick', torch.device('cpu'), args.dataset, args)
    cfg = apply_model_overrides(suite.runs[0].model_cfg, args)
    assert suite.dataset == 'social' and suite.dataset_kwargs['name'] == name
    assert cfg.fnn and cfg.fnn_topology_mode == 'observed_sparse'
    assert cfg.fnn_dt == 1 and not cfg.fnn_learn_dt
    assert cfg.fnn_max_dt_omega is None
    assert all(not r.model_cfg.fnn for r in suite.runs[1:])


def test_social_preserves_direction_empty_bins_and_timestamp_precision(tmp_path):
    name = 'college_msg'
    directory = tmp_path / name
    directory.mkdir()
    with gzip.open(directory / SOURCES[name][0], 'wt') as handle:
        handle.write('20 10 1700000180\n10 20 1700000000\n10 20 1700000020\n')
    ds = SocialEventDataset(SocialConfig(name, str(tmp_path), bin_size=20, split_by='time'))
    assert ds.num_bins == 10
    assert ds.bin_ids.tolist() == [0, 1, 9]
    batches = list(ds.bins())
    assert len(batches) == 7 and batches[2].num_events == 0
    assert batches[0].src.tolist() == [0] and batches[0].dst.tolist() == [1]
    assert list(ds.bins('test'))[-1].src.tolist() == [1]
    assert len(list(ds.bins())) == len(batches)
    assert ds.spec().source_id_range() == ds.spec().destination_id_range() == (0, 2)


def test_contacts_are_reciprocal_without_external_impulses(tmp_path):
    name = 'sociopatterns'
    directory = tmp_path / name
    directory.mkdir()
    with gzip.open(directory / SOURCES[name][0], 'wt') as handle:
        handle.write('1700000000 7 9 A B\n1700000180 9 7 B A\n')
    ds = SocialEventDataset(SocialConfig(name, str(tmp_path), split_by='time'))
    batch = next(iter(ds.bins()))
    assert set(zip(batch.src.tolist(), batch.dst.tolist())) == {(0, 1), (1, 0)}
    assert not batch.is_external.any()
    assert ds.spec().num_events == 4


def test_fnn_training_accepts_empty_target_bins(tmp_path):
    import argparse
    from interactiondynamics.models.model_factory import build_model
    from interactiondynamics.training.presets import build_suite
    from interactiondynamics.training.runner import train_one_epoch

    directory = tmp_path / 'college_msg'
    directory.mkdir()
    with gzip.open(directory / SOURCES['college_msg'][0], 'wt') as handle:
        handle.write('1 2 0\n2 1 20\n1 2 180\n')
    ds = SocialEventDataset(SocialConfig(root=str(tmp_path), bin_size=20, split_by='time'))
    suite = build_suite('quick', torch.device('cpu'), dataset_override='college_msg',
                        args=argparse.Namespace(seed=0))
    cfg = suite.runs[0].model_cfg
    cfg.fnn_topology_mode = 'dense'
    model = build_model(ds.spec(), cfg)
    for name, parameter in model.named_parameters():
        if name.endswith('_raw'):
            parameter.requires_grad_(False)
    suite.train_cfg.num_nodes = ds.spec().num_nodes
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    stats = train_one_epoch(model, ds.bins('train'), None, None, optimizer, suite.train_cfg)
    assert np.isfinite(stats['loss'])


def test_traffic_masks_train_statistics_and_split_isolation(tmp_path):
    pytest.importorskip('tables')
    directory = tmp_path / 'metr_la'
    directory.mkdir()
    values = np.full((100, 2), 10.0)
    values[0, 0] = 0
    values[1, 1] = np.nan
    values[70:] = 1000
    pd.DataFrame(values, index=pd.date_range('2020-01-01', periods=100, freq='5min')).to_hdf(
        directory / 'metr-la.h5', key='df')
    ds = TrafficDataset(TrafficConfig(root=str(tmp_path), history=2, horizon=2))
    assert ds.mean.item() == 10
    assert not ds.observed[0, 0, 0] and not ds.observed[1, 1, 0]
    train = ds.windows('train')
    assert len(train) == 67
    assert train[len(train)-1]['target_index'] == 68
    assert ds.windows('val')[0]['target_index'] == 72
    assert torch.isfinite(train[0]['x']).all()
    assert ds.windows('test')[0]['y'].mean().item() == 1000
    with pytest.raises(IndexError):
        train[len(train)]


def test_traffic_windows_do_not_cross_clock_gaps(tmp_path):
    pytest.importorskip('tables')
    directory = tmp_path / 'metr_la'
    directory.mkdir()
    times = pd.date_range('2020-01-01', periods=101, freq='5min').delete(20)
    pd.DataFrame(np.ones((100, 2)), index=times).to_hdf(directory / 'metr-la.h5', key='df')
    ds = TrafficDataset(TrafficConfig(root=str(tmp_path), history=2, horizon=2))
    windows = ds.windows('train')
    assert len(windows) == 64  # three four-step windows would cross the gap
    for i in range(len(windows)):
        middle = windows[i]['target_index']
        assert times[middle+1] - times[middle-2] == pd.Timedelta(minutes=15)


def test_event_quantile_splits_keep_validation_populated(tmp_path):
    directory = tmp_path / 'college_msg'
    directory.mkdir()
    with gzip.open(directory / SOURCES['college_msg'][0], 'wt') as handle:
        for t in list(range(19)) + [10000]:
            handle.write(f'1 2 {t}\n')
    ds = SocialEventDataset(SocialConfig(root=str(tmp_path), bin_size=1))
    assert [sum(b.num_events for b in ds.bins(s)) for s in ('train','val','test')] == [14, 3, 3]
