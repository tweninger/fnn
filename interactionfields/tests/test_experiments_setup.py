import inspect
import pytest

from interactionfields.graphs import build_graph, _BUILDERS
from interactionfields.plot_results import make_plot_kwargs
from interactionfields.run_experiments import (
    SIZE_PROFILES,
    build_experiment_suite,
)
from interactionfields.simulate import SIMULATORS


GRIDISH_KINDS = {"grid", "gate", "torus_surface", "torus_grid", "cylinder_x", "cylinder_y"}


def test_size_profiles_are_sane():
    for key, prof in SIZE_PROFILES.items():
        assert prof.name == key
        assert prof.h > 0 and prof.w > 0
        assert prof.t_bins > 0
        assert 0 < prof.cut <= prof.t_bins
        assert prof.n > 0
        assert prof.epochs > 0


@pytest.mark.parametrize("profile_key", list(SIZE_PROFILES.keys()))
def test_experiment_suite_builds_and_graphs_valid(profile_key: str):
    prof = SIZE_PROFILES[profile_key]
    suite = build_experiment_suite(prof, seed=0)
    assert suite, "suite must be non-empty"
    names = [s.name for s in suite]
    assert len(names) == len(set(names)), "experiment names must be unique"

    for exp in suite:
        assert exp.simulator_kind in SIMULATORS
        assert exp.graph_kind in _BUILDERS
        A, meta = build_graph(exp.graph_kind, **exp.graph_kwargs)
        assert A.shape is not None
        assert A.shape[0] == A.shape[1]
        assert "kind" in meta

        if exp.graph_kind in GRIDISH_KINDS and "shape" in meta:
            h, w = int(meta["shape"][0]), int(meta["shape"][1])
            frame_kwargs, anim_kwargs = make_plot_kwargs(exp.graph_kind, h, w, meta, A)
            assert isinstance(frame_kwargs, dict)
            assert isinstance(anim_kwargs, dict)


@pytest.mark.parametrize("profile_key", list(SIZE_PROFILES.keys()))
def test_experiment_suite_simulator_kwargs_satisfy_required_params(profile_key: str):
    prof = SIZE_PROFILES[profile_key]
    suite = build_experiment_suite(prof, seed=0)
    assert suite, "suite must be non-empty"

    for exp in suite:
        sim_fn = SIMULATORS[exp.simulator_kind]
        sig = inspect.signature(sim_fn)
        required = {
            name
            for name, param in sig.parameters.items()
            if param.default is inspect._empty
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        # run_one injects adj and t_bins, and resolves center-based defaults
        required -= {"adj", "t_bins"}
        missing = required - set(exp.simulator_kwargs.keys())
        assert not missing, f"{exp.name} missing simulator kwargs: {sorted(missing)}"
