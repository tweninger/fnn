from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, cast, List, Sequence

import torch # hi tensors and randomness

from core.events import EventBatch # hi thing trainer.py consumes
from datasets.interfaces import DataSpec, EventStreamDataset # hi dataset interface

"""
It's an EventStreamDataset that outputs one EventBatch per time bin (a temporal sequence of event bins), just like JODIE loader does!!
Except we make interactions synthetically from a basic 1D chain of "masses" (a thing) and springs (maybe a rubberband)
- At time bin b, node i interacted with node j, and the intereaction had dx, dv, force, and extension features. Nice.
- Outputs bin 0: EventBatch(src, dst, t, features)
- simple chain: sparse interactions, local structure, easier learning
- fully connected: dense interactions, global coupling, harder learning, more events
"""


# config class - aka bag of settings 
@dataclass
class SpringMassConfig:
    name: str = "spring_mass"

    # number of masses (aka nodes, entities, whatever)
    num_nodes: int = 64
    # number of simulated time steps to run
    num_bins: int = 1024

    # physics-ish params 
    dt: float = 0.05 # integration time step (how big each time step is)
    spring_k: float = 1.0 # hooke's law constant (how strongly neighbors pull/push each other)
    damping: float = 0.995 # velocity damping multiplier each step (multiply velocity by this each step, so motion slowly dies down)
    rest_length: float = 1.0 # preferred spacing between neighboring masses (the spring wants neighboring nodes to be 1 unit apart)

    # random initial condition scales
    init_pos_noise: float = 0.05 # how much random displacement initially
    init_vel_noise: float = 0.05 # how much random initial motion

    # only emit an interaction if force is above this threshold (aka "strong enough")
    use_force_threshold: bool = False
    force_threshold: float = 0.015
    
    # if true, store both i->j and j->i for each active spring
    bidirectional: bool = True

    # train/val/test split by TIME
    split_fracs: tuple = (0.7, 0.15, 0.15)

    #hi seed
    seed: int = 0

    #frankly idek what this is
    device: Optional[torch.device] = None

_EDGE_FEATURE_NAMES: Sequence[str] = (
    # "dx",
    # "dy",
    # "dvx",
    # "dvy",
    #"dist",
    #"extension",
    # "fx",
    # "fy",
)
# dataset class shell - aka actual dataset object
class SpringMassDataset(EventStreamDataset):
    def __init__(self, cfg: SpringMassConfig):
        self.cfg = cfg

        #we have 4, dx, dv, extension, and force... JODIE has msg features fyi
        self._event_dim = len(_EDGE_FEATURE_NAMES)

        # build the synthetic simulation and define train/val/test splits
        self._build()
        self._split()

    #create local random number generator 
    def _build(self):
        # torch RNG w/ fixed seed so config reproduces same data
        g = torch.Generator().manual_seed(self.cfg.seed)


        #pull config values into local variables and give shorter names
        N = self.cfg.num_nodes
        T = self.cfg.num_bins

        dt = self.cfg.dt
        k = self.cfg.spring_k
        damping = self.cfg.damping
        rest = self.cfg.rest_length
        thr = self.cfg.force_threshold
        bidir = self.cfg.bidirectional

        # initiailize physical state
        # HERE, x: node positions and y: node velocities and v:random initial velocity per node

        #start masses roughly on 1D line spaced by rest_length... aka (1,2,3,4...)
        x = torch.arange(N, dtype=torch.float32) * rest

        # add random displacement noise so they start rougly evenly spaced but perturbed a bit
        x += self.cfg.init_pos_noise * torch.randn(N, generator=g)

        # small random initial velocities so each node begins with a small random movement
        v = self.cfg.init_vel_noise * torch.randn(N, generator=g)

        # hi (src, dst, t, features)
        # these lists hold one tensor per ACTIVE time bin
        self.src_bins: List[torch.Tensor] = []
        self.dst_bins: List[torch.Tensor] = []
        self.t_bins: List[torch.Tensor] = []
        self.feat_bins: List[torch.Tensor] = []
        self.node_target_bins: List[torch.Tensor] = []

        # simulate forward in time!! -> main simulation loop
        # simulate one time step at a time, for T total steps
        # at each step, compute which interactions happened and update world state for next step
        for b in range(T):

            # net_force[n] accumulates total force on node n this step
            net_force = torch.zeros(N)

            # temporary event lists for this one bin
            src_list, dst_list, feat_list = [], [], []

            # choose ur topology here

            #simple chain topology
            for i in range(N - 1):
                j = i + 1  # only nearest-neighbor springs i.e. (0,1), (1,2), ...

                # dx: how far apart the two neighboring nodes currently are
                # dv: how different their velocities are
                # extension: how far the spring is from its preferred length (e.g. if dx > rest, then spring is stretched)
                # force: k * extension (hooke's law choice)

                # so bigger the extension/compression, stronger the interaction!                
                
                # relative displacement and velocity
                dx = x[j] - x[i]
                dv = v[j] - v[i]

                # hooke spring extension relative to rest length
                extension = dx - rest

                # hooke's law force magnitude/sign
                force = k * extension
                dist = torch.norm(dx) + 1e-8

                # newton's third law:
                # i gets +force, g gets -force
                # aka interaction between i and j contributes to motion of both nodes
                net_force[i] += force
                net_force[j] -= force

                # only create an edge if the interaction is "strong enough" -> bc i say so
                keep_edge = (torch.abs(force) > thr) if self.cfg.use_force_threshold else True

                if keep_edge:
                    #feat = [float(dist)]
                    feat = []


                    src_list.append(i)
                    dst_list.append(j)
                    feat_list.append(feat)

                    if bidir:
                        src_list.append(j)
                        dst_list.append(i)
                        feat_list.append([])

            node_targets = torch.stack([x, v], dim=-1)   # [N, 2]
            # if at least one spring was active, store this as a real event bin
            # aka convert python lists to tensors, create timestamp tensor t, and store the bin
            if len(src_list) > 0:
                src = torch.tensor(src_list, dtype=torch.long)
                dst = torch.tensor(dst_list, dtype=torch.long)
                feats = torch.zeros((len(src_list), self._event_dim), dtype=torch.float32)
                # every event in this bin gets the same discrete timestamp b
                t = torch.full((len(src_list),), b, dtype=torch.long)
                
                self.src_bins.append(src)
                self.dst_bins.append(dst)
                self.t_bins.append(t)
                self.feat_bins.append(feats)
                self.node_target_bins.append(node_targets.clone()) # update the physical system
            
            # a = F / m ; here m is implicitly 1
            a = net_force # acceleration is being treated as force, because mass is 1

            #damped Euhler update
            v = damping * (v + dt * a) # new velocity = old velocity plus acceleration effect, then shurnk by damping
            x = x + dt * v # move positions using updated velocity


        
        #_num_bins is not cfg.num_bins, it's only # of bins that actually contained events
        self._num_bins = len(self.src_bins)

        #total number of event records across all active bins
        self._num_events = sum(x.numel() for x in self.src_bins)

    def _split(self):
        # split over STORED bins (active bins only!!!)
        T = self._num_bins
        f_tr, f_va, f_te = self.cfg.split_fracs

        n_tr = int(T * f_tr)
        n_va = int(T * f_va)

        self.split_bins = {
            "train": list(range(0, n_tr)),
            "val": list(range(n_tr, n_tr + n_va)),
            "test": list(range(n_tr + n_va, T)),
        }

    def spec(self) -> DataSpec:
        # returns a re-iterable stream object (which we need for our framework lol)
        return DataSpec(
            name=self.cfg.name,
            num_nodes=self.cfg.num_nodes,
            event_dim=self._event_dim,
            num_events=self._num_events,
            num_bins=self._num_bins,
            extra={
                "node_target_dim": 2,
                "node_target_names": ["x", "v"],
            }
        )

    # returns re-iterable stream object
    def bins(self, split: str = "train") -> Iterable[EventBatch]:
        return _Stream(
            self.src_bins,
            self.dst_bins,
            self.t_bins,
            self.feat_bins,
            self.node_target_bins,
            self.split_bins[split],
            self.cfg.device,
        )

# stores references
class _Stream(Iterable[EventBatch]):
    def __init__(self, src, dst, t, feat, node_targets, idxs, device):
        self.src = src
        self.dst = dst
        self.t = t
        self.feat = feat
        self.node_targets = node_targets
        self.idxs = idxs
        self.device = device

    def __iter__(self) -> Iterator[EventBatch]:
        for i in self.idxs:
            eb = EventBatch(
                src=cast(torch.LongTensor, self.src[i]),
                dst=cast(torch.LongTensor, self.dst[i]),
                t=cast(torch.LongTensor, self.t[i]),
                features=self.feat[i],
                node_targets=self.node_targets[i],
            )
            if self.device is not None:
                eb = eb.to(self.device)
            yield eb