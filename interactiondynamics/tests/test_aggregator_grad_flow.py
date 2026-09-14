import pytest
import torch

from interactiondynamics.aggregators.hopfield import HopfieldAggregator
from interactiondynamics.aggregators.settransformers import SetTransformerAggregator
from interactiondynamics.core.events import EventBatch
from interactiondynamics.core.interfaces import ModelState
from interactiondynamics.encoders.event_encoder import TGNEventEncoder


@pytest.mark.parametrize("kind", ["settransformer", "hopfield"])
@pytest.mark.parametrize("capacity", [1, 4])
def test_packed_messages_backpropagate_to_event_encoder(kind, capacity):
    torch.manual_seed(42)
    encoder = TGNEventEncoder(8, 2, 8, hidden_dim=16)
    kwargs = dict(msg_dim=8, num_heads=2, max_events_per_node=capacity)
    aggregator = (
        SetTransformerAggregator(**kwargs)
        if kind == "settransformer"
        else HopfieldAggregator(node_dim=8, hidden_dim=16, **kwargs)
    )
    events = EventBatch(
        src=torch.tensor([0, 0, 1]),
        dst=torch.tensor([1, 2, 2]),
        features=torch.randn(3, 2),
    )
    state = ModelState(node=torch.randn(4, 8))
    embeddings = encoder(state, events)
    embeddings.retain_grad()
    messages = aggregator(state, embeddings, events, num_nodes=4)
    assert torch.isfinite(messages).all()
    assert torch.equal(messages[3], torch.zeros(8))
    messages.square().sum().backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert sum(p.grad.abs().sum() for p in encoder.parameters()) > 0
