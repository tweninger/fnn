import numpy as np
from collections import defaultdict




def predict_link_probabilities(edge_memories: set, edges_tuple: tuple):
    """
    get the link probabilities by predicting whether each edge in edges_tuple appears in edge_memories
    :param edge_memories: set, store the edges in memory, {(src_node_id, dst_node_id), ...}
    :param edges_tuple: tuple, edges with (src_node_ids, dst_node_ids)
    :return:
    """
    src_node_ids, dst_node_ids = edges_tuple
    # probabilities of all the edges
    probabilities = []
    for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids):
        if (src_node_id, dst_node_id) in edge_memories:
            probabilities.append(1.0)
        else:
            probabilities.append(0.0)

    return np.array(probabilities)


def edge_bank_unlimited_memory(history_src_node_ids: np.ndarray, history_dst_node_ids: np.ndarray):
    """
    EdgeBank with unlimited memory, which stores every edge that it has seen
    :param history_src_node_ids: ndarray, shape (num_historical_edges, )
    :param history_dst_node_ids: ndarray, shape (num_historical_edges, )
    :return:
    """
    edge_memories = set((history_src_node_id, history_dst_node_id) for history_src_node_id, history_dst_node_id
                        in zip(history_src_node_ids, history_dst_node_ids))
    return edge_memories

