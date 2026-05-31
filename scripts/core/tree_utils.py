"""Pure tree helpers shared by target verification tests and backend code."""

from __future__ import annotations

from typing import List, Sequence


def paths_from_tree(tree_input_ids: Sequence[int], parent_indices: Sequence[int]) -> List[List[int]]:
    paths: List[List[int]] = []
    for idx in range(len(tree_input_ids)):
        reverse_path = []
        current = idx
        seen = set()
        while current >= 0 and current not in seen:
            seen.add(current)
            reverse_path.append(int(tree_input_ids[current]))
            current = int(parent_indices[current])
        paths.append(list(reversed(reverse_path)))
    return paths


def indices_for_path(node_idx: int, parent_indices: Sequence[int]) -> List[int]:
    indices = []
    current = node_idx
    seen = set()
    while current >= 0 and current not in seen:
        seen.add(current)
        indices.append(current)
        current = int(parent_indices[current])
    return list(reversed(indices))
