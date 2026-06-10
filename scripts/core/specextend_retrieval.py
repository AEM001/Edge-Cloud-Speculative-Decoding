"""Edge-side retrieval state for SpecExtend-style draft KV management.

The tensor KV cache itself is backend-specific. This module keeps the portable
part of the algorithm: chunk bookkeeping and target-attention based selection.
Backends use the returned token indices to materialize their working KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class RetrievalChunk:
    chunk_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class SpecExtendRetrievalState:
    def __init__(self, chunk_size: int = 64, top_k_chunks: int = 16):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if top_k_chunks <= 0:
            raise ValueError("top_k_chunks must be positive")

        self.chunk_size = chunk_size
        self.top_k_chunks = top_k_chunks
        self.total_seq_len = 0
        self.chunks: List[RetrievalChunk] = []
        self.selected_chunks: List[RetrievalChunk] = []

    def append_tokens(self, token_count: int) -> bool:
        """Record new tokens appended to the full draft KV cache.

        Returns True when one or more new chunks were created.
        """
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        if token_count == 0:
            return False

        old_chunk_count = len(self.chunks)
        previous_total = self.total_seq_len
        self.total_seq_len += token_count

        if not self.chunks:
            self._rebuild_chunks()
        else:
            self._extend_chunks(previous_total)

        if not self.selected_chunks:
            self.selected_chunks = self.chunks[-min(self.top_k_chunks, len(self.chunks)) :]
        else:
            if len(self.chunks) > old_chunk_count:
                selected_ids = {chunk.chunk_id for chunk in self.selected_chunks}
                self.selected_chunks.extend(
                    chunk for chunk in self.chunks[old_chunk_count:] if chunk.chunk_id not in selected_ids
                )
            self._refresh_selected_tail()

        return len(self.chunks) > old_chunk_count

    def select_by_attention(self, attention_scores: Sequence[float]) -> List[RetrievalChunk]:
        """Select chunks by mean target attention score.

        The input is expected to cover token positions in the full draft cache.
        Extra scores are ignored; missing scores make the corresponding suffix
        chunks ineligible until the backend can provide complete scores.
        """
        selected = select_chunks_by_attention(
            self.chunks,
            attention_scores,
            top_k_chunks=self.top_k_chunks,
        )
        self.selected_chunks = selected
        self._refresh_selected_tail()
        return list(self.selected_chunks)

    def selected_token_indices(self) -> List[int]:
        indices: List[int] = []
        for chunk in self.selected_chunks:
            indices.extend(range(chunk.start, chunk.end))
        return sorted(set(indices))

    def selected_chunk_ids(self) -> List[int]:
        return [chunk.chunk_id for chunk in self.selected_chunks]

    def set_selected_chunk_ids(self, chunk_ids: Iterable[int]) -> List[int]:
        wanted = set(chunk_ids)
        self.selected_chunks = [chunk for chunk in self.chunks if chunk.chunk_id in wanted]
        self._refresh_selected_tail()
        return self.selected_token_indices()

    def _rebuild_chunks(self) -> None:
        self.chunks = []
        start = 0
        chunk_id = 0
        while start < self.total_seq_len:
            end = min(start + self.chunk_size, self.total_seq_len)
            self.chunks.append(RetrievalChunk(chunk_id=chunk_id, start=start, end=end))
            start = end
            chunk_id += 1

    def _extend_chunks(self, previous_total: int) -> None:
        if previous_total == self.total_seq_len:
            return
        while self.chunks and self.chunks[-1].end < self.total_seq_len:
            last = self.chunks[-1]
            if last.length < self.chunk_size:
                end = min(last.start + self.chunk_size, self.total_seq_len)
                self.chunks[-1] = RetrievalChunk(last.chunk_id, last.start, end)
                continue
            start = last.end
            end = min(start + self.chunk_size, self.total_seq_len)
            self.chunks.append(RetrievalChunk(last.chunk_id + 1, start, end))

    def _refresh_selected_tail(self) -> None:
        if not self.chunks or not self.selected_chunks:
            return

        selected_ids = {chunk.chunk_id for chunk in self.selected_chunks}
        last_chunk = self.chunks[-1]
        if last_chunk.chunk_id in selected_ids:
            self.selected_chunks = [
                last_chunk if chunk.chunk_id == last_chunk.chunk_id else chunk
                for chunk in self.selected_chunks
            ]


def build_chunks(total_seq_len: int, chunk_size: int) -> List[RetrievalChunk]:
    if total_seq_len < 0:
        raise ValueError("total_seq_len must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks: List[RetrievalChunk] = []
    start = 0
    chunk_id = 0
    while start < total_seq_len:
        end = min(start + chunk_size, total_seq_len)
        chunks.append(RetrievalChunk(chunk_id=chunk_id, start=start, end=end))
        start = end
        chunk_id += 1
    return chunks


def select_chunks_by_attention(
    chunks: Sequence[RetrievalChunk],
    attention_scores: Sequence[float],
    top_k_chunks: int,
) -> List[RetrievalChunk]:
    if top_k_chunks <= 0:
        raise ValueError("top_k_chunks must be positive")
    if not chunks:
        return []
    if len(attention_scores) <= 0:
        raise ValueError("attention_scores must not be empty")

    scored = []
    for chunk in chunks:
        if chunk.end > len(attention_scores):
            continue
        values = attention_scores[chunk.start : chunk.end]
        if not values:
            continue
        scored.append((sum(values) / len(values), chunk))

    if not scored:
        raise ValueError("attention_scores do not cover any retrieval chunk")

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for _, chunk in scored[:top_k_chunks]]
    selected.sort(key=lambda chunk: chunk.chunk_id)
    return selected
