import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from core.protocol import SpecExtendRequest, SpecExtendResponse
from core.qwen_specextend_backend import SpecExtendDraftKVCache
from core.specextend_retrieval import build_chunks, select_chunks_by_attention, SpecExtendRetrievalState


class SpecExtendCoreTests(unittest.TestCase):
    def test_protocol_round_trip(self):
        req = SpecExtendRequest(
            request_id="r1",
            prefix_ids=[1, 2],
            draft_ids=[3, 4],
            retrieve_attn_scores=True,
            metadata={},
        )
        self.assertEqual(SpecExtendRequest.from_dict(req.to_dict()), req)

        resp = SpecExtendResponse(
            request_id="r1",
            accepted_len=1,
            correction_token_id=9,
            accepted_indices=[0],
            server_verify_time_ms=1.5,
            target_attn_scores=[0.1, 0.9],
        )
        self.assertEqual(SpecExtendResponse.from_dict(resp.to_dict()), resp)

    def test_retrieval_selects_highest_mean_chunks(self):
        state = SpecExtendRetrievalState(chunk_size=2, top_k_chunks=2)
        state.append_tokens(6)

        selected = state.select_by_attention([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])

        self.assertEqual([chunk.chunk_id for chunk in selected], [1, 2])
        self.assertEqual(state.selected_token_indices(), [2, 3, 4, 5])

    def test_retrieval_appends_new_chunks_to_selected_working_set(self):
        state = SpecExtendRetrievalState(chunk_size=2, top_k_chunks=1)
        state.append_tokens(4)
        state.set_selected_chunk_ids([0])

        state.append_tokens(1)

        self.assertEqual(state.selected_chunk_ids(), [0, 2])
        self.assertEqual(state.selected_token_indices(), [0, 1, 4])

    def test_target_side_chunk_selection_matches_edge_retrieval(self):
        scores = [0.1, 0.2, 0.9, 0.8, 0.3, 0.4]
        chunks = build_chunks(total_seq_len=len(scores), chunk_size=2)

        selected = select_chunks_by_attention(chunks, scores, top_k_chunks=2)

        self.assertEqual([chunk.chunk_id for chunk in selected], [1, 2])

    def test_draft_cache_keeps_chunks_since_last_retrieval_update(self):
        cache = SpecExtendDraftKVCache.__new__(SpecExtendDraftKVCache)
        cache.full_token_ids = list(range(10))
        cache.chunk_size = 2
        cache.chunks = build_chunks(total_seq_len=10, chunk_size=2)
        cache.selected_chunks = []
        cache.working_token_indices = []
        cache._retrieval_base_seq_len = 0
        cache._rebuild_working_cache = lambda: None

        cache.select_chunks([0], retrieval_selection_updated=True)
        self.assertEqual([chunk.chunk_id for chunk in cache.selected_chunks], [0])

        cache.full_token_ids = list(range(14))
        cache.chunks = build_chunks(total_seq_len=14, chunk_size=2)
        cache.select_chunks([0], retrieval_selection_updated=False)

        self.assertEqual([chunk.chunk_id for chunk in cache.selected_chunks], [0, 5, 6])
        self.assertEqual(cache.working_token_indices, [0, 1, 10, 11, 12, 13])

if __name__ == "__main__":
    unittest.main()
