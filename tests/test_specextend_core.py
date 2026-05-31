import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from core.protocol import SpecExtendTreeRequest, SpecExtendTreeResponse
from core.qwen_specextend_backend import QwenSpecExtendDraftBackend, QwenSpecExtendTargetBackend
from core.specextend_retrieval import build_chunks, select_chunks_by_attention, SpecExtendRetrievalState


class SpecExtendCoreTests(unittest.TestCase):
    def test_protocol_round_trip(self):
        req = SpecExtendTreeRequest(
            request_id="r1",
            prefix_ids=[1, 2],
            tree_input_ids=[3, 4],
            tree_position_ids=[2, 3],
            parent_indices=[-1, 0],
            tree_attention_mask=[[1, 0], [1, 1]],
            retrieve_attn_scores=True,
            metadata={},
        )
        self.assertEqual(SpecExtendTreeRequest.from_dict(req.to_dict()), req)

        resp = SpecExtendTreeResponse(
            request_id="r1",
            accepted_len=1,
            correction_token_id=9,
            accepted_tree_indices=[0],
            server_verify_time_ms=1.5,
            target_attn_scores=[0.1, 0.9],
        )
        self.assertEqual(SpecExtendTreeResponse.from_dict(resp.to_dict()), resp)

    def test_retrieval_selects_highest_mean_chunks(self):
        state = SpecExtendRetrievalState(chunk_size=2, top_k_chunks=2)
        state.append_tokens(6)

        selected = state.select_by_attention([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])

        self.assertEqual([chunk.chunk_id for chunk in selected], [1, 2])
        self.assertEqual(state.selected_token_indices(), [2, 3, 4, 5])

    def test_target_side_chunk_selection_matches_edge_retrieval(self):
        scores = [0.1, 0.2, 0.9, 0.8, 0.3, 0.4]
        chunks = build_chunks(total_seq_len=len(scores), chunk_size=2)

        selected = select_chunks_by_attention(chunks, scores, top_k_chunks=2)

        self.assertEqual([chunk.chunk_id for chunk in selected], [1, 2])

    def test_sparse_draft_context_uses_only_selected_retrieval_indices(self):
        prefix = [10, 11, 12, 13, 14, 15]

        context = QwenSpecExtendDraftBackend._draft_context(prefix, [0, 1, 4, 5])

        self.assertEqual(context.token_ids, [10, 11, 14, 15])
        self.assertEqual(context.position_ids, [0, 1, 4, 5])

    def test_tree_attention_mask_contains_ancestors(self):
        mask = QwenSpecExtendDraftBackend._tree_attention_mask([-1, 0, 1, 0])

        self.assertEqual(mask[0], [1, 0, 0, 0])
        self.assertEqual(mask[2], [1, 1, 1, 0])
        self.assertEqual(mask[3], [1, 0, 0, 1])

    def test_target_tree_path_helpers(self):
        tree_input_ids = [10, 11, 12, 13]
        parent_indices = [-1, 0, 1, 0]

        paths = QwenSpecExtendTargetBackend._paths_from_tree(tree_input_ids, parent_indices)
        indices = QwenSpecExtendTargetBackend._indices_for_path(2, parent_indices)

        self.assertEqual(paths[2], [10, 11, 12])
        self.assertEqual(paths[3], [10, 13])
        self.assertEqual(indices, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
