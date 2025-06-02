import torch
import unittest
from model.optimal_transport_layer import Optimal_Transport_Layer, log_optimal_transport_extended

class TestOptimalTransportLayer(unittest.TestCase):
    def setUp(self):
        # Setup configuration for the model
        self.config = {
            'sinkhorn_iterations': 100,
            'feature_dim': 128,
            'matched_threshold': 0.5,
            'epsilon': 0.1,
            'tau': 1.0
        }
        self.model = Optimal_Transport_Layer(self.config)
        
        # Create 20 mock descriptor pairs for testing
        self.video_descriptors = []
        for _ in range(20):
            mdesc0 = torch.randn(1, 10, 128)
            mdesc1 = torch.randn(1, 10, 128)
            match_gt = {'a2b': torch.tensor([[0, 0], [1, 1]]), 'un_a': torch.tensor([2]), 'un_b': torch.tensor([2])}
            self.video_descriptors.append((mdesc0, mdesc1, match_gt))

    def test_log_optimal_transport_extended(self):
        for idx, (mdesc0, mdesc1, match_gt) in enumerate(self.video_descriptors):
            with self.subTest(video_index=idx):
                scores, indices0, indices1, mscores0, mscores1 = self.model(mdesc0, mdesc1, match_gt=match_gt)

                # Assert shape is (M+1, N+1) for 128 descriptors + dustbin
                self.assertEqual(scores.shape, (129, 129))

                # Assert that indices0 and indices1 have the correct shape
                self.assertEqual(indices0.shape, (128,))
                self.assertEqual(indices1.shape, (128,))

                # Test the enhanced matching loss calculation with ground truth
                self.assertTrue(self.model.matching_loss is not None)
                self.assertTrue(self.model.hard_pair_loss is not None)

if __name__ == '__main__':
    unittest.main()