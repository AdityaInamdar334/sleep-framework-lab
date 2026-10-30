import unittest
import torch
from sleep_lab import (
    SmallNet, FrozenExpandable, task_data, soft_distill,
    select_dreams, old_logits_unchanged, run_once,
)

class LabTests(unittest.TestCase):
    def test_task_labels(self):
        x, y = task_data(1, 64, seed=7)
        self.assertTrue(torch.equal(y, (x[:,1] > 0).long()))
        self.assertTrue(bool(torch.all(x[:,2] == 1)))

    def test_old_exact(self):
        torch.manual_seed(22)
        net = SmallNet()
        expandable = FrozenExpandable(net)
        x,_ = task_data(0, 32, seed=8)
        self.assertTrue(old_logits_unchanged(net, expandable, x))
        with torch.no_grad():
            expandable.up.weight.fill_(2.0)
        self.assertTrue(old_logits_unchanged(net, expandable, x))

    def test_soft_distill(self):
        x = torch.randn(16,2)
        self.assertLess(abs(float(soft_distill(x,x))), 1e-5)

    def test_dream_selection(self):
        torch.manual_seed(10)
        net = SmallNet()
        student = FrozenExpandable(net)
        x,_ = task_data(1, 256, 5)
        dx, dy = select_dreams(student, net, x, top_n=20, random_n=10)
        self.assertEqual(dx.shape[0], dy.shape[0])
        self.assertLessEqual(len(dx), 30)

    def test_full_quick_benchmark(self):
        out = run_once(0, quick=True)
        self.assertEqual(out["old_checkpoint"]["old"], out["expanded_after_dream_rehearsal"]["old"])
        self.assertGreater(out["selected_dreams"], 0)

if __name__ == '__main__': unittest.main()

class ConceptTests(unittest.TestCase):
    def test_edit_distance(self):
        from concepts import edit_distance
        self.assertEqual(edit_distance('kitten', 'sitting'), 3)
        self.assertEqual(edit_distance('', ''), 0)

    def test_imitation_reward(self):
        from concepts import imitation_reward
        self.assertEqual(imitation_reward('answer', 'answer', True), 1.0)
        self.assertEqual(imitation_reward('wrong', 'answer', False,
                                          max_normalized_distance=0.0), 0.0)
