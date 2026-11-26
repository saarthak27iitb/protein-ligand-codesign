"""Scientific invariants and policy-gradient regressions; no mock performance claims."""

import sys, tempfile, unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.common import kabsch, residues, xyz
from research.models import Agent, PPO
from research.proposals import project_pocket


class ScientificChecks(unittest.TestCase):
    def test_alignment_preserves_handedness(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        fitted, t, err = kabsch(p, p @ r + 4)
        self.assertLess(err, 1e-10)
        self.assertAlmostEqual(np.linalg.det(fitted), 1.0)

    def test_masked_actions_never_selected(self):
        a = Agent()
        s = np.zeros(40)
        features = np.zeros((3, 40))
        mask = [False, True, False]
        for _ in range(50):
            index, record = a.choose(s, features, mask)
            self.assertEqual(index, 1)
            self.assertAlmostEqual(float(record["old_logp"]), 0.0, places=6)

    def test_ppo_prefers_rewarded_actions(self):
        torch.manual_seed(10)
        cfg = {
            "learning_rate": 0.003,
            "gamma": 1.0,
            "ppo_epochs": 4,
            "clip_ratio": 0.2,
            "entropy": 0.001,
            "max_grad_norm": 1.0,
        }
        learner = PPO(cfg)
        s = np.zeros(40)
        features = np.zeros((2, 40))
        features[1, 0] = 1
        before = float(
            learner.agents[0]
            .distribution(
                torch.tensor(s, dtype=torch.float32),
                torch.tensor(features, dtype=torch.float32),
                torch.ones(2, dtype=torch.bool),
            )
            .probs[1]
        )
        for epoch in range(20):
            for _ in range(16):
                action, record = learner.agents[0].choose(s, features, [True, True])
                learner.add_episode(
                    [[record], [None]], [[1.0 if action else -1.0], [0.0]]
                )
            learner.update()
        after = float(
            learner.agents[0]
            .distribution(
                torch.tensor(s, dtype=torch.float32),
                torch.tensor(features, dtype=torch.float32),
                torch.ones(2, dtype=torch.bool),
            )
            .probs[1]
        )
        self.assertGreater(after, before + 0.2)

    def test_projection_fixes_backbone_and_mutation_budget(self):
        current = ROOT / "vendor/PocketGen/examples/2p16/2p16.pdb"
        proposed = ROOT / "runs/cooperative-454953/round_0_pockets/0_whole_relaxed.pdb"
        if not proposed.exists():
            self.skipTest("Optional real PocketGen artifact absent")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "projected.pdb"
            project_pocket(current, proposed, current, out, 2, 42)
            a, b = residues(current), residues(out)
            self.assertEqual(a.keys(), b.keys())
            count = sum(
                next(iter(a[k].values()))[17:20] != next(iter(b[k].values()))[17:20]
                for k in a
            )
            self.assertLessEqual(count, 2)
            for k in a:
                for atom in ["N", "CA", "C", "O"]:
                    if atom in a[k]:
                        np.testing.assert_array_equal(xyz(a[k][atom]), xyz(b[k][atom]))

    def test_invalid_state_rejected(self):
        a = Agent()
        with self.assertRaises(ValueError):
            a.choose(np.zeros(40), np.zeros((2, 40)), [False, False])


if __name__ == "__main__":
    unittest.main()
