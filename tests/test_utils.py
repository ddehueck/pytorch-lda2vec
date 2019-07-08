import sys,os
sys.path.append(os.path.realpath('..'))

import unittest
import torch as t
import utils


class TestUtils(unittest.TestCase):

    def test_get_sparsity_score(self):
        s_vec = t.tensor([0.0, 1.0, 0.0, 0.0])
        u_vec = t.tensor([0.25, 0.25, 0.25, 0.25])
        
        self.assertAlmostEqual(1, utils.get_sparsity_score(s_vec))
        self.assertAlmostEqual(0, utils.get_sparsity_score(u_vec))
                

if __name__ == '__main__':
    unittest.main()
