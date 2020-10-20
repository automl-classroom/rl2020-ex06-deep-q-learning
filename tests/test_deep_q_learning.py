import unittest
import copy


class TestDeepQLearning(unittest.TestCase):

    def test_deep_q_learning(self):
        from deep_q_learning import q_learning, Q
        Q_before_training = copy.deepcopy(Q)
        q_learning(10)
        for p1, p2 in zip(Q_before_training.parameters(), Q.parameters()):
            assert p1.data.ne(p2.data).sum() > 0


if __name__ == '__main__':
    unittest.main()
