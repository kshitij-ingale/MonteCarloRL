import unittest
from agent import MonteCarlo
class Parameters:
    env_name = 'Blackjack-v0'

class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        ''' Create class instance and Gym environment instance '''
        p = Parameters()
        self.obj = MonteCarlo(p)

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()