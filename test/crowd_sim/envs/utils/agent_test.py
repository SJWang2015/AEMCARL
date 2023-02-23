import unittest
import configparser
import os
from crowd_sim.envs.utils.human import Human
from test.test_configs.get_configs import get_config


class AgentTest(unittest.TestCase):

    def test_get_scan(self):
        env_config = configparser.RawConfigParser()
        env_config.read(get_config('env.config'))
        humans = []
        x_sign = [1, -1, -1, 1]
        y_sign = [1, 1, -1, -1]
        for i in range(4):
            humans.append(Human(env_config, 'humans'))
            humans[i].set_position([10*x_sign[i], 10*y_sign[i]])

        expected_angles = [0.7853981633974483, 2.356194490192345, -2.356194490192345, -0.7853981633974483]
        expected_distance = 14.142135623730951

        for i in range(4):
            scan = humans[i].get_scan(8, 0, 0)
            self.assertEqual(len(scan), 1)
            self.assertEqual(scan[expected_angles[i]], expected_distance)


if __name__ == '__main__':
    unittest.main()
