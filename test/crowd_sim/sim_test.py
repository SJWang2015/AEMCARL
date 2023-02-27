import unittest
import configparser
import numpy as np
from test.test_configs.get_configs import get_config
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.action import ActionXY, ActionRot


# Integration tests combining multiple components across the simulation
class SimTest(unittest.TestCase):

    def test_move_robot(self):
        # set up config
        env_config = configparser.RawConfigParser()
        env_config.read(get_config('env.config'))

        # set up robot and params
        dt = 0.1
        vx = 1.
        robot = Robot(env_config, 'robot')
        robot.kinematics = 'holonomic'
        robot.set_position([0., 0.])
        robot.time_step = dt

        # step through (holonomically)
        for i in range(10):
            robot.step(ActionXY(vx, 0.))
            self.assertTrue(robot.px - ((i+1) * dt * vx) < 0.0001)  # allow for some accumulation error

    def test_lidar_scanning(self):
        # set up config
        env_config = configparser.RawConfigParser()
        env_config.read(get_config('env.config'))

        # set up env
        dt = 0.1
        num_steps = 5
        robot = Robot(env_config, 'robot')
        robot.set_position([0., 0.])
        robot.theta = 0
        robot.time_step = dt
        scan_res = 360
        humans = []
        human_positions = [[2., 0.], [0., 2.], [0., -2.]]
        for p in human_positions:
            humans.append(Human(env_config, 'humans'))
            humans[-1].set_position(p)

        # keep the action constant throughout sim
        # note this is non-holonomic, v=1m/s r=0.2rad/s
        action = ActionRot(1., 0.2)  # v, r

        # get scans and robot position at each time step
        scans = []
        positions = []
        for i in range(num_steps):
            robot.step(action)  # step through sim
            scans.append(self.scan_lidar(robot, humans, scan_res).tolist())  # take lidar scan
            positions.append([robot.px, robot.py, robot.theta])  # log robot position

    # utility function for getting a lidar scan given a robot and a list of humans
    # TODO: factor this out, most likely into crowd_sim.py
    def scan_lidar(self, robot, humans, res):
        # get scan as a dictionary {angle_index : distance}
        full_scan = {}
        for h in humans:
            scan = h.get_scan(res, robot.px, robot.py)
            for angle in scan:
                if scan[angle] < full_scan.get(angle, np.inf):
                    full_scan[angle] = scan[angle]

        # convert to array of length res, with inf at angles with no reading
        out_scan = np.zeros(res) + np.inf
        for k in full_scan.keys():
            out_scan[k] = full_scan[k]
        return out_scan


if __name__ == '__main__':
    unittest.main()
