#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Move Arc line(linear arc motion)
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
"""
Just for test example
"""
# if len(sys.argv) >= 2:
#     ip = sys.argv[1]
# else:
#     try:
#         from configparser import ConfigParser
#         parser = ConfigParser()
#         parser.read('../robot.conf')
#         ip = parser.get('xArm', 'ip')
#     except:
#         ip = input('Please input the xArm ip address:')
#         if not ip:
#             print('input error, exit')
#             sys.exit(1)
########################################################
ip = "192.168.1.208"

arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# arm.move_gohome(wait=True)

x_max, x_min, y_max, y_min, z_max, z_min = 750, 50, 600, -600, 1000, 50
arm.set_reduced_tcp_boundary([x_max, x_min, y_max, y_min, z_max, z_min])

arm.set_fense_mode(True)

paths = [
    [300, 0, 150, -180, 0, 0],
    [300, 200, 250, -180, 0, 0],
    [500, 200, 150, -180, 0, 0],
    [500, -200, 250, -180, 0, 0],
    [300, -200, 150, -180, 0, 0],
    [300, 0, 250, -180, 0, 0],
    [300, 200, 350, -180, 0, 0],
    [500, 200, 250, -180, 0, 0],
    [500, -200, 350, -180, 0, 0],
    [300, -200, 250, -180, 0, 0],
    [300, 0, 350, -180, 0, 0],
]

arm.set_position(*paths[0], wait=True)
_, angles = arm.get_servo_angle()
arm.set_pause_time(0.2)


def move():
    ret = arm.set_servo_angle(angle=angles, speed=100, wait=False)
    if ret < 0:
        print('set_servo_angle, ret={}'.format(ret))
        return -1
    for path in paths:
        ret = arm.set_position(*path[:6], radius=0, wait=False, speed=100)
        if ret < 0:
            print('set_position, ret={}'.format(ret))
            return -1
    return 0

for i in range(10):
    if move() != 0:
        break

# arm.move_gohome(wait=True)
arm.disconnect()
