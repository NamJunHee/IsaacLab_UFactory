#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Move line(linear motion)
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

arm.move_gohome(wait=True)

arm.set_position(x=300, y=0, z=150, roll=-180, pitch=0, yaw=0, speed=10, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=300, y=200, z=250, roll=-180, pitch=0, yaw=0, speed=20, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=500, y=200, z=150, roll=-180, pitch=0, yaw=0, speed=30, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=500, y=-200, z=250, roll=-180, pitch=0, yaw=0, speed=40, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=300, y=-200, z=150, roll=-180, pitch=0, yaw=0, speed=50, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=300, y=0, z=250, roll=-180, pitch=0, yaw=0, speed=60, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))

arm.move_gohome(wait=True)

arm.set_position(x=300, y=0, z=150, roll=-3.1415926, pitch=0, yaw=0, speed=10, is_radian=True, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=300, y=200, z=250, roll=-3.1415926, pitch=0, yaw=0, speed=20, is_radian=True, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=500, y=200, z=150, roll=-3.1415926, pitch=0, yaw=0, speed=30, is_radian=True, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=500, y=-200, z=250, roll=-3.1415926, pitch=0, yaw=0, speed=40, is_radian=True, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=300, y=-200, z=150, roll=-3.1415926, pitch=0, yaw=0, speed=50, is_radian=True, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))
arm.set_position(x=300, y=0, z=250, roll=-3.1415926, pitch=0, yaw=0, speed=60, is_radian=True, wait=True)
print(arm.get_position(), arm.get_position(is_radian=True))

arm.move_gohome(wait=True)
arm.disconnect()
