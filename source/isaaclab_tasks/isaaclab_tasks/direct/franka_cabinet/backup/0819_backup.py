# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from isaaclab.sensors import CameraCfg, Camera
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from builtin_interfaces.msg import Time

# from PIL import Image
import cv2
import numpy as np
from enum import Enum
import kornia
import math
import scipy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Point

from cv_bridge import CvBridge
import threading
import time

import pykinect_azure as pykinect
from xarm.wrapper import XArmAPI
from pykinect_azure.k4a import _k4a
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
from pyk4a.calibration import CalibrationType

import ctypes

import socket
import struct
import os

class RobotType(Enum):
    FRANKA = "franka"
    UF = "ufactory"
    DOOSAN = "doosan"
robot_type = RobotType.UF

class ObjectMoveType(Enum):
    STATIC = "static"
    CIRCLE = "circle"
    LINEAR = "linear"
    STOP = "stop"
object_move = ObjectMoveType.STATIC
# object_move = ObjectMoveType.LINEAR
# object_move = ObjectMoveType.STOP

class CameraType(Enum):
    Sim = "sim"
    Azure = "azure"
camera_type = CameraType.Sim

training_mode = False

foundationpose_mode = False
yolo_mode = False

camera_enable = True
image_publish = True

robot_action = False
robot_init_pose = False
robot_fix = False

init_reward = True
UFactory_set_mode = False

add_episode_length = 200
# add_episode_length = 600
# add_episode_length = -400 # 초기 학습 시 episode 길이

vel_ratio = 0.10

@configclass
class FrankaObjectTrackingEnvCfg(DirectRLEnvCfg):
    ## env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    
    if robot_type == RobotType.FRANKA:
        action_space = 9
        observation_space = 23
        
    elif robot_type == RobotType.UF:
        # action_space = 12
        # observation_space = 29

        action_space = 6
        observation_space = 17
        
    elif robot_type == RobotType.DOOSAN:
        action_space = 8
        observation_space = 21
    
    state_space = 0

    ## simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    ## scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    ## robot
    Franka_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1":  0.000,
                "panda_joint2": -0.831,
                "panda_joint3": -0.000,
                "panda_joint4": -1.796,
                "panda_joint5": -0.000,
                "panda_joint6":  1.733,
                "panda_joint7":  0.707,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                # velocity_limit=2.175,
                velocity_limit=0.22,
                stiffness=80.0,
                # stiffness=200.0,
                # damping=4.0,
                damping=25.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                # velocity_limit=2.61,
                velocity_limit=0.22,
                stiffness=80.0,
                # stiffness=200.0,
                # damping=4.0,
                damping=25.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )
    
    UF_robot = ArticulationCfg(
        # prim_path="/World/envs/env_.*/xarm6_with_gripper",
        prim_path="/World/envs/env_.*/xarm6",
        spawn=sim_utils.UsdFileCfg(
            # usd_path="/home/nmail-robot/IsaacLab/ROBOT/xarm6_with_gripper/xarm6_with_gripper.usd",
            usd_path="/home/nmail-robot/IsaacLab/ROBOT/xarm6_robot_white/xarm6_robot_white.usd",
            
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=24, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1":  0.00,
                "joint2": -1.22,
                # "joint3": -0.78,
                "joint3": -0.50,
                "joint4":  0.00,
                "joint5":  1.30,
                # "joint5":  0.80,
                "joint6":  0.00,
                # "left_finger_joint" : 0.0,
                # "right_finger_joint": 0.0
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ufactory_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit=87.0,
                
                velocity_limit=2.61 * vel_ratio,
                stiffness=2000.0,
                damping=100.0,
                
                # velocity_limit=0.8,
                # stiffness=80.0,
                # damping=18.0,
            ),
            "ufactory_forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint4", "joint5", "joint6"],
                effort_limit=12.0,
                
                velocity_limit=2.61 * vel_ratio,
                stiffness=2000.0,
                damping=100.0,
                
                # velocity_limit=0.8,
                # stiffness=80.0,
                # damping=18.0,
            ),
            # "ufactory_hand": ImplicitActuatorCfg(
            #     joint_names_expr=["left_finger_joint", "right_finger_joint"],
            #     effort_limit=200.0,
            #     velocity_limit=0.2,
            #     stiffness=2e3,
            #     damping=1e2,
            # ),
        },
    )

    Doosan_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Doosan_M1013",
        # prim_path="/World/envs/env_.*/m1013",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/nmail-robot/IsaacLab/ROBOT/Doosan_M1013/M1013_onrobot_with_gripper/M1013_onrobot.usda",
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/Doosan_M1013/m1013_with_gripper/m1013_with_gripper.usd",
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/Doosan_M1013/m1013/m1013.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "J1_joint":  0.00,
                "J2_joint": -0.60,
                "J3_joint":  1.80,
                "J4_joint":  0.00,
                "J5_joint":  1.25,
                "J6_joint":  0.00,
                "left_joint" : 0.0,
                "right_joint": 0.0
                
                # "joint1":  0.00,
                # "joint2": -0.60,
                # "joint3":  1.80,
                # "joint4":  0.00,
                # "joint5":  1.25,
                # "joint6" : 0.00,
                # "left_joint" : 0.0,
                # "right_joint": 0.0
            },
            pos=(1.0, 0.0, 0.05),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "doosan_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["J1_joint", "J2_joint", "J3_joint"],
                # joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit=87.0,
                # velocity_limit=2.175,
                velocity_limit=0.25,
                stiffness=20.0,
                # stiffness=200.0,
                # damping=4.0,
                damping=30.0,
            ),
            "doosan_forearm": ImplicitActuatorCfg(
                joint_names_expr=["J4_joint", "J5_joint", "J6_joint"],
                # joint_names_expr=["joint4", "joint5", "joint6"],
                effort_limit=12.0,
                # velocity_limit=2.61,
                velocity_limit=0.25,
                stiffness=20.0,
                # stiffness=200.0,
                # damping=4.0,
                damping=30.0,
            ),
            "doosan_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_joint", "right_joint"],
                effort_limit=200.0,
                velocity_limit=0.3,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    ## camera
    if camera_enable:
        if robot_type == RobotType.FRANKA:
            camera = CameraCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand/hand_camera", 
                update_period=0.03,
                height=480,
                width=640,
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=25.0, # 값이 클수록 확대
                    focus_distance=60.0,
                    horizontal_aperture=50.0,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.05),
                    rot=(0.0, 0.707, 0.707, 0.0),
                    convention="ROS",
                )
            )
            
        elif robot_type == RobotType.UF:
            camera = CameraCfg(
                # prim_path="/World/envs/env_.*/xarm6_with_gripper/link6/hand_camera", 
                prim_path="/World/envs/env_.*/xarm6/link6/hand_camera",
                update_period=0.03,
                height=480,
                width=640,
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=10.0, # 값이 클수록 확대
                    focus_distance=60.0,
                    horizontal_aperture=50.0,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.1),
                    rot=(0.0, 0.707, 0.707, 0.0),
                    convention="ROS",
                )
            )
            
        elif robot_type == RobotType.DOOSAN:
            camera = CameraCfg(
                # prim_path="/World/envs/env_.*/Doosan_M1013/gripper/onrobot_2fg_14/base/hand_camera", 
                prim_path="/World/envs/env_.*/Doosan_M1013/J6/hand_camera", 
                # prim_path="/World/envs/env_.*/m1013/link6/hand_camera", 
                update_period=0.03,
                height=480,
                width=640,
                data_types=["rgb", "depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=15.0, # 값이 클수록 확대
                    focus_distance=60.0,
                    horizontal_aperture=50.0,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 1.5),
                    # rot=(-0.5, 0.5, -0.5, -0.5), #ROS
                    # rot=(-0.5, -0.5, -0.5, 0.5), #ros
                    rot=(0.0, -0.707, 0.707, 0.0),
                    convention="ROS",
                )
            )
    
    ## cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    ## ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    ## cube
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0, 0.055), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",ee
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                ),
        ),
    )

    ## mustard
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/base_link",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0, 0.05), rot=(0.923, 0, 0, -0.382)),
        spawn=UsdFileCfg(
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/003_cracker_box/003_cracker_box.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/005_tomato_soup_can/005_tomato_soup_can.usd",
                usd_path="/home/nmail-robot/objects_usd/google_objects_usd/006_mustard_bottle/006_mustard_bottle.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/004_sugar_box/004_sugar_box.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/025_mug/025_mug.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/Travel_Mate_P_series_Notebook/Travel_Mate_P_series_Notebook.usd",
                # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/objects_usd/google_objects_usd/Mens_ASV_Billfish_Boat_Shoe_in_Dark_Brown_Leather_zdHVHXueI3w/Mens_ASV_Billfish_Boat_Shoe_in_Dark_Brown_Leather_zdHVHXueI3w.usd",
                
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                    kinematic_enabled = True,
                ),
            ),
    )
    
    # action_scale = 7.5
    # dof_velocity_scale = 0.1
    action_scale = 2.0
    dof_velocity_scale = 0.05

    # reward scales
    # dist_reward_scale = 1.5
    # rot_reward_scale = 1.5
    # open_reward_scale = 10.0
    # action_penalty_scale = 0.05
    # finger_reward_scale = 2.0
    
    #time
    current_time = 0.0

class FrankaObjectTrackingEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaObjectTrackingEnvCfg

    def __init__(self, cfg: FrankaObjectTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # 학습 초기 (좁은 범위)
        self.rand_pos_range = {
            # "x" : ( 0.7,  1.0),
            # "y" : ( -0.001,  0.001),
            # "z" : (  0.75, 0.75),

            # "x" : (  0.30, 0.70),
            # "y" : ( -0.35, 0.35),
            # "z" : (  0.10, 0.60),
                
            "x" : (  0.30,  0.50),
            "y" : ( -0.30,  0.30),
            "z" : (  0.05,  0.30)
        }
        
        ## 학습 후기 (넓은 범위)
        # self.rand_pos_range = {
        #     "x" : ( -0.20,  0.35),
        #     "y" : ( -0.35,  0.35),
        #     "z" : (  0.055, 0.3)
        # }
        
        ## Doosan (넓은 범위)
        if robot_type == RobotType.DOOSAN:
            self.rand_pos_range = {
                # "x" : ( -0.10,  -0.10),
                # "y" : ( -0.001,  0.001),
                # "z" : (  0.1, 0.1),

                # "x" : ( -0.15,  0.30),
                # "y" : ( -0.30,  0.30),
                # "z" : (  0.055, 0.3)
            }
            
        if robot_type == RobotType.FRANKA:
            self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
            "panda_finger_joint1", "panda_finger_joint2"
            ]
            self.joint_init_values = [0.000, -0.831, 0.000, -1.796, 0.000, 2.033, 0.707, 0.035, 0.035]
        elif robot_type == RobotType.UF:
            self.joint_names = [
            "joint1", "joint2", "joint3", "joint4","joint5", "joint6", ]
            # self.joint_init_values = [0.000, -1.220, -0.780, -0.000, 0.800, 0.000]
            # self.joint_init_values = [0.000, -1.220, -0.780, -0.000, 1.300, 0.000]
            self.joint_init_values = [0.000, -1.220, -0.50, -0.000, 1.300, 0.000]
        elif robot_type == RobotType.DOOSAN:
            self.joint_names = [
            "J1_joint", "J2_joint", "J3_joint", "J4_joint","J5_joint", "J6_joint" ]
            # "joint1", "joint2", "joint3","joint4", "joint5","joint6" ]
            self.joint_init_values = [0.000, -0.600, 1.800, 0.000, 1.250, 0.000] 

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        stage = get_current_stage()
        
        if robot_type == RobotType.FRANKA:
            self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
            self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
            
            hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
            )
            lfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
                self.device,
            )
            rfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
                self.device,
            )
            self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
            self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
            self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
            
        elif robot_type == RobotType.UF:
            # self.robot_dof_speed_scales[self._robot.find_joints("left_finger_joint")[0]] = 0.1
            # self.robot_dof_speed_scales[self._robot.find_joints("right_finger_joint")[0]] = 0.1
            
            hand_pose = get_env_local_pose(
                self.scene.env_origins[0],
                # UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6_with_gripper/link6")),
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
                self.device,
            )
            lfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                # UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6_with_gripper/left_finger")),
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
                self.device,
            )
            rfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                # UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6_with_gripper/right_finger")),
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/xarm6/link6")),
                self.device,
            )
            self.hand_link_idx = self._robot.find_bodies("link6")[0][0]
            # self.left_finger_link_idx = self._robot.find_bodies("left_finger")[0][0]
            # self.right_finger_link_idx = self._robot.find_bodies("right_finger")[0][0]
             
        elif robot_type == RobotType.DOOSAN:
            
            self.robot_dof_speed_scales[self._robot.find_joints("left_joint")[0]] = 0.1
            self.robot_dof_speed_scales[self._robot.find_joints("right_joint")[0]] = 0.1
            
            hand_pose = get_env_local_pose(
                self.scene.env_origins[0],
                # UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Doosan_M1013/gripper/onrobot_2fg_14/base")),
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Doosan_M1013/J6")),
                self.device,
            )
            lfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Doosan_M1013/gripper/onrobot_2fg_14/Left")),
                self.device,
            )
            rfinger_pose = get_env_local_pose(
                self.scene.env_origins[0],
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Doosan_M1013/gripper/onrobot_2fg_14/Right")),
                self.device,
            )
            # self.hand_link_idx = self._robot.find_bodies("base")[0][0]
            self.hand_link_idx = self._robot.find_bodies("J6")[0][0]
            self.left_finger_link_idx = self._robot.find_bodies("Left")[0][0]
            self.right_finger_link_idx = self._robot.find_bodies("Right")[0][0]
        
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))
        
        box_local_pose = torch.tensor([0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.box_local_pos = box_local_pose[0:3].repeat((self.num_envs, 1))
        self.box_local_rot = box_local_pose[3:7].repeat((self.num_envs, 1))

        if robot_type == RobotType.FRANKA or robot_type == RobotType.UF:
            self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
                (self.num_envs, 1)
            )
        elif robot_type == RobotType.DOOSAN:
            self.gripper_forward_axis = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(
                (self.num_envs, 1)
            )
            
        self.gripper_up_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        
        # self.cube_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
        #     (self.num_envs,1)
        # )
        self.box_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs,1)
        )
        
        # self.cube_idx = self._cube.find_bodies("cube")[0][0]
        self.box_idx = self._box.find_bodies("base_link")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # self.cube_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # self.cube_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.cube_center = self._cube.data.body_link_pos_w[:,0,:].clone()
        
        self.box_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.box_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.box_center = self._box.data.body_link_pos_w[:,0,:].clone()
        
        self.box_pos_cam = torch.zeros((self.num_envs, 4), device=self.device)
        
        self.fixed_z = 0.055
        
        self.current_box_pos = None
        self.current_box_rot = None
        
        self.target_box_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["x"][1] - self.rand_pos_range["x"][0]) + self.rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["y"][1] - self.rand_pos_range["y"][0]) + self.rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["z"][1] - self.rand_pos_range["z"][0]) + self.rand_pos_range["z"][0],
            ], dim = 1)
        
        # self.target_box_pos = self.target_box_pos + self.box_center
        self.target_box_pos = self.target_box_pos + self.scene.env_origins

        # self.rand_pos_step = 0
        # self.new_box_pos_rand = self._box.data.body_link_pos_w[:,0,:].clone()
        
        # self.obj_speed = 0.0005
        self.obj_speed = 0.001
        # self.obj_speed = 0.002
        # self.obj_speed = 0.0025
        
        rclpy.init()
        self.last_publish_time = 0.0
        self.position_error = 0.0
        self.obj_origin_distance = 0.0
        self.out_of_fov_cnt = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        if image_publish:
            qos_profile = QoSProfile(depth=10)
            qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
            
            # self.node = rclpy.create_node('isaac_camera_publisher')
            # self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/isaac_camera_info_rect',10)
            # self.rgb_publisher = self.node.create_publisher(Image, '/isaac_image_rect',10)
            # self.depth_publisher = self.node.create_publisher(Image, '/isaac_depth',10)

            self.node = rclpy.create_node('camera_publisher')
            self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/camera_info_rect',10)
            self.rgb_publisher = self.node.create_publisher(Image, '/image_rect',10)
            self.depth_publisher = self.node.create_publisher(Image, '/depth',10)
            
            self.bridge = CvBridge()

            if camera_type == CameraType.Azure:
                print("[IsaacLab] Initializing Azure camera...")
                # pykinect.initialize_libraries()
                # self.device_config = pykinect.default_configuration

                # self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
                # self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
                # self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        
                # self.azure_device = pykinect.start_device(config=self.device_config)

                # self.k4a = PyK4A(
                #     Config(
                #         color_resolution=ColorResolution.RES_720P,
                #         depth_mode=DepthMode.NFOV_UNBINNED,
                #         synchronized_images_only=True,
                #     )
                # )
                # self.k4a.start()
                
        if foundationpose_mode:
            self.foundationpose_msg = None
            self.foundationpose_node = rclpy.create_node('foundationpose_receiver')
            self.foundationpose_node.create_subscription(
                Point,
                '/object_position',
                self.foundationpose_callback,
                10
            )

        if yolo_mode:
            print("[IsaacLab] Initializing YOLO receiver node...")
            self.yolo_msg = None
            self.yolo_node = rclpy.create_node('yolo_receiver')
            self.yolo_node.create_subscription(
                Point,
                '/yolo/point',
                self.yolo_callback,
                10
            )
        
        self.init_cnt = 0

        if UFactory_set_mode:
            ip = "192.168.1.208"
            self.arm = XArmAPI(ip)

            self.arm.motion_enable(enable=True)
            self.arm.set_mode(6) ## joint Online Trajectory Planning
            # self.arm.set_mode(0) ## position Control Mode
            self.arm.set_state(state=0)

            x_max, x_min, y_max, y_min, z_max, z_min = 750, 50, 600, -600, 1000, 50
            self.arm.set_reduced_tcp_boundary([x_max, x_min, y_max, y_min, z_max, z_min])
            self.arm.set_fense_mode(True)
        
    def publish_camera_data(self):
        env_id = 0
        
        current_stamp = self.node.get_clock().now().to_msg() 
        current_stamp.sec = current_stamp.sec % 50000
        current_stamp.nanosec = 0
        
        # current_stamp = Time()
        # current_stamp.sec = 1
        # current_stamp.nanosec = 0
                
        if image_publish:
            # if camera_type == CameraType.Sim:
            #     rgb_data = self._camera.data.output["rgb"]
            #     depth_data = self._camera.data.output["depth"]

            #     rgb_image = (rgb_data.cpu().numpy()[env_id]).astype(np.uint8)
            #     depth_image = (depth_data.cpu().numpy()[env_id]).astype(np.float32)
            #     depth_image[depth_image > 1.5] = 1.5

            #     rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
            #     depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
            
            rgb_data = self._camera.data.output["rgb"]
            depth_data = self._camera.data.output["depth"]
            rgb_image = (rgb_data.cpu().numpy()[env_id]).astype(np.uint8)
            depth_image = (depth_data.cpu().numpy()[env_id]).astype(np.float32)
            depth_image[depth_image > 1.5] = 1.5
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
            
            # Publish Camera Info
            camera_info_msg = CameraInfo()
            camera_info_msg.header.stamp = current_stamp
            
            camera_info_msg.header.frame_id = 'tf_camera'
        
            camera_info_msg.height = 480 
            camera_info_msg.width = 640 
            
            camera_info_msg.distortion_model = 'plumb_bob'
        
            intrinsic_matrices = self._camera.data.intrinsic_matrices.cpu().numpy().flatten().tolist()
            camera_info_msg.k = intrinsic_matrices[:9]
            camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info_msg.r = [1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0]
            camera_info_msg.p = intrinsic_matrices[:3] + [0.0] + intrinsic_matrices[3:6] + [0.0] + [0.0, 0.0, 1.0, 0.0]

            camera_info_msg.binning_x = 0
            camera_info_msg.binning_y = 0

            camera_info_msg.roi.x_offset = 0
            camera_info_msg.roi.y_offset = 0
            camera_info_msg.roi.height = 0
            camera_info_msg.roi.width = 0
            camera_info_msg.roi.do_rectify = False
        
            self.camera_info_publisher.publish(camera_info_msg)
        
            # Publish RGB Image
            # rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
            rgb_msg.header.stamp = current_stamp
            rgb_msg.header.frame_id = 'tf_camera'
            self.rgb_publisher.publish(rgb_msg)

            # Publish Depth Image
            # depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
            depth_msg.header.stamp = current_stamp
            depth_msg.header.frame_id = 'tf_camera'
            self.depth_publisher.publish(depth_msg)
            depth_msg.step = depth_image.shape[1] * 4
    
    def subscribe_foundationpose(self):
        msg = self.foundationpose_msg
        
        if msg is None:
            return None

        return torch.tensor([msg.x, msg.y, msg.z], device=self.device)
    
    def subscribe_yolo(self):
        msg = self.yolo_msg
        
        if msg is None:
            return None

        return torch.tensor([msg.x, msg.y, msg.z], device=self.device)
        
    def foundationpose_callback(self,msg):
        self.foundationpose_msg = msg
    
    def yolo_callback(self,msg):
        self.yolo_msg = msg

    def quat_mul(self, q, r):
        x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        quat = torch.stack((x, y, z, w), dim=-1)
        # return kornia.geometry.quaternion.normalize_quaternion(quat)
        return kornia.geometry.conversions.normalize_quaternion(quat)
    
    def quat_conjugate(self, q):
        q_conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
        return q_conj
    
    def get_real_hand_pose(self):

        code, pose_mm_deg = self.arm.get_position(is_radian=False)

        if code != 0:
            print(f"Error: 실제 로봇 TCP 자세를 읽는 데 실패했습니다. 오류 코드: {code}")
            return None, None

        pos_m = [p / 1000.0 for p in pose_mm_deg[:3]]

        roll_deg, pitch_deg, yaw_deg = pose_mm_deg[3:]

        r = scipy.spatial.transform.Rotation.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
        quat_xyzw = r.as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        hand_pos_real = torch.tensor([pos_m], device=self.device, dtype=torch.float32)
        hand_rot_real = torch.tensor([quat_wxyz], device=self.device, dtype=torch.float32)

        return hand_pos_real, hand_rot_real

    def compute_camera_world_pose(self, hand_pos, hand_rot):
        if robot_type == RobotType.FRANKA:
            cam_offset_pos = torch.tensor([0.0, 0.3, 0.06], device=hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.0, 0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
        elif robot_type == RobotType.UF:
            if camera_type == CameraType.Sim:
                cam_offset_pos = torch.tensor([0.0, 0.0, 0.1], device=hand_pos.device).repeat(self.num_envs, 1)
                q_cam_in_hand = torch.tensor([0.0, 0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
            elif camera_type == CameraType.Azure:
                cam_offset_pos = torch.tensor([[0.1013, 0.0233, 0.0606]], device=hand_pos.device).repeat(self.num_envs, 1)
                q_cam_in_hand = torch.tensor([[0.7170, -0.0334, -0.0516, 0.6942]], device=hand_pos.device).repeat(self.num_envs, 1)

                # cam_offset_pos = torch.tensor([-0.02, 0.03, 0.08], device=hand_pos.device).repeat(self.num_envs, 1)
                # cam_offset_pos = torch.tensor([0.00, 0.03, 0.06], device=hand_pos.device).repeat(self.num_envs, 1)
                # q_cam_in_hand = torch.tensor([0.0, 0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
                # q_cam_in_hand = torch.tensor([0.707, 0.707, 0.0, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
        elif robot_type == RobotType.DOOSAN:
            cam_offset_pos = torch.tensor([0.0, 0.0, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
            # q_cam_in_hand = torch.tensor([-0.5, 0.5, -0.5, -0.5], device=hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.0, -0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)

        hand_rot_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(hand_rot)
        cam_offset_pos_world = torch.bmm(hand_rot_matrix, cam_offset_pos.unsqueeze(-1)).squeeze(-1)

        camera_pos_w = hand_pos + cam_offset_pos_world
        camera_pos_w = camera_pos_w - self.scene.env_origins
        
        # camera_rot_w = self.quat_mul(hand_rot, q_cam_in_hand)
        # camera_rot_w = hand_rot
        camera_rot_w = self.robot_grasp_rot
        
        return camera_pos_w, camera_rot_w

    def world_to_camera_pose(self, camera_pos_w, camera_rot_w, obj_pos_w, obj_rot_w):
        rel_pos = obj_pos_w - camera_pos_w

        cam_rot_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(camera_rot_w)
        
        obj_pos_cam = torch.bmm(cam_rot_matrix.transpose(1, 2), rel_pos.unsqueeze(-1)).squeeze(-1)

        cam_rot_inv = self.quat_conjugate(camera_rot_w)
        obj_rot_cam = self.quat_mul(cam_rot_inv, obj_rot_w)

        return obj_pos_cam, obj_rot_cam
    
    def camera_to_world_pose(self, camera_pos_w, camera_rot_w, obj_pos_cam, obj_rot_cam):
        cam_rot_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(camera_rot_w)
        
        obj_pos_world = torch.bmm(cam_rot_matrix, obj_pos_cam.unsqueeze(-1)).squeeze(-1) + camera_pos_w
        obj_rot_world = self.quat_mul(camera_rot_w, obj_rot_cam)
        
        return obj_pos_world, obj_rot_world
    
    # def recv_exact(self, sock, n_bytes):
    #     buf = b''
    #     while len(buf) < n_bytes:
    #         chunk = sock.recv(n_bytes - len(buf))
    #         if not chunk:
    #             return None
    #         buf += chunk
    #     return buf
        
    def _setup_scene(self):
        
        if robot_type == RobotType.FRANKA:
            self._robot = Articulation(self.cfg.Franka_robot)
        elif robot_type == RobotType.UF:
            self._robot = Articulation(self.cfg.UF_robot)
        elif robot_type == RobotType.DOOSAN:
            self._robot = Articulation(self.cfg.Doosan_robot)
    
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # 카메라 추가
        if camera_enable:
            self._camera = Camera(self.cfg.camera)
            self.scene.sensors["hand_camera"] = self._camera
        
        # 큐브 추가
        # self._cube = RigidObject(self.cfg.cube)
        # self.scene.rigid_objects["cube"] = self._cube
        
        # 상자 추가
        self._box = RigidObject(self.cfg.box)
        self.scene.rigid_objects["base_link"] = self._box

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.cfg.current_time = self.cfg.current_time + self.dt
        current_time = torch.tensor(self.cfg.current_time, device=self.device, dtype=torch.float32)
        
        # 카메라 ros2 publish----------------------------------------------------------------------------------------------
        if image_publish:   
            self.last_publish_time += self.dt
            if self.last_publish_time >= (1.0 / 15.0):  # 30fps 기준
                self.publish_camera_data()
                rclpy.spin_once(self.node, timeout_sec=0.001)
                self.last_publish_time = 0.0

        # 물체 원 운동 (실제 운동 제어 코드)-------------------------------------------------------------------------------------------
        if object_move == ObjectMoveType.CIRCLE:
            R = 0.10
            omega = 0.7 # Speed

            offset_x = R * torch.cos(omega * current_time) - 0.35
            offset_y = R * torch.sin(omega * current_time) 
            offset_z = 0.055

            offset_pos = torch.tensor([offset_x, offset_y, offset_z], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

            new_box_pos_circle = self.box_center + offset_pos
            new_box_rot_circle = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)

            new_box_pose_circle = torch.cat([new_box_pos_circle, new_box_rot_circle], dim = -1)

            self._box.write_root_pose_to_sim(new_box_pose_circle)
        
        # 물체 위치 랜덤 선형 이동 --------------------------------------------------------------------------------------------------
        if object_move == ObjectMoveType.LINEAR:
            distance_to_target = torch.norm(self.target_box_pos - self.new_box_pos_rand, p=2, dim = -1)
            if torch.any(distance_to_target < 0.01):
                self.target_box_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["x"][1] - self.rand_pos_range["x"][0]) + self.rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["y"][1] - self.rand_pos_range["y"][0]) + self.rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["z"][1] - self.rand_pos_range["z"][0]) + self.rand_pos_range["z"][0],
                ], dim = 1)

                self.target_box_pos = self.target_box_pos + self.scene.env_origins

                self.current_box_pos = self._box.data.body_link_pos_w[:, 0, :].clone()
                self.current_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()

                self.new_box_pos_rand = self.current_box_pos

                direction = self.target_box_pos - self.current_box_pos
                direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
                self.rand_pos_step = (direction / direction_norm * self.obj_speed)

            self.new_box_pos_rand = self.new_box_pos_rand + self.rand_pos_step
            new_box_rot_rand = self.current_box_rot 

            if self.new_box_pos_rand is not None and new_box_rot_rand is not None:
                new_box_pose_rand = torch.cat([self.new_box_pos_rand, new_box_rot_rand], dim=-1)
            else:
                raise ValueError("self.new_box_pos_rand or new_box_rot_rand is None")
            self._box.write_root_pose_to_sim(new_box_pose_rand)
        
    def _apply_action(self):
        
        global robot_action
        global robot_init_pose
        
        target_pos = self.robot_dof_targets.clone()
        # print(f"target_pos: {target_pos}")
        
        if robot_type == RobotType.FRANKA:
            joint3_index = self._robot.find_joints(["panda_joint3"])[0]
            joint5_index = self._robot.find_joints(["panda_joint5"])[0]
            joint7_index = self._robot.find_joints(["panda_joint7"])[0]
            target_pos[:, joint3_index] = 0.0
            target_pos[:, joint5_index] = 0.0
            target_pos[:, joint7_index] = 0.0
        elif robot_type == RobotType.UF:
            joint4_index = self._robot.find_joints(["joint4"])[0]
            joint6_index = self._robot.find_joints(["joint6"])[0]
            target_pos[:, joint4_index] = 0.0
            target_pos[:, joint6_index] = 0.0
            target_pos[:, 7:] = 0.0
        elif robot_type == RobotType.DOOSAN:
            joint4_index = self._robot.find_joints(["J4_joint"])[0]
            joint6_index = self._robot.find_joints(["J6_joint"])[0]
            # joint4_index = self._robot.find_joints(["joint4"])[0]
            # joint6_index = self._robot.find_joints(["joint6"])[0]
            target_pos[:, joint4_index] = 0.0
            target_pos[:, joint6_index] = 0.0
        
        if training_mode == False and robot_fix == False:

            if robot_action and robot_init_pose:
                self._robot.set_joint_position_target(target_pos)
                ##----xarm python API------
                if UFactory_set_mode:
                    # print("target_pos :", target_pos)
                    xarm_actions = self._robot.data.joint_pos[:, :6]

                    if robot_type == RobotType.UF:
                        joint4_index = self._robot.find_joints(["joint4"])[0]
                        joint6_index = self._robot.find_joints(["joint6"])[0]
                        xarm_actions[:, joint4_index] = 0.0
                        xarm_actions[:, joint6_index] = 0.0

                    # print("xarm_actions :", xarm_actions)
                    angle_cmd = xarm_actions.detach().cpu().numpy().flatten().tolist()

                    ang_speed = 200
                    angmvacc = 100.0

                    rad_speed = math.radians(ang_speed)
                    rad_mvacc = math.radians(angmvacc)

                    self.arm.set_servo_angle(angle=angle_cmd, speed=rad_speed, wait=False, is_radian=True, mvacc = rad_mvacc)

                    # print("self.box_grasp_pos : ", self.box_grasp_pos)

                    # x = float(self.box_grasp_pos[0, 0].cpu()) * 1000
                    # y = float(self.box_grasp_pos[0, 1].cpu()) * 1000
                    # z = float(self.box_grasp_pos[0, 2].cpu()) * 1000

                    # self.arm.set_position(x=x, y=y, z=z, roll=-180, pitch=0, yaw=0, speed=200, wait=False)

            elif robot_action == False and robot_init_pose == False:
                init_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

                for name, val in zip(self.joint_names, self.joint_init_values):
                    index = self._robot.find_joints(name)[0]
                    init_pos[:, index] = val

                self._robot.set_joint_position_target(init_pos)

                joint_err = torch.abs(self._robot.data.joint_pos - init_pos)
                max_err = torch.max(joint_err).item()
                
                if foundationpose_mode:
                    pos = self.subscribe_foundationpose()
                    if (max_err < 0.3) and (pos is not None):
                        self.init_cnt += 1
                        print(f"init_cnt : {self.init_cnt}")
                        
                        if self.init_cnt > 300: 
                            robot_action = True
                            robot_init_pose = True
                
                elif yolo_mode :
                    print("[IsaacLab] Waiting for YOLO detection...")
                    pos = self.subscribe_yolo()
                    print(f"[IsaacLab] YOLO detected position: {pos}")
                    if (max_err < 0.3) and (pos is not None):
                        self.init_cnt += 1
                        print(f"init_cnt : {self.init_cnt}")
                              
                        if self.init_cnt > 300: 
                            robot_action = True
                            robot_init_pose = True
                            
                elif foundationpose_mode == False and yolo_mode == False and max_err < 0.3:
                    self.init_cnt += 1
                    print(f"init_cnt : {self.init_cnt}")
                    if self.init_cnt > 300:
                        robot_init_pose = True
                        robot_action = True
                
                               
        elif training_mode == True and robot_fix == False:
            if robot_init_pose == False:
                init_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
                for name, val in zip(self.joint_names, self.joint_init_values):
                    index = self._robot.find_joints(name)[0]
                    init_pos[:, index] = val

                self._robot.set_joint_position_target(init_pos)

                joint_err = torch.abs(self._robot.data.joint_pos - init_pos)
                max_err = torch.max(joint_err).item()
                
                if max_err < 0.3:
                    robot_init_pose = True
                    robot_action = True
                
            elif robot_init_pose:
                self._robot.set_joint_position_target(target_pos)
                

        
    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
                
        if training_mode or object_move == ObjectMoveType.CIRCLE:

            terminated = 0
            truncated = self.episode_length_buf >= self.max_episode_length + add_episode_length
        else:
            terminated = 0
            truncated = self.episode_length_buf >= self.max_episode_length #- 400 # 물체 램덤 생성 환경 초기화 주기
        
        #환경 고정
        # terminated = 0
        # truncated = 0
        
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        
        # robot_left_finger_pos = self._robot.data.body_link_pos_w[:, self.left_finger_link_idx]
        # robot_right_finger_pos = self._robot.data.body_link_pos_w[:, self.right_finger_link_idx]

        # camera_pos_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
        # camera_rot_w = self.robot_grasp_rot
        
        if camera_type == CameraType.Azure:
            hand_pos_real, hand_rot_real = self.get_real_hand_pose()
            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(hand_pos_real, hand_rot_real)
        elif camera_type == CameraType.Sim:
            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)

        self.box_pos_cam, box_rot_cam = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,
        )

        # print(f"hand_pos_real: {hand_pos_real}")
        # print(f"hand_rot_real: {hand_rot_real}")
        # print(f"self.robot_grasp_pos: {self.robot_grasp_pos}")
        # print(f"self.robot_grasp_rot: {self.robot_grasp_rot}")

        # print("*" * 20)
        # print(f"box_pos_cam: {self.box_pos_cam}")
        # print(f"self._robot.data.joint_pos : {self._robot.data.joint_pos}")
        
        return self._compute_rewards(
            self.actions,
            self.robot_grasp_pos,
            self.box_grasp_pos,
            self.robot_grasp_rot,
            self.box_grasp_rot,
            self.box_pos_cam,
            box_rot_cam,
            self.gripper_forward_axis,
            self.gripper_up_axis,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        
        # robot state ---------------------------------------------------------------------------------
        if training_mode:
            # joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            #     -0.125,
            #     0.125,
            #     (len(env_ids), self._robot.num_joints),
            #     self.device,
            # )

            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            joint_vel = torch.zeros_like(joint_pos) ## 여기 뭔가 다르다
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:
            # 최초 한 번만 실행
            if not hasattr(self, "_initialized"):
                self._initialized = False

            if not self._initialized:
                joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
                -0.125,
                0.125,
                (len(env_ids), self._robot.num_joints),
                self.device,
                )
                joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
                joint_vel = torch.zeros_like(joint_pos)
                self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
                self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
                self._initialized = True
        
        # 물체 원 운동 (원 운동 시 환경 초기화 코드)------------------------------------------------------------------------------------------------------------
        reset_pos = self.box_center
        reset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
        reset_box_pose = torch.cat([reset_pos, reset_rot], dim = -1)
        
        if object_move == ObjectMoveType.CIRCLE:
            self._box.write_root_pose_to_sim(reset_box_pose)
        
        # 물체 위치 랜덤 생성 (Static) (실제 물체 생성 코드) -----------------------------------------------------------------------------------------------------------
        self.rand_pos = torch.stack([
            torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["x"][1] - self.rand_pos_range["x"][0]) + self.rand_pos_range["x"][0],
            torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["y"][1] - self.rand_pos_range["y"][0]) + self.rand_pos_range["y"][0],
            torch.rand(self.num_envs, device=self.device) * (self.rand_pos_range["z"][1] - self.rand_pos_range["z"][0]) + self.rand_pos_range["z"][0],
        ], dim = 1)
        
        rand_reset_pos = self.rand_pos + self.box_center
        
        random_angles = torch.rand(self.num_envs, device=self.device) * 2 * torch.pi  # 0 ~ 2π 랜덤 값
        rand_reset_rot = torch.stack([
            torch.cos(random_angles / 2),  # w
            torch.zeros(self.num_envs, device=self.device),  # x
            torch.zeros(self.num_envs, device=self.device),  # y
            torch.sin(random_angles / 2)  # z (z축 회전)
        ], dim=1)
        
        rand_reset_box_pose = torch.cat([rand_reset_pos, rand_reset_rot], dim=-1)
        zero_root_velocity = torch.zeros((self.num_envs, 6), device=self.device)

        if object_move == ObjectMoveType.STATIC:
            self._box.write_root_pose_to_sim(rand_reset_box_pose)
            self._box.write_root_velocity_to_sim(zero_root_velocity)
        
        # 물체 위치 선형 랜덤 이동 (Linear) ---------------------------------------------------------------
        if object_move == ObjectMoveType.LINEAR:
            self.new_box_pos_rand = self._box.data.body_link_pos_w[:, 0, :].clone()
            self.current_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()

            # self.new_box_pos_rand = self.current_box_pos
            # self.target_box_pos = self.rand_pos

            direction = self.target_box_pos - self.new_box_pos_rand
            direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
            self.rand_pos_step = (direction / direction_norm * self.obj_speed)
        #--------------------------------------------------------------------------------
               
        self.cfg.current_time = 0
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        global robot_action
        
        camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
        box_pos_cam, box_rot_cam = self.world_to_camera_pose(camera_pos_w, camera_rot_w, self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,)
                
        if foundationpose_mode:
            rclpy.spin_once(self.foundationpose_node, timeout_sec=0.01)
            pos = self.subscribe_object_pos()
            
            if (pos is not None): #and robot_init_pose:

                # camera_pos_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
                # camera_rot_w = self.robot_grasp_rot
                
                foundationpose_pos = pos.repeat(self.num_envs, 1)
                foundationpose_pos_converted = torch.zeros_like(foundationpose_pos)
                
                if robot_type == RobotType.FRANKA:
                    foundationpose_pos_converted[:, 0] = -foundationpose_pos[:, 1]  # x = -y_fp
                    foundationpose_pos_converted[:, 1] =  foundationpose_pos[:, 0]  # y = x_fp
                    foundationpose_pos_converted[:, 2] =  foundationpose_pos[:, 2]  # z = z_fp
                
                elif robot_type == RobotType.UF:
                    foundationpose_pos_converted[:, 0] =  foundationpose_pos[:, 0]
                    foundationpose_pos_converted[:, 1] =  foundationpose_pos[:, 1] 
                    foundationpose_pos_converted[:, 2] =  foundationpose_pos[:, 2] 
                    
                elif robot_type == RobotType.DOOSAN :
                    foundationpose_pos_converted[:, 0] =  foundationpose_pos[:, 0]
                    foundationpose_pos_converted[:, 1] =  foundationpose_pos[:, 1] 
                    foundationpose_pos_converted[:, 2] =  foundationpose_pos[:, 2] 
                
                fp_world_pos, _ = self.camera_to_world_pose(camera_pos_w, camera_rot_w, foundationpose_pos_converted, self.box_grasp_rot,)

                # print(f"isaac_cam_pos : {box_pos_cam}")
                # print(f"fp_cam_pos : {foundationpose_pos_converted}")
                # print(f"isaac_world_pos : {self.box_grasp_pos}")
                # print(f"fp_world_pos : {fp_world_pos}")

                to_target = fp_world_pos - self.robot_grasp_pos
            else:
                robot_action = False
                to_target = self.box_grasp_pos - self.robot_grasp_pos 
        
        elif yolo_mode: # yolo mode
            rclpy.spin_once(self.yolo_node, timeout_sec=0.01)
            yolo_pos = self.subscribe_yolo()

            if (yolo_pos is not None):
                yolo_pos = yolo_pos.repeat(self.num_envs, 1)
                yolo_pos_converted = torch.zeros_like(yolo_pos)

                if robot_type == RobotType.UF:
                    yolo_pos_converted[:, 0] = -yolo_pos[:, 1]
                    yolo_pos_converted[:, 1] =  yolo_pos[:, 0] 
                    yolo_pos_converted[:, 2] =  yolo_pos[:, 2]

                yolo_world_pos, _ = self.camera_to_world_pose(camera_pos_w, camera_rot_w, yolo_pos_converted, self.box_grasp_rot,)
                # yolo_world_pos, _ = self.camera_to_world_pose(camera_pos_w, camera_rot_w, yolo_pos, self.box_grasp_rot,)

                print(f"yolo_cam_pos : {yolo_pos_converted}")
                # print(f"isaac_world_pos : {self.box_grasp_pos}")
                print(f"yolo_world_pos : {yolo_world_pos}")

                to_target = yolo_world_pos - self.robot_grasp_pos

            else:
                robot_action = False
                to_target = self.box_grasp_pos - self.robot_grasp_pos 
        
        else:
            to_target = self.box_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._box.data.body_link_pos_w[:, 0, 2].unsqueeze(-1),
                self._box.data.body_link_vel_w[:, 0, 2].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0),}
    
    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]

        box_pos_world = self._box.data.body_link_pos_w[env_ids, self.box_idx]
        box_rot_world = self._box.data.body_link_quat_w[env_ids, self.box_idx]
        
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.box_grasp_rot[env_ids],
            self.box_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            box_rot_world,
            box_pos_world,
            self.box_local_rot[env_ids],
            self.box_local_pos[env_ids],
        )
        
    def _compute_rewards(
        self,
        actions,
        franka_grasp_pos, 
        box_pos_w,    
        franka_grasp_rot,
        box_rot_w,
        box_pos_cam,
        box_rot_cam,
        gripper_forward_axis,
        gripper_up_axis,
    ):
        # distance_reward_scale = 10.0
        # vector_align_reward_scale = 8.0
        # position_align_reward_scale = 6.0
        # pview_reward_scale = 10.0
        # veloity_align_reward_scale = 0.0
        # joint_penalty_scale = 0.0
        # ee_motion_penalty_weight = 0.0
        
        distance_reward_scale = 9.5
        vector_align_reward_scale = 8.0
        position_align_reward_scale = 6.0
        pview_reward_scale = 10.0
        veloity_align_reward_scale = 0.0
        joint_penalty_scale = 2.0
        
        if not hasattr(self, "init_robot_joint_position"):
            self.init_robot_joint_position = self._robot.data.joint_pos.clone()
        
        eps = 1e-6
        
        ## 거리 유지 보상 (그리퍼와 물체 간 거리 일정 유지)
        if robot_type == RobotType.FRANKA or robot_type == RobotType.UF:
            # min_dist = 0.20
            # max_dist = 0.30
            # target_distance = 0.25
            
            min_dist = 0.25
            max_dist = 0.35
            target_distance = 0.30
            
        elif robot_type == RobotType.DOOSAN:
            min_dist = 0.30
            max_dist = 0.40
            target_distance = 0.35
        
        gripper_to_box_dist = torch.norm(franka_grasp_pos - box_pos_w, p=2, dim=-1)
        distance_error = torch.abs(gripper_to_box_dist - target_distance)
        
        within_range = (gripper_to_box_dist >= min_dist) & (gripper_to_box_dist <= max_dist)
        too_close = gripper_to_box_dist < min_dist
        too_far = gripper_to_box_dist > max_dist
        # too_close_or_far = ~within_range
        
        distance_reward = torch.zeros_like(gripper_to_box_dist)
       
        #학습 초기 상수 보상
        # distance_reward[within_range] = 1.0
        # distance_reward[too_close_or_far] = -1.0 * torch.tanh(5.0 * distance_error[too_close_or_far])

        #학습 후기 선형 보상
        k = 2.0  # 보상 기울기 
        distance_reward[within_range] = 1.0 - k * distance_error[within_range]
        # distance_reward[too_close_or_far] = -1.0 * torch.tanh(5.0 * distance_error[too_close_or_far])
        
        distance_reward[too_close] = -3.0 * torch.tanh(10.0 * distance_error[too_close])
        distance_reward[too_far] = -3.0 * torch.tanh(5.0 * distance_error[too_far])

        ## 잡기축 정의 (그리퍼 초기 위치 → 물체 위치 벡터) 그리퍼 위치가 잡기축 위에 있는지 확인
        # robot_origin = self.scene.env_origins + torch.tensor([1.0, 0.0, 0.0], device=self.scene.env_origins.device)
        # xy_vec = box_pos_w[:, :2]  - robot_origin[:, :2]
        # xy_dir = xy_vec / (torch.norm(xy_vec, dim=-1, keepdim=True) + eps)  
        # xy_scaled = xy_dir * (2**0.5 / 2)                                   
        # z_component = torch.full_like(xy_scaled[:, :1], -(2**0.5 / 2))
        # grasp_axis = torch.cat([xy_scaled, z_component], dim=-1)
        
        if not hasattr(self, "init_grasp_pos"):
            self.init_grasp_pos = franka_grasp_pos.clone().detach()
        # print(f"init_grasp_pos : {self.init_grasp_pos}")
        
        # (25.08.13) 로봇 스폰 시 env origin에 대한 상대 위치가 (1.0, 0.0, ...)으로 설정된 것을 기준으로 합니다.
        robot_base_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        robot_base_pos_xy = self.scene.env_origins + robot_base_offset
        
        grasp_axis_origin = torch.zeros_like(self.init_grasp_pos)
        grasp_axis_origin[:, 0] = robot_base_pos_xy[:, 0]
        grasp_axis_origin[:, 1] = robot_base_pos_xy[:, 1]
        grasp_axis_origin[:, 2] = 0.55
            
        # grasp_vec = box_pos_w - self.init_grasp_pos  # [num_envs, 3]
        grasp_vec = box_pos_w - grasp_axis_origin 
        grasp_axis = grasp_vec / (torch.norm(grasp_vec, dim=-1, keepdim=True) + eps)
        
        gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)
        
        alignment_cos = torch.sum(gripper_forward * grasp_axis, dim=-1).clamp(-1.0, 1.0)

        if robot_type == RobotType.DOOSAN:
            alignment_cos =  alignment_cos + 0.22 ##doosan
        
        # vector_align_margin = 0.85
        vector_align_margin = 0.90
        # vector_align_margin = 0.95
        
        vector_alignment_reward = torch.where(
            alignment_cos >= vector_align_margin,
            # 1, # 학습 초기 상수 보상
            alignment_cos, # 학습 후기 선형 보상
            -1.0 * (1.0 - alignment_cos)
        )
        
        ## position_alignment_reward (그리퍼가 물체의 잡기축에 수직으로 위치하는지 확인)
        gripper_proj_dist = torch.norm(torch.cross(franka_grasp_pos - box_pos_w, grasp_axis, dim=-1),dim=-1)        

        # position_align_margin = 0.1
        position_align_margin = 0.05
        
        max_proj_dist = 0.15 
        start_reward = 0.0        
        slope = -10.0

        position_alignment_reward = slope * (gripper_proj_dist - position_align_margin) + start_reward
        position_alignment_reward = torch.clamp(position_alignment_reward, min=-3.0)
        
        positive_mask = position_alignment_reward > 0.0
        position_alignment_reward = torch.where(
            positive_mask,
            position_alignment_reward + 1,
            position_alignment_reward
        )
        
        ## 카메라 veiw 중심으로부터 거리 (XY 평면 기준) 시야 이탈 판단
        # 1. 카메라 앞에 있는지(Z > 0) 먼저 확인하는 마스크 생성
        is_in_front_mask = box_pos_cam[:, 2] > 0

        # 2. 기존과 동일하게 XY 평면 거리 계산
        center_offset = torch.norm(box_pos_cam[:, :2], dim=-1)
        
        pview_margin = 0.20 # 학습 초기
        # pview_margin = 0.15 # 학습 중기
        # # pview_margin = 0.10 # 학습 후기
        
        out_of_fov_mask = center_offset > pview_margin

        # 3. 시야각 내에 있는지에 대한 보상 후보 값을 먼저 계산
        pview_reward_candidate = torch.where(
            out_of_fov_mask,
            torch.full_like(center_offset, -3.0),
            torch.where(
                center_offset <= 0.1,
                torch.full_like(center_offset, 2.0),
                torch.exp(-10.0 * (center_offset - 0.15))
            )
        )

        # 4. 최종적으로, 카메라 앞에 있을 때만 후보 보상을 적용하고, 뒤에 있으면 큰 페널티를 부여
        pview_reward = torch.where(
            is_in_front_mask,
            pview_reward_candidate,
            torch.full_like(center_offset, -5.0) # 카메라 뒤에 있을 경우 더 큰 페널티
        )
        
        # center_offset = torch.norm(box_pos_cam[:, :2], dim=-1)

        # # pview_margin = 0.20 # 학습 초기
        # pview_margin = 0.15 # 학습 중기
        # # pview_margin = 0.10 # 학습 후기
        # out_of_fov_mask = center_offset > pview_margin

        # pview_reward = torch.where(
        #     out_of_fov_mask,
        #     torch.full_like(center_offset, -3.0),
        #     torch.where(
        #         center_offset <= 0.1,
        #         torch.full_like(center_offset,2.0),
        #         torch.exp(-10.0 * (center_offset - 0.15))
        #     )
        # )
        
        ## 속도 정렬 보상
        if not hasattr(self, "prev_box_pos"):
            self.prev_box_pos = box_pos_w.clone()
            self.prev_gripper_pos = franka_grasp_pos.clone()

        oject_velocity = (box_pos_w - self.prev_box_pos) / self.dt
        gripper_velocity = (franka_grasp_pos - self.prev_gripper_pos) / self.dt
        
        dot = torch.sum(oject_velocity * gripper_velocity, dim=-1)
        norm = torch.norm(oject_velocity, p=2, dim=-1) * torch.norm(gripper_velocity, p=2, dim=-1) + eps
        velocity_alignment_reward_raw = dot / norm
        
        velocity_align_mask = (gripper_to_box_dist < max_dist)  # 또는 적절한 거리 기준
        
        if init_reward:
            # velocity_alignment_reward = torch.where(
            #     velocity_align_mask & (velocity_alignment_reward > 0.7),
            #     torch.ones_like(velocity_alignment_reward),
            #     torch.full_like(velocity_alignment_reward, -1.0)
            # )
            
            high_threshold = 0.8
            min_val = -1.0  

            velocity_alignment_reward = torch.zeros_like(velocity_alignment_reward_raw)

            well_aligned_mask = velocity_align_mask & (velocity_alignment_reward_raw >= high_threshold)
            velocity_alignment_reward[well_aligned_mask] = 1.0

            poorly_aligned_mask = velocity_align_mask & ~well_aligned_mask
            velocity_alignment_reward[poorly_aligned_mask] = (
                (velocity_alignment_reward_raw[poorly_aligned_mask] - high_threshold)
                / (high_threshold - min_val + eps)
            )
            
            velocity_alignment_reward = torch.clamp(velocity_alignment_reward, -1.0, 1.0)
        else:
            velocity_alignment_reward = velocity_align_mask * velocity_alignment_reward_raw
        
        self.prev_box_pos = box_pos_w.clone()
        self.prev_grasp_pos = franka_grasp_pos.clone()

        ## 자세 안정성 유지 패널티
        joint_deviation = torch.abs(self._robot.data.joint_pos - self.init_robot_joint_position)
        joint_weights = torch.ones_like(joint_deviation)
        
        if robot_type == RobotType.FRANKA:
            joint_weights[:, 2] = 0.0
            joint_weights[:, 4] = 0.0 
        elif robot_type == RobotType.UF:
            joint4_idx = self._robot.find_joints(["joint4"])[0]
            joint6_idx = self._robot.find_joints(["joint6"])[0]
            joint_weights[:, joint4_idx] = 0.0
            joint_weights[:, joint6_idx] = 0.0
        elif robot_type == RobotType.DOOSAN:
            joint4_idx = self._robot.find_joints(["J4_joint"])[0]
            joint6_idx = self._robot.find_joints(["J6_joint"])[0]
            joint_weights[:, joint4_idx] = 0.0
            joint_weights[:, joint6_idx] = 0.0
            
        weighted_joint_deviation = joint_deviation * joint_weights
        joint_penalty = torch.sum(weighted_joint_deviation, dim=-1)
        joint_penalty = torch.tanh(joint_penalty)
        
        ## 안정성 유지 패널티
        if not hasattr(self, "prev_ee_pos_for_stability"):
           self.prev_ee_pos_for_stability = franka_grasp_pos.clone()
        
        ee_motion = torch.norm(franka_grasp_pos - self.prev_ee_pos_for_stability, dim=-1)

        ee_motion_threshold = 0.01
        ee_motion_scale = 100.0  
        
        ee_motion_penalty = torch.where(
            ee_motion < ee_motion_threshold,
            torch.zeros_like(ee_motion),
            torch.exp(ee_motion_scale * (ee_motion - ee_motion_threshold)) - 1.0
        )

        self.prev_ee_pos_for_stability = franka_grasp_pos.clone()
        
        ## 최종 보상 계산
        rewards = (
            distance_reward_scale * distance_reward  
            + vector_align_reward_scale * vector_alignment_reward
            + position_align_reward_scale * position_alignment_reward
            + pview_reward_scale * pview_reward
            # + veloity_align_reward_scale * velocity_alignment_reward
            - joint_penalty_scale * joint_penalty 
            # - ee_motion_penalty_weight * ee_motion_penalty
        )
        
        # print("=====================================")
        # print("gripper_to_box_dist : ", gripper_to_box_dist)
        # print("distance_reward : ", distance_reward)
        # print("alignment_cos : ", alignment_cos)
        # print("grasp_axis_origin : ", {grasp_axis_origin})
        # print("vector_alignment_reward:", vector_alignment_reward)
        # # print("position_alignment_reward:", position_alignment_reward)
        # print("center_offset:", center_offset)
        # print("pview_reward:", pview_reward)
        # print(f"ee_motion_penalty : {ee_motion_penalty}")

        #2025.08.19

        return rewards
        
    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        box_rot,
        box_pos,
        box_local_rot,
        box_local_pos,

    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        
        global_box_rot, global_box_pos = tf_combine(
            box_rot, box_pos, box_local_rot, box_local_pos
        )

        return global_franka_rot, global_franka_pos, global_box_rot, global_box_pos
        
        
        