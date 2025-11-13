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
# from builtin_interfaces.msg import Time

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
# object_move = ObjectMoveType.STATIC
object_move = ObjectMoveType.LINEAR
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
UFactory_set_mode = True

add_episode_length = 200
# add_episode_length = 600
# add_episode_length = -400 # 초기 학습 시 episode 길이

vel_ratio = 1.0
obj_speed = 0.0015

rand_pos_range = {
    "x" : (  0.35, 0.80),
    "y" : ( -0.3, 0.3),
    "z" : (  0.08, 0.8),
    
    # "x" : (  0.40, 0.65),
    # "y" : ( -0.0, 0.0),
    # "z" : (  0.15, 0.55),
    
    # "x" : (  0.6, 0.60),
    # "y" : ( -0.3, 0.3),
    # "z" : (  0.1, 0.1),
    
    # "x" : (  0.5, 0.5),
    # "y" : ( -0.0, 0.0),
    # "z" : (  0.05, 0.75),
    
    # "x" : (  0.35, 0.35),
    # "y" : ( -0.0, 0.0),
    # "z" : (  0.78, 0.78),
    
    # "x" : (  0.35, 0.35),
    # "y" : ( -0.0, 0.0),
    # "z" : (  0.30, 0.30),
}

reward_curriculum_levels = [
    # Level 0: (Static, Robot Speed 0.5) - 가장 넓은 마진
    {
        "reward_scales": {"pview": 1.0, "distance": 1.0, "vector_align": 0.8, "position_align": 0.7, "joint_penalty": 0.5},
        "success_multiplier": 1.5, "failure_multiplier": 1.2, 
        "y_range" : ( -0.35, 0.35),

        "distance_margin" : 0.30,
        "vector_align_margin" : math.radians(30.0),
        "position_align_margin" : 0.30,
        "pview_margin" : 0.30,
        "fail_margin" : 0.4,
    },
    # [신규] Level 1: (Moving 0.0005, Robot Speed 0.5) - 물체 이동 "먼저" 학습
    {
        "reward_scales": {"pview": 1.0, "distance": 1.0, "vector_align": 0.8, "position_align": 0.7, "joint_penalty": 0.5},
        "success_multiplier": 1.5, "failure_multiplier": 1.2, 
        "y_range" : ( -0.35, 0.35),

        "distance_margin" : 0.25, # 마진 약간 좁힘
        "vector_align_margin" : math.radians(25.0),
        "position_align_margin" : 0.25,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # [신규] Level 2: (Moving 0.0005, Robot Speed 1.0) - "그다음" 로봇 속도 증가
    {
        "reward_scales": {"pview": 1.0, "distance": 0.8, "vector_align": 1.0, "position_align": 0.8, "joint_penalty": 0.5},
        "success_multiplier": 1.0, "failure_multiplier": 1.0, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.20,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.20,
        "fail_margin" : 0.3
    },
    # [신규] Level 3: (Moving Random, Robot Speed 1.0) - "그다음" 물체 속도 증가
    {
        "reward_scales": {"pview": 1.0, "distance": 0.8, "vector_align": 1.0, "position_align": 0.8, "joint_penalty": 0.5},
        "success_multiplier": 1.0, "failure_multiplier": 1.0, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.15,
        "vector_align_margin" : math.radians(15.0),
        "position_align_margin" : 0.15,
        "pview_margin" : 0.15,
        "fail_margin" : 0.3
    },
    # [신규] Level 4: (Moving Random, Robot Speed 1.5) - 최종
    {
        "reward_scales": {"pview": 1.2, "distance": 1.0, "vector_align": 1.2, "position_align": 1.2, "joint_penalty": 0.5},
        "success_multiplier": 2.0, "failure_multiplier": 1.2, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.10,
        "vector_align_margin" : math.radians(10.0),
        "position_align_margin" : 0.10,
        "pview_margin" : 0.1,
        "fail_margin" : 0.2,
    },
]

# vector_align_margin = math.radians(15.0)
# vector_align_margin = math.radians(10.0)
vector_align_margin = math.radians(5.0)

position_align_margin = 0.05
# position_align_margin = 0.03
# position_align_margin = 0.01

pview_margin = 0.20 
# pview_margin = 0.15 
# pview_margin = 0.10

pose_candidate = {
    # "zero" : {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(0.0), 
    #                   "joint3": math.radians(0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(0.0), 
    #                   "joint6": math.radians(0.0)},
    
    ## top------------------------------------------------
    # "top_close":   {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-50.0), 
    #                   "joint3": math.radians(-30.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-30.0), 
    #                   "joint6": math.radians(0.0)},
        
    "top_close":   {"joint1": math.radians(0.0), 
                      "joint2": math.radians(-75.0), 
                      "joint3": math.radians(-40.0), 
                      "joint4": math.radians(0.0), 
                      "joint5": math.radians(0.0), 
                      "joint6": math.radians(0.0)},

    # "top_close_2":   {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-110.0), 
    #                   "joint3": math.radians(5.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-5.0), 
    #                   "joint6": math.radians(0.0)},
    
    # "top_middle":   {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians(-30.0), 
    #                   "joint3": math.radians(-30.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians( -45.0), 
    #                   "joint6": math.radians(  0.0)},
    
    "top_middle":   {"joint1": math.radians(  0.0), 
                      "joint2": math.radians(-25.0), 
                      "joint3": math.radians(-60.0), 
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians( -30.0), 
                      "joint6": math.radians(  0.0)},
    
    # "top_middle_2":   {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians(-5.0), 
    #                   "joint3": math.radians(-60.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians( -35.0), 
    #                   "joint6": math.radians(  0.0)},
    
    # "top_far":     {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians( -20.0),  
    #                   "joint3": math.radians(-45.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians(  -35.0), 
    #                   "joint6": math.radians(  0.0)},
    
    "top_far":     {"joint1": math.radians(  0.0), 
                    "joint2": math.radians(  0.0),  
                    "joint3": math.radians(-90.0), 
                    "joint4": math.radians(  0.0), 
                    "joint5": math.radians(-20.0), 
                    "joint6": math.radians(  0.0)},
    
    # "top_far_2":     {"joint1": math.radians(  0.0), 
    #                   "joint2": math.radians(  0.0),  
    #                   "joint3": math.radians(-65.0), 
    #                   "joint4": math.radians(  0.0), 
    #                   "joint5": math.radians(-35.0), 
    #                   "joint6": math.radians(  0.0)},
    
    ##middle------------------------------------------------
    # "middle_close":  {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-110.0),
    #                   "joint3": math.radians( -5.0),  
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(45.0), 
    #                   "joint6": math.radians(0.0)},
    
    # "middle_middle": {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-40.0), 
    #                   "joint3": math.radians(0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-45.0),  
    #                   "joint6": math.radians(0.0)},
    
    # "middle_far":    {"joint1": math.radians(0.0),  
    #                   "joint2": math.radians(15.0),  
    #                   "joint3": math.radians(-50.0),
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-50.0), 
    #                   "joint6": math.radians(0.0)},
    
    "middle_close":  {"joint1": math.radians(  0.0), 
                      "joint2": math.radians(-90.0),
                      "joint3": math.radians(-25.0),  
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians( 25.0), 
                      "joint6": math.radians( 0.0)},
    
    "middle_middle": {"joint1": math.radians(  0.0), 
                      "joint2": math.radians(-45.0), 
                      "joint3": math.radians(-40.0), 
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians( -5.0),  
                      "joint6": math.radians(  0.0)},
    
    "middle_far":    {"joint1": math.radians(  0.0),  
                      "joint2": math.radians(  5.0),  
                      "joint3": math.radians(-80.0),
                      "joint4": math.radians(  0.0), 
                      "joint5": math.radians(-15.0), 
                      "joint6": math.radians(  0.0)},

    ##bottom------------------------------------------------
    # "bottom_close":  {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-95.0),   
    #                   "joint3": math.radians(-5.0),   
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians( 50.0),
    #                   "joint6": math.radians(0.0)},
    
    # "bottom2_close2":  {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-70.0),   
    #                   "joint3": math.radians( 0.0),   
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians( 35.0),
    #                   "joint6": math.radians(0.0)},
    
    # "bottom_middle": {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-60.0),  
    #                   "joint3": math.radians(-0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(10.0),
    #                   "joint6": math.radians(0.0)},
    
    # "bottom_middle_2": {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-30.0),  
    #                   "joint3": math.radians(-0.0), 
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-10.0),
    #                   "joint6": math.radians(0.0)},
    
    # "bottom_far":    {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(-25.0),  
    #                   "joint3": math.radians(-15.0),
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-5.0), 
    #                   "joint6": math.radians(0.0)},
    
    # "bottom_far_2":    {"joint1": math.radians(0.0), 
    #                   "joint2": math.radians(15.0),  
    #                   "joint3": math.radians(-45.0),
    #                   "joint4": math.radians(0.0), 
    #                   "joint5": math.radians(-5.0), 
    #                   "joint6": math.radians(0.0)},
    
    "bottom_close":    {"joint1": math.radians(  0.0), 
                        "joint2": math.radians(-95.0),  
                        "joint3": math.radians(-10.0),
                        "joint4": math.radians(  0.0), 
                        "joint5": math.radians( 60.0), 
                        "joint6": math.radians(  0.0)},
    
    "bottom_middle":   {"joint1": math.radians(  0.0), 
                        "joint2": math.radians(-40.0),  
                        "joint3": math.radians(-25.0),
                        "joint4": math.radians(  0.0), 
                        "joint5": math.radians( 20.0), 
                        "joint6": math.radians(  0.0)},
    
    "bottom_far":      {"joint1": math.radians(  0.0), 
                        "joint2": math.radians(  5.0),  
                        "joint3": math.radians(-55.0),
                        "joint4": math.radians(  0.0), 
                        "joint5": math.radians(  5.0), 
                        "joint6": math.radians(  0.0)},
}

# initial_pose = pose_candidate["bottom_close"]
initial_pose = pose_candidate["middle_close"]
# initial_pose = pose_candidate["top_close"]
# initial_pose = pose_candidate["zero"]

workspace_zones = {
    "x": {"close" : 0.35, "middle": 0.50,"far": 0.70},
    "z": {"bottom": 0.30, "middle": 0.50,"top": 0.78}
}

x_weights = {"far": 1.0, "middle": 1.0, "close" : 1.0}
z_weights = {"top": 1.0, "middle": 1.0, "bottom": 1.0}

zone_activation = {
    "top_close":    True,
    "top_middle":   True,
    "top_far":      False, # << 이 값을 False로 바꾸면 제외됩니다.
    "middle_close": True,
    "middle_middle":True,
    "middle_far":   True,
    "bottom_close": True,
    "bottom_middle":True,
    "bottom_far":   True,
}

CSV_FILEPATH = "/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/tracking_data.csv"

zone_definitions = {
    "top_close":    {"x": (workspace_zones["x"]["middle"], workspace_zones["x"]["far"]),   "z": (workspace_zones["z"]["middle"], rand_pos_range["z"][1])},
    "top_middle":   {"x": (workspace_zones["x"]["middle"], workspace_zones["x"]["far"]),   "z": (workspace_zones["z"]["middle"], rand_pos_range["z"][1])},
    "top_far":      {"x": (workspace_zones["x"]["far"],   rand_pos_range["x"][1]),         "z": (workspace_zones["z"]["middle"], rand_pos_range["z"][1])},
    "middle_close": {"x": (rand_pos_range["x"][0], workspace_zones["x"]["middle"]), "z": (workspace_zones["z"]["bottom"], workspace_zones["z"]["middle"])},
    "middle_middle":{"x": (workspace_zones["x"]["middle"], workspace_zones["x"]["far"]),   "z": (workspace_zones["z"]["bottom"], workspace_zones["z"]["middle"])},
    "middle_far":   {"x": (workspace_zones["x"]["far"],   rand_pos_range["x"][1]),         "z": (workspace_zones["z"]["bottom"], workspace_zones["z"]["middle"])},
    "bottom_close": {"x": (rand_pos_range["x"][0], workspace_zones["x"]["middle"]), "z": (rand_pos_range["z"][0], workspace_zones["z"]["bottom"])},
    "bottom_middle":{"x": (workspace_zones["x"]["middle"], workspace_zones["x"]["far"]),   "z": (rand_pos_range["z"][0], workspace_zones["z"]["bottom"])},
    "bottom_far":   {"x": (workspace_zones["x"]["far"],   rand_pos_range["x"][1]),         "z": (rand_pos_range["z"][0], workspace_zones["z"]["bottom"])},
    # "bottom2_close2": {"x": (rand_pos_range["x"][0], workspace_zones["x"]["close"]), "z": (rand_pos_range["z"][0], workspace_zones["z"]["bottom"])}
}
zone_keys = list(pose_candidate.keys())


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
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/xarm6_with_gripper/xarm6_with_gripper.usd",
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
                "joint3": -0.50,
                "joint4":  0.00,
                "joint5":  1.30,
                # "joint5":  0.30,
                # "joint5":  0.80,
                "joint6":  0.00,
                # "left_finger_joint" : 0.0,
                # "right_finger_joint": 0.0,
                
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
                effort_limit=87.0,
                
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.15, 0.1), rot=(0.923, 0, 0, -0.382)),
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

    action_scale = 1.0
    dof_velocity_scale = 0.07

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
        
        self.target_angle_matrix = torch.tensor([
            [-40.0, -30.0, -20.0],  # z < 0.5 (bottom) 일 때의 x 구간별 각도
            [  0.0,  0.0,   0.0],  # 0.5 <= z < 0.7 (middle) 일 때의 x 구간별 각도
            [ 40.0,  30.0,  20.0]   # z >= 0.7 (top) 일 때의 x 구간별 각도
        ], device=self.device)
        
        self.boundaries_x = torch.tensor([workspace_zones["x"]["middle"], workspace_zones["x"]["far"]], device=self.device)
        self.boundaries_z = torch.tensor([workspace_zones["z"]["middle"], workspace_zones["z"]["top"]], device=self.device)
        
        self.log_counter = 0
        self.LOG_INTERVAL = 5  # 1번의 리셋 묶음마다 한 번씩 로그 출력
        
        # 성능 모니터링을 위한 버퍼
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device)
        
        # 1. 보상 스케일만 조절하는 새로운 커리큘럼 레벨 정의
        self.max_reward_level = len(reward_curriculum_levels) - 1
        self.baseline_avg_reward = 0.20 # 계산된 기준 보상값

        # 2. 보상 커리큘럼을 위한 독립적인 상태 변수들
        self.current_reward_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_successes_reward = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_failures_reward = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.PROMOTION_COUNT_REWARD = 10
        self.DEMOTION_COUNT_REWARD = 5
        
        self.episode_init_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        
        self.curriculum_factor_k0 = 0.25  # k_c의 초기값 (논문 권장값)
        self.curriculum_factor_kd = 0.997 # k_c의 진전 속도
        
        # k_c (커리큘럼 계수) 상태 변수. 모든 환경이 k_c의 초기값에서 시작.
        # k_c는 (num_envs, 1) 형태로 저장됨
        self.curriculum_factor_k_c = torch.full((self.num_envs, 1), self.curriculum_factor_k0, device=self.device)
        
        # [추가] ------------------------------------------------------------------
        # 물체 이동 상태를 정의하는 상수
        self.MOVE_STATE_STATIC = 0
        self.MOVE_STATE_LINEAR = 1

        # 4096개 환경의 이동 상태를 개별적으로 저장하는 텐서 (0 = STATIC, 1 = LINEAR)
        self.object_move_state = torch.full(
            (self.num_envs,), self.MOVE_STATE_STATIC, dtype=torch.long, device=self.device
        )
        
        # 4096개 환경의 물체 이동 속도를 개별적으로 저장하는 텐서
        self.obj_speed = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        
        # [추가] ------------------------------------------------------------------
        # 4096개 환경의 액션 스케일(반응 속도)을 개별적으로 저장하는 텐서
        # Level 0의 기본값(낮은 속도)으로 초기화합니다.
        self.action_scale_tensor = torch.full(
            (self.num_envs,), 0.5, device=self.device, dtype=torch.float32
        )
        # ------------------------------------------------------------------------
        

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
            self.joint_init_values = [0.000, -1.220, -0.50, -0.000, 1.300, 0.000]
            # self.joint_init_values = [0.000, -1.220, -0.50, -0.000, 0.300, 0.000]
            # self.joint_init_values = [0.000, -1.220, -0.50, -0.000, 0.800, 0.000]
        elif robot_type == RobotType.DOOSAN:
            self.joint_names = [
            "J1_joint", "J2_joint", "J3_joint", "J4_joint","J5_joint", "J6_joint" ]
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
        robot_local_pose_pos += torch.tensor([0, 0.00, 0], device=self.device)
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
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0],
            ], dim = 1)
        
        self.target_box_pos = self.target_box_pos + self.scene.env_origins
        self.new_box_pos_rand = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.current_box_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.current_box_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.rand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.rand_pos_step = torch.zeros((self.num_envs, 3), device=self.device)
        
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
            self.latest_detection_msg = None
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
        
        # --- 1. "hand_eye_calibration.py" 스크립트 실행 결과 입력 ---
        t_cam_to_gripper_mm = torch.tensor(
            [70.03869874385214, 32.93603944336598, -129.5520585552788],
            device=self.device, dtype=torch.float32
        )
        
        # --- 2. 회전(Rotation) 값 ---
        R_cam_to_gripper_quat_ROS = torch.tensor(
            [-0.08403050810066812, 0.7031366469474544, 0.038105476236599455, 0.7050430498264744],
            device=self.device, dtype=torch.float32
        )

        # --- 3. '역변환' (T_gripper_to_cam) 계산 ---
        # T_cam_to_gripper (캘리브레이션 결과)
        R_cam_to_gripper_ROS = R_cam_to_gripper_quat_ROS.clone()
        t_cam_to_gripper_ROS = (t_cam_to_gripper_mm / 1000.0).unsqueeze(0) # [1, 3]

        # T_gripper_to_cam (T_cam_to_gripper의 역변환)
        # R_gripper_to_cam = R_cam_to_gripper^-1
        self.R_gripper_to_cam = R_cam_to_gripper_ROS.clone().unsqueeze(0) # [1, 4]
        self.R_gripper_to_cam[:, 1:] *= -1.0 # 켤레 (Inverse rotation)
        
        # t_gripper_to_cam = - (R_gripper_to_cam @ t_cam_to_gripper)
        R_gripper_to_cam_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(self.R_gripper_to_cam) # [1, 3, 3]
        
        # [1, 3, 1] = - ( [1, 3, 3] @ [1, 3, 1] )
        t_gripper_to_cam_ROS = -torch.bmm(R_gripper_to_cam_matrix, t_cam_to_gripper_ROS.unsqueeze(-1))
        
        # 최종적으로 사용할 역변환 값을 멤버 변수에 저장
        self.R_gripper_to_cam = self.R_gripper_to_cam.squeeze(0) # [4]
        self.t_gripper_to_cam = t_gripper_to_cam_ROS.squeeze(-1).squeeze(0) # [3]

        # [추가] 물체 가시성(visibility) 마스크. False(보이지 않음)로 초기화
        self.is_object_visible_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # [추가] 로봇 "정지" 명령을 위한 현재 조인트 위치 버퍼
        self.current_joint_pos_buffer = self._robot.data.joint_pos.clone()

    def publish_camera_data(self):
        env_id = 0
        
        current_stamp = self.node.get_clock().now().to_msg() 
        current_stamp.sec = current_stamp.sec % 50000
        current_stamp.nanosec = 0
        
        # current_stamp = Time()
        # current_stamp.sec = 1
        # current_stamp.nanosec = 0
                
        if image_publish:            
            rgb_data = self._camera.data.output["rgb"]
            depth_data = self._camera.data.output["depth"]
            
            rgb_image = (rgb_data.cpu().numpy()[env_id]).astype(np.uint8)
            depth_image = (depth_data.cpu().numpy()[env_id]).astype(np.float32)

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
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
            rgb_msg.header.stamp = current_stamp
            rgb_msg.header.frame_id = 'tf_camera'
            self.rgb_publisher.publish(rgb_msg)

            # Publish Depth Image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
            depth_msg.header.stamp = current_stamp
            depth_msg.header.frame_id = 'tf_camera'
            self.depth_publisher.publish(depth_msg)
            depth_msg.step = depth_image.shape[1] * 4
        
    def subscribe_object_pos(self):
        msg = self.latest_detection_msg
        
        if msg is None:
            return None

        return torch.tensor([msg.x, msg.y, msg.z], device=self.device)
        
    def foundationpose_callback(self,msg):
        self.latest_detection_msg = msg
    
    def subscribe_yolo(self):
        msg = self.yolo_msg

        self.yolo_msg = None

        if msg is None:
            return None

        return torch.tensor([msg.x, msg.y, msg.z], device=self.device)

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
        # r = scipy.spatial.transform.Rotation.from_euler('xzy', [yaw_deg, pitch_deg, roll_deg], degrees=True)
        quat_xyzw = r.as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        hand_pos_real = torch.tensor([pos_m], device=self.device, dtype=torch.float32)
        hand_rot_real = torch.tensor([quat_wxyz], device=self.device, dtype=torch.float32)

        # print("*" * 50)
        # print("hand_pos_real:",hand_pos_real)
        # print("hand_rot_real:",hand_rot_real)
        
        return hand_pos_real, hand_rot_real
    
    def compute_camera_world_pose(self, hand_pos, hand_rot):
        if robot_type == RobotType.FRANKA:
            cam_offset_pos = torch.tensor([0.0, 0.0, 0.05], device=hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.0, 0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
        elif robot_type == RobotType.UF:
            if camera_type == CameraType.Azure:
                cam_offset_pos = torch.tensor([-0.0, -0.075, 0.075], device=hand_pos.device).repeat(self.num_envs, 1)
                q_cam_in_hand = torch.tensor([0.0, -0.7071, 0.0, 0.7071], device=hand_pos.device).repeat(self.num_envs, 1)
            if camera_type == CameraType.Sim:
                cam_offset_pos = torch.tensor([0.0, 0.0, 0.1], device=hand_pos.device).repeat(self.num_envs, 1)
                q_cam_in_hand = torch.tensor([0.0, -0.7071, 0.0, 0.7071], device=hand_pos.device).repeat(self.num_envs, 1)
        elif robot_type == RobotType.DOOSAN:
            cam_offset_pos = torch.tensor([0.0, 0.0, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)
            # q_cam_in_hand = torch.tensor([-0.5, 0.5, -0.5, -0.5], device=hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.0, -0.707, 0.707, 0.0], device=hand_pos.device).repeat(self.num_envs, 1)

        hand_rot_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(hand_rot)
        cam_offset_pos_world = torch.bmm(hand_rot_matrix, cam_offset_pos.unsqueeze(-1)).squeeze(-1)

        camera_pos_w = hand_pos + cam_offset_pos_world
        camera_pos_w = camera_pos_w - self.scene.env_origins
        
        camera_rot_w = self.quat_mul(hand_rot, q_cam_in_hand)
        # camera_rot_w = self.robot_grasp_rot

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
        
        # [수정 시작] ---------------------------------------------------------
        # (self.num_envs,) 텐서를 (self.num_envs, 1)로 브로드캐스팅
        # current_action_scale = self.action_scale_tensor.unsqueeze(-1) 

        # 글로벌 self.cfg.action_scale 대신 개별 텐서(current_action_scale)를 사용
        # targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * current_action_scale
        # [수정 끝] -----------------------------------------------------------
        
        # targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        # self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        # [수정 시작] ---------------------------------------------------------
        # 1. 정책(actions)에 따른 잠재적 다음 목표 위치 계산
        current_action_scale = self.action_scale_tensor.unsqueeze(-1) 
        potential_targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * current_action_scale
        potential_targets_clamped = torch.clamp(potential_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        if training_mode:
            # [수정] 훈련 모드일 때는 가시성과 관계없이 항상 정책의 행동을 적용
            self.robot_dof_targets[:] = potential_targets_clamped
        
        else:
            # [수정] 테스트 모드일 때만 가시성에 따른 조건부 정지/행동 로직 적용
            
            # 2. "정지" 목표 위치 (이전 스텝에서 저장해둔 현재 조인트 위치)
            # self.current_joint_pos_buffer는 _get_observations에서 매 스텝 업데이트됩니다.
            hold_targets = self.current_joint_pos_buffer

            # 3. 가시성 마스크를 사용하여 목표 위치 선택
            # self.is_object_visible_mask는 _get_rewards에서 매 스텝 업데이트됩니다.
            visible_mask_expanded = self.is_object_visible_mask.unsqueeze(-1) 
            
            # self.robot_dof_targets를 최종 목표로 업데이트
            self.robot_dof_targets[:] = torch.where(
                visible_mask_expanded, 
                potential_targets_clamped,  # 시야 O: 행동 적용
                hold_targets                # 시야 X: 현재 위치 고수 (정지)
            )
        # [수정 끝] -----------------------------------------------------------
        
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
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0],
                ], dim = 1)

                self.target_box_pos = self.target_box_pos + self.scene.env_origins

                self.current_box_pos = self._box.data.body_link_pos_w[:, 0, :].clone()
                self.current_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()

                self.new_box_pos_rand = self.current_box_pos

                direction = self.target_box_pos - self.current_box_pos
                direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
                self.rand_pos_step = (direction / direction_norm * obj_speed)

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

                    ang_speed = 150
                    angmvacc = 100.0

                    rad_speed = math.radians(ang_speed)
                    rad_mvacc = math.radians(angmvacc)

                    # self.arm.set_servo_angle(angle=angle_cmd, speed=rad_speed, wait=False, is_radian=True, mvacc = rad_mvacc)

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
                    pos = self.subscribe_object_pos()
                    if (max_err < 0.3) and (pos is not None):
                        self.init_cnt += 1
                        print(f"init_cnt : {self.init_cnt}")
                        
                        if self.init_cnt > 50: 
                            robot_action = True
                            robot_init_pose = True
                
                elif yolo_mode :
                    print("[IsaacLab] Waiting for YOLO detection...")
                    pos = self.subscribe_yolo()
                    print(f"[IsaacLab] YOLO detected position: {pos}")
                    if (max_err < 0.3) and (pos is not None):
                        self.init_cnt += 1
                        print(f"init_cnt : {self.init_cnt}")
                              
                        if self.init_cnt > 50: 
                            robot_action = True
                            robot_init_pose = True
                            
                elif foundationpose_mode == False and yolo_mode == False and max_err < 0.3:
                    self.init_cnt += 1
                    print(f"init_cnt : {self.init_cnt}")
                    if self.init_cnt > 30:
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
        
        if camera_type == CameraType.Azure:
            hand_pos_real, hand_rot_real = self.get_real_hand_pose()
            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(hand_pos_real, hand_rot_real)
        elif camera_type == CameraType.Sim:
            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)

        self.box_pos_cam, box_rot_cam = self.world_to_camera_pose(
            camera_pos_w, camera_rot_w,
            self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,
        )

        gripper_to_box_dist = torch.norm(self.robot_grasp_pos - self.box_grasp_pos, p=2, dim=-1)

        # 1. 시야 중심 이탈 마스크 (center_offset > margin)
        center_offset = torch.norm(self.box_pos_cam[:, [2, 1]], dim=-1)
        out_of_fov_mask = center_offset > 0.3

        # 2. 물체가 카메라 뒤에 위치하는 마스크 (is_in_front_mask 반대)
        # print("self.box_pos_cam[:, 0] :", self.box_pos_cam[:, 0])
        is_behind_mask = self.box_pos_cam[:, 0] > 0 

        # 3. 최종 PView 실패 마스크
        self.is_pview_fail = out_of_fov_mask | is_behind_mask
        
        # --- [추가] 객체 가시성 마스크 업데이트 ---
        # self.is_pview_fail가 True이면 "시야 밖"이므로, 
        # is_object_visible_mask는 그 반대(~)가 됩니다.
        self.is_object_visible_mask = ~self.is_pview_fail

        reward = self._compute_rewards(
            self.actions,
            gripper_to_box_dist,
            self.robot_grasp_pos,
            self.box_grasp_pos,
            self.robot_grasp_rot,
            self.box_grasp_rot,
            self.box_pos_cam,
            box_rot_cam,
            self.gripper_forward_axis,
            self.gripper_up_axis,
        )

        return reward
    
    def _perform_static_reset(self, env_ids: torch.Tensor):        
        
        num_resets = len(env_ids)
        if num_resets == 0:
            return
            
        final_weights = []
        for key in zone_keys:
            if not zone_activation.get(key, False): # .get()으로 안전하게 접근
                final_weights.append(0.0)
                continue
            z_part, x_part = key.split('_')
            combined_weight = x_weights.get(x_part, 1.0) * z_weights.get(z_part, 1.0)
            final_weights.append(combined_weight)
        
        weights_tensor = torch.tensor(final_weights, dtype=torch.float, device=self.device)
        selected_zone_indices = torch.multinomial(weights_tensor, num_resets, replacement=True)

        x_mins = torch.tensor([zone_definitions[zone_keys[i]]["x"][0] for i in selected_zone_indices], device=self.device)
        x_maxs = torch.tensor([zone_definitions[zone_keys[i]]["x"][1] for i in selected_zone_indices], device=self.device)
        z_mins = torch.tensor([zone_definitions[zone_keys[i]]["z"][0] for i in selected_zone_indices], device=self.device)
        z_maxs = torch.tensor([zone_definitions[zone_keys[i]]["z"][1] for i in selected_zone_indices], device=self.device)

        x_pos = torch.rand(num_resets, device=self.device) * (x_maxs - x_mins) + x_mins
        z_pos = torch.rand(num_resets, device=self.device) * (z_maxs - z_mins) + z_mins

        current_levels = self.current_reward_level[env_ids]
        y_pos = torch.zeros(num_resets, device=self.device)
        
        for level_idx in range(self.max_reward_level + 1):
            level_mask = (current_levels == level_idx)
            num_in_level = torch.sum(level_mask)
            
            if num_in_level > 0:
                y_range = reward_curriculum_levels[level_idx]["y_range"]
                y_pos[level_mask] = torch.rand(num_in_level, device=self.device) * (y_range[1] - y_range[0]) + y_range[0]

        # [수정] self.rand_pos[env_ids] 에 할당
        self.rand_pos[env_ids] = torch.stack([x_pos, y_pos, z_pos], dim=1)
        rand_reset_pos = self.rand_pos[env_ids] + self.scene.env_origins[env_ids]
        
        random_angles = torch.rand(num_resets, device=self.device) * 2 * torch.pi
        rand_reset_rot = torch.stack([
            torch.cos(random_angles / 2),
            torch.zeros(num_resets, device=self.device),
            torch.zeros(num_resets, device=self.device),
            torch.sin(random_angles / 2)  
        ], dim=1)
        
        rand_reset_box_pose = torch.cat([rand_reset_pos, rand_reset_rot], dim=-1)
        zero_root_velocity = torch.zeros((self.num_envs, 6), device=self.device)
        
        self._box.write_root_pose_to_sim(rand_reset_box_pose, env_ids=env_ids)
        self._box.write_root_velocity_to_sim(zero_root_velocity[env_ids], env_ids=env_ids)
        
        if training_mode == True:
            joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
            joint1_idx = self._robot.find_joints(["joint1"])[0]
            
            YAW_CANDIDATE_ANGLES = { 15.0: math.radians(15.0), 45.0: math.radians(45.0), 75.0: math.radians(75.0) }
            ANGLE_BOUNDARIES = [30.0, 60.0, 90.0]
            
            for i, env_id in enumerate(env_ids):
                object_pos_local = rand_reset_pos[i] - self.scene.env_origins[env_id]
                obj_x, obj_y, obj_z = object_pos_local[0], object_pos_local[1], object_pos_local[2]
                        
                if obj_x >= workspace_zones["x"]["far"]: x_zone = "far"
                elif obj_x >= workspace_zones["x"]["middle"]: x_zone = "middle"
                else: x_zone = "close"
                    
                if obj_z >= workspace_zones["z"]["top"]: z_zone = "top"
                elif obj_z >= workspace_zones["z"]["bottom"]: z_zone = "middle"
                else: z_zone = "bottom"
                    
                zone_key = f"{z_zone}_{x_zone}"
                target_pose_dict = pose_candidate[zone_key]
                
                for joint_name, pos in target_pose_dict.items():
                    if joint_name != "joint1":
                        joint_idx = self._robot.find_joints(joint_name)[0]
                        joint_pos[i, joint_idx] = pos
                        
                target_yaw_rad = torch.atan2(obj_y, obj_x)
                abs_yaw_deg = torch.abs(torch.rad2deg(target_yaw_rad))

                if abs_yaw_deg <= ANGLE_BOUNDARIES[0]: target_angle_deg = 15.0
                elif abs_yaw_deg <= ANGLE_BOUNDARIES[1]: target_angle_deg = 45.0
                else: target_angle_deg = 75.0

                final_yaw_rad = YAW_CANDIDATE_ANGLES[target_angle_deg] * torch.sign(obj_y)
                joint_pos[i, joint1_idx] = final_yaw_rad
                
            joint_pos[:, joint1_idx] = torch.clamp(joint_pos[:, joint1_idx], self.robot_dof_lower_limits[joint1_idx], self.robot_dof_upper_limits[joint1_idx])
            joint_vel = torch.zeros_like(joint_pos)
            
            # [추가] 액추에이터 목표 변수(self.robot_dof_targets)도 리셋합니다.
            self.robot_dof_targets[env_ids] = joint_pos
            
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            
            self.episode_init_joint_pos[env_ids] = joint_pos

    def _perform_linear_reset(self, env_ids: torch.Tensor):
        
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        
        # 1. 로봇 및 물체 위치 재설정 (STATIC 리셋과 거의 동일)
        final_weights = []
        for key in zone_keys:
            if not zone_activation.get(key, False):
                final_weights.append(0.0)
                continue
            z_part, x_part = key.split('_')
            combined_weight = x_weights.get(x_part, 1.0) * z_weights.get(z_part, 1.0)
            final_weights.append(combined_weight)
        
        weights_tensor = torch.tensor(final_weights, dtype=torch.float, device=self.device)
        selected_zone_indices = torch.multinomial(weights_tensor, num_resets, replacement=True)

        x_mins = torch.tensor([zone_definitions[zone_keys[i]]["x"][0] for i in selected_zone_indices], device=self.device)
        x_maxs = torch.tensor([zone_definitions[zone_keys[i]]["x"][1] for i in selected_zone_indices], device=self.device)
        z_mins = torch.tensor([zone_definitions[zone_keys[i]]["z"][0] for i in selected_zone_indices], device=self.device)
        z_maxs = torch.tensor([zone_definitions[zone_keys[i]]["z"][1] for i in selected_zone_indices], device=self.device)

        x_pos = torch.rand(num_resets, device=self.device) * (x_maxs - x_mins) + x_mins
        z_pos = torch.rand(num_resets, device=self.device) * (z_maxs - z_mins) + z_mins

        current_levels = self.current_reward_level[env_ids]
        y_pos = torch.zeros(num_resets, device=self.device)
        
        for level_idx in range(self.max_reward_level + 1):
            level_mask = (current_levels == level_idx)
            num_in_level = torch.sum(level_mask)
            
            if num_in_level > 0:
                y_range = reward_curriculum_levels[level_idx]["y_range"]
                y_pos[level_mask] = torch.rand(num_in_level, device=self.device) * (y_range[1] - y_range[0]) + y_range[0]

        # [수정] self.rand_pos[env_ids] 에 할당
        self.rand_pos[env_ids] = torch.stack([x_pos, y_pos, z_pos], dim=1)
        rand_reset_pos = self.rand_pos[env_ids] + self.scene.env_origins[env_ids]
        
        random_angles = torch.rand(num_resets, device=self.device) * 2 * torch.pi
        rand_reset_rot = torch.stack([
            torch.cos(random_angles / 2),
            torch.zeros(num_resets, device=self.device),
            torch.zeros(num_resets, device=self.device),
            torch.sin(random_angles / 2)  
        ], dim=1)
        
        rand_reset_box_pose = torch.cat([rand_reset_pos, rand_reset_rot], dim=-1)
        zero_root_velocity = torch.zeros((self.num_envs, 6), device=self.device)
        
        self._box.write_root_pose_to_sim(rand_reset_box_pose, env_ids=env_ids)
        self._box.write_root_velocity_to_sim(zero_root_velocity[env_ids], env_ids=env_ids)

        # 2. LINEAR 이동을 위한 상태 초기화
        self.new_box_pos_rand[env_ids] = self._box.data.body_link_pos_w[env_ids, 0, :]
        self.current_box_rot[env_ids] = self._box.data.body_link_quat_w[env_ids, 0, :]

        # 2.2. 목표 위치를 *다른* 랜덤 위치로 새로 생성
        new_targets_x = torch.rand(num_resets, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0]
        new_targets_y = torch.rand(num_resets, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0]
        new_targets_z = torch.rand(num_resets, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0]
        new_targets = torch.stack([new_targets_x, new_targets_y, new_targets_z], dim = 1)
        
        self.target_box_pos[env_ids] = new_targets + self.scene.env_origins[env_ids]

        # 2.3. 이동 방향 및 스텝 계산
        direction = self.target_box_pos[env_ids] - self.new_box_pos_rand[env_ids]
        direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
        
        # [핵심] 개별 속도를 사용합니다.
        speed = self.obj_speed[env_ids].unsqueeze(-1)
        self.rand_pos_step[env_ids] = (direction / direction_norm * speed)

        # 3. 로봇 자세 초기화 (STATIC 리셋과 동일)
        if training_mode == True:
            joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
            joint1_idx = self._robot.find_joints(["joint1"])[0]
            
            YAW_CANDIDATE_ANGLES = { 15.0: math.radians(15.0), 45.0: math.radians(45.0), 75.0: math.radians(75.0) }
            ANGLE_BOUNDARIES = [30.0, 60.0, 90.0]
            
            for i, env_id in enumerate(env_ids):
                object_pos_local = rand_reset_pos[i] - self.scene.env_origins[env_id]
                obj_x, obj_y, obj_z = object_pos_local[0], object_pos_local[1], object_pos_local[2]
                        
                if obj_x >= workspace_zones["x"]["far"]: x_zone = "far"
                elif obj_x >= workspace_zones["x"]["middle"]: x_zone = "middle"
                else: x_zone = "close"
                    
                if obj_z >= workspace_zones["z"]["top"]: z_zone = "top"
                elif obj_z >= workspace_zones["z"]["bottom"]: z_zone = "middle"
                else: z_zone = "bottom"
                    
                zone_key = f"{z_zone}_{x_zone}"
                target_pose_dict = pose_candidate[zone_key]
                
                for joint_name, pos in target_pose_dict.items():
                    if joint_name != "joint1":
                        joint_idx = self._robot.find_joints(joint_name)[0]
                        joint_pos[i, joint_idx] = pos
                        
                target_yaw_rad = torch.atan2(obj_y, obj_x)
                abs_yaw_deg = torch.abs(torch.rad2deg(target_yaw_rad))

                if abs_yaw_deg <= ANGLE_BOUNDARIES[0]: target_angle_deg = 15.0
                elif abs_yaw_deg <= ANGLE_BOUNDARIES[1]: target_angle_deg = 45.0
                else: target_angle_deg = 75.0

                final_yaw_rad = YAW_CANDIDATE_ANGLES[target_angle_deg] * torch.sign(obj_y)
                joint_pos[i, joint1_idx] = final_yaw_rad
                
            joint_pos[:, joint1_idx] = torch.clamp(joint_pos[:, joint1_idx], self.robot_dof_lower_limits[joint1_idx], self.robot_dof_upper_limits[joint1_idx])
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            
            self.episode_init_joint_pos[env_ids] = joint_pos


    def _reset_idx(self, env_ids: torch.Tensor | None):         
        current_reward_levels = self.current_reward_level[env_ids]
        avg_reward = self.episode_reward_buf[env_ids] / self.episode_length_buf[env_ids]

        # 2. 새로운 보상 커리큘럼의 임계값 가져오기
        success_multipliers = torch.tensor([reward_curriculum_levels[l.item()]["success_multiplier"] for l in current_reward_levels], device=self.device)
        failure_multipliers = torch.tensor([reward_curriculum_levels[l.item()]["failure_multiplier"] for l in current_reward_levels], device=self.device)
        
        success_thresholds_reward = self.baseline_avg_reward * success_multipliers
        failure_thresholds_reward = self.baseline_avg_reward * failure_multipliers 
        
        success_mask_reward = avg_reward >= success_thresholds_reward
        failure_mask_reward = avg_reward < failure_thresholds_reward
        
        # 3. 보상 커리큘럼의 연속 성공/실패 카운터 업데이트
        self.consecutive_successes_reward[env_ids] += success_mask_reward.long()
        self.consecutive_successes_reward[env_ids] *= (1 - failure_mask_reward.long())
        
        self.consecutive_failures_reward[env_ids] += failure_mask_reward.long()
        self.consecutive_failures_reward[env_ids] *= (1 - success_mask_reward.long())
              
        self.log_counter += 1
        print("log_counter: ", self.log_counter)
        
        if self.log_counter % self.LOG_INTERVAL == 0:
            avg_successes = torch.mean(self.consecutive_successes_reward.float()).item()
            print(f"[Training Log] Avg Consecutive Successes: {avg_successes:.2f}")
            self.log_counter = 0 # 카운터 초기화
        
        promotion_candidate_mask_reward = self.consecutive_successes_reward[env_ids] >= self.PROMOTION_COUNT_REWARD
        
        if torch.any(promotion_candidate_mask_reward):
            promotion_env_ids = env_ids[promotion_candidate_mask_reward]
            self.current_reward_level[promotion_env_ids] = (self.current_reward_level[promotion_env_ids] + 1).clamp(max=self.max_reward_level)
            self.consecutive_successes_reward[promotion_env_ids] = 0
            
        demotion_candidate_mask_reward = self.consecutive_failures_reward[env_ids] >= self.DEMOTION_COUNT_REWARD
        
        if torch.any(demotion_candidate_mask_reward):
            demotion_env_ids = env_ids[demotion_candidate_mask_reward]
            self.current_reward_level[demotion_env_ids] = (self.current_reward_level[demotion_env_ids] - 1).clamp(min=0)
            self.consecutive_failures_reward[demotion_env_ids] = 0
        
        # 에피소드 보상 버퍼 초기화
        self.episode_reward_buf[env_ids] = 0.0
                
        # robot state ---------------------------------------------------------------------------------
        if training_mode:            
            new_k_c = torch.pow(self.curriculum_factor_k_c[env_ids], self.curriculum_factor_kd)
            self.curriculum_factor_k_c[env_ids] = new_k_c
            self.curriculum_factor_k_c.clamp_(max=1.0)    
        else:
            if not hasattr(self, "_initialized"):
                self._initialized = False

            if not self._initialized:
                joint_pos = self._robot.data.default_joint_pos[env_ids] 
                
                joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
                joint_vel = torch.zeros_like(joint_pos)
                self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
                self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
                self._initialized = True
        
        if training_mode:
            current_levels_for_reset = self.current_reward_level[env_ids]
        
            # [수정] 5단계로 마스크 확장
            mask_level_0 = (current_levels_for_reset == 0)
            mask_level_1 = (current_levels_for_reset == 1)
            mask_level_2 = (current_levels_for_reset == 2)
            mask_level_3 = (current_levels_for_reset == 3)
            mask_level_4_plus = (current_levels_for_reset >= 4)

            env_ids_level_0 = env_ids[mask_level_0]
            env_ids_level_1 = env_ids[mask_level_1]
            env_ids_level_2 = env_ids[mask_level_2]
            env_ids_level_3 = env_ids[mask_level_3]
            env_ids_level_4_plus = env_ids[mask_level_4_plus]

            # Level 0: (Static, Robot Speed 0.5)
            if len(env_ids_level_0) > 0:
                self.object_move_state[env_ids_level_0] = self.MOVE_STATE_STATIC
                self.obj_speed[env_ids_level_0] = 0.0
                self.action_scale_tensor[env_ids_level_0] = 0.5 
                self._perform_static_reset(env_ids_level_0) 

            # [신규] Level 1: (Moving 0.0005, Robot Speed 0.5) - 물체 이동 먼저
            if len(env_ids_level_1) > 0:
                self.object_move_state[env_ids_level_1] = self.MOVE_STATE_LINEAR
                self.obj_speed[env_ids_level_1] = 0.0005 # 물체 이동 시작
                self.action_scale_tensor[env_ids_level_1] = 0.5 # 로봇 속도 유지
                self._perform_linear_reset(env_ids_level_1)

            # [신규] Level 2: (Moving 0.0005, Robot Speed 1.0) - 다음 로봇 속도 증가
            if len(env_ids_level_2) > 0:
                self.object_move_state[env_ids_level_2] = self.MOVE_STATE_LINEAR
                self.obj_speed[env_ids_level_2] = 0.0005
                self.action_scale_tensor[env_ids_level_2] = 1.0 # 로봇 속도 증가
                self._perform_linear_reset(env_ids_level_2)

            # [신규] Level 3: (Moving Random, Robot Speed 1.0) - 다음 물체 속도 증가
            if len(env_ids_level_3) > 0:
                self.object_move_state[env_ids_level_3] = self.MOVE_STATE_LINEAR
                # 랜덤 속도
                num_level_3 = len(env_ids_level_3)
                random_speeds = torch.rand(num_level_3, device=self.device) * (0.0015 - 0.0007) + 0.0007
                self.obj_speed[env_ids_level_3] = random_speeds

                self.action_scale_tensor[env_ids_level_3] = 1.0 # 로봇 속도 유지
                self._perform_linear_reset(env_ids_level_3)

            # [신규] Level 4: (Moving Random, Robot Speed 1.5) - 최종
            if len(env_ids_level_4_plus) > 0:
                self.object_move_state[env_ids_level_4_plus] = self.MOVE_STATE_LINEAR

                num_level_4_plus = len(env_ids_level_4_plus)
                random_speeds = torch.rand(num_level_4_plus, device=self.device) * (0.0015 - 0.0007) + 0.0007
                self.obj_speed[env_ids_level_4_plus] = random_speeds
                self.action_scale_tensor[env_ids_level_4_plus] = 1.5 # 로봇 속도 증가
                self._perform_linear_reset(env_ids_level_4_plus)

        else: # training_mode == False (테스트 모드)
            print("action_scale_tensor :", self.action_scale_tensor[env_ids])
            self.action_scale_tensor[env_ids] = 1.0 # (4.0이 적용됨)
            
            if object_move == ObjectMoveType.STATIC:
                self.object_move_state[env_ids] = self.MOVE_STATE_STATIC
                self.obj_speed[env_ids] = 0.0
                self._perform_static_reset(env_ids) 
            
            elif object_move == ObjectMoveType.LINEAR:
                self.object_move_state[env_ids] = self.MOVE_STATE_LINEAR
                self.obj_speed[env_ids] = obj_speed 
                self._perform_linear_reset(env_ids)
            
        self.cfg.current_time = 0
        self._compute_intermediate_values(env_ids)
        
        # [추가] 리셋 시 가시성 및 조인트 버퍼 초기화
        # 리셋 직후에는 "보이지 않음" 상태로 시작하여 움직이지 않도록 합니다.
        self.is_object_visible_mask[env_ids] = False 
        # 리셋된 로봇의 조인트 위치를 버퍼에 저장합니다.
        self.current_joint_pos_buffer[env_ids] = self._robot.data.joint_pos[env_ids]

        super()._reset_idx(env_ids)
    
    # def _get_observations(self) -> dict:
        
    #     dof_pos_scaled = (
    #         2.0
    #         * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
    #         / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
    #         - 1.0
    #     )
        
    #     global robot_action

    #     if yolo_mode: 
    #         hand_pos_real, hand_rot_real = self.get_real_hand_pose() # (XArm API)
    #         if hand_pos_real is None:
    #             print("⚠️ [오류] 실제 로봇 포즈를 읽을 수 없습니다. (get_real_hand_pose 실패)")
    #             gripper_pos_world = self._robot.data.body_link_pos_w[:, self.hand_link_idx]
    #             gripper_rot_world = self._robot.data.body_link_quat_w[:, self.hand_link_idx]
    #         else:
    #             gripper_pos_world = hand_pos_real.repeat(self.num_envs, 1)
    #             gripper_rot_world = hand_rot_real.repeat(self.num_envs, 1)

    #     else: # 시뮬레이션 모드일 때
    #         gripper_pos_world = self._robot.data.body_link_pos_w[:, self.hand_link_idx]
    #         gripper_rot_world = self._robot.data.body_link_quat_w[:, self.hand_link_idx]
            
    #     cam_rot_world_ros, cam_pos_world_ros = tf_combine(
    #         gripper_rot_world, 
    #         gripper_pos_world,
    #         self.R_gripper_to_cam.repeat(self.num_envs, 1), # ⬅️ 2단계에서 만든 변수
    #         self.t_gripper_to_cam.repeat(self.num_envs, 1)  # ⬅️ 2단계에서 만든 변수
    #     )

    #     if yolo_mode:
    #         rclpy.spin_once(self.yolo_node, timeout_sec=0.01)
    #         yolo_pos_raw = self.subscribe_yolo() 

    #         if (yolo_pos_raw is not None):                
    #             yolo_pos_cam_cv = yolo_pos_raw.repeat(self.num_envs, 1)
    #             yolo_pos_cam_ros = torch.zeros_like(yolo_pos_cam_cv)

    #             yolo_pos_cam_ros[:, 0] =  yolo_pos_cam_cv[:, 2]
    #             yolo_pos_cam_ros[:, 1] = -yolo_pos_cam_cv[:, 0]
    #             yolo_pos_cam_ros[:, 2] = -yolo_pos_cam_cv[:, 1]

    #             object_pos_world_abs = tf_vector(cam_rot_world_ros, yolo_pos_cam_ros) + cam_pos_world_ros
                
    #             print("*" * 50)
    #             print(f"yolo_pose_cam_ros : {yolo_pos_cam_ros}")
    #             print(f"YOLO (World Td, NEW): {object_pos_world_abs[0]}") 

    #             self.last_known_world_pos = object_pos_world_abs
    #         else:
    #             robot_action = False
    #             print(f"\rYOLO (World Td, STALE): {self.last_known_world_pos[0]}   ", end="")

    #         final_target_obs = self.last_known_world_pos
        
    #     else: # 시뮬레이션 모드
    #         final_target_obs = self.box_grasp_pos - self.scene.env_origins
    #         self.last_known_world_pos = final_target_obs

    #     robot_base_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device) 
    #     final_target_relative = final_target_obs - robot_base_pos.repeat(self.num_envs, 1)

    #     to_target = self.box_grasp_pos - self.robot_grasp_pos

    #     obs = torch.cat(
    #         (
    #             dof_pos_scaled,
    #             self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
    #             # final_target_relative, # ⬅️ 올바르게 변환된 목표 좌표
    #             to_target,
    #             self._box.data.body_link_pos_w[:, 0, 2].unsqueeze(-1),
    #             self._box.data.body_link_vel_w[:, 0, 2].unsqueeze(-1),
    #         ),
    #         dim=-1,
    #     )
        
    #     return {"policy": torch.clamp(obs, -5.0, 5.0),}

    def _get_observations(self) -> dict:

        # [추가] 현재 조인트 위치 버퍼 업데이트
        # 이 버퍼는 다음 _pre_physics_step에서 "정지" 명령으로 사용됩니다.
        self.current_joint_pos_buffer[:] = self._robot.data.joint_pos
        
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        global robot_action

        if yolo_mode: 
            # 1. [REAL] 실제 로봇 그리퍼 '링크' 위치/회전 가져오기
            hand_pos_real, hand_rot_real = self.get_real_hand_pose() # (XArm API)
            if hand_pos_real is None:
                print("⚠️ [오류] 실제 로봇 포즈를 읽을 수 없습니다. (fallback to sim)")
                # 에러 발생 시 시뮬레이션 값으로 대체
                gripper_link_pos_w = self._robot.data.body_link_pos_w[:, self.hand_link_idx]
                gripper_link_rot_w = self._robot.data.body_link_quat_w[:, self.hand_link_idx]
            else:
                gripper_link_pos_w = hand_pos_real.repeat(self.num_envs, 1)
                gripper_link_rot_w = hand_rot_real.repeat(self.num_envs, 1)
            
            # 2. [REAL] 실제 로봇 '그리퍼 파지점' 계산 (로컬 오프셋 적용)
            # (self.robot_grasp_pos의 실제 로봇 버전)
            _, real_gripper_grasp_pos = tf_combine(
                gripper_link_rot_w, 
                gripper_link_pos_w,
                self.robot_local_grasp_rot, 
                self.robot_local_grasp_pos
            )

            # 3. [REAL] 실제 카메라 좌표계 계산 (YOLO 좌표 변환용)
            cam_rot_world_ros, cam_pos_world_ros = tf_combine(
                gripper_link_rot_w, 
                gripper_link_pos_w,
                self.R_gripper_to_cam.repeat(self.num_envs, 1), 
                self.t_gripper_to_cam.repeat(self.num_envs, 1)
            )

            # 4. [REAL] YOLO로부터 실제 물체 위치 가져오기
            rclpy.spin_once(self.yolo_node, timeout_sec=0.01)
            yolo_pos_raw = self.subscribe_yolo() 

            if (yolo_pos_raw is not None):                
                yolo_pos_cam_cv = yolo_pos_raw.repeat(self.num_envs, 1)
                yolo_pos_cam_ros = torch.zeros_like(yolo_pos_cam_cv)

                yolo_pos_cam_ros[:, 0] =  yolo_pos_cam_cv[:, 2]
                yolo_pos_cam_ros[:, 1] = -yolo_pos_cam_cv[:, 0]
                yolo_pos_cam_ros[:, 2] = -yolo_pos_cam_cv[:, 1]

                # YOLO 좌표를 월드 좌표로 변환
                object_pos_world_abs = tf_vector(cam_rot_world_ros, yolo_pos_cam_ros) + cam_pos_world_ros
                
                # print("*" * 50)
                # print(f"yolo_pose_cam_ros : {yolo_pos_cam_ros}")
                # print(f"YOLO (World Td, NEW): {object_pos_world_abs[0]}") 

                self.last_known_world_pos = object_pos_world_abs
            else:
                robot_action = False
                # print(f"\rYOLO (World Td, STALE): {self.last_known_world_pos[0]}   ", end="")
                # (YOLO가 감지를 놓치면 마지막으로 성공한 위치를 계속 사용)

            # 5. [REAL] 정책에 입력될 실제 관찰 값 계산
            # (self.box_grasp_pos의 실제 로봇 버전)
            # 참고: 물체의 로컬 파지점이 0,0,0으로 가정됨 (__init__의 box_local_pose)
            real_object_grasp_pos = self.last_known_world_pos
            
            # 핵심: 실제 물체 위치 - 실제 그리퍼 위치
            to_target = real_object_grasp_pos - real_gripper_grasp_pos
            object_z_pos = real_object_grasp_pos[:, 2].unsqueeze(-1)
            
            # Sim-to-Real 한계: YOLO는 속도를 제공하지 않으므로 0으로 설정
            object_z_vel = torch.zeros_like(object_z_pos)

        else: # 시뮬레이션 모드 (yolo_mode = False)
            # [SIM] 기존과 동일하게 시뮬레이션 내부의 ground-truth 값 사용
            # (이 값들은 _get_rewards -> _compute_intermediate_values에서 이미 계산됨)
            to_target = self.box_grasp_pos - self.robot_grasp_pos
            object_z_pos = self._box.data.body_link_pos_w[:, 0, 2].unsqueeze(-1)
            object_z_vel = self._box.data.body_link_vel_w[:, 0, 2].unsqueeze(-1)

        
        # (기존의 final_target_relative 및 to_target 계산 로직 삭제)

        # 6. 최종 관찰(Observation) 생성
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,      # <-- 이제 yolo_mode에 따라 올바른 값이 들어감
                object_z_pos,   # <-- 이제 yolo_mode에 따라 올바른 값이 들어감
                object_z_vel,   # <-- 이제 yolo_mode에 따라 올바른 값이 들어감
            ),
            dim=-1,
        )
        
        return {"policy": torch.clamp(obs, -5.0, 5.0),}

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
        gripper_to_box_dist,
        franka_grasp_pos, 
        box_pos_w,    
        franka_grasp_rot,
        box_rot_w,
        box_pos_cam,
        box_rot_cam,
        gripper_forward_axis,
        gripper_up_axis,
    ):
        # 커리큘럼 기반 가중치 설정 (Reward Scales)
        levels = self.current_reward_level
        distance_reward_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["distance"] for l in levels], device=self.device)
        vector_align_reward_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["vector_align"] for l in levels], device=self.device)
        position_align_reward_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["position_align"] for l in levels], device=self.device)
        pview_reward_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["pview"] for l in levels], device=self.device)
        joint_penalty_scale = torch.tensor([reward_curriculum_levels[l.item()]["reward_scales"]["joint_penalty"] for l in levels], device=self.device)
        
        # 커리큘럼 기반 마진 설정
        distance_margin_m = torch.tensor([reward_curriculum_levels[l.item()]["distance_margin"] for l in levels], device=self.device)
        vector_align_margin_rad = torch.tensor([reward_curriculum_levels[l.item()]["vector_align_margin"] for l in levels], device=self.device)
        position_align_margin_m = torch.tensor([reward_curriculum_levels[l.item()]["position_align_margin"] for l in levels], device=self.device)
        pview_margin_m = torch.tensor([reward_curriculum_levels[l.item()]["pview_margin"] for l in levels], device=self.device)
        
        ALPHA_DIST = 1.0 / (distance_margin_m + 1e-6)
        ALPHA_VEC = 1.0 / (vector_align_margin_rad + 1e-6)
        ALPHA_POS = 1.0 / (position_align_margin_m + 1e-6)
        ALPHA_PVIEW = 1.0 / (pview_margin_m + 1e-6)
        
        # [핵심 수정] 전 영역 그래디언트 확보를 위한 탈출 기울기 계수 (beta)
        ESCAPE_GRADIENT = 0.005 
        
        ## R1: 거리 유지 보상 (Distance Reward)
        target_distance = 0.25
        distance_error = torch.abs(gripper_to_box_dist - target_distance)
        distance_reward = (
            torch.exp(-ALPHA_DIST * distance_error) # <--- ALPHA_DIST 동적 적용
            # - ESCAPE_GRADIENT * distance_error
        )

        ## R2: 각도 정렬 보상 (Vector Alignment Reward)
        box_pos_local = box_pos_w - self.scene.env_origins
        obj_x, obj_z = box_pos_local[:, 0], box_pos_local[:, 2]
        x_indices = torch.bucketize(obj_x.contiguous(), self.boundaries_x)
        z_indices = torch.bucketize(obj_z.contiguous(), self.boundaries_z)
        gripper_forward = tf_vector(franka_grasp_rot, gripper_forward_axis)
        actual_angle_rad = torch.asin(gripper_forward[:, 2].clamp(-1.0, 1.0))
        target_angle_rad = torch.deg2rad(self.target_angle_matrix[z_indices, x_indices])
        angle_error_rad = torch.abs(actual_angle_rad - target_angle_rad)
        
        vector_alignment_reward = (
            torch.exp(-ALPHA_VEC * angle_error_rad) # <--- ALPHA_VEC 동적 적용
            # - ESCAPE_GRADIENT * angle_error_rad
        )

        ## R3: 그리퍼 위치 유지 보상 (Position Alignment Reward)
        robot_origin = self.scene.env_origins
        grasp_axis = box_pos_w - robot_origin
        grasp_axis[..., 2] = 0.0
        grasp_axis = torch.nn.functional.normalize(grasp_axis, p=2, dim=-1)
        box_to_gripper_vec_xy = franka_grasp_pos - box_pos_w
        box_to_gripper_vec_xy[..., 2] = 0.0
        gripper_proj_dist = torch.norm(torch.cross(box_to_gripper_vec_xy, grasp_axis, dim=-1), dim=-1)
        
        position_alignment_reward = (
            torch.exp(-ALPHA_POS * gripper_proj_dist) # <--- ALPHA_POS 동적 적용
            # - ESCAPE_GRADIENT * gripper_proj_dist
        )
                
        ## R4: 시야 유지 보상 (PView Reward)
        is_in_front_mask = box_pos_cam[:, 0] < 0 
        center_offset = torch.norm(box_pos_cam[:, [2,1]], dim=-1)
        
        # 카메라 중심 오차에 대한 연속 보상 항 (탈출 기울기 적용)
        pview_positive_reward = (
            torch.exp(-ALPHA_PVIEW * center_offset) # <--- ALPHA_PVIEW 동적 적용
            #- ESCAPE_GRADIENT * center_offset
        )
        
        # 물체가 카메라 뒤에 있을 때 강제 페널티 (R > 0 유지를 위해 1e-6)
        pview_reward = torch.where(is_in_front_mask, pview_positive_reward, torch.full_like(center_offset, 1e-6))
        
        ## P1: 자세 안정성 유지 페널티 (Joint Penalty) - 곱셈 보상과 분리하여 덧셈 페널티로 적용
        joint_deviation = torch.abs(self._robot.data.joint_pos - self.episode_init_joint_pos)
        joint_weights = torch.ones_like(joint_deviation)
        if robot_type == RobotType.UF:
            joint4_idx = self._robot.find_joints(["joint4"])[0]
            joint6_idx = self._robot.find_joints(["joint6"])[0]
            joint_weights[:, joint4_idx] = 0.0
            joint_weights[:, joint6_idx] = 0.0
        weighted_joint_deviation = joint_deviation * joint_weights
        joint_penalty = torch.sum(weighted_joint_deviation, dim=-1)
        joint_penalty = torch.tanh(joint_penalty)

        # --- 3. 최종 보상 계산: 순수 곱셈 구조 복원 ---        
        rewards = (
            torch.pow(distance_reward, distance_reward_scale) *
            torch.pow(vector_alignment_reward, vector_align_reward_scale) *
            torch.pow(position_alignment_reward, position_align_reward_scale) * 
            torch.pow(pview_reward, pview_reward_scale)
        )
        
        self.last_step_reward = rewards.detach()
        
        # print("*" * 50)
        # print("distance_reward :", distance_reward)
        # print("vector_alignment_reward :", vector_alignment_reward)
        # print("position_alignment_reward :", position_alignment_reward)
        # print("pview_reward :", pview_reward)
                
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
        
        
        