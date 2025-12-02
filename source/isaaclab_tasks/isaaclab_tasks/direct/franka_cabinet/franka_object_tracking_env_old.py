# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
torch.set_printoptions(precision=4, sci_mode=False)

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
import time

import pykinect_azure as pykinect
from xarm.wrapper import XArmAPI
from pykinect_azure.k4a import _k4a
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
from pyk4a.calibration import CalibrationType

from collections import deque

class RobotType(Enum):
    FRANKA = "franka"
    UF = "ufactory"
    DOOSAN = "doosan"
robot_type = RobotType.UF

class ObjectMoveType(Enum):
    STATIC = "static"
    CIRCLE = "circle"
    LINEAR = "linear"
    # CURRICULAR = "curricular"
# object_move = ObjectMoveType.STATIC
object_move = ObjectMoveType.LINEAR
# object_move = ObjectMoveType.CURRICULAR

training_mode = False

camera_enable = True
image_publish = True

robot_action = False
robot_init_pose = False
robot_fix = False

UFactory_set_mode = True
real_robot_move = False
yolo_mode = True

add_episode_length = 200
# add_episode_length = 600
# add_episode_length = -400 # 초기 학습 시 episode 길이

vel_ratio = 1.0
obj_speed = 0.0015

rand_pos_range = {
    "x" : (  0.65, 0.85),
    "y" : ( -0.40, 0.40),
    "z" : (  0.08, 0.10),
    
    # "x" : (  0.5, 0.70),
    # "y" : ( -0.35, 0.35),
    # "z" : (  0.08, 0.7),
    
    # "x" : (  0.5, 0.4),
    # "y" : (  -0.3, 0.3),
    # "z" : (  0.6, 0.6),
}

reward_curriculum_levels = [
    # Level 0: (Static) - 기초 단계부터 공격적으로 설정
    {
        "reward_scales": {
            "distance": 4.0,      # [핵심] 1.0 -> 4.0 (접근이 최우선)
            "pview": 1.0,         # 1.0 유지 (Gating을 위해 유지)
            "vector_align": 0.5,  # 0.6 -> 0.5 (각도는 나중에)
            "position_align": 0.5,# 0.8 -> 0.5 (중앙 정렬보다 거리 좁히기가 우선)
            "joint_penalty": 1.0,# [핵심] 1.0 -> 0.05 (팔 움직이는 비용 무료화)
            "blind_penalty": 0.5  # [상향] 0.1 -> 0.5 (놓치면 치명타)
        },
        "success_multiplier": 1.2, "failure_multiplier": 0.8, 
        "y_range" : ( -0.35, 0.35),

        "distance_margin" : 0.15,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # Level 1: (Moving Slow) - 추적 시작
    {
        "reward_scales": {
            "distance": 4.0,      # 접근 강조
            "pview": 1.0,
            "vector_align": 0.5,
            "position_align": 0.5,
            "joint_penalty": 1.0,# 움직임 자유 보장
            "blind_penalty": 0.5  # 놓치지 마라
        },
        "success_multiplier": 1.0, "failure_multiplier": 1.2, 
        "y_range" : ( -0.35, 0.35),

        "distance_margin" : 0.20, 
        "vector_align_margin" : math.radians(25.0),
        "position_align_margin" : 0.25,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # Level 2: (Moving Planar) - 여기가 고비였음
    {
        "reward_scales": {
            "distance": 4.0,      # 멀어지는 물체 잡으려면 보상이 커야 함
            "pview": 1.0,
            "vector_align": 0.5,
            "position_align": 0.5,
            "joint_penalty": 1.0,# 멀리 뻗어도 감점 없게 함
            "blind_penalty": 0.7
        },
        "success_multiplier": 0.9, "failure_multiplier": 1.0, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.15,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35
    },
    # Level 3: (Moving Fast)
    {
        "reward_scales": {
            "distance": 4.0, 
            "pview": 1.0, 
            "vector_align": 0.6, # 상위 레벨이니 정밀도 약간 요구
            "position_align": 0.6, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.0  # 속도가 빠르니 놓치는 거에 더 엄격하게
        },
        "success_multiplier": 0.8, "failure_multiplier": 1.0, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.10,
        "vector_align_margin" : math.radians(15.0),
        "position_align_margin" : 0.15,
        "pview_margin" : 0.20,
        "fail_margin" : 0.30
    },
    # Level 4: (Moving Very Fast)
    {
        "reward_scales": {
            "distance": 4.0, 
            "pview": 1.0, 
            "vector_align": 0.8, 
            "position_align": 0.8, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.5 # 최고 난이도
        },
        "success_multiplier": 1.0, "failure_multiplier": 1.2, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.10,
        "vector_align_margin" : math.radians(10.0),
        "position_align_margin" : 0.10,
        "pview_margin" : 0.15,
        "fail_margin" : 0.30,
    },
]

# vector_align_margin = math.radians(15.0)
# vector_align_margin = math.radians(10.0)
vector_align_margin = math.radians(5.0)

# position_align_margin = 0.15
# position_align_margin = 0.10
position_align_margin = 0.05

# pview_margin = 0.15
# pview_margin = 0.10
pview_margin = 0.05

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
    "x": {"close" : 0.35, "middle": 0.50,"far": 0.65},
    "z": {"bottom": 0.30, "middle": 0.50,"top": 0.65}
}

x_weights = {"far": 5.0, "middle": 1.0, "close" : 4.0}
z_weights = {"top": 4.0, "middle": 1.0, "bottom": 5.0}

zone_activation = {
    "top_close":    True,
    "top_middle":   True,
    "top_far":      True, # << 이 값을 False로 바꾸면 제외됩니다.
    "middle_close": True,
    "middle_middle":True,
    "middle_far":   True,
    "bottom_close": True,
    "bottom_middle":True,
    "bottom_far":   True,
}

# CSV_FILEPATH = "/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/tracking_data.csv"

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
        observation_space = 21
        
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
            # joint_pos={
            #     # "joint1" : math.radians(  0.0),
            #     # "joint2" : math.radians(-66.0),
            #     # "joint3" : math.radians(  8.0),
            #     # "joint4" : math.radians(  0.0),
            #     # "joint5" : math.radians( 15.0),
            #     # "joint6" : math.radians(  0.0),
            # },
            joint_pos = initial_pose,
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ufactory_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit = 87.0,
                
                velocity_limit = 2.61 * vel_ratio,
                stiffness = 2000.0,
                damping = 100.0,
                
                # velocity_limit=0.8,
                # stiffness=80.0,
                # damping=18.0,
            ),
            "ufactory_forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint4", "joint5", "joint6"],
                effort_limit = 87.0,
                
                velocity_limit = 2.61 * vel_ratio,
                stiffness = 2000.0,
                damping = 100.0,
                
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
            usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/Doosan_M1013/M1013_onrobot_with_gripper/M1013_onrobot.usda",
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
                    focal_length=30.0, # 값이 클수록 확대
                    focus_distance=60.0,
                    horizontal_aperture=50.0,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.07, 0.03, -0.13), # 위/아래, 좌/우, 앞/뒤
                    rot=(0.7071, 0.0, 0.0, 0.7071),
                    
                    # rot=(0.0, 0.707, 0.707, 0.0),                    
                    # convention="ROS",
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.25), rot=(0.923, 0, 0, -0.382)),
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
    
    # action_scale = 2.0
    # dof_velocity_scale = 0.05
    
    action_scale = 4.0
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
    cfg: FrankaObjectTrackingEnvCfg

    def __init__(self, cfg: FrankaObjectTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.boundaries_x = torch.tensor([workspace_zones["x"]["middle"], workspace_zones["x"]["far"]], device=self.device)
        self.boundaries_z = torch.tensor([workspace_zones["z"]["middle"], workspace_zones["z"]["top"]], device=self.device)
        
        self.log_counter = 0
        self.LOG_INTERVAL = 5  # 1번의 리셋 묶음마다 한 번씩 로그 출력
        
        # 성능 모니터링을 위한 버퍼
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device)
        
        # 1. 보상 스케일만 조절하는 새로운 커리큘럼 레벨 정의
        self.max_reward_level = len(reward_curriculum_levels) - 1
        self.baseline_avg_reward = 0.1 # 계산된 기준 보상값

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
            self.joint_init_values = [initial_pose[name] for name in self.joint_names]
            
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

        self.box_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs,1)
        )
        
        # self.cube_idx = self._cube.find_bodies("cube")[0][0]
        self.box_idx = self._box.find_bodies("base_link")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.box_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.box_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.box_center = self._box.data.body_link_pos_w[:,0,:].clone()
        
        self.box_pos_cam = torch.zeros((self.num_envs, 4), device=self.device)        
        
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
 
            self.node = rclpy.create_node('camera_publisher')
            self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/camera_info_rect',10)
            self.rgb_publisher = self.node.create_publisher(Image, '/image_rect',10)
            self.depth_publisher = self.node.create_publisher(Image, '/depth',10)
            
            self.bridge = CvBridge()

        if yolo_mode:
            print("[IsaacLab] Initializing YOLO receiver node...")
            self.yolo_msg = None
            self.yolo_pos_raw = None
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
        
        t_cam_to_gripper_mm = torch.tensor(
            [70, 35, -133],
            device=self.device, dtype=torch.float32
        )
        R_cam_to_gripper_quat_ROS = torch.tensor(
            [-0.08403050810066812, 0.7031366469474544, 0.038105476236599455, 0.7050430498264744],
            # [0.0, 0.7071, 0.0, 0.7071],
            # [0.0, 0.0, 0.0, 1.0],
            device=self.device, dtype=torch.float32
        )

        self.R_cam_to_gripper_local = R_cam_to_gripper_quat_ROS.clone()
        self.t_cam_to_gripper_local = (t_cam_to_gripper_mm / 1000.0)

        self.is_object_visible_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_joint_pos_buffer = self._robot.data.joint_pos.clone()

        self.last_known_world_pos = torch.zeros((self.num_envs, 3), 
                                                device=self.device, 
                                                dtype=torch.float32)
        
        self.prev_hand_pos_real = None
        self.prev_time_check = time.time()
        self.SYSTEM_LATENCY = 0.80  

        # [추가] 로봇의 과거 위치/회전을 저장할 버퍼 생성 (최대 200개, 약 2~3초 분량)
        self.pose_history = deque(maxlen=2000)

        self.last_filtered_pos = None 
        self.POSITION_NOISE_THRESHOLD = 0.005  # 5mm 이내의 미세한 변화는 무시 (떨림 방지)

        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.action_smoothing_alpha = 0.1

        self.prev_object_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.avg_distance_error_buf = torch.zeros(self.num_envs, device=self.device)

    def publish_camera_data(self):
        env_id = 0
        
        current_stamp = self.node.get_clock().now().to_msg() 
        current_stamp.sec = current_stamp.sec % 50000
        current_stamp.nanosec = 0
                
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
        if yolo_mode: # camera_type == CameraType.Azure:
            q_cam_in_hand = self.R_cam_to_gripper_local.repeat(self.num_envs, 1)
            cam_offset_pos = self.t_cam_to_gripper_local.repeat(self.num_envs, 1)

        else: # camera_type == CameraType.Sim:
            cam_offset_pos = torch.tensor([0.0, 0.0, 0.1], device=hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.0, -0.7071, 0.0, 0.7071], device=hand_pos.device).repeat(self.num_envs, 1)

        camera_rot_w, camera_pos_w_abs = tf_combine(
            hand_rot,           # R_wg, t_wg
            hand_pos,
            q_cam_in_hand,      # R_gc
            cam_offset_pos      # t_gc
        )
        
        camera_pos_w = camera_pos_w_abs - self.scene.env_origins
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
                
        # 1. 정책(actions)에 따른 잠재적 다음 목표 위치 계산
        current_action_scale = self.action_scale_tensor.unsqueeze(-1) 
        potential_targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * current_action_scale
        potential_targets_clamped = torch.clamp(potential_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        if training_mode:
            self.robot_dof_targets[:] = potential_targets_clamped
        else:            
            hold_targets = self.current_joint_pos_buffer
            
            if not yolo_mode:
                visible_mask_expanded = torch.ones_like(self.is_object_visible_mask.unsqueeze(-1), dtype=torch.bool)
            else:
                visible_mask_expanded = self.is_object_visible_mask.unsqueeze(-1) 
            
            self.robot_dof_targets[:] = torch.where(
                visible_mask_expanded, 
                potential_targets_clamped,  # 시야 O (혹은 YOLO off): 행동 적용
                hold_targets                # 시야 X: 현재 위치 고수 (정지)
            )
        
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
        if object_move == ObjectMoveType.LINEAR and yolo_mode == False:
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
                    xarm_actions = self._robot.data.joint_pos[:, :6]

                    if robot_type == RobotType.UF:
                        joint4_index = self._robot.find_joints(["joint4"])[0]
                        joint6_index = self._robot.find_joints(["joint6"])[0]
                        xarm_actions[:, joint4_index] = 0.0
                        xarm_actions[:, joint6_index] = 0.0

                    angle_cmd = xarm_actions.detach().cpu().numpy().flatten().tolist()

                    ang_speed = 100
                    angmvacc = 10.0

                    rad_speed = math.radians(ang_speed)
                    rad_mvacc = math.radians(angmvacc)

                    if real_robot_move:
                        self.arm.set_servo_angle(angle=angle_cmd, speed=rad_speed, wait=False, is_radian=True, mvacc = rad_mvacc)

            elif robot_action == False and robot_init_pose == False:
                init_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

                for name, val in zip(self.joint_names, self.joint_init_values):
                    index = self._robot.find_joints(name)[0]
                    init_pos[:, index] = val

                self._robot.set_joint_position_target(init_pos)

                joint_err = torch.abs(self._robot.data.joint_pos - init_pos)
                max_err = torch.max(joint_err).item()
                
                if yolo_mode :
                    print("[IsaacLab] Waiting for YOLO detection...")
                    is_visible = self.is_object_visible_mask[0].item()
                    print(f"[IsaacLab] Object is visible: {is_visible}")

                    if (max_err < 0.3) and is_visible: 
                        self.init_cnt += 1
                        print(f"init_cnt : {self.init_cnt}")
                              
                        if self.init_cnt > 5: 
                            robot_action = True
                            robot_init_pose = True
                            
                elif yolo_mode == False and max_err < 0.3 : #and foundationpose_mode == False:
                    self.init_cnt += 1
                    print(f"init_cnt : {self.init_cnt}")
                    if self.init_cnt > 5:
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

    # Refresh the intermediate values after the physics steps
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        
        if yolo_mode: 
            hand_pos_real, hand_rot_real = self.get_real_hand_pose()
            
            if hand_pos_real is None:
                hand_pos_input = self._robot.data.body_link_pos_w[:, self.hand_link_idx]
                hand_rot_input = self._robot.data.body_link_quat_w[:, self.hand_link_idx]
            else:
                hand_pos_input = hand_pos_real.repeat(self.num_envs, 1)
                hand_rot_input = hand_rot_real.repeat(self.num_envs, 1)
            
            sim_gripper_grasp_pos = self.robot_grasp_pos
            sim_gripper_grasp_rot = self.robot_grasp_rot
            
            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(hand_pos_input, hand_rot_input)
            
            real_object_grasp_pos = self.last_known_world_pos
            real_object_grasp_rot = self.box_grasp_rot 

            dist_input = torch.norm(sim_gripper_grasp_pos - real_object_grasp_pos, p=2, dim=-1)
            dist_input = torch.norm(hand_pos_real - real_object_grasp_pos, p=2, dim=-1)
            
            gripper_grasp_pos_input = hand_pos_real  
            object_grasp_pos_input = real_object_grasp_pos
            gripper_grasp_rot_input = hand_rot_real
            object_grasp_rot_input = real_object_grasp_rot
            box_rot_cam_input = object_grasp_rot_input

            real_object_pos_local = real_object_grasp_pos - self.scene.env_origins
            
            if self.yolo_pos_raw is not None:
                yolo_pos_cv = self.yolo_pos_raw.repeat(self.num_envs, 1)
                
                box_pos_cam_input = torch.zeros_like(yolo_pos_cv)
                box_pos_cam_input[:, 0] =  yolo_pos_cv[:, 0] # Z -> X
                box_pos_cam_input[:, 1] =  yolo_pos_cv[:, 1] # X -> -Y
                box_pos_cam_input[:, 2] =  yolo_pos_cv[:, 2] # Y -> -Z

                center_offset = torch.norm(box_pos_cam_input[:, [2, 1]], dim=-1)
                is_in_front = box_pos_cam_input[:, 0] > 0 
                out_of_fov_mask = center_offset > 0.3
                
                self.is_pview_fail = out_of_fov_mask | (~is_in_front)
                self.is_object_visible_mask = ~self.is_pview_fail
                
            else:
                box_pos_cam_input = torch.zeros((self.num_envs, 3), device=self.device)
                self.is_pview_fail = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                self.is_object_visible_mask[:] = False
            
        else: #not yolo
            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.robot_grasp_pos, self.robot_grasp_rot)
            
            box_pos_cam_sim, box_rot_cam_sim = self.world_to_camera_pose(
                camera_pos_w, camera_rot_w,
                self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,
            )
            
            dist_input = torch.norm(self.robot_grasp_pos - self.box_grasp_pos, p=2, dim=-1)
            gripper_grasp_pos_input = self.robot_grasp_pos
            object_grasp_pos_input = self.box_grasp_pos
            gripper_grasp_rot_input = self.robot_grasp_rot
            object_grasp_rot_input = self.box_grasp_rot
            box_pos_cam_input = box_pos_cam_sim
            box_rot_cam_input = box_rot_cam_sim
            
            center_offset = torch.norm(box_pos_cam_input[:, [2, 1]], dim=-1)
            out_of_fov_mask = center_offset > 0.3
            is_behind_mask = box_pos_cam_input[:, 0] > 0 
            self.is_pview_fail = out_of_fov_mask | is_behind_mask
            self.is_object_visible_mask = ~self.is_pview_fail

        # [수정] 함수 호출 시 인자 순서 및 개수 맞춤 (dist_input 제거)
        reward = self._compute_rewards(
            self.actions,
            gripper_grasp_pos_input,     # franka_grasp_pos (Arg 2)
            object_grasp_pos_input,      # box_pos_w (Arg 3)
            gripper_grasp_rot_input,     # franka_grasp_rot (Arg 4, Quaternion)
            object_grasp_rot_input,      # box_rot_w (Arg 5)
            box_pos_cam_input,           # box_pos_cam (Arg 6)
            box_rot_cam_input,           # box_rot_cam (Arg 7)
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
        if not training_mode:
            new_seed = int(time.time() * 1000) % (2**32 - 1)
            torch.manual_seed(new_seed)
        
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
            
            # [!!! 누락된 핵심 코드 추가 !!!] ------------------------------------
            # 로봇의 물리적 위치(joint_pos)와 제어기의 목표값(targets)을 일치시켜야 합니다.
            self.robot_dof_targets[env_ids] = joint_pos 
            # ----------------------------------------------------------------
            
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
                
                # [!!! 누락된 핵심 코드 추가 !!!] ------------------------------------
                self.robot_dof_targets[env_ids] = joint_pos 
                # ----------------------------------------------------------------
                
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
            self.action_scale_tensor[env_ids] = 2.5 # (4.0이 적용됨)
            
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
        
        self.is_object_visible_mask[env_ids] = False 
        self.current_joint_pos_buffer[env_ids] = self._robot.data.joint_pos[env_ids]
                
        super()._reset_idx(env_ids)
    
    def _get_observations(self) -> dict:
        global robot_action
        self.current_joint_pos_buffer[:] = self._robot.data.joint_pos
        
        # 1. Joint Position Scaled
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        # 2. Joint Velocity Scaled
        dof_vel_scaled = self._robot.data.joint_vel * self.cfg.dof_velocity_scale

        # 변수 초기화
        box_pos_cam_obs = torch.zeros((self.num_envs, 3), device=self.device)
        box_pos_w_obs = torch.zeros((self.num_envs, 3), device=self.device)
        box_vel_w_obs = torch.zeros((self.num_envs, 3), device=self.device)

        if yolo_mode: 
            # ------------------------------------------------------------------
            # [Real Robot] YOLO + 좌표 변환으로 Sim Observation 모사
            # ------------------------------------------------------------------
            
            # (1) 현재 로봇 상태 및 과거 상태 보간 (기존 코드 유지)
            hand_pos_real, hand_rot_real = self.get_real_hand_pose() 
            current_time = time.time()
            
            if hand_pos_real is not None:
                self.pose_history.append((current_time, hand_pos_real.clone(), hand_rot_real.clone()))

            target_time = current_time - self.SYSTEM_LATENCY
            past_hand_pos = None
            past_hand_rot = None

            if len(self.pose_history) >= 2:
                found_interval = False
                for i in range(len(self.pose_history) - 1, 0, -1):
                    t_next, pos_next, rot_next = self.pose_history[i]
                    t_prev, pos_prev, rot_prev = self.pose_history[i-1]
                    if t_prev <= target_time <= t_next:
                        alpha = (target_time - t_prev) / (t_next - t_prev + 1e-6)
                        past_hand_pos = (1 - alpha) * pos_prev + alpha * pos_next
                        rot_interp = (1 - alpha) * rot_prev + alpha * rot_next
                        past_hand_rot = kornia.geometry.conversions.normalize_quaternion(rot_interp)
                        found_interval = True
                        break
                if not found_interval:
                    if target_time < self.pose_history[0][0]:
                        past_hand_pos, past_hand_rot = self.pose_history[0][1], self.pose_history[0][2]
                    else:
                        past_hand_pos, past_hand_rot = self.pose_history[-1][1], self.pose_history[-1][2]
            elif len(self.pose_history) == 1:
                past_hand_pos, past_hand_rot = self.pose_history[0][1], self.pose_history[0][2]
            else:
                past_hand_pos = hand_pos_real if hand_pos_real is not None else torch.zeros(3, device=self.device)
                past_hand_rot = hand_rot_real if hand_rot_real is not None else torch.zeros(4, device=self.device)

            if hand_pos_real is None:
                gripper_link_pos_w = self._robot.data.body_link_pos_w[:, self.hand_link_idx]
                gripper_link_rot_w = self._robot.data.body_link_quat_w[:, self.hand_link_idx]
            else:
                gripper_link_pos_w = past_hand_pos.repeat(self.num_envs, 1)
                gripper_link_rot_w = past_hand_rot.repeat(self.num_envs, 1)
            
            cam_rot_world_ros, cam_pos_world_ros = tf_combine(
                gripper_link_rot_w,                                   
                gripper_link_pos_w,                                   
                self.R_cam_to_gripper_local.repeat(self.num_envs, 1), 
                self.t_cam_to_gripper_local.repeat(self.num_envs, 1)  
            )

            # (2) YOLO 데이터 처리
            rclpy.spin_once(self.yolo_node, timeout_sec=0.01)
            self.yolo_pos_raw = self.subscribe_yolo() 

            if (self.yolo_pos_raw is not None):
                self.is_object_visible_mask[:] = True
                
                # YOLO Camera Coord -> ROS Camera Coord (Sim과 동일하게 맞춤)
                yolo_pos_cam_cv = self.yolo_pos_raw.repeat(self.num_envs, 1)
                yolo_pos_cam_ros = torch.zeros_like(yolo_pos_cam_cv)
                yolo_pos_cam_ros[:, 0] =  yolo_pos_cam_cv[:, 2] # Z -> X (Forward)
                yolo_pos_cam_ros[:, 1] = -yolo_pos_cam_cv[:, 0] # X -> -Y (Left)
                yolo_pos_cam_ros[:, 2] = -yolo_pos_cam_cv[:, 1] # Y -> -Z (Up)

                # [Observation 3] Camera Frame Position
                box_pos_cam_obs = yolo_pos_cam_ros

                # Camera Frame -> World Frame 변환
                measured_pos_world_abs = tf_vector(cam_rot_world_ros, yolo_pos_cam_ros) + cam_pos_world_ros
                
                # 노이즈 필터링 (Low-pass like)
                if self.last_filtered_pos is None:
                    object_pos_world_abs = measured_pos_world_abs
                    self.last_filtered_pos = measured_pos_world_abs
                else:
                    diff = torch.norm(measured_pos_world_abs - self.last_filtered_pos, p=2, dim=-1)
                    update_mask = diff > self.POSITION_NOISE_THRESHOLD
                    object_pos_world_abs = torch.where(
                        update_mask.unsqueeze(-1), 
                        measured_pos_world_abs,     
                        self.last_filtered_pos      
                    )
                    self.last_filtered_pos = object_pos_world_abs

                # [Observation 4] World Frame Position
                box_pos_w_obs = object_pos_world_abs
                self.last_known_world_pos = object_pos_world_abs
                
                # Sim 업데이트 (Visual용)
                current_sim_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()
                new_sim_box_pose = torch.cat([object_pos_world_abs, current_sim_box_rot], dim=-1)
                self._box.write_root_pose_to_sim(new_sim_box_pose)
                
            else:
                # 놓쳤을 경우: 카메라는 0(혹은 마지막 값), 월드는 마지막 값 유지
                self.is_object_visible_mask[:] = False
                box_pos_cam_obs = torch.zeros((self.num_envs, 3), device=self.device) # 안 보임 처리
                box_pos_w_obs = self.last_known_world_pos # 마지막 위치 기억

            # [Observation 5] World Frame Velocity (차분 계산)
            # v = (current_pos - prev_pos) / dt
            if torch.sum(self.prev_object_pos_w) == 0: # 초기화 직후 튀는 값 방지
                 self.prev_object_pos_w = box_pos_w_obs.clone()

            box_vel_w_obs = (box_pos_w_obs - self.prev_object_pos_w) / self.dt
            self.prev_object_pos_w = box_pos_w_obs.clone()

            # (Real에서는 노이즈 때문에 속도가 매우 튈 수 있으니 클램핑 추천)
            box_vel_w_obs = torch.clamp(box_vel_w_obs, -2.0, 2.0)

        else: 
            # ------------------------------------------------------------------
            # [Simulation] Ground Truth (제공해주신 코드)
            # ------------------------------------------------------------------
            # [Observation 3] Camera Frame Position (GT 계산)
            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(self.hand_pos, self.hand_rot)
            box_pos_cam_sim, _ = self.world_to_camera_pose(
                camera_pos_w, camera_rot_w,
                self._box.data.body_link_pos_w[:, 0, 0:3] - self.scene.env_origins,
                self.box_grasp_rot 
            )
            box_pos_cam_obs = box_pos_cam_sim

            # [Observation 4] World Frame Position (GT)
            box_pos_w_obs = self._box.data.body_link_pos_w[:, 0, 0:3]

            # [Observation 5] World Frame Velocity (GT)
            box_vel_w_obs = self._box.data.body_link_vel_w[:, 0, 0:3]

        # ------------------------------------------------------------------
        # 최종 Observation 병합
        # 구조: [Joint(N), JointVel(N), CamPos(3), WorldPos(3), WorldVel(3)]
        # ------------------------------------------------------------------
        obs = torch.cat(
            (
                dof_pos_scaled,
                dof_vel_scaled,
                box_pos_cam_obs[:, 0:3], 
                box_pos_w_obs,
                box_vel_w_obs,
            ),
            dim=-1,
        )

        # 디버깅
        # print(f"Obs Shape: {obs.shape}")
        # print(f"Cam Pos: {box_pos_cam_obs[0]}")
        # print(f"World Vel: {box_vel_w_obs[0]}")

        return {"policy": torch.clamp(obs, -5.0, 5.0),}
    
    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.hand_pos = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        self.hand_rot = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]
        
        box_pos_world = self._box.data.body_link_pos_w[env_ids, self.box_idx]
        box_rot_world = self._box.data.body_link_quat_w[env_ids, self.box_idx]
                
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.box_grasp_rot[env_ids],
            self.box_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            self.hand_rot,
            self.hand_pos,
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
        # 커리큘럼 기반 가중치 설정 (Reward Scales)
        levels = self.current_reward_level
        max_idx = self.max_reward_level
        
        distance_reward_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["distance"] for l in levels], device=self.device)
        vector_align_reward_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["vector_align"] for l in levels], device=self.device)
        position_align_reward_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["position_align"] for l in levels], device=self.device)
        pview_reward_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["pview"] for l in levels], device=self.device)
        joint_penalty_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["joint_penalty"] for l in levels], device=self.device)
        blind_penalty_scale = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["reward_scales"]["blind_penalty"] for l in levels], device=self.device)
        
        # 커리큘럼 기반 마진 설정
        distance_margin_m = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["distance_margin"] for l in levels], device=self.device)
        vector_align_margin_rad = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["vector_align_margin"] for l in levels], device=self.device)
        position_align_margin_m = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["position_align_margin"] for l in levels], device=self.device)
        pview_margin_m = torch.tensor([reward_curriculum_levels[min(l.item(), max_idx)]["pview_margin"] for l in levels], device=self.device)
        
        ALPHA_DIST = 1.0 / (distance_margin_m + 1e-6)
        ALPHA_VEC = 1.0 / (vector_align_margin_rad + 1e-6)
        ALPHA_POS = 1.0 / (position_align_margin_m + 1e-6)
        ALPHA_PVIEW = 1.0 / (pview_margin_m + 1e-6)
        
        # ESCAPE_GRADIENT = 0.005 
        
        ## R1: 거리 유지 보상 (Distance Reward) - [카메라 기준 수정]
        target_distance = 0.40
        camera_real_distance = torch.norm(box_pos_cam, dim=-1) 
        distance_error = torch.abs(camera_real_distance - target_distance)
        
        distance_reward = (
            torch.exp(-ALPHA_DIST * distance_error)
        )
                
        self.avg_distance_error_buf += distance_error
        # self.episode_steps_buf += 1.0 # 매 스텝 1씩 증가

        ## R2: 각도 정렬 보상 (Vector Alignment Reward)
        box_pos_local = box_pos_w - self.scene.env_origins
        obj_z = box_pos_local[:, 2]
        
        q_cam_in_hand = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device).repeat(self.num_envs, 1)
        
        deg_bottom = -10.0
        deg_middle =   0.0
        deg_top    =  10.0

        target_angle_deg = torch.full_like(obj_z, deg_middle)
        target_angle_deg = torch.where(obj_z < 0.30, torch.tensor(deg_bottom, device=self.device), target_angle_deg)
        target_angle_deg = torch.where(obj_z >= 0.65, torch.tensor(deg_top, device=self.device), target_angle_deg)

        target_angle_rad = torch.deg2rad(target_angle_deg)

        camera_rot_w = self.quat_mul(franka_grasp_rot, q_cam_in_hand)
        camera_forward_axis_local = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        camera_forward_world = tf_vector(camera_rot_w, camera_forward_axis_local)
        actual_angle_rad = torch.asin(camera_forward_world[:, 2].clamp(-1.0, 1.0))
        
        angle_error_rad = torch.abs(actual_angle_rad - target_angle_rad)
        vector_alignment_reward = torch.exp(-ALPHA_VEC * angle_error_rad)

        ## R3: 그리퍼 위치 유지 보상 (Position Alignment Reward) - [카메라 기준 수정]
        is_in_front_mask = box_pos_cam[:, 2] > 0 
        center_offset_r3 = torch.norm(box_pos_cam[:, [0,1]], dim=-1)
        position_alignment_reward_raw = torch.exp(-ALPHA_POS * center_offset_r3)
        
        position_alignment_reward = torch.where(
            is_in_front_mask, 
            position_alignment_reward_raw, 
            torch.tensor(1e-6, device=self.device)
        )
                
        ## R4: 시야 유지 보상 (PView Reward) - [수정 없음]
        depth = torch.abs(box_pos_cam[:, 2]) + 1e-6
        physical_offset = torch.norm(box_pos_cam[:, [0,1]], dim=-1)
        view_error_ratio = physical_offset / depth

        pview_positive_reward = (
            torch.exp(-ALPHA_PVIEW * view_error_ratio) 
        )
        pview_reward = torch.where(is_in_front_mask, pview_positive_reward, torch.full_like(view_error_ratio, 1e-6))
        
        ## 접근 보상 (Approach Reward) - Shaping Reward
        if not hasattr(self, 'last_error'):
            self.last_error = distance_error.clone()
            
        error_improvement = (self.last_error - distance_error)
        approach_reward = torch.clamp(error_improvement, min=0.0) * 6.0
        self.last_error = distance_error.clone()
        
        ## Joint 5 (손목) 범위 제한 보상 (Soft Limit)
        joint5_val = self._robot.data.joint_pos[:, 4]
        
        # 제한 범위 설정 (라디안 변환)
        limit_min = torch.deg2rad(torch.tensor(-30.0, device=self.device))
        limit_max = torch.deg2rad(torch.tensor(-10.0, device=self.device))
    
        violation_min = torch.clamp(limit_min - joint5_val, min=0.0)
        violation_max = torch.clamp(joint5_val - limit_max, min=0.0)
        
        total_violation = violation_min + violation_max
        joint5_limit_penalty = (total_violation ** 2) * (-joint_penalty_scale)
        
        ## gating 기법
        gating_factor = torch.pow(pview_reward, pview_reward_scale)
        weighted_distance_reward = torch.pow(distance_reward, distance_reward_scale) * gating_factor
        
        task_reward = (
            weighted_distance_reward * # (거리 * 시야)
            torch.pow(vector_alignment_reward, vector_align_reward_scale) *
            torch.pow(position_alignment_reward, position_align_reward_scale)
        )
        
        # 최종 보상 조합 (하이브리드 구조)
        # A. Task Reward (성공 조건들 - 곱하기)
        # task_reward = (
        #     torch.pow(distance_reward, distance_reward_scale) *
        #     torch.pow(vector_alignment_reward, vector_align_reward_scale) *
        #     torch.pow(position_alignment_reward, position_align_reward_scale) * 
        #     torch.pow(pview_reward, pview_reward_scale)
        # )
        
        # B. Blind Penalty (실패 비용 - 빼기)
        # 시야를 놓치면 레벨에 따라 감점 (-0.1 ~ -1.0)
        is_blind = self.is_pview_fail.float()
        blind_penalty = is_blind * (-blind_penalty_scale)
        
        # C. 최종 합산
        # (잘했니?) + (다가갔니?) - (놓쳤니?)
        rewards = task_reward + approach_reward + blind_penalty + joint5_limit_penalty
        self.last_step_reward = rewards.detach()
        
        # print("*" * 50)
        # print("distance_reward :", distance_reward)
        # print("distance_error :", distance_error)
        # print("vector_alignment_reward :", vector_alignment_reward)
        # print("position_alignment_reward :", position_alignment_reward)
        # print("view_error_ratio :", view_error_ratio)
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