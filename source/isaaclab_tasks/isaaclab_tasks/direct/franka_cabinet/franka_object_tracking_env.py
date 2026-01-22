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
# from isaaclab.utils.math import sample_uniform

from isaaclab.sensors import CameraCfg, Camera
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg #, CollisionPropertiesCfg
from isaaclab.sim.schemas import modify_collision_properties, CollisionPropertiesCfg
from isaaclab.sensors import CameraCfg, Camera, ContactSensor, ContactSensorCfg # [수정] ContactSensor, ContactSensorCfg 추가

# from builtin_interfaces.msg import Time

# from PIL import Image
import cv2
import numpy as np
from enum import Enum
import kornia

import scipy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Point
import math

from cv_bridge import CvBridge
import time

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom
import omni.usd
from isaaclab.sim import spawn_from_usd 

import pykinect_azure as pykinect
from xarm.wrapper import XArmAPI
from pykinect_azure.k4a import _k4a
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
from pyk4a.calibration import CalibrationType

from collections import deque
import threading

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

camera_enable = False
image_publish = False

robot_action = False
robot_init_pose = False
robot_fix = False

UFactory_set_mode = True
real_robot_move = False
yolo_mode = True

grasping_mode = False

foundationpose_mode = False

init_reward = True
reset_flag = True

fixed_camera_enable = False

add_episode_length = 200
# add_episode_length = -800
# add_episode_length = -900
# add_episode_length = -500

rand_pos_range = {
    "x" : (  0.40, 0.75),
    "y" : ( -0.40, 0.40),
    "z" : (  0.08, 0.75),

    # "x" : (  0.3651, 0.3651),
    # "y" : (  0.0323, 0.0323),
    # "z" : (  0.1522, 0.1522),
    
}

reward_curriculum_levels = [
    # Level 0: (Static) - 기초 단계
    {
        "reward_scales": {
            "distance": 6.0,      # [핵심] 1.0 -> 6.0 (접근이 깡패다)
            "pview": 0.5,         # [하향] 1.0 -> 0.5 (시야는 Gating용)
            "vector_align": 0.5,  # 0.6 -> 0.5
            "position_align": 0.5,# 0.8 -> 0.5
            "joint_penalty": 1.0, # [유지] 손목 보호를 위해 1.0 유지
            "blind_penalty": 1.0  # [상향] 0.1 -> 1.0 (놓치면 점수 다 뱉어내라)
        },
        "success_multiplier": 1.2, "failure_multiplier": 0.8, 
        "y_range" : ( -0.40, 0.40),

        "distance_margin" : 0.15,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # Level 1: (Moving Slow) - 추적 시작
    {
        "reward_scales": {
            "distance": 6.0,      # [핵심] 공격적 접근 유도
            "pview": 0.5,
            "vector_align": 0.5,
            "position_align": 0.5,
            "joint_penalty": 1.0, # [유지]
            "blind_penalty": 1.0  # [상향]
        },
        "success_multiplier": 1.0, "failure_multiplier": 1.2, 
        "y_range" : ( -0.35, 0.35),

        "distance_margin" : 0.20, 
        "vector_align_margin" : math.radians(25.0),
        "position_align_margin" : 0.25,
        "pview_margin" : 0.25,
        "fail_margin" : 0.35,
    },
    # Level 2: (Moving Planar) - 여기가 고비였던 구간
    {
        "reward_scales": {
            "distance": 6.0,      # [핵심] 멀어지는 물체 쫓아가려면 보상이 커야 함
            "pview": 0.5,
            "vector_align": 0.5,
            "position_align": 0.5,
            "joint_penalty": 1.0, # [유지]
            "blind_penalty": 1.0  # [상향]
        },
        "success_multiplier": 0.9, "failure_multiplier": 1.0, 
        "y_range": (-0.35, 0.35),

        "distance_margin" : 0.15,
        "vector_align_margin" : math.radians(20.0),
        "position_align_margin" : 0.20,
        "pview_margin" : 0.20,
        "fail_margin" : 0.30
    },
    # Level 3: (Moving Fast)
    {
        "reward_scales": {
            "distance": 6.0, 
            "pview": 0.5, 
            "vector_align": 0.5, 
            "position_align": 0.5, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.0 
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
            "distance": 6.0, 
            "pview": 0.5, 
            "vector_align": 0.5, 
            "position_align": 0.5, 
            "joint_penalty": 1.0, 
            "blind_penalty": 1.5 # [최상위] 더 엄격하게
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

initial_pose = pose_candidate["bottom_close"]
# initial_pose = pose_candidate["middle_close"]
# initial_pose = pose_candidate["top_close"]
# initial_pose = pose_candidate["zero"]

workspace_zones = {
    "x": {"close" : 0.35, "middle": 0.50,"far": 0.65},
    "z": {"bottom": 0.30, "middle": 0.50,"top": 0.65}
}

x_weights = {"far": 5.0, "middle": 1.0, "close" : 4.0}
z_weights = {"top": 1.0, "middle": 1.0, "bottom": 10.0}

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

# CSV_FILEPATH = "/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/tracking_data.csv"

@configclass
class FrankaObjectTrackingEnvCfg(DirectRLEnvCfg):
    ## env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 6
    # observation_space = 21
    # observation_space = 22
    observation_space = 23
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
    
    ## robot
    UF_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/xarm6",
        spawn=sim_utils.UsdFileCfg(
            # usd_path="/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/IsaacLab/ROBOT/xarm6_robot_white/xarm6_robot_white.usd",
            # usd_path="/home/nmail-robot/IsaacLab/ROBOT/xarm6_robot_white/xarm6_robot_white.usd",
            usd_path="/home/nmail-robot/IsaacLab/ROBOT/Ufactory/xarm6/xarm6.usd",

            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=24, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos = initial_pose,
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ufactory_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit = 87.0,
                
                # velocity_limit = 2.61,
                velocity_limit = 3.5,
                stiffness = 2000.0,
                damping = 100.0,
            ),
            "ufactory_forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint4", "joint5"],
                effort_limit = 87.0,
                
                # velocity_limit = 2.61,
                velocity_limit = 3.5,
                stiffness = 2000.0,
                damping = 100.0,
            ),
            "ufactory_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint6"],
                effort_limit = 87.0,
                velocity_limit = 5.0, # 회전 속도 한계 품
                stiffness = 400.0,    # [핵심] 2000 -> 500 (유연하게 만듦)
                damping = 40.0,       # 댐핑 조절
            ),
            "ufactory_hand": ImplicitActuatorCfg(
                # xArm6 그리퍼의 메인 관절 이름이 "drive_joint" 입니다.
                joint_names_expr=[".*drive_joint"], 
                effort_limit=100.0,   # 충분한 힘
                velocity_limit=2.0,   # 적절한 속도
                stiffness=5000.0,     # [중요] 위치 고정을 위해 높은 강성(Stiffness) 필요
                damping=100.0,        # 진동 방지
            ),
        },
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/xarm6/gripper/.*(finger|knuckle|base_link)",
        update_period=0.0, 
        history_length=6, 
        debug_vis=True
    )

    ## camera
    if camera_enable:
        camera = CameraCfg(
            # prim_path="/World/envs/env_.*/xarm6_with_gripper/link6/hand_camera",
            prim_path="/World/envs/env_.*/xarm6/link5/hand_camera",
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
            )
        )
            
    
    ## mustard
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/base_link",
        # prim_path="/World/envs/env_.*/bottle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, 0.08), rot=(0.923, 0, 0, -0.382)),
        spawn=UsdFileCfg(
            usd_path="/home/nmail-robot/objects_usd/google_objects_usd/006_mustard_bottle/006_mustard_bottle.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity = True,
                kinematic_enabled = False,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled = False,  # [핵심] 0으로 설정하면 "아무것과도 충돌하지 않음"
            ),
        ),    
    )
    
    action_scale = 4.0
    dof_velocity_scale = 0.07
    blind_penalty_scale = 0.5

    #time
    current_time = 0.0

class FrankaObjectTrackingEnv(DirectRLEnv):
    cfg: FrankaObjectTrackingEnvCfg

    def __init__(self, cfg: FrankaObjectTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.log_counter = 0
        self.LOG_INTERVAL = 6 
        
        # ----------------------------------------------------------------
        # 3. 커리큘럼 및 평가 기준 설정 (Curriculum & Evaluation)
        # ----------------------------------------------------------------
        self.max_reward_level = len(reward_curriculum_levels) - 1
        self.current_reward_level = torch.full((self.num_envs,), 0, dtype=torch.long, device=self.device)
        self.baseline_avg_reward = 0.05 # 계산된 기준 보상값
        
        # 커리큘럼 계수 (Teacher-Student Learning 관련)
        self.curriculum_factor_k0 = 0.25  # k_c의 초기값 (논문 권장값)
        self.curriculum_factor_kd = 0.997 # k_c의 진전 속도
        self.curriculum_factor_k_c = torch.full((self.num_envs, 1), self.curriculum_factor_k0, device=self.device)
        
        # 에피소드 성공 판단 기준
        self.EVAL_BATCH_SIZE = 10     # 20판마다 성적 평가
        self.PROMOTION_RATE = 0.80     # 승률 90% 이상이면 레벨업 (20판 중 18승)
        self.DEMOTION_RATE = 0.10      # 승률 40% 이하면 레벨다운 (20판 중 8승 이하)

        # 세부 성공 조건
        self.MIN_PVIEW_RATIO = 0.90  # 에피소드 길이의 90% 이상 유지
        self.MAX_DISTANCE_ERROR = 0.05  # 평균 거리 오차 5cm 미만
        self.MIN_TOTAL_REWARD = 200.0    # 에피소드 총 보상 최소 60.0 이상
        
        # 성적 기록용 버퍼
        self.success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.failure_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.avg_distance_error_buf = torch.zeros(self.num_envs, device=self.device)
        self.success_steps_buf = torch.zeros(self.num_envs, device=self.device)
        
        # ----------------------------------------------------------------
        # 4. 로봇 설정 (Robot Config)
        # ----------------------------------------------------------------
        self.joint_names = ["joint1", "joint2", "joint3", "joint4","joint5", "joint6"]
        self.joint_init_values = [initial_pose[name] for name in self.joint_names]
        
        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        self.episode_init_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.current_joint_pos_buffer = self._robot.data.joint_pos.clone()

        self.action_scale_tensor = torch.full(
            (self.num_envs,), 0.5, device=self.device, dtype=torch.float32
        )
        
        # ----------------------------------------------------------------
        # 5. 물체(Box) 이동 및 상태 설정 (Object State)
        # ----------------------------------------------------------------
        self.MOVE_STATE_STATIC = 0
        self.MOVE_STATE_LINEAR = 1

        self.object_move_state = torch.full(
            (self.num_envs,), self.MOVE_STATE_STATIC, dtype=torch.long, device=self.device
        )
        self.obj_speed = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        
        self.speed_change_timer = torch.zeros(self.num_envs, device=self.device)
        self.current_speed_factor = torch.ones(self.num_envs, device=self.device)        
        
        self.level1_axis_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.level2_plane_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ----------------------------------------------------------------
        # 6. 작업 공간 및 초기 자세 설정 (Workspace & Reference Poses)
        # ----------------------------------------------------------------
        self.zone_names_ordered = [
            "top_close", "top_middle", "top_far",
            "middle_close", "middle_middle", "middle_far",
            "bottom_close", "bottom_middle", "bottom_far"
        ]
        
        ref_poses = []
        for name in self.zone_names_ordered:
            pose_dict = pose_candidate[name]
            joint_vals = [pose_dict[jn] for jn in self.joint_names if jn in pose_dict]
            
            if len(joint_vals) < len(self.joint_names):
                joint_vals = [0.0] + joint_vals 
            ref_poses.append(joint_vals)
        self.ref_pose_tensor = torch.tensor(ref_poses, device=self.device, dtype=torch.float32)
        
        # ----------------------------------------------------------------
        # 7. 기하학적 계산 및 링크 인덱싱 (Geometry & Kinematics)
        # ----------------------------------------------------------------
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
        
        stage = get_current_stage()
        
        self.hand_link_idx = self._robot.find_bodies("link6")[0][0]
        self.box_idx = self._box.find_bodies("base_link")[0][0]
        
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
        
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.gripper_up_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.box_z_axis = torch.tensor([0,0,1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs,1)
        )
        
        # ----------------------------------------------------------------
        # 8. 런타임 상태 버퍼 할당 (Runtime Buffers)
        # ----------------------------------------------------------------
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        
        self.box_center = self._box.data.body_link_pos_w[:,0,:].clone()
        self.target_box_pos = torch.stack([
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0],
                torch.rand(self.num_envs, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0],
            ], dim = 1) + self.scene.env_origins
        
        self.new_box_pos_rand = torch.zeros((self.num_envs, 3), device=self.device)
        self.rand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.rand_pos_step = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.current_box_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.current_box_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.box_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.box_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.box_pos_cam = torch.zeros((self.num_envs, 4), device=self.device)  
        
        # 상태 플래그
        self.init_cnt = 0
        self.out_of_fov_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_object_visible_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_publish_time = 0.0
        self.position_error = 0.0
        self.obj_origin_distance = 0.0
                
        rclpy.init()
        
        if image_publish:      
            qos_profile = QoSProfile(depth=10)
            qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

            self.node = rclpy.create_node('camera_publisher')
            self.camera_info_publisher = self.node.create_publisher(CameraInfo, '/camera_info_rect',10)
            self.rgb_publisher = self.node.create_publisher(Image, '/image_rect',10)
            self.depth_publisher = self.node.create_publisher(Image, '/depth',10)
            
            self.bridge = CvBridge()
        
        self.init_cnt = 0

        if UFactory_set_mode:
            ip = "192.168.1.208"
            self.arm = XArmAPI(ip)

            self.arm.motion_enable(enable=True)
            self.arm.set_mode(6) ## joint Online Trajectory Planning
            # self.arm.set_mode(0) ## position Control Mode
            self.arm.set_state(state=0)

            x_max, x_min, y_max, y_min, z_max, z_min = 1200, 50, 600, -600, 1000, 50
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
        self.action_smoothing_alpha = 1.0

        self.prev_object_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.avg_distance_error_buf = torch.zeros(self.num_envs, device=self.device)

        self.prev_box_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_box_pos_c = torch.zeros((self.num_envs, 3), device=self.device)
        
        if yolo_mode:
            self.yolo_msg = None
            self.yolo_pos_raw = None
            self.last_valid_yolo_raw = None 
            self.last_yolo_time = 0.0

            self.yolo_fixed_msg = None
            self.last_valid_yolo_fixed = None
            self.last_yolo_fixed_time = 0.0

            self.yolo_node = rclpy.create_node('yolo_receiver')

            # [핵심] 스레드 간 데이터 공유를 위한 변수와 락(Lock)
            self.yolo_lock = threading.Lock()
            self.shared_yolo_robot = None # 로봇 카메라 좌표 [x, y, z]
            self.shared_yolo_fixed = None # 고정 카메라 좌표 [x, y, z]
            self.shared_yolo_robot_time = 0.0
            
            # 콜백 함수 정의 (여기서 락을 걸고 데이터 저장)
            def robot_cam_callback(msg):
                with self.yolo_lock:
                    self.shared_yolo_robot = torch.tensor([msg.x, msg.y, msg.z], device=self.device)
                    # 메시지 시간 기록 (Latency 체크용)
                    self.shared_yolo_robot_time = time.time()

            def fixed_cam_callback(msg):
                with self.yolo_lock:
                    self.shared_yolo_fixed = torch.tensor([msg.x, msg.y, msg.z], device=self.device)

            # 구독 설정
            self.yolo_node.create_subscription(Point, '/yolo/point', robot_cam_callback, 10)
            self.yolo_node.create_subscription(Point, '/yolo/point_world', fixed_cam_callback, 10)

            # [핵심] YOLO 데이터 수신 전용 스레드 시작
            self.stop_yolo_thread = False
            def _yolo_worker():
                print("[Thread] YOLO Receiver Thread Started!")
                # 별도 실행기(Executor)를 사용하여 스핀
                executor = rclpy.executors.SingleThreadedExecutor()
                executor.add_node(self.yolo_node)
                while not self.stop_yolo_thread:
                    executor.spin_once(timeout_sec=0.001)
                    # time.sleep(0.001) # CPU 점유율 조절 (필요시)
            
            self.yolo_thread = threading.Thread(target=_yolo_worker, daemon=True)
            self.yolo_thread.start()

            # (기존) 로봇 폴링 스레드는 그대로 유지...
            self.real_robot_pose_lock = threading.Lock()
            self.shared_real_hand_pos = None 
            self.shared_real_hand_rot = None
            self.shared_real_joints = None   
            self.stop_polling = False

            def _polling_worker():
                print("[Thread] Worker started. Trying to connect robot...")
                
                while not self.stop_polling:
                    # 1. 위치(TCP) 물어보기
                    code_pos, pose_mm_deg = self.arm.get_position(is_radian=False)
                    
                    if code_pos == 0:
                        # 통신 성공!
                        pos_m = [p / 1000.0 for p in pose_mm_deg[:3]]
                        roll_deg, pitch_deg, yaw_deg = pose_mm_deg[3:]
                        r = scipy.spatial.transform.Rotation.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
                        quat_xyzw = r.as_quat()
                        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

                        with self.real_robot_pose_lock:
                            self.shared_real_hand_pos = torch.tensor([pos_m], device=self.device, dtype=torch.float32) 
                            self.shared_real_hand_rot = torch.tensor([quat_wxyz], device=self.device, dtype=torch.float32)
                    else:
                        # [디버깅] 에러 발생 시 코드 출력 (너무 자주 뜨면 주석 처리)
                        print(f"[Thread Error] get_position failed! code: {code_pos}")

                    # 2. 관절 각도 물어보기
                    code_joint, angles = self.arm.get_servo_angle(is_radian=True)
                    
                    if code_joint == 0:
                        with self.real_robot_pose_lock:
                            self.shared_real_joints = angles[:6] 
                    else:
                        print(f"[Thread Error] get_servo_angle failed! code: {code_joint}")
                    
                    # time.sleep(0.01)
                    time.sleep(0.05)

            self.poll_thread = threading.Thread(target=_polling_worker, daemon=True)
            self.poll_thread.start()
            print("[IsaacLab] Robot polling thread started!")
        
        # [추가] Grasping 상태 관리 플래그
        self.is_grasping = False       # 현재 잡기 동작 수행 중인지 여부
        self.grasp_done = False        # 잡기가 끝났는지 여부
        self.GRASP_THRESHOLD = 0.25    # 잡기 시도 거리 (5cm)

        self.gripper_drive_idx = self._robot.find_joints(".*drive_joint")[0][0]
        self.joint4_index = self._robot.find_joints(["joint4"])[0]
        self.joint6_index = self._robot.find_joints(["joint6"])[0]

        self.target_grasp_width = torch.zeros(self.num_envs, device=self.device)
        self.target_grasp_angle = torch.zeros(self.num_envs, device=self.device)

    def subscribe_yolo(self):
        msg = self.yolo_msg
        self.yolo_msg = None

        if msg is None:
            return None

        return torch.tensor([msg.x, msg.y, msg.z], device=self.device)

    def subscribe_yolo_fixed(self):
        msg = self.yolo_fixed_msg
        self.yolo_fixed_msg = None # 읽으면 초기화 (중복 방지)
        if msg is None: return None
        return torch.tensor([msg.x, msg.y, msg.z], device=self.device)

    def yolo_callback(self,msg):
        self.yolo_msg = msg
    
    def yolo_fixed_callback(self, msg):
        self.yolo_fixed_msg = msg

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
    
    def quat_mul(self, q, r):
        x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        quat = torch.stack((x, y, z, w), dim=-1)
        return kornia.geometry.conversions.normalize_quaternion(quat)
    
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

    def quat_conjugate(self, q):
        q_conj = torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
        return q_conj
    
    def compute_camera_world_pose(self, cam_offset_pos, q_cam_in_hand, hand_pos, hand_rot):
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
        self._robot = Articulation(self.cfg.UF_robot)

        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        if camera_enable:
            self._camera = Camera(self.cfg.camera)
            self.scene.sensors["hand_camera"] = self._camera
        
        self._box = RigidObject(self.cfg.box)
        self.scene.rigid_objects["base_link"] = self._box

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
                
        # current_action_scale = self.action_scale_tensor.unsqueeze(-1) 
        # potential_targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * current_action_scale
        # potential_targets_clamped = torch.clamp(potential_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        current_action_scale = self.action_scale_tensor.unsqueeze(-1) 

        # [수정 1] 전체 12개 중 '팔 관절 6개'만 잘라냅니다.
        arm_targets = self.robot_dof_targets[:, :6]
        arm_speed_scales = self.robot_dof_speed_scales[:6]
        
        # [수정 2] 6개짜리끼리 계산 (이제 에러 안 남)
        new_arm_targets = arm_targets + arm_speed_scales * self.dt * self.actions * current_action_scale
        
        # [수정 3] 리미트도 6개만 가져와서 적용
        lower_limits = self.robot_dof_lower_limits[:6]
        upper_limits = self.robot_dof_upper_limits[:6]
        potential_targets_clamped = torch.clamp(new_arm_targets, lower_limits, upper_limits)

        # [수정 4] 계산된 6개를 원본 배열의 '앞쪽 6칸'에만 업데이트
        self.robot_dof_targets[:, :6] = potential_targets_clamped

        if yolo_mode:
            now_ros = self.yolo_node.get_clock().now()
            current_time_sec = now_ros.nanoseconds / 1e9 

            current_joints = None
            if hasattr(self, 'real_robot_pose_lock'):
                with self.real_robot_pose_lock:
                    if self.shared_real_joints is not None:
                        current_joints = self.shared_real_joints[:] # 복사
            
            if current_joints is not None:
                self.pose_history.append((current_time_sec, current_joints))

        if training_mode:
            self.robot_dof_targets[:,:6] = potential_targets_clamped
        else:            
            hold_targets = self.current_joint_pos_buffer[:, :6]
            
            if not yolo_mode:
                visible_mask_expanded = torch.ones_like(self.is_object_visible_mask.unsqueeze(-1), dtype=torch.bool)
            else:
                visible_mask_expanded = self.is_object_visible_mask.unsqueeze(-1) 
            
            self.robot_dof_targets[:, :6] = torch.where(
                visible_mask_expanded, 
                potential_targets_clamped,  # 시야 O (혹은 YOLO off): 행동 적용
                hold_targets                # 시야 X: 현재 위치 고수 (정지)
            )
        
        if hasattr(self, 'gripper_drive_idx'):
            g_idx = self.gripper_drive_idx
            
            # [핵심] 그리퍼 타겟을 '상한값(Upper Limit)'으로 강제 고정 -> 최대 벌림
            self.robot_dof_targets[:, g_idx] = 0.00

        self.cfg.current_time = self.cfg.current_time + self.dt
        current_time = torch.tensor(self.cfg.current_time, device=self.device, dtype=torch.float32)
        
        if image_publish:   
            self.last_publish_time += self.dt
            if self.last_publish_time >= (1.0 / 15.0):  # 30fps 기준
                self.publish_camera_data()
                rclpy.spin_once(self.node, timeout_sec=0.000)
                self.last_publish_time = 0.0

        # 물체 위치 랜덤 선형 이동 (Per-Environment)
        # 1. LINEAR 상태인 환경의 인덱스를 찾습니다.
        linear_move_mask = (self.object_move_state == self.MOVE_STATE_LINEAR)
        linear_env_ids = torch.where(linear_move_mask)[0]

        if len(linear_env_ids) > 0:
            # 2. 현재 시뮬레이션 상의 '실제' 위치 가져오기 (매우 중요!)
            current_pos_world = self._box.data.root_pos_w[linear_env_ids] # (N, 3)
            
            # 3. 목표 도달 확인 (거리 1cm 미만)
            target_pos = self.target_box_pos[linear_env_ids]
            distance_to_target = torch.norm(target_pos - current_pos_world, p=2, dim=-1)
            reached_target_mask = (distance_to_target < 0.01)

            # --- [A] 목표 도달 시: 새로운 목표 설정 (기존 로직과 유사하지만 변수 정리) ---
            if torch.any(reached_target_mask):
                env_ids_to_update = linear_env_ids[reached_target_mask]
                
                num_to_update = len(env_ids_to_update)
                current_levels = self.current_reward_level[env_ids_to_update]

                # 다음 목표 생성을 위한 랜덤 후보군 & 기준 위치 설정

                # 1. 전체 범위 내 랜덤 좌표 생성 (Level 2, 3용 차기 목표 후보)
                rand_x = torch.rand(num_to_update, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0]
                rand_y = torch.rand(num_to_update, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0]
                rand_z = torch.rand(num_to_update, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0]

                # 2. 기준 위치 가져오기 (축 고정용)
                # "축을 고정한다" = "방금 도달한 목표 위치(직전 Target)를 그대로 유지한다"
                # (현재 물체 위치인 new_box_pos_rand 대신 target_box_pos를 써야 오차가 누적되지 않음)
                curr_target_world = self.target_box_pos[env_ids_to_update]
                curr_target_local = curr_target_world - self.scene.env_origins[env_ids_to_update]

                curr_x = curr_target_local[:, 0]
                curr_y = curr_target_local[:, 1]
                curr_z = curr_target_local[:, 2]

                # 3. 최종 목표 변수 초기화 (기본값: 3D Random)
                final_target_x = rand_x
                final_target_y = rand_y
                final_target_z = rand_z

                mask_lv1 = (current_levels == 1)
                mask_lv2 = (current_levels == 2)

                # [Level 1] 1차원 왕복 운동 (Reciprocating)
                if torch.any(mask_lv1):
                    # 1. Level 1에 해당하는 데이터만 추출 (Size: K)
                    ids_lv1 = env_ids_to_update[mask_lv1]
                    axis_modes = self.level1_axis_mode[ids_lv1]

                    # 2. Level 1용 임시 변수 추출
                    t_x_lv1 = final_target_x[mask_lv1]
                    t_y_lv1 = final_target_y[mask_lv1]
                    t_z_lv1 = final_target_z[mask_lv1]
                    
                    c_x_lv1 = curr_x[mask_lv1]
                    c_y_lv1 = curr_y[mask_lv1]
                    c_z_lv1 = curr_z[mask_lv1]

                    x_min, x_max = rand_pos_range["x"]
                    y_min, y_max = rand_pos_range["y"]
                    z_min, z_max = rand_pos_range["z"]

                    # --- X축 왕복 ---
                    # (axis_modes는 Size K이므로 바로 연산 가능)
                    sub_cond_x = (axis_modes == 0)
                    
                    dist_to_max_x = torch.abs(c_x_lv1 - x_max)
                    dist_to_min_x = torch.abs(c_x_lv1 - x_min)
                    next_x = torch.where(dist_to_max_x < dist_to_min_x, torch.tensor(x_min, device=self.device), torch.tensor(x_max, device=self.device))

                    t_x_lv1 = torch.where(sub_cond_x, next_x, t_x_lv1)
                    t_y_lv1 = torch.where(sub_cond_x, c_y_lv1, t_y_lv1) # Y 고정
                    t_z_lv1 = torch.where(sub_cond_x, c_z_lv1, t_z_lv1) # Z 고정

                    # --- Y축 왕복 ---
                    sub_cond_y = (axis_modes == 1)
                    dist_to_max_y = torch.abs(c_y_lv1 - y_max)
                    dist_to_min_y = torch.abs(c_y_lv1 - y_min)
                    next_y = torch.where(dist_to_max_y < dist_to_min_y, torch.tensor(y_min, device=self.device), torch.tensor(y_max, device=self.device))

                    t_x_lv1 = torch.where(sub_cond_y, c_x_lv1, t_x_lv1) # X 고정
                    t_y_lv1 = torch.where(sub_cond_y, next_y, t_y_lv1)
                    t_z_lv1 = torch.where(sub_cond_y, c_z_lv1, t_z_lv1) # Z 고정

                    # --- Z축 왕복 ---
                    sub_cond_z = (axis_modes == 2)
                    dist_to_max_z = torch.abs(c_z_lv1 - z_max)
                    dist_to_min_z = torch.abs(c_z_lv1 - z_min)
                    next_z = torch.where(dist_to_max_z < dist_to_min_z, torch.tensor(z_min, device=self.device), torch.tensor(z_max, device=self.device))

                    t_x_lv1 = torch.where(sub_cond_z, c_x_lv1, t_x_lv1) # X 고정
                    t_y_lv1 = torch.where(sub_cond_z, c_y_lv1, t_y_lv1) # Y 고정
                    t_z_lv1 = torch.where(sub_cond_z, next_z, t_z_lv1)
                    
                    # 3. 계산된 결과를 원본 텐서에 덮어쓰기
                    final_target_x[mask_lv1] = t_x_lv1
                    final_target_y[mask_lv1] = t_y_lv1
                    final_target_z[mask_lv1] = t_z_lv1

                # [Level 2] 2차원 연속 이동 (Continuous Planar Random Walk) - [수정됨]
                if torch.any(mask_lv2):
                    ids_lv2 = env_ids_to_update[mask_lv2]
                    plane_modes = self.level2_plane_mode[ids_lv2]
                    
                    t_x_lv2 = final_target_x[mask_lv2]
                    t_y_lv2 = final_target_y[mask_lv2]
                    t_z_lv2 = final_target_z[mask_lv2]
                    
                    c_x_lv2 = curr_x[mask_lv2]
                    c_y_lv2 = curr_y[mask_lv2]
                    c_z_lv2 = curr_z[mask_lv2]

                    # XY 평면 (Z 고정)
                    sub_cond_xy = (plane_modes == 0)
                    t_z_lv2 = torch.where(sub_cond_xy, c_z_lv2, t_z_lv2)

                    # XZ 평면 (Y 고정)
                    sub_cond_xz = (plane_modes == 1)
                    t_y_lv2 = torch.where(sub_cond_xz, c_y_lv2, t_y_lv2)

                    # YZ 평면 (X 고정)
                    sub_cond_yz = (plane_modes == 2)
                    t_x_lv2 = torch.where(sub_cond_yz, c_x_lv2, t_x_lv2)
                    
                    # 2. 결과 덮어쓰기
                    final_target_x[mask_lv2] = t_x_lv2
                    final_target_y[mask_lv2] = t_y_lv2
                    final_target_z[mask_lv2] = t_z_lv2

                new_targets = torch.stack([final_target_x, final_target_y, final_target_z], dim=1)

                self.target_box_pos[env_ids_to_update] = new_targets + self.scene.env_origins[env_ids_to_update]

            # 1. 타이머 감소
            self.speed_change_timer[linear_env_ids] -= self.dt
            
            # 2. 타이머가 0 이하로 떨어진 환경들 찾기 (속도를 바꿀 때가 된 환경들)
            time_up_mask = (self.speed_change_timer <= 0.0) & linear_move_mask
            env_ids_to_change_speed = torch.where(time_up_mask)[0]
            
            if len(env_ids_to_change_speed) > 0:
                # 3. 새로운 노이즈 비율 생성 (0.7 ~ 1.3)
                new_noise = (torch.rand(len(env_ids_to_change_speed), device=self.device) * 0.6) + 0.7
                self.current_speed_factor[env_ids_to_change_speed] = new_noise
                
                # 4. 타이머 리셋 (0.5초 ~ 1.5초 사이 랜덤 유지)
                # 즉, 한 번 속도가 변하면 최소 0.5초, 최대 1.5초 동안은 그 속도를 유지함
                new_duration = (torch.rand(len(env_ids_to_change_speed), device=self.device) * 1.0) + 0.5
                self.speed_change_timer[env_ids_to_change_speed] = new_duration
            
            # 1) 방향 벡터 (매 프레임 갱신 - 타겟을 향해 계속 조향해야 하므로)
            target_pos_updated = self.target_box_pos[linear_env_ids]
            direction = target_pos_updated - current_pos_world
            direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
            unit_direction = direction / direction_norm
            
            # 2) 속도 크기 (저장된 factor 사용)
            # 매 프레임 바뀌는 게 아니라, 타이머에 의해 갱신된 값을 계속 사용
            base_speed = self.obj_speed[linear_env_ids].unsqueeze(-1)
            active_noise = self.current_speed_factor[linear_env_ids].unsqueeze(-1)
            
            final_speed = base_speed * active_noise
            
            lin_vel = unit_direction * final_speed
            
            # 3) 물리 엔진 적용
            velocity_command = torch.zeros((len(linear_env_ids), 6), device=self.device)
            velocity_command[:, 0:3] = lin_vel
            self._box.write_root_velocity_to_sim(velocity_command, env_ids=linear_env_ids)

    def _execute_grasp(self, target_pos_m, grasp_angle_rad, grasp_width_m):
        """
        [Joint Interpolation Approach]
        IK 불안정으로 인한 '구불거림'을 제거하기 위해,
        끝점의 관절 각도를 미리 계산한 뒤, 현재 각도와 끝 각도 사이를
        부드럽게 연결(Interpolation)하여 이동합니다.
        """
        print(f"\n[Grasp Triggered] Calculating Smooth Path...")

        if not UFactory_set_mode:
            self.is_grasping = False
            return

        # ------------------------------------------------------------------
        # 1. 현재 로봇 상태(Cartesian + Joint) 읽기
        # ------------------------------------------------------------------
        # 관절 각도 (시작점)
        code, curr_joints = self.arm.get_servo_angle(is_radian=True)
        if code != 0: self.is_grasping = False; return
        start_joints = np.array(curr_joints[:6]) # Numpy로 변환

        # TCP 위치
        code_pos, curr_pose = self.arm.get_position(is_radian=False)
        if code_pos != 0: self.is_grasping = False; return
        
        curr_x, curr_y, curr_z = curr_pose[0], curr_pose[1], curr_pose[2]
        curr_roll, curr_pitch, curr_yaw = curr_pose[3], curr_pose[4], curr_pose[5]

        # ------------------------------------------------------------------
        # 2. 목표지점(Cartesian) 계산
        # ------------------------------------------------------------------
        target_x = target_pos_m[0] * 1000.0
        target_y = target_pos_m[1] * 1000.0
        target_z = max(20.0, target_pos_m[2] * 1000.0)

        vec_x, vec_y, vec_z = target_x - curr_x, target_y - curr_y, target_z - curr_z
        distance = math.sqrt(vec_x**2 + vec_y**2 + vec_z**2) + 1e-6
        unit_x, unit_y, unit_z = vec_x/distance, vec_y/distance, vec_z/distance

        # 깊이 / 후퇴 거리 설정
        EXTRA_DEPTH_MM = 100.0  # 5cm 진입
        RETRACT_DIST = 100.0    # 5cm 후퇴

        # [진입 목표점]
        final_x = target_x + (unit_x * EXTRA_DEPTH_MM)
        final_y = target_y + (unit_y * EXTRA_DEPTH_MM)
        final_z = max(15.0, target_z + (unit_z * EXTRA_DEPTH_MM))

        # [후퇴 목표점] (진입점에서 뒤로 빠짐)
        retract_x = final_x - (unit_x * RETRACT_DIST)
        retract_y = final_y - (unit_y * RETRACT_DIST)
        retract_z = final_z - (unit_z * RETRACT_DIST)

        open_target = 850
        close_target = max(0, grasp_width_m * 10000.0)

        # ------------------------------------------------------------------
        # [Helper] 관절 보간 경로 생성 (핵심 수정)
        # ------------------------------------------------------------------
        def generate_joint_trajectory(start_j, target_pose_list, speed_mm_s):
            """
            Cartesian 목표점(target_pose_list)에 대한 IK를 '딱 한번' 풀고,
            현재 관절각(start_j)에서 목표 관절각까지 선형 보간하여 리스트 생성.
            -> 경로가 절대 구불거리지 않음.
            """
            # 1. 목표점의 IK 솔루션 구하기
            code, target_j = self.arm.get_inverse_kinematics(
                pose=target_pose_list, 
                input_is_radian=False, return_is_radian=True
            )
            
            if code != 0:
                print("[Error] IK Solution Not Found!")
                return None
            
            target_j = np.array(target_j[:6])
            
            # [Safety] 관절이 너무 많이 회전(Flip)하는지 체크 (30도 이상 튀면 위험)
            diff_deg = np.degrees(np.abs(target_j - start_j))
            if np.max(diff_deg) > 45.0:
                print(f"[Warning] IK Solution FLIP detected! Max diff: {np.max(diff_deg):.1f} deg. Aborting.")
                return None

            # 2. 이동 거리 및 스텝 수 계산
            # Cartesian 거리로 시간 계산 (정확하진 않지만 근사치)
            # 현재 위치와 목표 위치 사이 거리
            dist = math.sqrt((target_pose_list[0]-curr_pose[0])**2 + 
                             (target_pose_list[1]-curr_pose[1])**2 + 
                             (target_pose_list[2]-curr_pose[2])**2)
            
            duration = dist / speed_mm_s
            steps = int(duration * 50) # 50Hz 제어
            if steps < 1: steps = 1

            # 3. 관절각 선형 보간 (Linear Interpolation)
            traj = []
            for i in range(1, steps + 1):
                t = i / steps
                # 시작 각도와 목표 각도 사이를 t 비율만큼 섞음
                interp_j = start_j + (target_j - start_j) * t
                traj.append(interp_j.tolist())
                
            return traj, target_j # 마지막 관절각도 반환 (다음 경로 시작점용)

        def execute_joints(joint_list):
            """50Hz 속도로 전송"""
            period = 1.0 / 50.0
            for j_pos in joint_list:
                start_t = time.perf_counter()
                self.arm.set_servo_angle(angle=j_pos, is_radian=True, wait=False)
                elapsed = time.perf_counter() - start_t
                if period > elapsed:
                    time.sleep(period - elapsed)

        def wait_mode6(duration_sec):
            steps = int(duration_sec / 0.02)
            c, j = self.arm.get_servo_angle(is_radian=True)
            if c != 0: return
            cur_j = j[:6]
            for _ in range(steps):
                self.arm.set_servo_angle(angle=cur_j, is_radian=True, wait=False)
                time.sleep(0.02)

        # ------------------------------------------------------------------
        # [Step 0] 경로 생성 (Joint Interpolation)
        # ------------------------------------------------------------------
        # 진입 목표 (Orientation은 현재 자세 유지)
        final_pose_list = [final_x, final_y, final_z, curr_roll, curr_pitch, curr_yaw]
        
        # 1. 진입 경로 생성 (속도 40mm/s)
        res_approach = generate_joint_trajectory(start_joints, final_pose_list, speed_mm_s=40.0)
        if res_approach is None:
            self.is_grasping = False; return
        approach_traj, joints_at_grasp = res_approach

        # 2. 후퇴 경로 생성 (속도 200mm/s)
        # 후퇴는 '잡은 위치(joints_at_grasp)'에서 시작해야 함
        retract_pose_list = [retract_x, retract_y, retract_z, curr_roll, curr_pitch, curr_yaw]
        res_retract = generate_joint_trajectory(joints_at_grasp, retract_pose_list, speed_mm_s=200.0)
        if res_retract is None:
            self.is_grasping = False; return
        retract_traj, _ = res_retract

        # ------------------------------------------------------------------
        # 실행 시퀀스
        # ------------------------------------------------------------------
        self.arm.set_gripper_position(int(open_target), wait=False)

        # 1. 진입
        print(f"[Step 1] Smooth Approach (Speed: 40mm/s)...")
        execute_joints(approach_traj)

        print("[Step 2] Stabilizing...")
        wait_mode6(1.0)

        # 2. 잡기
        print("[Step 3] Grasping...")
        self.arm.set_gripper_position(int(close_target), wait=True)
        wait_mode6(0.5)

        # 3. 후퇴
        print(f"[Step 4] Fast Retract (Speed: 200mm/s)...")
        execute_joints(retract_traj)

        # 4. 확인
        print("[Step 5] Checking Visibility...")
        wait_mode6(1.0)

        # --- 결과 판단 ---
        object_still_visible = False
        current_time = time.time()
        with self.yolo_lock:
            if self.shared_yolo_robot is not None:
                if (current_time - getattr(self, 'shared_yolo_robot_time', 0.0)) < 1.0: 
                    object_still_visible = True

        if object_still_visible:
            print(">>> [Retry] Object still detected!")
        else:
            print(">>> [Success] Drop & Reset.")
            wait_mode6(1.0)
            self.arm.set_gripper_position(int(open_target), wait=True)

        # --- 동기화 ---
        print(">>> Syncing Simulation...")
        code, current_joints = self.arm.get_servo_angle(is_radian=True)
        if code == 0:
            real_joints_tensor = torch.tensor(current_joints[:6], device=self.device)
            self.robot_dof_targets[:] = real_joints_tensor
            self.current_joint_pos_buffer[:] = real_joints_tensor
            if training_mode: self.episode_init_joint_pos[:] = real_joints_tensor

        self.grasp_done = False   
        self.is_grasping = False
        print(">>> Resume Tracking.")

    def _apply_action(self):
        global robot_action
        global robot_init_pose
        
        # [Grasping State Check]
        # 이미 잡기를 수행 중이거나 완료했다면 더 이상 Tracking 명령을 보내지 않음
        if self.is_grasping or self.grasp_done:
            return

        target_pos = self.robot_dof_targets.clone()
        
        # xArm 6축 제어를 위한 마스킹 (4, 6번 관절 0 처리)
        target_pos[:, self.joint4_index] = 0.0
        target_pos[:, self.joint6_index] = 0.0
        target_pos[:, 7:] = 0.0
        
        # ------------------------------------------------------------------
        # TEST Mode & Robot NOT Fixed
        # ------------------------------------------------------------------
        if training_mode == False and robot_fix == False:
            
            # [Case 1] 로봇 제어 시작됨 (Tracking Active)
            if robot_action and robot_init_pose:
                
                # ==========================================================
                # [Grasping Trigger] 25cm 진입 시 잡기 시작
                # ==========================================================
                # 1. 로봇(TCP)과 물체(Target) 사이의 거리 계산
                # env_id=0 기준
                if hasattr(self, 'robot_grasp_pos') and hasattr(self, 'last_known_world_pos'):
                    current_dist = torch.norm(self.robot_grasp_pos[0] - self.last_known_world_pos[0], p=2)
                    
                    # 2. 거리가 임계값(0.25m) 이내인지 확인
                    if current_dist < self.GRASP_THRESHOLD and real_robot_move and grasping_mode: 
                        self.is_grasping = True # 플래그 설정 -> Tracking 중단
                        
                        target_obj_pos = self.last_known_world_pos[0].cpu().numpy()
                        
                        # [너비 설정] 머스타드 병 기준 약 5.5cm (0.055m)
                        detected_width = 0.055 
                        
                        # 잡기 시퀀스 함수 호출 (Blocking)
                        self._execute_grasp(target_obj_pos, grasp_angle_rad=0.0, grasp_width_m=detected_width)
                        
                        self.grasp_done = True
                        self.is_grasping = False
                        return # 함수 종료 (더 이상 Tracking 명령 전송 안 함)
                # ==========================================================

                # [Existing] Tracking Logic (Grasping 조건이 아닐 때)
                self._robot.set_joint_position_target(target_pos)

                if UFactory_set_mode:
                    xarm_actions = self._robot.data.joint_pos[:, :6]

                    if robot_type == RobotType.UF:
                        xarm_actions[:, self.joint4_index] = 0.0
                        xarm_actions[:, self.joint6_index] = 0.0

                    angle_cmd = xarm_actions.detach().cpu().numpy().flatten().tolist()

                    # [기존 설정 유지] 로봇이 튀지 않도록 원래 속도값 사용
                    ang_speed = 150    
                    angmvacc = 60.0    
                    
                    rad_speed = math.radians(ang_speed)
                    rad_mvacc = math.radians(angmvacc)

                    if real_robot_move:
                        # 실제 로봇에 명령 전송 (wait=False로 비동기 연속 전송)
                        self.arm.set_servo_angle(
                            angle=angle_cmd, 
                            speed=rad_speed, 
                            wait=False, 
                            is_radian=True, 
                            mvacc=rad_mvacc
                        )

            # [Case 2] 로봇 제어 대기 중 (Start 조건 확인)
            elif robot_action == False and robot_init_pose == False:
                # 시뮬레이션 로봇은 초기 자세로 이동 시도 (Visual)
                init_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
                for name, val in zip(self.joint_names, self.joint_init_values):
                    index = self._robot.find_joints(name)[0]
                    init_pos[:, index] = val
                self._robot.set_joint_position_target(init_pos)
                
                # 오차 계산
                joint_err = torch.abs(self._robot.data.joint_pos - init_pos)
                max_err = torch.max(joint_err).item()
                
                # YOLO 모드일 때 물체가 보여야 시작
                if yolo_mode:
                    is_visible = self.is_object_visible_mask[0].item()
                    
                    if is_visible: 
                        self.init_cnt += 1
                        # print(f"init_cnt : {self.init_cnt}") # 로그 너무 많으면 주석 처리

                        if self.init_cnt > 5: 
                            # 시작 시 튀는 현상 방지를 위해 현재 위치로 타겟 동기화
                            self.robot_dof_targets[:] = self.current_joint_pos_buffer.clone()
                            
                            robot_action = True
                            robot_init_pose = True

                # YOLO 모드가 아닐 때 (Sim Only) 위치만 맞으면 시작
                elif yolo_mode == False and max_err < 0.3:
                    self.init_cnt += 1
                    # print(f"init_cnt : {self.init_cnt}")
                    if self.init_cnt > 5:
                        robot_init_pose = True
                        robot_action = True
                               
        # ------------------------------------------------------------------
        # Training Mode
        # ------------------------------------------------------------------
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
        if hasattr(self, 'is_pview_fail'):
            is_high_level = (self.current_reward_level >= 7)
            current_fail = self.is_pview_fail
            
            self.out_of_fov_counter = torch.where(
                is_high_level & current_fail,
                self.out_of_fov_counter + 1,
                torch.zeros_like(self.out_of_fov_counter)
            )
        
            MAX_CONSECUTIVE_FAIL_STEPS = 45
            terminated = (self.out_of_fov_counter >= MAX_CONSECUTIVE_FAIL_STEPS)
    
        else:
            terminated = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        
        truncated = self.episode_length_buf >= self.max_episode_length + add_episode_length

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        
        if yolo_mode: 
            real_object_grasp_pos = self.last_known_world_pos
            real_object_grasp_rot = self.box_grasp_rot 

            object_grasp_pos_input = real_object_grasp_pos
            object_grasp_rot_input = real_object_grasp_rot

            box_rot_cam_input = object_grasp_rot_input
            
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
            cam_offset_pos = torch.tensor([0.07, 0.03, -0.13], device=self.robot_grasp_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.robot_grasp_pos.device).repeat(self.num_envs, 1)

            cam_offset_pos = torch.tensor([0.07, 0.03, -0.13], device=self.robot_grasp_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.robot_grasp_pos.device).repeat(self.num_envs, 1)

            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(cam_offset_pos, q_cam_in_hand, self.robot_grasp_pos, self.robot_grasp_rot)
            
            box_pos_cam_sim, box_rot_cam_sim = self.world_to_camera_pose(
                camera_pos_w, camera_rot_w,
                self.box_grasp_pos - self.scene.env_origins, self.box_grasp_rot,
            )
            
            # dist_input = torch.norm(self.robot_grasp_pos - self.box_grasp_pos, p=2, dim=-1)
            box_pos_cam_input = box_pos_cam_sim
            box_rot_cam_input = box_rot_cam_sim

            object_grasp_pos_input = self.box_grasp_pos
            object_grasp_rot_input = self.box_grasp_rot

            center_offset = torch.norm(box_pos_cam_input[:, [2, 1]], dim=-1)
            out_of_fov_mask = center_offset > 0.3
            is_behind_mask = box_pos_cam_input[:, 0] > 0 
            self.is_pview_fail = out_of_fov_mask | is_behind_mask
            self.is_object_visible_mask = ~self.is_pview_fail

        gripper_grasp_pos_input = self.robot_grasp_pos
        gripper_grasp_rot_input = self.robot_grasp_rot

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
        
        # 1. 현재 레벨 확인
        current_levels = self.current_reward_level[env_ids]
        mask_lv1 = (current_levels == 1)
        mask_lv2 = (current_levels == 2)
                
        # [공통] 새로운 시작 위치(Start Position) 생성 (완전 랜덤)
        # X, Y, Z 모두 전체 범위 내에서 균등 랜덤 생성 (가중치 제거됨)
        start_x = torch.rand(num_resets, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0]
        start_y = torch.rand(num_resets, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0]
        start_z = torch.rand(num_resets, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0]
        
        start_pos = torch.stack([start_x, start_y, start_z], dim=1)
        
        # [중요] 물체 강제 이동 (Teleport) - 시뮬레이션 상의 위치를 즉시 변경
        random_angles = torch.rand(num_resets, device=self.device) * 2 * torch.pi
        reset_rot = torch.stack([
            torch.cos(random_angles / 2),
            torch.zeros(num_resets, device=self.device),
            torch.zeros(num_resets, device=self.device),
            torch.sin(random_angles / 2)  
        ], dim=1)
        
        reset_pose = torch.cat([start_pos + self.scene.env_origins[env_ids], reset_rot], dim=-1)
        zero_velocity = torch.zeros((num_resets, 6), device=self.device)
        
        self._box.write_root_pose_to_sim(reset_pose, env_ids=env_ids)
        self._box.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        start_pos_world = start_pos + self.scene.env_origins[env_ids]
        
        # 내부 상태 변수 초기화 (현재 위치 = 방금 이동시킨 랜덤 위치)
        self.new_box_pos_rand[env_ids] = start_pos_world
        self.current_box_pos[env_ids] = start_pos_world
        self.current_box_rot[env_ids] = reset_rot

        target_pos = start_pos.clone()
        
        # --- Level 1: 1차원 왕복 운동 초기화 ---
        if torch.any(mask_lv1):
            ids_lv1 = env_ids[mask_lv1]
            num_lv1 = len(ids_lv1)
            
            # 1. 축 랜덤 선택 (0:X, 1:Y, 2:Z) 및 저장 (새로운 에피소드용 축)
            axis_mode = torch.randint(0, 3, (num_lv1,), device=self.device)
            self.level1_axis_mode[ids_lv1] = axis_mode
            
            # 2. 범위 가져오기
            x_min, x_max = rand_pos_range["x"]
            y_min, y_max = rand_pos_range["y"]
            z_min, z_max = rand_pos_range["z"]
            
            # 3. 첫 번째 목표 설정: "현재 위치에서 가장 먼 끝점"으로 설정
            # (그래야 시작하자마자 긴 거리를 이동하며 왕복을 시작함)
            
            # X축 왕복 (Y, Z는 시작 위치 고정)
            cond_x = (axis_mode == 0)
            # 현재 x가 중간보다 작으면 Max로, 크면 Min으로
            target_x_dest = torch.where(start_x[mask_lv1] < (x_min + x_max)/2, torch.tensor(x_max, device=self.device), torch.tensor(x_min, device=self.device))
            target_pos[mask_lv1, 0] = torch.where(cond_x, target_x_dest, target_pos[mask_lv1, 0])
            
            # Y축 왕복 (X, Z는 시작 위치 고정)
            cond_y = (axis_mode == 1)
            target_y_dest = torch.where(start_y[mask_lv1] < (y_min + y_max)/2, torch.tensor(y_max, device=self.device), torch.tensor(y_min, device=self.device))
            target_pos[mask_lv1, 1] = torch.where(cond_y, target_y_dest, target_pos[mask_lv1, 1])
            
            # Z축 왕복 (X, Y는 시작 위치 고정)
            cond_z = (axis_mode == 2)
            target_z_dest = torch.where(start_z[mask_lv1] < (z_min + z_max)/2, torch.tensor(z_max, device=self.device), torch.tensor(z_min, device=self.device))
            target_pos[mask_lv1, 2] = torch.where(cond_z, target_z_dest, target_pos[mask_lv1, 2])

        # --- Level 2+: 3차원 연속 이동 초기화 ---
        if torch.any(mask_lv2):
            ids_lv2 = env_ids[mask_lv2]
            num_lv2 = len(ids_lv2)
            
            # 1. 평면 랜덤 선택 및 저장 (이번 에피소드 동안 고정)
            # 0:XY(Z고정), 1:XZ(Y고정), 2:YZ(X고정)
            plane_mode = torch.randint(0, 3, (num_lv2,), device=self.device)
            self.level2_plane_mode[ids_lv2] = plane_mode
            
            # 2. 시작 위치(Start)는 3D 공간상 완전 랜덤 (순간이동용)
            # (이미 위 공통 로직에서 start_pos가 계산되어 있음)
            
            # 3. 첫 번째 목표(Target) 설정
            # "선택된 평면 위"에 있어야 하므로, 고정축은 start_pos 값을 유지하고 나머지만 랜덤
            
            # 랜덤 후보군 생성
            rand_tx = torch.rand(num_lv2, device=self.device) * (rand_pos_range["x"][1] - rand_pos_range["x"][0]) + rand_pos_range["x"][0]
            rand_ty = torch.rand(num_lv2, device=self.device) * (rand_pos_range["y"][1] - rand_pos_range["y"][0]) + rand_pos_range["y"][0]
            rand_tz = torch.rand(num_lv2, device=self.device) * (rand_pos_range["z"][1] - rand_pos_range["z"][0]) + rand_pos_range["z"][0]
            
            # start_pos는 공통 로직에서 계산된 값 사용
            # 해당 환경(ids_lv2)에 맞는 start_pos 추출
            # start_pos 텐서는 (num_resets, 3) 크기이므로 마스크를 사용해 추출해야 함
            # 주의: start_pos는 전체 env_ids에 대한 것이므로, mask_lv2를 그대로 쓰면 됨
            
            start_x_lv2 = start_pos[mask_lv2, 0]
            start_y_lv2 = start_pos[mask_lv2, 1]
            start_z_lv2 = start_pos[mask_lv2, 2]
            
            # 평면 구속 조건 적용
            # XY 평면 (Z 고정)
            cond_xy = (plane_mode == 0)
            final_tz = torch.where(cond_xy, start_z_lv2, rand_tz)
            
            # XZ 평면 (Y 고정)
            cond_xz = (plane_mode == 1)
            final_ty = torch.where(cond_xz, start_y_lv2, rand_ty)
            
            # YZ 평면 (X 고정)
            cond_yz = (plane_mode == 2)
            final_tx = torch.where(cond_yz, start_x_lv2, rand_tx)
            
            # 나머지 축은 랜덤값 그대로 사용 (조건이 False인 곳은 rand값이 들어감)
            final_tx = torch.where(cond_yz, final_tx, rand_tx)
            final_ty = torch.where(cond_xz, final_ty, rand_ty)
            final_tz = torch.where(cond_xy, final_tz, rand_tz)

            # 최종 목표 할당
            target_pos[mask_lv2, 0] = final_tx
            target_pos[mask_lv2, 1] = final_ty
            target_pos[mask_lv2, 2] = final_tz
        
        # [공통] 최종 목표 적용 및 이동 벡터 계산
        self.target_box_pos[env_ids] = target_pos + self.scene.env_origins[env_ids]

        # direction = self.target_box_pos[env_ids] - self.new_box_pos_rand[env_ids]
        # direction_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-6
        # speed = self.obj_speed[env_ids].unsqueeze(-1)
        
        # self.rand_pos_step[env_ids] = (direction / direction_norm * speed)

        # [로봇 자세 초기화] (기존 코드 유지)
        if training_mode:
            joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
            joint1_idx = self._robot.find_joints(["joint1"])[0]
            
            YAW_CANDIDATE_ANGLES = { 15.0: math.radians(15.0), 45.0: math.radians(45.0), 75.0: math.radians(75.0) }
            ANGLE_BOUNDARIES = [30.0, 60.0, 90.0]
            
            for i, env_id in enumerate(env_ids):
                object_pos_local = start_pos[i] # 위에서 만든 랜덤 위치 사용
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
            
            # [핵심] 로봇 목표값 동기화 (튀는 현상 방지)
            self.robot_dof_targets[env_ids] = joint_pos 
            self.episode_init_joint_pos[env_ids] = joint_pos
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        global reset_flag 
        
        if reset_flag:
            if training_mode == False:
                reset_flag = False
            
            # [신규 로직] 복합 조건 평가
            actual_lengths = self.episode_length_buf[env_ids].float() + 1e-6

            # 1. 성공률 (Tracking Ratio) 계산
            # (시야 유지 스텝 / 전체 에피소드 길이)
            success_ratio = self.success_steps_buf[env_ids] / actual_lengths
            pass_ratio = success_ratio >= self.MIN_PVIEW_RATIO

            # 2. 평균 거리 오차 계산
            avg_distance = self.avg_distance_error_buf[env_ids] / actual_lengths
            pass_distance = avg_distance <= self.MAX_DISTANCE_ERROR

            # 3. 총 보상 계산
            total_reward = self.episode_reward_buf[env_ids]
            pass_reward = total_reward >= self.MIN_TOTAL_REWARD

            # [최종 성공 판단] 3가지 조건을 모두(AND) 만족해야 성공
            success_mask_reward = pass_ratio & pass_distance #& pass_reward
            failure_mask_reward = ~success_mask_reward

            #승률 기반 레벨 이동 로직
            # 1. 성공/실패 각각 카운트 누적 (초기화 없음)
            self.success_count[env_ids] += success_mask_reward.long()
            self.failure_count[env_ids] += failure_mask_reward.long()

            # 2. 총 시도 횟수 계산
            total_attempts = self.success_count[env_ids] + self.failure_count[env_ids]

            # 3. 평가 주기(EVAL_BATCH_SIZE)가 된 환경들만 골라내기
            # 예: 20판을 채운 환경들
            check_mask = (total_attempts >= self.EVAL_BATCH_SIZE)

            if torch.any(check_mask):
                check_env_ids = env_ids[check_mask]

                # 승률 계산 (성공 / 전체)
                current_success_rate = self.success_count[check_env_ids].float() / total_attempts[check_env_ids].float()

                # A. 승급 심사 (90% 이상)
                promote_mask = current_success_rate >= self.PROMOTION_RATE
                if torch.any(promote_mask):
                    promote_ids = check_env_ids[promote_mask]
                    new_levels = self.current_reward_level[promote_ids] + 1
                    
                    # [핵심] 최대 레벨을 초과하지 않도록 clamp
                    new_levels = torch.clamp(new_levels, max=self.max_reward_level)
                    self.current_reward_level[promote_ids] = (self.current_reward_level[promote_ids] + 1).clamp(max=self.max_reward_level)

                # B. 강등 심사 (40% 이하)
                demote_mask = current_success_rate < self.DEMOTION_RATE
                if torch.any(demote_mask):
                    demote_ids = check_env_ids[demote_mask]
                    
                    # [수정] 현재 레벨 - 1 계산
                    new_levels = self.current_reward_level[demote_ids] - 1
                    
                    # [핵심] 0 미만으로 내려가지 않도록 clamp
                    new_levels = torch.clamp(new_levels, min=0)
                    
                    # 레벨다운
                    self.current_reward_level[demote_ids] = (self.current_reward_level[demote_ids] - 1).clamp(min=0)

                # [중요] 평가가 끝난 환경들은 카운터 리셋 (다음 20판을 위해)
                # 20판 채우면 무조건 리셋해야, 과거의 성적에 발목 잡히지 않음
                self.success_count[check_env_ids] = 0
                self.failure_count[check_env_ids] = 0

            self.log_counter += 5
            if 1 :#self.log_counter % self.LOG_INTERVAL == 0:
                level_counts = torch.bincount(self.current_reward_level, minlength=self.max_reward_level + 1)

                # [기존] 1. 평균 승률 계산
                total_attempts = self.success_count + self.failure_count
                valid_mask = total_attempts > 0
                if torch.any(valid_mask):
                    avg_rate = (self.success_count[valid_mask].float() / total_attempts[valid_mask].float()).mean().item()
                else:
                    avg_rate = 0.0

                # [기존] 현재 평균 몇 판째인지 계산
                avg_episodes = total_attempts.float().mean().item()
                max_episodes = total_attempts.max().item()

                # [기존] 전체 평균 통계 계산
                if len(env_ids) > 0:
                    current_actual_lengths = self.episode_length_buf[env_ids].float() + 1e-6

                    # A. 평균 성공률
                    avg_success_ratio_val = (self.success_steps_buf[env_ids] / current_actual_lengths).mean().item()
                    # B. 평균 거리 오차
                    avg_distance_error_val = (self.avg_distance_error_buf[env_ids] / current_actual_lengths).mean().item()
                    # C. 평균 총 보상
                    avg_total_reward_val = self.episode_reward_buf[env_ids].mean().item()
                else:
                    avg_success_ratio_val = 0.0
                    avg_distance_error_val = 0.0
                    avg_total_reward_val = 0.0

                print("=" * 80) # 구분선 길이 약간 늘림
                print(f"📊 Curriculum Level Distribution (Total: {self.num_envs})")
                print(f"🔄 Progress: {avg_episodes:.1f} / {self.EVAL_BATCH_SIZE} episodes (Max: {max_episodes})")
                print(f"📈 Level Up/Down Win Rate: {avg_rate * 100:.2f}% (Target: {self.PROMOTION_RATE*100:.0f}%)")
                print("-" * 80)

                # 1. 전체 평균 출력
                print(f"🔍 [Global Stats] Avg of {len(env_ids)} reset envs:")
                print(f"   Total  | Success: {avg_success_ratio_val * 100:6.2f}% | Dist: {avg_distance_error_val * 100:5.2f} cm | Reward: {avg_total_reward_val:6.1f}")

                print("-" * 80)
                print("🔍 [Level-wise Stats]")

                # 2. [추가됨] 각 레벨별 통계 계산 및 출력
                if len(env_ids) > 0:
                    current_levels_reset = self.current_reward_level[env_ids] # 현재 리셋되는 환경들의 레벨

                    # 각 레벨을 순회하며 통계 계산
                    for lvl in range(self.max_reward_level + 1):
                        # 현재 리셋된 환경들 중, 해당 레벨(lvl)인 것들만 마스킹
                        lvl_mask = (current_levels_reset == lvl)
                        lvl_count = torch.sum(lvl_mask).item()

                        if lvl_count > 0:
                            # 해당 레벨의 데이터 추출
                            # env_ids[lvl_mask]는 안됨. env_ids 자체가 인덱스이므로, 불리언 마스크를 사용하여 필터링해야 함
                            # 올바른 방법: 값을 추출한 뒤 마스킹

                            lvl_lengths = current_actual_lengths[lvl_mask]

                            # A. 성공률
                            lvl_success = (self.success_steps_buf[env_ids][lvl_mask] / lvl_lengths).mean().item()
                            # B. 거리 오차
                            lvl_dist = (self.avg_distance_error_buf[env_ids][lvl_mask] / lvl_lengths).mean().item()
                            # C. 총 보상
                            lvl_reward = self.episode_reward_buf[env_ids][lvl_mask].mean().item()

                            print(f"   Level {lvl} ({lvl_count:3d}) | Success: {lvl_success * 100:6.2f}% | Dist: {lvl_dist * 100:5.2f} cm | Reward: {lvl_reward:6.1f}")

                print("-" * 80)

                # 레벨 분포 바 그래프 출력 (기존 코드)
                for level_idx, count in enumerate(level_counts):
                    count_val = count.item()
                    ratio = (count_val / self.num_envs) * 100
                    bar = "#" * int(ratio / 5) 
                    print(f"  Level {level_idx}: {count_val:4d} envs ({ratio:5.1f}%) | {bar}")
                print("=" * 80)

                self.log_counter = 0 # 카운터 초기화

            self.episode_reward_buf[env_ids] = 0.0
            self.avg_distance_error_buf[env_ids] = 0.0
            self.success_steps_buf[env_ids] = 0.0

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

                    self.robot_dof_targets[env_ids] = joint_pos 

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
                    self.action_scale_tensor[env_ids_level_0] = 1.0 
                    self._perform_static_reset(env_ids_level_0) 

                # [신규] Level 1: (Moving 0.0005, Robot Speed 0.5) - 물체 이동 먼저
                if len(env_ids_level_1) > 0:
                    self.object_move_state[env_ids_level_1] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids_level_1] = 0.05 # 물체 이동 시작
                    self.action_scale_tensor[env_ids_level_1] = 1.0 # 로봇 속도 유지
                    self._perform_linear_reset(env_ids_level_1)

                # [신규] Level 2: (Moving 0.0005, Robot Speed 1.0) - 다음 로봇 속도 증가
                if len(env_ids_level_2) > 0:
                    self.object_move_state[env_ids_level_2] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids_level_2] = 0.07
                    self.action_scale_tensor[env_ids_level_2] = 1.0 # 로봇 속도 증가
                    self._perform_linear_reset(env_ids_level_2)

                # [신규] Level 3: (Moving Random, Robot Speed 1.0) - 다음 물체 속도 증가
                if len(env_ids_level_3) > 0:
                    self.object_move_state[env_ids_level_3] = self.MOVE_STATE_LINEAR
                    num_level_3 = len(env_ids_level_3)
                    random_speeds = torch.rand(num_level_3, device=self.device) * (0.0015 - 0.0007) + 0.0007
                    self.obj_speed[env_ids_level_3] = 0.1
                    self.action_scale_tensor[env_ids_level_3] = 1.0 # 로봇 속도 유지
                    self._perform_linear_reset(env_ids_level_3)

                # [신규] Level 4: (Moving Random, Robot Speed 1.5) - 최종
                if len(env_ids_level_4_plus) > 0:
                    self.object_move_state[env_ids_level_4_plus] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids_level_4_plus] = 0.15
                    self.action_scale_tensor[env_ids_level_4_plus] = 1.0 # 로봇 속도 증가
                    self._perform_linear_reset(env_ids_level_4_plus)

            else: # training_mode == False (테스트 모드)
                self.action_scale_tensor[env_ids] = 3.0 # (4.0이 적용됨)

                if object_move == ObjectMoveType.STATIC:
                    self.object_move_state[env_ids] = self.MOVE_STATE_STATIC
                    self.obj_speed[env_ids] = 0.0
                    self._perform_static_reset(env_ids) 

                elif object_move == ObjectMoveType.LINEAR:
                    self.object_move_state[env_ids] = self.MOVE_STATE_LINEAR
                    self.obj_speed[env_ids] = 0.3 
                    self._perform_linear_reset(env_ids)

            self.cfg.current_time = 0
            self._compute_intermediate_values(env_ids)

            self.is_object_visible_mask[env_ids] = False 
            self.current_joint_pos_buffer[env_ids] = self._robot.data.joint_pos[env_ids]
            self.out_of_fov_counter[env_ids] = 0

            if hasattr(self, 'last_error'):
                current_dist = torch.norm(self.robot_grasp_pos[env_ids] - self.box_grasp_pos[env_ids], p=2, dim=-1)
                self.last_error[env_ids] = current_dist
        
        super()._reset_idx(env_ids)

        self._compute_intermediate_values(env_ids)

        if yolo_mode:
            # 1. World Position 초기화 (Real)
            # _get_observations에서 계산된 최신 월드 좌표를 가져옴
            if hasattr(self, 'last_known_world_pos'):
                current_pos_w = self.last_known_world_pos[env_ids].clone()
            else:
                current_pos_w = torch.zeros((len(env_ids), 3), device=self.device)
            
            self.prev_box_pos_w[env_ids] = current_pos_w

            # 2. Camera Position 초기화 (Real)
            # YOLO raw 데이터가 있다면 좌표 변환(CV -> ROS) 후 저장
            if self.yolo_pos_raw is not None:
                yolo_cv = self.yolo_pos_raw.repeat(len(env_ids), 1)
            else:
                yolo_cv = torch.zeros((len(env_ids), 3), device=self.device)

            self.prev_box_pos_c[env_ids] = yolo_cv
        else:
            # 1. 월드 좌표 초기화
            current_pos_w = self._box.data.body_link_pos_w[env_ids, 0, 0:3] - self.scene.env_origins[env_ids]
            self.prev_box_pos_w[env_ids] = current_pos_w.clone()

            cam_offset_pos = torch.tensor([0.07, 0.03, -0.13], device=self.hand_pos[env_ids].device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.hand_pos[env_ids].device).repeat(self.num_envs, 1)

            camera_pos_w, camera_rot_w = self.compute_camera_world_pose(cam_offset_pos, q_cam_in_hand, self.hand_pos[env_ids], self.hand_rot[env_ids])
            current_pos_c, _ = self.world_to_camera_pose(
                camera_pos_w, camera_rot_w,
                current_pos_w, 
                self.box_grasp_rot[env_ids]
            )
            self.prev_box_pos_c[env_ids] = current_pos_c[:, 0:3].clone()

    def _get_observations(self) -> dict:
        # 1. 시뮬레이션 로봇 관절값
        self.current_joint_pos_buffer[:] = self._robot.data.joint_pos
        
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        box_pos_w_cur = torch.zeros((self.num_envs, 3), device=self.device)

        # ------------------------------------------------------------------
        # [A] YOLO Mode (Offset Compensation Switching)
        # ------------------------------------------------------------------
        if yolo_mode:
            # [신규] 오차 저장용 변수 초기화
            if not hasattr(self, 'camera_offset'):
                self.camera_offset = torch.zeros((self.num_envs, 3), device=self.device)
            
            # 스레드 데이터 수신
            new_yolo_raw = None
            raw_fixed_cam_world = None
            last_msg_time = 0.0 # 메시지 수신 시간

            with self.yolo_lock:
                if self.shared_yolo_robot is not None:
                    new_yolo_raw = self.shared_yolo_robot.clone()
                    last_msg_time = getattr(self, 'shared_yolo_robot_time', 0.0)

                if self.shared_yolo_fixed is not None:
                    raw_fixed_cam_world = self.shared_yolo_fixed.clone()
            
            current_system_time = time.time()
            if not hasattr(self, 'last_detection_system_time'):
                self.last_detection_system_time = 0.0

            # ---------------------------------------------------------
            # [Step 1] Robot Camera World Pos 계산
            # ---------------------------------------------------------

            # 즉, 0.2초 동안 새 데이터가 안 왔으면 "놓쳤다"고 판단하고 강제로 버림
            if (current_system_time - last_msg_time) > 0.2:
                new_yolo_raw = None  # <--- 강제로 None으로 만듬!

            pos_w_from_robot = None
            dist_from_hand = 10.0

            if new_yolo_raw is not None:
                self.last_detection_system_time = current_system_time
                yolo_cv = new_yolo_raw.repeat(self.num_envs, 1)
                dist_from_hand = yolo_cv[0, 2].item()

                current_real_pos_local = None 
                current_real_rot = None
                
                if hasattr(self, 'real_robot_pose_lock'):
                    with self.real_robot_pose_lock:
                        if self.shared_real_hand_pos is not None:
                            current_real_pos_local = self.shared_real_hand_pos.clone()
                            current_real_rot = self.shared_real_hand_rot.clone()
                
                if current_real_pos_local is None:
                    hand_pos_input_global = self._robot.data.body_link_pos_w[:, self.hand_link_idx]
                    hand_rot_input = self._robot.data.body_link_quat_w[:, self.hand_link_idx]
                else:
                    hand_pos_input_global = current_real_pos_local.repeat(self.num_envs, 1) + self.scene.env_origins
                    hand_rot_input = current_real_rot.repeat(self.num_envs, 1)

                q_cam_real = self.R_cam_to_gripper_local.repeat(self.num_envs, 1)
                t_cam_real = self.t_cam_to_gripper_local.repeat(self.num_envs, 1)

                camera_rot_w_real, camera_pos_w_real_abs = tf_combine(
                    hand_rot_input, hand_pos_input_global, q_cam_real, t_cam_real
                )
                
                box_pos_c_ros = torch.zeros_like(yolo_cv)
                box_pos_c_ros[:, 0] =  yolo_cv[:, 2] 
                box_pos_c_ros[:, 1] = -yolo_cv[:, 0] 
                box_pos_c_ros[:, 2] = -yolo_cv[:, 1] 

                pos_w_from_robot, _ = self.camera_to_world_pose(
                    camera_pos_w_real_abs, camera_rot_w_real, 
                    box_pos_c_ros,  
                    self.box_grasp_rot 
                )

            # ---------------------------------------------------------
            # [Step 2] Fixed Camera World Pos 계산
            # ---------------------------------------------------------
            pos_w_from_fixed = None
            if raw_fixed_cam_world is not None and fixed_camera_enable:
                # 로봇 카메라가 죽었을 때를 대비해 시간 갱신 (고정 카메라라도 있으면 살아있는 것임)
                if pos_w_from_robot is None:
                    self.last_detection_system_time = current_system_time
                pos_w_from_fixed = raw_fixed_cam_world.repeat(self.num_envs, 1) + self.scene.env_origins

            # ---------------------------------------------------------
            # [Step 3] Seamless Switching Logic (핵심 수정)
            # ---------------------------------------------------------
            target_pos_raw = None
            
            # 로봇 카메라 신뢰 조건 (> 15cm)
            is_robot_reliable = (pos_w_from_robot is not None) or (dist_from_hand < 0.15)

            # print("*"* 50)
            # print("pos_w_from_robot :", pos_w_from_robot)
            # print("is_robot_reliable :", is_robot_reliable)
            # print("pos_w_from_fixed :", pos_w_from_fixed)
            # print("yolo_cv :", yolo_cv)

            # [상황 A] 둘 다 데이터가 양호함 -> 오차(Offset) 업데이트
            if is_robot_reliable and (pos_w_from_fixed is not None):
                # 로봇 카메라를 메인으로 사용
                target_pos_raw = pos_w_from_robot
                self.is_object_visible_mask[:] = True
                
                # [핵심] "고정 카메라가 로봇 카메라에 비해 얼마나 틀어져 있는지" 계산 후 저장
                current_offset = pos_w_from_robot - pos_w_from_fixed
                
                # 오차를 부드럽게 업데이트 (급격한 변화 방지, Alpha 0.1)
                self.camera_offset = (self.camera_offset * 0.9) + (current_offset * 0.1)

            # [상황 B] 로봇 카메라만 양호함
            elif is_robot_reliable:
                target_pos_raw = pos_w_from_robot
                self.is_object_visible_mask[:] = True
                # 고정 카메라 데이터가 없으므로 Offset 업데이트 안 함 (기존 값 유지)

            # [상황 C] 로봇 카메라 불안정/없음 + 고정 카메라는 있음 -> 전환!
            elif pos_w_from_fixed is not None:
                self.is_object_visible_mask[:] = True
                
                # [핵심] 고정 카메라 값에 아까 저장해둔 오차를 더해서 사용
                # 이렇게 하면 로봇 카메라 좌표계와 거의 일치하는 값으로 변환됨
                target_pos_raw = pos_w_from_fixed + self.camera_offset

            # [상황 D] 둘 다 없음
            else:
                if pos_w_from_robot is not None: # 거리 15cm 이하지만 이거라도 써야 함
                     target_pos_raw = pos_w_from_robot
                     self.is_object_visible_mask[:] = True
                else:
                    # 진짜 없음 -> 마지막 위치 유지 or 타임아웃
                    time_since = current_system_time - self.last_detection_system_time
                    if hasattr(self, 'last_known_world_pos'):
                        target_pos_raw = self.last_known_world_pos.clone()
                    else:
                        target_pos_raw = self._box.data.body_link_pos_w[:, 0, 0:3]

                    if time_since > 0.5: self.is_object_visible_mask[:] = False 
                    else: self.is_object_visible_mask[:] = True 

            # ---------------------------------------------------------
            # [Step 4] Smoothing Filter (EMA)
            # ---------------------------------------------------------
            if not hasattr(self, 'filtered_pos_w'):
                self.filtered_pos_w = target_pos_raw.clone()
            
            # 점프 감지 (15cm 이상 튀면 필터 리셋)
            dist_jump = torch.norm(target_pos_raw - self.filtered_pos_w, dim=-1)
            jump_mask = dist_jump > 0.15 
            
            ALPHA = 0.5 
            self.filtered_pos_w = (target_pos_raw * ALPHA) + (self.filtered_pos_w * (1.0 - ALPHA))
            
            if torch.any(jump_mask):
                self.filtered_pos_w[jump_mask] = target_pos_raw[jump_mask]

            final_world_pos = self.filtered_pos_w.clone()
            
            self.last_known_world_pos = final_world_pos.clone()
            box_pos_w_global = final_world_pos.clone()

            # [Step 5 ~ 7] (기존 변환 및 시각화 코드 동일)
            current_sim_box_rot = self._box.data.body_link_quat_w[:, 0, :].clone()
            new_sim_box_pose = torch.cat([box_pos_w_global, current_sim_box_rot], dim=-1)
            self._box.write_root_pose_to_sim(new_sim_box_pose)

            box_pos_w_cur = box_pos_w_global - self.scene.env_origins

            hand_pos_sim_global = self._robot.data.body_link_pos_w[:, self.hand_link_idx]
            hand_rot_sim = self._robot.data.body_link_quat_w[:, self.hand_link_idx]
            sim_offset_pos = torch.tensor([0.07, 0.03, -0.13], device=self.device).repeat(self.num_envs, 1)
            sim_q_cam = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device).repeat(self.num_envs, 1)

            camera_rot_w_sim, camera_pos_w_sim_abs = tf_combine(
                hand_rot_sim, hand_pos_sim_global, sim_q_cam, sim_offset_pos
            )
            
            box_pos_c_cur_calculated, _ = self.world_to_camera_pose(
                camera_pos_w_sim_abs, camera_rot_w_sim,
                box_pos_w_global, 
                self.box_grasp_rot 
            )
            box_pos_c_cur = box_pos_c_cur_calculated[:, 0:3]

        # ------------------------------------------------------------------
        # [B] Sim Mode
        # ------------------------------------------------------------------
        else:
            box_pos_w_cur = self._box.data.body_link_pos_w[:, 0, 0:3] - self.scene.env_origins
            cam_offset_pos = torch.tensor([0.07, 0.03, -0.13], device=self.hand_pos.device).repeat(self.num_envs, 1)
            q_cam_in_hand = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.hand_pos.device).repeat(self.num_envs, 1)
            camera_rot_w, camera_pos_w_abs = tf_combine(self.hand_rot, self.hand_pos, q_cam_in_hand, cam_offset_pos)
            box_pos_w_global_sim = box_pos_w_cur + self.scene.env_origins
            box_pos_c_cur_full, _ = self.world_to_camera_pose(camera_pos_w_abs, camera_rot_w, box_pos_w_global_sim, self.box_grasp_rot)
            box_pos_c_cur = box_pos_c_cur_full[:, 0:3]

        # [C] Observation 구성 (동일)
        if torch.sum(self.prev_box_pos_w) == 0:
            self.prev_box_pos_w = box_pos_w_cur.clone()
            self.prev_box_pos_c = box_pos_c_cur.clone()
            if yolo_mode:
                self.filtered_pos_w = (box_pos_w_cur + self.scene.env_origins).clone()

        target_dist = 0.40
        current_depth = box_pos_c_cur[:, 2]
        z_error = (current_depth - target_dist).unsqueeze(-1)
        xy_offset = torch.norm(box_pos_c_cur[:, 0:2], p=2, dim=-1).unsqueeze(-1)

        obs = torch.cat(
            (
                dof_pos_scaled[:,:6],                                    
                # self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                (self._robot.data.joint_vel * self.cfg.dof_velocity_scale)[:, :6],
                box_pos_c_cur,           
                box_pos_w_cur,
                self.prev_box_pos_w,     
                z_error,
                xy_offset,
            ),
            dim=-1,
        )
        
        self.prev_box_pos_w = box_pos_w_cur.clone()
        self.prev_box_pos_c = box_pos_c_cur.clone()
        
        return {"policy": torch.clamp(obs, -5.0, 5.0),}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # self.hand_pos = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        # self.hand_rot = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]
        
        self.hand_pos[env_ids] = self._robot.data.body_link_pos_w[env_ids, self.hand_link_idx]
        self.hand_rot[env_ids] = self._robot.data.body_link_quat_w[env_ids, self.hand_link_idx]
        
        box_pos_world = self._box.data.body_link_pos_w[env_ids, self.box_idx]
        box_rot_world = self._box.data.body_link_quat_w[env_ids, self.box_idx]
                
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.box_grasp_rot[env_ids],
            self.box_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            # self.hand_rot,
            # self.hand_pos,
            self.hand_rot[env_ids],          # [수정] env_ids 추가!
            self.hand_pos[env_ids],          # [수정] env_ids 추가!
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