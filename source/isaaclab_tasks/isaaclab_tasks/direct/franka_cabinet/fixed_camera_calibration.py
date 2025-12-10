import sys
import time
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
from pyk4a import PyK4A, Config, CalibrationType, ColorResolution, DepthMode

# ================= [설정] =================
ROBOT_IP = "192.168.1.208"
FIXED_CAMERA_DEVICE_ID = 0

# ChArUco 보드 설정 (사용 중인 설정 유지)
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 7
SQUARE_LENGTH = 0.035
MARKER_LENGTH = 0.026
DICTIONARY_VAR = aruco.DICT_6X6_250
# ===========================================

def get_robot_pose_matrix(pose):
    """xArm 포즈 -> 4x4 행렬 (Rotation, Translation 분리 반환)"""
    x, y, z = pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0
    roll, pitch, yaw = pose[3], pose[4], pose[5]
    
    # xArm 회전 (Extrinsic XYZ)
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rot_mat = r.as_matrix()
    return rot_mat, np.array([x, y, z])

def main():
    print(f"Connecting to xArm at {ROBOT_IP}...")
    try:
        arm = XArmAPI(ROBOT_IP)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
    except:
        print("❌ 로봇 연결 실패")
        return

    print(f"Opening Azure Kinect...")
    try:
        k4a = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True
        ), device_id=FIXED_CAMERA_DEVICE_ID)
        k4a.start()
    except:
        print("❌ 카메라 연결 실패")
        arm.disconnect()
        return

    cam_mat = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    dist_coef = k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    dictionary = aruco.getPredefinedDictionary(DICTIONARY_VAR)
    board = aruco.CharucoBoard(
        (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), 
        SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    all_board_corners = board.getChessboardCorners()
    charuco_detector = aruco.CharucoDetector(board)

    # 데이터 수집용 리스트
    R_gripper2base = [] 
    t_gripper2base = [] 
    R_base2gripper = [] # [추가] 역방향 계산용
    t_base2gripper = [] # [추가] 역방향 계산용
    
    R_target2cam = []   
    t_target2cam = []   

    print("\n✅ 캘리브레이션 시작! (다양한 각도로 5장 이상 찍으세요)")
    print("⚠️ 주의: 로봇 손목을 비틀어서(Roll/Pitch) 회전을 섞어야 계산됩니다!")
    print("---------------------------------------------------------")
    print(" [ SPACE ]: 캡처 (Capture)")
    print(" [   C   ]: 계산 (Calculate)")
    print(" [   Q   ]: 종료 (Quit)")
    print("---------------------------------------------------------")

    while True:
        capture = k4a.get_capture()
        if capture.color is None: continue
        frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        
        valid_pose = False
        rvec, tvec = None, None

        if charuco_ids is not None and len(charuco_ids) > 5:
            aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids, (0, 0, 255))
            obj_points = all_board_corners[charuco_ids.flatten()]
            try:
                valid, rvec, tvec = cv2.solvePnP(obj_points, charuco_corners, cam_mat, dist_coef)
                if valid:
                    valid_pose = True
                    cv2.drawFrameAxes(display_frame, cam_mat, dist_coef, rvec, tvec, 0.1)
                    cv2.putText(display_frame, "Ready", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except: pass

        cv2.putText(display_frame, f"Captured: {len(R_gripper2base)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Hand-Eye Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if valid_pose:
                code, pose = arm.get_position(is_radian=False)
                if code == 0:
                    # 1. Base -> Gripper (Original)
                    R_b, t_b = get_robot_pose_matrix(pose)
                    
                    # 2. Gripper -> Base (Inverted for Eye-to-Hand)
                    R_b_inv = R_b.T
                    t_b_inv = -R_b_inv @ t_b

                    # 3. Target -> Camera
                    R_t, _ = cv2.Rodrigues(rvec)
                    t_t = tvec

                    # 리스트 저장
                    R_gripper2base.append(R_b)
                    t_gripper2base.append(t_b)
                    
                    R_base2gripper.append(R_b_inv)
                    t_base2gripper.append(t_b_inv)
                    
                    R_target2cam.append(R_t)
                    t_target2cam.append(t_t)
                    
                    print(f"📸 캡처 {len(R_gripper2base)}/10 (현재 로봇 Z: {t_b[2]:.3f}m)")
                else:
                    print("❌ 로봇 에러")
            else:
                print("⚠️ 보드 인식 안됨")

        elif key == ord('c'):
            if len(R_gripper2base) < 5:
                print("⚠️ 데이터 부족 (최소 5장)")
                continue
                
            print("\n🔄 계산 중... (두 가지 알고리즘 시도)")

            # [방법 1] 표준 입력 (Eye-to-Hand)
            # 보통 Base->Gripper를 넣고, RobotWorldHandEye를 쓰거나 파라미터를 조정함
            try:
                # DANIILIDIS 솔버 사용 (더 안정적)
                R_cam, t_cam = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base, 
                    R_target2cam, t_target2cam, 
                    method=cv2.CALIB_HAND_EYE_DANIILIDIS
                )
                dist1 = np.linalg.norm(t_cam)
                print(f"\n[결과 1] 입력: Base->Gripper (거리: {dist1:.2f}m)")
                print(f"Position: {t_cam.flatten()}")
                
                # [방법 2] 역방향 입력 (Gripper->Base)
                # Eye-to-Hand에서 종종 이 입력이 Base->Camera 변환을 정확히 줌
                R_cam2, t_cam2 = cv2.calibrateHandEye(
                    R_base2gripper, t_base2gripper, 
                    R_target2cam, t_target2cam, 
                    method=cv2.CALIB_HAND_EYE_DANIILIDIS
                )
                dist2 = np.linalg.norm(t_cam2)
                print(f"\n[결과 2] 입력: Gripper->Base (거리: {dist2:.2f}m)")
                print(f"Position: {t_cam2.flatten()}")

                # --- 자동 선택 로직 ---
                # 사용자 측정치(1.4m)와 더 가까운 것 선택
                target_dist = 1.4
                diff1 = abs(dist1 - target_dist)
                diff2 = abs(dist2 - target_dist)
                
                final_R, final_t = (R_cam, t_cam) if diff1 < diff2 else (R_cam2, t_cam2)
                
                print("\n========================================")
                print(f"✅ 최종 선택된 결과 (오차 더 적음)")
                print("========================================")
                print(f"Position (x, y, z):\n{final_t.flatten()}")
                print(f"Rotation Matrix:\n{final_R}")
                
                # 저장
                T_base_cam = np.eye(4)
                T_base_cam[:3, :3] = final_R
                T_base_cam[:3, 3] = final_t.flatten()
                np.savez("final_camera_pose.npz", T_base_cam=T_base_cam)
                print("💾 저장 완료!")
                
            except Exception as e:
                print(f"❌ 계산 오류: {e}")

        elif key == ord('q'):
            break

    k4a.stop()
    cv2.destroyAllWindows()
    arm.disconnect()

if __name__ == "__main__":
    main()