import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI
from pyk4a import PyK4A, Config, CalibrationType
from pyk4a import ColorResolution, DepthMode

# ====================================================================
# --- 1. 사용자 설정 (USER CONFIGURATION) ---
# ====================================================================
CHECKERBOARD_DIMS = (8, 6) 
SQUARE_SIZE_MM = 25.0
ROBOT_IP = "192.168.1.208"
NUM_SAMPLES = 15
# ====================================================================

# [ 💥 1. 카메라 좌표계 변환 행렬 (CV -> ROS) 💥 ]
# (X:우, Y:하, Z:전) -> (X:전, Y:좌, Z:상)
R_CV_TO_ROS = np.array([
    [0,  0,  1],
    [-1, 0,  0],
    [0, -1,  0]
], dtype=np.float32)

# ====================================================================
# [ 💥 2. 로봇 좌표계 변환 행렬 (XArm -> ROS) 💥 ]
# XArm (X:전, Y:우, Z:상) -> ROS (X:전, Y:좌, Z:상)
# 이 변환은 T_gripper2base 전체에 적용됩니다.
# T_ROS = T_fix @ T_XArm @ inv(T_fix)
R_XARM_TO_ROS_FIX = np.array([
    [1,  0,  0],
    [0, -1,  0], # Y축 반전
    [0,  0, -1]  # Z축 반전 (Euler 변환 방식 때문에 필요)
])
# 오일러 각 변환을 위한 보정 (Y, Z 부호 반전)
# t_ROS = T_fix @ t_XArm
# R_ROS = R_fix @ R_XArm @ R_fix_inv
# (XArm_Y -> -ROS_Y), (XArm_Z -> ROS_Z)
# (XArm_roll -> ROS_roll), (XArm_pitch -> -ROS_pitch), (XArm_yaw -> -ROS_yaw)
# ====================================================================

def main():
    """ Hand-Eye 캘리브레이션 메인 함수 """

    objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM # (mm 단위)

    arm = None
    k4a = None
    try:
        print("🤖 로봇에 연결하는 중...")
        arm = XArmAPI(ROBOT_IP, request_timeout=5) 
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(1)
        if not arm.connected:
            print("❌ 로봇 연결 실패.")
            return
        print("✅ 로봇 연결 성공. (좌표계: XArm)")

        print("\n📷 Azure Kinect 카메라 초기화 중...")
        k4a = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        k4a.start()
        print("✅ 카메라 초기화 성공. (좌표계: OpenCV)")

        calibration = k4a.calibration
        camera_matrix = calibration.get_camera_matrix(CalibrationType.COLOR)
        dist_coeffs = calibration.get_distortion_coefficients(CalibrationType.COLOR)

        R_gripper2base_list, t_gripper2base_list = [], []
        R_target2cam_list, t_target2cam_list = [], []

        print(f"\n--- 데이터 수집 시작 ({NUM_SAMPLES}개 필요) ---")
        
        while len(R_gripper2base_list) < NUM_SAMPLES:
            capture = k4a.get_capture()
            if capture.color is None:
                continue
            
            frame = cv2.undistort(capture.color, camera_matrix, dist_coeffs)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, flags)
            
            status_text = "✅ 인식 성공!" if ret else "❌ 인식 실패..."
            progress_text = f"[{len(R_gripper2base_list)}/{NUM_SAMPLES}]"
            print(f"\r{progress_text} | 체커보드: {status_text}", end="")
        
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(frame, CHECKERBOARD_DIMS, corners2, ret)
                _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners2, camera_matrix, dist_coeffs)
        
            cv2.imshow('Hand-Eye Calibration', frame)
            key = cv2.waitKey(1) & 0xFF
        
            if key == ord('c') and ret:
                print(f"\n[{len(R_gripper2base_list)+1}/{NUM_SAMPLES}] 포즈 저장 중...")
        
                # --- 1. 단서 A (로봇) 수집 (XArm 좌표계) ---
                code, pose = arm.get_position(is_radian=False)
                # print("robot_pose:")
                if code != 0:
                    print(f"  > ⚠️ 로봇 포즈 읽기 실패! 에러 코드: {code}")
                    continue
                
                # [ 💥 3. "좌표 설정" 수정: XArm -> ROS 변환 💥 ]
                # (x, y, z, roll, pitch, yaw)
                x, y, z, roll, pitch, yaw = pose
                
                # XArm(Y-Right) -> ROS(Y-Left) 변환
                t_gripper2base_ROS = np.array([x, -y, z]).reshape(3, 1) # Y축 부호 반전
                
                # 회전 변환: R_ROS = R_fix @ R_XArm
                # XArm의 (roll, pitch, yaw)는 'xyz' 순서
                R_gripper2base_XArm = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
                
                # R_ROS = R_fix @ R_XArm @ R_fix_inv (Tait-Bryan ZYX convention)
                # 더 간단한 오일러 각 변환: (roll, -pitch, -yaw)
                R_gripper2base_ROS = Rotation.from_euler('xyz', [roll, -pitch, -yaw], degrees=True).as_matrix()
                # ==================================================

                # --- 2. 단서 B (카메라) 수집 (OpenCV 좌표계) ---
                R_target2cam_CV, _ = cv2.Rodrigues(rvec)
                t_target2cam_CV = tvec.reshape(3, 1)

                # [ 💥 4. "좌표 설정" 수정: CV -> ROS 변환 💥 ]
                # (이 부분은 이전과 동일하게 올바름)
                R_target2cam_ROS = R_CV_TO_ROS @ R_target2cam_CV
                t_target2cam_ROS = R_CV_TO_ROS @ t_target2cam_CV
                # ==================================================

                # --- 3. 변환된 "ROS" 데이터만 리스트에 추가 ---
                R_gripper2base_list.append(R_gripper2base_ROS) # ⬅️ ROS로 변환된 값
                t_gripper2base_list.append(t_gripper2base_ROS) # ⬅️ ROS로 변환된 값
                R_target2cam_list.append(R_target2cam_ROS)     # ⬅️ ROS로 변환된 값
                t_target2cam_list.append(t_target2cam_ROS)     # ⬅️ ROS로 변환된 값
                
                print("  > ✅ 저장 완료!")

            elif key == ord('q'):
                print("\n\n사용자가 수동으로 데이터 수집을 종료했습니다.")
                break
        
        if len(R_gripper2base_list) < 4:
            print("\n⚠️ 캘리브레이션을 계산하기에 데이터가 부족합니다.")
            return

        print("\n\n--- 캘리브레이션 계산 시작 (모든 좌표 ROS 통일됨) ---")
        
        # 이제 (ROS, ROS, ROS, ROS) 데이터를 사용하므로 결과가 올바릅니다.
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base_list,
            t_gripper2base=t_gripper2base_list,
            R_target2cam=R_target2cam_list,
            t_target2cam=t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        print("\n--- 🔬 캘리브레이션 결과 (ROS 좌표계 기준) ---")
        print("[위치 오프셋 (Translation)] T_cam_in_gripper (mm):")
        print(t_cam2gripper.flatten())

        quat_xyzw = Rotation.from_matrix(R_cam2gripper).as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        print("\n[회전 오프셋 (Quaternion)] Q_cam_in_gripper (w, x, y, z):")
        print(np.array(quat_wxyz))

        euler_deg = Rotation.from_matrix(R_cam2gripper).as_euler('xyz', degrees=True)
        print("\n[회전 오프셋 (Euler Angles)] R_cam_in_gripper (roll, pitch, yaw, degrees):")
        print(euler_deg)

    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")
    finally:
        if k4a and k4a.is_running:
            k4a.stop()
            print("\n📷 카메라가 안전하게 정지되었습니다.")
        cv2.destroyAllWindows()
        if arm and arm.connected:
            arm.disconnect()
            print("🤖 로봇 연결이 안전하게 해제되었습니다.")
        print("\n프로그램을 종료합니다.")

if __name__ == '__main__':
    main()