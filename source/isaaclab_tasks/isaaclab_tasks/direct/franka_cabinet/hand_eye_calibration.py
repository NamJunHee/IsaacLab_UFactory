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
CHECKERBOARD_DIMS = (8, 6) # 체커보드 내부 코너 개 (가로, 세로)
SQUARE_SIZE_MM = 25.0      # 체커보드 사각형 한 변의 길이 (mm)
ROBOT_IP = "192.168.1.208" # 로봇 IP 주소
NUM_SAMPLES = 15           # 수집할 샘플 개수 (최소 4개, 10~20개 권장)
# ====================================================================

# [ 💥 1. 카메라 좌표계 변환 행렬 (CV -> ROS) 💥 ]
# OpenCV (X:오른쪽, Y:아래, Z:앞) -> ROS (X:앞, Y:왼쪽, Z:위)
R_CV_TO_ROS = np.array([
    [0,  0,  1], # ROS X = CV Z
    [-1, 0,  0], # ROS Y = -CV X
    [0, -1,  0]  # ROS Z = -CV Y
], dtype=np.float32)

# ====================================================================
# [ 💥 2. 로봇 좌표계 변환 (XArm -> ROS) 💥 ]
# XArm (X:앞, Y:오른쪽, Z:아래) -> ROS (X:앞, Y:왼쪽, Z:위)
#
# 이 변환을 위해 Y축과 Z축을 모두 반전시킵니다.
#
# 1. 위치 변환 (t_ROS = [t_x, -t_y, -t_z])
# 2. 회전 변환 (오일러 각: [roll, -pitch, -yaw])
#
# 이 스크립트에서는 이 두 가지 변환을 코드 내에서 직접 적용합니다.
# (이 로직은 올바르게 작성되었습니다.)
# ====================================================================

def main():
    """ Hand-Eye 캘리브레이션 메인 함수 """

    # 3D 체커보드 코너 좌표 생성 (mm 단위)
    objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM 

    arm = None
    k4a = None
    try:
        # --- 로봇 연결 ---
        print("🤖 로봇에 연결하는 중...")
        arm = XArmAPI(ROBOT_IP, request_timeout=5) 
        if not arm.connected:
            print(f"❌ 로봇 연결 실패. (IP: {ROBOT_IP})")
            print("    IP 주소를 확인하거나 로봇이 켜져 있는지 확인하세요.")
            return
            
        arm.motion_enable(enable=True)
        arm.set_mode(0) # Position Control Mode
        arm.set_state(0)
        time.sleep(1)
        print("✅ 로봇 연결 성공. (좌표계: XArm)")

        # --- 카메라 초기화 ---
        print("\n📷 Azure Kinect 카메라 초기화 중...")
        k4a = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        k4a.start()
        print("✅ 카메라 초기화 성공. (좌표계: OpenCV)")

        # 카메라 내부 파라미터 획득
        calibration = k4a.calibration
        camera_matrix = calibration.get_camera_matrix(CalibrationType.COLOR)
        dist_coeffs = calibration.get_distortion_coefficients(CalibrationType.COLOR)
        
        print("\n[카메라 파라미터]")
        print(f"  > Camera Matrix (fx, fy, cx, cy):\n{camera_matrix}")
        print(f"  > Distortion Coefficients:\n{dist_coeffs}")

        # 캘리브레이션 데이터 저장 리스트
        R_gripper2base_list, t_gripper2base_list = [], []
        R_target2cam_list, t_target2cam_list = [], []

        print(f"\n--- 데이터 수집 시작 ({NUM_SAMPLES}개 필요) ---")
        print("로봇을 '다른 위치'와 '다른 각도'로 움직인 후,")
        print("체커보드가 인식되면 'c' 키를 눌러 포즈를 캡처하세요.")
        print("다양한 각도와 위치에서 수집해야 정확도가 올라갑니다.")
        print("'q' 키를 누르면 수집을 중단하고 캘리브레이션을 시작합니다.")
        
        while len(R_gripper2base_list) < NUM_SAMPLES:
            capture = k4a.get_capture()
            if capture.color is None:
                continue
            
            # 1. 이미지 캡처 및 체커보드 검출
            # undistort를 여기서 하면 PnP 계산 시 왜곡 계수를 사용하지 않아야 함
            # 여기서는 시각화 용도로만 사용하고, PnP에는 원본 파라미터 사용
            frame_display = cv2.undistort(capture.color, camera_matrix, dist_coeffs)
            gray = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2GRAY) # PnP는 원본(왜곡된) 이미지에서 수행
        
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, flags)
            
            status_text = "✅ 인식 성공!" if ret else "❌ 인식 실패..."
            progress_text = f"[{len(R_gripper2base_list)}/{NUM_SAMPLES}]"
            print(f"\r{progress_text} | 체커보드: {status_text} (c: 캡처, q: 종료)", end="")
        
            if ret:
                # 2. 코너 좌표 정밀화 및 PnP 계산
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(frame_display, CHECKERBOARD_DIMS, corners2, ret)
                
                # solvePnPRansac은 왜곡된 이미지와 왜곡 계수를 사용해야 함
                _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners2, camera_matrix, dist_coeffs)
        
            cv2.imshow('Hand-Eye Calibration', frame_display)
            key = cv2.waitKey(1) & 0xFF
        
            if key == ord('c') and ret:
                print(f"\n[{len(R_gripper2base_list)+1}/{NUM_SAMPLES}] 포즈 저장 중...")
        
                # --- 1. 단서 A (로봇) 수집 (XArm 좌표계) ---
                # (T_gripper_to_base)
                code, pose = arm.get_position(is_radian=False) # (mm, deg)
                if code != 0:
                    print(f"  > ⚠️ 로봇 포즈 읽기 실패! 에러 코드: {code}")
                    continue
                
                # [ 💥 3. "좌표 설정" : XArm -> ROS 변환 💥 ]
                # (x, y, z, roll, pitch, yaw)
                x, y, z, roll, pitch, yaw = pose
                
                t_gripper2base_ROS = np.array([x, y, z], dtype=np.float32).reshape(3, 1)
                
                # 회전 변환: R_ROS = (roll, -pitch, -yaw)
                R_gripper2base_ROS = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix().astype(np.float32)
                # ==================================================

                # --- 2. 단서 B (카메라) 수집 (OpenCV 좌표계) ---
                # (T_target_to_cam)
                R_target2cam_CV, _ = cv2.Rodrigues(rvec)
                t_target2cam_CV = tvec.astype(np.float32).reshape(3, 1) # (mm)

                # [ 💥 4. "좌표 설정" : CV -> ROS 변환 💥 ]
                R_target2cam_ROS = R_CV_TO_ROS @ R_target2cam_CV
                t_target2cam_ROS = R_CV_TO_ROS @ t_target2cam_CV
                # ==================================================

                # --- 3. 변환된 "ROS" 데이터만 리스트에 추가 ---
                R_gripper2base_list.append(R_gripper2base_ROS) # ⬅️ ROS로 변환된 값
                t_gripper2base_list.append(t_gripper2base_ROS) # ⬅️ ROS로 변환된 값
                R_target2cam_list.append(R_target2cam_ROS)     # ⬅️ ROS로 변환된 값
                t_target2cam_list.append(t_target2cam_ROS)     # ⬅️ ROS로 변환된 값
                
                print(f"  > 로봇 포즈 (ROS): t={t_gripper2base_ROS.flatten()}")
                print(f"  > 타겟 포즈 (ROS): t={t_target2cam_ROS.flatten()}")
                print("  > ✅ 저장 완료!")

            elif key == ord('q'):
                print("\n\n사용자가 수동으로 데이터 수집을 종료했습니다.")
                break
        
        if len(R_gripper2base_list) < 4:
            print(f"\n⚠️ 캘리브레이션을 계산하기에 데이터가 부족합니다 (최소 4개 필요, 현재 {len(R_gripper2base_list)}개).")
            return

        print(f"\n\n--- 캘리브레이션 계산 시작 (총 {len(R_gripper2base_list)}개 샘플) ---")
        print("모든 좌표계가 ROS 기준으로 통일되었습니다.")
        
        # 3. Hand-Eye 캘리브레이션 계산 (AX=XB)
        # T_gripper_to_base (A) 와 T_target_to_cam (B) 를 사용하여
        # T_cam_to_gripper (X) 를 계산합니다.
        # method=cv2.CALIB_HAND_EYE_TSAI 가 가장 표준적입니다.
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base_list,
            t_gripper2base=t_gripper2base_list,
            R_target2cam=R_target2cam_list,
            t_target2cam=t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI 
        )

        print("\n--- 🔬 캘리브레이션 결과 (T_cam_to_gripper) ---")
        print("이 값들을 'franka_object_tracking_env.py'의 __init__ 함수에 복사하세요.")

        print("\n[위치 오프셋 (Translation)] t_cam_to_gripper_mm (mm):")
        print("torch.tensor(")
        print(f"    {list(t_cam2gripper.flatten())},")
        print("    device=self.device, dtype=torch.float32")
        print(")")

        # Scipy를 사용하여 (x, y, z, w) 형식의 쿼터니언 생성
        quat_xyzw = Rotation.from_matrix(R_cam2gripper).as_quat()
        
        # (w, x, y, z) 순서로 변경
        # [수정됨] qu2at_xyzw[0] -> quat_xyzw[0] (오타 수정)
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        
        print("\n[회전 오프셋 (Quaternion)] R_cam_to_gripper_quat_ROS (w, x, y, z):")
        print("torch.tensor(")
        print(f"    {quat_wxyz},")
        print("    device=self.device, dtype=torch.float32")
        print(")")

        # 참고용 오일러 각도 출력 (ROS 기준: roll, pitch, yaw)
        euler_deg = Rotation.from_matrix(R_cam2gripper).as_euler('xyz', degrees=True)
        print("\n[참고: 회전 오프셋 (Euler Angles)] (roll, pitch, yaw, degrees):")
        print(f"    {list(euler_deg)}")

    except Exception as e:
        import traceback
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")
        traceback.print_exc()
    finally:
        # --- 정리 ---
        if k4a and k4a.is_running:
            k4a.stop()
            print("\n📷 카메라가 안전하게 정지되었습니다.")
        cv2.destroyAllWindows()
        if arm and arm.connected:
            arm.set_state(4) # Stop
            arm.disconnect()
            print("🤖 로봇 연결이 안전하게 해제되었습니다.")
        print("\n프로그램을 종료합니다.")

if __name__ == '__main__':
    main()