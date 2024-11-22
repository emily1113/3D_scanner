import cv2
import cv2.aruco as aruco
import numpy as np

# 摄像头1的内参矩阵和畸变系数
camera_matrix1 = np.array([ [2424.2794834773194, 0, 963.0166430088262],
                            [0, 2424.6822621489446, 621.0569418586338],
                            [  0,           0,           1        ]], dtype=float)
dist_coeffs1 = np.array((0.0, 0.0, 0.0, 0.0,  0.0))

# 摄像头2的内参矩阵和畸变系数
# camera_matrix2 = np.array([[386.62339387,   0,         319.76436843],
#                             [  0,         387.06062011, 183.7132677 ],
#                             [  0,           0,           1        ]], dtype=float)
# dist_coeffs2 = np.array((0.09940914, -0.24385382, 0.0005279, -0.00157842, 0.11749123))

def detect_aruco_from_two_cameras(camera_id1, camera_id2, dictionary=aruco.DICT_6X6_50):
    """
    使用两个摄像头同时检测 ArUco 标记，并在标记上绘制三维坐标系。

    参数：
    - camera_id1: 第一个摄像头的 ID（通常 0 或 1）
    - camera_id2: 第二个摄像头的 ID（通常 1 或 2）
    - dictionary: ArUco 字典
    """
    # 打开两个摄像头
    cap1 = cv2.VideoCapture(camera_id1, cv2.CAP_DSHOW)
    # cap2 = cv2.VideoCapture(camera_id2, cv2.CAP_DSHOW)

    # 确保摄像头已打开
    if not cap1.isOpened() :
        print("无法打开摄像头")
        return

    # 获取 ArUco 字典和检测参数
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    parameters = aruco.DetectorParameters()

    while True:
        # 从两个摄像头捕获图像
        ret1, frame1 = cap1.read()
        # ret2, frame2 = cap2.read()

        if not ret1 :
            print("无法从摄像头获取图像")
            break

        # 转换为灰度图像
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 检测 ArUco 标记
        corners1, ids1, _ = aruco.detectMarkers(gray1, aruco_dict, parameters=parameters)
        # corners2, ids2, _ = aruco.detectMarkers(gray2, aruco_dict, parameters=parameters)

        # 如果检测到 ArUco 标记，估计其姿态并绘制坐标轴
        if ids1 is not None:
            frame1 = aruco.drawDetectedMarkers(frame1, corners1, ids1)
            rvecs1, tvecs1, _ = aruco.estimatePoseSingleMarkers(corners1, 0.05, camera_matrix1, dist_coeffs1)
            for rvec, tvec in zip(rvecs1, tvecs1):
                # 使用 cv2.drawFrameAxes 而不是 aruco.drawAxis
                cv2.drawFrameAxes(frame1, camera_matrix1, dist_coeffs1, rvec, tvec, 0.1)

        # if ids2 is not None:
        #     frame2 = aruco.drawDetectedMarkers(frame2, corners2, ids2)
        #     rvecs2, tvecs2, _ = aruco.estimatePoseSingleMarkers(corners2, 0.05, camera_matrix2, dist_coeffs2)
        #     for rvec, tvec in zip(rvecs2, tvecs2):
        #         # 使用 cv2.drawFrameAxes 而不是 aruco.drawAxis
        #         cv2.drawFrameAxes(frame2, camera_matrix2, dist_coeffs2, rvec, tvec, 0.1)

        # 显示两幅图像
        cv2.imshow('Camera 1 - ArUco Markers with 3D Axis', frame1)
        # cv2.imshow('Camera 2 - ArUco Markers with 3D Axis', frame2)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap1.release()
    # cap2.release()
    cv2.destroyAllWindows()

# 调用函数，假设第一个摄像头 ID 为 1，第二个摄像头 ID 为 2
detect_aruco_from_two_cameras(1, 2)
