# import cv2
# import cv2.aruco as aruco
# import numpy as np

# def generate_aruco_marker(marker_id, marker_size=151, dictionary=aruco.DICT_4X4_50):      # 5cm的aruco Pixel為188.98, 4cm Pixel為151.18
#     # 获取指定的字典
#     aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
#     # 生成标记图像
#     img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
#     # 保存标记图像
#     cv2.imwrite(f'aruco_marker_{marker_id}.png', img)
#     print(f"已保存 ArUco 标记 ID {marker_id} 为 aruco_marker_{marker_id}.png")

# # 生成 ID 为 0 到 4 的 ArUco 标记
# for marker_id in range(1):
#     generate_aruco_marker(marker_id)







import cv2
import cv2.aruco as aruco
import numpy as np

# 摄像头1的内参矩阵和畸变系数
camera_matrix1 = np.array([[2424.2794834773194, 0, 963.0166430088262],
                           [0, 2424.6822621489446, 621.0569418586338],
                           [0, 0, 1]], dtype=float)
dist_coeffs1 = np.array((0.0, 0.0, 0.0, 0.0, 0.0))


def detect_aruco_from_images(image_path1, image_path2=None, dictionary=aruco.DICT_6X6_50):
    """
    使用两个图像检测 ArUco 标记，并在标记上绘制三维坐标系。

    参数：
    - image_path1: 第一张图像路径
    - image_path2: 第二张图像路径（可选）
    - dictionary: ArUco 字典
    """
    # 读取图像
    frame1 = cv2.imread(image_path1)
    if frame1 is None:
        print(f"无法读取图像：{image_path1}")
        return

    frame2 = None
    if image_path2:
        frame2 = cv2.imread(image_path2)
        if frame2 is None:
            print(f"无法读取图像：{image_path2}")

    # 获取 ArUco 字典和检测参数
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    parameters = aruco.DetectorParameters()

    # 转换为灰度图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = None
    if frame2 is not None:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 检测 ArUco 标记
    corners1, ids1, _ = aruco.detectMarkers(gray1, aruco_dict, parameters=parameters)
    if frame2 is not None:
        corners2, ids2, _ = aruco.detectMarkers(gray2, aruco_dict, parameters=parameters)
    else:
        corners2, ids2 = None, None

    # 如果检测到 ArUco 标记，估计其姿态并绘制坐标轴
    if ids1 is not None:
        frame1 = aruco.drawDetectedMarkers(frame1, corners1, ids1)
        rvecs1, tvecs1, _ = aruco.estimatePoseSingleMarkers(corners1, 0.05, camera_matrix1, dist_coeffs1)
        for rvec, tvec in zip(rvecs1, tvecs1):
            cv2.drawFrameAxes(frame1, camera_matrix1, dist_coeffs1, rvec, tvec, 0.1)

    if ids2 is not None and frame2 is not None:
        frame2 = aruco.drawDetectedMarkers(frame2, corners2, ids2)
        rvecs2, tvecs2, _ = aruco.estimatePoseSingleMarkers(corners2, 0.05, camera_matrix1, dist_coeffs1)
        for rvec, tvec in zip(rvecs2, tvecs2):
            cv2.drawFrameAxes(frame2, camera_matrix1, dist_coeffs1, rvec, tvec, 0.1)

    # 显示检测结果
    cv2.imshow('Image 1 - ArUco Markers with 3D Axis', frame1)
    if frame2 is not None:
        cv2.imshow('Image 2 - ArUco Markers with 3D Axis', frame2)

    # 按下 'q' 键退出
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭窗口
    cv2.destroyAllWindows()


# 调用函数，假设图像路径为 'image1.jpg' 和 'image2.jpg'
detect_aruco_from_images('image1.jpg', 'image2.jpg')
