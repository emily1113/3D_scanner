import cv2
import cv2.aruco as aruco
import numpy as np

# 摄像头内参矩阵和畸变系数
camera_matrix = np.array([[2424.2794834773194, 0, 963.0166430088262],
                          [0, 2424.6822621489446, 621.0569418586338],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.array((0.0, 0.0, 0.0, 0.0, 0.0))

def resize_image(image, max_width, max_height):
    """
    调整图像大小以适应屏幕范围。
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale < 1:  # 只有当图像超过限制时才进行缩放
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image

def get_transformation_matrix(rvec, tvec):
    """
    根据旋转向量和位移向量计算 4x4 齐次变换矩阵。

    参数：
    - rvec: 旋转向量 (3x1)
    - tvec: 位移向量 (3x1)

    返回：
    - transformation_matrix: 4x4 齐次变换矩阵
    """
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # 构建 4x4 齐次变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = tvec.ravel()
    return transformation_matrix

def detect_aruco_from_image(image_path, dictionary=aruco.DICT_6X6_50, max_width=800, max_height=600):
    """
    从图片中检测 ArUco 标记，并在标记上绘制三维坐标系，同时计算相机到 ArUco 的变换矩阵。

    参数：
    - image_path: 图片路径
    - dictionary: ArUco 字典
    - max_width: 显示图像的最大宽度
    - max_height: 显示图像的最大高度
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图片")
        return

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取 ArUco 字典和检测参数
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    parameters = aruco.DetectorParameters()

    # 检测 ArUco 标记
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 如果检测到 ArUco 标记
    if ids is not None:
        image = aruco.drawDetectedMarkers(image, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            # 使用 cv2.drawFrameAxes 绘制坐标轴
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # 计算并打印相机到 ArUco 的转换矩阵
            transformation_matrix = get_transformation_matrix(rvec, tvec)
            print(f"ArUco ID: {ids.ravel()}")
            print("Transformation Matrix (Camera to ArUco):")
            print(transformation_matrix)

    # 调整图像大小以适应屏幕
    image = resize_image(image, max_width, max_height)

    # 显示结果
    cv2.imshow('ArUco Markers with 3D Axis', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 替换图片路径
image_path = "C:/Users/ASUS/Desktop/POINT/3D_scanner/ArUco/rgb_image_00000.png"
detect_aruco_from_image(image_path, max_width=800, max_height=600)
